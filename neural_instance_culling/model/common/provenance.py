"""Canonical provenance helpers shared by the new PVS experiment line.

The experiment writes the same digest at each boundary (protocol, relation
CSR, checkpoint, and runtime export).  JSON is canonicalized before hashing so
key ordering or whitespace cannot create a false distinction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def protocol_digest(protocol: Mapping[str, Any]) -> str:
    """Hash a protocol after removing its self-referential digest field."""
    payload = dict(protocol)
    payload.pop("protocolDigest", None)
    return canonical_json_sha256(payload)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relation_artifact_digest(
    relation_dir: str | Path,
    metadata: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    """Hash relation metadata plus every declared binary provenance file.

    This is deliberately a digest of the offline artifact, not a runtime
    asset.  It includes hierarchy and censoring files when they are declared
    in metadata, so a changed group mapping cannot reuse an old checkpoint.
    """
    root = Path(relation_dir)
    files: dict[str, str] = {}

    def add(relative: str) -> None:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing relation provenance file: {path}")
        files[relative] = sha256_file(path)

    declared = metadata.get("files")
    if not isinstance(declared, Mapping):
        raise ValueError("relation metadata has no declared files")
    for value in declared.values():
        if isinstance(value, str) and value:
            add(value)

    hierarchy = metadata.get("hierarchy")
    if isinstance(hierarchy, Mapping) and isinstance(hierarchy.get("files"), Mapping):
        for value in hierarchy["files"].values():
            if isinstance(value, str) and value:
                add(value)

    observations = metadata.get("survivalObservations")
    if isinstance(observations, Mapping) and isinstance(observations.get("files"), Mapping):
        for value in observations["files"].values():
            if isinstance(value, str) and value:
                add(value)

    payload = {
        "schema": metadata.get("schema"),
        "metadata": dict(metadata),
        "files": dict(sorted(files.items())),
    }
    return canonical_json_sha256(payload), files


__all__ = [
    "canonical_json_bytes",
    "canonical_json_sha256",
    "protocol_digest",
    "sha256_file",
    "relation_artifact_digest",
]
