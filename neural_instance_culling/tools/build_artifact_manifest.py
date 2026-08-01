#!/usr/bin/env python3
"""Create a small reproducibility manifest for local or external artifacts.

The manifest records hashes and sizes only. It never copies model weights,
datasets, GLBs, or benchmark outputs into the source repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "neuralstreamweb3d-artifact-manifest-v1"
CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}.")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"NAME and PATH must be non-empty, got {value!r}.")
    return name, Path(raw_path)


def resolve_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {candidate}")
    if candidate.is_symlink():
        raise ValueError(f"Symlink artifact paths are not supported: {candidate}")
    return candidate


def iter_files(path: Path) -> Iterable[tuple[str, Path]]:
    if path.is_file():
        yield path.name, path
        return
    if not path.is_dir():
        raise ValueError(f"Artifact path is neither a regular file nor directory: {path}")
    for child in sorted(path.rglob("*")):
        if child.is_symlink():
            raise ValueError(f"Symlink inside artifact directory is not supported: {child}")
        if child.is_file():
            yield child.relative_to(path).as_posix(), child


def tree_digest(entries: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        digest.update(str(entry["relativePath"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(entry["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = run("status", "--porcelain=v1")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "statusPorcelain": status or "",
    }


def build_manifest(root: Path, files: list[str], directories: list[str]) -> dict[str, Any]:
    if not files and not directories:
        raise ValueError("At least one --artifact or --directory is required.")
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for kind, values in (("file", files), ("directory", directories)):
        for raw in values:
            name, raw_path = parse_name_path(raw)
            path = resolve_path(root, raw_path)
            if kind == "file" and not path.is_file():
                raise ValueError(f"--artifact requires a regular file: {path}")
            if kind == "directory" and not path.is_dir():
                raise ValueError(f"--directory requires a directory: {path}")
            entries: list[dict[str, Any]] = []
            for relative, child in iter_files(path):
                if child in seen:
                    raise ValueError(f"Artifact paths overlap at {child}")
                seen.add(child)
                digest, size = sha256_file(child)
                entries.append({"relativePath": relative, "bytes": size, "sha256": digest})
            total = sum(int(entry["bytes"]) for entry in entries)
            records.append(
                {
                    "name": name,
                    "kind": kind,
                    "path": str(path),
                    "fileCount": len(entries),
                    "bytes": total,
                    "treeSha256": tree_digest(entries),
                    "files": entries,
                }
            )
    return {
        "schema": SCHEMA,
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root.resolve()),
        "git": git_state(root),
        "artifacts": records,
        "totalBytes": sum(int(record["bytes"]) for record in records),
        "totalFiles": sum(int(record["fileCount"]) for record in records),
    }


def run_self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "weights").mkdir()
        (root / "weights" / "a.bin").write_bytes(b"abc")
        (root / "weights" / "b.bin").write_bytes(b"defgh")
        (root / "meta.json").write_bytes(b"x")
        manifest = build_manifest(root, ["meta=" + str(root / "meta.json")], ["weights=" + str(root / "weights")])
        assert manifest["schema"] == SCHEMA
        assert manifest["totalBytes"] == 9
        assert manifest["totalFiles"] == 3
        assert manifest["artifacts"][0]["files"][0]["sha256"] == hashlib.sha256(b"x").hexdigest()
    return {"status": "passed", "checked": ["file hash", "directory tree hash", "size accounting", "overlap guard"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hash local artifacts without copying them into Git.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact", action="append", default=[], help="NAME=FILE; may be repeated")
    parser.add_argument("--directory", action="append", default=[], help="NAME=DIRECTORY; may be repeated")
    parser.add_argument("--output", type=Path, help="Output JSON path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    root = args.root.resolve()
    manifest = build_manifest(root, args.artifact, args.directory)
    if args.output is None:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(output.resolve()), "totalFiles": manifest["totalFiles"], "totalBytes": manifest["totalBytes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
