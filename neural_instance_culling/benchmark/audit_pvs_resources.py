#!/usr/bin/env python3
"""Audit PVS CSR, runtime instance mapping, and offline GLB point-cache semantics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime_meta(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("componentRecords") or []
    if not records:
        raise ValueError(f"{path} has no componentRecords")
    count = max(int(row["componentGlobalId"]) for row in records) + 1
    aabbs = np.zeros((count, 6), dtype=np.float32)
    instance_to_glb = np.zeros((count,), dtype=np.int64)
    for row in records:
        instance_id = int(row["componentGlobalId"])
        bounds = row["bounds"]
        if "min" in bounds and "max" in bounds:
            minimum = np.asarray(bounds["min"], dtype=np.float32)
            maximum = np.asarray(bounds["max"], dtype=np.float32)
        else:
            center = np.asarray(bounds["center"], dtype=np.float32)
            size = np.asarray(bounds["size"], dtype=np.float32)
            minimum = center - size * 0.5
            maximum = center + size * 0.5
        aabbs[instance_id, :3] = minimum
        aabbs[instance_id, 3:] = maximum
        instance_to_glb[instance_id] = int(row["globalGlbId"])
    return aabbs, instance_to_glb, payload


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_candidate_subset(dataset_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    visible_offsets = np.memmap(dataset_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
    visible_ids = np.memmap(dataset_dir / "visible_ids.bin", dtype=np.uint32, mode="r")
    candidate_offsets_path = dataset_dir / "candidate_offsets.bin"
    candidate_ids_path = dataset_dir / "candidate_ids.bin"
    result: dict[str, Any] = {
        "poseCount": int(visible_offsets.size - 1),
        "visibleRefCount": int(visible_ids.size),
        "candidateFilesPresent": candidate_offsets_path.exists() and candidate_ids_path.exists(),
        "candidateMissVisible": None,
        "candidateMissPoseCount": None,
        "candidateSemantics": meta.get("candidateSemantics"),
        "candidateRawIndependentlyAvailable": False,
        "candidateIncludesForcedVisiblePositives": False,
    }
    if not result["candidateFilesPresent"]:
        return result
    semantics = str(result.get("candidateSemantics") or "").lower()
    result["candidateIncludesForcedVisiblePositives"] = bool(
        "plus" in semantics and "visible" in semantics
    )
    # The current builders overwrite the raw AABB candidate with the unioned
    # set and only retain the post-union file.  A subset check on that file is
    # therefore not evidence that the executable AABB stage was safe.
    result["candidateRawIndependentlyAvailable"] = bool(
        meta.get("rawCandidateFile") or meta.get("rawCandidateIds")
    )
    candidate_offsets = np.memmap(candidate_offsets_path, dtype=np.uint64, mode="r")
    candidate_ids = np.memmap(candidate_ids_path, dtype=np.uint32, mode="r")
    if candidate_offsets.size != visible_offsets.size:
        raise ValueError("candidate_offsets and visible_offsets have different pose counts")
    missing_refs = 0
    missing_poses = 0
    for pose_index in range(int(visible_offsets.size - 1)):
        visible_start = int(visible_offsets[pose_index])
        visible_end = int(visible_offsets[pose_index + 1])
        candidate_start = int(candidate_offsets[pose_index])
        candidate_end = int(candidate_offsets[pose_index + 1])
        visible = np.asarray(visible_ids[visible_start:visible_end], dtype=np.uint32)
        candidate = np.asarray(candidate_ids[candidate_start:candidate_end], dtype=np.uint32)
        missing = np.setdiff1d(visible, candidate, assume_unique=False)
        if missing.size:
            missing_refs += int(missing.size)
            missing_poses += 1
    result["candidateMissVisible"] = int(missing_refs)
    result["candidateMissPoseCount"] = int(missing_poses)
    return result


def audit_point_cache(
    glb_points: Path,
    glb_index: Path | None,
    instance_to_glb: np.ndarray,
) -> dict[str, Any]:
    meta_path = glb_points.with_name(glb_points.stem + "_meta.json")
    point_meta = read_json(meta_path) if meta_path.exists() else {}
    runtime_glb_count = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    point_glb_count = int(point_meta.get("numGlbs", -1))
    result: dict[str, Any] = {
        "path": str(glb_points),
        "metaPath": str(meta_path),
        "runtimeGlbCount": runtime_glb_count,
        "pointCacheRowCount": point_glb_count,
        "rowCountMatchesRuntimeGlbs": point_glb_count == runtime_glb_count,
        "decoded": point_meta.get("decoded"),
        "failed": point_meta.get("failed"),
        "fallback": point_meta.get("fallback"),
        "pointsPerGlb": point_meta.get("pointsPerGlb"),
        "hashMatchCount": None,
        "hashComparedCount": None,
    }
    if glb_index and glb_index.exists() and isinstance(point_meta.get("entries"), list):
        index_entries = read_json(glb_index).get("entries") or []
        index_hash = {int(row["globalId"]): str(row.get("hash", "")) for row in index_entries}
        compared = 0
        matched = 0
        for row in point_meta["entries"]:
            global_id = int(row.get("globalId", -1))
            if global_id not in index_hash:
                continue
            compared += 1
            matched += int(str(row.get("hash", "")) == index_hash[global_id])
        result["hashMatchCount"] = int(matched)
        result["hashComparedCount"] = int(compared)
    return result


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    runtime_meta_path = Path(args.runtime_meta)
    glb_points = Path(args.glb_points)
    glb_index = Path(args.glb_index) if args.glb_index else None
    dataset_meta = read_json(dataset_dir / "dataset_meta.json")
    aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    candidate = audit_candidate_subset(dataset_dir, dataset_meta)
    point_cache = audit_point_cache(glb_points, glb_index, instance_to_glb)

    checks = {
        "visibleIdsWithinRuntimeInstances": bool(
            not (dataset_dir / "visible_ids.bin").exists()
            or int(np.memmap(dataset_dir / "visible_ids.bin", dtype=np.uint32, mode="r").max(initial=0)) < aabbs.shape[0]
        ),
        "candidateContainsVisibleWithoutForcedUnion": bool(
            candidate.get("candidateMissVisible") == 0
            and candidate.get("candidateRawIndependentlyAvailable")
            and not candidate.get("candidateIncludesForcedVisiblePositives")
        ),
        "pointCacheRowsMatchRuntimeGlbs": point_cache["rowCountMatchesRuntimeGlbs"],
        "pointCacheHasNoDecodeFailure": point_cache.get("failed") == 0,
        "pointCacheHasNoFallback": point_cache.get("fallback") == 0,
    }
    if point_cache.get("hashComparedCount") is not None:
        checks["pointCacheHashesMatchGlbIndex"] = point_cache["hashMatchCount"] == point_cache["hashComparedCount"]
    result = {
        "schema": "pvs-resource-audit-v1",
        "datasetDir": str(dataset_dir),
        "datasetMeta": dataset_meta,
        "runtimeMeta": {
            "path": str(runtime_meta_path),
            "sha256": sha256_file(runtime_meta_path),
            "instanceCount": int(aabbs.shape[0]),
            "runtimeGlbCount": int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0,
            "source": runtime_meta.get("sceneName") or runtime_meta.get("source"),
        },
        "candidateAudit": candidate,
        "pointCacheAudit": point_cache,
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--glb-points", required=True)
    parser.add_argument("--glb-index", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = audit(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    raise SystemExit(0 if result["status"] == "pass" else 2)


if __name__ == "__main__":
    main()
