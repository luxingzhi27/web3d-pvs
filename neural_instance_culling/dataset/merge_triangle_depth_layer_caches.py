#!/usr/bin/env python3
"""Merge completed hardware triangle-depth cache shards.

The browser renderer writes one cache per shard.  This utility validates every
shard before concatenating it in pose-index order.  It deliberately refuses
partial caches, duplicate poses, mismatched render settings, failed GPU gates,
or a pose set that does not match the registered split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

MODEL_DIR = Path(__file__).resolve().parents[1] / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
from common.candidate_identity import (  # noqa: E402
    candidate_digest_for_pose_sequence,
    pose_sequence_for_splits,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


CACHE_SCHEMA = "triangle-depth-layer-cache-v1"
CACHE_SCHEMA_V2 = "triangle-depth-layer-cache-v2"
SUPPORTED_CACHE_SCHEMAS = {CACHE_SCHEMA, CACHE_SCHEMA_V2}
EVIDENCE_SCHEMA = "triangle-depth-layer-gpu-evidence-v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_completed_cache(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.memmap, np.memmap]:
    meta_path = path / "layer_cache_meta.json"
    if not meta_path.is_file():
        raise ValueError(f"cache is not complete: missing {meta_path}")
    meta = _read_json(meta_path)
    if meta.get("schema") not in SUPPORTED_CACHE_SCHEMAS:
        raise ValueError(f"{path}: expected one of {sorted(SUPPORTED_CACHE_SCHEMAS)}, got {meta.get('schema')!r}")
    if not bool(meta.get("gpuGate", {}).get("hardware")):
        raise ValueError(f"{path}: cache did not pass the browser hardware GPU gate")
    evidence_path = next(path.glob("*.gpu_evidence.json"), None)
    if evidence_path is None:
        raise ValueError(f"{path}: missing GPU evidence JSON")
    evidence = _read_json(evidence_path)
    before = evidence.get("hostGpuBefore") or {}
    during = evidence.get("hostGpuDuring") or {}
    formal_gpu = (
        evidence.get("schema") == EVIDENCE_SCHEMA
        and evidence.get("formalReady") is True
        and evidence.get("gpuGate", {}).get("hardware") is True
        and before.get("nvidiaSmi", {}).get("available") is True
        and before.get("nvidiaSmiPmon", {}).get("available") is True
        and during.get("nvidiaSmi", {}).get("available") is True
        and during.get("nvidiaSmiPmon", {}).get("available") is True
    )
    if not formal_gpu:
        raise ValueError(f"{path}: GPU evidence is not formal-ready")
    pose_count = int(meta["poseCount"])
    layers = int(meta["maxLayers"])
    width = int(meta["width"])
    height = int(meta["height"])
    if min(pose_count, layers, width, height) <= 0:
        raise ValueError(f"{path}: invalid cache dimensions")
    files = meta["files"]
    render_pose_ids = np.fromfile(_resolve(path, files["poseIndices"]), dtype="<u4")
    if render_pose_ids.size != pose_count:
        raise ValueError(f"{path}: render pose count mismatch ({render_pose_ids.size} != {pose_count})")
    if np.unique(render_pose_ids).size != render_pose_ids.size:
        raise ValueError(f"{path}: duplicate render pose IDs")
    if meta.get("schema") == CACHE_SCHEMA_V2:
        first_layer_reference = meta.get("firstLayerReference") or {}
        if (
            int(first_layer_reference.get("checks", 0)) != pose_count
            or int(first_layer_reference.get("mismatches", 0)) != 0
        ):
            raise ValueError(
                f"{path}: v2 cache must prove one matching unpeeled Color-ID reference per pose"
            )
        source_name = files.get("sourcePoseIndices")
        render_name = files.get("renderPoseIds")
        if not source_name or not render_name:
            raise ValueError(f"{path}: v2 cache must declare sourcePoseIndices and renderPoseIds")
        explicit_render_ids = np.fromfile(_resolve(path, render_name), dtype="<u4")
        if explicit_render_ids.size != pose_count or not np.array_equal(explicit_render_ids, render_pose_ids):
            raise ValueError(f"{path}: explicit renderPoseIds do not match poseIndices")
        source_pose_indices = np.fromfile(_resolve(path, source_name), dtype="<u4")
        if source_pose_indices.size != pose_count:
            raise ValueError(f"{path}: source pose count mismatch ({source_pose_indices.size} != {pose_count})")
    else:
        source_pose_indices = render_pose_ids.copy()
    shape = (pose_count, layers, height, width)
    ids_path = _resolve(path, files["instanceIds"])
    depth_path = _resolve(path, files["linearDepth"])
    expected_bytes = int(np.prod(shape)) * 4
    if ids_path.stat().st_size != expected_bytes or depth_path.stat().st_size != expected_bytes:
        raise ValueError(f"{path}: binary cache size does not match metadata")
    return meta, render_pose_ids.astype(np.int64), source_pose_indices.astype(np.int64), np.memmap(ids_path, dtype="<u4", mode="r", shape=shape), np.memmap(depth_path, dtype="<f4", mode="r", shape=shape)


def merge_caches(
    shard_dirs: list[Path],
    output_dir: Path,
    expected_pose_indices: np.ndarray | None = None,
    candidate_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not shard_dirs:
        raise ValueError("at least one cache shard is required")
    loaded = [(path, *load_completed_cache(path)) for path in shard_dirs]
    reference = loaded[0][1]
    fixed = ("schema", "modelInputFovYDeg", "width", "height", "maxLayers")
    for path, meta, _render_ids, _source_ids, _ids, _depths in loaded[1:]:
        for key in fixed:
            if meta.get(key) != reference.get(key):
                raise ValueError(f"{path}: {key} does not match the first shard")
    if any(item[1].get("schema") != reference.get("schema") for item in loaded):
        raise ValueError("merged shards use different cache schemas")
    if reference.get("schema") == CACHE_SCHEMA_V2:
        if any(
            int(meta.get("firstLayerReference", {}).get("checks", 0)) != int(meta["poseCount"])
            or int(meta.get("firstLayerReference", {}).get("mismatches", 0)) != 0
            for _path, meta, _render_ids, _source_ids, _ids, _depths in loaded
        ):
            raise ValueError("v2 merge requires a matching unpeeled Color-ID reference for every pose")
    render_parts = [item[2] for item in loaded]
    source_parts = [item[3] for item in loaded]
    all_render_ids = np.concatenate(render_parts).astype(np.int64, copy=False)
    all_source_ids = np.concatenate(source_parts).astype(np.int64, copy=False)
    if np.unique(all_render_ids).size != all_render_ids.size:
        raise ValueError("merged shards contain duplicate render pose IDs")
    if expected_pose_indices is not None:
        expected = np.asarray(expected_pose_indices, dtype=np.int64)
        if not np.array_equal(np.sort(all_render_ids), np.sort(expected)):
            missing = np.setdiff1d(expected, all_render_ids)
            extra = np.setdiff1d(all_render_ids, expected)
            raise ValueError(f"render pose coverage mismatch: missing={missing[:8].tolist()}, extra={extra[:8].tolist()}")
    order = np.argsort(all_render_ids, kind="stable")
    output_dir.mkdir(parents=True, exist_ok=True)
    layers, height, width = int(reference["maxLayers"]), int(reference["height"]), int(reference["width"])
    shape = (all_render_ids.size, layers, height, width)
    ids_out = output_dir / "triangle_depth.bin.instance_ids.bin"
    depth_out = output_dir / "triangle_depth.bin.linear_depth.bin"
    poses_out = output_dir / "triangle_depth.bin.pose_indices.bin"
    ids_mem = np.memmap(ids_out, dtype="<u4", mode="w+", shape=shape)
    depth_mem = np.memmap(depth_out, dtype="<f4", mode="w+", shape=shape)
    pose_mem = np.memmap(poses_out, dtype="<u4", mode="w+", shape=(all_render_ids.size,))
    source_out = None
    source_mem = None
    render_out = None
    render_mem = None
    if reference.get("schema") == CACHE_SCHEMA_V2:
        source_out = output_dir / "triangle_depth.bin.source_pose_indices.bin"
        render_out = output_dir / "triangle_depth.bin.render_pose_ids.bin"
        source_mem = np.memmap(source_out, dtype="<u4", mode="w+", shape=(all_render_ids.size,))
        render_mem = np.memmap(render_out, dtype="<u4", mode="w+", shape=(all_render_ids.size,))
    for output_row, flat_index in enumerate(order.tolist()):
        shard_index = 0
        local_index = int(flat_index)
        for shard_index, (_path, _meta, render_ids, source_ids, ids, depths) in enumerate(loaded):
            if local_index < render_ids.size:
                ids_mem[output_row] = ids[local_index]
                depth_mem[output_row] = depths[local_index]
                pose_mem[output_row] = np.uint32(render_ids[local_index])
                if source_mem is not None and render_mem is not None:
                    source_mem[output_row] = np.uint32(source_ids[local_index])
                    render_mem[output_row] = np.uint32(render_ids[local_index])
                break
            local_index -= int(render_ids.size)
        else:  # pragma: no cover - guarded by concatenated size
            raise RuntimeError("failed to locate merged pose row")
    ids_mem.flush(); depth_mem.flush(); pose_mem.flush()
    if source_mem is not None and render_mem is not None:
        source_mem.flush(); render_mem.flush()
    schema = reference["schema"]
    files = {"poseIndices": poses_out.name, "instanceIds": ids_out.name, "linearDepth": depth_out.name}
    if schema == CACHE_SCHEMA_V2:
        files.update({"sourcePoseIndices": source_out.name, "renderPoseIds": render_out.name})
    meta = {
        "schema": schema,
        "modelInputFovYDeg": reference["modelInputFovYDeg"],
        "width": width,
        "height": height,
        "maxLayers": layers,
        "poseCount": int(all_render_ids.size),
        "sourceShards": [str(path.resolve()) for path in shard_dirs],
        "poseIdSemantics": reference.get("poseIdSemantics", "poseIndices stores renderPoseId; sourcePoseIndices stores sourcePoseIndex") if schema == CACHE_SCHEMA_V2 else "poseIndices stores canonical PoseCSR poseIndex",
        "gpuEvidence": "all source shards passed triangle-depth-layer-gpu-evidence-v1 formalReady=true",
        "firstLayerReference": {
            "checks": int(sum(int(meta.get("firstLayerReference", {}).get("checks", 0)) for _path, meta, *_rest in loaded)),
            "mismatches": int(sum(int(meta.get("firstLayerReference", {}).get("mismatches", 0)) for _path, meta, *_rest in loaded)),
            "semantics": "each cache layer zero was compared byte-for-byte with a repeated unpeeled Color-ID pass",
        },
        "files": files,
    }
    if candidate_identity is not None:
        meta["candidateIdentity"] = candidate_identity
    elif reference.get("candidateIdentity") is not None:
        meta["candidateIdentity"] = reference["candidateIdentity"]
    # Preserve the v2 render manifest so downstream evidence construction can
    # recover the actual subpose camera instead of silently using the
    # canonical view-cell pose.
    if reference.get("sourceManifest"):
        meta["sourceManifest"] = reference["sourceManifest"]
    if reference.get("assetsDir"):
        meta["assetsDir"] = reference["assetsDir"]
    (output_dir / "layer_cache_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def _candidate_identity_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any] | None:
    declared = manifest.get("candidateIdentity")
    if isinstance(declared, dict) and declared.get("canonicalCandidateDigest") and declared.get("renderCandidateDigest"):
        return declared
    dataset_value = manifest.get("datasetDir")
    split = str(manifest.get("split", ""))
    if not dataset_value or not split:
        return None
    dataset_dir = _resolve(manifest_path.parent, str(dataset_value))
    dataset = PoseCSRDataset(dataset_dir, num_instances=1)
    canonical = pose_sequence_for_splits(dataset, [split])
    render = np.asarray(
        [int(row.get("sourcePoseIndex", row.get("poseIndex", index))) for index, row in enumerate(manifest.get("poses", []))],
        dtype=np.int64,
    )
    return {
        "canonicalCandidateDigest": candidate_digest_for_pose_sequence(dataset, canonical),
        "canonicalCandidateDigestScope": "all stored candidate CSR rows in the selected dataset split, canonical PoseCSR order",
        "renderCandidateDigest": candidate_digest_for_pose_sequence(dataset, render),
        "renderCandidateDigestScope": "stored candidate CSR rows repeated in this manifest renderPose order",
        "canonicalPoseCount": int(canonical.size),
        "renderPoseCount": int(render.size),
        "derivedByMergeTool": True,
    }


def _split_pose_indices(dataset_dir: Path, split: str) -> np.ndarray:
    meta = _read_json(dataset_dir / "dataset_meta.json")
    split_id = int(meta["splitIds"][split])
    pose_dtype = np.dtype(meta.get("poseDtype", "<f4"))
    # The current CSR pose record is fixed-size and its split byte is exposed
    # by pose_csr_dataset; import it rather than guessing the binary layout.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from pose_csr_dataset import PoseCSRDataset
    dataset = PoseCSRDataset(dataset_dir, num_instances=1)
    return np.flatnonzero(dataset.poses["split"] == split_id).astype(np.int64)


def self_test() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        expected = np.asarray([2, 0, 1, 3], dtype=np.int64)
        for shard_number, poses in enumerate((expected[::2], expected[1::2])):
            shard = root / f"shard_{shard_number}"
            shard.mkdir()
            shape = (poses.size, 2, 2, 3)
            ids_name, depth_name, pose_name = "ids.bin", "depth.bin", "poses.bin"
            np.arange(np.prod(shape), dtype="<u4").reshape(shape).tofile(shard / ids_name)
            np.arange(np.prod(shape), dtype="<f4").reshape(shape).tofile(shard / depth_name)
            poses.astype("<u4").tofile(shard / pose_name)
            (shard / "layer_cache_meta.json").write_text(json.dumps({"schema": CACHE_SCHEMA, "modelInputFovYDeg": 66, "width": 3, "height": 2, "maxLayers": 2, "poseCount": poses.size, "gpuGate": {"hardware": True}, "files": {"poseIndices": pose_name, "instanceIds": ids_name, "linearDepth": depth_name}}), encoding="utf-8")
            (shard / "run.gpu_evidence.json").write_text(json.dumps({
                "schema": EVIDENCE_SCHEMA,
                "formalReady": True,
                "gpuGate": {"hardware": True},
                "hostGpuBefore": {
                    "nvidiaSmi": {"available": True},
                    "nvidiaSmiPmon": {"available": True},
                },
                "hostGpuDuring": {
                    "nvidiaSmi": {"available": True},
                    "nvidiaSmiPmon": {"available": True},
                },
            }), encoding="utf-8")
        result = merge_caches([root / "shard_0", root / "shard_1"], root / "merged", expected)
        assert result["poseCount"] == 4
        assert np.array_equal(np.fromfile(root / "merged" / "poses.bin" if False else root / "merged" / result["files"]["poseIndices"], dtype="<u4"), np.asarray([0, 1, 2, 3], dtype="<u4"))
    return {"status": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=False)
    parser.add_argument("--shard-root", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="v2 render manifest; validates the complete renderPoseId set")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2)); return
    shards = list(args.shard or [])
    if args.shard_root is not None:
        shards.extend(sorted(path for path in args.shard_root.glob("shard_*") if path.is_dir()))
    if args.output_dir is None or not shards:
        raise ValueError("--output-dir and at least one --shard/--shard-root are required")
    if args.manifest is not None:
        manifest = _read_json(args.manifest)
        expected = np.asarray([int(row.get("renderPoseId", index)) for index, row in enumerate(manifest.get("poses", []))], dtype=np.int64)
        candidate_identity = _candidate_identity_from_manifest(manifest, args.manifest)
    else:
        expected = _split_pose_indices(args.dataset_dir, args.split) if args.dataset_dir is not None else None
        candidate_identity = None
    result = merge_caches(shards, args.output_dir, expected, candidate_identity)
    if args.manifest is not None:
        # A shard stores its own sliced manifest.  The merged cache must point
        # to the complete render manifest so downstream pose-count and camera
        # identity validation covers every merged row.
        result["sourceManifest"] = str(args.manifest.resolve())
        (args.output_dir / "layer_cache_meta.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
