#!/usr/bin/env python3
"""Write reproducible scene, asset, split, candidate, and GT statistics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402

from aabb_ray_baseline_config import DATA_ROOT, SCENES, scene_paths  # noqa: E402


SPLITS = ("train", "calibration", "validation", "test")
_GLB_HEADER = struct.Struct("<4sII")
_GLB_CHUNK = struct.Struct("<II")


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < _GLB_HEADER.size:
        raise ValueError(f"GLB is shorter than its header: {path}")
    magic, version, total_length = _GLB_HEADER.unpack_from(data, 0)
    if magic != b"glTF" or version != 2 or total_length != len(data):
        raise ValueError(f"invalid GLB header: {path}")
    cursor = _GLB_HEADER.size
    while cursor + _GLB_CHUNK.size <= len(data):
        chunk_length, chunk_type = _GLB_CHUNK.unpack_from(data, cursor)
        cursor += _GLB_CHUNK.size
        end = cursor + int(chunk_length)
        if end > len(data):
            raise ValueError(f"GLB chunk exceeds file length: {path}")
        if chunk_type == 0x4E4F534A:
            try:
                value = json.loads(data[cursor:end].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid GLB JSON chunk: {path}") from error
            if not isinstance(value, dict):
                raise ValueError(f"GLB JSON chunk is not an object: {path}")
            return value
        cursor = end
    raise ValueError(f"GLB has no JSON chunk: {path}")


def _accessor_count(document: Mapping[str, Any], accessor_index: Any) -> int:
    accessors = document.get("accessors")
    if not isinstance(accessors, list):
        raise ValueError("GLB mesh references an absent accessor table")
    index = int(accessor_index)
    if index < 0 or index >= len(accessors) or not isinstance(accessors[index], Mapping):
        raise ValueError("GLB primitive references an invalid accessor")
    count = int(accessors[index].get("count", -1))
    if count < 0:
        raise ValueError("GLB accessor has an invalid count")
    return count


def glb_triangle_count(path: str | Path) -> int:
    """Count indexed or non-indexed triangles in one prototype GLB."""
    document = _read_glb_json(Path(path))
    meshes = document.get("meshes", [])
    if not isinstance(meshes, list):
        raise ValueError(f"GLB meshes is not a list: {path}")
    triangles = 0
    for mesh in meshes:
        if not isinstance(mesh, Mapping):
            raise ValueError(f"GLB mesh is not an object: {path}")
        primitives = mesh.get("primitives", [])
        if not isinstance(primitives, list):
            raise ValueError(f"GLB mesh primitives is not a list: {path}")
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                raise ValueError(f"GLB primitive is not an object: {path}")
            if "indices" in primitive:
                vertex_count = _accessor_count(document, primitive["indices"])
            else:
                attributes = primitive.get("attributes", {})
                if not isinstance(attributes, Mapping) or "POSITION" not in attributes:
                    continue
                vertex_count = _accessor_count(document, attributes["POSITION"])
            mode = int(primitive.get("mode", 4))
            if mode == 4:
                triangles += vertex_count // 3
            elif mode in (5, 6):
                triangles += max(0, vertex_count - 2)
    return int(triangles)


def _distribution(values: np.ndarray, prefix: str) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            f"{prefix}_min": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_p95": 0.0,
            f"{prefix}_max": 0.0,
        }
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{prefix} contains non-finite values")
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_max": float(np.max(array)),
    }


def _split_statistics(dataset: PoseCSRDataset, split_name: str) -> dict[str, Any]:
    if split_name not in dataset.split_ids:
        return {f"{split_name}_pose_count": 0}
    split = dataset.split(split_name)
    indices = np.asarray(split.pose_indices, dtype=np.int64)
    candidates = np.asarray(dataset.candidate_counts[indices], dtype=np.float64)
    gt = np.asarray(dataset.visible_counts[indices], dtype=np.float64)
    weights = np.asarray(
        [float(np.asarray(dataset.visible_slice(int(pose))[1], dtype=np.float64).sum()) for pose in indices.tolist()],
        dtype=np.float64,
    )
    result: dict[str, Any] = {f"{split_name}_pose_count": int(indices.size)}
    result.update(_distribution(candidates, f"{split_name}_candidate_count"))
    result.update(_distribution(gt, f"{split_name}_gt_count"))
    result.update(_distribution(weights, f"{split_name}_visible_weight_sum"))
    result.update(
        {
            f"{split_name}_candidate_reference_count": int(candidates.sum()),
            f"{split_name}_gt_reference_count": int(gt.sum()),
            f"{split_name}_positive_rate": float(gt.sum() / candidates.sum()) if candidates.sum() > 0 else 0.0,
            f"{split_name}_gt_over_candidate": float(gt.sum() / candidates.sum()) if candidates.sum() > 0 else 0.0,
            f"{split_name}_zero_gt_pose_count": int(np.count_nonzero(gt == 0.0)),
            f"{split_name}_empty_candidate_pose_count": int(np.count_nonzero(candidates == 0.0)),
            f"{split_name}_total_visible_weight": float(weights.sum()),
        }
    )
    return result


def collect_scene_statistics(
    scene_name: str,
    dataset_dir: str | Path,
    runtime_meta_path: str | Path,
    glb_index_path: str | Path,
    glb_root: str | Path,
) -> dict[str, Any]:
    """Collect one row without reading any test predictions."""
    runtime_path = Path(runtime_meta_path).resolve()
    index_path = Path(glb_index_path).resolve()
    root = Path(glb_root).resolve()
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime metadata must be an object")
    world_aabbs, instance_to_glb, _ = load_runtime_meta(runtime_path)
    instance_count = int(world_aabbs.shape[0])
    mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
    if mapping.size != instance_count or (mapping.size and int(mapping.min()) < 0):
        raise ValueError("runtime instance-to-GLB mapping is invalid")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries") if isinstance(index, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"GLB index has no entries: {index_path}")
    entry_by_id: dict[int, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("GLB index entry is not an object")
        gid = int(entry.get("globalId", -1))
        if gid in entry_by_id or gid < 0:
            raise ValueError("GLB index IDs must be unique and non-negative")
        entry_by_id[gid] = entry
    glb_count = max(entry_by_id) + 1
    if sorted(entry_by_id) != list(range(glb_count)):
        raise ValueError("GLB index IDs must form a dense range")
    if mapping.size and int(mapping.max()) >= glb_count:
        raise ValueError("runtime mapping references a GLB absent from glbIndex")

    glb_bytes = np.zeros((glb_count,), dtype=np.float64)
    prototype_triangles = np.zeros((glb_count,), dtype=np.float64)
    for gid in range(glb_count):
        relative = str(entry_by_id[gid].get("path", ""))
        path = root / relative
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing registered GLB: {path}")
        glb_bytes[gid] = float(path.stat().st_size)
        prototype_triangles[gid] = float(glb_triangle_count(path))

    bounds = runtime.get("sceneBounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("runtime metadata has no sceneBounds")
    _scene_min, _scene_max, scene_size = scene_min_max(bounds)
    scene_size = np.asarray(scene_size, dtype=np.float64)
    if scene_size.shape != (3,) or not bool(np.isfinite(scene_size).all()):
        raise ValueError("scene bounds size is invalid")

    dataset_path = Path(dataset_dir).resolve()
    dataset = PoseCSRDataset(dataset_path, num_instances=instance_count)
    row: dict[str, Any] = {
        "scene": str(scene_name),
        "dataset_dir": str(dataset_path),
        "runtime_meta": str(runtime_path),
        "glb_index": str(index_path),
        "instance_count": instance_count,
        "glb_count": glb_count,
        "scene_size_x": float(scene_size[0]),
        "scene_size_y": float(scene_size[1]),
        "scene_size_z": float(scene_size[2]),
        "scene_diagonal": float(np.linalg.norm(scene_size)),
        "prototype_triangle_count": int(prototype_triangles.sum()),
        "expanded_triangle_count": int(prototype_triangles[mapping].sum()) if mapping.size else 0,
        "instance_glb_reuse_factor": float(instance_count / glb_count),
        "triangle_reuse_factor": float(prototype_triangles[mapping].sum() / max(1.0, prototype_triangles.sum())) if mapping.size else 0.0,
        "glb_bytes_total": float(glb_bytes.sum()),
        "glb_bytes_per_instance": float(glb_bytes.sum() / max(1, instance_count)),
        "prototype_triangle_per_instance_mean": float(prototype_triangles[mapping].mean()) if mapping.size else 0.0,
        "expanded_triangle_per_instance_mean": float(prototype_triangles[mapping].sum() / max(1, instance_count)) if mapping.size else 0.0,
    }
    row.update(_distribution(glb_bytes, "glb_bytes"))
    row.update(_distribution(prototype_triangles, "prototype_triangles"))
    for split_name in SPLITS:
        row.update(_split_statistics(dataset, split_name))
    return row


def write_scene_statistics_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError("cannot write an empty scene statistics table")
    fields = list(values[0])
    for row in values[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out/paper_results/scene_statistics.csv",
    )
    args = parser.parse_args(argv)
    names = [value.strip() for value in str(args.scenes).split(",") if value.strip()]
    if not names:
        parser.error("--scenes must contain at least one registered scene")
    rows = []
    for name in names:
        paths = scene_paths(args.data_root, name)
        rows.append(collect_scene_statistics(name, paths["dataset"], paths["runtimeMeta"], paths["glbIndex"], paths["glbRoot"]))
    output = write_scene_statistics_csv(args.output, rows)
    print(json.dumps({"output": str(output), "scenes": names}, ensure_ascii=False))


if __name__ == "__main__":
    main()
