"""Explicit input loading for the GLB streaming experiment."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
MODEL_DIR = BENCHMARK_DIR.parent / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pose_csr_dataset import PoseCSRDataset, _project_aabb_features_numpy  # noqa: E402

from glb_streaming import GlbAsset, PoseRecord, StreamingContractError


def require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def read_json(path: str | Path, label: str = "JSON file") -> Any:
    resolved = require_file(path, label)
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise StreamingContractError(f"invalid {label}: {resolved}: {error}") from error


def _bounds_array(value: Mapping[str, Any], label: str) -> np.ndarray:
    if "min" not in value or "max" not in value:
        raise StreamingContractError(f"{label} must contain min and max")
    try:
        result = np.asarray(list(value["min"]) + list(value["max"]), dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise StreamingContractError(f"{label} has invalid bounds") from error
    if result.shape != (6,) or not np.all(np.isfinite(result)) or np.any(result[3:] < result[:3]):
        raise StreamingContractError(f"{label} has non-finite or inverted bounds")
    return result


def _union_component_bounds(
    component_records: list[Mapping[str, Any]],
    component_ids: Iterable[int],
) -> np.ndarray:
    selected = [component_records[int(index)] for index in component_ids]
    if not selected:
        return np.zeros((6,), dtype=np.float32)
    bounds = np.stack([_bounds_array(row["bounds"], "component bounds") for row in selected], axis=0)
    return np.concatenate([bounds[:, :3].min(axis=0), bounds[:, 3:].max(axis=0)]).astype(np.float32)


def load_glb_assets(
    runtime_meta_path: str | Path,
    glb_index_path: str | Path,
    glb_root: str | Path,
) -> tuple[dict[int, GlbAsset], dict[int, int], dict[str, Any]]:
    """Load byte sizes, original order, AABBs, and instance-to-GLB mapping."""

    runtime_path = require_file(runtime_meta_path, "runtime metadata")
    index_path = require_file(glb_index_path, "GLB index")
    root = Path(glb_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"GLB root does not exist: {root}")
    runtime = read_json(runtime_path, "runtime metadata")
    index = read_json(index_path, "GLB index")
    component_records = runtime.get("componentRecords")
    glb_records = runtime.get("globalGlbRecords")
    entries = index.get("entries")
    if not isinstance(component_records, list) or not isinstance(glb_records, list) or not isinstance(entries, list):
        raise StreamingContractError("runtime metadata and GLB index are missing record arrays")
    expected_instances = int(runtime.get("instanceCount", len(component_records)))
    expected_glbs = int(runtime.get("globalGlbCount", len(entries)))
    if len(component_records) != expected_instances or len(entries) != expected_glbs:
        raise StreamingContractError("runtime metadata and GLB index counts are inconsistent")

    instance_to_glb: dict[int, int] = {}
    for fallback_id, record in enumerate(component_records):
        instance_id = int(record.get("instanceId", record.get("componentGlobalId", fallback_id)))
        glb_id = int(record.get("globalGlbId", -1))
        if instance_id in instance_to_glb or instance_id < 0 or glb_id < 0:
            raise StreamingContractError(f"invalid or duplicate instance mapping at row {fallback_id}")
        instance_to_glb[instance_id] = glb_id
    if set(instance_to_glb) != set(range(expected_instances)):
        raise StreamingContractError("runtime metadata does not provide a dense instance-to-GLB mapping")
    if any(glb_id >= expected_glbs for glb_id in instance_to_glb.values()):
        raise StreamingContractError("instance-to-GLB mapping points outside the GLB inventory")

    record_by_glb: dict[int, Mapping[str, Any]] = {}
    for record in glb_records:
        glb_id = int(record.get("globalGlbId", -1))
        if glb_id in record_by_glb or glb_id < 0:
            raise StreamingContractError(f"invalid or duplicate global GLB record: {glb_id}")
        record_by_glb[glb_id] = record
    if set(record_by_glb) != set(range(expected_glbs)):
        raise StreamingContractError("runtime metadata GLB records are not a dense inventory")

    assets: dict[int, GlbAsset] = {}
    seen_paths: set[Path] = set()
    for original_rank, entry in enumerate(entries):
        glb_id = int(entry.get("globalId", -1))
        if glb_id in assets or glb_id < 0 or glb_id >= expected_glbs:
            raise StreamingContractError(f"invalid or duplicate glbIndex globalId: {glb_id}")
        relative = str(entry.get("path") or "")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise StreamingContractError(f"GLB path escapes --glb-root: {relative}") from error
        if not candidate.is_file() or candidate.stat().st_size <= 0:
            raise FileNotFoundError(f"missing or empty candidate GLB {glb_id}: {candidate}")
        if candidate in seen_paths:
            raise StreamingContractError(f"multiple GLB IDs refer to the same file: {candidate}")
        seen_paths.add(candidate)
        record = record_by_glb[glb_id]
        raw_aabb = record.get("aabb") or record.get("bounds")
        aabb = _bounds_array(raw_aabb, f"GLB {glb_id} AABB") if isinstance(raw_aabb, Mapping) else None
        if aabb is None:
            component_ids = [
                int(value)
                for value in (record.get("componentGlobalIds") or [])
            ]
            aabb = _union_component_bounds(component_records, component_ids)
        assets[glb_id] = GlbAsset(
            global_id=glb_id,
            byte_size=int(candidate.stat().st_size),
            original_rank=int(original_rank),
            aabb=aabb,
        )
    if set(assets) != set(range(expected_glbs)):
        raise StreamingContractError("glbIndex does not cover the complete runtime GLB inventory")
    return assets, instance_to_glb, {
        "sceneName": str(
            runtime.get("sceneName")
            or runtime_path.parent.parent.name
            or runtime_path.parent.name
        ),
        "runtimeMeta": str(runtime_path),
        "glbIndex": str(index_path),
        "glbRoot": str(root),
        "instanceCount": expected_instances,
        "globalGlbCount": expected_glbs,
        "indexOrdering": index.get("ordering"),
    }


@dataclass(frozen=True)
class LoadedPose:
    record: PoseRecord
    candidate_instance_ids: np.ndarray
    visible_instance_ids: np.ndarray
    camera_world: np.ndarray
    camera_forward: np.ndarray
    camera_view: np.ndarray
    mvp: np.ndarray | None


def _load_jsonl_rows(path: str | Path, label: str) -> list[dict[str, Any]]:
    resolved = require_file(path, label)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise StreamingContractError(f"invalid {label} line {line_number}") from error
            if not isinstance(value, dict):
                raise StreamingContractError(f"{label} line {line_number} is not an object")
            rows.append(value)
    return rows


def load_pose_sidecar(path: str | Path | None, field: str, label: str) -> dict[int, Any]:
    """Read a JSONL sidecar keyed by poseId, accepting one object per pose."""

    if path is None:
        return {}
    resolved = require_file(path, label)
    if resolved.suffix.lower() == ".jsonl":
        rows = _load_jsonl_rows(resolved, label)
    else:
        payload = read_json(resolved, label)
        if isinstance(payload, list):
            rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict) and isinstance(payload.get("poses"), list):
            rows = [row for row in payload["poses"] if isinstance(row, dict)]
        elif isinstance(payload, dict):
            rows = []
            for pose_id, value in payload.items():
                rows.append({"poseId": int(pose_id), field: value})
        else:
            raise StreamingContractError(f"{label} must be JSONL, a pose list, or a pose map")
    result: dict[int, Any] = {}
    for row in rows:
        if "poseId" not in row or field not in row:
            raise StreamingContractError(f"{label} row must contain poseId and {field}")
        pose_id = int(row["poseId"])
        if pose_id in result:
            raise StreamingContractError(f"{label} contains duplicate poseId {pose_id}")
        result[pose_id] = row[field]
    return result


def _histogram_to_utility(value: Any, label: str) -> dict[int, float]:
    if not isinstance(value, Mapping):
        raise StreamingContractError(f"{label} histogram must be an object keyed by GLB ID")
    result: dict[int, float] = {}
    for key, count in value.items():
        glb_id = int(key)
        pixels = float(count)
        if glb_id < 0 or not np.isfinite(pixels) or pixels < 0.0:
            raise StreamingContractError(f"{label} contains invalid pixel count")
        result[glb_id] = pixels
    return result


def load_pose_inputs(
    dataset_dir: str | Path,
    runtime_meta_path: str | Path,
    assets: Mapping[int, GlbAsset],
    instance_to_glb: Mapping[int, int],
    *,
    split: str = "test",
    pose_limit: int = 0,
    pose_ids: Iterable[int] | None = None,
    utility_source: str = "binary_gt",
    reference_frontmost_path: str | Path | None = None,
    hzb_visible_path: str | Path | None = None,
) -> list[LoadedPose]:
    """Load the exact candidate/GT rows and attach optional utility/HZB data."""

    runtime = read_json(runtime_meta_path, "runtime metadata")
    dataset = PoseCSRDataset(dataset_dir, num_instances=len(instance_to_glb))
    if pose_ids is None:
        selected = dataset.split(split).pose_indices.astype(np.int64, copy=False).tolist()
    else:
        selected = [int(value) for value in pose_ids]
    if pose_limit > 0:
        selected = selected[: int(pose_limit)]
    if not selected:
        raise StreamingContractError(f"no poses selected from split {split}")

    frontmost = load_pose_sidecar(
        reference_frontmost_path,
        "histogram",
        "reference-frontmost sidecar",
    )
    hzb = load_pose_sidecar(hzb_visible_path, "hzbVisibleGlbIds", "HZB visible sidecar")
    if utility_source not in {"binary_gt", "visible_weights", "reference_frontmost_pixels"}:
        raise StreamingContractError(f"unknown utility source: {utility_source}")
    if utility_source == "reference_frontmost_pixels" and not reference_frontmost_path:
        raise StreamingContractError("reference-frontmost-pixels utility requires --reference-frontmost")

    component_records = runtime.get("componentRecords") or []
    if len(component_records) != len(instance_to_glb):
        raise StreamingContractError("runtime metadata instance count disagrees with the dataset")
    loaded: list[LoadedPose] = []
    for ordinal, pose_id in enumerate(selected):
        if pose_id < 0 or pose_id >= dataset.poses.size:
            raise StreamingContractError(f"selected pose ID is outside poses.bin: {pose_id}")
        candidate_instances = np.asarray(dataset.candidate_slice(pose_id), dtype=np.uint32)
        visible_instances, visible_weights = dataset.visible_slice(pose_id)
        visible_instances = np.asarray(visible_instances, dtype=np.uint32)
        invalid_candidates = [
            int(value) for value in candidate_instances.tolist() if int(value) not in instance_to_glb
        ]
        invalid_visible = [
            int(value) for value in visible_instances.tolist() if int(value) not in instance_to_glb
        ]
        if invalid_candidates or invalid_visible:
            raise StreamingContractError(
                f"pose {pose_id} has instance IDs outside runtime metadata: "
                f"candidates={invalid_candidates[:8]}, visible={invalid_visible[:8]}"
            )
        if np.unique(candidate_instances).size != candidate_instances.size:
            raise StreamingContractError(f"pose {pose_id} has duplicate candidate instance IDs")
        if np.unique(visible_instances).size != visible_instances.size:
            raise StreamingContractError(f"pose {pose_id} has duplicate visible instance IDs")
        candidate_glbs = tuple(sorted({int(instance_to_glb[int(value)]) for value in candidate_instances}))
        gt_glbs = tuple(sorted({int(instance_to_glb[int(value)]) for value in visible_instances}))
        utility: dict[int, float] = {}
        if utility_source == "visible_weights":
            for instance_id, weight in zip(visible_instances.tolist(), visible_weights.tolist()):
                glb_id = int(instance_to_glb[int(instance_id)])
                utility[glb_id] = utility.get(glb_id, 0.0) + float(weight)
        elif utility_source == "reference_frontmost_pixels":
            if pose_id not in frontmost:
                raise StreamingContractError(f"reference-frontmost sidecar has no selected pose {pose_id}")
            utility = _histogram_to_utility(frontmost[pose_id], f"pose {pose_id}")
        camera_world = np.asarray(dataset.poses["camera_world"][pose_id], dtype=np.float32)
        camera_forward = np.asarray(dataset.poses["camera_forward"][pose_id], dtype=np.float32)
        camera_view = dataset.camera_view(pose_id)
        mvp = dataset.mvp_slice(pose_id).astype(np.float32, copy=False) if dataset.mvp is not None else None
        loaded.append(
            LoadedPose(
                record=PoseRecord(
                    pose_id=pose_id,
                    ordinal=ordinal,
                    candidate_glb_ids=candidate_glbs,
                    gt_glb_ids=gt_glbs,
                    utility_by_glb=utility,
                    hzb_visible_glb_ids=(
                        tuple(int(value) for value in hzb[pose_id])
                        if pose_id in hzb
                        else None
                    ),
                ),
                candidate_instance_ids=candidate_instances,
                visible_instance_ids=visible_instances,
                camera_world=camera_world,
                camera_forward=camera_forward,
                camera_view=camera_view,
                mvp=mvp,
            )
        )
    return loaded


def attach_geometry_scores(
    poses: list[LoadedPose],
    assets: Mapping[int, GlbAsset],
    score_by_pose: Mapping[int, Mapping[str, np.ndarray]],
    instance_to_glb: Mapping[int, int],
) -> list[PoseRecord]:
    """Aggregate instance scores and add geometry-only score maps per GLB."""

    result: list[PoseRecord] = []
    glb_aabbs = np.stack([assets[index].aabb for index in sorted(assets)], axis=0).astype(np.float32)
    glb_ids = sorted(assets)
    glb_index = {glb_id: index for index, glb_id in enumerate(glb_ids)}
    for loaded in poses:
        pose = loaded.record
        candidate = np.asarray(pose.candidate_glb_ids, dtype=np.int64)
        rank_scores: dict[str, dict[int, float]] = {}
        score_fields = score_by_pose.get(pose.pose_id, {})
        for method in ("full", "aabb"):
            values = score_fields.get(method)
            if values is None:
                continue
            if values.shape != loaded.candidate_instance_ids.shape:
                raise StreamingContractError(
                    f"pose {pose.pose_id} {method} scores do not match candidate instance rows"
                )
            grouped: dict[int, float] = {}
            for instance_id, score in zip(loaded.candidate_instance_ids.tolist(), values.tolist()):
                glb_id = int(instance_to_glb[int(instance_id)])
                grouped[glb_id] = max(grouped.get(glb_id, 0.0), float(score))
            rank_scores[method] = grouped

        if loaded.camera_world.shape != (3,):
            raise StreamingContractError(f"pose {pose.pose_id} camera position is invalid")
        selected_indices = np.asarray([glb_index[index] for index in candidate.tolist()], dtype=np.int64)
        mins = glb_aabbs[selected_indices, :3]
        maxs = glb_aabbs[selected_indices, 3:]
        delta = np.maximum(np.maximum(mins - loaded.camera_world, loaded.camera_world - maxs), 0.0)
        distance = np.linalg.norm(delta, axis=1)
        rank_scores["distance"] = {
            int(glb_id): float(1.0 / (1.0 + distance[index]))
            for index, glb_id in enumerate(candidate.tolist())
        }
        if loaded.mvp is not None:
            _rect, area, _depth, valid = _project_aabb_features_numpy(
                glb_aabbs[selected_indices], loaded.mvp
            )
            area = np.where(valid, np.clip(area, 0.0, 1.0), 0.0)
            rank_scores["projected_area"] = {
                int(glb_id): float(area[index]) for index, glb_id in enumerate(candidate.tolist())
            }
            rank_scores["projected_area_per_byte"] = {
                int(glb_id): float(area[index]) / max(1, assets[int(glb_id)].byte_size)
                for index, glb_id in enumerate(candidate.tolist())
            }
        result.append(
            PoseRecord(
                pose_id=pose.pose_id,
                ordinal=pose.ordinal,
                candidate_glb_ids=pose.candidate_glb_ids,
                gt_glb_ids=pose.gt_glb_ids,
                utility_by_glb=pose.utility_by_glb,
                rank_scores=rank_scores,
                hzb_visible_glb_ids=pose.hzb_visible_glb_ids,
            )
        )
    return result


def load_score_sidecar(
    result_dir: str | Path,
    dataset_dir: str | Path,
    selected_pose_ids: Iterable[int],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    """Read score arrays and prove they align with the selected CSR rows."""

    root = Path(result_dir).expanduser().resolve()
    manifest = read_json(root / "score_manifest.json", "streaming score manifest")
    if manifest.get("schema") != "pvs-glb-streaming-score-sidecar-v1":
        raise StreamingContractError("unsupported streaming score sidecar schema")
    arrays_path = require_file(root / str(manifest.get("arraysFile", "scores.npz")), "streaming score arrays")
    arrays = np.load(arrays_path, allow_pickle=False)
    required = {"pose_ids", "pose_offsets", "candidate_ids"}
    if not required.issubset(set(arrays.files)):
        raise StreamingContractError("streaming score sidecar is missing alignment arrays")
    pose_ids = np.asarray(arrays["pose_ids"], dtype=np.int64)
    offsets = np.asarray(arrays["pose_offsets"], dtype=np.int64)
    candidate_ids = np.asarray(arrays["candidate_ids"], dtype=np.uint32)
    if offsets.size != pose_ids.size + 1 or int(offsets[0]) != 0 or int(offsets[-1]) != candidate_ids.size:
        raise StreamingContractError("streaming score sidecar offsets are invalid")
    if np.any(offsets[1:] < offsets[:-1]) or np.unique(pose_ids).size != pose_ids.size:
        raise StreamingContractError("streaming score sidecar pose alignment is invalid")
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(manifest["numInstances"]))
    expected_pose_ids = [int(value) for value in selected_pose_ids]
    sidecar_rows = {int(pose_id): row for row, pose_id in enumerate(pose_ids.tolist())}
    missing_pose_ids = [pose_id for pose_id in expected_pose_ids if pose_id not in sidecar_rows]
    if missing_pose_ids:
        raise StreamingContractError(
            f"score sidecar is missing selected pose IDs: {missing_pose_ids[:8]}"
        )
    for pose_id in expected_pose_ids:
        row = sidecar_rows[pose_id]
        start, end = int(offsets[row]), int(offsets[row + 1])
        expected = np.asarray(dataset.candidate_slice(pose_id), dtype=np.uint32)
        if not np.array_equal(candidate_ids[start:end], expected):
            raise StreamingContractError(f"score sidecar candidate rows disagree at pose {pose_id}")
    fields = [str(value) for value in manifest.get("scoreFields", [])]
    output: dict[int, dict[str, np.ndarray]] = {}
    for field in fields:
        if field not in arrays.files:
            raise StreamingContractError(f"score sidecar manifest requests missing field {field}")
        values = np.asarray(arrays[field], dtype=np.float32)
        if values.shape != candidate_ids.shape or not np.all(np.isfinite(values)):
            raise StreamingContractError(f"score sidecar field {field} is invalid")
        for pose_id in expected_pose_ids:
            row = sidecar_rows[pose_id]
            start, end = int(offsets[row]), int(offsets[row + 1])
            output.setdefault(pose_id, {})[field] = values[start:end]
    return output, manifest
