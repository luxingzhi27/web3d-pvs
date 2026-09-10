#!/usr/bin/env python3
"""Convert a complete HZB Region66 test result to a formal-v2 image manifest."""
from __future__ import annotations

import argparse
import array
import copy
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Sequence

try:
    from .instance_id_render_schema import (
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
        PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
        PREDICTION_KEY_FIELD,
        validate_formal_instance_render_manifest,
    )
    from .run_test_image_evaluation import validate_test_manifest
except ImportError:
    from instance_id_render_schema import (
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
        PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
        PREDICTION_KEY_FIELD,
        validate_formal_instance_render_manifest,
    )
    from run_test_image_evaluation import validate_test_manifest


HZB_RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v2"
HZB_WORKLOAD_SCHEMA = "geometry-shell-hzb-browser-workload-v1"
HZB_REGION_SCHEMA = "geometry-shell-hzb-region-sampling-v1"
POSE_STRIDE_BYTES = 64
POSE_SPLIT_OFFSET_BYTES = 44
FOV_TOLERANCE = 1e-6


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and positive")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and positive") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _require_fov(value: Any, expected: float, label: str) -> float:
    actual = _finite_positive(value, label)
    if abs(actual - expected) > FOV_TOLERANCE:
        raise ValueError(f"{label} must be {expected:g} degrees, got {actual:g}")
    return actual


def _absolute_path(value: Any, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _scene_from_asset_root(value: Any, label: str) -> str:
    path = _absolute_path(value, label)
    if path.name == "assets":
        return path.parent.name
    return path.name

def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing JSON input: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    return value


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer: {value!r}")
    return value


def _ids(value: Any, label: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list of component IDs")
    result = [_int(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if len(result) != len(set(result)):
        raise ValueError(f"{label} contains duplicate component IDs")
    return result


def _base_rows(base: dict[str, Any]) -> dict[int, str]:
    rows: dict[int, str] = {}
    for index, sample in enumerate(base.get("samples") or []):
        row = _int(sample.get("viewcellRow"), f"base sample {index} viewcellRow")
        key = sample.get(PREDICTION_KEY_FIELD)
        if not isinstance(key, str) or not key:
            raise ValueError(f"base sample {index} is missing {PREDICTION_KEY_FIELD}")
        if row in rows and rows[row] != key:
            raise ValueError(f"base view-cell row {row} uses multiple prediction keys")
        rows[row] = key
    if not rows:
        raise ValueError("base formal manifest must contain view-cell samples")
    return rows


def _base_pose_contract(base: dict[str, Any]) -> tuple[dict[int, str], dict[int, float]]:
    rows = _base_rows(base)
    aspects: dict[int, float] = {}
    for index, sample in enumerate(base.get("samples") or []):
        row = _int(sample.get("viewcellRow"), f"base sample {index} viewcellRow")
        aspect = _finite_positive(sample.get("aspect"), f"base sample {index} aspect")
        previous = aspects.get(row)
        if previous is not None and abs(previous - aspect) > FOV_TOLERANCE:
            raise ValueError(
                f"base view-cell row {row} has inconsistent per-pose aspect: "
                f"{previous:g} vs {aspect:g}"
            )
        aspects[row] = aspect
        _require_fov(
            sample.get("renderFovYDeg"), 60.0, f"base sample {index} renderFovYDeg"
        )
        _require_fov(
            sample.get("modelInputFovYDeg"), 66.0, f"base sample {index} modelInputFovYDeg"
        )
    if set(aspects) != set(rows):
        raise ValueError("base formal manifest does not provide an aspect for every view-cell")
    return rows, aspects

def _hzb_rows(result: dict[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    if result.get("schema") != HZB_RESULT_SCHEMA:
        raise ValueError(f"unsupported HZB result schema: {result.get('schema')!r}")
    if result.get("mode") != "Region66":
        raise ValueError("HZB image conversion requires mode=Region66")
    if result.get("error"):
        raise ValueError("cannot convert an HZB result that records an execution error")
    if result.get("formalReady") is not True:
        raise ValueError("HZB result must declare formalReady=true")
    if result.get("executionClass") != "formal-hardware-gpu":
        raise ValueError("HZB result must declare executionClass=formal-hardware-gpu")
    gpu_gate = result.get("gpuGate")
    if not isinstance(gpu_gate, dict) or gpu_gate.get("required") is not True or gpu_gate.get("hardware") is not True:
        raise ValueError("HZB result must pass its formal hardware GPU gate")
    workload = result.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("schema") != HZB_WORKLOAD_SCHEMA
        or workload.get("split") != "test"
    ):
        raise ValueError("HZB Region66 result must contain a test browser workload")
    if not isinstance(workload.get("scene"), str) or not workload["scene"]:
        raise ValueError("HZB workload must declare a scene")
    samples = result.get("samples")
    if not isinstance(samples, list) or not samples or any(not isinstance(row, dict) for row in samples):
        raise ValueError("HZB result must contain object samples")
    rows: dict[int, dict[str, Any]] = {}
    for index, sample in enumerate(samples):
        pose_id = _int(sample.get("poseId"), f"HZB sample {index} poseId")
        if pose_id in rows:
            raise ValueError(f"HZB result has duplicate poseId: {pose_id}")
        rows[pose_id] = sample
    if workload.get("poseCount") != len(rows):
        raise ValueError("HZB workload poseCount does not match result samples")
    return workload, rows


def _workload_pose_contract(workload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    provenance = workload.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("HZB workload is missing provenance")

    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("HZB workload provenance is missing configuration")
    _require_fov(configuration.get("fovYDeg"), 60.0, "HZB provenance configuration.fovYDeg")
    _require_fov(configuration.get("regionFovYDeg"), 66.0, "HZB provenance configuration.regionFovYDeg")
    workload_fov = _require_fov(workload.get("fovYDeg"), 66.0, "HZB workload fovYDeg")

    pose_selection = workload.get("poseSelection")
    if not isinstance(pose_selection, dict):
        pose_selection = provenance.get("poseSelection")
    if not isinstance(pose_selection, dict):
        raise ValueError("HZB workload is missing pose selection provenance")
    if pose_selection.get("split") != "test":
        raise ValueError("HZB pose selection provenance must declare split=test")
    selected_indices = pose_selection.get("selectedPoseIndices")
    if not isinstance(selected_indices, list):
        raise ValueError("HZB pose selection provenance must list selectedPoseIndices")
    selected = [_int(value, f"HZB selectedPoseIndices[{index}]") for index, value in enumerate(selected_indices)]
    if len(selected) != len(set(selected)):
        raise ValueError("HZB pose selection provenance contains duplicate pose IDs")
    if pose_selection.get("selectedPoseCount") != len(selected):
        raise ValueError("HZB pose selection count does not match selectedPoseIndices")
    if pose_selection.get("limit") not in (None, 0):
        raise ValueError("formal HZB result must not truncate the test pose set")
    if pose_selection.get("representative") is not True:
        raise ValueError("formal HZB result must record a complete representative pose selection")

    pose_rows = workload.get("poses")
    if not isinstance(pose_rows, list) or not pose_rows:
        pose_rows = provenance.get("poseAspects")
    if not isinstance(pose_rows, list) or not pose_rows:
        raise ValueError("HZB workload must record per-pose aspect metadata")
    rows: dict[int, dict[str, Any]] = {}
    for index, pose in enumerate(pose_rows):
        if not isinstance(pose, dict):
            raise ValueError(f"HZB workload pose metadata {index} must be an object")
        pose_id = _int(pose.get("poseId"), f"HZB workload pose {index} poseId")
        if pose_id in rows:
            raise ValueError(f"HZB workload has duplicate pose metadata: {pose_id}")
        # The compact result emitted by the HZB runner stores this table as
        # provenance.poseAspects and carries the common Region66 FOV only at
        # workload/configuration level. An expanded workload.poses row may
        # still repeat fovYDeg, which must agree when present.
        _require_fov(
            pose.get("fovYDeg", workload_fov), 66.0,
            f"HZB workload pose {pose_id} fovYDeg",
        )
        aspect = _finite_positive(pose.get("aspect"), f"HZB workload pose {pose_id} aspect")
        camera_view = pose.get("cameraView")
        if camera_view is not None:
            if not isinstance(camera_view, list) or len(camera_view) != 2:
                raise ValueError(f"HZB workload pose {pose_id} cameraView must contain tan_x and tan_y")
            tan_x = _finite_positive(camera_view[0], f"HZB workload pose {pose_id} cameraView.tanX")
            tan_y = _finite_positive(camera_view[1], f"HZB workload pose {pose_id} cameraView.tanY")
            if abs(tan_x / tan_y - aspect) > FOV_TOLERANCE:
                raise ValueError(f"HZB workload pose {pose_id} cameraView aspect disagrees with aspect")
        row_metadata: dict[str, Any] = {"aspect": aspect, "fovYDeg": 66.0}
        if pose.get("candidateCount") is not None:
            row_metadata["candidateCount"] = _int(
                pose["candidateCount"], f"HZB workload pose {pose_id} candidateCount"
            )
        rows[pose_id] = row_metadata
    if set(rows) != set(selected):
        raise ValueError("HZB workload pose metadata does not exactly cover selectedPoseIndices")
    if len(rows) != int(workload.get("poseCount", -1)):
        raise ValueError("HZB workload pose metadata count does not match poseCount")
    warmup_count = workload.get("warmupCount", 0)
    if warmup_count not in (None, 0):
        raise ValueError("formal HZB result must not omit test poses through warmup")
    return rows

def _read_array(path: Path, typecode: str, label: str) -> array.array:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    values = array.array(typecode)
    if len(raw) % values.itemsize:
        raise ValueError(f"{label} has a truncated element: {path}")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    return values

def _candidate_rows(
    dataset_dir: Path, pose_ids: Sequence[int], *, pose_count: int | None = None
) -> dict[int, list[int]]:
    offsets = _read_array(dataset_dir / "candidate_offsets.bin", "Q", "candidate offsets")
    values = _read_array(dataset_dir / "candidate_ids.bin", "I", "candidate IDs")
    if not offsets or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("candidate CSR offsets do not match candidate IDs")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError("candidate CSR offsets must be monotonic")
    if pose_count is not None and len(offsets) != pose_count + 1:
        raise ValueError("candidate CSR row count does not match candidate dataset metadata")
    rows: dict[int, list[int]] = {}
    for pose_id in pose_ids:
        if pose_id + 1 >= len(offsets):
            raise ValueError(f"candidate CSR has no row for HZB poseId {pose_id}")
        row = [int(value) for value in values[offsets[pose_id]:offsets[pose_id + 1]]]
        if len(row) != len(set(row)):
            raise ValueError(f"candidate CSR poseId {pose_id} contains duplicate component IDs")
        rows[pose_id] = row
    return rows


def _candidate_dataset_contract(
    dataset_dir: Path,
    base_manifest: dict[str, Any],
    base_runtime_meta: dict[str, Any],
    expected_scene: str,
    expected_pose_aspects: dict[int, float],
) -> tuple[dict[str, Any], dict[int, list[int]]]:
    meta = read_json(dataset_dir / "dataset_meta.json")
    pose_count = _int(meta.get("poseCount"), "candidate dataset poseCount")
    num_instances = _int(meta.get("numInstances"), "candidate dataset numInstances")
    candidate_count = _int(meta.get("candidateCount"), "candidate dataset candidateCount")
    if num_instances != len((base_manifest.get("instanceBindings") or {}).get("componentToBinding") or {}):
        raise ValueError("candidate dataset numInstances does not match the base manifest instance inventory")
    if meta.get("modelInputFovYDeg") is None or meta.get("frontendRenderFovYDeg") is None:
        raise ValueError("candidate dataset must declare model and frontend FOV")
    _require_fov(meta.get("modelInputFovYDeg"), 66.0, "candidate dataset modelInputFovYDeg")
    _require_fov(meta.get("frontendRenderFovYDeg"), 60.0, "candidate dataset frontendRenderFovYDeg")
    if not isinstance(meta.get("candidateSemantics"), str) or not meta["candidateSemantics"].strip():
        raise ValueError("candidate dataset must declare candidateSemantics")

    candidate_runtime_meta = _absolute_path(meta.get("runtimeMeta"), "candidate dataset runtimeMeta")
    if not candidate_runtime_meta.is_file():
        raise FileNotFoundError(f"candidate dataset runtimeMeta is missing: {candidate_runtime_meta}")
    candidate_runtime_payload = read_json(candidate_runtime_meta)
    if candidate_runtime_payload.get("sceneName") not in (None, expected_scene):
        raise ValueError("candidate runtime metadata scene does not match the base manifest/HZB result")
    _compare_runtime_metadata(base_runtime_meta, candidate_runtime_payload)

    declared_scene = meta.get("scene") or meta.get("sceneName")
    if declared_scene is not None and str(declared_scene) != expected_scene:
        raise ValueError("candidate dataset scene does not match the base manifest/HZB result")

    pose_path = dataset_dir / "poses.bin"
    raw_poses = pose_path.read_bytes()
    if len(raw_poses) != pose_count * POSE_STRIDE_BYTES:
        raise ValueError("candidate dataset poses.bin length does not match poseCount")
    split_ids = meta.get("splitIds")
    if not isinstance(split_ids, dict) or split_ids.get("test") is None:
        raise ValueError("candidate dataset must declare splitIds.test")
    test_split_id = _int(split_ids["test"], "candidate dataset splitIds.test")
    test_rows = {
        pose_id
        for pose_id in range(pose_count)
        if raw_poses[pose_id * POSE_STRIDE_BYTES + POSE_SPLIT_OFFSET_BYTES] == test_split_id
    }
    if not test_rows:
        raise ValueError("candidate dataset has no test pose rows")
    split_counts = meta.get("splitCounts")
    if not isinstance(split_counts, dict) or split_counts.get("test") is None:
        raise ValueError("candidate dataset must declare splitCounts.test")
    if _int(split_counts["test"], "candidate dataset splitCounts.test") != len(test_rows):
        raise ValueError("candidate dataset splitCounts.test does not match poses.bin")
    expected_viewcell_rows = set(expected_pose_aspects)
    if test_rows != expected_viewcell_rows:
        missing = sorted(expected_viewcell_rows - test_rows)
        extra = sorted(test_rows - expected_viewcell_rows)
        raise ValueError(
            "candidate dataset test pose coverage does not match the base manifest: "
            f"missing={missing[:16]}, extra={extra[:16]}"
        )

    for pose_id, expected_aspect in expected_pose_aspects.items():
        tan_x, tan_y = struct.unpack_from(
            "<ff", raw_poses, pose_id * POSE_STRIDE_BYTES + 36
        )
        if not math.isfinite(tan_x) or not math.isfinite(tan_y) or tan_x <= 0 or tan_y <= 0:
            raise ValueError(f"candidate dataset poseId {pose_id} has invalid camera_view")
        if abs(tan_x / tan_y - expected_aspect) > FOV_TOLERANCE:
            raise ValueError(
                f"candidate dataset poseId {pose_id} aspect does not match the base manifest"
            )

    candidate_id_values = _read_array(dataset_dir / "candidate_ids.bin", "I", "candidate IDs")
    if candidate_count != len(candidate_id_values):
        raise ValueError("candidate dataset candidateCount does not match candidate IDs")
    candidates = _candidate_rows(dataset_dir, sorted(expected_viewcell_rows), pose_count=pose_count)
    return meta, candidates


def _compare_runtime_metadata(base: dict[str, Any], candidate: dict[str, Any]) -> None:
    for field in ("sceneName", "componentCount", "instanceCount", "globalGlbCount"):
        base_value = base.get(field)
        candidate_value = candidate.get(field)
        if base_value is not None and candidate_value != base_value:
            raise ValueError(f"candidate runtime metadata {field} does not match the base manifest")

    def component_mapping(payload: dict[str, Any]) -> dict[int, int] | None:
        records = payload.get("componentRecords")
        if not isinstance(records, list):
            return None
        mapping: dict[int, int] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"runtime metadata componentRecords[{index}] must be an object")
            component_id = _int(record.get("componentGlobalId"), "runtime componentGlobalId")
            global_glb_id = _int(record.get("globalGlbId"), "runtime globalGlbId")
            if component_id in mapping and mapping[component_id] != global_glb_id:
                raise ValueError("runtime metadata contains conflicting component mappings")
            mapping[component_id] = global_glb_id
        return mapping

    base_components = component_mapping(base)
    candidate_components = component_mapping(candidate)
    if base_components is not None and candidate_components != base_components:
        raise ValueError("candidate runtime component mapping does not match the base manifest")

    def glb_mapping(payload: dict[str, Any]) -> dict[int, tuple[int, ...]] | None:
        records = payload.get("globalGlbRecords")
        if not isinstance(records, list):
            return None
        mapping: dict[int, tuple[int, ...]] = {}
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"runtime metadata globalGlbRecords[{index}] must be an object")
            global_glb_id = _int(record.get("globalGlbId"), "runtime globalGlbId")
            component_ids = record.get("componentGlobalIds") or []
            if not isinstance(component_ids, list):
                raise ValueError(f"runtime metadata globalGlbRecords[{index}] componentGlobalIds must be a list")
            ids = tuple(_int(value, "runtime componentGlobalId") for value in component_ids)
            if global_glb_id in mapping and mapping[global_glb_id] != ids:
                raise ValueError("runtime metadata contains conflicting GLB mappings")
            mapping[global_glb_id] = ids
        return mapping

    base_glbs = glb_mapping(base)
    candidate_glbs = glb_mapping(candidate)
    if base_glbs is not None and candidate_glbs != base_glbs:
        raise ValueError("candidate runtime GLB mapping does not match the base manifest")


def _base_scene_contract(base_manifest: dict[str, Any]) -> tuple[str, Path, dict[str, Any]]:
    runtime_meta_path = _absolute_path(base_manifest.get("runtimeMeta"), "base manifest runtimeMeta")
    if not runtime_meta_path.is_file():
        raise FileNotFoundError(f"base manifest runtimeMeta is missing: {runtime_meta_path}")
    runtime_meta = read_json(runtime_meta_path)
    component_to_binding = (base_manifest.get("instanceBindings") or {}).get("componentToBinding")
    if not isinstance(component_to_binding, dict) or not component_to_binding:
        raise ValueError("base formal manifest has no complete component binding table")
    runtime_instance_count = runtime_meta.get("instanceCount", runtime_meta.get("componentCount"))
    if runtime_instance_count is not None and _int(runtime_instance_count, "base runtimeMeta instanceCount") != len(component_to_binding):
        raise ValueError("base runtimeMeta instance count does not match the base manifest")
    component_records = runtime_meta.get("componentRecords")
    if isinstance(component_records, list):
        runtime_components: dict[int, int] = {}
        for index, record in enumerate(component_records):
            if not isinstance(record, dict):
                raise ValueError(f"base runtimeMeta componentRecords[{index}] must be an object")
            component_id = _int(record.get("componentGlobalId"), "base runtime componentGlobalId")
            global_glb_id = _int(record.get("globalGlbId"), "base runtime globalGlbId")
            if component_id in runtime_components and runtime_components[component_id] != global_glb_id:
                raise ValueError("base runtimeMeta contains conflicting component mappings")
            runtime_components[component_id] = global_glb_id
        expected_components = {
            _int(component_id, "base componentGlobalId"): _int(
                binding.get("globalGlbId"), "base component globalGlbId"
            )
            for component_id, binding in component_to_binding.items()
        }
        if runtime_components != expected_components:
            raise ValueError("base runtimeMeta component mapping does not match the base manifest")

    scene_values = [base_manifest.get("scene"), base_manifest.get("sceneName")]
    asset_root = base_manifest.get("glbRoot")
    if asset_root is not None:
        scene_values.append(_scene_from_asset_root(asset_root, "base manifest glbRoot"))
    runtime_scene = runtime_meta.get("sceneName")
    if runtime_scene:
        scene_values.append(str(runtime_scene))
    scenes = {str(value) for value in scene_values if isinstance(value, str) and value.strip()}
    if not scenes:
        raise ValueError("base formal manifest does not identify a scene")
    if len(scenes) != 1:
        raise ValueError(f"base manifest has conflicting scene identities: {sorted(scenes)}")
    return next(iter(scenes)), runtime_meta_path, runtime_meta


def _validated_source_result(source_result: str | Path, expected: dict[str, Any]) -> Path:
    path = _absolute_path(source_result, "sourceResult")
    if not path.is_file():
        raise FileNotFoundError(f"sourceResult does not exist: {path}")
    payload = read_json(path)
    if payload != expected:
        raise ValueError("sourceResult does not match the supplied HZB result")
    return path


def _validate_optional_shell_provenance(
    result: dict[str, Any], expected_scene: str, expected_instance_count: int
) -> None:
    shell_dir = result.get("shellDir")
    if not isinstance(shell_dir, (str, Path)) or not str(shell_dir).strip():
        return
    shell_meta_path = _absolute_path(shell_dir, "HZB result shellDir") / "shell_meta.json"
    if not shell_meta_path.is_file():
        return
    shell_meta = read_json(shell_meta_path)
    if shell_meta.get("schema") != "geometry-shell-hzb-v2":
        raise ValueError("HZB result shell metadata has an unsupported schema")
    if shell_meta.get("sceneName") != expected_scene:
        raise ValueError("HZB shell scene does not match the base manifest/HZB result")
    if shell_meta.get("instanceCount") is not None and _int(
        shell_meta["instanceCount"], "HZB shell instanceCount"
    ) != expected_instance_count:
        raise ValueError("HZB shell instance count does not match the base manifest")

def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and non-negative")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result

def _baseline(
    result: dict[str, Any], workload: dict[str, Any], rows: dict[int, dict[str, Any]],
    asset_variant: str, source_result: str | Path,
) -> dict[str, Any]:
    if not isinstance(asset_variant, str) or not asset_variant.strip():
        raise ValueError("assetVariant must be supplied as a non-empty string")
    width = _int(workload.get("width"), "workload.width")
    height = _int(workload.get("height"), "workload.height")
    if not width or not height:
        raise ValueError("workload.width and workload.height must be positive")
    depth_biases = []
    for pose_id, sample in rows.items():
        timings = sample.get("timings")
        if not isinstance(timings, dict):
            raise ValueError(f"HZB poseId {pose_id} is missing timings")
        depth_biases.append(_number(timings.get("depthBiasM"), f"poseId {pose_id} timings.depthBiasM"))
    if any(value != depth_biases[0] for value in depth_biases[1:]):
        raise ValueError("HZB samples record different depthBiasM values")
    region_sampling = result.get("regionSampling")
    if not isinstance(region_sampling, dict) or region_sampling.get("schema") != HZB_REGION_SCHEMA:
        raise ValueError("HZB result is missing regionSampling metadata")
    region_count = _int(region_sampling.get("requestedCount"), "regionSampling.requestedCount")
    return {
        "method": "geometry-shell-hzb",
        "selectionSplit": "calibration",
        "testRead": False,
        "assetVariant": asset_variant.strip(),
        "resolution": [width, height],
        "depthBiasM": depth_biases[0],
        "regionSampleCount": region_count,
        "sourceResult": str(_absolute_path(source_result, "sourceResult")),
    }

def build_hzb_image_manifest(
    base_manifest: dict[str, Any], hzb_result: dict[str, Any], *,
    candidate_dataset_dir: Path, asset_variant: str, source_result: str | Path,
) -> dict[str, Any]:
    """Replace keyed predictions while retaining the base camera/reference contract."""
    validate_test_manifest(base_manifest)
    if base_manifest.get("schema") != FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise ValueError("base image manifest must use formal-render-manifest-v2")
    validate_formal_instance_render_manifest(base_manifest)
    _require_fov(base_manifest.get("renderFovYDeg"), 60.0, "base manifest renderFovYDeg")
    _require_fov(base_manifest.get("modelInputFovYDeg"), 66.0, "base manifest modelInputFovYDeg")
    base_rows, base_aspects = _base_pose_contract(base_manifest)
    coverage = base_manifest.get("testCoverage")
    if not isinstance(coverage, dict):
        raise ValueError("base formal manifest is missing testCoverage")
    if coverage.get("viewcellCount") != len(base_rows):
        raise ValueError("base testCoverage.viewcellCount does not match base pose coverage")
    if coverage.get("sampleCount") != len(base_manifest.get("samples") or []):
        raise ValueError("base testCoverage.sampleCount does not match base samples")

    base_scene, _base_runtime_meta_path, base_runtime_meta = _base_scene_contract(base_manifest)
    workload, hzb_rows = _hzb_rows(hzb_result)
    base_component_count = len(
        (base_manifest.get("instanceBindings") or {}).get("componentToBinding") or {}
    )
    _validate_optional_shell_provenance(hzb_result, base_scene, base_component_count)
    if hzb_result.get("scene") not in (None, base_scene):
        raise ValueError("HZB result scene does not match the base manifest")
    if workload.get("scene") != base_scene:
        raise ValueError(
            f"HZB scene {workload.get('scene')!r} does not match base scene {base_scene!r}"
        )
    workload_pose_metadata = _workload_pose_contract(workload)
    missing = sorted(set(base_rows) - set(hzb_rows))
    extra = sorted(set(hzb_rows) - set(base_rows))
    if missing or extra:
        raise ValueError(
            "HZB poseIds do not exactly cover base viewcellRow: "
            f"missing={missing}, extra={extra}"
        )
    source_dataset = workload.get("source")
    if not isinstance(source_dataset, dict):
        result_dataset = hzb_result.get("datasetDir")
        if not isinstance(result_dataset, (str, Path)) or not str(result_dataset).strip():
            raise ValueError("HZB result is missing candidate dataset provenance")
        source_dataset = {"datasetDirName": _absolute_path(result_dataset, "HZB result datasetDir").name}
    dataset_dir = _absolute_path(candidate_dataset_dir, "candidate dataset directory")
    dataset_name = dataset_dir.name
    declared_dataset_name = source_dataset.get("datasetDirName")
    if declared_dataset_name != dataset_name:
        raise ValueError(
            "candidate dataset does not match HZB workload provenance: "
            f"expected={declared_dataset_name!r}, actual={dataset_name!r}"
        )
    result_dataset = hzb_result.get("datasetDir")
    if result_dataset is not None and _absolute_path(result_dataset, "HZB result datasetDir").name != dataset_name:
        raise ValueError("candidate dataset does not match HZB result datasetDir")
    candidate_meta, candidates = _candidate_dataset_contract(
        dataset_dir,
        base_manifest,
        base_runtime_meta,
        base_scene,
        base_aspects,
    )
    candidate_semantics = source_dataset.get("candidateSemantics")
    if candidate_semantics is not None and candidate_semantics != candidate_meta.get("candidateSemantics"):
        raise ValueError("candidate dataset semantics do not match HZB workload provenance")
    source_path = _validated_source_result(source_result, hzb_result)

    bindings = base_manifest.get("instanceBindings") or {}
    component_to_binding = bindings.get("componentToBinding")
    if not isinstance(component_to_binding, dict):
        raise ValueError("base formal manifest has no complete component binding table")
    try:
        component_ids = {int(key) for key in component_to_binding}
    except (TypeError, ValueError) as error:
        raise ValueError("base component binding keys must be component IDs") from error
    predictions: dict[str, list[int]] = {}
    selected_candidate_count = 0
    for pose_id in sorted(hzb_rows):
        sample = hzb_rows[pose_id]
        candidate = candidates[pose_id]
        selected_candidate_count += len(candidate)
        unknown_candidates = sorted(set(candidate) - component_ids)
        if unknown_candidates:
            raise ValueError(f"HZB poseId {pose_id} candidate IDs are outside the instance range: {unknown_candidates}")
        if _int(sample.get("candidateCount"), f"HZB poseId {pose_id} candidateCount") != len(candidate):
            raise ValueError(f"HZB poseId {pose_id} candidateCount does not match candidate IDs")
        metadata_candidate_count = workload_pose_metadata[pose_id].get("candidateCount")
        if metadata_candidate_count is not None and metadata_candidate_count != len(candidate):
            raise ValueError(f"HZB poseId {pose_id} workload candidateCount does not match candidate IDs")
        if abs(workload_pose_metadata[pose_id]["aspect"] - base_aspects[pose_id]) > FOV_TOLERANCE:
            raise ValueError(
                f"HZB poseId {pose_id} aspect does not match the base manifest: "
                f"{workload_pose_metadata[pose_id]['aspect']:g} vs {base_aspects[pose_id]:g}"
            )
        predicted = _ids(sample.get("visibleInstanceIds"), f"HZB poseId {pose_id} visibleInstanceIds")
        unknown = sorted(set(predicted) - component_ids)
        if unknown:
            raise ValueError(f"HZB poseId {pose_id} predicts component IDs outside the instance range: {unknown}")
        outside = sorted(set(predicted) - set(candidate))
        if outside:
            raise ValueError(f"HZB poseId {pose_id} predicts component IDs outside its candidate set: {outside}")
        key = base_rows[pose_id]
        if key in predictions and predictions[key] != predicted:
            raise ValueError(f"predictionKey {key!r} maps to conflicting HZB predictions")
        predictions[key] = predicted
    if workload.get("candidateCount") != selected_candidate_count:
        raise ValueError("HZB workload candidateCount does not match the candidate dataset rows")

    output = copy.deepcopy(base_manifest)
    for field in ("threshold", "thresholdSelection", "thresholdProvenance"):
        output.pop(field, None)
    output["baselineSelection"] = _baseline(hzb_result, workload, hzb_rows, asset_variant, source_path)
    output[PREDICTION_COMPONENT_IDS_BY_KEY_FIELD] = predictions
    validate_test_manifest(output)
    validate_formal_instance_render_manifest(output)
    return output

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--hzb-result", type=Path, required=True)
    parser.add_argument("--candidate-dataset-dir", type=Path, required=True)
    parser.add_argument("--asset-variant", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result_path = args.hzb_result.resolve()
    output = build_hzb_image_manifest(
        read_json(args.base_manifest.resolve()), read_json(result_path),
        candidate_dataset_dir=args.candidate_dataset_dir.resolve(),
        asset_variant=args.asset_variant, source_result=result_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(output['samples'])} samples)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"build_hzb_image_manifest: {error}", file=sys.stderr)
        raise
