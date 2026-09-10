#!/usr/bin/env python3
"""Orchestrate the registered Geometry-shell HZB paper baseline.

The orchestrator owns only the narrow HZB paper matrix.  It validates the
existing shell, Pose CSR and Region66 assets, keeps calibration independent
from frozen test, and records every browser/evaluator subprocess in its own
directory.  It never enables the software GPU path.

Calibration and frozen test need one visibility pass per pose.  The separate
120-pose timing phase uses five rounds to measure runtime distributions without
multiplying the full accuracy matrix.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "neural_instance_culling" / "benchmark"
RUNNER = ROOT / "slm2viewer" / "scripts" / "run_geometry_shell_hzb_benchmark.mjs"
EVALUATOR = BENCHMARK_DIR / "evaluate_geometry_shell_hzb.py"
SELECTOR = BENCHMARK_DIR / "select_geometry_shell_hzb.py"
POINT60_PLAN_BUILDER = BENCHMARK_DIR / "build_hzb_point60_gt_plan.py"
SAMPLER = ROOT / "neural_instance_culling" / "sampler" / "run_sampler.mjs"

EXPERIMENT = "baseline_geometry_shell_hzb_paper_2026-09-09"
RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v2"
METRICS_SCHEMA = "geometry-shell-hzb-metrics-v2"
SELECTION_SCHEMA = "geometry-shell-hzb-calibration-selection-v1"
PREFLIGHT_SCHEMA = "geometry-shell-hzb-paper-preflight-v1"
TASK_SCHEMA = "geometry-shell-hzb-paper-task-v1"

CALIBRATION_RESOLUTIONS = ((512, 288), (1024, 576))
CALIBRATION_DEPTH_BIASES_M = (0.0001, 0.001, 0.01)
# One calibration round means one browser task per (height, bias) member.
CALIBRATION_ROUNDS = 1
CALIBRATION_INVOCATIONS_PER_CONFIGURATION = 1
CALIBRATION_RUNNER_TIMING_ROUNDS = 1
TEST_RUNNER_TIMING_ROUNDS = 1
TIMING_POSE_COUNT = 120
TIMING_ROUNDS = 5
REGION_COUNTS = {
    "hkust": (1, 5, 9, 0),
    "ifcbench": (1, 0),
}
VARIANTS = ("lossless", "equal_asset")
EXPECTED_SPLITS = {
    "hkust": {"train": 5926, "validation": 730, "calibration": 659, "test": 684, "guard": 0},
    "ifcbench": {"train": 19647, "validation": 2712, "calibration": 2183, "test": 2710, "guard": 0},
}

RUNNER_FILES = (
    "result.json",
    "geometry_shell_hzb_gpu_evidence.json",
    "geometry_shell_hzb_workload.json",
    "geometry_shell_hzb_candidates_uint32.bin",
    "stdout.log",
    "stderr.log",
)
VISIBILITY_FILES = RUNNER_FILES + ("metrics.json",)
TIMING_FILES = RUNNER_FILES


class IncompleteArtifactError(RuntimeError):
    """Raised when a pre-existing task directory cannot be resumed safely."""


class FormalArtifactError(RuntimeError):
    """Raised when a browser artifact cannot be used as a formal result."""


@dataclass(frozen=True)
class SceneSpec:
    key: str
    scene_name: str
    dataset_dir: Path
    region_dataset_dir: Path
    runtime_meta: Path
    glb_index: Path
    glb_root: Path
    shell_dirs: Mapping[str, Path]
    timing_plan: Path
    point60_plan_dir: Path
    representative_plan: Path
    expected_splits: Mapping[str, int]


@dataclass(frozen=True)
class TaskSpec:
    phase: str
    scene: str
    name: str
    directory: Path
    commands: tuple[tuple[str, tuple[str, ...]], ...]
    expected_artifacts: tuple[str, ...]
    config: Mapping[str, Any]

    @property
    def result_path(self) -> Path:
        return self.directory / "result.json"

    @property
    def metrics_path(self) -> Path:
        return self.directory / "metrics.json"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer") from error
    if result != value and not isinstance(value, str):
        raise ValueError(f"{label} must be an integer")
    return result


def _read_u64(path: Path, count: int) -> tuple[int, ...]:
    data = _require_file(path, "uint64 array").read_bytes()
    expected = count * 8
    if len(data) != expected:
        raise ValueError(f"{path} has {len(data)} bytes; expected {expected}")
    return struct.unpack(f"<{count}Q", data)


def _check_monotonic_offsets(offsets: Sequence[int], label: str) -> None:
    if not offsets or offsets[0] != 0:
        raise ValueError(f"{label} must start at zero")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ValueError(f"{label} must be monotonic")


def _check_csr(directory: Path, offsets_name: str, ids_name: str, row_count: int, label: str) -> int:
    offsets = _read_u64(directory / offsets_name, row_count + 1)
    _check_monotonic_offsets(offsets, f"{label} offsets")
    ids = _require_file(directory / ids_name, f"{label} IDs")
    if ids.stat().st_size % 4:
        raise ValueError(f"{label} IDs are not uint32 aligned: {ids}")
    item_count = ids.stat().st_size // 4
    if offsets[-1] != item_count:
        raise ValueError(f"{label} final offset {offsets[-1]} != ID count {item_count}")
    return item_count


def _check_region_assets(directory: Path, row_count: int, num_instances: int) -> dict[str, Any]:
    meta = _read_json(_require_file(directory / "dataset_meta.json", "Region66 dataset metadata"))
    if meta.get("schema") != "proxy-viewcell-pvs-dataset-v1":
        raise ValueError(f"unsupported Region66 dataset schema: {directory}")
    if _int(meta.get("viewcellCount", -1), "Region66 viewcellCount") != row_count:
        raise ValueError("Region66 viewcellCount does not match Pose CSR poseCount")
    if _int(meta.get("numInstances", -1), "Region66 numInstances") != num_instances:
        raise ValueError("Region66 numInstances does not match Pose CSR dataset")
    for name in (
        "subpose_camera_forward.bin",
        "subpose_camera_pos.bin",
        "subpose_offsets.bin",
        "subpose_pose_indices.bin",
        "viewcell_centers.bin",
    ):
        _require_file(directory / name, f"Region66 {name}")
    offsets = _read_u64(directory / "subpose_offsets.bin", row_count + 1)
    _check_monotonic_offsets(offsets, "Region66 subpose offsets")
    subpose_count = offsets[-1]
    expected_vec3_bytes = subpose_count * 3 * 4
    for name in ("subpose_camera_forward.bin", "subpose_camera_pos.bin"):
        if (directory / name).stat().st_size != expected_vec3_bytes:
            raise ValueError(f"Region66 {name} length does not match subpose offsets")
    if (directory / "subpose_pose_indices.bin").stat().st_size != subpose_count * 4:
        raise ValueError("Region66 subpose pose-index length does not match offsets")
    if (directory / "viewcell_centers.bin").stat().st_size != row_count * 3 * 4:
        raise ValueError("Region66 viewcell center length does not match viewcellCount")
    return {
        "schema": meta["schema"],
        "viewcellCount": row_count,
        "subposeCount": subpose_count,
        "numInstances": num_instances,
    }


def _check_runtime_meta(path: Path, num_instances: int) -> dict[str, Any]:
    meta = _read_json(_require_file(path, "runtime visibility metadata"))
    if _int(meta.get("instanceCount", -1), "runtime instanceCount") != num_instances:
        raise ValueError("runtime metadata instanceCount does not match dataset")
    records = meta.get("componentRecords")
    if not isinstance(records, list) or len(records) != num_instances:
        raise ValueError("runtime metadata componentRecords are not dense")
    ids = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("runtime component record must be an object")
        ids.append(_int(record.get("componentGlobalId", -1), "componentGlobalId"))
    if sorted(ids) != list(range(num_instances)):
        raise ValueError("runtime component IDs must be unique and dense")
    return {
        "instanceCount": num_instances,
        "globalGlbCount": _int(meta.get("globalGlbCount", -1), "runtime globalGlbCount"),
    }


def _check_glb_index(path: Path, root: Path, expected_count: int) -> dict[str, Any]:
    index = _read_json(_require_file(path, "GLB index"))
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(f"GLB index entry count does not match runtime metadata: {path}")
    resolved_root = _require_dir(root, "GLB root").resolve()
    global_ids: list[int] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("GLB index entries must be objects")
        global_id = _int(entry.get("globalId", -1), "GLB globalId")
        relative = entry.get("path")
        if global_id < 0 or not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("GLB index contains an invalid entry")
        candidate = (resolved_root / relative).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError("GLB index path escapes GLB root") from error
        _require_file(candidate, f"GLB {global_id}")
        global_ids.append(global_id)
    if sorted(global_ids) != list(range(expected_count)):
        raise ValueError("GLB index global IDs must be dense")
    return {"entryCount": expected_count, "root": str(resolved_root)}


def _check_shell(path: Path, scene: SceneSpec, variant: str, num_instances: int) -> dict[str, Any]:
    _require_dir(path, f"{scene.key} {variant} geometry shell")
    meta = _read_json(_require_file(path / "shell_meta.json", "geometry shell metadata"))
    if meta.get("schema") != "geometry-shell-hzb-v2":
        raise ValueError(f"unsupported geometry shell schema: {path}")
    if meta.get("variant") != ("equal-asset" if variant == "equal_asset" else "lossless"):
        raise ValueError(f"geometry shell variant mismatch: {path}")
    if meta.get("sceneName") != scene.scene_name:
        raise ValueError(f"geometry shell scene mismatch: {path}")
    if _int(meta.get("instanceCount", -1), "shell instanceCount") != num_instances:
        raise ValueError(f"geometry shell instanceCount mismatch: {path}")
    for name in (
        "indices.meshopt.bin",
        "instance_aabb_fp32.bin",
        "instance_occluder_uint32.bin",
        "instance_to_glb_uint32.bin",
        "positions.meshopt.bin",
        "shell_instance_component_ids_uint32.bin",
        "transforms.meshopt.bin",
    ):
        file = _require_file(path / name, f"geometry shell {name}")
        if file.stat().st_size == 0:
            raise ValueError(f"geometry shell asset is empty: {file}")
    if (path / "instance_aabb_fp32.bin").stat().st_size != num_instances * 6 * 4:
        raise ValueError("geometry shell AABB table length does not match instanceCount")
    for name in ("instance_occluder_uint32.bin", "instance_to_glb_uint32.bin"):
        if (path / name).stat().st_size != num_instances * 4:
            raise ValueError(f"geometry shell {name} length does not match instanceCount")
    shell_instance_count = sum(
        _int(prototype.get("instanceCount", -1), "shell prototype instanceCount")
        for prototype in meta.get("prototypes", [])
    )
    if (path / "shell_instance_component_ids_uint32.bin").stat().st_size != shell_instance_count * 4:
        raise ValueError("geometry shell component ID table length does not match packed shell instances")
    return {
        "variant": meta["variant"],
        "instanceCount": num_instances,
        "occluderInstanceCount": _int(meta.get("occluderInstanceCount", -1), "occluderInstanceCount"),
        "path": str(path.resolve()),
    }


def _check_timing_plan(path: Path, dataset_dir: Path, expected_test_count: int) -> dict[str, Any]:
    payload = _read_json(_require_file(path, "120-pose timing plan"))
    if payload.get("schema") != "geometry-shell-hzb-timing-pose-index-plan-v1":
        raise ValueError(f"unsupported timing plan schema: {path}")
    if payload.get("split") != "test":
        raise ValueError(f"timing plan must use test split: {path}")
    if _int(payload.get("targetPoseCount", -1), "timing targetPoseCount") != TIMING_POSE_COUNT:
        raise ValueError(f"timing plan must contain {TIMING_POSE_COUNT} poses: {path}")
    indices = payload.get("poseIndices")
    if not isinstance(indices, list) or len(indices) != TIMING_POSE_COUNT:
        raise ValueError(f"timing plan poseIndices must contain {TIMING_POSE_COUNT} entries: {path}")
    normalized = [_int(value, "timing pose index") for value in indices]
    if len(set(normalized)) != TIMING_POSE_COUNT or min(normalized) < 0:
        raise ValueError(f"timing plan poseIndices must be unique and non-negative: {path}")
    source = payload.get("source")
    if isinstance(source, Mapping):
        declared_dataset = source.get("datasetDir")
        if declared_dataset is not None and Path(str(declared_dataset)).resolve() != dataset_dir.resolve():
            raise ValueError(f"timing plan dataset does not match Pose CSR dataset: {path}")
        if source.get("splitPoseCount") is not None and _int(source["splitPoseCount"], "timing splitPoseCount") != expected_test_count:
            raise ValueError(f"timing plan test count does not match dataset: {path}")
    return {
        "schema": payload["schema"],
        "split": payload["split"],
        "targetPoseCount": TIMING_POSE_COUNT,
        "strataCount": _int(payload.get("strataCount", -1), "timing strataCount"),
        "poseIndices": normalized,
    }


def _point60_handoff(
    scene: SceneSpec,
    output_root: Path,
    *,
    point60_plan_builder: Path = POINT60_PLAN_BUILDER,
    sampler: Path = SAMPLER,
    node: str = "node",
) -> dict[str, Any]:
    plan_dir = scene.point60_plan_dir.resolve()
    raw_dir = (output_root / "point60_raw_gt" / scene.key).resolve()
    plan_files = sorted(plan_dir.glob("*.jsonl")) if plan_dir.is_dir() else []
    commands: list[dict[str, Any]] = []
    if plan_files:
        for plan in plan_files:
            output = raw_dir / plan.name
            commands.append({
                "plan": str(plan.resolve()),
                "output": str(output),
                "stdout": str((raw_dir / f"{plan.stem}_stdout.log").resolve()),
                "stderr": str((raw_dir / f"{plan.stem}_stderr.log").resolve()),
                "command": [
                    node,
                    str(sampler.resolve()),
                    "--assets-dir", str(scene.glb_root.resolve()),
                    "--pose-plan", str(plan.resolve()),
                    "--output", str(output),
                    "--point60-gt",
                    "--require-hardware-gpu",
                ],
            })
        status = "pending-raw-gt"
    else:
        planned_dir = output_root / "point60_gt_plans" / scene.key
        commands.append({
            "command": [
                sys.executable,
                str(point60_plan_builder.resolve()),
                "--dataset-dir", str(scene.dataset_dir.resolve()),
                "--representative-plan", str(scene.representative_plan.resolve()),
                "--output-dir", str(planned_dir.resolve()),
                "--scene", scene.key,
            ],
            "reason": "existing Point60 plan directory is missing",
        })
        status = "pending-point60-plan"
    return {
        "blocking": False,
        "status": status,
        "planDir": str(plan_dir),
        "rawOutputDir": str(raw_dir),
        "commands": commands,
        "evaluator": [
            sys.executable,
            str(EVALUATOR.resolve()),
            "--point-gt-raw-dir", str(raw_dir),
        ],
        "note": "Point60 raw Color-ID GT is a separate hardware sampler handoff; it never blocks Region66 calibration, selection, test or timing.",
    }


def preflight_scene(
    scene: SceneSpec,
    output_root: Path,
    *,
    point60_plan_builder: Path = POINT60_PLAN_BUILDER,
    sampler: Path = SAMPLER,
    node: str = "node",
) -> dict[str, Any]:
    dataset = _read_json(_require_file(scene.dataset_dir / "dataset_meta.json", "Pose CSR dataset metadata"))
    if dataset.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError(f"unsupported Pose CSR schema: {scene.dataset_dir}")
    num_instances = _int(dataset.get("numInstances", -1), "Pose CSR numInstances")
    pose_count = _int(dataset.get("poseCount", -1), "Pose CSR poseCount")
    if num_instances <= 0 or pose_count <= 0:
        raise ValueError("Pose CSR dataset has invalid instance or pose count")
    split_counts = {
        str(key): _int(value, f"splitCounts.{key}")
        for key, value in (dataset.get("splitCounts") or {}).items()
    }
    if split_counts != dict(scene.expected_splits):
        raise ValueError(
            f"{scene.key} split counts changed: expected={dict(scene.expected_splits)}, actual={split_counts}"
        )
    if _int(dataset.get("poseStrideBytes", -1), "poseStrideBytes") != 64:
        raise ValueError("formal HZB workloads require 64-byte Pose CSR poses")
    split_ids = dataset.get("splitIds") or {}
    for split in ("calibration", "test"):
        if split not in split_ids:
            raise ValueError(f"Pose CSR metadata has no {split} split")
    _require_file(scene.dataset_dir / "poses.bin", "Pose CSR poses")
    if (scene.dataset_dir / "poses.bin").stat().st_size != pose_count * 64:
        raise ValueError("Pose CSR poses.bin length does not match poseCount")
    _check_csr(scene.dataset_dir, "candidate_offsets.bin", "candidate_ids.bin", pose_count, "candidate CSR")
    _check_csr(scene.dataset_dir, "visible_offsets.bin", "visible_ids.bin", pose_count, "visible CSR")
    visible_count = (scene.dataset_dir / "visible_ids.bin").stat().st_size // 4
    if (scene.dataset_dir / "visible_weights.bin").stat().st_size != visible_count * 4:
        raise ValueError("visible_weights.bin length does not match visible_ids.bin")
    candidate_semantics = str(dataset.get("candidateSemantics", ""))
    if "no GT positive union" not in candidate_semantics:
        raise ValueError("Pose CSR candidate semantics do not declare a non-GT candidate set")

    region_report = _check_region_assets(scene.region_dataset_dir, pose_count, num_instances)
    runtime_report = _check_runtime_meta(scene.runtime_meta, num_instances)
    glb_report = _check_glb_index(scene.glb_index, scene.glb_root, runtime_report["globalGlbCount"])
    shell_report = {
        variant: _check_shell(path, scene, variant, num_instances)
        for variant, path in scene.shell_dirs.items()
    }
    timing_report = _check_timing_plan(scene.timing_plan, scene.dataset_dir, scene.expected_splits["test"])
    point60 = _point60_handoff(
        scene,
        output_root,
        point60_plan_builder=point60_plan_builder,
        sampler=sampler,
        node=node,
    )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "experiment": EXPERIMENT,
        "scene": scene.key,
        "sceneName": scene.scene_name,
        "testRead": False,
        "splitCounts": split_counts,
        "numInstances": num_instances,
        "poseCount": pose_count,
        "candidateSemantics": candidate_semantics,
        "groundTruthSemantics": dataset.get("gtSemantics"),
        "shells": shell_report,
        "regionDataset": region_report,
        "runtime": runtime_report,
        "glb": glb_report,
        "timingPlan": timing_report,
        "point60": point60,
        "paths": {
            "datasetDir": str(scene.dataset_dir.resolve()),
            "regionDatasetDir": str(scene.region_dataset_dir.resolve()),
            "runtimeMeta": str(scene.runtime_meta.resolve()),
            "glbIndex": str(scene.glb_index.resolve()),
            "glbRoot": str(scene.glb_root.resolve()),
            "timingPlan": str(scene.timing_plan.resolve()),
        },
    }


def _format_bias(value: float) -> str:
    return format(float(value), ".10g").replace("-", "m").replace(".", "p")


def _region_name(count: int) -> str:
    if count == 0:
        return "all"
    return f"region{count}"


def _scene_specs(data_root: Path, hzb_root: Path) -> dict[str, SceneSpec]:
    data_root = data_root.resolve()
    hzb_root = hzb_root.resolve()
    dataset_root = data_root / "neural_instance_culling" / "dataset" / "out"
    return {
        "hkust": SceneSpec(
            key="hkust",
            scene_name="hkust-v3",
            dataset_dir=dataset_root / "pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
            region_dataset_dir=dataset_root / "hkust_v3_viewcell_colorid_fov66_source",
            runtime_meta=data_root / "hkust-v3" / "assets" / "runtimeVisibilityMeta.json",
            glb_index=data_root / "hkust-v3" / "assets" / "glbIndex.json",
            glb_root=data_root / "hkust-v3" / "assets",
            shell_dirs={
                "lossless": hzb_root / "geometry_shell_hzb_lossless_hkust",
                "equal_asset": hzb_root / "geometry_shell_hzb_equal_asset_hkust",
            },
            timing_plan=hzb_root / "hkust_test_timing_120_pose_plan.json",
            point60_plan_dir=hzb_root / "point60_gt_plans" / "hkust_v3",
            representative_plan=data_root / "neural_instance_culling" / "sampler" / "out" / "hkust_v3_viewcell_fov66" / "representatives.jsonl",
            expected_splits=EXPECTED_SPLITS["hkust"],
        ),
        "ifcbench": SceneSpec(
            key="ifcbench",
            scene_name="ifcbench_fantasy_metropolis_instanced_v2",
            dataset_dir=dataset_root / "pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
            region_dataset_dir=dataset_root / "ifcbench_fantasy_metropolis_viewcell_colorid_fov66_source_v1",
            runtime_meta=data_root / "ifcbench_fantasy_metropolis_instanced_v2" / "assets" / "runtimeVisibilityMeta.json",
            glb_index=data_root / "ifcbench_fantasy_metropolis_instanced_v2" / "assets" / "glbIndex.json",
            glb_root=data_root / "ifcbench_fantasy_metropolis_instanced_v2" / "assets",
            shell_dirs={
                "lossless": hzb_root / "geometry_shell_hzb_lossless_ifcbench",
                "equal_asset": hzb_root / "geometry_shell_hzb_equal_asset_ifcbench",
            },
            timing_plan=hzb_root / "ifcbench_test_timing_120_pose_plan.json",
            point60_plan_dir=hzb_root / "point60_gt_plans" / "ifcbench_fantasy_metropolis",
            representative_plan=data_root / "neural_instance_culling" / "sampler" / "out" / "ifcbench_fantasy_metropolis_instanced_v2" / "pose_plan_fov66.jsonl",
            expected_splits=EXPECTED_SPLITS["ifcbench"],
        ),
    }


def _runner_command(
    scene: SceneSpec,
    shell_variant: str,
    result: Path,
    *,
    split: str,
    width: int,
    height: int,
    depth_bias_m: float,
    region_sample_count: int,
    timing_rounds: int,
    node: str,
    runner: Path,
    pose_index_plan: Path | None = None,
) -> tuple[str, ...]:
    command = [
        node,
        str(runner.resolve()),
        "--shell-dir", str(scene.shell_dirs[shell_variant].resolve()),
        "--dataset-dir", str(scene.dataset_dir.resolve()),
        "--region-dataset-dir", str(scene.region_dataset_dir.resolve()),
        "--output", str(result.resolve()),
        "--mode", "Region66",
        "--split", split,
        "--limit", "0",
        "--width", str(int(width)),
        "--height", str(int(height)),
        "--fov-y-deg", "60",
        "--region-fov-y-deg", "66",
        "--near", "0.05",
        "--far", "20000",
        "--depth-bias-m", format(float(depth_bias_m), ".10g"),
        "--warmup-count", "0",
        "--timing-rounds", str(int(timing_rounds)),
        "--region-sample-count", str(int(region_sample_count)),
        "--require-hardware-gpu",
    ]
    if pose_index_plan is not None:
        command.extend(["--pose-index-plan", str(pose_index_plan.resolve())])
    return tuple(command)


def _evaluator_command(
    scene: SceneSpec,
    result: Path,
    metrics: Path,
    split: str,
    *,
    evaluator: Path = EVALUATOR,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(evaluator.resolve()),
        "--result", str(result.resolve()),
        "--dataset-dir", str(scene.dataset_dir.resolve()),
        "--runtime-meta", str(scene.runtime_meta.resolve()),
        "--output", str(metrics.resolve()),
        "--split", split,
        "--glb-index", str(scene.glb_index.resolve()),
        "--glb-root", str(scene.glb_root.resolve()),
    )


def build_calibration_tasks(
    scene: SceneSpec,
    output_root: Path,
    *,
    node: str = "node",
    runner: Path = RUNNER,
    evaluator: Path = EVALUATOR,
) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for width, height in CALIBRATION_RESOLUTIONS:
        for bias in CALIBRATION_DEPTH_BIASES_M:
            name = f"h{height}_b{_format_bias(bias)}"
            directory = output_root / "calibration" / scene.key / "lossless" / name
            result = directory / "result.json"
            metrics = directory / "metrics.json"
            tasks.append(TaskSpec(
                phase="calibrate",
                scene=scene.key,
                name=name,
                directory=directory,
                commands=(
                    (
                        "browser-calibration",
                        _runner_command(
                            scene,
                            "lossless",
                            result,
                            split="calibration",
                            width=width,
                            height=height,
                            depth_bias_m=bias,
                            region_sample_count=0,
                            timing_rounds=CALIBRATION_RUNNER_TIMING_ROUNDS,
                            node=node,
                            runner=runner,
                        ),
                    ),
                    (
                        "evaluate-calibration",
                        _evaluator_command(scene, result, metrics, "calibration", evaluator=evaluator),
                    ),
                ),
                expected_artifacts=VISIBILITY_FILES,
                config={
                    "shellVariant": "lossless",
                    "width": width,
                    "height": height,
                    "depthBiasM": bias,
                    "regionSampleCount": 0,
                    "calibrationRounds": CALIBRATION_ROUNDS,
                    "calibrationInvocations": CALIBRATION_INVOCATIONS_PER_CONFIGURATION,
                    "runnerTimingRounds": CALIBRATION_RUNNER_TIMING_ROUNDS,
                },
            ))
    return tasks


def _selection_path(output_root: Path, scene: SceneSpec) -> Path:
    return output_root / "select" / scene.key / "selection.json"


def build_selection_task(
    scene: SceneSpec,
    output_root: Path,
    calibration_tasks: Sequence[TaskSpec],
    *,
    selector: Path = SELECTOR,
) -> TaskSpec:
    directory = output_root / "select" / scene.key
    output = directory / "selection.json"
    candidates = tuple(task.metrics_path for task in calibration_tasks)
    command = [sys.executable, str(selector.resolve())]
    for candidate in candidates:
        command.extend(["--candidate", str(candidate.resolve())])
    command.extend(["--output", str(output.resolve())])
    return TaskSpec(
        phase="select",
        scene=scene.key,
        name="selection",
        directory=directory,
        commands=(("freeze-calibration-selection", tuple(command)),),
        expected_artifacts=("selection.json", "stdout.log", "stderr.log"),
        config={"candidateCount": len(candidates), "selectionSplit": "calibration"},
    )


def _selected_config(selection: Mapping[str, Any]) -> dict[str, Any]:
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selected HZB configuration does not use the registered selection schema")
    if selection.get("selectionSplit") != "calibration" or selection.get("testRead") is not False:
        raise ValueError("selected HZB configuration is not calibration-frozen")
    selected = selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("frozen HZB selection has no selected row")
    width = _int(selected.get("width", -1), "selected width")
    height = _int(selected.get("height", -1), "selected height")
    bias = _number(selected.get("depthBiasM"), "selected depthBiasM")
    if width < 8 or height < 8 or bias < 0:
        raise ValueError("frozen HZB selection has invalid runtime configuration")
    if selected.get("assetVariant") != "lossless":
        raise ValueError("frozen HZB selection must originate from the lossless calibration scan")
    return {"width": width, "height": height, "depthBiasM": bias}


def build_test_tasks(
    scene: SceneSpec,
    output_root: Path,
    selection: Mapping[str, Any],
    *,
    node: str = "node",
    runner: Path = RUNNER,
    evaluator: Path = EVALUATOR,
) -> list[TaskSpec]:
    selected = _selected_config(selection)
    tasks: list[TaskSpec] = []
    for variant in VARIANTS:
        for region_count in REGION_COUNTS[scene.key]:
            name = _region_name(region_count)
            directory = output_root / "test" / scene.key / variant / name
            result = directory / "result.json"
            metrics = directory / "metrics.json"
            tasks.append(TaskSpec(
                phase="test",
                scene=scene.key,
                name=f"{variant}_{name}",
                directory=directory,
                commands=(
                    (
                        "browser-test",
                        _runner_command(
                            scene,
                            variant,
                            result,
                            split="test",
                            width=selected["width"],
                            height=selected["height"],
                            depth_bias_m=selected["depthBiasM"],
                            region_sample_count=region_count,
                            timing_rounds=TEST_RUNNER_TIMING_ROUNDS,
                            node=node,
                            runner=runner,
                        ),
                    ),
                    (
                        "evaluate-test",
                        _evaluator_command(scene, result, metrics, "test", evaluator=evaluator),
                    ),
                ),
                expected_artifacts=VISIBILITY_FILES,
                config={
                    "shellVariant": variant,
                    "width": selected["width"],
                    "height": selected["height"],
                    "depthBiasM": selected["depthBiasM"],
                    "regionSampleCount": region_count,
                    "runnerTimingRounds": TEST_RUNNER_TIMING_ROUNDS,
                    "selection": str(_selection_path(output_root, scene).resolve()),
                    "selectionConfigSource": (
                        "dry-run-placeholder"
                        if selection.get("status") == "dry-run-placeholder"
                        else "frozen-calibration-selection"
                    ),
                    "testRead": True,
                },
            ))
    return tasks


def build_timing_tasks(
    scene: SceneSpec,
    output_root: Path,
    selection: Mapping[str, Any],
    *,
    node: str = "node",
    runner: Path = RUNNER,
) -> list[TaskSpec]:
    selected = _selected_config(selection)
    tasks: list[TaskSpec] = []
    for variant in VARIANTS:
        directory = output_root / "timing" / scene.key / variant
        result = directory / "result.json"
        tasks.append(TaskSpec(
            phase="timing",
            scene=scene.key,
            name=variant,
            directory=directory,
            commands=(
                (
                    "browser-timing",
                    _runner_command(
                        scene,
                        variant,
                        result,
                        split="test",
                        width=selected["width"],
                        height=selected["height"],
                        depth_bias_m=selected["depthBiasM"],
                        region_sample_count=0,
                        timing_rounds=TIMING_ROUNDS,
                        node=node,
                        runner=runner,
                        pose_index_plan=scene.timing_plan,
                    ),
                ),
            ),
            expected_artifacts=TIMING_FILES,
            config={
                "shellVariant": variant,
                "width": selected["width"],
                "height": selected["height"],
                "depthBiasM": selected["depthBiasM"],
                "regionSampleCount": 0,
                "poseIndexPlan": str(scene.timing_plan.resolve()),
                "poseCount": TIMING_POSE_COUNT,
                "timingRounds": TIMING_ROUNDS,
                "selectionConfigSource": (
                    "dry-run-placeholder"
                    if selection.get("status") == "dry-run-placeholder"
                    else "frozen-calibration-selection"
                ),
                "testRead": True,
            },
        ))
    return tasks


def _task_state(task: TaskSpec, validator: Callable[[TaskSpec], None] | None = None) -> str:
    directory = task.directory
    if not directory.exists():
        return "missing"
    if not directory.is_dir():
        raise IncompleteArtifactError(f"task path is not a directory: {directory}")
    if not any(directory.iterdir()):
        return "missing"
    missing = [name for name in task.expected_artifacts if not (directory / name).is_file()]
    if missing:
        raise IncompleteArtifactError(
            f"incomplete {task.phase} task {task.name}; missing {missing}: {directory}"
        )
    if validator is not None:
        try:
            validator(task)
        except Exception as error:
            raise IncompleteArtifactError(
                f"incomplete or invalid {task.phase} task {task.name}: {error}"
            ) from error
    return "complete"


def _command_text(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _run_logged(command: Sequence[str], task: TaskSpec, timeout_seconds: float, append: bool) -> dict[str, Any]:
    task.directory.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    started = time.time()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    try:
        with (task.directory / "stdout.log").open(mode, encoding="utf-8") as stdout, (task.directory / "stderr.log").open(mode, encoding="utf-8") as stderr:
            process = subprocess.run(
                [str(part) for part in command],
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{task.phase} task timed out after {timeout_seconds:g}s: {task.name}; see {task.directory}"
        ) from error
    record = {
        "command": [str(part) for part in command],
        "commandText": _command_text(command),
        "returnCode": int(process.returncode),
        "elapsedSeconds": time.time() - started,
    }
    if process.returncode != 0:
        raise RuntimeError(
            f"{task.phase} task command failed ({process.returncode}): {task.name}; see {task.directory / 'stderr.log'}"
        )
    return record


def _write_task_command(task: TaskSpec) -> None:
    _write_json(task.directory / "command.json", {
        "schema": "geometry-shell-hzb-paper-command-v1",
        "experiment": EXPERIMENT,
        "phase": task.phase,
        "scene": task.scene,
        "name": task.name,
        "commands": [
            {"label": label, "argv": list(command), "command": _command_text(command)}
            for label, command in task.commands
        ],
        "config": _json_value(task.config),
    })


def _write_task_manifest(task: TaskSpec, command_records: Sequence[Mapping[str, Any]]) -> None:
    _write_json(task.directory / "task.json", {
        "schema": TASK_SCHEMA,
        "experiment": EXPERIMENT,
        "phase": task.phase,
        "scene": task.scene,
        "name": task.name,
        "status": "complete",
        "formal": task.phase in {"calibrate", "test", "timing"},
        "testRead": bool(task.config.get("testRead", False)),
        "config": _json_value(task.config),
        "commands": _json_value(command_records),
        "artifacts": {
            name: str((task.directory / name).resolve())
            for name in task.expected_artifacts
        },
    })


def _execute_task(
    task: TaskSpec,
    *,
    dry_run: bool,
    timeout_seconds: float,
    validator: Callable[[TaskSpec], None] | None = None,
    after_step: Callable[[TaskSpec, str], None] | None = None,
) -> dict[str, Any]:
    existing = _task_state(task, validator)
    if existing == "complete":
        return {"phase": task.phase, "scene": task.scene, "name": task.name, "status": "skipped", "directory": str(task.directory.resolve())}
    if dry_run:
        return {
            "phase": task.phase,
            "scene": task.scene,
            "name": task.name,
            "status": "planned",
            "directory": str(task.directory.resolve()),
            "commands": [
                {"label": label, "argv": list(command), "command": _command_text(command)}
                for label, command in task.commands
            ],
            "config": _json_value(task.config),
        }
    task.directory.mkdir(parents=True, exist_ok=True)
    _write_task_command(task)
    records = []
    for label, command in task.commands:
        record = _run_logged(command, task, timeout_seconds, append=bool(records))
        record = {"label": label, **record}
        records.append(record)
        if after_step is not None:
            after_step(task, label)
    if validator is not None:
        validator(task)
    _write_task_manifest(task, records)
    return {
        "phase": task.phase,
        "scene": task.scene,
        "name": task.name,
        "status": "completed",
        "directory": str(task.directory.resolve()),
    }


def _validate_preflight_task(task: TaskSpec) -> None:
    payload = _read_json(task.directory / "preflight.json")
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("testRead") is not False:
        raise ValueError("preflight artifact has an invalid schema or testRead state")
    if payload.get("scene") != task.scene:
        raise ValueError("preflight scene does not match task")


def _validate_formal_result(task: TaskSpec) -> dict[str, Any]:
    result = _read_json(task.result_path)
    # This check deliberately precedes all evaluator calls.  A smoke result is
    # retained for diagnosis but can never become a metrics input.
    if result.get("formalReady") is not True:
        raise FormalArtifactError(
            f"{task.phase} result formalReady is not true; evaluator was not run: {task.result_path}"
        )
    if result.get("schema") != RESULT_SCHEMA:
        raise FormalArtifactError("browser result schema is not formal HZB schema")
    if result.get("executionClass") != "formal-hardware-gpu":
        raise FormalArtifactError("browser result is not classified as formal-hardware-gpu")
    if result.get("error") not in (None, ""):
        raise FormalArtifactError("formal browser result contains an error")
    gate = result.get("gpuGate")
    adapter = gate.get("adapter") if isinstance(gate, Mapping) else None
    concurrency = result.get("gpuConcurrency")
    if (
        not isinstance(gate, Mapping)
        or gate.get("required") is not True
        or gate.get("hardware") is not True
        or not isinstance(adapter, Mapping)
        or "nvidia" not in str(adapter.get("vendor", "")).lower()
        or not isinstance(concurrency, Mapping)
        or concurrency.get("concurrentComputeDetected") is not False
    ):
        raise FormalArtifactError("formal browser result lacks independent NVIDIA/no-concurrency evidence")
    workload = result.get("workload")
    if not isinstance(workload, Mapping):
        raise FormalArtifactError("formal browser result has no workload")
    if workload.get("schema") != "geometry-shell-hzb-browser-workload-v1":
        raise FormalArtifactError("browser workload schema is invalid")
    if workload.get("mode") != "Region66" or workload.get("split") != ("calibration" if task.phase == "calibrate" else "test"):
        raise FormalArtifactError("browser workload mode or split does not match task")
    if workload.get("poseSelection", {}).get("representative") is not True:
        raise FormalArtifactError("formal task cannot use a limited/non-representative pose selection")
    if _int(workload.get("width", -1), "workload width") != _int(task.config["width"], "task width"):
        raise FormalArtifactError("browser workload width does not match task")
    if _int(workload.get("height", -1), "workload height") != _int(task.config["height"], "task height"):
        raise FormalArtifactError("browser workload height does not match task")
    if not math.isclose(_number(workload.get("depthBiasM"), "workload depthBiasM"), float(task.config["depthBiasM"]), rel_tol=0.0, abs_tol=1e-9):
        raise FormalArtifactError("browser workload depth bias does not match task")
    if _int(workload.get("timingRounds", -1), "workload timingRounds") != _int(task.config.get("runnerTimingRounds", task.config.get("timingRounds", -1)), "task timingRounds"):
        raise FormalArtifactError("browser workload timing round count does not match task")
    region_sampling = workload.get("regionSampling")
    if not isinstance(region_sampling, Mapping) or _int(region_sampling.get("requestedCount", -1), "region requestedCount") != _int(task.config["regionSampleCount"], "task regionSampleCount"):
        raise FormalArtifactError("browser workload region sampling does not match task")
    samples = result.get("samples")
    expected_count = int(task.config["poseCount"]) if task.phase == "timing" else EXPECTED_SPLITS[task.scene]["calibration" if task.phase == "calibrate" else "test"]
    if not isinstance(samples, list) or len(samples) != expected_count:
        raise FormalArtifactError(f"browser result has {len(samples) if isinstance(samples, list) else 'no'} samples; expected {expected_count}")
    if task.phase == "timing":
        plan = _read_json(Path(str(task.config["poseIndexPlan"])))
        planned_indices = [_int(value, "timing plan pose index") for value in plan["poseIndices"]]
        actual_indices = [_int(sample.get("poseId", -1), "timing result poseId") for sample in samples]
        if actual_indices != planned_indices:
            raise FormalArtifactError("timing result pose IDs do not match the explicit 120-pose plan")
        timing_summary = result.get("timingSummary")
        if not isinstance(timing_summary, Mapping) or _int(timing_summary.get("roundCount", -1), "timing roundCount") != TIMING_ROUNDS:
            raise FormalArtifactError("timing result does not contain five summarized rounds")
    return result


def _validate_gpu_evidence(task: TaskSpec) -> None:
    evidence = _read_json(task.directory / "geometry_shell_hzb_gpu_evidence.json")
    if evidence.get("schema") != "geometry-shell-hzb-gpu-evidence-v1":
        raise FormalArtifactError("GPU evidence schema is invalid")
    if evidence.get("formalReady") is not True:
        raise FormalArtifactError("GPU evidence formalReady is not true")


def _validate_metrics(task: TaskSpec) -> None:
    _validate_gpu_evidence(task)
    _validate_formal_result(task)
    metrics = _read_json(task.metrics_path)
    expected_split = "calibration" if task.phase == "calibrate" else "test"
    expected_count = EXPECTED_SPLITS[task.scene][expected_split]
    if metrics.get("schema") != METRICS_SCHEMA:
        raise ValueError("HZB metrics schema is invalid")
    if metrics.get("mode") != "Region66" or metrics.get("split") != expected_split:
        raise ValueError("HZB metrics mode or split is invalid")
    if metrics.get("formalReady") is not True or metrics.get("executionClass") != "formal-hardware-gpu":
        raise FormalArtifactError("HZB metrics are not formal hardware metrics")
    if _int(metrics.get("evaluatedPoseCount", -1), "evaluatedPoseCount") != expected_count:
        raise ValueError("HZB metrics do not cover the complete split")
    if not isinstance(metrics.get("aggregate"), Mapping) or not isinstance(metrics.get("poseMacro"), Mapping):
        raise ValueError("HZB metrics lack aggregate/poseMacro metrics")
    for field in ("weightedRecall", "weightedRecallLower95"):
        _number(metrics.get(field), field)
    ground_truth = metrics.get("groundTruth")
    if not isinstance(ground_truth, Mapping) or ground_truth.get("mode") != "viewcell-union":
        raise ValueError("Region66 metrics must use viewcell-union ground truth")


def _validate_visibility_task(task: TaskSpec) -> None:
    _validate_metrics(task)


def _validate_timing_task(task: TaskSpec) -> None:
    _validate_gpu_evidence(task)
    _validate_formal_result(task)


def _validate_selection_task(task: TaskSpec) -> None:
    selection = _read_json(task.directory / "selection.json")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("selection schema is invalid")
    if selection.get("selectionSplit") != "calibration" or selection.get("testRead") is not False:
        raise ValueError("selection is not calibration-only")
    rows = selection.get("rows")
    if not isinstance(rows, list) or len(rows) != len(CALIBRATION_RESOLUTIONS) * len(CALIBRATION_DEPTH_BIASES_M):
        raise ValueError("selection does not contain the complete calibration matrix")
    _selected_config(selection)


def _selection_payload(task: TaskSpec) -> dict[str, Any]:
    _task_state(task, _validate_selection_task)
    return _read_json(task.directory / "selection.json")


def _run_preflight_task(
    scene: SceneSpec,
    output_root: Path,
    *,
    dry_run: bool,
    point60_plan_builder: Path = POINT60_PLAN_BUILDER,
    sampler: Path = SAMPLER,
    node: str = "node",
) -> dict[str, Any]:
    directory = output_root / "preflight" / scene.key
    task = TaskSpec(
        phase="preflight",
        scene=scene.key,
        name="preflight",
        directory=directory,
        commands=(),
        expected_artifacts=("preflight.json", "stdout.log", "stderr.log"),
        config={"testRead": False},
    )
    existing = _task_state(task, _validate_preflight_task)
    if existing == "complete":
        return {"phase": "preflight", "scene": scene.key, "name": "preflight", "status": "skipped", "directory": str(directory.resolve())}
    report = preflight_scene(
        scene,
        output_root,
        point60_plan_builder=point60_plan_builder,
        sampler=sampler,
        node=node,
    )
    if dry_run:
        return {
            "phase": "preflight",
            "scene": scene.key,
            "name": "preflight",
            "status": "planned",
            "directory": str(directory.resolve()),
            "point60": report["point60"],
        }
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "preflight.json", report)
    (directory / "stdout.log").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "stderr.log").write_text("", encoding="utf-8")
    _write_task_manifest(task, ())
    return {"phase": "preflight", "scene": scene.key, "name": "preflight", "status": "completed", "directory": str(directory.resolve())}


def _run_calibration(
    scenes: Sequence[SceneSpec],
    output_root: Path,
    *,
    dry_run: bool,
    node: str,
    runner: Path,
    evaluator: Path,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, list[TaskSpec]]]:
    records: list[dict[str, Any]] = []
    calibration_by_scene: dict[str, list[TaskSpec]] = {}
    for scene in scenes:
        tasks = build_calibration_tasks(
            scene,
            output_root,
            node=node,
            runner=runner,
            evaluator=evaluator,
        )
        calibration_by_scene[scene.key] = tasks
        for task in tasks:
            records.append(_execute_task(
                task,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
                validator=_validate_visibility_task,
                after_step=lambda current, label: _validate_formal_result(current) if label == "browser-calibration" else None,
            ))
    return records, calibration_by_scene


def _run_selection(
    scenes: Sequence[SceneSpec],
    output_root: Path,
    calibration_by_scene: Mapping[str, Sequence[TaskSpec]],
    *,
    dry_run: bool,
    selector: Path,
    evaluator: Path,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        calibration_tasks = list(
            calibration_by_scene.get(scene.key)
            or build_calibration_tasks(scene, output_root, evaluator=evaluator)
        )
        if not dry_run:
            for task in calibration_tasks:
                _task_state(task, _validate_visibility_task)
        task = build_selection_task(scene, output_root, calibration_tasks, selector=selector)
        record = _execute_task(
            task,
            dry_run=dry_run,
            timeout_seconds=timeout_seconds,
            validator=_validate_selection_task,
        )
        records.append(record)
        if not dry_run or record["status"] == "skipped":
            selections[scene.key] = _selection_payload(task)
    return records, selections


def _require_frozen_selections(scenes: Sequence[SceneSpec], output_root: Path) -> dict[str, dict[str, Any]]:
    selections: dict[str, dict[str, Any]] = {}
    for scene in scenes:
        path = _selection_path(output_root, scene)
        task = TaskSpec(
            phase="select",
            scene=scene.key,
            name="selection",
            directory=path.parent,
            commands=(),
            expected_artifacts=("selection.json", "stdout.log", "stderr.log"),
            config={"selectionSplit": "calibration"},
        )
        if not path.exists():
            raise FileNotFoundError(
                f"frozen selection is required before touching test/timing tasks: {path}"
            )
        selections[scene.key] = _selection_payload(task)
    return selections


def _run_test_or_timing(
    phase: str,
    scenes: Sequence[SceneSpec],
    output_root: Path,
    selections: Mapping[str, Mapping[str, Any]],
    *,
    dry_run: bool,
    node: str,
    runner: Path,
    evaluator: Path,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in scenes:
        selection = selections[scene.key]
        tasks = (
            build_test_tasks(
                scene,
                output_root,
                selection,
                node=node,
                runner=runner,
                evaluator=evaluator,
            )
            if phase == "test"
            else build_timing_tasks(scene, output_root, selection, node=node, runner=runner)
        )
        for task in tasks:
            records.append(_execute_task(
                task,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
                validator=_validate_visibility_task if phase == "test" else _validate_timing_task,
                after_step=lambda current, label: _validate_formal_result(current) if label.startswith("browser-") else None,
            ))
    return records


def _dry_run_selection() -> dict[str, Any]:
    """Provide dimensions only so all-mode dry-run can print later commands."""
    return {
        "schema": SELECTION_SCHEMA,
        "selectionSplit": "calibration",
        "status": "dry-run-placeholder",
        "testRead": False,
        "selected": {
            "assetVariant": "lossless",
            "width": CALIBRATION_RESOLUTIONS[0][0],
            "height": CALIBRATION_RESOLUTIONS[0][1],
            "depthBiasM": CALIBRATION_DEPTH_BIASES_M[1],
        },
    }


def _parse_scene_keys(value: str) -> list[str]:
    aliases = {"hkust": "hkust", "hkust_v3": "hkust", "ifc": "ifcbench", "ifcbench": "ifcbench"}
    keys = []
    for token in str(value).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in aliases:
            raise ValueError(f"unknown scene {token!r}; choose hkust or ifcbench")
        key = aliases[token]
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("at least one scene is required")
    return keys


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "calibrate", "select", "test", "timing", "all"))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--hzb-root", type=Path, default=ROOT / "neural_instance_culling" / "benchmark" / "out" / "paper_results" / "hzb")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scenes", default="hkust,ifcbench")
    parser.add_argument("--node", default="node")
    parser.add_argument("--runner", type=Path, default=RUNNER)
    parser.add_argument("--evaluator", type=Path, default=EVALUATOR)
    parser.add_argument("--selector", type=Path, default=SELECTOR)
    parser.add_argument("--timeout-seconds", type=float, default=3 * 60 * 60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is None:
        args.output_root = args.hzb_root / "geometry_shell_hzb_paper_2026-09-09"
    if not math.isfinite(float(args.timeout_seconds)) or args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive and finite")
    try:
        args.scene_keys = _parse_scene_keys(args.scenes)
    except ValueError as error:
        parser.error(str(error))
    args.data_root = args.data_root.resolve()
    args.hzb_root = args.hzb_root.resolve()
    args.output_root = args.output_root.resolve()
    args.runner = args.runner.resolve()
    args.evaluator = args.evaluator.resolve()
    args.selector = args.selector.resolve()
    return args


def run(args: argparse.Namespace) -> dict[str, Any]:
    specs = _scene_specs(args.data_root, args.hzb_root)
    scenes = [specs[key] for key in args.scene_keys]
    mode = str(args.mode)
    records: list[dict[str, Any]] = []
    calibration_by_scene: dict[str, list[TaskSpec]] = {}
    selections: dict[str, dict[str, Any]] = {}

    # Test and timing require selection before any preflight task is allowed to
    # create a test directory.  This check is deliberately first for those
    # direct modes.
    if mode in {"test", "timing"}:
        selections = _require_frozen_selections(scenes, args.output_root)

    if mode in {"preflight", "calibrate", "select", "all", "test", "timing"}:
        for scene in scenes:
            records.append(
                _run_preflight_task(
                    scene,
                    args.output_root,
                    dry_run=args.dry_run,
                    point60_plan_builder=args.data_root / "neural_instance_culling" / "benchmark" / "build_hzb_point60_gt_plan.py",
                    sampler=args.data_root / "neural_instance_culling" / "sampler" / "run_sampler.mjs",
                    node=args.node,
                )
            )

    if mode in {"calibrate", "all"}:
        calibration_records, calibration_by_scene = _run_calibration(
            scenes,
            args.output_root,
            dry_run=args.dry_run,
            node=args.node,
            runner=args.runner,
            evaluator=args.evaluator,
            timeout_seconds=args.timeout_seconds,
        )
        records.extend(calibration_records)

    if mode in {"select", "all"}:
        selection_records, new_selections = _run_selection(
            scenes,
            args.output_root,
            calibration_by_scene,
            dry_run=args.dry_run,
            selector=args.selector,
            evaluator=args.evaluator,
            timeout_seconds=args.timeout_seconds,
        )
        records.extend(selection_records)
        selections.update(new_selections)

    if mode in {"test", "timing", "all"}:
        if mode == "all":
            selections = (
                _require_frozen_selections(scenes, args.output_root)
                if not args.dry_run
                else {
                    scene.key: selections.get(scene.key, _dry_run_selection())
                    for scene in scenes
                }
            )
        if mode in {"test", "all"}:
            records.extend(_run_test_or_timing(
                "test", scenes, args.output_root, selections,
                dry_run=args.dry_run, node=args.node, runner=args.runner,
                evaluator=args.evaluator,
                timeout_seconds=args.timeout_seconds,
            ))
        if mode in {"timing", "all"}:
            records.extend(_run_test_or_timing(
                "timing", scenes, args.output_root, selections,
                dry_run=args.dry_run, node=args.node, runner=args.runner,
                evaluator=args.evaluator,
                timeout_seconds=args.timeout_seconds,
            ))

    formal_gpu_executed = any(
        row.get("status") == "completed"
        and row.get("phase") in {"calibrate", "test", "timing"}
        for row in records
    )
    point60_handoffs = {
        scene.key: _point60_handoff(
            scene,
            args.output_root,
            point60_plan_builder=args.data_root / "neural_instance_culling" / "benchmark" / "build_hzb_point60_gt_plan.py",
            sampler=args.data_root / "neural_instance_culling" / "sampler" / "run_sampler.mjs",
            node=args.node,
        )
        for scene in scenes
    }
    summary = {
        "schema": "geometry-shell-hzb-paper-execution-v1",
        "experiment": EXPERIMENT,
        "mode": mode,
        "dryRun": bool(args.dry_run),
        "formalGpuExecuted": formal_gpu_executed,
        "testRead": mode in {"test", "timing", "all"} and not args.dry_run,
        "calibration": {
            "resolutions": [list(value) for value in CALIBRATION_RESOLUTIONS],
            "depthBiasM": list(CALIBRATION_DEPTH_BIASES_M),
            "rounds": CALIBRATION_ROUNDS,
            "invocationsPerConfiguration": CALIBRATION_INVOCATIONS_PER_CONFIGURATION,
            "runnerTimingRounds": CALIBRATION_RUNNER_TIMING_ROUNDS,
        },
        "testRegionSampleCounts": {key: list(value) for key, value in REGION_COUNTS.items()},
        "timing": {"poseCount": TIMING_POSE_COUNT, "rounds": TIMING_ROUNDS},
        "scenes": [scene.key for scene in scenes],
        "outputRoot": str(args.output_root),
        "records": records,
        "point60": point60_handoffs,
    }
    if not args.dry_run:
        _write_json(args.output_root / "execution_summary.json", summary)
    print(json.dumps(_json_value(summary), ensure_ascii=False, indent=2))
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, IncompleteArtifactError, FormalArtifactError, RuntimeError) as error:
        print(f"geometry-shell-hzb paper orchestration failed: {error}", file=sys.stderr)
        raise SystemExit(1)
