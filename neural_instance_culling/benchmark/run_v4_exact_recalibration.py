#!/usr/bin/env python3
"""Orchestrate exact V4 recalibration for the existing paper checkpoints.

This runner never trains and never evaluates the test split.  It registers the
three existing HKUST and three existing IFCBench Full V4 checkpoints, scores
calibration and validation with :mod:`v4_exact_calibration`, freezes one
checkpoint-specific threshold from calibration, and replays that threshold on
validation.  ``plan`` and ``preflight`` are read-only with respect to model
artifacts; ``run`` writes only the new exact-recalibration result tree.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

try:
    from .v4_exact_calibration import (  # noqa: E402
        CALIBRATION_SCHEMA,
        FORMAL_BOOTSTRAP_REPLICATES,
        SCORE_SIDECAR_SCHEMA,
        VALIDATION_SCHEMA,
    )
except ImportError:  # Direct script execution.
    from v4_exact_calibration import (  # noqa: E402
        CALIBRATION_SCHEMA,
        FORMAL_BOOTSTRAP_REPLICATES,
        SCORE_SIDECAR_SCHEMA,
        VALIDATION_SCHEMA,
    )


ROOT = Path(__file__).resolve().parents[2]
V4_EXACT_CALIBRATION = ROOT / "neural_instance_culling" / "benchmark" / "v4_exact_calibration.py"
DEFAULT_OUTPUT_ROOT = (
    ROOT
    / "neural_instance_culling"
    / "benchmark"
    / "out"
    / "paper_results"
    / "exact_recalibration_v1"
)

RUNNER_SCHEMA = "pvs-v4-exact-recalibration-runner-v1"
PLAN_SCHEMA = "pvs-v4-exact-recalibration-plan-v1"
PREFLIGHT_SCHEMA = "pvs-v4-exact-recalibration-preflight-v1"
COMMAND_SCHEMA = "pvs-v4-exact-recalibration-command-v1"
RUN_SCHEMA = "pvs-v4-exact-recalibration-run-v1"
SUMMARY_SCHEMA = "pvs-v4-exact-recalibration-summary-v1"

SEEDS = (20260801, 20260802, 20260803)
EPOCHS = 40
GEOMETRY_DIM = 96
CALIBRATION_BOOTSTRAP_SEED = 20260909
VALIDATION_BOOTSTRAP_SEED = 20260910
WEIGHTED_RECALL_FLOOR = 0.99

EXPECTED_SPLITS: dict[str, dict[str, int]] = {
    "hkust": {
        "train": 5926,
        "calibration": 659,
        "validation": 730,
        "test": 684,
        "guard": 0,
    },
    "ifcbench": {
        "train": 19647,
        "calibration": 2183,
        "validation": 2712,
        "test": 2710,
        "guard": 0,
    },
}


@dataclass(frozen=True)
class SceneSpec:
    key: str
    scene_name: str
    dataset_rel: str
    runtime_meta_rel: str
    geometry_96d_rel: str
    split_counts: Mapping[str, int]


SCENES: dict[str, SceneSpec] = {
    "hkust": SceneSpec(
        key="hkust",
        scene_name="HKUST",
        dataset_rel=(
            "neural_instance_culling/dataset/out/"
            "pose_csr_hkust_v3_main_stratified_calibration_fov66_v1"
        ),
        runtime_meta_rel="hkust-v3/assets/runtimeVisibilityMeta.json",
        geometry_96d_rel=(
            "neural_instance_culling/dataset/out/"
            "fixed_geometry_features_hkust_v3/instance_geo_features_fp16.bin"
        ),
        split_counts=EXPECTED_SPLITS["hkust"],
    ),
    "ifcbench": SceneSpec(
        key="ifcbench",
        scene_name="IFCBench/Fantasy Metropolis",
        dataset_rel=(
            "neural_instance_culling/dataset/out/"
            "pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1"
        ),
        runtime_meta_rel=(
            "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json"
        ),
        geometry_96d_rel=(
            "neural_instance_culling/dataset/out/"
            "fixed_geometry_features_metropolis_v2/instance_geo_features_fp16.bin"
        ),
        split_counts=EXPECTED_SPLITS["ifcbench"],
    ),
}


CHECKPOINT_RELATIVE: dict[str, str] = {
    "hkust": (
        "neural_instance_culling/model/out/"
        "pvs_v4_integrated_visibility_mainline_v1/"
        "paper_full_seed{seed}_e40/best_safe.pt"
    ),
    "ifcbench": (
        "neural_instance_culling/model/out/"
        "pvs_mainline_v4_ifcbench_fantasy_metropolis_v1/"
        "full_seed{seed}_e40/best_safe.pt"
    ),
}


@dataclass(frozen=True)
class MemberSpec:
    key: str
    scene_key: str
    scene_name: str
    seed: int
    checkpoint: Path
    dataset: Path
    runtime_meta: Path
    geometry_96d: Path

    @property
    def label(self) -> str:
        return self.checkpoint.parent.name

    @property
    def output_name(self) -> str:
        return f"{self.scene_key}_{self.label}"


@dataclass(frozen=True)
class JobSpec:
    member: MemberSpec
    step: str
    command: tuple[str, ...]
    output: Path
    expected_artifact: Path
    bootstrap: Path | None

    @property
    def name(self) -> str:
        return f"{self.member.output_name}_{self.step}"

    @property
    def command_record(self) -> Path:
        return self.member_root / "commands" / f"{self.step}.json"

    @property
    def member_root(self) -> Path:
        return self.output.parent

    @property
    def log_dir(self) -> Path:
        return self.member_root / "logs"

    def manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "scene": self.member.scene_key,
            "sceneName": self.member.scene_name,
            "member": self.member.output_name,
            "seed": int(self.member.seed),
            "step": self.step,
            "command": list(self.command),
            "output": str(self.output.resolve()),
            "expectedArtifact": str(self.expected_artifact.resolve()),
            "bootstrap": None if self.bootstrap is None else str(self.bootstrap.resolve()),
            "testRead": False,
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _assert_test_free_payload(payload: Mapping[str, Any], label: str) -> None:
    if payload.get("testRead") is not False:
        raise ValueError(f"{label} must declare testRead=false")
    if str(payload.get("split", "")).casefold() == "test":
        raise ValueError(f"{label} is a test-split artifact")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _require_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def scene_paths(data_root: str | Path, scene_key: str) -> dict[str, Path]:
    if scene_key not in SCENES:
        raise ValueError(f"unknown registered scene: {scene_key}")
    root = Path(data_root).expanduser().resolve()
    scene = SCENES[scene_key]
    paths = {
        "dataset": root / scene.dataset_rel,
        "runtime_meta": root / scene.runtime_meta_rel,
        "geometry_96d": root / scene.geometry_96d_rel,
    }
    return paths


def registered_members(data_root: str | Path = ROOT) -> tuple[MemberSpec, ...]:
    members: list[MemberSpec] = []
    for scene_key, scene in SCENES.items():
        paths = scene_paths(data_root, scene_key)
        for seed in SEEDS:
            checkpoint = Path(data_root).expanduser().resolve() / CHECKPOINT_RELATIVE[scene_key].format(seed=seed)
            members.append(
                MemberSpec(
                    key=f"{scene_key}_full_seed{seed}_e{EPOCHS}",
                    scene_key=scene_key,
                    scene_name=scene.scene_name,
                    seed=seed,
                    checkpoint=checkpoint,
                    dataset=paths["dataset"],
                    runtime_meta=paths["runtime_meta"],
                    geometry_96d=paths["geometry_96d"],
                )
            )
    return tuple(members)


REGISTERED_MEMBERS = registered_members(ROOT)


def _output_root_is_safe(output_root: str | Path, members: Sequence[MemberSpec]) -> Path:
    root = Path(output_root).expanduser().resolve()
    for member in members:
        checkpoint_dir = member.checkpoint.parent.resolve()
        if root == checkpoint_dir or checkpoint_dir in root.parents:
            raise ValueError(
                "exact recalibration output must not be inside a checkpoint directory: "
                f"{root}"
            )
    return root


def _runtime_instance_count(payload: Mapping[str, Any], label: str) -> int:
    records = payload.get("componentRecords")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} has no componentRecords")
    ids = [int(record["componentGlobalId"]) for record in records]
    if min(ids) < 0 or len(set(ids)) != len(ids):
        raise ValueError(f"{label} componentGlobalId values are not unique and non-negative")
    inferred = max(ids) + 1
    declared = int(payload.get("instanceCount", inferred))
    if declared != inferred or len(records) != declared:
        raise ValueError(f"{label} instance count disagrees with component records")
    return declared


def _preflight_scene(data_root: Path, scene: SceneSpec) -> dict[str, Any]:
    paths = scene_paths(data_root, scene.key)
    _require_dir(paths["dataset"], f"{scene.key} dataset")
    for name in ("runtime_meta", "geometry_96d"):
        _require_file(paths[name], f"{scene.key} {name}")

    dataset_meta = _read_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("testRead") is True:
        raise ValueError(f"{scene.key} dataset metadata declares testRead=true")
    if dataset_meta.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError(f"{scene.key} does not use the explicit four-way Pose CSR schema")
    split_counts = {
        str(name): int(count)
        for name, count in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != dict(scene.split_counts):
        raise ValueError(
            f"{scene.key} split metadata changed: expected={dict(scene.split_counts)}, "
            f"actual={split_counts}"
        )
    files = dataset_meta.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{scene.key} dataset metadata has no files map")
    for field in (
        "poses",
        "visibleOffsets",
        "visibleIds",
        "visibleWeights",
        "candidateOffsets",
        "candidateIds",
        "queryCenterWorld",
        "candidateCameraWorld",
        "viewcellRadiusM",
    ):
        if field not in files:
            raise ValueError(f"{scene.key} dataset metadata is missing {field}")
        _require_file(paths["dataset"] / str(files[field]), f"{scene.key} {field}")

    runtime_meta = _read_json(paths["runtime_meta"])
    instance_count = _runtime_instance_count(runtime_meta, f"{scene.key} runtime metadata")
    if int(dataset_meta.get("numInstances", -1)) != instance_count:
        raise ValueError(f"{scene.key} dataset/runtime instance counts disagree")
    expected_geometry_bytes = instance_count * GEOMETRY_DIM * 2
    if paths["geometry_96d"].stat().st_size != expected_geometry_bytes:
        raise ValueError(
            f"{scene.key} 96D geometry size is wrong: expected {expected_geometry_bytes} bytes, "
            f"actual {paths['geometry_96d'].stat().st_size}"
        )
    return {
        "scene": scene.key,
        "sceneName": scene.scene_name,
        "instanceCount": instance_count,
        "geometryDim": GEOMETRY_DIM,
        "splitCounts": split_counts,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
        "testRead": False,
    }


def preflight(
    data_root: str | Path = ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    members = registered_members(root)
    output = _output_root_is_safe(output_root, members)
    _require_file(V4_EXACT_CALIBRATION, "generic V4 exact calibration CLI")
    scenes = [_preflight_scene(root, scene) for scene in SCENES.values()]
    for member in members:
        _require_file(member.checkpoint, f"{member.output_name} checkpoint")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "runnerSchema": RUNNER_SCHEMA,
        "experiment": "pvs_v4_exact_recalibration_existing_full_v1",
        "outputRoot": str(output),
        "formalBootstrapReplicates": FORMAL_BOOTSTRAP_REPLICATES,
        "calibrationBootstrapSeed": CALIBRATION_BOOTSTRAP_SEED,
        "validationBootstrapSeed": VALIDATION_BOOTSTRAP_SEED,
        "scenes": scenes,
        "members": [
            {
                "key": member.key,
                "scene": member.scene_key,
                "sceneName": member.scene_name,
                "seed": member.seed,
                "checkpoint": str(member.checkpoint.resolve()),
                "dataset": str(member.dataset.resolve()),
                "runtimeMeta": str(member.runtime_meta.resolve()),
                "geometry96D": str(member.geometry_96d.resolve()),
            }
            for member in members
        ],
        "testRead": False,
    }


def _member_root(output_root: Path, member: MemberSpec) -> Path:
    return output_root / member.scene_key / member.output_name


def _bootstrap_path(output_root: Path, scene_key: str, split: str) -> Path:
    if split not in {"calibration", "validation"}:
        raise ValueError(f"exact recalibration does not support split={split!r}")
    return output_root / scene_key / "fixed_bootstrap" / f"{split}_pose_indices_i32.bin"


def build_score_command(
    member: MemberSpec,
    split: str,
    output_root: str | Path,
    *,
    device: str = "cuda",
) -> list[str]:
    if split not in {"calibration", "validation"}:
        raise ValueError("exact recalibration scoring is limited to calibration and validation")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"unsupported exact recalibration device: {device}")
    output = _member_root(Path(output_root).expanduser().resolve(), member) / f"{split}_scores"
    command = [
        sys.executable,
        str(V4_EXACT_CALIBRATION),
        "score",
        "--checkpoint",
        str(member.checkpoint.resolve()),
        "--dataset-dir",
        str(member.dataset.resolve()),
        "--runtime-meta",
        str(member.runtime_meta.resolve()),
        "--initial-geo-features",
        str(member.geometry_96d.resolve()),
        "--scene",
        member.scene_name,
        "--split",
        split,
        "--output-dir",
        str(output),
        "--poses-per-batch",
        "2",
        "--device",
        device,
    ]
    if any("test" == part.casefold() for part in command):
        raise ValueError("exact recalibration command unexpectedly contains a test token")
    return command


def build_calibrate_command(
    member: MemberSpec,
    output_root: str | Path,
) -> list[str]:
    root = Path(output_root).expanduser().resolve()
    member_root = _member_root(root, member)
    return [
        sys.executable,
        str(V4_EXACT_CALIBRATION),
        "calibrate",
        "--sidecar",
        str((member_root / "calibration_scores").resolve()),
        "--bootstrap-indexes",
        str(_bootstrap_path(root, member.scene_key, "calibration").resolve()),
        "--output",
        str((member_root / "exact_calibration.json").resolve()),
        "--bootstrap-replicates",
        str(FORMAL_BOOTSTRAP_REPLICATES),
        "--bootstrap-seed",
        str(CALIBRATION_BOOTSTRAP_SEED),
    ]


def build_validation_command(
    member: MemberSpec,
    output_root: str | Path,
) -> list[str]:
    root = Path(output_root).expanduser().resolve()
    member_root = _member_root(root, member)
    return [
        sys.executable,
        str(V4_EXACT_CALIBRATION),
        "evaluate",
        "--sidecar",
        str((member_root / "validation_scores").resolve()),
        "--calibration",
        str((member_root / "exact_calibration.json").resolve()),
        "--bootstrap-indexes",
        str(_bootstrap_path(root, member.scene_key, "validation").resolve()),
        "--output",
        str((member_root / "validation_frozen.json").resolve()),
        "--bootstrap-replicates",
        str(FORMAL_BOOTSTRAP_REPLICATES),
        "--bootstrap-seed",
        str(VALIDATION_BOOTSTRAP_SEED),
    ]


def _job_specs(
    members: Sequence[MemberSpec],
    output_root: Path,
    *,
    device: str,
) -> dict[str, list[JobSpec]]:
    phases: dict[str, list[JobSpec]] = {
        "score_calibration": [],
        "calibrate": [],
        "score_validation": [],
        "evaluate_validation": [],
    }
    for member in members:
        member_root = _member_root(output_root, member)
        calibration_scores = member_root / "calibration_scores"
        validation_scores = member_root / "validation_scores"
        phases["score_calibration"].append(
            JobSpec(
                member=member,
                step="calibration_scores",
                command=tuple(build_score_command(member, "calibration", output_root, device=device)),
                output=calibration_scores,
                expected_artifact=calibration_scores / "sidecar_manifest.json",
                bootstrap=None,
            )
        )
        calibration = member_root / "exact_calibration.json"
        phases["calibrate"].append(
            JobSpec(
                member=member,
                step="exact_calibration",
                command=tuple(build_calibrate_command(member, output_root)),
                output=calibration,
                expected_artifact=calibration,
                bootstrap=_bootstrap_path(output_root, member.scene_key, "calibration"),
            )
        )
        phases["score_validation"].append(
            JobSpec(
                member=member,
                step="validation_scores",
                command=tuple(build_score_command(member, "validation", output_root, device=device)),
                output=validation_scores,
                expected_artifact=validation_scores / "sidecar_manifest.json",
                bootstrap=None,
            )
        )
        validation = member_root / "validation_frozen.json"
        phases["evaluate_validation"].append(
            JobSpec(
                member=member,
                step="validation_frozen",
                command=tuple(build_validation_command(member, output_root)),
                output=validation,
                expected_artifact=validation,
                bootstrap=_bootstrap_path(output_root, member.scene_key, "validation"),
            )
        )
    return phases


def build_plan(
    data_root: str | Path = ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    root = Path(data_root).expanduser().resolve()
    members = registered_members(root)
    output = _output_root_is_safe(output_root, members)
    phases = _job_specs(members, output, device=device)
    return {
        "schema": PLAN_SCHEMA,
        "runnerSchema": RUNNER_SCHEMA,
        "experiment": "pvs_v4_exact_recalibration_existing_full_v1",
        "outputRoot": str(output),
        "device": device,
        "formalBootstrapReplicates": FORMAL_BOOTSTRAP_REPLICATES,
        "phases": {
            name: [job.manifest() for job in jobs]
            for name, jobs in phases.items()
        },
        "jobCount": sum(len(jobs) for jobs in phases.values()),
        "testRead": False,
    }


def _expected_payload(job: JobSpec) -> dict[str, Any]:
    payload = _read_json(job.expected_artifact)
    _assert_test_free_payload(payload, str(job.expected_artifact))
    if job.step == "calibration_scores":
        if payload.get("schema") != SCORE_SIDECAR_SCHEMA or payload.get("split") != "calibration":
            raise ValueError(f"invalid calibration sidecar artifact: {job.expected_artifact}")
    elif job.step == "validation_scores":
        if payload.get("schema") != SCORE_SIDECAR_SCHEMA or payload.get("split") != "validation":
            raise ValueError(f"invalid validation sidecar artifact: {job.expected_artifact}")
    elif job.step == "exact_calibration":
        if payload.get("schema") != CALIBRATION_SCHEMA or payload.get("split") != "calibration":
            raise ValueError(f"invalid exact calibration artifact: {job.expected_artifact}")
        if job.bootstrap is None or not job.bootstrap.is_file() or not Path(f"{job.bootstrap}.json").is_file():
            raise ValueError(f"exact calibration bootstrap artifact is missing: {job.bootstrap}")
    elif job.step == "validation_frozen":
        if payload.get("schema") != VALIDATION_SCHEMA or payload.get("split") != "validation":
            raise ValueError(f"invalid frozen validation artifact: {job.expected_artifact}")
        if job.bootstrap is None or not job.bootstrap.is_file() or not Path(f"{job.bootstrap}.json").is_file():
            raise ValueError(f"validation bootstrap artifact is missing: {job.bootstrap}")
    return payload


def _job_complete(job: JobSpec) -> bool:
    if job.expected_artifact.is_file():
        _expected_payload(job)
        return True
    if job.output.is_dir() and any(job.output.iterdir()):
        raise RuntimeError(f"incomplete exact recalibration output: {job.output}")
    if job.output.is_file():
        raise RuntimeError(f"incomplete exact recalibration output: {job.output}")
    if job.command_record.is_file():
        raise RuntimeError(f"command record exists without a completed artifact: {job.command_record}")
    return False


def _run_job(job: JobSpec, gpu_id: int) -> dict[str, Any]:
    if _job_complete(job):
        if not job.command_record.is_file():
            record = {
                "schema": COMMAND_SCHEMA,
                **job.manifest(),
                "status": "skipped_existing",
                "gpuId": None,
                "testRead": False,
            }
            _write_json(job.command_record, record)
        return {"job": job.name, "status": "skipped_existing", "gpuId": None, "testRead": False}

    job.log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    environment["PYTHONUNBUFFERED"] = "1"
    record = {
        "schema": COMMAND_SCHEMA,
        **job.manifest(),
        "status": "running",
        "gpuId": int(gpu_id),
        "cwd": str(ROOT),
        "environment": {"CUDA_VISIBLE_DEVICES": str(int(gpu_id)), "PYTHONUNBUFFERED": "1"},
        "stdout": str((job.log_dir / f"{job.step}.stdout.log").resolve()),
        "stderr": str((job.log_dir / f"{job.step}.stderr.log").resolve()),
        "startedAtUnix": time.time(),
        "testRead": False,
    }
    _write_json(job.command_record, record)
    started = time.perf_counter()
    with (job.log_dir / f"{job.step}.stdout.log").open("w", encoding="utf-8") as stdout, (
        job.log_dir / f"{job.step}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            list(job.command),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - started
    record.update(
        {
            "status": "completed" if result.returncode == 0 else "failed",
            "returnCode": int(result.returncode),
            "elapsedSeconds": float(elapsed),
            "finishedAtUnix": time.time(),
            "testRead": False,
        }
    )
    _write_json(job.command_record, record)
    result_row = {
        "job": job.name,
        "status": record["status"],
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "gpuId": int(gpu_id),
        "testRead": False,
    }
    if result.returncode != 0:
        raise RuntimeError(f"exact recalibration job failed: {job.name}; see {job.log_dir}")
    return result_row


def run_jobs(jobs: Sequence[JobSpec], gpu_ids: Sequence[int]) -> list[dict[str, Any]]:
    if not jobs:
        return []
    slots = [int(value) for value in gpu_ids]
    if not slots:
        raise ValueError("at least one GPU slot is required")
    available: Queue[int] = Queue()
    for gpu_id in slots:
        # Deliberately keep repeated IDs as independent slots.  This permits
        # controlled same-GPU concurrency when the device has spare capacity.
        available.put(gpu_id)

    def execute(job: JobSpec) -> dict[str, Any]:
        gpu_id = available.get()
        try:
            return _run_job(job, gpu_id)
        finally:
            available.put(gpu_id)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(jobs), len(slots))) as executor:
        futures = {executor.submit(execute, job): job.name for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    return results


def _run_phase(
    jobs: Sequence[JobSpec],
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    pending = [job for job in jobs if not _job_complete(job)]
    completed = [
        {"job": job.name, "status": "skipped_existing", "gpuId": None, "testRead": False}
        for job in jobs
        if job not in pending
    ]
    return completed + run_jobs(pending, gpu_ids)


def _run_bootstrap_ordered(
    jobs: Sequence[JobSpec],
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    by_scene: dict[str, list[JobSpec]] = {scene_key: [] for scene_key in SCENES}
    for job in jobs:
        by_scene[job.member.scene_key].append(job)
    results: list[dict[str, Any]] = []
    first_jobs = [scene_jobs[0] for scene_jobs in by_scene.values() if scene_jobs]
    results.extend(_run_phase(first_jobs, gpu_ids))
    remaining = [job for scene_jobs in by_scene.values() for job in scene_jobs[1:]]
    results.extend(_run_phase(remaining, gpu_ids))
    return results


def run(
    data_root: str | Path = ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    *,
    gpu_ids: Sequence[int] = (0, 1, 2),
    device: str = "cuda",
) -> dict[str, Any]:
    preflight_payload = preflight(data_root, output_root)
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "preflight.json", preflight_payload)
    plan = build_plan(data_root, output, device=device)
    _write_json(output / "plan.json", plan)
    phases = _job_specs(registered_members(data_root), output, device=device)
    phase_results: dict[str, list[dict[str, Any]]] = {}
    phase_results["score_calibration"] = _run_phase(phases["score_calibration"], gpu_ids)
    phase_results["calibrate"] = _run_bootstrap_ordered(phases["calibrate"], gpu_ids)
    phase_results["score_validation"] = _run_phase(phases["score_validation"], gpu_ids)
    phase_results["evaluate_validation"] = _run_bootstrap_ordered(
        phases["evaluate_validation"], gpu_ids
    )
    payload = {
        "schema": RUN_SCHEMA,
        "runnerSchema": RUNNER_SCHEMA,
        "experiment": "pvs_v4_exact_recalibration_existing_full_v1",
        "outputRoot": str(output),
        "gpuSlots": [int(value) for value in gpu_ids],
        "device": device,
        "phaseResults": phase_results,
        "testRead": False,
    }
    _write_json(output / "run_manifest.json", payload)
    return payload


def _metric(metrics: Mapping[str, Any], name: str) -> float:
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError(f"validation metric {name} is not finite")
    return value


def validation_safe(calibration: Mapping[str, Any], validation: Mapping[str, Any]) -> bool:
    _assert_test_free_payload(calibration, "exact calibration")
    _assert_test_free_payload(validation, "frozen validation")
    if calibration.get("schema") != CALIBRATION_SCHEMA or calibration.get("status") != "safe":
        return False
    if validation.get("schema") != VALIDATION_SCHEMA or validation.get("split") != "validation":
        return False
    metrics = validation.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    return (
        _metric(metrics, "aggregateWeightedRecall") > WEIGHTED_RECALL_FLOOR
        and _metric(metrics, "aggregateWeightedRecallLowerConfidenceBound") > WEIGHTED_RECALL_FLOOR
    )


def validation_selection_key(metrics: Mapping[str, Any], seed: int) -> tuple[float, ...]:
    return (
        _metric(metrics, "agg_useful_cull"),
        _metric(metrics, "agg_balanced_accuracy"),
        _metric(metrics, "candidate_normalized_occlusion_recall"),
        _metric(metrics, "agg_specificity"),
        _metric(metrics, "agg_precision"),
        -_metric(metrics, "avg_pred_count"),
        -float(seed),
    )


def summarize(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    data_root: str | Path = ROOT,
) -> dict[str, Any]:
    output = Path(output_root).expanduser().resolve()
    members = registered_members(data_root)
    _output_root_is_safe(output, members)
    safe_rows: list[dict[str, Any]] = []
    for member in members:
        root = _member_root(output, member)
        calibration_path = root / "exact_calibration.json"
        validation_path = root / "validation_frozen.json"
        if not calibration_path.is_file() or not validation_path.is_file():
            continue
        calibration = _read_json(calibration_path)
        validation = _read_json(validation_path)
        _assert_test_free_payload(calibration, str(calibration_path))
        _assert_test_free_payload(validation, str(validation_path))
        if str(calibration.get("checkpoint", "")).strip() != str(member.checkpoint.resolve()):
            raise ValueError(f"calibration checkpoint mismatch: {calibration_path}")
        if str(validation.get("checkpoint", "")).strip() != str(member.checkpoint.resolve()):
            raise ValueError(f"validation checkpoint mismatch: {validation_path}")
        if str(calibration.get("scene", "")).strip() != member.scene_name:
            raise ValueError(f"calibration scene mismatch: {calibration_path}")
        if str(validation.get("scene", "")).strip() != member.scene_name:
            raise ValueError(f"validation scene mismatch: {validation_path}")
        if not validation_safe(calibration, validation):
            continue
        metrics = validation.get("metrics")
        assert isinstance(metrics, Mapping)
        safe_rows.append(
            {
                "member": member.output_name,
                "scene": member.scene_key,
                "sceneName": member.scene_name,
                "seed": int(member.seed),
                "checkpoint": str(member.checkpoint.resolve()),
                "calibration": str(calibration_path.resolve()),
                "validation": str(validation_path.resolve()),
                "threshold": _metric(metrics, "threshold"),
                "aggregateWeightedRecall": _metric(metrics, "aggregateWeightedRecall"),
                "aggregateWeightedRecallLowerConfidenceBound": _metric(
                    metrics, "aggregateWeightedRecallLowerConfidenceBound"
                ),
                "aggregate": {
                    "usefulCull": _metric(metrics, "agg_useful_cull"),
                    "balancedAccuracy": _metric(metrics, "agg_balanced_accuracy"),
                    "candidateNormalizedOcclusionRecall": _metric(
                        metrics, "candidate_normalized_occlusion_recall"
                    ),
                    "occlusionRecall": _metric(metrics, "agg_specificity"),
                    "precision": _metric(metrics, "agg_precision"),
                    "avgPredCount": _metric(metrics, "avg_pred_count"),
                },
                "testRead": False,
            }
        )
    def row_key(row: Mapping[str, Any]) -> tuple[float, ...]:
        aggregate = row["aggregate"]
        return validation_selection_key(
            {
                "agg_useful_cull": aggregate["usefulCull"],
                "agg_balanced_accuracy": aggregate["balancedAccuracy"],
                "candidate_normalized_occlusion_recall": aggregate[
                    "candidateNormalizedOcclusionRecall"
                ],
                "agg_specificity": aggregate["occlusionRecall"],
                "agg_precision": aggregate["precision"],
                "avg_pred_count": aggregate["avgPredCount"],
            },
            int(row["seed"]),
        )

    scene_selections = {}
    for scene_key in SCENES:
        scene_rows = [row for row in safe_rows if row["scene"] == scene_key]
        scene_selections[scene_key] = {
            "validationSafeMemberCount": len(scene_rows),
            "members": scene_rows,
            "selected": max(scene_rows, key=row_key, default=None),
        }
    payload = {
        "schema": SUMMARY_SCHEMA,
        "runnerSchema": RUNNER_SCHEMA,
        "experiment": "pvs_v4_exact_recalibration_existing_full_v1",
        "selectionSplit": "validation",
        "selectionRule": (
            "validation-safe members only; useful cull, balanced accuracy, CNOR, "
            "occlusion recall, precision, then fewer predictions"
        ),
        "registeredMemberCount": len(members),
        "validationSafeMemberCount": len(safe_rows),
        "members": safe_rows,
        "sceneSelections": scene_selections,
        "testRead": False,
    }
    _write_json(output / "summary.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "run", "summarize"))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "plan":
        payload = build_plan(args.data_root, args.output_root, device=args.device)
        output = Path(args.output_root).expanduser().resolve()
        _write_json(output / "plan.json", payload)
    elif args.mode == "preflight":
        payload = preflight(args.data_root, args.output_root)
        output = Path(args.output_root).expanduser().resolve()
        _write_json(output / "preflight.json", payload)
    elif args.mode == "run":
        payload = run(
            args.data_root,
            args.output_root,
            gpu_ids=args.gpu_ids,
            device=args.device,
        )
    else:
        payload = summarize(args.output_root, args.data_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
