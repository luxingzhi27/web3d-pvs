#!/usr/bin/env python3
"""Run the Connected-SAH V4 Full training matrix.

This runner owns training only.  It does not evaluate a test split, finalize
selection, or export runtime assets.  Training is scheduled in seed rounds:
all requested scenes for one seed finish before the next seed round starts.
GPU arguments are slots, rather than a set of device IDs, so repeated values
such as ``--gpu-slots 1 1 2`` intentionally launch two jobs on device 1.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
from typing import Any, Sequence

try:
    from .standard_graphics_connected_sah_config import (
        EXPERIMENT,
        RELATION_K,
        ROOT,
        SCENES,
        SEEDS,
    )
except ImportError:  # direct ``python path/to/script.py`` invocation
    from standard_graphics_connected_sah_config import (  # type: ignore
        EXPERIMENT,
        RELATION_K,
        ROOT,
        SCENES,
        SEEDS,
    )


TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
LOG_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT / "full"

EPOCHS = 40
STEPS_PER_EPOCH = 900
POSES_PER_BATCH = 8
OBSERVATION_BATCH_SIZE = 8192
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 4
SNAPSHOT_EVERY = 4
CALIBRATION_BOOTSTRAP_REPLICATES = 10_000

POSE_SAMPLING = "ambiguity_balanced"
HARD_POSE_FRACTION = 0.35
HARD_POSE_QUANTILE = 0.65

SURVIVAL_RANK = 4
GUARD_WEIGHT = 0.30
TAIL_SEPARATION_WEIGHT = 0.30
TAIL_ZERO_FRACTION = 0.10
TAIL_RAMP_END_FRACTION = 0.30
GUARD_ZERO_FRACTION = 0.30
GUARD_MIDDLE_END_FRACTION = 0.50
GUARD_MIDDLE_SCALE = 0.25
POSITIVE_TAIL_MASS_FRACTION = 0.01
NEGATIVE_FRONTIER_FRACTION = 0.02
POSITIVE_IMPORTANCE_FLOOR = 0.25
POSITIVE_IMPORTANCE_POWER = 0.5

DATASET_SCHEMA = "pose-csr-explicit-four-way-split-v1"
RELATION_SCHEMA = "pvs-viewcell-train-observed-relation-csr-v3"
CALIBRATION_SUMMARY_SCHEMA = (
    "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4"
)
GEOMETRY_DIM = 96

SCENE_ORDER = ("sponza_128k", "viking_village_128k", "bigcity_128k")
MODES = ("plan", "preflight", "smoke", "train")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def scene_paths(scene: str) -> dict[str, Path]:
    if scene not in SCENES:
        raise ValueError(f"unknown Connected-SAH scene: {scene}")
    return {
        key: Path(value)
        for key, value in SCENES[scene].items()
        if isinstance(value, (str, Path))
    }


def member_dir(model_root: Path, scene: str, seed: int) -> Path:
    return model_root / scene / f"full_seed{seed}_e{EPOCHS}"


def smoke_member_dir(model_root: Path, scene: str, seed: int) -> Path:
    return model_root / "smoke" / scene / f"seed{seed}_e1"


def _expected_splits(scene: str) -> dict[str, int]:
    config = SCENES[scene]
    return {str(key): int(value) for key, value in config["split_counts"].items()}  # type: ignore[union-attr]


def _dataset_files(dataset_meta: dict[str, Any]) -> dict[str, str]:
    files = dataset_meta.get("files")
    if not isinstance(files, dict):
        raise ValueError("Pose CSR metadata has no files object")
    return {str(key): str(value) for key, value in files.items()}


def preflight(scene: str) -> dict[str, Any]:
    """Validate the exact Connected-SAH data contract without reading test rows."""
    paths = scene_paths(scene)
    required = (
        "assets",
        "runtime_meta",
        "glb_index",
        "geometry",
        "geometry_meta",
        "dataset",
        "relation",
    )
    missing = [str(paths[key]) for key in required if not paths[key].exists()]
    if missing:
        raise FileNotFoundError(f"{scene} Connected-SAH Full input(s) missing: {missing}")

    dataset_meta = read_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("schema") != DATASET_SCHEMA:
        raise ValueError(f"{scene} dataset schema is not {DATASET_SCHEMA}")
    split_counts = {
        str(key): int(value)
        for key, value in (dataset_meta.get("splitCounts") or {}).items()
    }
    expected_splits = _expected_splits(scene)
    if split_counts != expected_splits:
        raise ValueError(
            f"{scene} split counts changed: expected={expected_splits}, actual={split_counts}"
        )
    if float(dataset_meta.get("modelInputFovYDeg", 0.0)) != 66.0:
        raise ValueError(f"{scene} dataset must use model FOV 66")
    if float(dataset_meta.get("frontendRenderFovYDeg", 0.0)) != 60.0:
        raise ValueError(f"{scene} dataset must use render FOV 60")
    files = _dataset_files(dataset_meta)
    for field in (
        "poses",
        "queryCenterWorld",
        "candidateCameraWorld",
        "viewcellRadiusM",
        "candidateIds",
        "visibleIds",
        "visibleWeights",
    ):
        if field not in files or not (paths["dataset"] / files[field]).is_file():
            raise ValueError(f"{scene} dataset is missing {field}")

    runtime_meta = read_json(paths["runtime_meta"])
    instance_count = int(runtime_meta.get("instanceCount", -1))
    if instance_count <= 0:
        raise ValueError(f"{scene} runtime metadata has no positive instanceCount")
    if int(dataset_meta.get("numInstances", -1)) != instance_count:
        raise ValueError(f"{scene} dataset/runtime instance counts disagree")
    geometry_meta = read_json(paths["geometry_meta"])
    if geometry_meta.get("shape") != [instance_count, GEOMETRY_DIM]:
        raise ValueError(f"{scene} geometry metadata must have shape [{instance_count}, {GEOMETRY_DIM}]")
    if geometry_meta.get("dtype") != "float16":
        raise ValueError(f"{scene} geometry table must be float16")
    expected_geometry_bytes = instance_count * GEOMETRY_DIM * 2
    if paths["geometry"].stat().st_size != expected_geometry_bytes:
        raise ValueError(
            f"{scene} geometry table has wrong size: expected={expected_geometry_bytes}, "
            f"actual={paths['geometry'].stat().st_size}"
        )

    relation_meta = read_json(paths["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != RELATION_SCHEMA:
        raise ValueError(f"{scene} relation schema is not {RELATION_SCHEMA}")
    if relation_meta.get("trainOnly") is not True or relation_meta.get("splitNames") != ["train"]:
        raise ValueError(f"{scene} relation must be train-only")
    if int(relation_meta.get("sourceTopK", -1)) != RELATION_K:
        raise ValueError(f"{scene} relation sourceTopK must be {RELATION_K}")
    evidence_top_k = relation_meta.get("evidenceTopK") or {}
    if int(evidence_top_k.get("k", -1)) != RELATION_K:
        raise ValueError(f"{scene} relation evidenceTopK must be {RELATION_K}")
    if int(relation_meta.get("numInstances", -1)) != instance_count:
        raise ValueError(f"{scene} relation/runtime instance counts disagree")
    return {
        "schema": "standard-graphics-connected-sah-full-preflight-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "datasetSchema": dataset_meta["schema"],
        "relationSchema": relation_meta["schema"],
        "relationK": RELATION_K,
        "geometry": {"shape": [instance_count, GEOMETRY_DIM], "dtype": "float16"},
        "splitCounts": split_counts,
        "testRead": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }


def train_command(
    scene: str,
    output: Path,
    seed: int,
    *,
    smoke: bool = False,
) -> list[str]:
    """Build a from-scratch V4 command; it has no test/finalize/export action."""
    paths = scene_paths(scene)
    command = [
        sys.executable,
        "-u",
        str(TRAIN),
        "--dataset-dir",
        str(paths["dataset"]),
        "--relation-dir",
        str(paths["relation"]),
        "--runtime-meta",
        str(paths["runtime_meta"]),
        "--initial-geo-features",
        str(paths["geometry"]),
        "--glb-index",
        str(paths["glb_index"]),
        "--glb-root",
        str(paths["glb_root"]),
        "--output-dir",
        str(output),
        "--experiment-name",
        f"{EXPERIMENT}{'_smoke' if smoke else ''}_{scene}_seed{seed}",
        "--variant",
        "full_integrated_visibility_mainline",
        "--occlusion-representation",
        "survival",
        "--survival-rank",
        str(SURVIVAL_RANK),
        "--relation-source",
        "bounded_hierarchical",
        "--spectral-mode",
        "moment_envelope",
        "--instance-calibration-mode",
        "residual",
        "--loss-variant",
        "pose_balanced_rvl_contrastive",
        "--train-split",
        "train",
        "--calibration-split",
        "calibration",
        "--validation-split",
        "validation",
        "--epochs",
        str(1 if smoke else EPOCHS),
        "--steps-per-epoch",
        str(2 if smoke else STEPS_PER_EPOCH),
        "--poses-per-batch",
        str(POSES_PER_BATCH),
        "--pose-sampling",
        POSE_SAMPLING,
        "--hard-pose-fraction",
        str(HARD_POSE_FRACTION),
        "--hard-pose-quantile",
        str(HARD_POSE_QUANTILE),
        "--observation-batch-size",
        str(32768 if smoke else OBSERVATION_BATCH_SIZE),
        "--eval-every",
        str(1 if smoke else EVAL_EVERY),
        "--snapshot-every",
        str(1 if smoke else SNAPSHOT_EVERY),
        "--max-eval-poses",
        str(2 if smoke else 0),
        "--calibration-bootstrap-replicates",
        str(2 if smoke else CALIBRATION_BOOTSTRAP_REPLICATES),
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--learning-rate",
        str(LEARNING_RATE),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--survival-loss-weight",
        "0.25",
        "--relation-consistency-weight",
        "0.10",
        "--instance-calibration-regularization-weight",
        "0.02",
        "--instance-calibration-max-abs",
        "4.0",
        "--sparse-instance-penalty",
        "3.0",
        "--instance-calibration-warmup-fraction",
        "0.10",
        "--instance-calibration-ramp-fraction",
        "0.20",
        "--relation-gradient-cap",
        "0.25",
        "--integrated-rvl-recall-guard-weight",
        str(GUARD_WEIGHT),
        "--integrated-rvl-recall-target",
        "0.99",
        "--integrated-rvl-recall-temperature",
        "0.05",
        "--integrated-rvl-pose-cvar-fraction",
        "0.25",
        "--integrated-rvl-pose-cvar-weight",
        "0.25",
        "--integrated-separation-weight",
        str(TAIL_SEPARATION_WEIGHT),
        "--integrated-rvl-recall-guard-zero-fraction",
        str(GUARD_ZERO_FRACTION),
        "--integrated-rvl-recall-guard-middle-end-fraction",
        str(GUARD_MIDDLE_END_FRACTION),
        "--integrated-rvl-recall-guard-middle-scale",
        str(GUARD_MIDDLE_SCALE),
        "--integrated-tail-zero-fraction",
        str(TAIL_ZERO_FRACTION),
        "--integrated-tail-ramp-fraction",
        str(TAIL_RAMP_END_FRACTION),
        "--frontier-positive-mass-fraction",
        str(POSITIVE_TAIL_MASS_FRACTION),
        "--frontier-positive-count-cap",
        "64",
        "--frontier-negative-fraction",
        str(NEGATIVE_FRONTIER_FRACTION),
        "--frontier-negative-count-cap",
        "256",
        "--frontier-margin",
        "0.50",
        "--frontier-temperature",
        "0.25",
        "--frontier-positive-importance-floor",
        str(POSITIVE_IMPORTANCE_FLOOR),
        "--frontier-positive-importance-power",
        str(POSITIVE_IMPORTANCE_POWER),
    ]
    return command


def is_valid_completed_calibration_summary(member: Path) -> bool:
    """Return true only for a final, test-free train_pvs calibration summary."""
    summary_path = member / "calibration_ready_summary.json"
    if not summary_path.is_file():
        return False
    try:
        summary = read_json(summary_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return False
    return bool(
        summary.get("schema") == CALIBRATION_SUMMARY_SCHEMA
        and summary.get("testRead") is False
        and summary.get("status") in {"safe", "no_qualified_safety_workpoint"}
        and "calibration" in summary
    )


def pending_members(
    scenes: Sequence[str],
    seed: int,
    model_root: Path,
    *,
    smoke: bool = False,
) -> list[tuple[str, Path]]:
    """Return only members without a valid completed calibration summary."""
    result = []
    for scene in scenes:
        member = smoke_member_dir(model_root, scene, seed) if smoke else member_dir(model_root, scene, seed)
        if not is_valid_completed_calibration_summary(member):
            result.append((scene, member))
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _command_record(
    path: Path,
    *,
    scene: str,
    seed: int,
    gpu_id: int,
    command: list[str],
    stdout: Path,
    stderr: Path,
    return_code: int | None = None,
) -> None:
    record: dict[str, Any] = {
        "schema": "standard-graphics-connected-sah-full-command-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "seed": int(seed),
        "gpuId": int(gpu_id),
        "cudaVisibleDevices": str(gpu_id),
        "command": command,
        "cwd": str(ROOT),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "testRead": False,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if return_code is not None:
        record["returnCode"] = int(return_code)
    _write_json(path, record)


def run_job(
    scene: str,
    seed: int,
    gpu_id: int,
    command: list[str],
    *,
    log_root: Path,
) -> None:
    """Run one job in one declared GPU slot and persist all process output."""
    job_name = f"seed{seed}_{scene}_gpu{gpu_id}"
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = log_root / f"{job_name}.stdout.log"
    stderr = log_root / f"{job_name}.stderr.log"
    command_path = log_root / f"{job_name}.command.json"
    _command_record(
        command_path,
        scene=scene,
        seed=seed,
        gpu_id=gpu_id,
        command=command,
        stdout=stdout,
        stderr=stderr,
    )
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with stdout.open("w", encoding="utf-8") as stdout_stream, stderr.open(
        "w", encoding="utf-8"
    ) as stderr_stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
        )
    _command_record(
        command_path,
        scene=scene,
        seed=seed,
        gpu_id=gpu_id,
        command=command,
        stdout=stdout,
        stderr=stderr,
        return_code=int(result.returncode),
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_round(
    scenes: Sequence[str],
    seed: int,
    gpu_slots: Sequence[int],
    model_root: Path,
    log_root: Path,
    *,
    smoke: bool = False,
) -> list[str]:
    """Run one seed round, assigning one scene job to each slot at a time."""
    if not gpu_slots:
        raise ValueError("at least one GPU slot is required")
    pending = pending_members(scenes, seed, model_root, smoke=smoke)
    skipped = [scene for scene in scenes if scene not in {name for name, _ in pending}]
    for start in range(0, len(pending), len(gpu_slots)):
        batch = pending[start:start + len(gpu_slots)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(
                    run_job,
                    scene,
                    seed,
                    int(gpu_slots[index]),
                    train_command(scene, member, seed, smoke=smoke),
                    log_root=log_root / f"seed{seed}",
                ): scene
                for index, (scene, member) in enumerate(batch)
            }
            errors = []
            for future in as_completed(futures):
                scene = futures[future]
                try:
                    future.result()
                except Exception as error:  # wait for the complete round before reporting
                    errors.append(f"{scene}: {error}")
            if errors:
                raise RuntimeError("Connected-SAH Full seed round failed: " + "; ".join(errors))
    return skipped


def plan(scenes: Sequence[str], model_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Return a side-effect-free plan for all seed rounds."""
    rounds = []
    for seed in SEEDS:
        rounds.append({
            "seed": int(seed),
            "sceneOrder": list(scenes),
            "jobs": [
                {
                    "scene": scene,
                    "member": str(smoke_member_dir(model_root, scene, seed) if smoke else member_dir(model_root, scene, seed)),
                    "command": train_command(
                        scene,
                        smoke_member_dir(model_root, scene, seed) if smoke else member_dir(model_root, scene, seed),
                        seed,
                        smoke=smoke,
                    ),
                }
                for scene in scenes
            ],
        })
    return {
        "schema": "standard-graphics-connected-sah-full-plan-v1",
        "experiment": EXPERIMENT,
        "sceneFirst": True,
        "sceneOrder": list(scenes),
        "seeds": list(SEEDS),
        "epochs": 1 if smoke else EPOCHS,
        "stepsPerEpoch": 2 if smoke else STEPS_PER_EPOCH,
        "testRead": False,
        "rounds": rounds,
    }


def _scene_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise ValueError("at least one scene is required")
    unknown = sorted(set(names) - set(SCENES))
    if unknown:
        raise ValueError(f"unknown scene(s) {unknown}; choose from {sorted(SCENES)}")
    return [scene for scene in SCENE_ORDER if scene in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--scenes", default=",".join(SCENE_ORDER))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    args = parser.parse_args()
    if any(int(value) < 0 for value in args.gpu_slots):
        parser.error("GPU slot IDs must be non-negative")
    scenes = _scene_names(args.scenes)
    model_root = args.model_root.resolve()
    log_root = args.log_root.resolve()

    if args.mode == "plan":
        print(json.dumps(plan(scenes, model_root), ensure_ascii=False, indent=2))
        return
    if args.mode == "preflight":
        reports = [preflight(scene) for scene in scenes]
        print(json.dumps({"experiment": EXPERIMENT, "reports": reports, "testRead": False}, ensure_ascii=False, indent=2))
        return
    if args.mode == "smoke":
        for scene in scenes:
            preflight(scene)
        for seed in (SEEDS[0],):
            run_round(scenes, seed, args.gpu_slots, model_root, log_root, smoke=True)
        return
    for scene in scenes:
        preflight(scene)
    for seed in SEEDS:
        skipped = run_round(scenes, seed, args.gpu_slots, model_root, log_root)
        if skipped:
            print(json.dumps({"seed": seed, "skippedCompleted": skipped}), flush=True)


if __name__ == "__main__":
    main()
