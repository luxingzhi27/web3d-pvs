#!/usr/bin/env python3
"""Run the registered AABB-plus-ray MLP paper baseline.

The runner owns the scan, calibration, validation comparison, three-seed
confirmation, and one-shot frozen test replay.  Training and selection never
open the test split, and every subprocess writes durable stdout/stderr logs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "neural_instance_culling" / "benchmark"
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
for path in (BENCHMARK_DIR, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aabb_ray_baseline_config import (  # noqa: E402
    CALIBRATION_BOOTSTRAP_REPLICATES,
    DATA_ROOT,
    EXPERIMENT,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
    LOSS_CONFIG,
    SCAN_EPOCHS,
    SCAN_LEARNING_RATES,
    SCAN_STEPS_PER_EPOCH,
    SCENES,
    WEIGHTED_RECALL_TARGET,
    scene_paths,
)
from evaluate_unified_pvs_metrics import (  # noqa: E402
    evaluate_runner,
    load_glb_byte_costs,
    threshold_grid,
)
from model_runners import load_learned_aabb_ray_runner, select_device  # noqa: E402
from pvs_contract import validate_native_candidate_contract  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402


EXPECTED_SPLITS = {
    "hkust_v3": {"train": 5926, "validation": 730, "calibration": 659, "test": 684, "guard": 0},
    "ifcbench_fantasy_metropolis": {"train": 19647, "validation": 2712, "calibration": 2183, "test": 2710, "guard": 0},
    "sponza_128k": {"train": 1752, "validation": 240, "calibration": 192, "test": 240, "guard": 0},
    "bigcity_128k": {"train": 5832, "validation": 804, "calibration": 648, "test": 804, "guard": 0},
    "viking_village_128k": {"train": 1104, "validation": 156, "calibration": 120, "test": 156, "guard": 0},
}
TRAIN_SCRIPT = BENCHMARK_DIR / "train_aabb_ray_baseline.py"
EVALUATION_SCHEMA = "pvs-aabb-ray-mlp-evaluation-v1"
CALIBRATION_SCHEMA = "pvs-aabb-ray-mlp-calibration-v1"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def preflight(data_root: str | Path, scene: str) -> dict[str, Any]:
    paths = scene_paths(data_root, scene)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if not TRAIN_SCRIPT.is_file():
        missing.append(str(TRAIN_SCRIPT))
    if missing:
        raise FileNotFoundError(f"missing AABB-ray baseline inputs: {missing}")
    dataset_meta = _load_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError("AABB-ray formal baseline requires the explicit four-way split dataset")
    split_counts = {str(key): int(value) for key, value in (dataset_meta.get("splitCounts") or {}).items()}
    expected = EXPECTED_SPLITS.get(scene)
    if expected is not None and split_counts != expected:
        raise ValueError(f"{scene} split counts changed: expected={expected}, actual={split_counts}")
    files = dataset_meta.get("files") or {}
    for field in ("candidateIds", "visibleIds", "visibleWeights"):
        if field not in files or not (paths["dataset"] / str(files[field])).is_file():
            raise ValueError(f"AABB-ray dataset is missing {field}")
    if not (paths["dataset"] / "mvp.bin").is_file():
        raise ValueError("AABB-ray formal baseline requires mvp.bin")
    if dataset_meta.get("candidateVisibleUnionAllowed") is True:
        raise ValueError("formal AABB-ray baseline cannot use a candidate-visible union")

    world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(paths["runtimeMeta"])
    dataset = PoseCSRDataset(paths["dataset"], num_instances=int(world_aabbs.shape[0]))
    # Check only data used by training and selection.  Test labels remain closed
    # until the explicit frozen test replay.
    train_indices = np.concatenate(
        [dataset.split(name).pose_indices for name in ("train", "calibration", "validation")]
    )
    contract = validate_native_candidate_contract(
        dataset,
        train_indices.tolist(),
        num_instances=int(world_aabbs.shape[0]),
    )
    return {
        "schema": "pvs-aabb-ray-formal-preflight-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "splitCounts": split_counts,
        "lossConfig": dict(LOSS_CONFIG),
        "scanLearningRates": list(SCAN_LEARNING_RATES),
        "scanEpochs": SCAN_EPOCHS,
        "scanStepsPerEpoch": SCAN_STEPS_PER_EPOCH,
        "formalSeeds": list(FORMAL_SEEDS),
        "formalEpochs": FORMAL_EPOCHS,
        "formalStepsPerEpoch": FORMAL_STEPS_PER_EPOCH,
        "trainCalibrationValidationCandidateContract": contract,
        "testRead": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }


def _format_learning_rate(value: float) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def build_train_command(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    scene: str,
    learning_rate: float,
    seed: int,
    epochs: int,
    steps_per_epoch: int,
) -> list[str]:
    paths = scene_paths(data_root, scene)
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset-dir", str(paths["dataset"]),
        "--runtime-meta", str(paths["runtimeMeta"]),
        "--output-dir", str(Path(output_dir).resolve()),
        "--epochs", str(int(epochs)),
        "--steps-per-epoch", str(int(steps_per_epoch)),
        "--batch-size", "8192",
        "--positive-samples-per-pose", "16",
        "--negative-samples-per-pose", "128",
        "--learning-rate", str(float(learning_rate)),
        "--weight-decay", "0.00001",
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--loss-variant", str(LOSS_CONFIG["lossVariant"]),
        "--log-every", "100",
        "--integrated-rvl-recall-guard-weight", str(float(LOSS_CONFIG["rvlRecallGuardWeight"])),
        "--integrated-rvl-recall-target", str(float(LOSS_CONFIG["rvlRecallTarget"])),
        "--integrated-rvl-recall-temperature", str(float(LOSS_CONFIG["rvlRecallTemperature"])),
        "--integrated-rvl-pose-cvar-fraction", str(float(LOSS_CONFIG["rvlPoseCvarFraction"])),
        "--integrated-rvl-pose-cvar-weight", str(float(LOSS_CONFIG["rvlPoseCvarWeight"])),
        "--integrated-separation-weight", str(float(LOSS_CONFIG["sharedTailSeparationWeight"])),
        "--integrated-tail-ramp-fraction", str(float(LOSS_CONFIG["tailRampFraction"])),
        "--frontier-positive-mass-fraction", str(float(LOSS_CONFIG["tailPositiveMassFraction"])),
        "--frontier-positive-count-cap", str(int(LOSS_CONFIG["tailPositiveCountCap"])),
        "--frontier-negative-fraction", str(float(LOSS_CONFIG["tailNegativeFraction"])),
        "--frontier-negative-count-cap", str(int(LOSS_CONFIG["tailNegativeCountCap"])),
        "--frontier-margin", str(float(LOSS_CONFIG["tailMargin"])),
        "--frontier-temperature", str(float(LOSS_CONFIG["tailTemperature"])),
        "--frontier-positive-importance-floor", str(float(LOSS_CONFIG["positiveImportanceFloor"])),
        "--frontier-positive-importance-power", str(float(LOSS_CONFIG["positiveImportancePower"])),
    ]
    return command


def formal_test_allowed(
    *,
    split: str,
    checkpoint: Path | None,
    calibration: Path | None,
    threshold: float | None,
) -> bool:
    if str(split) != "test":
        return False
    if checkpoint is None or calibration is None or threshold is None:
        return False
    return bool(np.isfinite(float(threshold)) and 0.0 <= float(threshold) <= 1.0)


def _member_name(stage: str, scene: str, learning_rate: float, seed: int, epochs: int, steps: int) -> str:
    return f"{stage}_{scene}_lr{_format_learning_rate(learning_rate)}_seed{int(seed)}_{int(epochs)}x{int(steps)}"


def _checkpoint_complete(output_dir: Path, *, learning_rate: float, seed: int, epochs: int, steps: int) -> bool:
    summary_path = output_dir / "training_summary.json"
    if not (output_dir / "best.pt").is_file() or not summary_path.is_file():
        return False
    try:
        summary = _load_json(summary_path)
        return bool(
            summary.get("schema") == "neuralstreamweb3d-learned-aabb-ray-summary-v1"
            and int(summary.get("epochs", -1)) == int(epochs)
            and int(summary.get("stepsPerEpoch", -1)) == int(steps)
            and int(summary.get("seed", -1)) == int(seed)
            and np.isclose(float(summary.get("learningRate", np.nan)), float(learning_rate), rtol=0.0, atol=1e-12)
            and summary.get("lossVariant") == LOSS_CONFIG["lossVariant"]
            and summary.get("testRead") is False
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _run_logged(command: Sequence[str], gpu: int, stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    return {
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.time() - started),
        "gpu": int(gpu),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
    }


def _run_jobs(jobs: Sequence[tuple[str, Sequence[str]]], gpu_ids: Sequence[int], log_dir: Path) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")

    available_gpus: Queue[int] = Queue()
    for gpu in gpu_ids:
        available_gpus.put(int(gpu))

    def run(name: str, command: Sequence[str]) -> dict[str, Any]:
        gpu = available_gpus.get()
        try:
            result = _run_logged(
                command,
                gpu,
                log_dir / f"{name}.stdout.log",
                log_dir / f"{name}.stderr.log",
            )
            result["name"] = name
            return result
        finally:
            available_gpus.put(gpu)

    results: list[dict[str, Any]] = []
    log_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=min(len(gpu_ids), len(jobs))) as executor:
        futures = {
            executor.submit(run, name, command): name
            for name, command in jobs
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["returnCode"] != 0:
                raise RuntimeError(f"AABB-ray training job failed: {result['name']}; see {result['stderr']}")
    return sorted(results, key=lambda item: str(item["name"]))


def _build_runner(scene: str, checkpoint: Path, data_root: Path, device: torch.device):
    paths = scene_paths(data_root, scene)
    return load_learned_aabb_ray_runner(
        f"aabb_ray_mlp_{scene}",
        checkpoint,
        paths["runtimeMeta"],
        None,
        0.5,
        device,
    )


def _evaluate_raw(
    data_root: Path,
    scene: str,
    checkpoint: Path,
    split_name: str,
    thresholds: np.ndarray,
    *,
    bootstrap_replicates: int,
    sidecar_dir: Path | None = None,
    calibration: Path | None = None,
) -> dict[str, Any]:
    paths = scene_paths(data_root, scene)
    world_aabbs, instance_to_glb, _runtime = load_runtime_meta(paths["runtimeMeta"])
    dataset = PoseCSRDataset(paths["dataset"], num_instances=int(world_aabbs.shape[0]))
    split = dataset.split(split_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = _build_runner(scene, checkpoint, data_root, device)
    num_glbs = int(np.max(instance_to_glb)) + 1 if instance_to_glb.size else 0
    glb_bytes = load_glb_byte_costs(paths["glbIndex"], paths["glbRoot"], num_glbs, instance_to_glb)
    writer = None
    if sidecar_dir is not None:
        from score_sidecar import ScoreSidecarWriter

        writer = ScoreSidecarWriter(
            sidecar_dir,
            split=split_name,
            threshold=float(thresholds[0]) if thresholds.size == 1 else None,
            checkpoint=checkpoint,
            calibration=calibration,
        )
    result = evaluate_runner(
        runner,
        split,
        thresholds.astype(np.float32, copy=False),
        (),
        glb_bytes,
        poses_per_batch=4,
        max_eval_poses=0,
        max_candidates_per_pose=0,
        seed=20260909,
        target_recall=0.95,
        target_weighted_recall=WEIGHTED_RECALL_TARGET,
        target_utility_recall=0.98,
        sample_with_replacement=False,
        allow_candidate_visible_union=False,
        bootstrap_replicates=int(bootstrap_replicates),
        sidecar_writer=writer,
        collect_score_stats=True,
    )
    if writer is not None:
        result["scoreSidecar"] = str(writer.close())
    return result


def _calibration_safe(row: Mapping[str, Any]) -> bool:
    lower = row.get("aggregateWeightedRecallLowerConfidenceBound", row.get("aggregate_weighted_recall_lower_confidence_bound"))
    recall = row.get("aggregateWeightedRecall", row.get("aggregate_weighted_recall"))
    return (
        lower is not None
        and recall is not None
        and float(recall) > WEIGHTED_RECALL_TARGET
        and float(lower) > WEIGHTED_RECALL_TARGET
    )


def _aggregate_cull_rates(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    useful = row.get("agg_useful_cull")
    bad = row.get("agg_bad_cull")
    if useful is not None and bad is not None:
        return float(useful), float(bad)
    counts = [row.get(key) for key in ("tp", "fp", "fn", "tn")]
    if any(value is None for value in counts):
        return None, None
    total = sum(float(value) for value in counts)
    if total <= 0:
        return 0.0, 0.0
    return float(row["tn"]) / total, float(row["fn"]) / total


def _row_key(row: Mapping[str, Any], *, safe_first: bool = True) -> tuple[float, ...]:
    useful_cull, _ = _aggregate_cull_rates(row)
    recall_lcb = float(row.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0)
    if not safe_first:
        return (
            recall_lcb,
            float(useful_cull if useful_cull is not None else -1.0),
            float(row.get("agg_balanced_accuracy", 0.0)),
            float(row.get("agg_specificity", 0.0)),
            float(row.get("agg_precision", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        )
    return (
        float(_calibration_safe(row)),
        float(useful_cull if useful_cull is not None else -1.0),
        float(row.get("agg_balanced_accuracy", 0.0)),
        float(row.get("agg_specificity", 0.0)),
        float(row.get("agg_precision", 0.0)),
        recall_lcb,
        -float(row.get("avg_pred_count", 0.0)),
    )


def calibrate_checkpoint(data_root: Path, scene: str, checkpoint: Path, output: Path) -> dict[str, Any]:
    result = _evaluate_raw(
        data_root,
        scene,
        checkpoint,
        "calibration",
        threshold_grid(),
        bootstrap_replicates=CALIBRATION_BOOTSTRAP_REPLICATES,
    )
    rows = result["thresholdRows"]
    safe = [row for row in rows if _calibration_safe(row)]
    diagnostic = max(rows, key=lambda row: _row_key(row, safe_first=False)) if rows else None
    selected = max(safe, key=lambda row: _row_key(row)) if safe else diagnostic
    if selected is None:
        raise ValueError("AABB-ray calibration produced no threshold rows")
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "scene": scene,
        "checkpoint": str(checkpoint.resolve()),
        "status": "safe" if safe else "no_qualified_safety_workpoint",
        "selectionSplit": "calibration",
        "thresholdSource": "calibration_only_frozen_before_validation_and_test",
        "bootstrapReplicates": CALIBRATION_BOOTSTRAP_REPLICATES,
        "bestSafe": {"selection": selected} if safe else None,
        "bestDiagnostic": {"selection": diagnostic},
        "thresholdRows": rows,
        "testEvaluationCount": 0,
        "testRead": False,
    }
    _write_json(output, payload)
    return payload


def _metric_payload(raw: Mapping[str, Any], *, threshold: float) -> dict[str, Any]:
    useful_cull, bad_cull = _aggregate_cull_rates(raw)
    aggregate = {
        "precision": raw.get("agg_precision"),
        "recall": raw.get("agg_recall"),
        "weightedRecall": raw.get("aggregateWeightedRecall", raw.get("agg_weighted_recall")),
        "weightedRecallLowerConfidenceBound": raw.get("aggregateWeightedRecallLowerConfidenceBound"),
        "f1": raw.get("agg_f1"),
        "accuracy": raw.get("agg_accuracy"),
        "balancedAccuracy": raw.get("agg_balanced_accuracy"),
        "specificity": raw.get("agg_specificity"),
        "usefulCull": useful_cull,
        "badCull": bad_cull,
        "avgCandidateCount": raw.get("avg_candidate_count"),
        "avgGtCount": raw.get("avg_gt_count"),
        "avgPredCount": raw.get("avg_pred_count"),
        "predictedGlbCount": raw.get("avg_pred_glb_count"),
        "predictedGlbBytes": raw.get("avg_pred_glb_bytes"),
        "candidateGlbCount": raw.get("avg_candidate_glb_count"),
        "candidateGlbBytes": raw.get("avg_candidate_glb_bytes"),
        "glbByteReduction": raw.get("pose_candidate_byte_reduction_ratio"),
        "averagePrecision": raw.get("aggregateAveragePrecision"),
        "positiveRate": raw.get("aggregatePositiveRate"),
        "apLift": raw.get("aggregateApLift"),
    }
    pose = {
        "precision": raw.get("pose_precision"),
        "recall": raw.get("pose_recall"),
        "weightedRecall": raw.get("pose_weighted_recall"),
        "f1": raw.get("pose_f1"),
        "jaccard": raw.get("pose_jaccard"),
        "accuracy": raw.get("pose_accuracy"),
        "balancedAccuracy": raw.get("pose_balanced_accuracy"),
        "specificity": raw.get("pose_specificity"),
        "usefulCull": raw.get("pose_useful_cull_candidate_ratio"),
        "badCull": raw.get("pose_bad_cull_candidate_ratio"),
        "averagePrecision": raw.get("poseMacroAveragePrecision"),
        "positiveRate": raw.get("poseMacroPositiveRate"),
        "apLift": raw.get("poseMacroApLift"),
    }
    return {"threshold": float(threshold), "aggregate": aggregate, "poseMacro": pose}


def _validation_key(item: Mapping[str, Any]) -> tuple[float, ...]:
    aggregate = item["validation"]["aggregate"]
    recall = float(aggregate.get("weightedRecall") or -1.0)
    lower = float(aggregate.get("weightedRecallLowerConfidenceBound") or -1.0)
    safe = recall > WEIGHTED_RECALL_TARGET and lower > WEIGHTED_RECALL_TARGET
    useful = float(aggregate.get("usefulCull") or 0.0)
    balanced = float(aggregate.get("balancedAccuracy") or 0.0)
    specificity = float(aggregate.get("specificity") or 0.0)
    precision = float(aggregate.get("precision") or 0.0)
    predicted = -float(aggregate.get("avgPredCount") or 0.0)
    if safe:
        return 1.0, useful, balanced, specificity, precision, lower, predicted
    return 0.0, lower, useful, balanced, specificity, precision, predicted


def evaluate_checkpoint(
    data_root: Path,
    scene: str,
    checkpoint: Path,
    calibration: Path,
    threshold: float,
    split_name: str,
    output: Path,
    sidecar_dir: Path | None = None,
) -> dict[str, Any]:
    if split_name == "test" and not formal_test_allowed(
        split=split_name, checkpoint=checkpoint, calibration=calibration, threshold=threshold
    ):
        raise ValueError("formal AABB-ray test requires explicit checkpoint, calibration, and threshold")
    if split_name == "test":
        calibration_payload = _load_json(calibration)
        if calibration_payload.get("testRead") is not False:
            raise ValueError("formal AABB-ray test calibration is not test-free")
        test_evaluation_count = calibration_payload.get("testEvaluationCount", 0)
        if test_evaluation_count is None:
            test_evaluation_count = 0
        if int(test_evaluation_count) != 0:
            raise ValueError("formal AABB-ray calibration already read test")
        selected_threshold = _selected_threshold(calibration_payload, require_safe=True)
        if not np.isclose(float(threshold), selected_threshold, rtol=0.0, atol=1e-7):
            raise ValueError("formal AABB-ray test threshold disagrees with calibration")
        declared_checkpoint = calibration_payload.get("checkpoint")
        if declared_checkpoint is not None and Path(str(declared_checkpoint)).resolve() != checkpoint.resolve():
            raise ValueError("formal AABB-ray calibration checkpoint disagrees")
    raw = _evaluate_raw(
        data_root,
        scene,
        checkpoint,
        split_name,
        np.asarray([float(threshold)], dtype=np.float32),
        bootstrap_replicates=CALIBRATION_BOOTSTRAP_REPLICATES if split_name in {"calibration", "validation", "test"} else 0,
        sidecar_dir=sidecar_dir,
        calibration=calibration,
    )
    row = raw["thresholdRows"][0]
    payload = {
        "schema": EVALUATION_SCHEMA,
        "scene": scene,
        "method": "aabb_ray_mlp",
        "split": split_name,
        "testRead": split_name == "test",
        "testEvaluationCount": 1 if split_name == "test" else 0,
        "checkpoint": str(checkpoint.resolve()),
        "calibration": str(calibration.resolve()),
        "threshold": float(threshold),
        "thresholdSource": "checkpoint_calibration_frozen_before_test" if split_name == "test" else "calibration_only",
        "metrics": _metric_payload(row, threshold=threshold)["aggregate"],
        "aggregate": _metric_payload(row, threshold=threshold)["aggregate"],
        "poseMacro": _metric_payload(row, threshold=threshold)["poseMacro"],
        "zeroGtPoseCount": int(raw.get("zeroGtPoseCount", 0)),
        "evaluatedPoseCount": int(row.get("eval_pose_count", 0)),
        "scoreSidecar": raw.get("scoreSidecar"),
        "runnerInfo": raw.get("runnerInfo", {}),
    }
    _write_json(output, payload)
    return payload


def _selected_threshold(
    calibration: Mapping[str, Any], *, require_safe: bool = False
) -> float:
    keys = ("bestSafe",) if require_safe else ("bestSafe", "bestDiagnostic")
    if require_safe and calibration.get("status") != "safe":
        raise ValueError("formal AABB-ray test requires a safe calibration workpoint")
    for key in keys:
        value = calibration.get(key)
        if isinstance(value, Mapping):
            selection = value.get("selection", value)
            if isinstance(selection, Mapping) and selection.get("threshold") is not None:
                return float(selection["threshold"])
    raise ValueError("calibration summary has no frozen threshold")


def _scan_member_dir(model_root: Path, scene: str, learning_rate: float) -> Path:
    return model_root / _member_name("scan", scene, learning_rate, FORMAL_SEEDS[0], SCAN_EPOCHS, SCAN_STEPS_PER_EPOCH)


def _formal_member_dir(model_root: Path, scene: str, learning_rate: float, seed: int) -> Path:
    return model_root / _member_name("formal", scene, learning_rate, seed, FORMAL_EPOCHS, FORMAL_STEPS_PER_EPOCH)


def run_scene(
    data_root: Path,
    scene: str,
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
    *,
    run_scan: bool = True,
    run_formal: bool = True,
    run_test: bool = True,
) -> dict[str, Any]:
    contract = preflight(data_root, scene)
    _write_json(benchmark_root / "preflight" / f"{scene}.json", contract)
    if run_scan:
        scan_jobs: list[tuple[str, Sequence[str]]] = []
        for learning_rate in SCAN_LEARNING_RATES:
            member = _scan_member_dir(model_root, scene, learning_rate)
            if _checkpoint_complete(member, learning_rate=learning_rate, seed=FORMAL_SEEDS[0], epochs=SCAN_EPOCHS, steps=SCAN_STEPS_PER_EPOCH):
                continue
            if member.exists() and any(member.iterdir()):
                raise RuntimeError(f"incomplete AABB-ray scan member exists; inspect it first: {member}")
            member.mkdir(parents=True, exist_ok=True)
            scan_jobs.append(
                (
                    member.name,
                    build_train_command(
                        data_root,
                        member,
                        scene=scene,
                        learning_rate=learning_rate,
                        seed=FORMAL_SEEDS[0],
                        epochs=SCAN_EPOCHS,
                        steps_per_epoch=SCAN_STEPS_PER_EPOCH,
                    ),
                )
            )
        _run_jobs(scan_jobs, gpu_ids, benchmark_root / "logs" / "scan" / scene)

        scan_rows: list[dict[str, Any]] = []
        for learning_rate in SCAN_LEARNING_RATES:
            member = _scan_member_dir(model_root, scene, learning_rate)
            if not _checkpoint_complete(member, learning_rate=learning_rate, seed=FORMAL_SEEDS[0], epochs=SCAN_EPOCHS, steps=SCAN_STEPS_PER_EPOCH):
                raise RuntimeError(f"scan member is incomplete: {member}")
            checkpoint = member / "best.pt"
            calibration_path = benchmark_root / "scan" / scene / f"lr{_format_learning_rate(learning_rate)}" / "calibration.json"
            calibration = calibrate_checkpoint(data_root, scene, checkpoint, calibration_path)
            threshold = _selected_threshold(calibration)
            validation = evaluate_checkpoint(
                data_root,
                scene,
                checkpoint,
                calibration_path,
                threshold,
                "validation",
                calibration_path.with_name("validation.json"),
            )
            scan_rows.append(
                {
                    "learningRate": float(learning_rate),
                    "member": str(member.resolve()),
                    "checkpoint": str(checkpoint.resolve()),
                    "calibration": str(calibration_path.resolve()),
                    "threshold": threshold,
                    "calibrationStatus": calibration["status"],
                    "validation": validation,
                    "testRead": False,
                }
            )
        selected_scan = max(scan_rows, key=_validation_key)
        selection = {
            "schema": "pvs-aabb-ray-scan-selection-v1",
            "scene": scene,
            "selectionSplit": "validation",
            "learningRate": selected_scan["learningRate"],
            "selectedMember": selected_scan["member"],
            "scanRows": scan_rows,
            "testRead": False,
        }
        _write_json(benchmark_root / "scan" / scene / "selection.json", selection)
    else:
        selection = _load_json(benchmark_root / "scan" / scene / "selection.json")
        if selection.get("selectionSplit") != "validation" or selection.get("testRead") is not False:
            raise ValueError("AABB-ray formal training requires a validation-only scan selection")
        scan_rows = selection.get("scanRows")
        if not isinstance(scan_rows, list) or not scan_rows:
            raise ValueError("AABB-ray scan selection has no rows")
        selected_scan = max(scan_rows, key=lambda item: float(item.get("learningRate", -1.0)) == float(selection["learningRate"]))

    if not run_formal:
        summary = {
            "schema": "pvs-aabb-ray-formal-summary-v1",
            "experiment": EXPERIMENT,
            "scene": scene,
            "selectedLearningRate": float(selection["learningRate"]),
            "selection": selection,
            "formalRows": [],
            "lossConfig": dict(LOSS_CONFIG),
            "scanMatrix": {"learningRates": list(SCAN_LEARNING_RATES), "epochs": SCAN_EPOCHS, "stepsPerEpoch": SCAN_STEPS_PER_EPOCH},
            "confirmationMatrix": {"seeds": list(FORMAL_SEEDS), "epochs": FORMAL_EPOCHS, "stepsPerEpoch": FORMAL_STEPS_PER_EPOCH},
            "testRead": False,
        }
        _write_json(benchmark_root / "formal" / scene / "summary.json", summary)
        return summary

    learning_rate = float(selected_scan["learningRate"])
    formal_jobs: list[tuple[str, Sequence[str]]] = []
    for seed in FORMAL_SEEDS:
        member = _formal_member_dir(model_root, scene, learning_rate, seed)
        if _checkpoint_complete(member, learning_rate=learning_rate, seed=seed, epochs=FORMAL_EPOCHS, steps=FORMAL_STEPS_PER_EPOCH):
            continue
        if member.exists() and any(member.iterdir()):
            raise RuntimeError(f"incomplete AABB-ray formal member exists; inspect it first: {member}")
        member.mkdir(parents=True, exist_ok=True)
        formal_jobs.append(
            (
                member.name,
                build_train_command(
                    data_root,
                    member,
                    scene=scene,
                    learning_rate=learning_rate,
                    seed=seed,
                    epochs=FORMAL_EPOCHS,
                    steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
                ),
            )
        )
    if run_formal:
        _run_jobs(formal_jobs, gpu_ids, benchmark_root / "logs" / "formal" / scene)

    formal_rows = []
    for seed in FORMAL_SEEDS:
        member = _formal_member_dir(model_root, scene, learning_rate, seed)
        if not _checkpoint_complete(member, learning_rate=learning_rate, seed=seed, epochs=FORMAL_EPOCHS, steps=FORMAL_STEPS_PER_EPOCH):
            raise RuntimeError(f"formal member is incomplete: {member}")
        checkpoint = member / "best.pt"
        calibration_path = benchmark_root / "formal" / scene / f"seed{seed}" / "calibration.json"
        calibration = calibrate_checkpoint(data_root, scene, checkpoint, calibration_path)
        threshold = _selected_threshold(calibration)
        validation = evaluate_checkpoint(
            data_root,
            scene,
            checkpoint,
            calibration_path,
            threshold,
            "validation",
            calibration_path.with_name("validation.json"),
        )
        test = None
        if run_test:
            test_output = benchmark_root / "test_metrics" / scene / f"aabb_ray_mlp_seed{seed}.json"
            test = evaluate_checkpoint(
                data_root,
                scene,
                checkpoint,
                calibration_path,
                threshold,
                "test",
                test_output,
                sidecar_dir=benchmark_root / "test_metrics" / scene / f"aabb_ray_mlp_seed{seed}.sidecar",
            )
        formal_rows.append(
            {
                "seed": seed,
                "member": str(member.resolve()),
                "checkpoint": str(checkpoint.resolve()),
                "calibration": str(calibration_path.resolve()),
                "threshold": threshold,
                "calibrationStatus": calibration["status"],
                "validation": validation,
                "test": test,
                "testRead": False,
            }
        )
    summary = {
        "schema": "pvs-aabb-ray-formal-summary-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "selectedLearningRate": learning_rate,
        "selection": selection,
        "formalRows": formal_rows,
        "lossConfig": dict(LOSS_CONFIG),
        "scanMatrix": {"learningRates": list(SCAN_LEARNING_RATES), "epochs": SCAN_EPOCHS, "stepsPerEpoch": SCAN_STEPS_PER_EPOCH},
        "confirmationMatrix": {"seeds": list(FORMAL_SEEDS), "epochs": FORMAL_EPOCHS, "stepsPerEpoch": FORMAL_STEPS_PER_EPOCH},
        "testRead": False,
    }
    _write_json(benchmark_root / "formal" / scene / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "scan", "train-formal", "all"))
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--scenes", default=",".join(SCENES))
    parser.add_argument("--model-root", type=Path, default=ROOT / "neural_instance_culling/model/out" / EXPERIMENT)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out/paper_results")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    scenes = [value.strip() for value in str(args.scenes).split(",") if value.strip()]
    for scene in scenes:
        if scene not in SCENES:
            parser.error(f"unknown scene {scene!r}; choose from {sorted(SCENES)}")
    args.model_root = args.model_root.resolve()
    args.benchmark_root = args.benchmark_root.resolve()
    if args.mode == "preflight":
        for scene in scenes:
            contract = preflight(args.data_root, scene)
            output = _write_json(
                args.benchmark_root / "preflight" / f"{scene}.json", contract
            )
            print(
                json.dumps(
                    {
                        "scene": scene,
                        "preflight": str(output),
                        "splitCounts": contract["splitCounts"],
                        "testRead": False,
                    },
                    ensure_ascii=False,
                )
            )
        return
    if args.dry_run:
        commands = []
        for scene in scenes:
            for learning_rate in SCAN_LEARNING_RATES:
                commands.append(
                    build_train_command(
                        args.data_root,
                        _scan_member_dir(args.model_root, scene, learning_rate),
                        scene=scene,
                        learning_rate=learning_rate,
                        seed=FORMAL_SEEDS[0],
                        epochs=SCAN_EPOCHS,
                        steps_per_epoch=SCAN_STEPS_PER_EPOCH,
                    )
                )
        print(json.dumps({"commands": commands, "testRead": False}, ensure_ascii=False, indent=2))
        return

    summaries = []
    for scene in scenes:
        summaries.append(
            run_scene(
                args.data_root,
                scene,
                args.model_root,
                args.benchmark_root,
                args.gpu_ids,
                run_scan=args.mode in {"scan", "all"},
                run_formal=args.mode in {"train-formal", "all"},
                run_test=(args.mode == "all" and not args.skip_test),
            )
        )
    print(json.dumps({"experiment": EXPERIMENT, "scenes": scenes, "summaries": summaries, "testRead": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
