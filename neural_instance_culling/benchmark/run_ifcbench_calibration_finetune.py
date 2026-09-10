#!/usr/bin/env python3
"""Run the registered IFCBench calibration and warm-start fine-tune protocol.

All large source assets are resolved from ``--data-root``. New model and
benchmark outputs are written below this worktree unless explicit output roots
are supplied. The runner has no test path: it trains, calibrates, validates,
selects, and starts the registered three-seed confirmation only.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from statistics import mean, stdev
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling" / "model" / "train_pvs.py"
CALIBRATE = ROOT / "neural_instance_culling" / "benchmark" / "ifcbench_exact_calibration.py"
DATA_ROOT_DEFAULT = Path("/mnt/sda/rhyang/slm")
EXPERIMENT = "pvs_ifcbench_v4_calibration_finetune_v1"
REFINE_EXPERIMENT = "pvs_ifcbench_v4_calibration_finetune_v2_refine"
SCAN_SEED = 20260802
CONFIRM_SEEDS = (20260801, 20260802, 20260803)
SCAN_EPOCHS = 4
CONFIRM_EPOCHS = 8
REFINE_SCAN_SEED = 20260802
REFINE_SCAN_EPOCHS = 2
REFINE_CONFIRM_EPOCHS = 4
STEPS_PER_EPOCH = 900
V1_BOUNDARY_REMOVED_CONFIG = "boundary_removed_rvl030_boundary000_margin050_temp025_lr2e-5"
REFINE_SELECTION_FILENAME = "refine_selection_summary.json"
REFINE_CONFIRMATION_FILENAME = "refine_confirmation_summary.json"
EXPECTED_SPLITS = {
    "train": 19647,
    "validation": 2712,
    "calibration": 2183,
    "test": 2710,
    "guard": 0,
}
CONFIGS: dict[str, dict[str, float]] = {
    "original_loss_rvl030_boundary020_margin050_temp025_lr2e-5": {
        "rvl": 0.30, "boundary": 0.20, "margin": 0.50, "temperature": 0.25, "lr": 2e-5,
    },
    "boundary_half_rvl030_boundary010_margin050_temp025_lr2e-5": {
        "rvl": 0.30, "boundary": 0.10, "margin": 0.50, "temperature": 0.25, "lr": 2e-5,
    },
    "boundary_removed_rvl030_boundary000_margin050_temp025_lr2e-5": {
        "rvl": 0.30, "boundary": 0.00, "margin": 0.50, "temperature": 0.25, "lr": 2e-5,
    },
    "soft_boundary_rvl030_boundary020_margin025_temp050_lr2e-5": {
        "rvl": 0.30, "boundary": 0.20, "margin": 0.25, "temperature": 0.50, "lr": 2e-5,
    },
    "higher_lr_rvl030_boundary020_margin050_temp025_lr5e-5": {
        "rvl": 0.30, "boundary": 0.20, "margin": 0.50, "temperature": 0.25, "lr": 5e-5,
    },
}
REFINE_CONFIGS: dict[str, dict[str, float]] = {
    "boundary_half_rvl030_boundary010_margin050_temp025_lr2e-5": {
        "rvl": 0.30, "boundary": 0.10, "margin": 0.50, "temperature": 0.25, "lr": 2e-5,
    },
    "boundary_removed_rvl035_boundary000_margin050_temp025_lr2e-5": {
        "rvl": 0.35, "boundary": 0.00, "margin": 0.50, "temperature": 0.25, "lr": 2e-5,
    },
    "boundary_removed_rvl030_boundary000_margin050_temp025_lr1e-5": {
        "rvl": 0.30, "boundary": 0.00, "margin": 0.50, "temperature": 0.25, "lr": 1e-5,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def paths(data_root: Path) -> dict[str, Path]:
    root = Path(data_root).resolve()
    return {
        "dataset": root / "neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "relation": root / "neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_v4_bounded_relation_csr_v1",
        "runtime_meta": root / "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json",
        "geometry": root / "neural_instance_culling/dataset/out/fixed_geometry_features_metropolis_v2/instance_geo_features_fp16.bin",
        "glb_index": root / "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json",
        "glb_root": root / "ifcbench_fantasy_metropolis_instanced_v2/assets",
        "source_model_root": root / "neural_instance_culling/model/out/pvs_mainline_v4_ifcbench_fantasy_metropolis_v1",
    }


def preflight(data_root: Path) -> dict[str, Any]:
    registered = paths(data_root)
    required = dict(registered)
    required.pop("source_model_root")
    required.update({"train": TRAIN, "calibration": CALIBRATE})
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing IFCBench calibration input(s): {missing}")
    dataset_meta = _read_json(registered["dataset"] / "dataset_meta.json")
    split_counts = {str(name): int(count) for name, count in (dataset_meta.get("splitCounts") or {}).items()}
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(f"IFCBench split changed: expected={EXPECTED_SPLITS}, actual={split_counts}")
    files = dataset_meta.get("files") or {}
    for name in ("candidateIds", "visibleIds", "visibleWeights", "queryCenterWorld", "candidateCameraWorld", "viewcellRadiusM"):
        if name not in files or not (registered["dataset"] / str(files[name])).is_file():
            raise ValueError(f"IFCBench dataset is missing {name}")
    for seed in CONFIRM_SEEDS:
        checkpoint = registered["source_model_root"] / f"full_seed{seed}_e40" / "best_safe.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing original IFCBench checkpoint: {checkpoint}")
    relation_meta = _read_json(registered["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3":
        raise ValueError("IFCBench relation artifact has the wrong schema")
    return {
        "schema": "pvs-ifcbench-calibration-finetune-preflight-v1",
        "experiment": EXPERIMENT,
        "dataRoot": str(Path(data_root).resolve()),
        "splitCounts": split_counts,
        "scanSeed": SCAN_SEED,
        "confirmSeeds": list(CONFIRM_SEEDS),
        "scanEpochs": SCAN_EPOCHS,
        "confirmEpochs": CONFIRM_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "testRead": False,
        "paths": {name: str(path.resolve()) for name, path in registered.items()},
    }


def source_checkpoint(data_root: Path, seed: int) -> Path:
    return paths(data_root)["source_model_root"] / f"full_seed{int(seed)}_e40" / "best_safe.pt"


def v1_confirm_checkpoint(
    data_root: Path,
    seed: int,
    v1_model_root: Path | None = None,
) -> Path:
    root = (
        Path(v1_model_root).resolve()
        if v1_model_root is not None
        else Path(data_root).resolve()
        / "neural_instance_culling/model/out"
        / EXPERIMENT
    )
    return (
        root
        / "confirm"
        / V1_BOUNDARY_REMOVED_CONFIG
        / f"seed{int(seed)}_e{CONFIRM_EPOCHS}"
        / "last.pt"
    )


def refine_preflight(
    data_root: Path,
    v1_model_root: Path | None = None,
) -> dict[str, Any]:
    base = preflight(data_root)
    inherited = {
        str(seed): v1_confirm_checkpoint(data_root, seed, v1_model_root)
        for seed in CONFIRM_SEEDS
    }
    missing = [str(path) for path in inherited.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing v1 boundary_removed confirmation checkpoint(s): "
            f"{missing}"
        )
    return {
        "schema": "pvs-ifcbench-calibration-finetune-refine-preflight-v1",
        "experiment": REFINE_EXPERIMENT,
        "baseExperiment": EXPERIMENT,
        "dataRoot": base["dataRoot"],
        "splitCounts": base["splitCounts"],
        "scanSeed": REFINE_SCAN_SEED,
        "scanEpochs": REFINE_SCAN_EPOCHS,
        "confirmSeeds": list(CONFIRM_SEEDS),
        "confirmEpochs": REFINE_CONFIRM_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "v1ConfirmConfig": V1_BOUNDARY_REMOVED_CONFIG,
        "v1ConfirmCheckpoints": {
            seed: str(path.resolve()) for seed, path in inherited.items()
        },
        "refineConfigs": {name: dict(values) for name, values in REFINE_CONFIGS.items()},
        "testRead": False,
        "paths": {
            **base["paths"],
            "v1FineTuneModelRoot": str(
                (
                    Path(v1_model_root).resolve()
                    if v1_model_root is not None
                    else Path(data_root).resolve()
                    / "neural_instance_culling/model/out"
                    / EXPERIMENT
                )
            ),
        },
    }


def member_dir(model_root: Path, stage: str, config_name: str, seed: int, epochs: int) -> Path:
    return Path(model_root).resolve() / stage / config_name / f"seed{int(seed)}_e{int(epochs)}"


def _build_train_command(
    data_root: Path,
    output: Path,
    config_name: str,
    seed: int,
    init_checkpoint: Path,
    epochs: int,
    eval_every: int,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    max_eval_poses: int = 0,
    *,
    config_table: Mapping[str, Mapping[str, float]],
    experiment: str,
) -> list[str]:
    p = paths(data_root)
    hp = config_table[config_name]
    return [
        sys.executable, str(TRAIN),
        "--dataset-dir", str(p["dataset"]),
        "--relation-dir", str(p["relation"]),
        "--runtime-meta", str(p["runtime_meta"]),
        "--initial-geo-features", str(p["geometry"]),
        "--glb-index", str(p["glb_index"]),
        "--glb-root", str(p["glb_root"]),
        "--output-dir", str(Path(output).resolve()),
        "--init-checkpoint", str(Path(init_checkpoint).resolve()),
        "--experiment-name", f"{experiment}_{config_name}_seed{int(seed)}",
        "--variant", f"ifcbench_warm_start_{config_name}",
        "--occlusion-representation", "survival",
        "--survival-rank", "4",
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_envelope",
        "--instance-calibration-mode", "residual",
        "--loss-variant", "pose_balanced_rvl_contrastive",
        "--epochs", str(int(epochs)),
        "--steps-per-epoch", str(int(steps_per_epoch)),
        "--poses-per-batch", "4",
        "--observation-batch-size", "8192",
        "--eval-every", str(int(eval_every)),
        "--snapshot-every", str(int(eval_every)),
        "--max-eval-poses", str(int(max_eval_poses)),
        # Exact 10,000-draw threshold selection is performed by the sidecar
        # stage.  One draw keeps the trainer's legacy diagnostic evaluation
        # inexpensive without changing the formal calibration result.
        "--calibration-bootstrap-replicates", "1",
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--learning-rate", str(hp["lr"]),
        "--weight-decay", "0.00001",
        "--survival-loss-weight", "0.25",
        "--relation-consistency-weight", "0.10",
        "--instance-calibration-regularization-weight", "0.02",
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--relation-gradient-cap", "0.25",
        "--integrated-rvl-recall-guard-weight", str(hp["rvl"]),
        "--integrated-rvl-recall-target", "0.99",
        "--integrated-rvl-recall-temperature", "0.05",
        "--integrated-rvl-pose-cvar-fraction", "0.25",
        "--integrated-rvl-pose-cvar-weight", "0.25",
        "--integrated-separation-weight", str(hp["boundary"]),
        "--integrated-tail-ramp-fraction", "0.15",
        "--frontier-positive-mass-fraction", "0.005",
        "--frontier-positive-count-cap", "64",
        "--frontier-negative-fraction", "0.01",
        "--frontier-negative-count-cap", "256",
        "--frontier-margin", str(hp["margin"]),
        "--frontier-temperature", str(hp["temperature"]),
        "--frontier-positive-importance-floor", "0.5",
        "--frontier-positive-importance-power", "0.5",
    ]


def train_command(
    data_root: Path,
    output: Path,
    config_name: str,
    seed: int,
    init_checkpoint: Path,
    epochs: int,
    eval_every: int,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    max_eval_poses: int = 0,
) -> list[str]:
    return _build_train_command(
        data_root,
        output,
        config_name,
        seed,
        init_checkpoint,
        epochs,
        eval_every,
        steps_per_epoch,
        max_eval_poses,
        config_table=CONFIGS,
        experiment=EXPERIMENT,
    )


def refine_train_command(
    data_root: Path,
    output: Path,
    config_name: str,
    seed: int,
    init_checkpoint: Path,
    epochs: int,
    eval_every: int,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    max_eval_poses: int = 0,
) -> list[str]:
    return _build_train_command(
        data_root,
        output,
        config_name,
        seed,
        init_checkpoint,
        epochs,
        eval_every,
        steps_per_epoch,
        max_eval_poses,
        config_table=REFINE_CONFIGS,
        experiment=REFINE_EXPERIMENT,
    )


def score_command(data_root: Path, checkpoint: Path, dataset_split: str, output_dir: Path) -> list[str]:
    split_name = str(dataset_split).lower()
    if split_name not in {"calibration", "validation"}:
        raise ValueError("IFCBench fine-tune scoring is limited to calibration and validation")
    p = paths(data_root)
    return [
        sys.executable, str(CALIBRATE), "score",
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--dataset-dir", str(p["dataset"]),
        "--runtime-meta", str(p["runtime_meta"]),
        "--initial-geo-features", str(p["geometry"]),
        "--split", split_name,
        "--output-dir", str(Path(output_dir).resolve()),
        "--poses-per-batch", "2",
        "--device", "cuda",
    ]


def calibrate_command(sidecar: Path, bootstrap: Path, output: Path) -> list[str]:
    return [
        sys.executable, str(CALIBRATE), "calibrate",
        "--sidecar", str(Path(sidecar).resolve()),
        "--bootstrap-indexes", str(Path(bootstrap).resolve()),
        "--output", str(Path(output).resolve()),
        "--bootstrap-replicates", "10000",
        "--bootstrap-seed", "20260909",
    ]


def evaluate_command(sidecar: Path, calibration: Path, bootstrap: Path, output: Path) -> list[str]:
    return [
        sys.executable, str(CALIBRATE), "evaluate",
        "--sidecar", str(Path(sidecar).resolve()),
        "--calibration", str(Path(calibration).resolve()),
        "--bootstrap-indexes", str(Path(bootstrap).resolve()),
        "--output", str(Path(output).resolve()),
        "--bootstrap-replicates", "10000",
        "--bootstrap-seed", "20260910",
    ]


def _run_one(name: str, command: list[str], gpu: int, log_dir: Path) -> tuple[str, int, float]:
    log_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    environment["PYTHONUNBUFFERED"] = "1"
    started = time.time()
    with (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
        log_dir / f"{name}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    elapsed = time.time() - started
    return name, int(result.returncode), float(elapsed)


def run_jobs(jobs: Sequence[tuple[str, list[str]]], gpu_ids: Sequence[int], log_dir: Path) -> None:
    if not jobs:
        return
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(_run_one, name, command, gpu_ids[index % len(gpu_ids)], log_dir): name
            for index, (name, command) in enumerate(jobs)
        }
        for future in as_completed(futures):
            name, return_code, elapsed = future.result()
            print(json.dumps({"job": name, "returnCode": return_code, "elapsedSeconds": elapsed}), flush=True)
            if return_code != 0:
                raise RuntimeError(f"IFCBench job failed: {name}; see {log_dir}")


def _members_for_configs(
    model_root: Path,
    config_names: Sequence[str],
    seed: int,
    epochs: int,
) -> list[tuple[str, Path]]:
    return [
        (name, member_dir(model_root, "scan", name, seed, epochs))
        for name in config_names
    ]


def _scan_members(model_root: Path) -> list[tuple[str, Path]]:
    return _members_for_configs(model_root, tuple(CONFIGS), SCAN_SEED, SCAN_EPOCHS)


def _refine_scan_members(model_root: Path) -> list[tuple[str, Path]]:
    return _members_for_configs(
        model_root, tuple(REFINE_CONFIGS), REFINE_SCAN_SEED, REFINE_SCAN_EPOCHS
    )


def _run_exact_calibration_for_members(
    data_root: Path,
    members: Sequence[tuple[str, Path, Path]],
    benchmark_root: Path,
    gpu_ids: Sequence[int],
    log_prefix: str,
) -> None:
    members = list(members)
    if not members:
        raise ValueError("exact calibration requires at least one member")
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    benchmark_root = Path(benchmark_root).resolve()
    bootstrap_root = benchmark_root / "fixed_bootstrap"
    calibration_bootstrap = bootstrap_root / "calibration_pose_indices_i32.bin"
    validation_bootstrap = bootstrap_root / "validation_pose_indices_i32.bin"
    score_jobs: list[tuple[str, list[str]]] = []
    for name, member, stage_root in members:
        checkpoint = member / "last.pt"
        score_jobs.append((
            f"score_calibration_{name}",
            score_command(data_root, checkpoint, "calibration", stage_root / "calibration_scores"),
        ))
    run_jobs(score_jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_score_calibration")

    first_name, _first_member, first_root = members[0]
    first_calibration = first_root / "exact_calibration.json"
    first_calibrate = calibrate_command(
        first_root / "calibration_scores", calibration_bootstrap, first_calibration
    )
    first_result = _run_one(
        f"calibrate_{first_name}", first_calibrate, gpu_ids[0], benchmark_root / f"logs/{log_prefix}_calibrate"
    )
    if first_result[1] != 0:
        raise RuntimeError(f"IFCBench exact calibration failed: {first_name}")

    calibration_jobs: list[tuple[str, list[str]]] = []
    for name, _member, stage_root in members[1:]:
        calibration_jobs.append((
            f"calibrate_{name}",
            calibrate_command(stage_root / "calibration_scores", calibration_bootstrap, stage_root / "exact_calibration.json"),
        ))
    run_jobs(calibration_jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_calibrate")

    validation_score_jobs: list[tuple[str, list[str]]] = []
    for name, member, stage_root in members:
        checkpoint = member / "last.pt"
        validation_score_jobs.append((
            f"score_validation_{name}",
            score_command(data_root, checkpoint, "validation", stage_root / "validation_scores"),
        ))
    run_jobs(validation_score_jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_score_validation")

    # Create the shared validation bootstrap before parallel evaluation so all
    # workers observe the same immutable index file rather than racing on its
    # first write.
    first_validation_name, _first_member, first_validation_root = members[0]
    first_validation_result = _run_one(
        f"evaluate_{first_validation_name}",
        evaluate_command(
            first_validation_root / "validation_scores",
            first_validation_root / "exact_calibration.json",
            validation_bootstrap,
            first_validation_root / "validation_frozen.json",
        ),
        gpu_ids[0],
        benchmark_root / f"logs/{log_prefix}_validation",
    )
    if first_validation_result[1] != 0:
        raise RuntimeError(f"IFCBench frozen validation failed: {first_validation_name}")
    validation_jobs: list[tuple[str, list[str]]] = []
    for name, _member, stage_root in members[1:]:
        validation_jobs.append((
            f"evaluate_{name}",
            evaluate_command(
                stage_root / "validation_scores",
                stage_root / "exact_calibration.json",
                validation_bootstrap,
                stage_root / "validation_frozen.json",
            ),
        ))
    run_jobs(validation_jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_validation")


def run_exact_calibration(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
) -> None:
    root = Path(benchmark_root).resolve()
    members = [
        (name, member, root / "scan" / name)
        for name, member in _scan_members(model_root)
    ]
    _run_exact_calibration_for_members(
        data_root, members, root, gpu_ids, log_prefix="exact"
    )


def run_refine_exact_calibration(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
) -> None:
    root = Path(benchmark_root).resolve()
    members = [
        (name, member, root / "scan" / name)
        for name, member in _refine_scan_members(model_root)
    ]
    _run_exact_calibration_for_members(
        data_root, members, root, gpu_ids, log_prefix="refine_exact"
    )


def _selection_rows(
    benchmark_root: Path,
    config_names: Sequence[str],
    *,
    include_provenance: bool = True,
    require_test_free: bool = True,
) -> list[dict[str, Any]]:
    rows = []
    root = Path(benchmark_root).resolve()
    for name in config_names:
        stage_root = root / "scan" / name
        calibration = _read_json(stage_root / "exact_calibration.json")
        validation = _read_json(stage_root / "validation_frozen.json")
        if require_test_free and (
            calibration.get("testRead") is not False
            or validation.get("testRead") is not False
        ):
            raise ValueError(f"selection input is not test-free: {stage_root}")
        metrics = validation.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"frozen validation has no metrics: {stage_root}")
        safe = (
            calibration.get("status") == "safe"
            and float(metrics.get("aggregateWeightedRecall", 0.0)) > 0.99
            and float(metrics.get("aggregateWeightedRecallLowerConfidenceBound", 0.0)) > 0.99
        )
        row: dict[str, Any] = {
            "config": name,
            "calibrationStatus": calibration.get("status"),
            "threshold": float(calibration["selection"]["threshold"]),
            "validationSafetyPassed": bool(safe),
            "calibration": calibration["selected"],
            "validation": dict(metrics),
        }
        if include_provenance:
            row.update({
                "checkpoint": calibration.get("checkpoint"),
                "testRead": False,
            })
        rows.append(row)
    return rows


def _choose_selection(rows: Sequence[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    if not rows:
        raise ValueError("IFCBench selection requires at least one evaluated member")
    safe_rows = [row for row in rows if row["validationSafetyPassed"]]
    pool = safe_rows or rows
    if safe_rows:
        status = "safe_member_selected"
        key = lambda row: (
            float(row["validation"].get("agg_useful_cull", 0.0)),
            float(row["validation"].get("agg_balanced_accuracy", 0.0)),
            float(row["validation"].get("agg_precision", 0.0)),
            -float(row["validation"].get("avg_pred_count", 0.0)),
        )
    else:
        status = "no_safe_member_selected_relative_best"
        key = lambda row: (
            float(row["validation"].get("aggregateWeightedRecallLowerConfidenceBound", 0.0)),
            float(row["validation"].get("aggregateWeightedRecall", 0.0)),
            float(row["validation"].get("agg_useful_cull", 0.0)),
            float(row["validation"].get("agg_balanced_accuracy", 0.0)),
            float(row["validation"].get("agg_precision", 0.0)),
            -float(row["validation"].get("avg_pred_count", 0.0)),
        )
    selected = max(pool, key=key)
    selected_name = selected.get("config")
    if not isinstance(selected_name, str) or not selected_name:
        raise ValueError("IFCBench selection produced an empty config")
    return status, selected


def selection_payload(benchmark_root: Path) -> dict[str, Any]:
    rows = _selection_rows(
        benchmark_root,
        tuple(CONFIGS),
        include_provenance=False,
        require_test_free=False,
    )
    status, selected = _choose_selection(rows)
    return {
        "schema": "pvs-ifcbench-finetune-selection-v1",
        "experiment": EXPERIMENT,
        "selectionStatus": status,
        "selectionRule": "validation safety first; then useful cull, balanced accuracy, precision, and fewer predictions",
        "selected": selected,
        "rows": rows,
        "testRead": False,
    }


def refine_selection_payload(benchmark_root: Path) -> dict[str, Any]:
    rows = _selection_rows(benchmark_root, tuple(REFINE_CONFIGS))
    status, selected = _choose_selection(rows)
    selected_name = str(selected["config"])
    if selected_name not in REFINE_CONFIGS:
        raise ValueError(f"refine selection returned an unknown config: {selected_name}")
    return {
        "schema": "pvs-ifcbench-calibration-finetune-refine-selection-v1",
        "experiment": REFINE_EXPERIMENT,
        "baseExperiment": EXPERIMENT,
        "selectionStatus": status,
        "selectionRule": (
            "safe validation members first; then useful cull, balanced accuracy, "
            "precision, and fewer predictions; without a safe member, highest "
            "validation weighted-recall LCB is selected as the relative best"
        ),
        "selectedConfig": selected_name,
        "selected": selected,
        "rows": rows,
        "testRead": False,
    }


def confirmation_members(
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
) -> list[tuple[str, Path, Path]]:
    return [
        (
            f"seed{seed}",
            member_dir(model_root, "confirm", selected_name, seed, CONFIRM_EPOCHS),
            Path(benchmark_root).resolve() / "confirm" / selected_name / f"seed{seed}",
        )
        for seed in CONFIRM_SEEDS
    ]


def refine_scan_jobs(
    data_root: Path,
    model_root: Path,
    v1_model_root: Path | None = None,
) -> list[tuple[str, list[str]]]:
    initial_checkpoint = v1_confirm_checkpoint(
        data_root, REFINE_SCAN_SEED, v1_model_root
    )
    return [
        (
            f"refine_scan_{name}",
            refine_train_command(
                data_root,
                member,
                name,
                REFINE_SCAN_SEED,
                initial_checkpoint,
                REFINE_SCAN_EPOCHS,
                REFINE_SCAN_EPOCHS,
            ),
        )
        for name, member in _refine_scan_members(model_root)
    ]


def refine_confirmation_members(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    v1_model_root: Path | None = None,
) -> list[tuple[str, Path, Path, Path]]:
    if selected_name not in REFINE_CONFIGS:
        raise ValueError(f"unknown refine config: {selected_name}")
    return [
        (
            f"seed{seed}",
            member_dir(model_root, "confirm", selected_name, seed, REFINE_CONFIRM_EPOCHS),
            Path(benchmark_root).resolve() / "confirm" / selected_name / f"seed{seed}",
            v1_confirm_checkpoint(data_root, seed, v1_model_root),
        )
        for seed in CONFIRM_SEEDS
    ]


def refine_confirmation_jobs(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    v1_model_root: Path | None = None,
) -> list[tuple[str, list[str]]]:
    members = refine_confirmation_members(
        data_root, model_root, benchmark_root, selected_name, v1_model_root
    )
    return [
        (
            f"refine_confirm_{selected_name}_{label}",
            refine_train_command(
                data_root,
                member,
                selected_name,
                seed,
                initial_checkpoint,
                REFINE_CONFIRM_EPOCHS,
                2,
            ),
        )
        for seed, (label, member, _result_root, initial_checkpoint) in zip(
            CONFIRM_SEEDS, members
        )
    ]


def _run_confirmation_exact_for_members(
    data_root: Path,
    members: Sequence[tuple[str, Path, Path]],
    benchmark_root: Path,
    gpu_ids: Sequence[int],
    log_prefix: str,
) -> None:
    members = list(members)
    if not members:
        raise ValueError("exact confirmation requires at least one member")
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    benchmark_root = Path(benchmark_root).resolve()
    bootstrap_root = benchmark_root / "fixed_bootstrap"
    calibration_bootstrap = bootstrap_root / "calibration_pose_indices_i32.bin"
    validation_bootstrap = bootstrap_root / "validation_pose_indices_i32.bin"
    jobs = []
    for label, member, result_root in members:
        jobs.append((
            f"confirm_score_calibration_{label}",
            score_command(data_root, member / "last.pt", "calibration", result_root / "calibration_scores"),
        ))
    run_jobs(jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_score_calibration")

    jobs = []
    for label, _member, result_root in members:
        jobs.append((
            f"confirm_calibrate_{label}",
            calibrate_command(
                result_root / "calibration_scores",
                calibration_bootstrap,
                result_root / "exact_calibration.json",
            ),
        ))
    run_jobs(jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_calibrate")

    jobs = []
    for label, member, result_root in members:
        jobs.append((
            f"confirm_score_validation_{label}",
            score_command(data_root, member / "last.pt", "validation", result_root / "validation_scores"),
        ))
    run_jobs(jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_score_validation")

    jobs = []
    for label, _member, result_root in members:
        jobs.append((
            f"confirm_evaluate_{label}",
            evaluate_command(
                result_root / "validation_scores",
                result_root / "exact_calibration.json",
                validation_bootstrap,
                result_root / "validation_frozen.json",
            ),
        ))
    run_jobs(jobs, gpu_ids, benchmark_root / f"logs/{log_prefix}_validation")


def run_confirmation_exact(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    gpu_ids: Sequence[int],
) -> None:
    _run_confirmation_exact_for_members(
        data_root,
        confirmation_members(model_root, benchmark_root, selected_name),
        benchmark_root,
        gpu_ids,
        log_prefix="confirm_exact",
    )


def run_refine_confirmation_exact(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    gpu_ids: Sequence[int],
    v1_model_root: Path | None = None,
) -> None:
    members = refine_confirmation_members(
        data_root, model_root, benchmark_root, selected_name, v1_model_root
    )
    _run_confirmation_exact_for_members(
        data_root,
        [(label, member, result_root) for label, member, result_root, _ in members],
        benchmark_root,
        gpu_ids,
        log_prefix="refine_confirm_exact",
    )


def confirmation_summary(
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
) -> dict[str, Any]:
    rows = []
    for seed, (_label, member, result_root) in zip(
        CONFIRM_SEEDS,
        confirmation_members(model_root, benchmark_root, selected_name),
    ):
        calibration = _read_json(result_root / "exact_calibration.json")
        validation = _read_json(result_root / "validation_frozen.json")
        rows.append({
            "seed": seed,
            "checkpoint": str((member / "last.pt").resolve()),
            "threshold": float(calibration["selection"]["threshold"]),
            "calibrationStatus": calibration["status"],
            "validation": validation["metrics"],
            "testRead": False,
        })
    metrics = (
        "agg_precision", "agg_recall", "agg_weighted_recall",
        "aggregateWeightedRecallLowerConfidenceBound", "agg_accuracy",
        "agg_balanced_accuracy", "agg_specificity", "agg_useful_cull",
        "agg_bad_cull", "avg_pred_count", "positiveFraction",
    )
    aggregate = {}
    for metric in metrics:
        values = [float(row["validation"][metric]) for row in rows]
        aggregate[metric] = {"mean": mean(values), "sampleStd": stdev(values)}
    return {
        "schema": "pvs-ifcbench-finetune-confirmation-summary-v1",
        "experiment": EXPERIMENT,
        "selectedConfig": selected_name,
        "training": {"epochs": CONFIRM_EPOCHS, "stepsPerEpoch": STEPS_PER_EPOCH},
        "rows": rows,
        "aggregate": aggregate,
        "testRead": False,
    }


def _numeric_metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(
        {
            str(name)
            for row in rows
            for name in row.get("validation", {})
        }
    )
    aggregate: dict[str, Any] = {}
    for name in names:
        values: list[float] = []
        for row in rows:
            value = row.get("validation", {}).get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                values = []
                break
            numeric = float(value)
            if not math.isfinite(numeric):
                values = []
                break
            values.append(numeric)
        if len(values) == len(rows) and values:
            aggregate[name] = {
                "mean": mean(values),
                "sampleStd": stdev(values) if len(values) > 1 else 0.0,
            }
    return aggregate


def refine_confirmation_summary(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    v1_model_root: Path | None = None,
) -> dict[str, Any]:
    if not selected_name or selected_name not in REFINE_CONFIGS:
        raise ValueError("refine confirmation requires a known, non-empty selectedConfig")
    root = Path(benchmark_root).resolve()
    members = refine_confirmation_members(
        data_root, model_root, root, selected_name, v1_model_root
    )
    rows: list[dict[str, Any]] = []
    for seed, (_label, member, result_root, initial_checkpoint) in zip(
        CONFIRM_SEEDS, members
    ):
        calibration_path = result_root / "exact_calibration.json"
        validation_path = result_root / "validation_frozen.json"
        calibration = _read_json(calibration_path)
        validation = _read_json(validation_path)
        if calibration.get("testRead") is not False or validation.get("testRead") is not False:
            raise ValueError(f"refine confirmation result is not test-free: {result_root}")
        calibration_metrics = calibration.get("selected")
        validation_metrics = validation.get("metrics")
        if not isinstance(calibration_metrics, Mapping) or not isinstance(
            validation_metrics, Mapping
        ):
            raise ValueError(f"refine confirmation result has incomplete metrics: {result_root}")
        checkpoint = (member / "last.pt").resolve()
        rows.append({
            "seed": seed,
            "checkpoint": str(checkpoint),
            "initialCheckpoint": str(initial_checkpoint.resolve()),
            "checkpointSeed": calibration.get("checkpointSeed"),
            "checkpointEpoch": calibration.get("checkpointEpoch"),
            "calibrationSummary": str(calibration_path.resolve()),
            "calibrationStatus": calibration.get("status"),
            "threshold": float(calibration["selection"]["threshold"]),
            "calibration": dict(calibration_metrics),
            "validationSummary": str(validation_path.resolve()),
            "validation": dict(validation_metrics),
            "testRead": False,
        })
    return {
        "schema": "pvs-ifcbench-calibration-finetune-refine-confirmation-summary-v1",
        "experiment": REFINE_EXPERIMENT,
        "baseExperiment": EXPERIMENT,
        "selectedConfig": selected_name,
        "selectionSummary": str((root / REFINE_SELECTION_FILENAME).resolve()),
        "training": {
            "epochs": REFINE_CONFIRM_EPOCHS,
            "stepsPerEpoch": STEPS_PER_EPOCH,
            "initialCheckpointStage": "v1 confirm boundary_removed e8 last.pt",
        },
        "rows": rows,
        "aggregate": _numeric_metric_summary(rows),
        "metricScope": "all scalar frozen-validation metrics are aggregated; nested score distributions remain in each row",
        "testRead": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight", "smoke", "scan", "exact", "select", "confirm", "post-scan",
            "refine-preflight", "refine-scan", "refine-exact", "refine-select",
            "refine-confirm", "refine-post-scan", "refine",
        ),
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--v1-model-root", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    refine_mode = args.mode.startswith("refine")
    data_root = Path(args.data_root).resolve()
    default_v1_model_root = data_root / "neural_instance_culling/model/out" / EXPERIMENT
    default_v1_benchmark_root = data_root / "neural_instance_culling/benchmark/out" / EXPERIMENT
    if refine_mode:
        v1_model_root = Path(args.v1_model_root or default_v1_model_root).resolve()
        model_root = Path(
            args.model_root
            or data_root / "neural_instance_culling/model/out" / REFINE_EXPERIMENT
        ).resolve()
        benchmark_root = Path(
            args.benchmark_root
            or data_root / "neural_instance_culling/benchmark/out" / REFINE_EXPERIMENT
        ).resolve()
        if model_root in {default_v1_model_root, v1_model_root} or benchmark_root == default_v1_benchmark_root:
            raise ValueError("refine outputs must not reuse a v1 output root")
        contract = refine_preflight(data_root, v1_model_root)
    else:
        v1_model_root = default_v1_model_root
        model_root = Path(
            args.model_root or data_root / "neural_instance_culling/model/out" / EXPERIMENT
        ).resolve()
        benchmark_root = Path(
            args.benchmark_root or data_root / "neural_instance_culling/benchmark/out" / EXPERIMENT
        ).resolve()
        contract = preflight(data_root)
    _write_json(benchmark_root / "preflight.json", contract)
    if args.mode in ("preflight", "refine-preflight"):
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return

    if refine_mode:
        if args.mode in ("refine", "refine-scan"):
            run_jobs(
                refine_scan_jobs(data_root, model_root, v1_model_root),
                args.gpu_ids,
                benchmark_root / "logs/refine_scan",
            )
            if args.mode == "refine-scan":
                return
        if args.mode in ("refine", "refine-exact", "refine-post-scan"):
            run_refine_exact_calibration(
                data_root, model_root, benchmark_root, args.gpu_ids
            )
            if args.mode == "refine-exact":
                return
        refine_selection_path = benchmark_root / REFINE_SELECTION_FILENAME
        if args.mode in ("refine", "refine-select", "refine-post-scan"):
            payload = refine_selection_payload(benchmark_root)
            _write_json(refine_selection_path, payload)
            print(json.dumps({
                "selectedConfig": payload["selectedConfig"],
                "selectionStatus": payload["selectionStatus"],
                "testRead": False,
            }))
            if args.mode == "refine-select":
                return
        selection = _read_json(refine_selection_path)
        selected_name = selection.get("selectedConfig")
        if not isinstance(selected_name, str) or not selected_name:
            raise ValueError("refine selection summary has no selectedConfig")
        if selected_name not in REFINE_CONFIGS:
            raise ValueError(f"refine selection contains an unknown config: {selected_name}")
        run_jobs(
            refine_confirmation_jobs(
                data_root, model_root, benchmark_root, selected_name, v1_model_root
            ),
            args.gpu_ids,
            benchmark_root / "logs/refine_confirm",
        )
        run_refine_confirmation_exact(
            data_root,
            model_root,
            benchmark_root,
            selected_name,
            args.gpu_ids,
            v1_model_root,
        )
        summary = refine_confirmation_summary(
            data_root,
            model_root,
            benchmark_root,
            selected_name,
            v1_model_root,
        )
        summary_path = benchmark_root / REFINE_CONFIRMATION_FILENAME
        _write_json(summary_path, summary)
        print(json.dumps({
            "selectedConfig": selected_name,
            "confirmationSummary": str(summary_path),
            "testRead": False,
        }))
        return

    if args.mode == "smoke":
        smoke_output = member_dir(model_root, "smoke", "warm_start", SCAN_SEED, 1)
        command = train_command(
            args.data_root,
            smoke_output,
            next(iter(CONFIGS)),
            SCAN_SEED,
            source_checkpoint(args.data_root, SCAN_SEED),
            1,
            1,
            steps_per_epoch=2,
            max_eval_poses=2,
        )
        result = _run_one("warm_start_smoke", command, args.gpu_ids[0], benchmark_root / "logs/smoke")
        print(json.dumps({"job": result[0], "returnCode": result[1], "elapsedSeconds": result[2], "output": str(smoke_output), "testRead": False}))
        if result[1] != 0:
            raise RuntimeError("IFCBench warm-start smoke failed")
        return
    if args.mode == "scan":
        jobs = [
            (
                f"scan_{name}",
                train_command(
                    args.data_root,
                    member_dir(model_root, "scan", name, SCAN_SEED, SCAN_EPOCHS),
                    name,
                    SCAN_SEED,
                    source_checkpoint(args.data_root, SCAN_SEED),
                    SCAN_EPOCHS,
                    4,
                ),
            )
            for name in CONFIGS
        ]
        run_jobs(jobs, args.gpu_ids, benchmark_root / "logs/scan")
        return
    if args.mode in ("exact", "post-scan"):
        run_exact_calibration(args.data_root, model_root, benchmark_root, args.gpu_ids)
        if args.mode == "exact":
            return
    if args.mode in ("select", "post-scan"):
        payload = selection_payload(benchmark_root)
        _write_json(benchmark_root / "selection.json", payload)
        print(json.dumps({"selected": payload["selected"]["config"], "selectionStatus": payload["selectionStatus"], "testRead": False}))
        if args.mode == "select":
            return
    selection = _read_json(benchmark_root / "selection.json")
    selected_name = str(selection["selected"]["config"])
    jobs = [
        (
            f"confirm_{selected_name}_seed{seed}",
            train_command(
                args.data_root,
                member_dir(model_root, "confirm", selected_name, seed, CONFIRM_EPOCHS),
                selected_name,
                seed,
                source_checkpoint(args.data_root, seed),
                CONFIRM_EPOCHS,
                2,
            ),
        )
        for seed in CONFIRM_SEEDS
    ]
    run_jobs(jobs, args.gpu_ids, benchmark_root / "logs/confirm")
    run_confirmation_exact(args.data_root, model_root, benchmark_root, selected_name, args.gpu_ids)
    summary = confirmation_summary(model_root, benchmark_root, selected_name)
    _write_json(benchmark_root / "confirmation_summary.json", summary)
    print(json.dumps({
        "selected": selected_name,
        "confirmationSummary": str(benchmark_root / "confirmation_summary.json"),
        "testRead": False,
    }))


if __name__ == "__main__":
    main()
