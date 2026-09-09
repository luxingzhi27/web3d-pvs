#!/usr/bin/env python3
"""Run the registered IFCBench calibration and warm-start fine-tune protocol.

All large source assets are resolved from ``--data-root``.  New model and
benchmark outputs are written below this worktree unless explicit output roots
are supplied.  The runner has no test path: it trains, calibrates, validates,
selects, and starts the registered three-seed confirmation only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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
SCAN_SEED = 20260802
CONFIRM_SEEDS = (20260801, 20260802, 20260803)
SCAN_EPOCHS = 4
CONFIRM_EPOCHS = 8
STEPS_PER_EPOCH = 900
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


def member_dir(model_root: Path, stage: str, config_name: str, seed: int, epochs: int) -> Path:
    return Path(model_root).resolve() / stage / config_name / f"seed{int(seed)}_e{int(epochs)}"


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
    p = paths(data_root)
    hp = CONFIGS[config_name]
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
        "--experiment-name", f"{EXPERIMENT}_{config_name}_seed{int(seed)}",
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


def score_command(data_root: Path, checkpoint: Path, dataset_split: str, output_dir: Path) -> list[str]:
    p = paths(data_root)
    return [
        sys.executable, str(CALIBRATE), "score",
        "--checkpoint", str(Path(checkpoint).resolve()),
        "--dataset-dir", str(p["dataset"]),
        "--runtime-meta", str(p["runtime_meta"]),
        "--initial-geo-features", str(p["geometry"]),
        "--split", dataset_split,
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


def _scan_members(model_root: Path) -> list[tuple[str, Path]]:
    return [
        (name, member_dir(model_root, "scan", name, SCAN_SEED, SCAN_EPOCHS))
        for name in CONFIGS
    ]


def run_exact_calibration(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
) -> None:
    bootstrap_root = Path(benchmark_root).resolve() / "fixed_bootstrap"
    calibration_bootstrap = bootstrap_root / "calibration_pose_indices_i32.bin"
    validation_bootstrap = bootstrap_root / "validation_pose_indices_i32.bin"
    score_jobs: list[tuple[str, list[str]]] = []
    for name, member in _scan_members(model_root):
        checkpoint = member / "last.pt"
        stage_root = Path(benchmark_root).resolve() / "scan" / name
        score_jobs.append((
            f"score_calibration_{name}",
            score_command(data_root, checkpoint, "calibration", stage_root / "calibration_scores"),
        ))
    run_jobs(score_jobs, gpu_ids, Path(benchmark_root).resolve() / "logs/exact_score_calibration")

    first_name = next(iter(CONFIGS))
    first_root = Path(benchmark_root).resolve() / "scan" / first_name
    first_calibration = first_root / "exact_calibration.json"
    first_calibrate = calibrate_command(
        first_root / "calibration_scores", calibration_bootstrap, first_calibration
    )
    first_result = _run_one(
        f"calibrate_{first_name}", first_calibrate, gpu_ids[0], Path(benchmark_root).resolve() / "logs/exact_calibrate"
    )
    if first_result[1] != 0:
        raise RuntimeError(f"IFCBench exact calibration failed: {first_name}")

    calibration_jobs: list[tuple[str, list[str]]] = []
    for name in list(CONFIGS)[1:]:
        stage_root = Path(benchmark_root).resolve() / "scan" / name
        calibration_jobs.append((
            f"calibrate_{name}",
            calibrate_command(stage_root / "calibration_scores", calibration_bootstrap, stage_root / "exact_calibration.json"),
        ))
    run_jobs(calibration_jobs, gpu_ids, Path(benchmark_root).resolve() / "logs/exact_calibrate")

    validation_score_jobs: list[tuple[str, list[str]]] = []
    for name, member in _scan_members(model_root):
        checkpoint = member / "last.pt"
        stage_root = Path(benchmark_root).resolve() / "scan" / name
        validation_score_jobs.append((
            f"score_validation_{name}",
            score_command(data_root, checkpoint, "validation", stage_root / "validation_scores"),
        ))
    run_jobs(validation_score_jobs, gpu_ids, Path(benchmark_root).resolve() / "logs/exact_score_validation")

    # Create the shared validation bootstrap before parallel evaluation so all
    # workers observe the same immutable index file rather than racing on its
    # first write.
    first_validation_name = next(iter(CONFIGS))
    first_validation_root = Path(benchmark_root).resolve() / "scan" / first_validation_name
    first_validation_result = _run_one(
        f"evaluate_{first_validation_name}",
        evaluate_command(
            first_validation_root / "validation_scores",
            first_validation_root / "exact_calibration.json",
            validation_bootstrap,
            first_validation_root / "validation_frozen.json",
        ),
        gpu_ids[0],
        Path(benchmark_root).resolve() / "logs/exact_validation",
    )
    if first_validation_result[1] != 0:
        raise RuntimeError(f"IFCBench frozen validation failed: {first_validation_name}")
    validation_jobs: list[tuple[str, list[str]]] = []
    for name in list(CONFIGS)[1:]:
        stage_root = Path(benchmark_root).resolve() / "scan" / name
        validation_jobs.append((
            f"evaluate_{name}",
            evaluate_command(
                stage_root / "validation_scores",
                stage_root / "exact_calibration.json",
                validation_bootstrap,
                stage_root / "validation_frozen.json",
            ),
        ))
    run_jobs(validation_jobs, gpu_ids, Path(benchmark_root).resolve() / "logs/exact_validation")


def selection_payload(benchmark_root: Path) -> dict[str, Any]:
    rows = []
    for name in CONFIGS:
        root = Path(benchmark_root).resolve() / "scan" / name
        calibration = _read_json(root / "exact_calibration.json")
        validation = _read_json(root / "validation_frozen.json")
        metrics = validation["metrics"]
        safe = (
            calibration.get("status") == "safe"
            and float(metrics.get("aggregateWeightedRecall", 0.0)) > 0.99
            and float(metrics.get("aggregateWeightedRecallLowerConfidenceBound", 0.0)) > 0.99
        )
        rows.append({
            "config": name,
            "calibrationStatus": calibration.get("status"),
            "threshold": float(calibration["selection"]["threshold"]),
            "validationSafetyPassed": bool(safe),
            "calibration": calibration["selected"],
            "validation": metrics,
        })
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
    return {
        "schema": "pvs-ifcbench-finetune-selection-v1",
        "experiment": EXPERIMENT,
        "selectionStatus": status,
        "selectionRule": "validation safety first; then useful cull, balanced accuracy, precision, and fewer predictions",
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


def run_confirmation_exact(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    selected_name: str,
    gpu_ids: Sequence[int],
) -> None:
    members = confirmation_members(model_root, benchmark_root, selected_name)
    bootstrap_root = Path(benchmark_root).resolve() / "fixed_bootstrap"
    calibration_bootstrap = bootstrap_root / "calibration_pose_indices_i32.bin"
    validation_bootstrap = bootstrap_root / "validation_pose_indices_i32.bin"
    jobs = []
    for label, member, result_root in members:
        jobs.append((
            f"confirm_score_calibration_{label}",
            score_command(data_root, member / "last.pt", "calibration", result_root / "calibration_scores"),
        ))
    run_jobs(jobs, gpu_ids, Path(benchmark_root) / "logs/confirm_exact_score_calibration")

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
    run_jobs(jobs, gpu_ids, Path(benchmark_root) / "logs/confirm_exact_calibrate")

    jobs = []
    for label, member, result_root in members:
        jobs.append((
            f"confirm_score_validation_{label}",
            score_command(data_root, member / "last.pt", "validation", result_root / "validation_scores"),
        ))
    run_jobs(jobs, gpu_ids, Path(benchmark_root) / "logs/confirm_exact_score_validation")

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
    run_jobs(jobs, gpu_ids, Path(benchmark_root) / "logs/confirm_exact_validation")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "smoke", "scan", "exact", "select", "confirm", "post-scan"),
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT_DEFAULT)
    parser.add_argument("--model-root", type=Path, default=ROOT / "neural_instance_culling/model/out" / EXPERIMENT)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out" / EXPERIMENT)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    contract = preflight(args.data_root)
    benchmark_root = Path(args.benchmark_root).resolve()
    _write_json(benchmark_root / "preflight.json", contract)
    if args.mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    model_root = Path(args.model_root).resolve()
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
