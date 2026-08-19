#!/usr/bin/env python3
"""Scan and formally train the pose-balanced frontier visibility loss."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Empty, Queue
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py"
EXPERIMENT = "pvs_pose_balanced_frontier_visibility_loss_v1"
OUTPUT_TAG = f"{EXPERIMENT}_20260819"
SCAN_SEED = 20260801
FORMAL_SEEDS = (20260801, 20260802, 20260803)
SCAN_EPOCHS = 12
FORMAL_EPOCHS = 40
STEPS_PER_EPOCH = 100
WEIGHTED_RECALL_FLOOR = 0.99

ROUND1_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "balanced_only", "weight": 0.0, "margin": 0.5, "temperature": 0.25},
    {"name": "frontier_w010", "weight": 0.1, "margin": 0.5, "temperature": 0.25},
    {"name": "frontier_w020", "weight": 0.2, "margin": 0.5, "temperature": 0.25},
    {"name": "frontier_w040", "weight": 0.4, "margin": 0.5, "temperature": 0.25},
)


def _paths(root: Path) -> dict[str, Path]:
    shared_subpose = root / (
        "neural_instance_culling/dataset/out/"
        "pvs_v4_viewcell_extreme_support_scan_20260818/"
        "subpose_supervision_sidecar"
    )
    local_subpose = ROOT / (
        "neural_instance_culling/dataset/out/"
        "pvs_v4_viewcell_extreme_support_scan_20260818/"
        "subpose_supervision_sidecar"
    )
    return {
        "dataset": root / (
            "neural_instance_culling/dataset/out/"
            "pose_csr_hkust_v3_bounded_relation_moment_fov66_v3"
        ),
        "relation": root / (
            "neural_instance_culling/dataset/out/"
            "pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/"
            "bounded_relation_csr_v3"
        ),
        "runtime_meta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root / (
            "neural_instance_culling/model/out/"
            "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/"
            "instance_geo_features_fp16.bin"
        ),
        "glb_index": root / "hkust-v3/assets/glbIndex.json",
        "glb_root": root / "hkust-v3/assets",
        "subpose": shared_subpose if shared_subpose.is_dir() else local_subpose,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(root: Path) -> dict[str, Any]:
    paths = _paths(root.resolve())
    missing = [str(path) for path in paths.values() if not path.exists()]
    for entry in (TRAIN, EVALUATE):
        if not entry.is_file():
            missing.append(str(entry))
    if missing:
        raise FileNotFoundError(f"missing registered inputs: {missing}")
    return {
        "schema": "pvs-pose-balanced-frontier-visibility-loss-preflight-v1",
        "experiment": EXPERIMENT,
        "scanSeed": SCAN_SEED,
        "formalSeeds": list(FORMAL_SEEDS),
        "scanEpochs": SCAN_EPOCHS,
        "formalEpochs": FORMAL_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "initialCheckpoint": None,
        "lossVariant": "pose_balanced_frontier",
        "testRead": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }


def _round2_configs(base_weight: float) -> tuple[dict[str, Any], ...]:
    label = f"w{int(round(float(base_weight) * 1000)):03d}"
    return (
        {"name": f"frontier_{label}_m025_t025", "weight": base_weight, "margin": 0.25, "temperature": 0.25},
        {"name": f"frontier_{label}_m050_t020", "weight": base_weight, "margin": 0.50, "temperature": 0.20},
        {"name": f"frontier_{label}_m050_t035", "weight": base_weight, "margin": 0.50, "temperature": 0.35},
        {"name": f"frontier_{label}_m075_t025", "weight": base_weight, "margin": 0.75, "temperature": 0.25},
    )


def _member_name(stage: str, config_name: str, seed: int, epochs: int) -> str:
    return f"{stage}_{config_name}_seed{int(seed)}_e{int(epochs)}"


def _member_dir(
    model_root: Path,
    stage: str,
    config_name: str,
    seed: int,
    epochs: int,
) -> Path:
    return model_root.resolve() / _member_name(stage, config_name, seed, epochs)


def _member_complete(member: Path, epochs: int) -> bool:
    required = (
        member / "last.pt",
        member / "model_meta.json",
        member / "calibration_ready_summary.json",
        member / "train_history.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
        return (
            isinstance(history, list)
            and bool(history)
            and max(
                int(row["epoch"])
                for row in history
                if isinstance(row, Mapping)
            )
            >= int(epochs)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _checkpoint_for_evaluation(member: Path) -> tuple[Path, bool]:
    summary = _load_json(member / "calibration_ready_summary.json")
    if summary.get("testRead") is not False:
        raise ValueError(f"calibration summary read test data: {member}")
    safe = summary.get("status") == "safe" and summary.get("bestSafe") is not None
    checkpoint = member / ("best_safe.pt" if safe else "best_diagnostic.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected checkpoint: {checkpoint}")
    return checkpoint, bool(safe)


def build_train_command(
    root: Path,
    member: Path,
    config: Mapping[str, Any],
    *,
    seed: int,
    epochs: int,
    smoke: bool = False,
) -> list[str]:
    paths = _paths(root.resolve())
    effective_epochs = 1 if smoke else int(epochs)
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--relation-dir", str(paths["relation"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--subpose-sidecar", str(paths["subpose"].resolve()),
        "--output-dir", str(member.resolve()),
        "--experiment-name", f"{EXPERIMENT}_{member.name}",
        "--variant", "full_pose_balanced_frontier",
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_envelope",
        "--instance-calibration-mode", "residual",
        "--loss-variant", "pose_balanced_frontier",
        "--refinement-scope", "all",
        "--epochs", str(effective_epochs),
        "--steps-per-epoch", "1" if smoke else str(STEPS_PER_EPOCH),
        "--poses-per-batch", "4",
        "--observation-batch-size", "32768" if smoke else "8192",
        "--eval-every", "1" if smoke else "4",
        "--snapshot-every", "1" if smoke else "4",
        "--max-eval-poses", "2" if smoke else "0",
        "--calibration-bootstrap-replicates", (
            "2" if smoke else "10000" if effective_epochs == FORMAL_EPOCHS else "2000"
        ),
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--learning-rate", "0.0002",
        "--weight-decay", "0.00001",
        "--survival-loss-weight", "0.25",
        "--relation-consistency-weight", "0.10",
        "--utility-loss-weight", "0.0",
        "--download-loss-weight", "0.0",
        "--boundary-tail-weight", "0.0",
        "--negative-band-weight", "0.0",
        "--glb-resource-weight", "0.0",
        "--relation-gradient-cap", "0.25",
        "--schedule-gradient-cap", "0.0",
        "--efficiency-gradient-cap", "0.0",
        "--instance-calibration-regularization-weight", "0.02",
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--rvl-count-weight", "0.0",
        "--rvl-rank-weight", "0.0",
        "--query-tail-separator-family", "disabled",
        "--query-tail-separation-loss-weight", "0.0",
        "--frontier-loss-weight", str(float(config["weight"])),
        "--frontier-positive-mass-fraction", "0.005",
        "--frontier-positive-count-cap", "64",
        "--frontier-negative-fraction", "0.01",
        "--frontier-negative-count-cap", "256",
        "--frontier-margin", str(float(config["margin"])),
        "--frontier-temperature", str(float(config["temperature"])),
        "--frontier-positive-importance-floor", "0.5",
        "--frontier-positive-importance-power", "0.5",
    ]
    if "--initial-checkpoint" in command:
        raise RuntimeError("from-scratch command unexpectedly contains a checkpoint")
    return command


def build_evaluate_command(
    root: Path,
    member: Path,
    output: Path,
    *,
    seed: int,
) -> list[str]:
    paths = _paths(root.resolve())
    checkpoint, _safe = _checkpoint_for_evaluation(member)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--relation-dir", str(paths["relation"].resolve()),
        "--model-meta", str((member / "model_meta.json").resolve()),
        "--calibration", str((member / "calibration_ready_summary.json").resolve()),
        "--output", str(output.resolve()),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--allow-unsafe-diagnostic",
        "--persist-ids",
    ]


def _run_logged(
    command: Sequence[str],
    *,
    gpu: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "command": list(command),
        "gpu": int(gpu),
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "testRead": False,
    }


def _run_queue(
    jobs: Sequence[tuple[str, Sequence[str]]],
    *,
    gpu_ids: Sequence[int],
    log_root: Path,
) -> list[dict[str, Any]]:
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    pending: Queue[tuple[str, Sequence[str]]] = Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu: int) -> list[dict[str, Any]]:
        owned: list[dict[str, Any]] = []
        while True:
            try:
                name, command = pending.get_nowait()
            except Empty:
                return owned
            result = {
                "member": name,
                **_run_logged(
                    command,
                    gpu=gpu,
                    stdout_path=log_root / f"{name}.stdout.log",
                    stderr_path=log_root / f"{name}.stderr.log",
                ),
            }
            owned.append(result)
            pending.task_done()
            print(
                json.dumps(
                    {
                        key: result[key]
                        for key in ("member", "gpu", "returnCode", "elapsedSeconds")
                    }
                ),
                flush=True,
            )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, int(gpu)) for gpu in gpu_ids]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [row for row in results if int(row["returnCode"]) != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; inspect logs")
    return sorted(results, key=lambda row: str(row["member"]))


def _evaluation_path(benchmark_root: Path, member_name: str) -> Path:
    return benchmark_root / "members" / member_name / "validation_evaluation.json"


def _train_specs(
    root: Path,
    model_root: Path,
    specs: Sequence[tuple[str, Mapping[str, Any], int, int]],
    *,
    smoke: bool = False,
) -> list[tuple[str, Sequence[str]]]:
    jobs: list[tuple[str, Sequence[str]]] = []
    for stage, config, seed, epochs in specs:
        member = _member_dir(model_root, stage, str(config["name"]), seed, epochs)
        if not smoke and _member_complete(member, epochs):
            continue
        jobs.append(
            (
                member.name,
                build_train_command(
                    root,
                    member,
                    config,
                    seed=seed,
                    epochs=epochs,
                    smoke=smoke,
                ),
            )
        )
    return jobs


def _evaluate_specs(
    root: Path,
    model_root: Path,
    benchmark_root: Path,
    specs: Sequence[tuple[str, Mapping[str, Any], int, int]],
) -> list[tuple[str, Sequence[str]]]:
    jobs: list[tuple[str, Sequence[str]]] = []
    for stage, config, seed, epochs in specs:
        member = _member_dir(model_root, stage, str(config["name"]), seed, epochs)
        if not _member_complete(member, epochs):
            raise RuntimeError(f"incomplete member: {member}")
        output = _evaluation_path(benchmark_root, member.name)
        if output.is_file():
            continue
        jobs.append(
            (
                f"evaluate_{member.name}",
                build_evaluate_command(root, member, output, seed=seed),
            )
        )
    return jobs


def _row_for_spec(
    model_root: Path,
    benchmark_root: Path,
    spec: tuple[str, Mapping[str, Any], int, int],
) -> dict[str, Any]:
    stage, config, seed, epochs = spec
    member = _member_dir(model_root, stage, str(config["name"]), seed, epochs)
    calibration = _load_json(member / "calibration_ready_summary.json")
    evaluation = _load_json(_evaluation_path(benchmark_root, member.name))
    if evaluation.get("split") != "validation" or evaluation.get("testRead") is not False:
        raise ValueError(f"invalid scan evaluation split: {member}")
    aggregate = evaluation["aggregate"]
    calibration_safe = (
        calibration.get("status") == "safe" and calibration.get("bestSafe") is not None
    )
    weighted_recall = float(aggregate["weightedRecall"])
    weighted_lcb_raw = aggregate.get("weightedRecallLowerConfidenceBound")
    weighted_lcb = (
        None if weighted_lcb_raw is None else float(weighted_lcb_raw)
    )
    # The checkpoint's calibration status already includes the registered
    # one-sided LCB. The validation evaluator currently reports only the
    # frozen-threshold point estimate, so a missing validation LCB is recorded
    # rather than fabricated or treated as a numerical score.
    validation_safe = weighted_recall > WEIGHTED_RECALL_FLOOR
    return {
        "member": member.name,
        "stage": stage,
        "config": dict(config),
        "seed": int(seed),
        "epochs": int(epochs),
        "checkpointSafeOnCalibration": bool(calibration_safe),
        "safeOnValidation": bool(validation_safe),
        "eligibleSafe": bool(calibration_safe and validation_safe),
        "validationWeightedRecallLcbAvailable": weighted_lcb is not None,
        "threshold": float(evaluation["threshold"]),
        "aggregate": {
            key: (
                None
                if aggregate[key] is None
                else float(aggregate[key])
            )
            for key in (
                "precision",
                "recall",
                "weightedRecall",
                "weightedRecallLowerConfidenceBound",
                "accuracy",
                "balancedAccuracy",
                "specificity",
                "f1",
                "usefulCull",
                "badCull",
                "avgCandidateCount",
                "avgGtCount",
                "avgPredCount",
            )
        },
        "poseMacro": {
            key: float(evaluation["poseMacro"][key])
            for key in (
                "precision",
                "recall",
                "weightedRecall",
                "accuracy",
                "balancedAccuracy",
                "specificity",
                "f1",
                "usefulCull",
                "badCull",
            )
        },
        "testRead": False,
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["aggregate"]
    if bool(row["eligibleSafe"]):
        return (
            1.0,
            float(metrics["precision"]),
            float(metrics["balancedAccuracy"]),
            float(metrics["accuracy"]),
            float(metrics["usefulCull"]),
            -float(metrics["avgPredCount"]),
        )
    return (
        0.0,
        min(
            float(metrics["weightedRecall"]),
            (
                float(metrics["weightedRecallLowerConfidenceBound"])
                if metrics["weightedRecallLowerConfidenceBound"] is not None
                else float(metrics["weightedRecall"])
            ),
        ),
        float(metrics["balancedAccuracy"]),
        float(metrics["precision"]),
        float(metrics["accuracy"]),
        float(metrics["usefulCull"]),
    )


def _select(
    model_root: Path,
    benchmark_root: Path,
    specs: Sequence[tuple[str, Mapping[str, Any], int, int]],
    output: Path,
) -> dict[str, Any]:
    rows = [_row_for_spec(model_root, benchmark_root, spec) for spec in specs]
    if not rows:
        raise ValueError("cannot select from an empty scan")
    selected = max(rows, key=_rank)
    payload = {
        "schema": "pvs-pose-balanced-frontier-scan-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "thresholdSource": "each checkpoint's calibration split",
        "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
        "selectionRule": (
            "eligible safe first; then aggregate precision, balanced accuracy, "
            "accuracy, useful cull, and lower predicted count. If none are safe, "
            "select the relative best weighted-recall lower bound before classification metrics."
        ),
        "selected": selected,
        "rows": sorted(rows, key=lambda row: str(row["member"])),
        "testRead": False,
    }
    _write_json(output, payload)
    return payload


def _seed_aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("formal seed aggregation requires at least one row")

    def summarize(values: Sequence[Any]) -> dict[str, Any]:
        finite = [float(value) for value in values if value is not None]
        if not finite:
            return {
                "availableCount": 0,
                "mean": None,
                "populationStd": None,
                "minimum": None,
                "maximum": None,
            }
        return {
            "availableCount": len(finite),
            "mean": float(statistics.fmean(finite)),
            "populationStd": float(statistics.pstdev(finite)),
            "minimum": float(min(finite)),
            "maximum": float(max(finite)),
        }

    aggregate_keys = tuple(rows[0]["aggregate"].keys())
    pose_keys = tuple(rows[0]["poseMacro"].keys())
    return {
        "memberCount": len(rows),
        "safeMemberCount": sum(bool(row["eligibleSafe"]) for row in rows),
        "allMembersSafe": all(bool(row["eligibleSafe"]) for row in rows),
        "threshold": summarize([float(row["threshold"]) for row in rows]),
        "aggregate": {
            key: summarize([row["aggregate"][key] for row in rows])
            for key in aggregate_keys
        },
        "poseMacro": {
            key: summarize([row["poseMacro"][key] for row in rows])
            for key in pose_keys
        },
        "testRead": False,
    }


def _round1_specs() -> list[tuple[str, Mapping[str, Any], int, int]]:
    return [("scan_round1", config, SCAN_SEED, SCAN_EPOCHS) for config in ROUND1_CONFIGS]


def _best_nonzero_weight(selection: Mapping[str, Any]) -> float:
    rows = [
        row
        for row in selection["rows"]
        if float(row["config"]["weight"]) > 0.0
    ]
    if not rows:
        return 0.2
    return float(max(rows, key=_rank)["config"]["weight"])


def _formal_specs(config: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any], int, int]]:
    return [
        ("formal40", config, seed, FORMAL_EPOCHS)
        for seed in FORMAL_SEEDS
    ]


def _run_stage(
    specs: Sequence[tuple[str, Mapping[str, Any], int, int]],
    *,
    root: Path,
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
    label: str,
) -> None:
    train_jobs = _train_specs(root, model_root, specs)
    if train_jobs:
        results = _run_queue(
            train_jobs,
            gpu_ids=gpu_ids,
            log_root=benchmark_root / "logs" / label / "train",
        )
        _write_json(
            benchmark_root / f"{label}_training_results.json",
            {"results": results, "testRead": False},
        )
    evaluation_jobs = _evaluate_specs(root, model_root, benchmark_root, specs)
    if evaluation_jobs:
        results = _run_queue(
            evaluation_jobs,
            gpu_ids=gpu_ids,
            log_root=benchmark_root / "logs" / label / "evaluate",
        )
        _write_json(
            benchmark_root / f"{label}_evaluation_results.json",
            {"results": results, "testRead": False},
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("smoke", "scan", "formal40", "evaluate", "summarize", "all"),
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / OUTPUT_TAG,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / OUTPUT_TAG,
    )
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = preflight(args.root)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.benchmark_root / "preflight.json", contract)

    if args.mode == "smoke":
        config = ROUND1_CONFIGS[2]
        member = _member_dir(args.model_root / "smoke", "smoke", config["name"], SCAN_SEED, 1)
        jobs = [
            (
                member.name,
                build_train_command(
                    args.root,
                    member,
                    config,
                    seed=SCAN_SEED,
                    epochs=1,
                    smoke=True,
                ),
            )
        ]
        if args.dry_run:
            print(json.dumps({"jobs": jobs, "testRead": False}, indent=2))
            return
        _run_queue(jobs, gpu_ids=args.gpu_ids[:1], log_root=args.benchmark_root / "logs/smoke")
        return

    round1_specs = _round1_specs()
    if args.dry_run:
        jobs = _train_specs(args.root, args.model_root, round1_specs)
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "initialCheckpoint": None,
                    "round1JobCount": len(jobs),
                    "commands": [list(command) for _name, command in jobs],
                    "testRead": False,
                },
                indent=2,
            )
        )
        return

    if args.mode in {"scan", "all"}:
        _run_stage(
            round1_specs,
            root=args.root,
            model_root=args.model_root,
            benchmark_root=args.benchmark_root,
            gpu_ids=args.gpu_ids,
            label="scan_round1",
        )
        round1_selection = _select(
            args.model_root,
            args.benchmark_root,
            round1_specs,
            args.benchmark_root / "scan_round1_selection.json",
        )
        base_weight = _best_nonzero_weight(round1_selection)
        round2_specs = [
            ("scan_round2", config, SCAN_SEED, SCAN_EPOCHS)
            for config in _round2_configs(base_weight)
        ]
        _run_stage(
            round2_specs,
            root=args.root,
            model_root=args.model_root,
            benchmark_root=args.benchmark_root,
            gpu_ids=args.gpu_ids,
            label="scan_round2",
        )
        final_selection = _select(
            args.model_root,
            args.benchmark_root,
            [*round1_specs, *round2_specs],
            args.benchmark_root / "selected_configuration.json",
        )
    else:
        selection_path = args.benchmark_root / "selected_configuration.json"
        if not selection_path.is_file():
            raise FileNotFoundError(
                "selected_configuration.json is required; run scan first"
            )
        final_selection = _load_json(selection_path)

    selected_config = dict(final_selection["selected"]["config"])
    formal_specs = _formal_specs(selected_config)
    if args.mode in {"formal40", "all"}:
        _run_stage(
            formal_specs,
            root=args.root,
            model_root=args.model_root,
            benchmark_root=args.benchmark_root,
            gpu_ids=args.gpu_ids,
            label="formal40",
        )
    elif args.mode == "evaluate":
        evaluation_jobs = _evaluate_specs(
            args.root,
            args.model_root,
            args.benchmark_root,
            formal_specs,
        )
        if evaluation_jobs:
            _run_queue(
                evaluation_jobs,
                gpu_ids=args.gpu_ids,
                log_root=args.benchmark_root / "logs/formal40/evaluate",
            )

    if args.mode in {"formal40", "evaluate", "summarize", "all"}:
        summary = _select(
            args.model_root,
            args.benchmark_root,
            formal_specs,
            args.benchmark_root / "formal40_summary.json",
        )
        summary["selectedHyperparameters"] = selected_config
        summary["memberCount"] = len(formal_specs)
        summary["seeds"] = list(FORMAL_SEEDS)
        summary["epochs"] = FORMAL_EPOCHS
        summary["seedAggregate"] = _seed_aggregate(summary["rows"])
        _write_json(args.benchmark_root / "formal40_summary.json", summary)


if __name__ == "__main__":
    main()
