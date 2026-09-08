#!/usr/bin/env python3
"""Run the registered 2/4/8/12-direction survival-field capacity experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from neural_instance_culling.benchmark.run_pvs import (
    MAINLINE_CONFIG,
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
    ROOT,
    WEIGHTED_RECALL_FLOOR,
    _load_json,
    _run_queue,
    _write_json,
    build_evaluate_command,
    build_train_command,
    preflight as mainline_preflight,
)


EXPERIMENT = "pvs_survival_rank_capacity_sweep_v1"
CAPACITY_RANKS = (2, 4, 8, 12)
SURVIVAL_PARAMETER_DIM = 7
EXPECTED_RUNTIME_DIMS = {rank: 96 + rank * SURVIVAL_PARAMETER_DIM for rank in CAPACITY_RANKS}
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
BENCHMARK_ROOT = ROOT / "neural_instance_culling/benchmark/out" / EXPERIMENT


def _member_name(rank: int, seed: int) -> str:
    return (
        f"rank{int(rank):02d}_dim{int(rank) * SURVIVAL_PARAMETER_DIM:03d}"
        f"_seed{int(seed)}_e{FORMAL_EPOCHS}"
    )


def _member_dir(model_root: Path, rank: int, seed: int) -> Path:
    return model_root / _member_name(rank, seed)


def _evaluation_path(benchmark_root: Path, rank: int, seed: int) -> Path:
    return benchmark_root / "members" / _member_name(rank, seed) / "validation_evaluation.json"


def _argument(command: Sequence[str], name: str) -> str:
    index = list(command).index(name)
    return str(command[index + 1])


def build_capacity_train_command(
    data_root: Path,
    member: Path,
    *,
    rank: int,
    seed: int,
    smoke: bool = False,
) -> list[str]:
    if int(rank) not in CAPACITY_RANKS:
        raise ValueError(f"rank must be one of {CAPACITY_RANKS}")
    command = build_train_command(
        data_root,
        member,
        MAINLINE_CONFIG,
        "full",
        seed=seed,
        epochs=FORMAL_EPOCHS,
        smoke=smoke,
        steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
        eval_every=4,
        survival_rank=rank,
    )
    command[command.index("--experiment-name") + 1] = f"{EXPERIMENT}_{member.name}"
    command[command.index("--variant") + 1] = "survival_rank_capacity"
    return command


def preflight(data_root: Path) -> dict[str, Any]:
    mainline = mainline_preflight(data_root)
    return {
        "schema": "pvs-survival-rank-capacity-preflight-v1",
        "experiment": EXPERIMENT,
        "ranks": list(CAPACITY_RANKS),
        "survivalDims": [rank * SURVIVAL_PARAMETER_DIM for rank in CAPACITY_RANKS],
        "runtimeFeatureDims": [EXPECTED_RUNTIME_DIMS[rank] for rank in CAPACITY_RANKS],
        "seeds": list(FORMAL_SEEDS),
        "epochs": FORMAL_EPOCHS,
        "stepsPerEpoch": FORMAL_STEPS_PER_EPOCH,
        "fromScratch": True,
        "thresholdSplit": "calibration",
        "selectionSplit": "validation",
        "testRead": False,
        "mainline": mainline,
    }


def _member_complete(member: Path, *, rank: int, seed: int, smoke: bool) -> bool:
    required = (
        member / "last.pt",
        member / "model_meta.json",
        member / "calibration_ready_summary.json",
        member / "train_history.json",
        member / "run_manifest.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        manifest = _load_json(member / "run_manifest.json")
        arguments = manifest.get("arguments")
        history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
        expected_epochs = 1 if smoke else FORMAL_EPOCHS
        expected_steps = 1 if smoke else FORMAL_STEPS_PER_EPOCH
        return bool(
            isinstance(arguments, Mapping)
            and int(arguments.get("survival_rank", -1)) == int(rank)
            and int(arguments.get("seed", -1)) == int(seed)
            and int(arguments.get("epochs", -1)) == expected_epochs
            and int(arguments.get("steps_per_epoch", -1)) == expected_steps
            and arguments.get("loss_variant") == "pose_balanced_rvl_contrastive"
            and arguments.get("relation_source") == "bounded_hierarchical"
            and arguments.get("spectral_mode") == "moment_envelope"
            and arguments.get("instance_calibration_mode") == "residual"
            and manifest.get("testRead") is False
            and isinstance(history, list)
            and len(history) >= expected_epochs
            and max(int(row["epoch"]) for row in history) >= expected_epochs
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _evaluation_checkpoint(member: Path) -> Path:
    calibration = _load_json(member / "calibration_ready_summary.json")
    safe = calibration.get("status") == "safe" and calibration.get("bestSafe") is not None
    checkpoint = member / ("best_safe.pt" if safe else "best_diagnostic.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _evaluation_current(output: Path, member: Path, rank: int, seed: int) -> bool:
    if not output.is_file():
        return False
    try:
        payload = _load_json(output)
        return bool(
            payload.get("split") == "validation"
            and payload.get("testRead") is False
            and int(payload.get("checkpointSeed", -1)) == int(seed)
            and Path(str(payload.get("checkpoint"))).resolve()
            == _evaluation_checkpoint(member).resolve()
            and int(payload["runtime"]["runtimeFeatureDim"])
            == EXPECTED_RUNTIME_DIMS[int(rank)]
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _run(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    ranks: Sequence[int],
    gpu_ids: Sequence[int],
    *,
    smoke: bool,
) -> None:
    seeds = (FORMAL_SEEDS[0],) if smoke else FORMAL_SEEDS
    # Seed-major ordering starts one member of every rank in each four-GPU wave.
    specs = [(rank, seed) for seed in seeds for rank in ranks]
    train_jobs: list[tuple[str, Sequence[str]]] = []
    for rank, seed in specs:
        member = _member_dir(model_root, rank, seed)
        if not _member_complete(member, rank=rank, seed=seed, smoke=smoke):
            if member.exists() and any(member.iterdir()):
                raise RuntimeError(
                    f"incomplete capacity member already exists; inspect before removing: {member}"
                )
            member.mkdir(parents=True, exist_ok=True)
            train_jobs.append(
                (
                    member.name,
                    build_capacity_train_command(
                        data_root,
                        member,
                        rank=rank,
                        seed=seed,
                        smoke=smoke,
                    ),
                )
            )
    if train_jobs:
        _run_queue(train_jobs, gpu_ids, benchmark_root / "logs/train")
    if smoke:
        evaluate_jobs: list[tuple[str, Sequence[str]]] = []
        for rank, seed in specs:
            member = _member_dir(model_root, rank, seed)
            output = _evaluation_path(benchmark_root, rank, seed)
            output.parent.mkdir(parents=True, exist_ok=True)
            evaluate_jobs.append(
                (
                    f"evaluate_{member.name}",
                    build_evaluate_command(data_root, member, output, seed=seed),
                )
            )
        _run_queue(evaluate_jobs, gpu_ids, benchmark_root / "logs/evaluate")
        for rank, seed in specs:
            member = _member_dir(model_root, rank, seed)
            if not _evaluation_current(
                _evaluation_path(benchmark_root, rank, seed), member, rank, seed
            ):
                raise RuntimeError(f"capacity smoke evaluation failed for rank {rank}")
        return

    evaluate_jobs: list[tuple[str, Sequence[str]]] = []
    for rank, seed in specs:
        member = _member_dir(model_root, rank, seed)
        if not _member_complete(member, rank=rank, seed=seed, smoke=False):
            raise RuntimeError(f"capacity training member is incomplete: {member}")
        calibration = _load_json(member / "calibration_ready_summary.json")
        bootstrap = int((calibration.get("calibration") or {}).get("bootstrapReplicates", 0))
        if bootstrap < FORMAL_BOOTSTRAP_REPLICATES:
            raise ValueError(f"capacity member used only {bootstrap} calibration bootstrap replicates")
        output = _evaluation_path(benchmark_root, rank, seed)
        if not _evaluation_current(output, member, rank, seed):
            output.parent.mkdir(parents=True, exist_ok=True)
            evaluate_jobs.append(
                (
                    f"evaluate_{member.name}",
                    build_evaluate_command(data_root, member, output, seed=seed),
                )
            )
    if evaluate_jobs:
        _run_queue(evaluate_jobs, gpu_ids, benchmark_root / "logs/evaluate")


def _parameter_count(checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint modelState is missing: {checkpoint_path}")
    return int(sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)))


def summarize(model_root: Path, benchmark_root: Path, ranks: Sequence[int]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_keys = (
        "precision",
        "recall",
        "f1",
        "accuracy",
        "balancedAccuracy",
        "specificity",
        "weightedRecall",
        "usefulCull",
        "badCull",
        "avgPredCount",
        "predOverCandidate",
    )
    for rank in ranks:
        for seed in FORMAL_SEEDS:
            member = _member_dir(model_root, rank, seed)
            output = _evaluation_path(benchmark_root, rank, seed)
            if not _evaluation_current(output, member, rank, seed):
                raise RuntimeError(f"capacity validation output is missing or stale: {output}")
            evaluation = _load_json(output)
            calibration = _load_json(member / "calibration_ready_summary.json")
            aggregate = evaluation["aggregate"]
            pose_macro = evaluation["poseMacro"]
            lcb = float(aggregate["weightedRecallLowerConfidenceBound"])
            weighted_recall = float(aggregate["weightedRecall"])
            rows.append(
                {
                    "rank": int(rank),
                    "survivalDim": int(rank) * SURVIVAL_PARAMETER_DIM,
                    "seed": int(seed),
                    "member": member.name,
                    "epoch": int(evaluation["epoch"]),
                    "threshold": float(evaluation["threshold"]),
                    "calibrationSafe": bool(
                        calibration.get("status") == "safe"
                        and calibration.get("bestSafe") is not None
                    ),
                    "validationSafe": bool(
                        weighted_recall > WEIGHTED_RECALL_FLOOR
                        and lcb > WEIGHTED_RECALL_FLOOR
                    ),
                    "weightedRecallLowerConfidenceBound": lcb,
                    "aggregate": {key: float(aggregate[key]) for key in metric_keys},
                    "poseMacro": {key: float(pose_macro[key]) for key in metric_keys},
                    "runtimeFeatureDim": int(evaluation["runtime"]["runtimeFeatureDim"]),
                    "runtimeFeatureBytes": int(evaluation["runtime"]["runtimeFeatureBytes"]),
                    "validationElapsedSeconds": float(evaluation["runtime"]["elapsedSeconds"]),
                    "modelStateParameterCount": _parameter_count(_evaluation_checkpoint(member)),
                    "testRead": False,
                }
            )
    by_rank: dict[str, Any] = {}
    for rank in ranks:
        members = [row for row in rows if row["rank"] == rank]
        by_rank[str(rank)] = {
            "memberCount": len(members),
            "safeMemberCount": sum(
                row["calibrationSafe"] and row["validationSafe"] for row in members
            ),
            "survivalDim": rank * SURVIVAL_PARAMETER_DIM,
            "runtimeFeatureDim": EXPECTED_RUNTIME_DIMS[rank],
            "runtimeFeatureBytes": int(statistics.fmean(row["runtimeFeatureBytes"] for row in members)),
            "modelStateParameterCount": int(statistics.fmean(row["modelStateParameterCount"] for row in members)),
            "aggregateMean": {
                key: float(statistics.fmean(row["aggregate"][key] for row in members))
                for key in metric_keys
            },
            "poseMacroMean": {
                key: float(statistics.fmean(row["poseMacro"][key] for row in members))
                for key in metric_keys
            },
        }
    payload = {
        "schema": "pvs-survival-rank-capacity-summary-v1",
        "experiment": EXPERIMENT,
        "ranks": list(ranks),
        "seeds": list(FORMAL_SEEDS),
        "thresholdSource": "each checkpoint's calibration split",
        "selectionSplit": "validation",
        "rows": rows,
        "byRank": by_rank,
        "testRead": False,
    }
    _write_json(benchmark_root / "capacity_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "run", "summarize"))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--ranks", type=int, nargs="+", default=list(CAPACITY_RANKS))
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    ranks = tuple(int(rank) for rank in args.ranks)
    if not ranks or len(set(ranks)) != len(ranks) or any(rank not in CAPACITY_RANKS for rank in ranks):
        raise ValueError(f"ranks must be a unique subset of {CAPACITY_RANKS}")
    contract = preflight(args.data_root)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.benchmark_root / "preflight.json", contract)
    if args.mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if args.dry_run:
        commands = [
            build_capacity_train_command(
                args.data_root,
                _member_dir(args.model_root, rank, FORMAL_SEEDS[0]),
                rank=rank,
                seed=FORMAL_SEEDS[0],
                smoke=args.mode == "smoke",
            )
            for rank in ranks
        ]
        print(json.dumps({"commands": commands, "testRead": False}, indent=2))
        return
    if args.mode in {"smoke", "run"}:
        _run(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            ranks,
            args.gpu_ids,
            smoke=args.mode == "smoke",
        )
    if args.mode in {"run", "summarize"}:
        print(json.dumps(summarize(args.model_root, args.benchmark_root, ranks), indent=2))


if __name__ == "__main__":
    main()
