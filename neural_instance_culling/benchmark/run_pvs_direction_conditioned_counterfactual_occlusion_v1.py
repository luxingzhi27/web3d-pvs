#!/usr/bin/env python3
"""Run the fixed-budget direction-conditioned counterfactual PVS pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "neural_instance_culling/benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_pvs_cross_pose_operating_exposure_representation_v1 as base  # noqa: E402


EXPERIMENT = "pvs_direction_conditioned_counterfactual_occlusion_v1"
OUTPUT_TAG = f"{EXPERIMENT}_20260819"
SEED = 20260801
EPOCHS = 8
STEPS_PER_EPOCH = 100
VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "variant": "A_basis_hidden_no_counterfactual",
        "relationFeatureMode": "basis",
        "exposureSource": "hidden",
        "exposureHiddenDim": 16,
        "counterfactualWeight": 0.0,
    },
    {
        "variant": "B_gated_contrast_no_counterfactual",
        "relationFeatureMode": "gated_contrast",
        "exposureSource": "relation_contrast",
        "exposureHiddenDim": 8,
        "counterfactualWeight": 0.0,
    },
    {
        "variant": "C_basis_hidden_counterfactual",
        "relationFeatureMode": "basis",
        "exposureSource": "hidden",
        "exposureHiddenDim": 16,
        "counterfactualWeight": 0.05,
    },
    {
        "variant": "D_gated_contrast_counterfactual",
        "relationFeatureMode": "gated_contrast",
        "exposureSource": "relation_contrast",
        "exposureHiddenDim": 8,
        "counterfactualWeight": 0.05,
    },
)


def _base_args(args: argparse.Namespace) -> argparse.Namespace:
    values = [
        "factor16",
        "--root", str(Path(args.root).resolve()),
        "--model-root", str(Path(args.model_root).resolve()),
        "--benchmark-root", str(Path(args.benchmark_root).resolve()),
        "--gpu-ids", *(str(value) for value in args.gpu_ids),
        "--poses-per-batch", str(int(args.poses_per_batch)),
    ]
    for option, value in (
        ("--dataset-dir", args.dataset_dir),
        ("--relation-dir", args.relation_dir),
        ("--runtime-meta", args.runtime_meta),
        ("--initial-geo-features", args.initial_geo_features),
        ("--subpose-sidecar", args.subpose_sidecar),
        ("--glb-index", args.glb_index),
        ("--glb-root", args.glb_root),
    ):
        if value:
            values.extend([option, str(value)])
    if args.allow_missing_glb_costs:
        values.append("--allow-missing-glb-costs")
    return base.parse_args(values)


def _mode_config() -> dict[str, Any]:
    config = dict(base._mode_config("scan8"))
    config.update(
        {
            "stage": "quick8",
            "epochs": EPOCHS,
            "stepsPerEpoch": STEPS_PER_EPOCH,
            "evalEvery": EPOCHS,
            "snapshotEvery": 4,
            "calibrationBootstrapReplicates": 1000,
        }
    )
    return config


def _jobs() -> list[dict[str, Any]]:
    shared = base._factor_spec("D_cross_pose_exposure", base._scan_config("S4"))
    jobs = []
    for spec in VARIANTS:
        name = str(spec["variant"])
        jobs.append(
            {
                **shared,
                **spec,
                "stage": "quick8",
                "configId": "S4",
                "seed": SEED,
                "epochs": EPOCHS,
                "member": f"quick8_{name}_seed{SEED}_e{EPOCHS}",
                "testRead": False,
            }
        )
    return jobs


def _replace_option(command: list[str], option: str, value: str) -> None:
    try:
        index = command.index(option)
    except ValueError as exc:
        raise RuntimeError(f"base command lacks registered option {option}") from exc
    if index + 1 >= len(command):
        raise RuntimeError(f"base command option has no value: {option}")
    command[index + 1] = str(value)


def build_train_command(
    args: argparse.Namespace,
    job: Mapping[str, Any],
    member: Path,
) -> list[str]:
    command = base.build_train_command(
        _base_args(args), job, member, _mode_config()
    )
    _replace_option(
        command,
        "--experiment-name",
        f"{EXPERIMENT}_{job['member']}",
    )
    _replace_option(
        command,
        "--exposure-supervision-hidden-dim",
        str(int(job["exposureHiddenDim"])),
    )
    command.extend(
        [
            "--exposure-supervision-source",
            str(job["exposureSource"]),
            "--runtime-relation-feature-mode",
            str(job["relationFeatureMode"]),
            "--counterfactual-view-rank-weight",
            str(float(job["counterfactualWeight"])),
            "--counterfactual-view-rank-margin",
            "0.50",
            "--counterfactual-view-rank-temperature",
            "0.25",
            "--counterfactual-view-positive-weight-power",
            "0.50",
            "--counterfactual-view-positive-importance-mix",
            "0.50",
        ]
    )
    if "--initial-checkpoint" in command:
        raise RuntimeError("counterfactual pilot must train from scratch")
    return command


def _complete(member: Path) -> bool:
    return base._training_artifacts_complete(member, EPOCHS)


def _load_completed_stage_results(
    path: Path,
    *,
    schema: str,
    expected_members: set[str],
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = base._load_json(path)
    rows = payload.get("results")
    if (
        payload.get("schema") != schema
        or payload.get("testRead") is not False
        or not isinstance(rows, list)
    ):
        raise ValueError(f"invalid completed stage record: {path}")
    normalized = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(normalized) != len(rows):
        raise ValueError(f"completed stage record contains a non-object row: {path}")
    actual_members = {str(row.get("member")) for row in normalized}
    if actual_members != expected_members or len(normalized) != len(expected_members):
        raise RuntimeError(f"completed stage record has the wrong members: {path}")
    if any(int(row.get("returnCode", 1)) != 0 for row in normalized):
        raise RuntimeError(f"completed stage record contains a failed member: {path}")
    return sorted(normalized, key=lambda row: str(row["member"]))


def _run_training(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    model_root = Path(args.model_root).resolve()
    base_args = _base_args(args)
    result_path = benchmark_root / "quick8_training_results.json"
    completed = _load_completed_stage_results(
        result_path,
        schema=f"{EXPERIMENT}-training-v1",
        expected_members={str(job["member"]) for job in jobs},
    )
    if completed is not None:
        for job in jobs:
            member = model_root / "quick8" / str(job["member"])
            if not _complete(member):
                raise RuntimeError(
                    f"completed training record points to an incomplete member: {member}"
                )
        return completed
    pending = []
    results = []
    for job in jobs:
        member = model_root / "quick8" / str(job["member"])
        if _complete(member):
            results.append(
                {
                    "member": str(job["member"]),
                    "outputDir": str(member),
                    "returnCode": 0,
                    "reusedCompletedMember": True,
                    "testRead": False,
                }
            )
        else:
            if member.exists():
                raise FileExistsError(f"incomplete member already exists: {member}")
            pending.append(
                (
                    str(job["member"]),
                    build_train_command(args, job, member),
                    member,
                )
            )
    if pending:
        results.extend(
            base._run_queue(
                pending,
                gpu_ids=args.gpu_ids,
                log_root=benchmark_root / "logs/quick8/train",
            )
        )
    failures = [row for row in results if int(row["returnCode"]) != 0]
    if failures:
        raise RuntimeError("counterfactual pilot training failed")
    for job in jobs:
        member = model_root / "quick8" / str(job["member"])
        if not _complete(member):
            raise RuntimeError(f"training member is incomplete: {member}")
    results.sort(key=lambda row: str(row["member"]))
    base._write_json_once(
        result_path,
        {
            "schema": f"{EXPERIMENT}-training-v1",
            "results": results,
            "testRead": False,
        },
    )
    return results


def _run_evaluation(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    model_root = Path(args.model_root).resolve()
    result_path = benchmark_root / "quick8_evaluation_results.json"
    completed = _load_completed_stage_results(
        result_path,
        schema=f"{EXPERIMENT}-evaluation-v1",
        expected_members={f"evaluate_{job['member']}" for job in jobs},
    )
    if completed is not None:
        for job in jobs:
            output = benchmark_root / "members/quick8" / f"{job['member']}.json"
            if not output.is_file():
                raise RuntimeError(
                    f"completed evaluation record points to a missing output: {output}"
                )
        return completed
    pending = []
    results = []
    for job in jobs:
        member = model_root / "quick8" / str(job["member"])
        output = benchmark_root / "members/quick8" / f"{job['member']}.json"
        if output.is_file():
            results.append(
                {
                    "member": f"evaluate_{job['member']}",
                    "output": str(output),
                    "returnCode": 0,
                    "reusedExisting": True,
                    "testRead": False,
                }
            )
        else:
            command = base.build_evaluate_command(
                base_args, job, member, output
            )
            pending.append((f"evaluate_{job['member']}", command, output))
    if pending:
        results.extend(
            base._run_queue(
                pending,
                gpu_ids=args.gpu_ids,
                log_root=benchmark_root / "logs/quick8/evaluate",
            )
        )
    failures = [row for row in results if int(row["returnCode"]) != 0]
    if failures:
        raise RuntimeError("counterfactual pilot validation replay failed")
    for row in results:
        row["output"] = str(row.get("output", row.get("outputDir")))
        row["testRead"] = False
    results.sort(key=lambda row: str(row["member"]))
    base._write_json_once(
        result_path,
        {
            "schema": f"{EXPERIMENT}-evaluation-v1",
            "results": results,
            "testRead": False,
        },
    )
    return results


def _summary_rows(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model_root = Path(args.model_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    rows = []
    for job in jobs:
        member = model_root / "quick8" / str(job["member"])
        evaluation = benchmark_root / "members/quick8" / f"{job['member']}.json"
        row = base._evaluation_row(job, member, evaluation)
        row["relationFeatureMode"] = str(job["relationFeatureMode"])
        row["exposureSource"] = str(job["exposureSource"])
        row["counterfactualWeight"] = float(job["counterfactualWeight"])
        rows.append(row)
    return rows


def _selection(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [row for row in rows if bool(row.get("eligibleSafe"))]
    pool = safe if safe else list(rows)

    def rank(row: Mapping[str, Any]) -> tuple[float, ...]:
        aggregate = row["aggregate"]
        if bool(row.get("eligibleSafe")):
            return (
                float(aggregate.get("usefulCull") or float("-inf")),
                float(aggregate.get("glbByteReduction") or float("-inf")),
                float(aggregate.get("balancedAccuracy") or float("-inf")),
                float(aggregate.get("precision") or float("-inf")),
                float(aggregate.get("accuracy") or float("-inf")),
                -float(aggregate.get("avgPredCount") or float("inf")),
            )
        lcb = aggregate.get("weightedRecallLowerConfidenceBound")
        return (
            float(
                lcb
                if lcb is not None
                else aggregate.get("weightedRecall") or float("-inf")
            ),
            float(aggregate.get("weightedRecall") or float("-inf")),
            float(aggregate.get("usefulCull") or float("-inf")),
            float(aggregate.get("glbByteReduction") or float("-inf")),
            float(aggregate.get("balancedAccuracy") or float("-inf")),
            float(aggregate.get("precision") or float("-inf")),
        )

    winner = max(pool, key=rank)
    return {
        "selectionPool": "safe_candidate_pool" if safe else "diagnostic_candidate_pool",
        "selectionRule": (
            "weighted-recall safety first; useful cull, GLB byte reduction, "
            "balanced accuracy, precision, accuracy, and fewer predictions"
        ),
        "selected": dict(winner),
        "testRead": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(set(args.gpu_ids)) != len(args.gpu_ids) or len(args.gpu_ids) != 4:
        raise ValueError("the registered pilot requires four distinct GPU IDs")
    benchmark_root = Path(args.benchmark_root).resolve()
    benchmark_root.mkdir(parents=True, exist_ok=True)
    contract = base.preflight(_base_args(args))
    contract.update(
        {
            "schema": f"{EXPERIMENT}-preflight-v1",
            "experiment": EXPERIMENT,
            "variants": [dict(row) for row in VARIANTS],
            "epochs": EPOCHS,
            "stepsPerEpoch": STEPS_PER_EPOCH,
            "runtimeFeatureDimension": 124,
            "runtimeFeatureByteLength": base.HKUST_RUNTIME_FEATURE_BYTES,
            "runtimeQueryInputDimension": 130,
            "maximumNeuralAssetBytes": base.MAX_NEURAL_ASSET_BYTES,
            "testRead": False,
        }
    )
    if args.dry_run:
        return {
            "status": "dry_run",
            "preflight": contract,
            "commands": [
                build_train_command(
                    args,
                    job,
                    Path(args.model_root) / "quick8" / str(job["member"]),
                )
                for job in _jobs()
            ],
            "testRead": False,
        }
    base._write_json_once(benchmark_root / "preflight.json", contract)
    jobs = _jobs()
    _run_training(args, jobs, benchmark_root)
    _run_evaluation(args, jobs, benchmark_root)
    rows = _summary_rows(args, jobs)
    summary = base._summarize_rows(rows)
    summary.update(
        {
            "schema": f"{EXPERIMENT}-summary-v1",
            "selection": _selection(rows),
            "testRead": False,
        }
    )
    base._write_json_once(benchmark_root / "quick8_summary.json", summary)
    return {"status": "complete", "summary": summary, "testRead": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--model-root", type=Path, default=ROOT / "neural_instance_culling/model/out" / OUTPUT_TAG)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out" / OUTPUT_TAG)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--relation-dir", default="")
    parser.add_argument("--runtime-meta", default="")
    parser.add_argument("--initial-geo-features", default="")
    parser.add_argument("--subpose-sidecar", default="")
    parser.add_argument("--glb-index", default="")
    parser.add_argument("--glb-root", default="")
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    print(json.dumps(base._jsonable(run(parse_args(argv))), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
