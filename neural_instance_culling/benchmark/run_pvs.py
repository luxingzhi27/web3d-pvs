#!/usr/bin/env python3
"""Run the fixed V4 paper model and its six core ablations.

Every member trains from scratch. Calibration freezes that checkpoint's
threshold, validation compares configurations at the frozen threshold, and
this runner never reads test.
"""
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

try:
    from .summarize_core_ablation import summarize as summarize_paired_ablation
except ImportError:  # Direct script execution.
    from summarize_core_ablation import summarize as summarize_paired_ablation


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs.py"
EXPERIMENT = "pvs_v4_integrated_visibility_mainline_v1"
FORMAL_SEEDS = (20260801, 20260802, 20260803)
FORMAL_EPOCHS = 40
FORMAL_STEPS_PER_EPOCH = 900
FORMAL_BOOTSTRAP_REPLICATES = 10000
WEIGHTED_RECALL_FLOOR = 0.99
EXPECTED_SPLITS = {
    "train": 5926,
    "calibration": 659,
    "validation": 730,
    "test": 684,
    "guard": 0,
}
EVALUATION_SCHEMA = (
    "pvs-bounded-relation-prior-instance-calibrated-moment-v4-evaluation-v1"
)

MAINLINE_CONFIG: dict[str, Any] = {
    "name": "mainline",
    "lr": 2e-4,
    "guard": 0.30,
    "separation": 0.20,
    "margin": 0.50,
}

PAPER_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "variant": "full_integrated_visibility_mainline",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "moment_envelope",
    },
    "no_relation": {
        "variant": "core_no_relation",
        "representation": "survival",
        "relation": "geometry_only",
        "calibration": "residual",
        "spectral": "moment_envelope",
    },
    "no_survival": {
        "variant": "core_no_survival",
        "representation": "none",
        "relation": "none",
        "calibration": "disabled",
        "spectral": "moment_envelope",
    },
    "generic28": {
        "variant": "core_generic28",
        "representation": "generic28",
        "relation": "none",
        "calibration": "disabled",
        "spectral": "moment_envelope",
    },
    "no_moment": {
        "variant": "core_no_moment",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "point",
    },
    "no_recall_guard": {
        "variant": "core_no_recall_guard",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "moment_envelope",
        "guard": 0.0,
    },
    "no_tail_margin": {
        "variant": "core_no_tail_margin",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "moment_envelope",
        "separation": 0.0,
    },
}
CORE_VARIANTS = tuple(name for name in PAPER_VARIANTS if name != "full")


def _paths(data_root: Path) -> dict[str, Path]:
    root = data_root.resolve()
    return {
        "dataset": root / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
        "relation": root / "neural_instance_culling/dataset/out/pvs_v4_integrated_visibility_mainline_v1/bounded_relation_csr_v3",
        "runtime_meta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root / "neural_instance_culling/dataset/out/fixed_geometry_features_hkust_v3/instance_geo_features_fp16.bin",
        "glb_index": root / "hkust-v3/assets/glbIndex.json",
        "glb_root": root / "hkust-v3/assets",
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(data_root: Path) -> dict[str, Any]:
    paths = _paths(data_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    missing.extend(str(path) for path in (TRAIN, EVALUATE) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing registered mainline inputs: {missing}")
    dataset_meta = _load_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError("mainline requires the explicit four-way split dataset")
    split_counts = {
        str(key): int(value)
        for key, value in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"mainline split counts changed: expected={EXPECTED_SPLITS}, actual={split_counts}"
        )
    files = dataset_meta.get("files") or {}
    for field in (
        "queryCenterWorld",
        "candidateCameraWorld",
        "viewcellRadiusM",
        "candidateIds",
        "visibleIds",
        "visibleWeights",
    ):
        if field not in files or not (paths["dataset"] / str(files[field])).is_file():
            raise ValueError(f"mainline dataset is missing {field}")
    relation_meta = _load_json(paths["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3":
        raise ValueError("mainline relation artifact has the wrong schema")
    return {
        "schema": "pvs-v4-paper-mainline-preflight-v1",
        "experiment": EXPERIMENT,
        "splitCounts": split_counts,
        "initialCheckpoint": None,
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_envelope",
        "instanceCalibrationMode": "residual",
        "lossVariant": "pose_balanced_rvl_contrastive",
        "paperVariants": list(PAPER_VARIANTS),
        "formalSeeds": list(FORMAL_SEEDS),
        "formalEpochs": FORMAL_EPOCHS,
        "formalStepsPerEpoch": FORMAL_STEPS_PER_EPOCH,
        "testRead": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }


def _effective_config(variant: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(MAINLINE_CONFIG)
    for key in ("guard", "separation", "margin"):
        if key in variant:
            result[key] = variant[key]
    return result


def build_train_command(
    data_root: Path,
    member: Path,
    config: Mapping[str, Any],
    variant_name: str,
    *,
    seed: int,
    epochs: int = FORMAL_EPOCHS,
    smoke: bool = False,
    steps_per_epoch: int = FORMAL_STEPS_PER_EPOCH,
    eval_every: int = 4,
    survival_rank: int | None = None,
) -> list[str]:
    del config
    paths = _paths(data_root)
    spec = PAPER_VARIANTS[variant_name]
    hp = _effective_config(spec)
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"]),
        "--relation-dir", str(paths["relation"]),
        "--runtime-meta", str(paths["runtime_meta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glb_index"]),
        "--glb-root", str(paths["glb_root"]),
        "--output-dir", str(member.resolve()),
        "--experiment-name", f"{EXPERIMENT}_{member.name}",
        "--variant", str(spec["variant"]),
        "--occlusion-representation", str(spec["representation"]),
        "--relation-source", str(spec["relation"]),
        "--spectral-mode", str(spec["spectral"]),
        "--instance-calibration-mode", str(spec["calibration"]),
        "--loss-variant", "pose_balanced_rvl_contrastive",
        "--epochs", "1" if smoke else str(int(epochs)),
        "--steps-per-epoch", "1" if smoke else str(int(steps_per_epoch)),
        "--poses-per-batch", "4",
        "--observation-batch-size", "32768" if smoke else "8192",
        "--eval-every", "1" if smoke else str(int(eval_every)),
        "--snapshot-every", "1" if smoke else str(int(eval_every)),
        "--max-eval-poses", "2" if smoke else "0",
        "--calibration-bootstrap-replicates", (
            "2" if smoke else str(FORMAL_BOOTSTRAP_REPLICATES)
        ),
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--learning-rate", str(float(hp["lr"])),
        "--weight-decay", "0.00001",
        "--survival-loss-weight", (
            "0.25" if spec["representation"] == "survival" else "0.0"
        ),
        "--relation-consistency-weight", (
            "0.10" if spec["representation"] == "survival" else "0.0"
        ),
        "--instance-calibration-regularization-weight", (
            "0.02" if spec["representation"] == "survival" else "0.0"
        ),
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--relation-gradient-cap", "0.25",
        "--integrated-rvl-recall-guard-weight", str(float(hp["guard"])),
        "--integrated-rvl-recall-target", "0.99",
        "--integrated-rvl-recall-temperature", "0.05",
        "--integrated-rvl-pose-cvar-fraction", "0.25",
        "--integrated-rvl-pose-cvar-weight", "0.25",
        "--integrated-separation-weight", str(float(hp["separation"])),
        "--integrated-tail-ramp-fraction", "0.15",
        "--frontier-positive-mass-fraction", "0.005",
        "--frontier-positive-count-cap", "64",
        "--frontier-negative-fraction", "0.01",
        "--frontier-negative-count-cap", "256",
        "--frontier-margin", str(float(hp["margin"])),
        "--frontier-temperature", "0.25",
        "--frontier-positive-importance-floor", "0.5",
        "--frontier-positive-importance-power", "0.5",
    ]
    if survival_rank is not None:
        command.extend(["--survival-rank", str(int(survival_rank))])
    return command


def _member_name(stage: str, variant: str, seed: int, epochs: int) -> str:
    return f"{stage}_{variant}_seed{int(seed)}_e{int(epochs)}"


def _specs(
    variants: Sequence[str], *, smoke: bool = False
) -> list[tuple[str, str, int, int]]:
    seeds = (FORMAL_SEEDS[0],) if smoke else FORMAL_SEEDS
    epochs = 1 if smoke else FORMAL_EPOCHS
    stage = "smoke" if smoke else "paper"
    return [(stage, variant, seed, epochs) for variant in variants for seed in seeds]


def _member_dir(model_root: Path, spec: tuple[str, str, int, int]) -> Path:
    return model_root / _member_name(*spec)


def _member_complete(member: Path, spec: tuple[str, str, int, int]) -> bool:
    _stage, variant, seed, epochs = spec
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
        expected = PAPER_VARIANTS[variant]
        return bool(
            isinstance(arguments, Mapping)
            and arguments.get("variant") == expected["variant"]
            and arguments.get("loss_variant") == "pose_balanced_rvl_contrastive"
            and int(arguments.get("seed", -1)) == seed
            and int(arguments.get("epochs", -1)) == epochs
            and int(arguments.get("steps_per_epoch", -1))
            == (1 if epochs == 1 else FORMAL_STEPS_PER_EPOCH)
            and manifest.get("testRead") is False
            and isinstance(history, list)
            and len(history) >= epochs
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _evaluation_checkpoint(member: Path) -> Path:
    summary = _load_json(member / "calibration_ready_summary.json")
    safe = summary.get("status") == "safe" and summary.get("bestSafe") is not None
    checkpoint = member / ("best_safe.pt" if safe else "best_diagnostic.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def build_evaluate_command(
    data_root: Path,
    member: Path,
    output: Path,
    *,
    seed: int,
) -> list[str]:
    paths = _paths(data_root)
    command = [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(_evaluation_checkpoint(member)),
        "--dataset-dir", str(paths["dataset"]),
        "--runtime-meta", str(paths["runtime_meta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glb_index"]),
        "--glb-root", str(paths["glb_root"]),
        "--relation-dir", str(paths["relation"]),
        "--model-meta", str(member / "model_meta.json"),
        "--calibration", str(member / "calibration_ready_summary.json"),
        "--split", "validation",
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--output", str(output.resolve()),
    ]
    summary = _load_json(member / "calibration_ready_summary.json")
    if summary.get("status") != "safe" or summary.get("bestSafe") is None:
        command.append("--allow-unsafe-diagnostic")
    return command


def _run_logged(
    command: Sequence[str], gpu: int, stdout: Path, stderr: Path
) -> dict[str, Any]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err)
    return {
        "returncode": int(result.returncode),
        "elapsedSeconds": float(time.time() - started),
        "gpu": int(gpu),
        "stdout": str(stdout),
        "stderr": str(stderr),
    }


def _run_queue(
    jobs: Sequence[tuple[str, Sequence[str]]],
    gpu_ids: Sequence[int],
    log_root: Path,
) -> list[dict[str, Any]]:
    if not jobs:
        return []
    if not gpu_ids:
        raise ValueError("at least one GPU id is required")
    available: Queue[int] = Queue()
    for gpu in gpu_ids:
        available.put(int(gpu))

    def run(name: str, command: Sequence[str]) -> dict[str, Any]:
        try:
            gpu = available.get(timeout=60)
        except Empty as error:
            raise RuntimeError("no GPU became available") from error
        try:
            result = _run_logged(
                command,
                gpu,
                log_root / f"{name}.stdout.log",
                log_root / f"{name}.stderr.log",
            )
            result["name"] = name
            return result
        finally:
            available.put(gpu)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {executor.submit(run, name, command): name for name, command in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result["returncode"] != 0:
                raise RuntimeError(f"{result['name']} failed; see {result['stderr']}")
    return results


def _evaluation_path(benchmark_root: Path, member: Path) -> Path:
    return benchmark_root / "members" / member.name / "validation_evaluation.json"


def _evaluation_current(output: Path, member: Path, seed: int) -> bool:
    if not output.is_file():
        return False
    try:
        payload = _load_json(output)
        return bool(
            payload.get("schema") == EVALUATION_SCHEMA
            and payload.get("split") == "validation"
            and payload.get("testRead") is False
            and Path(str(payload.get("checkpoint"))).resolve()
            == _evaluation_checkpoint(member).resolve()
            and int(payload.get("checkpointSeed", -1)) == seed
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _run_specs(
    data_root: Path,
    model_root: Path,
    benchmark_root: Path,
    specs: Sequence[tuple[str, str, int, int]],
    gpu_ids: Sequence[int],
) -> None:
    train_jobs: list[tuple[str, Sequence[str]]] = []
    for spec in specs:
        member = _member_dir(model_root, spec)
        if _member_complete(member, spec):
            continue
        if member.exists() and any(member.iterdir()):
            raise RuntimeError(f"incomplete member exists; inspect it first: {member}")
        member.mkdir(parents=True, exist_ok=True)
        train_jobs.append(
            (
                member.name,
                build_train_command(
                    data_root,
                    member,
                    MAINLINE_CONFIG,
                    spec[1],
                    seed=spec[2],
                    epochs=spec[3],
                    smoke=spec[3] == 1,
                ),
            )
        )
    _run_queue(train_jobs, gpu_ids, benchmark_root / "logs/train")

    evaluation_jobs: list[tuple[str, Sequence[str]]] = []
    for spec in specs:
        member = _member_dir(model_root, spec)
        if not _member_complete(member, spec):
            raise RuntimeError(f"training member is incomplete: {member}")
        if spec[3] != 1:
            calibration = _load_json(member / "calibration_ready_summary.json")
            replicates = int(
                (calibration.get("calibration") or {}).get("bootstrapReplicates", 0)
            )
            if replicates < FORMAL_BOOTSTRAP_REPLICATES:
                raise ValueError(
                    f"formal member used only {replicates} bootstrap replicates: {member}"
                )
        output = _evaluation_path(benchmark_root, member)
        if not _evaluation_current(output, member, spec[2]):
            output.parent.mkdir(parents=True, exist_ok=True)
            evaluation_jobs.append(
                (
                    f"evaluate_{member.name}",
                    build_evaluate_command(data_root, member, output, seed=spec[2]),
                )
            )
    _run_queue(evaluation_jobs, gpu_ids, benchmark_root / "logs/evaluate")


def summarize(
    model_root: Path, benchmark_root: Path, variants: Sequence[str]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metric_keys = (
        "precision",
        "recall",
        "weightedRecall",
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
    for spec in _specs(variants):
        member = _member_dir(model_root, spec)
        output = _evaluation_path(benchmark_root, member)
        if not _evaluation_current(output, member, spec[2]):
            raise RuntimeError(f"validation output is missing or stale: {output}")
        evaluation = _load_json(output)
        calibration = _load_json(member / "calibration_ready_summary.json")
        aggregate = evaluation["aggregate"]
        lower_bound = float(aggregate["weightedRecallLowerConfidenceBound"])
        weighted_recall = float(aggregate["weightedRecall"])
        rows.append(
            {
                "member": member.name,
                "variant": spec[1],
                "seed": spec[2],
                "epoch": int(evaluation["epoch"]),
                "threshold": float(evaluation["threshold"]),
                "calibrationSafe": bool(
                    calibration.get("status") == "safe"
                    and calibration.get("bestSafe") is not None
                ),
                "validationSafe": bool(
                    weighted_recall > WEIGHTED_RECALL_FLOOR
                    and lower_bound > WEIGHTED_RECALL_FLOOR
                ),
                "weightedRecallLowerConfidenceBound": lower_bound,
                "aggregate": {key: float(aggregate[key]) for key in metric_keys},
                "testRead": False,
            }
        )
    by_variant: dict[str, Any] = {}
    for variant in variants:
        members = [row for row in rows if row["variant"] == variant]
        by_variant[variant] = {
            "memberCount": len(members),
            "safeMemberCount": sum(
                row["calibrationSafe"] and row["validationSafe"] for row in members
            ),
            "aggregateMean": {
                key: float(statistics.fmean(row["aggregate"][key] for row in members))
                for key in metric_keys
            },
        }
    payload = {
        "schema": "pvs-v4-paper-mainline-summary-v1",
        "thresholdSource": "each checkpoint calibration split",
        "selectionSplit": "validation",
        "rows": rows,
        "byVariant": by_variant,
        "testRead": False,
    }
    _write_json(benchmark_root / "paper_mainline_summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("preflight", "smoke", "train-full", "core-ablation", "all", "summarize"),
    )
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / EXPERIMENT,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / EXPERIMENT,
    )
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = preflight(args.data_root)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.benchmark_root / "preflight.json", contract)
    if args.mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return

    variants = {
        "smoke": ("full",),
        "train-full": ("full",),
        "core-ablation": CORE_VARIANTS,
        "all": tuple(PAPER_VARIANTS),
        "summarize": tuple(PAPER_VARIANTS),
    }[args.mode]
    specs = _specs(variants, smoke=args.mode == "smoke")
    if args.dry_run:
        commands = [
            build_train_command(
                args.data_root,
                _member_dir(args.model_root, spec),
                MAINLINE_CONFIG,
                spec[1],
                seed=spec[2],
                epochs=spec[3],
                smoke=spec[3] == 1,
            )
            for spec in specs
        ]
        print(json.dumps({"commands": commands, "testRead": False}, indent=2))
        return
    if args.mode != "summarize":
        _run_specs(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            specs,
            args.gpu_ids,
        )
    if args.mode in {"all", "summarize"}:
        summary = summarize(args.model_root, args.benchmark_root, variants)
        paired = summarize_paired_ablation(
            args.benchmark_root,
            args.benchmark_root / "paper_core_ablation_summary.json",
        )
        summary["pairedCoreAblation"] = {
            "output": str(
                (args.benchmark_root / "paper_core_ablation_summary.json").resolve()
            ),
            "bootstrapReplicates": int(paired["bootstrap"]["replicates"]),
        }
        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
