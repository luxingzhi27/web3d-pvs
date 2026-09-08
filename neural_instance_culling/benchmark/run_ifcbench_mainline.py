#!/usr/bin/env python3
"""Train and validate the fixed V4 mainline on IFCBench Fantasy Metropolis.

This runner is scene-specific so the retained HKUST protocol and outputs stay
unchanged. Thresholds are frozen by each checkpoint's calibration split; test
is never opened by this entrypoint.
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
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs.py"
EXPERIMENT = "pvs_mainline_v4_ifcbench_fantasy_metropolis_v1"
SEEDS = (20260801, 20260802, 20260803)
EPOCHS = 40
STEPS_PER_EPOCH = 900
EXPECTED_SPLITS = {
    "train": 19647,
    "validation": 2712,
    "calibration": 2183,
    "test": 2710,
    "guard": 0,
}


def paths(data_root: Path) -> dict[str, Path]:
    root = data_root.resolve()
    return {
        "dataset": root / "neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "relation": root / "neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_v4_bounded_relation_csr_v1",
        "runtime_meta": root / "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json",
        "geometry": root / "neural_instance_culling/dataset/out/fixed_geometry_features_metropolis_v2/instance_geo_features_fp16.bin",
        "glb_index": root / "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json",
        "glb_root": root / "ifcbench_fantasy_metropolis_instanced_v2/assets",
    }


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(data_root: Path) -> dict[str, Any]:
    registered = paths(data_root)
    missing = [str(path) for path in registered.values() if not path.exists()]
    missing.extend(str(path) for path in (TRAIN, EVALUATE) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing IFCBench mainline input(s): {missing}")
    dataset_meta = load_json(registered["dataset"] / "dataset_meta.json")
    split_counts = {
        str(name): int(count)
        for name, count in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"IFCBench mainline split changed: expected={EXPECTED_SPLITS}, actual={split_counts}"
        )
    files = dataset_meta.get("files") or {}
    for name in (
        "queryCenterWorld",
        "candidateCameraWorld",
        "viewcellRadiusM",
        "candidateIds",
        "visibleIds",
        "visibleWeights",
    ):
        if name not in files or not (registered["dataset"] / str(files[name])).is_file():
            raise ValueError(f"IFCBench mainline dataset is missing {name}")
    relation_meta = load_json(registered["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3":
        raise ValueError("IFCBench relation artifact has the wrong schema")
    return {
        "schema": "pvs-mainline-v4-ifcbench-preflight-v1",
        "experiment": EXPERIMENT,
        "splitCounts": split_counts,
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "testRead": False,
        "paths": {name: str(path.resolve()) for name, path in registered.items()},
    }


def member_dir(model_root: Path, seed: int) -> Path:
    return model_root / f"full_seed{seed}_e{EPOCHS}"


def train_command(data_root: Path, output: Path, seed: int, smoke: bool) -> list[str]:
    p = paths(data_root)
    return [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(p["dataset"]),
        "--relation-dir", str(p["relation"]),
        "--runtime-meta", str(p["runtime_meta"]),
        "--initial-geo-features", str(p["geometry"]),
        "--glb-index", str(p["glb_index"]),
        "--glb-root", str(p["glb_root"]),
        "--output-dir", str(output),
        "--experiment-name", f"{EXPERIMENT}_seed{seed}",
        "--variant", "full_integrated_visibility_mainline",
        "--occlusion-representation", "survival",
        "--survival-rank", "4",
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_envelope",
        "--instance-calibration-mode", "residual",
        "--loss-variant", "pose_balanced_rvl_contrastive",
        "--epochs", "1" if smoke else str(EPOCHS),
        "--steps-per-epoch", "2" if smoke else str(STEPS_PER_EPOCH),
        "--poses-per-batch", "4",
        "--observation-batch-size", "32768" if smoke else "8192",
        "--eval-every", "1" if smoke else "4",
        "--snapshot-every", "1" if smoke else "4",
        "--max-eval-poses", "2" if smoke else "0",
        "--calibration-bootstrap-replicates", "2" if smoke else "10000",
        "--seed", str(seed),
        "--device", "cuda",
        "--learning-rate", "0.0002",
        "--weight-decay", "0.00001",
        "--survival-loss-weight", "0.25",
        "--relation-consistency-weight", "0.10",
        "--instance-calibration-regularization-weight", "0.02",
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--relation-gradient-cap", "0.25",
        "--integrated-rvl-recall-guard-weight", "0.30",
        "--integrated-rvl-recall-target", "0.99",
        "--integrated-rvl-recall-temperature", "0.05",
        "--integrated-rvl-pose-cvar-fraction", "0.25",
        "--integrated-rvl-pose-cvar-weight", "0.25",
        "--integrated-separation-weight", "0.20",
        "--integrated-tail-ramp-fraction", "0.15",
        "--frontier-positive-mass-fraction", "0.005",
        "--frontier-positive-count-cap", "64",
        "--frontier-negative-fraction", "0.01",
        "--frontier-negative-count-cap", "256",
        "--frontier-margin", "0.50",
        "--frontier-temperature", "0.25",
        "--frontier-positive-importance-floor", "0.5",
        "--frontier-positive-importance-power", "0.5",
    ]


def selected_checkpoint(member: Path) -> Path:
    summary = load_json(member / "calibration_ready_summary.json")
    safe = summary.get("status") == "safe" and summary.get("bestSafe") is not None
    checkpoint = member / ("best_safe.pt" if safe else "best_diagnostic.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def evaluate_command(data_root: Path, member: Path, output: Path, seed: int) -> list[str]:
    p = paths(data_root)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(selected_checkpoint(member)),
        "--dataset-dir", str(p["dataset"]),
        "--runtime-meta", str(p["runtime_meta"]),
        "--initial-geo-features", str(p["geometry"]),
        "--glb-index", str(p["glb_index"]),
        "--glb-root", str(p["glb_root"]),
        "--relation-dir", str(p["relation"]),
        "--model-meta", str(member / "model_meta.json"),
        "--calibration", str(member / "calibration_ready_summary.json"),
        "--output", str(output),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--seed", str(seed),
        "--device", "cuda",
        "--allow-unsafe-diagnostic",
        "--persist-ids",
    ]


def run_jobs(jobs: Sequence[tuple[str, list[str]]], gpu_ids: Sequence[int], log_dir: Path) -> None:
    if not jobs:
        return
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(name: str, command: list[str], gpu: int) -> tuple[str, int, float]:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        started = time.time()
        with (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
            log_dir / f"{name}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        return name, int(result.returncode), float(time.time() - started)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = {
            executor.submit(run, name, command, gpu_ids[index % len(gpu_ids)]): name
            for index, (name, command) in enumerate(jobs)
        }
        for future in as_completed(futures):
            name, return_code, elapsed = future.result()
            print(json.dumps({"job": name, "returnCode": return_code, "elapsedSeconds": elapsed}), flush=True)
            if return_code != 0:
                raise RuntimeError(f"IFCBench mainline job failed: {name}; see {log_dir}")


def summarize(model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        member = member_dir(model_root, seed)
        evaluation = load_json(benchmark_root / "members" / f"seed{seed}_validation.json")
        calibration = load_json(member / "calibration_ready_summary.json")
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(selected_checkpoint(member)),
                "calibrationStatus": calibration.get("status"),
                "threshold": float(evaluation["threshold"]),
                "aggregate": evaluation["aggregate"],
                "pose": evaluation.get("pose"),
                "testRead": False,
            }
        )
    summary = {
        "schema": "pvs-mainline-v4-ifcbench-validation-summary-v1",
        "experiment": EXPERIMENT,
        "thresholdSource": "each checkpoint's calibration split only",
        "validationPoseCount": EXPECTED_SPLITS["validation"],
        "rows": rows,
        "testRead": False,
    }
    write_json(benchmark_root / "validation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "evaluate", "all"))
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
    args = parser.parse_args()
    contract = preflight(args.data_root)
    write_json(args.benchmark_root / "preflight.json", contract)
    if args.mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if args.mode == "smoke":
        output = args.model_root / "smoke"
        run_jobs(
            [("smoke", train_command(args.data_root, output, SEEDS[0], True))],
            args.gpu_ids[:1],
            args.benchmark_root / "logs/smoke",
        )
        return
    if args.mode in {"train", "all"}:
        jobs = [
            (f"train_seed{seed}", train_command(args.data_root, member_dir(args.model_root, seed), seed, False))
            for seed in SEEDS
        ]
        run_jobs(jobs, args.gpu_ids[:3], args.benchmark_root / "logs/train")
    if args.mode in {"evaluate", "all"}:
        jobs = []
        for seed in SEEDS:
            member = member_dir(args.model_root, seed)
            output = args.benchmark_root / "members" / f"seed{seed}_validation.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((f"evaluate_seed{seed}", evaluate_command(args.data_root, member, output, seed)))
        run_jobs(jobs, args.gpu_ids[:3], args.benchmark_root / "logs/evaluate")
        print(json.dumps(summarize(args.model_root, args.benchmark_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
