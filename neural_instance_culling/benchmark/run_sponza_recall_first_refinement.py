#!/usr/bin/env python3
"""Continue Sponza from its strongest early checkpoints with recall-first training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

try:
    from .run_standard_graphics_connected_sah_full import preflight, train_command
    from .standard_graphics_connected_sah_config import ROOT
except ImportError:
    from run_standard_graphics_connected_sah_full import preflight, train_command  # type: ignore
    from standard_graphics_connected_sah_config import ROOT  # type: ignore


EXPERIMENT = "pvs_v4_sponza_recall_first_refinement_v1"
SCENE = "sponza_128k"
SEEDS = (20260801, 20260802, 20260803)
BASE_ROOT = ROOT / "neural_instance_culling/model/out/pvs_v4_standard_graphics_connected_sah_128k_v1/sponza_128k"
SOURCES = {
    seed: BASE_ROOT / f"full_seed{seed}_e40/best_diagnostic.pt"
    for seed in SEEDS
}
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT


@dataclass(frozen=True)
class Config:
    name: str
    learning_rate: float
    guard_weight: float
    pose_cvar_fraction: float
    pose_cvar_weight: float
    separation_weight: float
    positive_mass_fraction: float
    positive_count_cap: int
    negative_fraction: float
    negative_count_cap: int


CONFIGS = (
    Config("recall_lr1e6", 1e-6, 2.0, 0.75, 1.0, 0.02, 0.05, 512, 0.001, 32),
    Config("recall_lr2e6", 2e-6, 1.5, 0.50, 1.0, 0.05, 0.03, 256, 0.002, 64),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def replace_value(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def member_dir(stage: str, config: Config, seed: int) -> Path:
    return MODEL_ROOT / stage / f"{config.name}_seed{seed}"


def completed(path: Path) -> bool:
    summary = path / "calibration_ready_summary.json"
    return summary.is_file() and read_json(summary).get("testRead") is False


def command(config: Config, seed: int, output: Path, *, final: bool) -> list[str]:
    result = train_command(SCENE, output, seed, smoke=False)
    epochs = 4 if final else 2
    values = {
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{config.name}_seed{seed}",
        "--epochs": str(epochs),
        "--steps-per-epoch": "450",
        "--poses-per-batch": "16",
        "--hard-pose-fraction": "0.75",
        "--hard-pose-quantile": "0.40",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--learning-rate": str(config.learning_rate),
        "--survival-loss-weight": "0.05",
        "--relation-consistency-weight": "0.02",
        "--integrated-rvl-recall-guard-weight": str(config.guard_weight),
        "--integrated-rvl-recall-target": "0.999",
        "--integrated-rvl-pose-cvar-fraction": str(config.pose_cvar_fraction),
        "--integrated-rvl-pose-cvar-weight": str(config.pose_cvar_weight),
        "--integrated-separation-weight": str(config.separation_weight),
        "--integrated-rvl-recall-guard-zero-fraction": "0",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.01",
        "--integrated-rvl-recall-guard-middle-scale": "1",
        "--integrated-tail-zero-fraction": "0",
        "--integrated-tail-ramp-fraction": "0.01",
        "--frontier-positive-mass-fraction": str(config.positive_mass_fraction),
        "--frontier-positive-count-cap": str(config.positive_count_cap),
        "--frontier-negative-fraction": str(config.negative_fraction),
        "--frontier-negative-count-cap": str(config.negative_count_cap),
    }
    for flag, value in values.items():
        replace_value(result, flag, value)
    result.extend([
        "--training-course-total-steps", str(epochs * 450),
        "--init-checkpoint", str(SOURCES[seed]),
        "--init-instance-calibration-blend", "1",
    ])
    return result


def run_logged(name: str, args: list[str], gpu: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    record_path = log_root / f"{name}.command.json"
    record = {
        "schema": "pvs-sponza-recall-first-command-v1",
        "experiment": EXPERIMENT,
        "name": name,
        "gpuId": gpu,
        "command": args,
        "testRead": False,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(record_path, record)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(args, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    record["returnCode"] = int(result.returncode)
    record["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(record_path, record)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)


def run_parallel(jobs: list[tuple[str, list[str]]], gpu_slots: Sequence[int], log_root: Path) -> None:
    for start in range(0, len(jobs), len(gpu_slots)):
        batch = jobs[start:start + len(gpu_slots)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(run_logged, name, args, gpu_slots[index], log_root): name
                for index, (name, args) in enumerate(batch)
            }
            errors = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
            if errors:
                raise RuntimeError("; ".join(errors))


def metric(validation: dict[str, Any], *names: str) -> float:
    for name in names:
        if validation.get(name) is not None:
            return float(validation[name])
    return 0.0


def row(stage: str, config: Config, seed: int) -> dict[str, Any]:
    path = member_dir(stage, config, seed)
    summary = read_json(path / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    lcb = metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound")
    return {
        "seed": seed,
        "config": config.name,
        "member": str(path.resolve()),
        "safe": summary.get("status") == "safe" and wr > 0.99 and lcb > 0.99,
        "validation": validation,
        "testRead": False,
    }


def summarize() -> dict[str, Any]:
    candidates = []
    for config in CONFIGS:
        rows = [row("scan", config, seed) for seed in SEEDS]
        validations = [item["validation"] for item in rows]
        lcbs = [metric(value, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound") for value in validations]
        key = (
            sum(item["safe"] for item in rows),
            min(lcbs),
            sum(lcbs) / len(lcbs),
            sum(metric(value, "aggregateWeightedRecall", "agg_weighted_recall") for value in validations) / len(validations),
            sum(metric(value, "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall") for value in validations) / len(validations),
        )
        candidates.append({"config": config.name, "selectionKey": list(key), "rows": rows})
    selected = max(candidates, key=lambda item: tuple(item["selectionKey"]))
    payload = {
        "schema": "pvs-sponza-recall-first-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "selectedConfig": selected["config"],
        "candidates": candidates,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "scan_selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "scan", "summarize", "final"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    if not args.gpu_slots:
        parser.error("at least one GPU slot is required")
    if args.mode == "plan":
        print(json.dumps({
            "experiment": EXPERIMENT,
            "sources": {str(seed): str(path.resolve()) for seed, path in SOURCES.items()},
            "configs": [asdict(config) for config in CONFIGS],
            "scanUpdates": 900,
            "finalUpdates": 1800,
            "testRead": False,
        }, ensure_ascii=False, indent=2))
        return
    preflight(SCENE)
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source checkpoint(s): {missing}")
    if args.mode == "preflight":
        print(json.dumps({"experiment": EXPERIMENT, "status": "ready", "testRead": False}))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return
    jobs = []
    if args.mode == "scan":
        for seed in SEEDS:
            for config in CONFIGS:
                output = member_dir("scan", config, seed)
                if not completed(output):
                    jobs.append((f"{config.name}_seed{seed}", command(config, seed, output, final=False)))
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/scan")
        return
    selection = read_json(RESULT_ROOT / "scan_selection.json")
    config = next(item for item in CONFIGS if item.name == selection["selectedConfig"])
    for seed in SEEDS:
        output = member_dir("final", config, seed)
        if not completed(output):
            jobs.append((f"{config.name}_seed{seed}", command(config, seed, output, final=True)))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/final")


if __name__ == "__main__":
    main()
