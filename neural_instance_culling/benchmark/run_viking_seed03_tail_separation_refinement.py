#!/usr/bin/env python3
"""Improve Viking seed03 safety by separating positive and negative score tails."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any

try:
    from .run_standard_graphics_connected_sah_full import preflight, train_command
    from .standard_graphics_connected_sah_config import ROOT
except ImportError:
    from run_standard_graphics_connected_sah_full import preflight, train_command  # type: ignore
    from standard_graphics_connected_sah_config import ROOT  # type: ignore


EXPERIMENT = "pvs_v4_viking_seed03_tail_separation_refinement_v1"
SCENE = "viking_village_128k"
SEED = 20260803
SOURCE = (
    ROOT
    / "neural_instance_culling/model/out/pvs_v4_standard_graphics_checkpoint_rescue_v1"
    / "final/viking_village_128k/tailprotected_lr5e6_seed20260803/best_diagnostic.pt"
)
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT


@dataclass(frozen=True)
class Config:
    name: str
    learning_rate: float
    guard_weight: float
    recall_target: float
    pose_cvar_weight: float
    separation_weight: float


CONFIGS = (
    Config("tailrank_lr2e7", 2e-7, 0.10, 0.995, 0.25, 1.0),
    Config("tailrank_lr5e7", 5e-7, 0.10, 0.995, 0.25, 1.0),
    Config("balanced_guard050", 2e-7, 0.50, 0.997, 0.50, 0.30),
    Config("balanced_guard080", 2e-7, 0.80, 0.998, 0.75, 0.25),
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def set_value(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def member(stage: str, config: Config) -> Path:
    return MODEL_ROOT / stage / config.name


def completed(path: Path) -> bool:
    summary = path / "calibration_ready_summary.json"
    return summary.is_file() and read_json(summary).get("testRead") is False


def build_command(config: Config, *, final: bool) -> list[str]:
    output = member("final" if final else "scan", config)
    command = train_command(SCENE, output, SEED, smoke=False)
    epochs, steps = (4, 450) if final else (2, 450)
    values = {
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{config.name}",
        "--epochs": str(epochs),
        "--steps-per-epoch": str(steps),
        "--poses-per-batch": "16",
        "--hard-pose-fraction": "0.75",
        "--hard-pose-quantile": "0.40",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--calibration-weighted-recall-floor": "0.99",
        "--learning-rate": str(config.learning_rate),
        "--survival-loss-weight": "0.10",
        "--relation-consistency-weight": "0.05",
        "--integrated-rvl-recall-guard-weight": str(config.guard_weight),
        "--integrated-rvl-recall-target": str(config.recall_target),
        "--integrated-rvl-pose-cvar-fraction": "0.50",
        "--integrated-rvl-pose-cvar-weight": str(config.pose_cvar_weight),
        "--integrated-separation-weight": str(config.separation_weight),
        "--integrated-rvl-recall-guard-zero-fraction": "0",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.01",
        "--integrated-rvl-recall-guard-middle-scale": "1",
        "--integrated-tail-zero-fraction": "0",
        "--integrated-tail-ramp-fraction": "0.01",
        "--frontier-positive-mass-fraction": "0.03",
        "--frontier-positive-count-cap": "256",
        "--frontier-negative-fraction": "0.02",
        "--frontier-negative-count-cap": "256",
    }
    for flag, value in values.items():
        set_value(command, flag, value)
    command.extend([
        "--training-course-total-steps", str(epochs * steps),
        "--init-checkpoint", str(SOURCE),
        "--init-instance-calibration-blend", "1",
    ])
    return command


def run_one(config: Config, gpu: int, *, final: bool) -> None:
    stage = "final" if final else "scan"
    output = member(stage, config)
    if completed(output):
        return
    command = build_command(config, final=final)
    log_root = RESULT_ROOT / "logs" / stage
    log_root.mkdir(parents=True, exist_ok=True)
    write_json(log_root / f"{config.name}.command.json", {
        "schema": "pvs-viking-seed03-tail-separation-command-v1",
        "experiment": EXPERIMENT,
        "config": config.name,
        "gpuId": gpu,
        "command": command,
        "testRead": False,
    })
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with (log_root / f"{config.name}.stdout.log").open("w", encoding="utf-8") as stdout, (
        log_root / f"{config.name}.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def metric(value: dict[str, Any], *names: str) -> float:
    for name in names:
        if value.get(name) is not None:
            return float(value[name])
    return 0.0


def summarize() -> dict[str, Any]:
    rows = []
    for config in CONFIGS:
        summary = read_json(member("scan", config) / "calibration_ready_summary.json")
        validation = summary.get("validationAtFrozenThreshold") or {}
        wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
        lcb = metric(
            validation,
            "aggregateWeightedRecallLowerConfidenceBound",
            "weighted_recall_lower_confidence_bound",
        )
        cnor = metric(
            validation,
            "candidateNormalizedOcclusionRecall",
            "candidate_normalized_occlusion_recall",
        )
        safe = summary.get("status") == "safe" and wr > 0.99 and lcb > 0.99
        rows.append({
            "config": config.name,
            "safe": safe,
            "qualified": safe and cnor >= 0.80,
            "validation": validation,
            "testRead": False,
        })
    selected = max(rows, key=lambda row: (
        int(row["qualified"]),
        int(row["safe"]),
        metric(row["validation"], "candidate_normalized_occlusion_recall"),
        metric(row["validation"], "agg_useful_cull"),
        metric(row["validation"], "aggregateWeightedRecallLowerConfidenceBound"),
    ))
    payload = {
        "schema": "pvs-viking-seed03-tail-separation-selection-v1",
        "experiment": EXPERIMENT,
        "selectedConfig": selected["config"],
        "acceptance": {
            "validationWeightedRecallLowerConfidenceBound": 0.99,
            "validationCandidateNormalizedOcclusionRecall": 0.80,
        },
        "rows": rows,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "scan_selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "scan", "summarize", "final"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    preflight(SCENE)
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    if args.mode == "preflight":
        print(json.dumps({
            "experiment": EXPERIMENT,
            "configs": [config.name for config in CONFIGS],
            "testRead": False,
        }))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return
    if not args.gpu_slots:
        parser.error("at least one GPU slot is required")
    if args.mode == "scan":
        with ThreadPoolExecutor(max_workers=len(CONFIGS)) as executor:
            futures = {
                executor.submit(run_one, config, int(args.gpu_slots[index]), final=False): config.name
                for index, config in enumerate(CONFIGS)
            }
            errors = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
            if errors:
                raise RuntimeError("; ".join(errors))
        return
    selection = read_json(RESULT_ROOT / "scan_selection.json")
    config = next(item for item in CONFIGS if item.name == selection["selectedConfig"])
    run_one(config, int(args.gpu_slots[0]), final=True)


if __name__ == "__main__":
    main()
