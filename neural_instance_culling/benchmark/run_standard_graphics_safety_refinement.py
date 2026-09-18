#!/usr/bin/env python3
"""Tune and apply a short safety-loss refinement for Sponza and Viking."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

try:
    from .run_standard_graphics_connected_sah_full import (
        member_dir as base_member_dir,
        preflight as base_preflight,
        train_command as base_train_command,
    )
    from .standard_graphics_connected_sah_config import ROOT, SEEDS
except ImportError:
    from run_standard_graphics_connected_sah_full import (  # type: ignore
        member_dir as base_member_dir,
        preflight as base_preflight,
        train_command as base_train_command,
    )
    from standard_graphics_connected_sah_config import ROOT, SEEDS  # type: ignore


EXPERIMENT = "pvs_v4_standard_graphics_safety_refinement_v1"
SCENES = ("sponza_128k", "viking_village_128k")
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT
PILOT_SOURCE = {
    "sponza_128k": base_member_dir(
        ROOT / "neural_instance_culling/model/out/pvs_v4_standard_graphics_connected_sah_128k_v1",
        "sponza_128k",
        20260803,
    ) / "best_diagnostic.pt",
    "viking_village_128k": base_member_dir(
        ROOT / "neural_instance_culling/model/out/pvs_v4_standard_graphics_connected_sah_128k_v1",
        "viking_village_128k",
        20260801,
    ) / "best_safe.pt",
}


@dataclass(frozen=True)
class LossConfig:
    name: str
    tail_weight: float
    guard_weight: float
    pose_cvar_weight: float
    survival_weight: float
    relation_weight: float


LOSS_CONFIGS = (
    LossConfig("r1_tail15_guard30", 0.15, 0.30, 0.25, 0.25, 0.10),
    LossConfig("r2_tail15_guard45", 0.15, 0.45, 0.35, 0.25, 0.10),
    LossConfig("r3_tail10_guard45_auxlight", 0.10, 0.45, 0.35, 0.15, 0.05),
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


def pilot_member(scene: str, config: LossConfig) -> Path:
    return MODEL_ROOT / "pilot" / scene / config.name


def final_member(scene: str, config: LossConfig, seed: int) -> Path:
    return MODEL_ROOT / "final" / scene / f"{config.name}_seed{seed}_e4"


def refinement_command(
    scene: str,
    config: LossConfig,
    output: Path,
    source: Path,
    seed: int,
    *,
    final: bool,
) -> list[str]:
    command = base_train_command(scene, output, seed, smoke=False)
    steps_per_epoch = 900 if final else 450
    values = {
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{scene}_{config.name}_seed{seed}",
        "--epochs": "4",
        "--steps-per-epoch": str(steps_per_epoch),
        "--eval-every": "1",
        "--snapshot-every": "2" if final else "1",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--learning-rate": "0.00001",
        "--survival-loss-weight": str(config.survival_weight),
        "--relation-consistency-weight": str(config.relation_weight),
        "--integrated-rvl-recall-guard-weight": str(config.guard_weight),
        "--integrated-rvl-pose-cvar-weight": str(config.pose_cvar_weight),
        "--integrated-separation-weight": str(config.tail_weight),
        "--integrated-rvl-recall-guard-zero-fraction": "0",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.05",
        "--integrated-rvl-recall-guard-middle-scale": "1",
        "--integrated-tail-zero-fraction": "0",
        "--integrated-tail-ramp-fraction": "0.05",
    }
    for flag, value in values.items():
        replace_value(command, flag, value)
    command.extend([
        "--training-course-total-steps",
        str(4 * steps_per_epoch),
        "--init-checkpoint",
        str(source),
    ])
    return command


def completed(member: Path) -> bool:
    summary = member / "calibration_ready_summary.json"
    if not summary.is_file():
        return False
    value = read_json(summary)
    return value.get("testRead") is False and value.get("status") in {
        "safe", "no_qualified_safety_workpoint"
    }


def run_logged(name: str, command: list[str], gpu_id: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    record_path = log_root / f"{name}.command.json"
    record = {
        "schema": "pvs-standard-graphics-safety-refinement-command-v1",
        "experiment": EXPERIMENT,
        "name": name,
        "gpuId": gpu_id,
        "command": command,
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "testRead": False,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(record_path, record)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    record["returnCode"] = int(result.returncode)
    record["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(record_path, record)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_parallel(jobs: list[tuple[str, list[str]]], gpu_slots: Sequence[int], log_root: Path) -> None:
    for start in range(0, len(jobs), len(gpu_slots)):
        batch = jobs[start:start + len(gpu_slots)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(run_logged, name, command, int(gpu_slots[index]), log_root): name
                for index, (name, command) in enumerate(batch)
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


def result_row(scene: str, config: LossConfig) -> dict[str, Any]:
    member = pilot_member(scene, config)
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    lcb = metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound")
    return {
        "scene": scene,
        "config": config.name,
        "member": str(member.resolve()),
        "safe": summary.get("status") == "safe" and wr > 0.99 and lcb > 0.99,
        "validation": validation,
        "testRead": False,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    validation = row["validation"]
    if row["safe"]:
        return (
            1.0,
            metric(validation, "agg_useful_cull"),
            metric(validation, "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall"),
            metric(validation, "agg_balanced_accuracy"),
            -metric(validation, "avg_pred_count"),
        )
    return (
        0.0,
        metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound"),
        metric(validation, "aggregateWeightedRecall", "agg_weighted_recall"),
        metric(validation, "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall"),
        metric(validation, "agg_useful_cull"),
    )


def summarize() -> dict[str, Any]:
    selections = {}
    for scene in SCENES:
        rows = [result_row(scene, config) for config in LOSS_CONFIGS]
        selected = max(rows, key=selection_key)
        selections[scene] = {
            "selectedConfig": selected["config"],
            "selectionPool": "safe" if any(row["safe"] for row in rows) else "diagnostic",
            "rows": rows,
        }
    payload = {
        "schema": "pvs-standard-graphics-safety-refinement-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "selections": selections,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "pilot_selection.json", payload)
    return payload


def selected_config(scene: str) -> LossConfig:
    name = read_json(RESULT_ROOT / "pilot_selection.json")["selections"][scene]["selectedConfig"]
    return next(config for config in LOSS_CONFIGS if config.name == name)


def source_for_final(scene: str, seed: int) -> Path:
    member = base_member_dir(
        ROOT / "neural_instance_culling/model/out/pvs_v4_standard_graphics_connected_sah_128k_v1",
        scene,
        seed,
    )
    summary = read_json(member / "calibration_ready_summary.json")
    return member / ("best_safe.pt" if summary.get("status") == "safe" else "best_diagnostic.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "pilot", "summarize", "final"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 1, 1])
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    args = parser.parse_args()
    if not args.gpu_slots:
        parser.error("at least one GPU slot is required")

    if args.mode == "plan":
        payload = {
            "experiment": EXPERIMENT,
            "scenes": list(args.scenes),
            "configs": [config.__dict__ for config in LOSS_CONFIGS],
            "pilotSteps": 1800,
            "finalSteps": 3600,
            "testRead": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for scene in args.scenes:
        base_preflight(scene)
        if not PILOT_SOURCE[scene].is_file():
            raise FileNotFoundError(PILOT_SOURCE[scene])
    if args.mode == "preflight":
        print(json.dumps({"experiment": EXPERIMENT, "status": "ready", "testRead": False}))
        return
    if args.mode == "pilot":
        jobs = []
        for config in LOSS_CONFIGS:
            for scene in args.scenes:
                output = pilot_member(scene, config)
                if not completed(output):
                    jobs.append((
                        f"{scene}_{config.name}",
                        refinement_command(scene, config, output, PILOT_SOURCE[scene], 20260801, final=False),
                    ))
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/pilot")
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return

    jobs = []
    for scene in args.scenes:
        config = selected_config(scene)
        for seed in SEEDS:
            source = source_for_final(scene, seed)
            output = final_member(scene, config, seed)
            if not completed(output):
                jobs.append((
                    f"{scene}_{config.name}_seed{seed}",
                    refinement_command(scene, config, output, source, seed, final=True),
                ))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/final")


if __name__ == "__main__":
    main()
