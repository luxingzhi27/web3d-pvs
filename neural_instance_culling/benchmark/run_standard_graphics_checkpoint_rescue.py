#!/usr/bin/env python3
"""Continue Sponza and Viking from stable Full V4 checkpoints without reading test."""
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
    from .run_standard_graphics_connected_sah_full import (
        member_dir as base_member_dir,
        preflight as base_preflight,
        train_command as base_train_command,
    )
    from .standard_graphics_connected_sah_config import ROOT
except ImportError:
    from run_standard_graphics_connected_sah_full import (  # type: ignore
        member_dir as base_member_dir,
        preflight as base_preflight,
        train_command as base_train_command,
    )
    from standard_graphics_connected_sah_config import ROOT  # type: ignore


EXPERIMENT = "pvs_v4_standard_graphics_checkpoint_rescue_v1"
BASE_ROOT = ROOT / "neural_instance_culling/model/out/pvs_v4_standard_graphics_connected_sah_128k_v1"
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT

TARGETS = {
    ("sponza_128k", 20260801): BASE_ROOT / "sponza_128k/full_seed20260801_e40/checkpoint_epoch_024.pt",
    ("sponza_128k", 20260802): BASE_ROOT / "sponza_128k/full_seed20260802_e40/checkpoint_epoch_012.pt",
    ("sponza_128k", 20260803): BASE_ROOT / "sponza_128k/full_seed20260803_e40/checkpoint_epoch_016.pt",
    ("viking_village_128k", 20260803): BASE_ROOT / "viking_village_128k/full_seed20260803_e40/checkpoint_epoch_032.pt",
}


@dataclass(frozen=True)
class RescueConfig:
    name: str
    learning_rate: float
    guard_weight: float
    recall_target: float
    pose_cvar_fraction: float
    pose_cvar_weight: float
    separation_weight: float
    positive_mass_fraction: float
    positive_count_cap: int
    negative_fraction: float
    negative_count_cap: int


CONFIGS = (
    RescueConfig(
        "guarded_lr2e6", 2e-6, 0.75, 0.997, 0.35, 0.75, 0.15, 0.01, 128, 0.005, 128,
    ),
    RescueConfig(
        "tailprotected_lr5e6", 5e-6, 1.00, 0.998, 0.50, 1.00, 0.10, 0.02, 256, 0.0025, 64,
    ),
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


def member_dir(stage: str, scene: str, seed: int, config: RescueConfig) -> Path:
    return MODEL_ROOT / stage / scene / f"{config.name}_seed{seed}"


def completed(member: Path) -> bool:
    path = member / "calibration_ready_summary.json"
    if not path.is_file():
        return False
    value = read_json(path)
    return value.get("testRead") is False and value.get("status") in {
        "safe", "no_qualified_safety_workpoint",
    }


def rescue_command(
    scene: str,
    seed: int,
    config: RescueConfig,
    output: Path,
    source: Path,
    *,
    final: bool,
) -> list[str]:
    command = base_train_command(scene, output, seed, smoke=False)
    epochs = 4 if final else 2
    steps_per_epoch = 450
    values = {
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{scene}_{config.name}_seed{seed}",
        "--epochs": str(epochs),
        "--steps-per-epoch": str(steps_per_epoch),
        "--poses-per-batch": "8",
        "--hard-pose-fraction": "0.50",
        "--hard-pose-quantile": "0.50",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--learning-rate": str(config.learning_rate),
        "--survival-loss-weight": "0.10",
        "--relation-consistency-weight": "0.05",
        "--integrated-rvl-recall-guard-weight": str(config.guard_weight),
        "--integrated-rvl-recall-target": str(config.recall_target),
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
        replace_value(command, flag, value)
    command.extend([
        "--training-course-total-steps", str(epochs * steps_per_epoch),
        "--init-checkpoint", str(source),
    ])
    return command


def run_logged(name: str, command: list[str], gpu_id: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    record_path = log_root / f"{name}.command.json"
    record = {
        "schema": "pvs-standard-graphics-checkpoint-rescue-command-v1",
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


def result_row(stage: str, scene: str, seed: int, config: RescueConfig) -> dict[str, Any]:
    member = member_dir(stage, scene, seed, config)
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    lcb = metric(
        validation,
        "aggregateWeightedRecallLowerConfidenceBound",
        "weighted_recall_lower_confidence_bound",
    )
    return {
        "scene": scene,
        "seed": seed,
        "config": config.name,
        "member": str(member.resolve()),
        "safe": summary.get("status") == "safe" and wr > 0.99 and lcb > 0.99,
        "validation": validation,
        "testRead": False,
    }


def config_key(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    validations = [row["validation"] for row in rows]
    safe_count = sum(bool(row["safe"]) for row in rows)
    lcbs = [metric(v, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound") for v in validations]
    wrs = [metric(v, "aggregateWeightedRecall", "agg_weighted_recall") for v in validations]
    cnors = [metric(v, "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall") for v in validations]
    useful = [metric(v, "agg_useful_cull") for v in validations]
    return (
        float(safe_count),
        min(lcbs),
        sum(lcbs) / len(lcbs),
        sum(wrs) / len(wrs),
        sum(cnors) / len(cnors),
        sum(useful) / len(useful),
    )


def summarize_scan() -> dict[str, Any]:
    selections = {}
    for scene in ("sponza_128k", "viking_village_128k"):
        targets = [(target_scene, seed) for target_scene, seed in TARGETS if target_scene == scene]
        candidates = []
        for config in CONFIGS:
            rows = [result_row("scan", target_scene, seed, config) for target_scene, seed in targets]
            candidates.append({"config": config.name, "rows": rows, "selectionKey": list(config_key(rows))})
        selected = max(candidates, key=lambda item: tuple(item["selectionKey"]))
        selections[scene] = {"selectedConfig": selected["config"], "candidates": candidates}
    payload = {
        "schema": "pvs-standard-graphics-checkpoint-rescue-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "selections": selections,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "scan_selection.json", payload)
    return payload


def selected_config(scene: str) -> RescueConfig:
    name = read_json(RESULT_ROOT / "scan_selection.json")["selections"][scene]["selectedConfig"]
    return next(config for config in CONFIGS if config.name == name)


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
            "targets": {f"{scene}:seed{seed}": str(path.resolve()) for (scene, seed), path in TARGETS.items()},
            "configs": [asdict(config) for config in CONFIGS],
            "scanUpdates": 900,
            "finalUpdates": 1800,
            "testRead": False,
        }, ensure_ascii=False, indent=2))
        return

    for scene in sorted({scene for scene, _seed in TARGETS}):
        base_preflight(scene)
    missing = [str(path) for path in TARGETS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing rescue checkpoint(s): {missing}")
    if args.mode == "preflight":
        print(json.dumps({"experiment": EXPERIMENT, "status": "ready", "testRead": False}))
        return

    if args.mode == "scan":
        jobs = []
        for (scene, seed), source in TARGETS.items():
            for config in CONFIGS:
                output = member_dir("scan", scene, seed, config)
                if not completed(output):
                    name = f"{scene}_{config.name}_seed{seed}"
                    jobs.append((name, rescue_command(scene, seed, config, output, source, final=False)))
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/scan")
        return

    if args.mode == "summarize":
        print(json.dumps(summarize_scan(), ensure_ascii=False, indent=2))
        return

    jobs = []
    for (scene, seed), source in TARGETS.items():
        config = selected_config(scene)
        output = member_dir("final", scene, seed, config)
        if not completed(output):
            name = f"{scene}_{config.name}_seed{seed}"
            jobs.append((name, rescue_command(scene, seed, config, output, source, final=True)))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/final")


if __name__ == "__main__":
    main()
