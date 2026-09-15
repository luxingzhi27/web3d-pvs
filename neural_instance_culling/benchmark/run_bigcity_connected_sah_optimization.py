#!/usr/bin/env python3
"""Run the registered Big City occlusion-opportunity optimization.

The runner is train/calibration/validation only. It never evaluates test.
"""
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
        preflight as full_preflight,
        train_command as base_train_command,
    )
    from .standard_graphics_connected_sah_config import ROOT, SCENES, SEEDS
except ImportError:
    from run_standard_graphics_connected_sah_full import (  # type: ignore
        preflight as full_preflight,
        train_command as base_train_command,
    )
    from standard_graphics_connected_sah_config import ROOT, SCENES, SEEDS  # type: ignore


SCENE = "bigcity_128k"
EXPERIMENT = "pvs_v4_bigcity_connected_sah_occlusion_opportunity_v1"
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT
RELATION_BUILDER = ROOT / "neural_instance_culling/dataset/build_observed_relation_csr.py"
RELATION_K_VALUES = (8, 16, 24)

PILOT_EPOCHS = 8
PILOT_STEPS = 450
FULL_EPOCHS = 40
FULL_STEPS = 900
FULL_COURSE_STEPS = FULL_EPOCHS * FULL_STEPS


@dataclass(frozen=True)
class PilotConfig:
    name: str
    relation_k: int
    learning_rate: float
    negative_only_fraction: float
    hard_fraction: float


PILOT_CONFIGS = (
    PilotConfig("p0_k8_lr5e5_neg0", 8, 5e-5, 0.000, 0.350),
    PilotConfig("p1_k16_lr5e5_neg0", 16, 5e-5, 0.000, 0.350),
    PilotConfig("p2_k8_lr5e5_neg25", 8, 5e-5, 0.250, 0.375),
    PilotConfig("p3_k16_lr5e5_neg25", 16, 5e-5, 0.250, 0.375),
    PilotConfig("p4_k16_lr2e5_neg25", 16, 2e-5, 0.250, 0.375),
    PilotConfig("p5_k24_lr5e5_neg25", 24, 5e-5, 0.250, 0.375),
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


def relation_dir(relation_k: int) -> Path:
    scene_root = Path(SCENES[SCENE]["scene_root"])
    if relation_k == 8:
        return scene_root / "relation_csr"
    return scene_root / f"relation_csr_k{relation_k}"


def relation_command(relation_k: int) -> list[str]:
    config = SCENES[SCENE]
    return [
        sys.executable,
        "-u",
        str(RELATION_BUILDER),
        "--dataset-dir",
        str(config["dataset"]),
        "--runtime-meta",
        str(config["runtime_meta"]),
        "--sparse-cache-root",
        str(config["depth_cache"]),
        "--output-dir",
        str(relation_dir(relation_k)),
        "--splits",
        "train",
        "--source-k",
        str(relation_k),
    ]


def replace_value(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def member_dir(config: PilotConfig, *, full_seed: int | None = None) -> Path:
    if full_seed is None:
        return MODEL_ROOT / "pilot" / config.name
    return MODEL_ROOT / "full" / f"{config.name}_seed{full_seed}_e{FULL_EPOCHS}"


def train_command(
    config: PilotConfig,
    output: Path,
    seed: int,
    *,
    full: bool,
    smoke: bool = False,
) -> list[str]:
    command = base_train_command(SCENE, output, seed, smoke=False)
    values = {
        "--relation-dir": str(relation_dir(config.relation_k)),
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{config.name}_seed{seed}",
        "--epochs": str(1 if smoke else FULL_EPOCHS if full else PILOT_EPOCHS),
        "--steps-per-epoch": str(2 if smoke else FULL_STEPS if full else PILOT_STEPS),
        "--eval-every": str(1 if smoke else 4 if full else 2),
        "--snapshot-every": str(1 if smoke else 4 if full else 2),
        "--max-eval-poses": str(2 if smoke else 0),
        "--calibration-bootstrap-replicates": str(2 if smoke else 10000 if full else 2000),
        "--learning-rate": str(config.learning_rate),
        "--hard-pose-fraction": str(config.hard_fraction),
        "--integrated-rvl-recall-guard-zero-fraction": "0.10",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.30",
        "--integrated-rvl-recall-guard-middle-scale": "0.25",
        "--integrated-tail-zero-fraction": "0.10",
        "--integrated-tail-ramp-fraction": "0.30",
        "--frontier-negative-fraction": "0.04",
    }
    for flag, value in values.items():
        replace_value(command, flag, value)
    command.extend([
        "--training-course-total-steps",
        str(FULL_COURSE_STEPS),
        "--negative-only-pose-fraction",
        str(config.negative_only_fraction),
    ])
    return command


def relation_preflight(relation_k: int) -> dict[str, Any]:
    path = relation_dir(relation_k)
    metadata = read_json(path / "relation_csr_meta.json")
    evidence = metadata.get("evidenceTopK") or {}
    diagnostics = evidence.get("diagnostics") or {}
    if int(evidence.get("k", -1)) != relation_k:
        raise ValueError(f"relation K mismatch in {path}")
    if metadata.get("trainOnly") is not True:
        raise ValueError(f"relation is not train-only: {path}")
    return {
        "relationK": relation_k,
        "path": str(path.resolve()),
        "retainedQualityQuantiles": diagnostics.get("retainedQualityQuantiles"),
        "truncatedCellCount": diagnostics.get("truncatedCellCount"),
        "cellCount": diagnostics.get("cellCount"),
        "testRead": False,
    }


def completed(member: Path) -> bool:
    path = member / "calibration_ready_summary.json"
    if not path.is_file():
        return False
    summary = read_json(path)
    return summary.get("status") in {"safe", "no_qualified_safety_workpoint"} and summary.get("testRead") is False


def run_logged(name: str, command: list[str], gpu_id: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    record_path = log_root / f"{name}.command.json"
    record = {
        "schema": "pvs-bigcity-occlusion-opportunity-command-v1",
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
    if not gpu_slots:
        raise ValueError("at least one GPU slot is required")
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


def metric(validation: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = validation.get(key)
        if value is not None:
            return float(value)
    return 0.0


def pilot_row(config: PilotConfig) -> dict[str, Any]:
    member = member_dir(config)
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    lcb = metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound")
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    safe = summary.get("status") == "safe" and lcb > 0.99 and wr > 0.99
    return {
        "name": config.name,
        "relationK": config.relation_k,
        "learningRate": config.learning_rate,
        "negativeOnlyPoseFraction": config.negative_only_fraction,
        "hardPoseFraction": config.hard_fraction,
        "member": str(member.resolve()),
        "status": summary.get("status"),
        "safe": safe,
        "validation": validation,
        "testRead": False,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    validation = row["validation"]
    common = (
        metric(validation, "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall"),
        metric(validation, "agg_useful_cull"),
        metric(validation, "agg_balanced_accuracy"),
        metric(validation, "agg_specificity"),
        -metric(validation, "avg_pred_count"),
    )
    if row["safe"]:
        return (1.0, common[1], common[0], common[2], common[3], common[4])
    return (
        0.0,
        metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound"),
        metric(validation, "aggregateWeightedRecall", "agg_weighted_recall"),
        *common,
    )


def summarize() -> dict[str, Any]:
    rows = [pilot_row(config) for config in PILOT_CONFIGS]
    selected = max(rows, key=selection_key)
    payload = {
        "schema": "pvs-bigcity-occlusion-opportunity-pilot-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "selectedConfig": selected["name"],
        "selectionPool": "safe" if any(row["safe"] for row in rows) else "diagnostic",
        "rows": rows,
        "partitionRebuildRequired": (
            not selected["safe"]
            or metric(selected["validation"], "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall") < 0.50
        ),
        "testRead": False,
    }
    write_json(RESULT_ROOT / "pilot_selection.json", payload)
    return payload


def selected_config() -> PilotConfig:
    selection = read_json(RESULT_ROOT / "pilot_selection.json")
    name = str(selection["selectedConfig"])
    return next(config for config in PILOT_CONFIGS if config.name == name)


def plan() -> dict[str, Any]:
    return {
        "schema": "pvs-bigcity-occlusion-opportunity-plan-v1",
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "relations": [relation_command(k) for k in RELATION_K_VALUES],
        "pilots": [
            {
                "config": config.__dict__,
                "member": str(member_dir(config).resolve()),
                "command": train_command(config, member_dir(config), SEEDS[0], full=False),
            }
            for config in PILOT_CONFIGS
        ],
        "fullSeeds": list(SEEDS),
        "testRead": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "build-relations", "preflight", "smoke", "pilot", "summarize", "full"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[0, 3])
    args = parser.parse_args()

    if args.mode == "plan":
        print(json.dumps(plan(), ensure_ascii=False, indent=2))
        return
    full_preflight(SCENE)
    if args.mode == "build-relations":
        for relation_k in RELATION_K_VALUES:
            if relation_k == 8:
                continue
            if (relation_dir(relation_k) / "relation_csr_meta.json").is_file():
                continue
            run_logged(
                f"relation_k{relation_k}",
                relation_command(relation_k),
                int(args.gpu_slots[0]),
                RESULT_ROOT / "logs/relation",
            )
        return
    relation_reports = [relation_preflight(k) for k in RELATION_K_VALUES]
    if args.mode == "preflight":
        print(json.dumps({"relations": relation_reports, "testRead": False}, ensure_ascii=False, indent=2))
        return
    if args.mode == "smoke":
        config = next(item for item in PILOT_CONFIGS if item.negative_only_fraction > 0)
        output = MODEL_ROOT / "smoke" / config.name
        run_parallel(
            [(config.name, train_command(config, output, SEEDS[0], full=False, smoke=True))],
            args.gpu_slots,
            RESULT_ROOT / "logs/smoke",
        )
        return
    if args.mode == "pilot":
        jobs = [
            (config.name, train_command(config, member_dir(config), SEEDS[0], full=False))
            for config in PILOT_CONFIGS
            if not completed(member_dir(config))
        ]
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/pilot")
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return

    config = selected_config()
    jobs = []
    for seed in SEEDS:
        output = member_dir(config, full_seed=seed)
        if not completed(output):
            jobs.append((f"{config.name}_seed{seed}", train_command(config, output, seed, full=True)))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/full")


if __name__ == "__main__":
    main()
