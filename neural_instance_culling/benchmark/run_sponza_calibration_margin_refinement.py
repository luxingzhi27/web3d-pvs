#!/usr/bin/env python3
"""Tune a calibration-only safety margin for recall-first Sponza checkpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

try:
    from .run_standard_graphics_connected_sah_full import preflight, train_command
    from .standard_graphics_connected_sah_config import ROOT
except ImportError:
    from run_standard_graphics_connected_sah_full import preflight, train_command  # type: ignore
    from standard_graphics_connected_sah_config import ROOT  # type: ignore


EXPERIMENT = "pvs_v4_sponza_calibration_margin_refinement_v1"
SCENE = "sponza_128k"
SEEDS = (20260801, 20260802, 20260803)
SCAN_SEEDS = (20260802, 20260803)
SOURCE_ROOT = ROOT / "neural_instance_culling/model/out/pvs_v4_sponza_recall_first_refinement_v1/final"
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT


@dataclass(frozen=True)
class Margin:
    name: str
    floor: float


MARGINS = (Margin("margin995", 0.995), Margin("margin997", 0.997))


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source(seed: int) -> Path:
    member = SOURCE_ROOT / f"recall_lr2e6_seed{seed}"
    summary = read_json(member / "calibration_ready_summary.json")
    name = "best_safe.pt" if summary.get("status") == "safe" else "best_diagnostic.pt"
    return member / name


def replace_value(command: list[str], flag: str, value: str) -> None:
    if flag in command:
        command[command.index(flag) + 1] = value
    else:
        command.extend([flag, value])


def member_dir(stage: str, margin: Margin, seed: int) -> Path:
    return MODEL_ROOT / stage / f"{margin.name}_seed{seed}"


def completed(path: Path) -> bool:
    summary = path / "calibration_ready_summary.json"
    return summary.is_file() and read_json(summary).get("testRead") is False


def build_command(margin: Margin, seed: int, output: Path, *, final: bool) -> list[str]:
    result = train_command(SCENE, output, seed, smoke=False)
    epochs = 2
    steps = 450 if final else 300
    values = {
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{margin.name}_seed{seed}",
        "--epochs": str(epochs),
        "--steps-per-epoch": str(steps),
        "--poses-per-batch": "16",
        "--hard-pose-fraction": "0.75",
        "--hard-pose-quantile": "0.40",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--calibration-weighted-recall-floor": str(margin.floor),
        "--learning-rate": "0.0000005",
        "--survival-loss-weight": "0.05",
        "--relation-consistency-weight": "0.02",
        "--integrated-rvl-recall-guard-weight": "2.0",
        "--integrated-rvl-recall-target": "0.9995",
        "--integrated-rvl-pose-cvar-fraction": "0.75",
        "--integrated-rvl-pose-cvar-weight": "1.0",
        "--integrated-separation-weight": "0.01",
        "--integrated-rvl-recall-guard-zero-fraction": "0",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.01",
        "--integrated-rvl-recall-guard-middle-scale": "1",
        "--integrated-tail-zero-fraction": "0",
        "--integrated-tail-ramp-fraction": "0.01",
        "--frontier-positive-mass-fraction": "0.05",
        "--frontier-positive-count-cap": "512",
        "--frontier-negative-fraction": "0.0005",
        "--frontier-negative-count-cap": "32",
    }
    for flag, value in values.items():
        replace_value(result, flag, value)
    result.extend([
        "--training-course-total-steps", str(epochs * steps),
        "--init-checkpoint", str(source(seed)),
        "--init-instance-calibration-blend", "1",
    ])
    return result


def run_one(name: str, command: list[str], gpu: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    write_json(log_root / f"{name}.command.json", {
        "schema": "pvs-sponza-calibration-margin-command-v1",
        "experiment": EXPERIMENT,
        "name": name,
        "gpuId": gpu,
        "command": command,
        "testRead": False,
    })
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_parallel(jobs: list[tuple[str, list[str]]], gpu_slots: Sequence[int], log_root: Path) -> None:
    for start in range(0, len(jobs), len(gpu_slots)):
        batch = jobs[start:start + len(gpu_slots)]
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {
                executor.submit(run_one, name, command, gpu_slots[index], log_root): name
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


def metric(value: dict[str, Any], *names: str) -> float:
    for name in names:
        if value.get(name) is not None:
            return float(value[name])
    return 0.0


def summarize() -> dict[str, Any]:
    candidates = []
    for margin in MARGINS:
        rows = []
        for seed in SCAN_SEEDS:
            path = member_dir("scan", margin, seed)
            summary = read_json(path / "calibration_ready_summary.json")
            validation = summary.get("validationAtFrozenThreshold") or {}
            wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
            lcb = metric(validation, "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound")
            rows.append({
                "seed": seed,
                "safe": summary.get("status") == "safe" and wr > 0.99 and lcb > 0.99,
                "member": str(path.resolve()),
                "validation": validation,
                "testRead": False,
            })
        lcbs = [metric(row["validation"], "aggregateWeightedRecallLowerConfidenceBound", "weighted_recall_lower_confidence_bound") for row in rows]
        safe_count = sum(row["safe"] for row in rows)
        mean_useful = sum(metric(row["validation"], "agg_useful_cull") for row in rows) / len(rows)
        mean_cnor = sum(metric(row["validation"], "candidateNormalizedOcclusionRecall", "candidate_normalized_occlusion_recall") for row in rows) / len(rows)
        key = (
            safe_count,
            int(safe_count == len(rows)),
            mean_useful if safe_count == len(rows) else min(lcbs),
            mean_cnor if safe_count == len(rows) else mean_useful,
            min(lcbs),
        )
        candidates.append({"margin": margin.name, "selectionKey": list(key), "rows": rows})
    selected = max(candidates, key=lambda item: tuple(item["selectionKey"]))
    payload = {
        "schema": "pvs-sponza-calibration-margin-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "selectedMargin": selected["margin"],
        "candidates": candidates,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "scan_selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "scan", "summarize", "final"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    preflight(SCENE)
    missing = [str(source(seed)) for seed in SEEDS if not source(seed).is_file()]
    if missing:
        raise FileNotFoundError(f"missing source checkpoint(s): {missing}")
    if args.mode == "preflight":
        print(json.dumps({"experiment": EXPERIMENT, "status": "ready", "testRead": False}))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return
    if not args.gpu_slots:
        parser.error("at least one GPU slot is required")
    if args.mode == "scan":
        jobs = []
        for seed in SCAN_SEEDS:
            for margin in MARGINS:
                output = member_dir("scan", margin, seed)
                if not completed(output):
                    jobs.append((f"{margin.name}_seed{seed}", build_command(margin, seed, output, final=False)))
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/scan")
        return
    selection = read_json(RESULT_ROOT / "scan_selection.json")
    margin = next(item for item in MARGINS if item.name == selection["selectedMargin"])
    jobs = []
    for seed in SEEDS:
        output = member_dir("final", margin, seed)
        if not completed(output):
            jobs.append((f"{margin.name}_seed{seed}", build_command(margin, seed, output, final=True)))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/final")


if __name__ == "__main__":
    main()
