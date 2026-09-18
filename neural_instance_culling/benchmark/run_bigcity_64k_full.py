#!/usr/bin/env python3
"""Train the three Full V4 seeds on the finalized Big City 64 KiB dataset."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
from typing import Sequence

try:
    from .run_standard_graphics_connected_sah_full import (
        is_valid_completed_calibration_summary,
        preflight,
        train_command,
    )
    from .standard_graphics_connected_sah_config import ROOT, SEEDS
except ImportError:
    from run_standard_graphics_connected_sah_full import (  # type: ignore
        is_valid_completed_calibration_summary,
        preflight,
        train_command,
    )
    from standard_graphics_connected_sah_config import ROOT, SEEDS  # type: ignore


EXPERIMENT = "pvs_v4_bigcity_connected_sah_64k_full_v1"
SCENE = "bigcity_64k"
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
LOG_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT / "logs"


def member(seed: int, *, smoke: bool = False) -> Path:
    name = f"smoke_seed{seed}_e1" if smoke else f"full_seed{seed}_e40"
    return MODEL_ROOT / name


def command(seed: int, *, smoke: bool = False) -> list[str]:
    output = member(seed, smoke=smoke)
    result = train_command(SCENE, output, seed, smoke=smoke)
    result[result.index("--experiment-name") + 1] = f"{EXPERIMENT}_seed{seed}"
    return result


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run_one(seed: int, gpu: int, *, smoke: bool = False) -> None:
    output = member(seed, smoke=smoke)
    if is_valid_completed_calibration_summary(output):
        return
    args = command(seed, smoke=smoke)
    name = f"{'smoke' if smoke else 'full'}_seed{seed}_gpu{gpu}"
    stdout_path = LOG_ROOT / f"{name}.stdout.log"
    stderr_path = LOG_ROOT / f"{name}.stderr.log"
    write_json(LOG_ROOT / f"{name}.command.json", {
        "schema": "pvs-bigcity-64k-full-command-v1",
        "experiment": EXPERIMENT,
        "scene": SCENE,
        "seed": seed,
        "gpuId": gpu,
        "command": args,
        "testRead": False,
    })
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(args, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, args)


def run_all(gpu_slots: Sequence[int], *, smoke: bool = False) -> None:
    selected_seeds = SEEDS[:1] if smoke else SEEDS
    with ThreadPoolExecutor(max_workers=len(selected_seeds)) as executor:
        futures = {
            executor.submit(run_one, seed, int(gpu_slots[index]), smoke=smoke): seed
            for index, seed in enumerate(selected_seeds)
        }
        errors = []
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                errors.append(f"seed{futures[future]}: {error}")
        if errors:
            raise RuntimeError("; ".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "smoke", "train"))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    if len(args.gpu_slots) < (1 if args.mode == "smoke" else len(SEEDS)):
        parser.error("one GPU slot per concurrent seed is required")
    if args.mode == "plan":
        print(json.dumps({
            "experiment": EXPERIMENT,
            "scene": SCENE,
            "seeds": list(SEEDS),
            "members": {str(seed): str(member(seed).resolve()) for seed in SEEDS},
            "testRead": False,
        }, ensure_ascii=False, indent=2))
        return
    report = preflight(SCENE)
    report["experiment"] = EXPERIMENT
    if args.mode == "preflight":
        print(json.dumps({"experiment": EXPERIMENT, "report": report, "testRead": False}, ensure_ascii=False, indent=2))
        return
    run_all(args.gpu_slots, smoke=args.mode == "smoke")


if __name__ == "__main__":
    main()
