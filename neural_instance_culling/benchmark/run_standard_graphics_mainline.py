#!/usr/bin/env python3
"""Run the fixed Full V4 protocol on registered standard graphics scenes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs.py"
SEEDS = (20260801, 20260802, 20260803)
EPOCHS = 40
STEPS_PER_EPOCH = 900
WEIGHTED_RECALL_FLOOR = 0.99

SCENES: dict[str, dict[str, Any]] = {
    "sponza_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_sponza_standard_graphics_128k_fov66_v1",
        "relation": "neural_instance_culling/dataset/out/sponza_standard_graphics_v4_bounded_relation_csr_v1",
        "runtime_meta": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/runtimeVisibilityMeta.json",
        "geometry": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/instance_geo_features_fp16.bin",
        "glb_index": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/glbIndex.json",
        "glb_root": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets",
        "splits": {"train": 1752, "validation": 240, "calibration": 192, "test": 240, "guard": 0},
    },
    "bigcity_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_bigcity_standard_graphics_128k_fov66_v1",
        "relation": "neural_instance_culling/dataset/out/bigcity_standard_graphics_v4_bounded_relation_csr_v1",
        "runtime_meta": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/runtimeVisibilityMeta.json",
        "geometry": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/instance_geo_features_fp16.bin",
        "glb_index": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/glbIndex.json",
        "glb_root": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets",
        "splits": {"train": 5832, "validation": 804, "calibration": 648, "test": 804, "guard": 0},
    },
    "viking_village_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_viking_village_standard_graphics_128k_fov66_v1",
        "relation": "neural_instance_culling/dataset/out/viking_village_standard_graphics_v4_bounded_relation_csr_v1",
        "runtime_meta": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/runtimeVisibilityMeta.json",
        "geometry": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/instance_geo_features_fp16.bin",
        "glb_index": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/glbIndex.json",
        "glb_root": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets",
        "splits": {"train": 1104, "validation": 156, "calibration": 120, "test": 156, "guard": 0},
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def scene_paths(scene: str) -> dict[str, Path]:
    config = SCENES[scene]
    return {
        key: ROOT / str(value)
        for key, value in config.items()
        if key != "splits"
    }


def preflight(scene: str) -> dict[str, Any]:
    paths = scene_paths(scene)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing {scene} mainline input(s): {missing}")
    dataset_meta = read_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError(f"{scene} must use the explicit four-way Pose CSR schema")
    split_counts = {
        str(name): int(count)
        for name, count in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != SCENES[scene]["splits"]:
        raise ValueError(
            f"{scene} split changed: expected={SCENES[scene]['splits']}, actual={split_counts}"
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
            raise ValueError(f"{scene} dataset is missing {field}")
    relation_meta = read_json(paths["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3":
        raise ValueError(f"{scene} relation artifact has the wrong schema")
    return {
        "schema": "pvs-standard-graphics-mainline-preflight-v1",
        "scene": scene,
        "splitCounts": split_counts,
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "testRead": False,
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }


def member_dir(model_root: Path, scene: str, seed: int) -> Path:
    return model_root / scene / f"full_seed{seed}_e{EPOCHS}"


def train_command(scene: str, output: Path, seed: int, *, smoke: bool) -> list[str]:
    p = scene_paths(scene)
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
        "--experiment-name", f"pvs_mainline_v4_{scene}_seed{seed}",
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
    calibration = read_json(member / "calibration_ready_summary.json")
    checkpoint = member / (
        "best_safe.pt"
        if calibration.get("status") == "safe" and calibration.get("bestSafe") is not None
        else "best_diagnostic.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def evaluate_command(
    scene: str,
    member: Path,
    output: Path,
    seed: int,
    split: str,
    sidecar: Path | None = None,
) -> list[str]:
    p = scene_paths(scene)
    command = [
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
        "--split", split,
        "--poses-per-batch", "2",
        "--seed", str(seed),
        "--device", "cuda",
        "--persist-ids",
        "--scene-name", scene,
        "--method-name", "full_v4",
    ]
    if split != "test":
        command.append("--allow-unsafe-diagnostic")
    else:
        command.extend(["--persist-scores", "--sidecar-dir", str(sidecar)])
    return command


def run_jobs(jobs: Sequence[tuple[str, list[str]]], gpu_ids: Sequence[int], log_dir: Path) -> None:
    if not jobs:
        return
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(index: int, name: str, command: list[str]) -> tuple[str, int, float]:
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[index % len(gpu_ids)])
        started = time.time()
        with (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
            log_dir / f"{name}.stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
        return name, int(result.returncode), float(time.time() - started)

    with ThreadPoolExecutor(max_workers=min(len(jobs), len(gpu_ids))) as executor:
        futures = {
            executor.submit(run, index, name, command): name
            for index, (name, command) in enumerate(jobs)
        }
        for future in as_completed(futures):
            name, return_code, elapsed = future.result()
            print(json.dumps({"job": name, "returnCode": return_code, "elapsedSeconds": elapsed}), flush=True)
            if return_code != 0:
                raise RuntimeError(f"standard graphics job failed: {name}; see {log_dir}")


def validation_safe(payload: dict[str, Any]) -> bool:
    weighted_recall = float(payload.get("aggregateWeightedRecall", math.nan))
    lower = float(payload.get("aggregateWeightedRecallLowerConfidenceBound", math.nan))
    return weighted_recall > WEIGHTED_RECALL_FLOOR and lower > WEIGHTED_RECALL_FLOOR


def validation_key(payload: dict[str, Any], seed: int) -> tuple[float, ...]:
    aggregate = payload["aggregate"]
    return (
        float(aggregate["usefulCull"]),
        float(aggregate["balancedAccuracy"]),
        float(aggregate["specificity"]),
        float(aggregate["precision"]),
        float(payload["aggregateWeightedRecallLowerConfidenceBound"]),
        -float(aggregate["avgPredCount"]),
        -float(seed),
    )


def select_validation_member(scene: str, model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    rows = []
    for seed in SEEDS:
        member = member_dir(model_root, scene, seed)
        validation_path = benchmark_root / scene / "validation" / f"seed{seed}.json"
        validation = read_json(validation_path)
        calibration = read_json(member / "calibration_ready_summary.json")
        rows.append(
            {
                "seed": seed,
                "member": member,
                "checkpoint": selected_checkpoint(member),
                "calibration": member / "calibration_ready_summary.json",
                "calibrationStatus": calibration.get("status"),
                "validation": validation,
            }
        )
    safe = [row for row in rows if row["calibrationStatus"] == "safe" and validation_safe(row["validation"])]
    if not safe:
        raise RuntimeError(f"{scene} has no calibration-safe and validation-safe Full V4 member")
    selected = max(safe, key=lambda row: validation_key(row["validation"], row["seed"]))
    summary = {
        "schema": "pvs-standard-graphics-validation-selection-v1",
        "scene": scene,
        "selectionSplit": "validation",
        "rule": "safe pool; useful cull, balanced accuracy, occlusion recall, precision, WR LCB, fewer predictions",
        "selectedSeed": selected["seed"],
        "safePoolSeeds": [row["seed"] for row in safe],
        "rows": [
            {
                "seed": row["seed"],
                "checkpoint": str(row["checkpoint"].resolve()),
                "calibration": str(row["calibration"].resolve()),
                "calibrationStatus": row["calibrationStatus"],
                "validationSafe": validation_safe(row["validation"]),
                "aggregate": row["validation"]["aggregate"],
            }
            for row in rows
        ],
        "testRead": False,
    }
    write_json(benchmark_root / scene / "selection.json", summary)
    return selected


def run_scene(scene: str, mode: str, model_root: Path, benchmark_root: Path, gpu_ids: Sequence[int]) -> None:
    contract = preflight(scene)
    write_json(benchmark_root / scene / "preflight.json", contract)
    if mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if mode == "smoke":
        run_jobs(
            [("smoke", train_command(scene, model_root / scene / "smoke", SEEDS[0], smoke=True))],
            gpu_ids[:1],
            benchmark_root / scene / "logs/smoke",
        )
        return
    if mode in {"train", "all"}:
        jobs = [
            (f"train_seed{seed}", train_command(scene, member_dir(model_root, scene, seed), seed, smoke=False))
            for seed in SEEDS
        ]
        run_jobs(jobs, gpu_ids[:3], benchmark_root / scene / "logs/train")
    if mode in {"evaluate", "all"}:
        jobs = []
        for seed in SEEDS:
            output = benchmark_root / scene / "validation" / f"seed{seed}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                (f"validation_seed{seed}", evaluate_command(scene, member_dir(model_root, scene, seed), output, seed, "validation"))
            )
        run_jobs(jobs, gpu_ids[:3], benchmark_root / scene / "logs/validation")
    if mode in {"finalize", "all"}:
        selected = select_validation_member(scene, model_root, benchmark_root)
        output = benchmark_root / scene / "test" / "full_v4.json"
        sidecar = benchmark_root / scene / "test" / "full_v4.sidecar"
        output.parent.mkdir(parents=True, exist_ok=True)
        run_jobs(
            [("frozen_test", evaluate_command(scene, selected["member"], output, selected["seed"], "test", sidecar))],
            gpu_ids[:1],
            benchmark_root / scene / "logs/test",
        )
        test = read_json(output)
        expected = int(SCENES[scene]["splits"]["test"])
        if test.get("testRead") is not True or int(test.get("poseCount", -1)) != expected:
            raise RuntimeError(f"{scene} frozen test output is incomplete")
        write_json(
            benchmark_root / scene / "final_summary.json",
            {
                "schema": "pvs-standard-graphics-full-v4-final-v1",
                "scene": scene,
                "selectedSeed": selected["seed"],
                "test": str(output.resolve()),
                "scoreSidecar": str(sidecar.resolve()),
                "aggregate": test["aggregate"],
                "poseMacro": test.get("poseMacro"),
                "testRead": True,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "smoke", "train", "evaluate", "finalize", "all"))
    parser.add_argument("--scenes", default="sponza_128k")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out/paper_results/standard_graphics/test_metrics",
    )
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    unknown = sorted(set(scenes) - set(SCENES))
    if unknown:
        parser.error(f"unknown scene(s) {unknown}; choose from {sorted(SCENES)}")
    for scene in scenes:
        run_scene(scene, args.mode, args.model_root.resolve(), args.benchmark_root.resolve(), args.gpu_ids)


if __name__ == "__main__":
    main()
