#!/usr/bin/env python3
"""Run the fixed Full V4 protocol on registered standard graphics scenes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs.py"
EXPORT = ROOT / "neural_instance_culling/model/export_pvs.py"
SEEDS = (20260801, 20260802, 20260803)
EPOCHS = 40
STEPS_PER_EPOCH = 900
WEIGHTED_RECALL_FLOOR = 0.99
VIKING_FINETUNE_ROOT = ROOT / "neural_instance_culling/model/out/pvs_v4_viking_region_stability_finetune_v1"
VIKING_FINETUNE_MEMBERS = {
    20260801: "conservative_seed20260801_from_best_safe_lr5e-6_e4x450",
    20260802: "conservative_seed20260802_from_e032_lr5e-6_e4x450",
    20260803: "conservative_seed20260803_from_best_safe_lr5e-6_e4x450",
}
BIGCITY_RECOVERY_ROOT = ROOT / "neural_instance_culling/model/out/pvs_v4_bigcity_runtime_head_recovery_v1"
BIGCITY_RECOVERY_MEMBERS = {
    "balanced": "seed20260802_head_reset_balanced_e8x600",
    "strong_separation": "seed20260802_head_reset_strong_separation_e8x600",
}
BIGCITY_RECOVERY_CONFIGS = {
    "balanced": {"separation": "0.35", "negative_fraction": "0.05", "negative_cap": "512"},
    "strong_separation": {"separation": "0.60", "negative_fraction": "0.10", "negative_cap": "1024"},
}

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


def selection_members(
    scene: str,
    model_root: Path,
    finetune_root: Path = VIKING_FINETUNE_ROOT,
    bigcity_recovery_root: Path = BIGCITY_RECOVERY_ROOT,
) -> list[dict[str, Any]]:
    rows = [
        {
            "label": f"full_seed{seed}",
            "memberType": "full_from_scratch",
            "seed": seed,
            "member": member_dir(model_root, scene, seed),
        }
        for seed in SEEDS
    ]
    if scene == "viking_village_128k":
        finetunes = [
            {
                "label": f"region_stability_seed{seed}",
                "memberType": "region_stability_finetune",
                "seed": seed,
                "member": finetune_root / VIKING_FINETUNE_MEMBERS[seed],
            }
            for seed in SEEDS
        ]
        if all((row["member"] / "calibration_ready_summary.json").is_file() for row in finetunes):
            rows.extend(finetunes)
    if scene == "bigcity_128k":
        recoveries = [
            {
                "label": f"runtime_head_recovery_{label}",
                "memberType": "runtime_head_recovery",
                "seed": 20260802,
                "member": bigcity_recovery_root / member,
            }
            for label, member in BIGCITY_RECOVERY_MEMBERS.items()
        ]
        if all((row["member"] / "calibration_ready_summary.json").is_file() for row in recoveries):
            rows.extend(recoveries)
    return rows


def evaluation_method_name(member_type: str) -> str:
    return {
        "full_from_scratch": "full_v4",
        "region_stability_finetune": "full_v4_region_stability_finetune",
        "runtime_head_recovery": "full_v4_runtime_head_recovery",
    }[member_type]


def validation_output(benchmark_root: Path, scene: str, row: dict[str, Any]) -> Path:
    name = f"seed{row['seed']}.json" if row["memberType"] == "full_from_scratch" else f"{row['label']}.json"
    return benchmark_root / scene / "validation" / name


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


def _replace_command_value(command: list[str], flag: str, value: str) -> None:
    index = command.index(flag)
    command[index + 1] = value


def bigcity_recovery_command(label: str) -> list[str]:
    config = BIGCITY_RECOVERY_CONFIGS[label]
    output = BIGCITY_RECOVERY_ROOT / BIGCITY_RECOVERY_MEMBERS[label]
    command = train_command("bigcity_128k", output, 20260802, smoke=False)
    command.extend([
        "--init-checkpoint",
        str(member_dir(
            ROOT / "neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1",
            "bigcity_128k",
            20260802,
        ) / "last.pt"),
        "--reset-runtime-heads",
    ])
    for flag, value in {
        "--experiment-name": f"pvs_v4_bigcity_runtime_head_recovery_v1_{label}",
        "--epochs": "8",
        "--steps-per-epoch": "600",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--learning-rate": "0.0001",
        "--instance-calibration-warmup-fraction": "0",
        "--instance-calibration-ramp-fraction": "0",
        "--integrated-separation-weight": config["separation"],
        "--integrated-tail-ramp-fraction": "0",
        "--frontier-negative-fraction": config["negative_fraction"],
        "--frontier-negative-count-cap": config["negative_cap"],
    }.items():
        _replace_command_value(command, flag, value)
    return command


def viking_finetune_command(seed: int) -> list[str]:
    output = VIKING_FINETUNE_ROOT / VIKING_FINETUNE_MEMBERS[seed]
    command = train_command("viking_village_128k", output, seed, smoke=False)
    command.extend([
        "--init-checkpoint",
        str(member_dir(
            ROOT / "neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1",
            "viking_village_128k",
            seed,
        ) / "best_safe.pt"),
    ])
    for flag, value in {
        "--experiment-name": f"pvs_v4_viking_region_stability_finetune_v1_conservative_seed{seed}",
        "--epochs": "4",
        "--steps-per-epoch": "450",
        "--poses-per-batch": "8",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--learning-rate": "0.000005",
        "--instance-calibration-warmup-fraction": "0",
        "--instance-calibration-ramp-fraction": "0",
        "--integrated-rvl-recall-guard-weight": "0.75",
        "--integrated-rvl-recall-target": "0.997",
        "--integrated-rvl-pose-cvar-weight": "0.75",
        "--integrated-tail-ramp-fraction": "0",
    }.items():
        _replace_command_value(command, flag, value)
    return command


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
    method_name: str = "full_v4",
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
        "--method-name", method_name,
    ]
    if split != "test":
        command.append("--allow-unsafe-diagnostic")
    else:
        command.extend(["--persist-scores", "--sidecar-dir", str(sidecar)])
    return command


def export_command(scene: str, selected: dict[str, Any], output: Path) -> list[str]:
    return [
        sys.executable,
        str(EXPORT),
        "--checkpoint", str(selected["checkpoint"]),
        "--runtime-meta", str(scene_paths(scene)["runtime_meta"]),
        "--output-dir", str(output),
    ]


def run_jobs(jobs: Sequence[tuple[str, list[str]]], gpu_ids: Sequence[int], log_dir: Path) -> None:
    if not jobs:
        return
    if not gpu_ids:
        raise ValueError("at least one GPU ID is required")
    log_dir.mkdir(parents=True, exist_ok=True)

    available_gpus: Queue[int] = Queue()
    for gpu_id in gpu_ids:
        available_gpus.put(int(gpu_id))

    def run(name: str, command: list[str]) -> tuple[str, int, float]:
        gpu_id = available_gpus.get()
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            started = time.time()
            with (log_dir / f"{name}.stdout.log").open("w", encoding="utf-8") as stdout, (
                log_dir / f"{name}.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
            return name, int(result.returncode), float(time.time() - started)
        finally:
            available_gpus.put(gpu_id)

    with ThreadPoolExecutor(max_workers=min(len(jobs), len(gpu_ids))) as executor:
        futures = {
            executor.submit(run, name, command): name
            for name, command in jobs
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
    for candidate in selection_members(scene, model_root):
        seed = int(candidate["seed"])
        member = Path(candidate["member"])
        validation_path = validation_output(benchmark_root, scene, candidate)
        validation = read_json(validation_path)
        calibration = read_json(member / "calibration_ready_summary.json")
        rows.append(
            {
                "label": candidate["label"],
                "memberType": candidate["memberType"],
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
        "selectedLabel": selected["label"],
        "selectedMemberType": selected["memberType"],
        "selectedMember": str(selected["member"].resolve()),
        "safePoolMembers": [row["label"] for row in safe],
        "rows": [
            {
                "label": row["label"],
                "memberType": row["memberType"],
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


def load_validation_selection(scene: str, model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    summary = read_json(benchmark_root / scene / "selection.json")
    if summary.get("testRead") is not False or summary.get("selectionSplit") != "validation":
        raise ValueError(f"{scene} runtime export requires a frozen validation-only selection")
    seed = int(summary["selectedSeed"])
    member = Path(summary["selectedMember"]).resolve()
    return {
        "label": summary["selectedLabel"],
        "memberType": summary["selectedMemberType"],
        "seed": seed,
        "member": member,
        "checkpoint": selected_checkpoint(member),
        "calibration": member / "calibration_ready_summary.json",
    }


def export_selected_runtime(
    scene: str,
    selected: dict[str, Any],
    model_root: Path,
    benchmark_root: Path,
    gpu_ids: Sequence[int],
) -> None:
    output = model_root / scene / "runtime_selected_v1"
    run_jobs(
        [("export_runtime", export_command(scene, selected, output))],
        gpu_ids[:1],
        benchmark_root / scene / "logs/export",
    )
    meta = read_json(output / "model_meta.json")
    write_json(
        benchmark_root / scene / "runtime_export.json",
        {
            "schema": "pvs-standard-graphics-runtime-export-v1",
            "scene": scene,
            "selectedSeed": int(selected["seed"]),
            "selectedLabel": selected["label"],
            "selectedMemberType": selected["memberType"],
            "selectionSplit": "validation",
            "runtimeDir": str(output.resolve()),
            "runtimeSchema": meta.get("schema"),
            "threshold": meta.get("threshold"),
            "numInstances": meta.get("numInstances"),
            "testRead": False,
        },
    )


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
    if mode == "recover-bigcity":
        if scene != "bigcity_128k":
            raise ValueError("recover-bigcity only supports bigcity_128k")
        run_jobs(
            [
                (f"runtime_head_recovery_{label}", bigcity_recovery_command(label))
                for label in BIGCITY_RECOVERY_CONFIGS
            ],
            gpu_ids,
            BIGCITY_RECOVERY_ROOT / "run_logs",
        )
    if mode == "finetune-viking":
        if scene != "viking_village_128k":
            raise ValueError("finetune-viking only supports viking_village_128k")
        pending = [seed for seed in SEEDS if not (
            VIKING_FINETUNE_ROOT / VIKING_FINETUNE_MEMBERS[seed] / "calibration_ready_summary.json"
        ).is_file()]
        run_jobs(
            [
                (f"region_stability_seed{seed}", viking_finetune_command(seed))
                for seed in pending
            ],
            gpu_ids,
            VIKING_FINETUNE_ROOT / "run_logs",
        )
    if mode in {"train", "all"}:
        jobs = [
            (f"train_seed{seed}", train_command(scene, member_dir(model_root, scene, seed), seed, smoke=False))
            for seed in SEEDS
        ]
        run_jobs(jobs, gpu_ids[:3], benchmark_root / scene / "logs/train")
    if mode in {"evaluate", "all", "recover-bigcity", "finetune-viking"}:
        jobs = []
        for candidate in selection_members(scene, model_root):
            seed = int(candidate["seed"])
            output = validation_output(benchmark_root, scene, candidate)
            output.parent.mkdir(parents=True, exist_ok=True)
            jobs.append(
                (
                    f"validation_{candidate['label']}",
                    evaluate_command(
                        scene,
                        Path(candidate["member"]),
                        output,
                        seed,
                        "validation",
                        method_name=evaluation_method_name(candidate["memberType"]),
                    ),
                )
            )
        run_jobs(jobs, gpu_ids[:3], benchmark_root / scene / "logs/validation")
    selected: dict[str, Any] | None = None
    if mode in {"finalize", "all", "recover-bigcity", "finetune-viking"}:
        selected = select_validation_member(scene, model_root, benchmark_root)
        output = benchmark_root / scene / "test" / "full_v4.json"
        sidecar = benchmark_root / scene / "test" / "full_v4.sidecar"
        output.parent.mkdir(parents=True, exist_ok=True)
        run_jobs(
            [(
                "frozen_test",
                evaluate_command(
                    scene,
                    selected["member"],
                    output,
                    selected["seed"],
                    "test",
                    sidecar,
                    method_name=evaluation_method_name(selected["memberType"]),
                ),
            )],
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
                "selectedLabel": selected["label"],
                "selectedMemberType": selected["memberType"],
                "test": str(output.resolve()),
                "scoreSidecar": str(sidecar.resolve()),
                "aggregate": test["aggregate"],
                "poseMacro": test.get("poseMacro"),
                "testRead": True,
            },
        )
    if mode in {"export", "all"}:
        if selected is None:
            selected = load_validation_selection(scene, model_root, benchmark_root)
        export_selected_runtime(scene, selected, model_root, benchmark_root, gpu_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight", "smoke", "train", "evaluate", "finalize", "export", "all",
            "recover-bigcity", "finetune-viking",
        ),
    )
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
