#!/usr/bin/env python3
"""Run the registered train-only standard-scene optimization experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "neural_instance_culling/benchmark") not in sys.path:
    sys.path.insert(0, str(ROOT / "neural_instance_culling/benchmark"))

from run_standard_graphics_mainline import (  # noqa: E402
    SCENES,
    SEEDS,
    _replace_command_value,
    evaluate_command,
    read_json,
    run_jobs,
    selected_checkpoint,
    train_command,
    validation_safe,
    write_json,
)


EXPERIMENT = "pvs_v4_standard_graphics_ambiguity_sampling_relation_sweep_v1"
SOURCE_RELATIONS = {
    "sponza_128k": "sponza_standard_graphics_v4_bounded_relation_csr_v1",
    "viking_village_128k": "viking_village_standard_graphics_v4_bounded_relation_csr_v1",
    "bigcity_128k": "bigcity_standard_graphics_v4_bounded_relation_csr_v1",
}
SPARSE_CACHES = {
    "sponza_128k": "sponza_triangle_depth_train/cache",
    "viking_village_128k": "viking_triangle_depth_train/cache",
    "bigcity_128k": "bigcity_triangle_depth_train/cache",
}
V2_DEPTH_ROOTS = {
    "sponza_128k": "sponza_triangle_depth_train",
    "viking_village_128k": "viking_triangle_depth_train",
    "bigcity_128k": "bigcity_triangle_depth_train",
}
RELATION_K = (8, 16, 32)
RELATION_QUALITY_Q01_FLOOR = 0.8
PILOT_EPOCHS = 8
PILOT_STEPS = 450
V2_DATASETS = {
    "sponza_128k": "pose_csr_sponza_standard_graphics_128k_fov66_sampling_v2",
    "viking_village_128k": "pose_csr_viking_village_standard_graphics_128k_fov66_sampling_v2",
    "bigcity_128k": "pose_csr_bigcity_standard_graphics_128k_fov66_sampling_v2",
}
V2_SPLITS = {
    "sponza_128k": {"train": 4800, "calibration": 528, "validation": 672, "test": 672, "guard": 0},
    "viking_village_128k": {"train": 1944, "calibration": 216, "validation": 276, "test": 276, "guard": 0},
    "bigcity_128k": {"train": 11580, "calibration": 1284, "validation": 1608, "test": 1608, "guard": 0},
}


def relation_dir(scene: str, source_k: int) -> Path:
    if source_k == 8:
        return ROOT / "neural_instance_culling/dataset/out" / SOURCE_RELATIONS[scene]
    stem = SOURCE_RELATIONS[scene].removesuffix("_v1")
    return ROOT / "neural_instance_culling/dataset/out" / f"{stem}_topk{source_k}_v1"


def v2_relation_dir(scene: str, source_k: int) -> Path:
    stem = SOURCE_RELATIONS[scene].removesuffix("_v1")
    return ROOT / "neural_instance_culling/dataset/out" / f"{stem}_sampling_v2_topk{source_k}_v1"


def relation_build_command(scene: str, source_k: int) -> list[str]:
    config = SCENES[scene]
    return [
        sys.executable,
        str(ROOT / "neural_instance_culling/dataset/build_observed_relation_csr.py"),
        "--dataset-dir", str(ROOT / config["dataset"]),
        "--runtime-meta", str(ROOT / config["runtime_meta"]),
        "--sparse-cache-root", str(
            ROOT / "neural_instance_culling/benchmark/out/paper_results/standard_graphics/preprocessing"
            / SPARSE_CACHES[scene]
        ),
        "--output-dir", str(relation_dir(scene, source_k)),
        "--splits", "train",
        "--source-k", str(source_k),
    ]


def v2_relation_build_command(scene: str, source_k: int, result_root: Path) -> list[str]:
    config = SCENES[scene]
    return [
        sys.executable,
        str(ROOT / "neural_instance_culling/dataset/build_observed_relation_csr.py"),
        "--dataset-dir", str(ROOT / "neural_instance_culling/dataset/out" / V2_DATASETS[scene]),
        "--runtime-meta", str(ROOT / config["runtime_meta"]),
        "--sparse-cache-root", str(
            result_root / "sampling_v2" / V2_DEPTH_ROOTS[scene] / "cache"
        ),
        "--output-dir", str(v2_relation_dir(scene, source_k)),
        "--splits", "train",
        "--source-k", str(source_k),
    ]


def current_selected_checkpoint(scene: str) -> tuple[Path, int]:
    selection = read_json(
        ROOT
        / "neural_instance_culling/benchmark/out/paper_results/standard_graphics/test_metrics"
        / scene
        / "selection.json"
    )
    member = Path(str(selection["selectedMember"]))
    seed = int(selection["selectedSeed"])
    return selected_checkpoint(member), seed


def pilot_dir(model_root: Path, scene: str, source_k: int) -> Path:
    return model_root / scene / f"topk{source_k}_ambiguity_balanced_pilot"


def optimized_train_command(
    scene: str,
    output: Path,
    seed: int,
    source_k: int,
    *,
    pilot: bool,
    init_checkpoint: Path | None = None,
) -> list[str]:
    command = train_command(scene, output, seed, smoke=False)
    _replace_command_value(command, "--relation-dir", str(relation_dir(scene, source_k)))
    replacements = {
        "--experiment-name": f"{EXPERIMENT}_{scene}_topk{source_k}_seed{seed}",
        "--epochs": str(PILOT_EPOCHS if pilot else 40),
        "--steps-per-epoch": str(PILOT_STEPS if pilot else 900),
        "--eval-every": "1" if pilot else "4",
        "--snapshot-every": "1" if pilot else "4",
        "--calibration-bootstrap-replicates": "2000" if pilot else "10000",
        "--learning-rate": "0.00002" if pilot else "0.0002",
        "--instance-calibration-warmup-fraction": "0" if pilot else "0.10",
        "--instance-calibration-ramp-fraction": "0" if pilot else "0.20",
        "--integrated-tail-ramp-fraction": "0" if pilot else "0.15",
    }
    for flag, value in replacements.items():
        _replace_command_value(command, flag, value)
    command.extend([
        "--pose-sampling", "ambiguity_balanced",
        "--hard-pose-fraction", "0.5",
        "--hard-pose-quantile", "0.65",
    ])
    if init_checkpoint is not None:
        command.extend(["--init-checkpoint", str(init_checkpoint)])
    return command


def v2_member_dir(model_root: Path, scene: str, source_k: int, seed: int) -> Path:
    return model_root / "sampling_v2" / scene / f"topk{source_k}_full_seed{seed}_e40"


def v2_train_command(
    scene: str,
    output: Path,
    seed: int,
    source_k: int,
) -> list[str]:
    command = optimized_train_command(scene, output, seed, source_k, pilot=False)
    _replace_command_value(
        command,
        "--dataset-dir",
        str(ROOT / "neural_instance_culling/dataset/out" / V2_DATASETS[scene]),
    )
    _replace_command_value(command, "--relation-dir", str(v2_relation_dir(scene, source_k)))
    _replace_command_value(
        command,
        "--experiment-name",
        f"{EXPERIMENT}_sampling_v2_{scene}_topk{source_k}_seed{seed}",
    )
    return command


def v2_evaluate_command(
    scene: str,
    member: Path,
    output: Path,
    seed: int,
    source_k: int,
    sidecar: Path,
) -> list[str]:
    command = evaluate_command(
        scene,
        member,
        output,
        seed,
        "test",
        sidecar,
        method_name="full_v4_ambiguity_sampling_relation_capacity",
    )
    _replace_command_value(
        command,
        "--dataset-dir",
        str(ROOT / "neural_instance_culling/dataset/out" / V2_DATASETS[scene]),
    )
    _replace_command_value(command, "--relation-dir", str(v2_relation_dir(scene, source_k)))
    return command


def validation_payload(member: Path) -> dict[str, Any]:
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    return {
        "status": summary.get("status"),
        "safe": summary.get("status") == "safe" and validation_safe(validation),
        "epoch": (summary.get("bestSafe") or summary.get("bestDiagnostic") or {}).get("epoch"),
        "validation": validation,
        "testRead": False,
    }


def train_only_relation_k(scene: str) -> tuple[int, list[dict[str, Any]]]:
    return _train_only_relation_k(scene, relation_dir)


def train_only_v2_relation_k(scene: str) -> tuple[int, list[dict[str, Any]]]:
    return _train_only_relation_k(scene, v2_relation_dir)


def _train_only_relation_k(scene: str, resolver: Any) -> tuple[int, list[dict[str, Any]]]:
    rows = []
    for source_k in RELATION_K:
        metadata = read_json(resolver(scene, source_k) / "relation_csr_meta.json")
        diagnostics = ((metadata.get("evidenceTopK") or {}).get("diagnostics") or {})
        retained = diagnostics.get("retainedQualityQuantiles") or {}
        rows.append({
            "sourceTopK": source_k,
            "retainedQualityQ01": float(retained["q01"]),
        })
    qualified = [
        row for row in rows
        if row["retainedQualityQ01"] >= RELATION_QUALITY_Q01_FLOOR
    ]
    selected = min(qualified or rows, key=lambda row: row["sourceTopK"])
    return int(selected["sourceTopK"]), rows


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    validation = row["validation"]
    return (
        float(row["safe"]),
        float(validation.get("agg_useful_cull", 0.0)),
        float(validation.get("agg_balanced_accuracy", 0.0)),
        float(validation.get("agg_specificity", 0.0)),
        float(validation.get("agg_precision", 0.0)),
        float(validation.get("aggregateWeightedRecallLowerConfidenceBound", 0.0)),
        -float(validation.get("avg_pred_count", float("inf"))),
        -float(row["sourceTopK"]),
    )


def summarize_pilots(model_root: Path, result_root: Path, scenes: list[str]) -> dict[str, Any]:
    selections = {}
    for scene in scenes:
        rows = []
        for source_k in RELATION_K:
            member = pilot_dir(model_root, scene, source_k)
            row = {
                "scene": scene,
                "sourceTopK": source_k,
                "member": str(member.resolve()),
                **validation_payload(member),
            }
            rows.append(row)
        safe_rows = [row for row in rows if row["safe"]]
        pool = safe_rows or rows
        selected = max(pool, key=selection_key)
        formal_k, relation_quality = train_only_relation_k(scene)
        selections[scene] = {
            "validationBestSourceTopK": int(selected["sourceTopK"]),
            "validationSelectionPool": "safe" if safe_rows else "diagnostic",
            "trainOnlySelectedSourceTopK": formal_k,
            "trainOnlySelectionRule": (
                f"minimum sourceTopK with retained relation quality q01 >= {RELATION_QUALITY_Q01_FLOOR}"
            ),
            "relationQualityRows": relation_quality,
            "rows": rows,
            "testRead": False,
        }
    payload = {
        "schema": "pvs-standard-graphics-ambiguity-relation-pilot-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "validationDiagnosticRule": "safety, useful cull, balanced accuracy, occlusion recall, precision, WR LCB, fewer predictions",
        "formalCapacityRule": (
            f"minimum train-only relation sourceTopK with retained quality q01 >= {RELATION_QUALITY_Q01_FLOOR}"
        ),
        "selections": selections,
        "testRead": False,
    }
    write_json(result_root / "pilot_selection.json", payload)
    return payload


def run_relation_builds(scenes: list[str], log_root: Path) -> None:
    for scene in scenes:
        for source_k in (16, 32):
            output = relation_dir(scene, source_k)
            if (output / "relation_csr_meta.json").is_file():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            log_dir = log_root / "relation_build"
            log_dir.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ)
            with (log_dir / f"{scene}_topk{source_k}.stdout.log").open("w", encoding="utf-8") as stdout, (
                log_dir / f"{scene}_topk{source_k}.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    relation_build_command(scene, source_k),
                    cwd=ROOT,
                    env=environment,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"relation build failed for {scene} top-k {source_k}")


def run_v2_relation_builds(scenes: list[str], result_root: Path) -> None:
    log_dir = result_root / "logs/sampling_v2_relation_build"
    log_dir.mkdir(parents=True, exist_ok=True)
    for scene in scenes:
        for source_k in RELATION_K:
            output = v2_relation_dir(scene, source_k)
            if (output / "relation_csr_meta.json").is_file():
                continue
            with (log_dir / f"{scene}_topk{source_k}.stdout.log").open("w", encoding="utf-8") as stdout, (
                log_dir / f"{scene}_topk{source_k}.stderr.log"
            ).open("w", encoding="utf-8") as stderr:
                result = subprocess.run(
                    v2_relation_build_command(scene, source_k, result_root),
                    cwd=ROOT,
                    env=dict(os.environ),
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(f"sampling-v2 relation build failed for {scene} top-k {source_k}")


def select_v2_member(scene: str, model_root: Path, result_root: Path) -> dict[str, Any]:
    source_k, relation_quality = train_only_v2_relation_k(scene)
    rows = []
    for seed in SEEDS:
        member = v2_member_dir(model_root, scene, source_k, seed)
        rows.append({
            "scene": scene,
            "sourceTopK": source_k,
            "seed": seed,
            "member": str(member.resolve()),
            **validation_payload(member),
        })
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        raise RuntimeError(f"{scene} sampling-v2 has no calibration/validation-safe member")
    selected = max(safe_rows, key=selection_key)
    payload = {
        "schema": "pvs-standard-graphics-sampling-v2-validation-selection-v1",
        "scene": scene,
        "sourceTopK": source_k,
        "relationQualityRows": relation_quality,
        "selectedSeed": int(selected["seed"]),
        "selectedMember": selected["member"],
        "rows": rows,
        "selectionSplit": "validation",
        "testRead": False,
    }
    write_json(result_root / "sampling_v2" / scene / "selection.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "build-relations", "pilot", "summarize-pilot", "full",
            "build-v2-relations", "v2-full", "v2-finalize",
        ),
    )
    parser.add_argument(
        "--scenes",
        default="sponza_128k,viking_village_128k,bigcity_128k",
    )
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / EXPERIMENT,
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out/paper_results/standard_graphics/optimization_v1",
    )
    args = parser.parse_args()
    scenes = [value.strip() for value in args.scenes.split(",") if value.strip()]
    unknown = sorted(set(scenes) - set(SCENES))
    if unknown:
        parser.error(f"unknown scenes: {unknown}")
    model_root = args.model_root.resolve()
    result_root = args.result_root.resolve()

    if args.mode == "build-relations":
        run_relation_builds(scenes, result_root / "logs")
        return
    if args.mode == "build-v2-relations":
        run_v2_relation_builds(scenes, result_root)
        return
    if args.mode == "pilot":
        jobs = []
        for scene in scenes:
            checkpoint, seed = current_selected_checkpoint(scene)
            for source_k in RELATION_K:
                output = pilot_dir(model_root, scene, source_k)
                jobs.append((
                    f"{scene}_topk{source_k}",
                    optimized_train_command(
                        scene, output, seed, source_k, pilot=True, init_checkpoint=checkpoint
                    ),
                ))
        run_jobs(jobs, args.gpu_ids, result_root / "logs/pilot")
        return
    selection = summarize_pilots(model_root, result_root, scenes)
    if args.mode == "summarize-pilot":
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return

    if args.mode == "v2-full":
        jobs = []
        for scene in scenes:
            source_k, _relation_quality = train_only_v2_relation_k(scene)
            dataset_meta = read_json(
                ROOT / "neural_instance_culling/dataset/out" / V2_DATASETS[scene] / "dataset_meta.json"
            )
            if dataset_meta.get("splitCounts") != V2_SPLITS[scene]:
                raise ValueError(f"{scene} sampling-v2 split counts changed")
            if not (v2_relation_dir(scene, source_k) / "relation_csr_meta.json").is_file():
                raise FileNotFoundError(v2_relation_dir(scene, source_k))
            for seed in SEEDS:
                output = v2_member_dir(model_root, scene, source_k, seed)
                jobs.append((
                    f"sampling_v2_{scene}_topk{source_k}_seed{seed}",
                    v2_train_command(scene, output, seed, source_k),
                ))
        run_jobs(jobs, args.gpu_ids, result_root / "logs/sampling_v2_full")
        return

    if args.mode == "v2-finalize":
        for scene in scenes:
            selected = select_v2_member(scene, model_root, result_root)
            source_k = int(selected["sourceTopK"])
            seed = int(selected["selectedSeed"])
            member = Path(str(selected["selectedMember"]))
            output_root = result_root / "sampling_v2" / scene / "test"
            output = output_root / "full_v4.json"
            sidecar = output_root / "full_v4.sidecar"
            if output.exists() or sidecar.exists():
                raise FileExistsError(f"refusing to reread sampling-v2 test: {output_root}")
            output_root.mkdir(parents=True, exist_ok=True)
            run_jobs(
                [(
                    f"sampling_v2_{scene}_frozen_test",
                    v2_evaluate_command(scene, member, output, seed, source_k, sidecar),
                )],
                args.gpu_ids[:1],
                result_root / "logs/sampling_v2_test" / scene,
            )
            result = read_json(output)
            if result.get("testRead") is not True or int(result.get("poseCount", -1)) != V2_SPLITS[scene]["test"]:
                raise RuntimeError(f"{scene} sampling-v2 frozen test is incomplete")
            write_json(
                result_root / "sampling_v2" / scene / "final_summary.json",
                {
                    "schema": "pvs-standard-graphics-sampling-v2-final-v1",
                    "scene": scene,
                    "sourceTopK": source_k,
                    "selectedSeed": seed,
                    "selectedMember": str(member.resolve()),
                    "test": str(output.resolve()),
                    "scoreSidecar": str(sidecar.resolve()),
                    "aggregate": result.get("aggregate"),
                    "poseMacro": result.get("poseMacro"),
                    "testRead": True,
                },
            )
        return

    jobs = []
    for scene in scenes:
        source_k = int(selection["selections"][scene]["trainOnlySelectedSourceTopK"])
        for seed in SEEDS:
            output = model_root / scene / f"topk{source_k}_ambiguity_balanced_full_seed{seed}_e40"
            jobs.append((
                f"{scene}_topk{source_k}_seed{seed}",
                optimized_train_command(scene, output, seed, source_k, pilot=False),
            ))
    run_jobs(jobs, args.gpu_ids, result_root / "logs/full")


if __name__ == "__main__":
    main()
