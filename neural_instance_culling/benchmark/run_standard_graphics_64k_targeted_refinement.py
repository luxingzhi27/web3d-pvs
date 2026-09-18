#!/usr/bin/env python3
"""Run targeted checkpoint refinement for the standard-graphics 64 KiB models.

The scan is train/calibration/validation only. It waits for every registered
Full member to finish, tunes on one representative seed per scene, and expands
the selected scene-specific configuration to all three seeds only after an
explicit validation summary has been written.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Sequence

try:
    from .run_standard_graphics_connected_sah_full import preflight, train_command
    from .standard_graphics_connected_sah_config import ROOT, SEEDS
except ImportError:
    from run_standard_graphics_connected_sah_full import preflight, train_command  # type: ignore
    from standard_graphics_connected_sah_config import ROOT, SEEDS  # type: ignore


EXPERIMENT = "pvs_v4_standard_graphics_64k_targeted_refinement_v1"
SCENES = ("sponza_64k", "bigcity_64k", "viking_village_64k")
MODEL_ROOT = ROOT / "neural_instance_culling/model/out" / EXPERIMENT
RESULT_ROOT = ROOT / "neural_instance_culling/benchmark/out/paper_results" / EXPERIMENT
BASE_EXPERIMENTS = {
    "sponza_64k": "pvs_v4_sponza_connected_sah_64k_full_v1",
    "bigcity_64k": "pvs_v4_bigcity_connected_sah_64k_full_v1",
    "viking_village_64k": "pvs_v4_viking_connected_sah_64k_full_v1",
}
PILOT_SEEDS = {
    "sponza_64k": 20260803,
    "bigcity_64k": 20260801,
    "viking_village_64k": 20260803,
}
CONTROL_NAME = "control_no_refinement"


@dataclass(frozen=True)
class RefinementConfig:
    scene: str
    name: str
    learning_rate: float
    hard_pose_fraction: float
    hard_pose_quantile: float
    negative_only_pose_fraction: float
    recall_guard_weight: float
    recall_target: float
    pose_cvar_fraction: float
    pose_cvar_weight: float
    separation_weight: float
    positive_mass_fraction: float
    positive_count_cap: int
    negative_fraction: float
    negative_count_cap: int
    relation_k: int = 8
    survival_loss_weight: float = 0.25
    relation_consistency_weight: float = 0.10
    positive_importance_floor: float = 0.25
    positive_importance_power: float = 0.5
    source_kind: str = "full"


CONFIGS = (
    RefinementConfig(
        "sponza_64k", "s0_mild_safety_bridge", 1e-6,
        0.50, 0.55, 0.0, 0.40, 0.995, 0.35, 0.35, 0.40,
        0.02, 128, 0.03, 256,
    ),
    RefinementConfig(
        "sponza_64k", "s1_tail_separation_bridge", 2e-6,
        0.65, 0.45, 0.0, 0.35, 0.993, 0.30, 0.30, 0.60,
        0.02, 128, 0.05, 384,
    ),
    RefinementConfig(
        "bigcity_64k", "b0_balanced_neg12", 2e-6,
        0.50, 0.55, 0.125, 0.55, 0.995, 0.40, 0.50, 0.55,
        0.02, 256, 0.04, 512,
    ),
    RefinementConfig(
        "bigcity_64k", "b1_guarded_neg25", 2e-6,
        0.50, 0.50, 0.25, 0.70, 0.997, 0.50, 0.65, 0.65,
        0.025, 256, 0.05, 512,
    ),
    RefinementConfig(
        "bigcity_64k", "b2_bridge_neg12_sep80", 1e-6,
        0.55, 0.50, 0.125, 0.62, 0.996, 0.45, 0.55, 0.80,
        0.02, 256, 0.07, 768,
    ),
    RefinementConfig(
        "bigcity_64k", "b3_bridge_neg25_sep80", 1e-6,
        0.55, 0.50, 0.25, 0.62, 0.996, 0.45, 0.55, 0.80,
        0.02, 256, 0.07, 768,
    ),
    RefinementConfig(
        "bigcity_64k", "b4_k16_guarded_neg25", 2e-6,
        0.50, 0.50, 0.25, 0.70, 0.997, 0.50, 0.65, 0.65,
        0.025, 256, 0.05, 512, 16,
    ),
    RefinementConfig(
        "bigcity_64k", "b5_interpolated_neg25", 1.5e-6,
        0.525, 0.50, 0.25, 0.67, 0.9965, 0.475, 0.60, 0.72,
        0.0225, 256, 0.06, 640,
    ),
    RefinementConfig(
        "viking_village_64k", "v0_balanced_separation", 1e-6,
        0.65, 0.45, 0.0, 0.25, 0.992, 0.35, 0.25, 0.60,
        0.02, 128, 0.05, 384,
    ),
    RefinementConfig(
        "viking_village_64k", "v1_strong_separation", 1e-6,
        0.75, 0.40, 0.0, 0.20, 0.990, 0.30, 0.20, 0.90,
        0.02, 128, 0.07, 512,
    ),
    RefinementConfig(
        "sponza_64k", "s2_visual_utility", 5e-7,
        0.60, 0.45, 0.0, 0.25, 0.992, 0.30, 0.25, 0.30,
        0.01, 128, 0.05, 384,
        survival_loss_weight=0.10,
        relation_consistency_weight=0.05,
        positive_importance_floor=0.05,
        positive_importance_power=1.0,
    ),
    RefinementConfig(
        "sponza_64k", "s3_visual_utility_cull", 5e-7,
        0.70, 0.40, 0.0, 0.20, 0.990, 0.25, 0.20, 0.45,
        0.01, 128, 0.07, 512,
        survival_loss_weight=0.05,
        relation_consistency_weight=0.02,
        positive_importance_floor=0.01,
        positive_importance_power=1.0,
    ),
    RefinementConfig(
        "bigcity_64k", "b6_visual_utility", 5e-7,
        0.60, 0.45, 0.125, 0.25, 0.992, 0.30, 0.25, 0.30,
        0.01, 256, 0.05, 512,
        survival_loss_weight=0.10,
        relation_consistency_weight=0.05,
        positive_importance_floor=0.05,
        positive_importance_power=1.0,
        source_kind="bigcity_b1",
    ),
    RefinementConfig(
        "bigcity_64k", "b7_visual_utility_cull", 5e-7,
        0.70, 0.40, 0.25, 0.20, 0.990, 0.25, 0.20, 0.45,
        0.01, 256, 0.07, 640,
        survival_loss_weight=0.05,
        relation_consistency_weight=0.02,
        positive_importance_floor=0.01,
        positive_importance_power=1.0,
        source_kind="bigcity_b1",
    ),
    RefinementConfig(
        "viking_village_64k", "v2_visual_utility", 5e-7,
        0.60, 0.45, 0.0, 0.25, 0.992, 0.30, 0.25, 0.30,
        0.01, 128, 0.05, 384,
        survival_loss_weight=0.10,
        relation_consistency_weight=0.05,
        positive_importance_floor=0.05,
        positive_importance_power=1.0,
    ),
    RefinementConfig(
        "viking_village_64k", "v3_visual_utility_cull", 5e-7,
        0.70, 0.40, 0.0, 0.20, 0.990, 0.25, 0.20, 0.45,
        0.01, 128, 0.07, 512,
        survival_loss_weight=0.05,
        relation_consistency_weight=0.02,
        positive_importance_floor=0.01,
        positive_importance_power=1.0,
    ),
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


def replace_value(command: list[str], flag: str, value: str) -> None:
    command[command.index(flag) + 1] = value


def base_member(scene: str, seed: int) -> Path:
    return (
        ROOT
        / "neural_instance_culling/model/out"
        / BASE_EXPERIMENTS[scene]
        / f"full_seed{seed}_e40"
    )


def source_checkpoint(scene: str, seed: int) -> Path:
    return base_member(scene, seed) / "best_diagnostic.pt"


def config_source_checkpoint(config: RefinementConfig, seed: int) -> Path:
    if config.source_kind == "full":
        return source_checkpoint(config.scene, seed)
    if config.source_kind == "bigcity_b1":
        return (
            MODEL_ROOT
            / "final/bigcity_64k"
            / f"b1_guarded_neg25_seed{seed}"
            / "best_diagnostic.pt"
        )
    raise ValueError(f"unsupported refinement source: {config.source_kind}")


def configs_for_scene(scene: str) -> tuple[RefinementConfig, ...]:
    return tuple(config for config in CONFIGS if config.scene == scene)


def relation_dir(config: RefinementConfig) -> Path:
    scene_root = (
        ROOT
        / "neural_instance_culling/dataset/out/standard_graphics_connected_sah_64k_v1"
        / config.scene
    )
    return scene_root / ("relation_csr" if config.relation_k == 8 else f"relation_csr_k{config.relation_k}")


def member_dir(stage: str, config: RefinementConfig, seed: int) -> Path:
    return MODEL_ROOT / stage / config.scene / f"{config.name}_seed{seed}"


def completed(member: Path) -> bool:
    summary = member / "calibration_ready_summary.json"
    if not summary.is_file():
        return False
    value = read_json(summary)
    return value.get("testRead") is False and value.get("status") in {
        "safe", "no_qualified_safety_workpoint"
    }


def refinement_command(
    config: RefinementConfig,
    seed: int,
    output: Path,
    source: Path,
    *,
    final: bool,
) -> list[str]:
    command = train_command(config.scene, output, seed, smoke=False)
    epochs = 4 if final else 2
    steps_per_epoch = 450
    values = {
        "--relation-dir": str(relation_dir(config)),
        "--output-dir": str(output),
        "--experiment-name": f"{EXPERIMENT}_{config.name}_seed{seed}",
        "--epochs": str(epochs),
        "--steps-per-epoch": str(steps_per_epoch),
        "--poses-per-batch": "16",
        "--eval-every": "1",
        "--snapshot-every": "1",
        "--max-eval-poses": "0",
        "--calibration-bootstrap-replicates": "10000" if final else "2000",
        "--learning-rate": str(config.learning_rate),
        "--survival-loss-weight": str(config.survival_loss_weight),
        "--relation-consistency-weight": str(config.relation_consistency_weight),
        "--hard-pose-fraction": str(config.hard_pose_fraction),
        "--hard-pose-quantile": str(config.hard_pose_quantile),
        "--instance-calibration-warmup-fraction": "0",
        "--instance-calibration-ramp-fraction": "0",
        "--integrated-rvl-recall-guard-weight": str(config.recall_guard_weight),
        "--integrated-rvl-recall-target": str(config.recall_target),
        "--integrated-rvl-pose-cvar-fraction": str(config.pose_cvar_fraction),
        "--integrated-rvl-pose-cvar-weight": str(config.pose_cvar_weight),
        "--integrated-separation-weight": str(config.separation_weight),
        "--integrated-rvl-recall-guard-zero-fraction": "0",
        "--integrated-rvl-recall-guard-middle-end-fraction": "0.05",
        "--integrated-rvl-recall-guard-middle-scale": "1",
        "--integrated-tail-zero-fraction": "0",
        "--integrated-tail-ramp-fraction": "0.05",
        "--frontier-positive-mass-fraction": str(config.positive_mass_fraction),
        "--frontier-positive-count-cap": str(config.positive_count_cap),
        "--frontier-negative-fraction": str(config.negative_fraction),
        "--frontier-negative-count-cap": str(config.negative_count_cap),
        "--frontier-positive-importance-floor": str(config.positive_importance_floor),
        "--frontier-positive-importance-power": str(config.positive_importance_power),
    }
    for flag, value in values.items():
        replace_value(command, flag, value)
    command.extend([
        "--training-course-total-steps", str(epochs * steps_per_epoch),
        "--negative-only-pose-fraction", str(config.negative_only_pose_fraction),
        "--init-checkpoint", str(source),
    ])
    return command


def qualification_tier(validation: dict[str, Any]) -> str:
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    lcb = metric(
        validation,
        "aggregateWeightedRecallLowerConfidenceBound",
        "weighted_recall_lower_confidence_bound",
    )
    if wr > 0.99 and lcb > 0.99:
        return "confidence_target_met"
    if wr > 0.99:
        return "mean_target_met"
    return "mean_target_not_met"


def metric(validation: dict[str, Any], *names: str) -> float:
    for name in names:
        if validation.get(name) is not None:
            return float(validation[name])
    return 0.0


def result_row(config: RefinementConfig) -> dict[str, Any]:
    seed = PILOT_SEEDS[config.scene]
    member = member_dir("scan", config, seed)
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    return {
        "scene": config.scene,
        "config": config.name,
        "seed": seed,
        "member": str(member.resolve()),
        "qualificationTier": qualification_tier(validation),
        "validation": validation,
        "testRead": False,
    }


def control_row(scene: str) -> dict[str, Any]:
    seed = PILOT_SEEDS[scene]
    member = base_member(scene, seed)
    summary = read_json(member / "calibration_ready_summary.json")
    validation = summary.get("validationAtFrozenThreshold") or {}
    return {
        "scene": scene,
        "config": CONTROL_NAME,
        "seed": seed,
        "member": str(member.resolve()),
        "qualificationTier": qualification_tier(validation),
        "validation": validation,
        "testRead": False,
    }


def selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    validation = row["validation"]
    tier = row["qualificationTier"]
    tier_rank = {
        "mean_target_not_met": 0.0,
        "mean_target_met": 1.0,
        "confidence_target_met": 2.0,
    }[tier]
    cnor = metric(
        validation,
        "candidateNormalizedOcclusionRecall",
        "candidate_normalized_occlusion_recall",
    )
    useful = metric(validation, "agg_useful_cull")
    lcb = metric(
        validation,
        "aggregateWeightedRecallLowerConfidenceBound",
        "weighted_recall_lower_confidence_bound",
    )
    wr = metric(validation, "aggregateWeightedRecall", "agg_weighted_recall")
    if tier_rank > 0:
        return (tier_rank, cnor, useful, lcb, wr, -metric(validation, "avg_pred_count"))
    return (tier_rank, lcb, wr, cnor, useful, -metric(validation, "avg_pred_count"))


def summarize() -> dict[str, Any]:
    selections: dict[str, Any] = {}
    for scene in SCENES:
        rows = [control_row(scene), *(
            result_row(config) for config in configs_for_scene(scene)
        )]
        selected = max(rows, key=selection_key)
        selected_config = next(
            (config for config in configs_for_scene(scene) if config.name == selected["config"]),
            None,
        )
        if selected_config is not None and selected_config.relation_k > 8:
            same_tier_k8 = [
                row for row in rows
                if row["qualificationTier"] == selected["qualificationTier"]
                and any(
                    config.name == row["config"] and config.relation_k == 8
                    for config in configs_for_scene(scene)
                )
            ]
            if same_tier_k8:
                simpler = max(same_tier_k8, key=selection_key)
                selected_v = selected["validation"]
                simpler_v = simpler["validation"]
                cnor_gain = metric(
                    selected_v,
                    "candidateNormalizedOcclusionRecall",
                    "candidate_normalized_occlusion_recall",
                ) - metric(
                    simpler_v,
                    "candidateNormalizedOcclusionRecall",
                    "candidate_normalized_occlusion_recall",
                )
                useful_gain = metric(selected_v, "agg_useful_cull") - metric(
                    simpler_v, "agg_useful_cull"
                )
                if cnor_gain < 0.01 and useful_gain < 0.01:
                    selected = simpler
        selections[scene] = {
            "selectedConfig": selected["config"],
            "selectionTier": selected["qualificationTier"],
            "rows": rows,
        }
    payload = {
        "schema": "pvs-standard-graphics-64k-targeted-refinement-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "qualificationPolicy": {
            "confidenceTarget": "WR > 0.99 and one-sided 95% LCB > 0.99",
            "retentionFloor": "WR > 0.99",
            "cnorRole": "optimization target after qualification; not a hard gate",
            "relationComplexityTie": (
                "prefer K=8 when a higher-K member in the same qualification tier "
                "improves both CNOR and Useful Cull by less than 0.01"
            ),
        },
        "selections": selections,
        "testRead": False,
    }
    write_json(RESULT_ROOT / "scan_selection.json", payload)
    return payload


def selected_config(scene: str) -> RefinementConfig | None:
    summary = read_json(RESULT_ROOT / "scan_selection.json")
    name = summary["selections"][scene]["selectedConfig"]
    if name == CONTROL_NAME:
        return None
    return next(config for config in configs_for_scene(scene) if config.name == name)


def preflight_sources(scenes: Sequence[str]) -> dict[str, Any]:
    reports = {scene: preflight(scene) for scene in scenes}
    missing_summaries = []
    missing_checkpoints = []
    for scene in scenes:
        for seed in SEEDS:
            summary = base_member(scene, seed) / "calibration_ready_summary.json"
            if not summary.is_file():
                missing_summaries.append(str(summary))
        for config in configs_for_scene(scene):
            source = config_source_checkpoint(config, PILOT_SEEDS[scene])
            if not source.is_file():
                missing_checkpoints.append(str(source))
            relation_meta = relation_dir(config) / "relation_csr_meta.json"
            if not relation_meta.is_file():
                missing_checkpoints.append(str(relation_meta))
    if missing_summaries or missing_checkpoints:
        raise FileNotFoundError({
            "missingFullSummaries": missing_summaries,
            "missingPilotSources": missing_checkpoints,
        })
    return {
        "schema": "pvs-standard-graphics-64k-targeted-refinement-preflight-v1",
        "experiment": EXPERIMENT,
        "scenes": reports,
        "pilotSources": {
            scene: str(source_checkpoint(scene, PILOT_SEEDS[scene]).resolve())
            for scene in scenes
        },
        "configSources": {
            config.name: str(
                config_source_checkpoint(config, PILOT_SEEDS[config.scene]).resolve()
            )
            for config in CONFIGS
            if config.scene in scenes
        },
        "testRead": False,
    }


def run_logged(name: str, command: list[str], gpu: int, log_root: Path) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}.stdout.log"
    stderr_path = log_root / f"{name}.stderr.log"
    record_path = log_root / f"{name}.command.json"
    record = {
        "schema": "pvs-standard-graphics-64k-targeted-refinement-command-v1",
        "experiment": EXPERIMENT,
        "name": name,
        "gpuId": gpu,
        "command": command,
        "testRead": False,
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(record_path, record)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    record["returnCode"] = int(result.returncode)
    record["updatedAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(record_path, record)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_parallel(
    jobs: list[tuple[str, list[str]]],
    gpu_slots: Sequence[int],
    log_root: Path,
) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "preflight", "scan", "summarize", "final"))
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=list(SCENES))
    parser.add_argument("--gpu-slots", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    scenes = tuple(args.scenes)
    if args.mode == "plan":
        print(json.dumps({
            "experiment": EXPERIMENT,
            "scenes": list(scenes),
            "pilotSeeds": {scene: PILOT_SEEDS[scene] for scene in scenes},
            "configs": [asdict(config) for config in CONFIGS if config.scene in scenes],
            "scanUpdatesPerMember": 900,
            "finalUpdatesPerMember": 1800,
            "testRead": False,
        }, ensure_ascii=False, indent=2))
        return
    if args.mode == "summarize":
        print(json.dumps(summarize(), ensure_ascii=False, indent=2))
        return
    report = preflight_sources(scenes)
    if args.mode == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    jobs: list[tuple[str, list[str]]] = []
    if args.mode == "scan":
        for scene in scenes:
            seed = PILOT_SEEDS[scene]
            for config in configs_for_scene(scene):
                output = member_dir("scan", config, seed)
                if not completed(output):
                    jobs.append((
                        f"{config.name}_seed{seed}",
                        refinement_command(
                            config,
                            seed,
                            output,
                            config_source_checkpoint(config, seed),
                            final=False,
                        ),
                    ))
        run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/scan")
        return
    for scene in scenes:
        config = selected_config(scene)
        if config is None:
            continue
        for seed in SEEDS:
            output = member_dir("final", config, seed)
            if not completed(output):
                jobs.append((
                    f"{config.name}_seed{seed}",
                    refinement_command(
                        config,
                        seed,
                        output,
                        config_source_checkpoint(config, seed),
                        final=True,
                    ),
                ))
    run_parallel(jobs, args.gpu_slots, RESULT_ROOT / "logs/final")


if __name__ == "__main__":
    main()
