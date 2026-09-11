"""Registered configuration for the paper AABB-plus-ray MLP baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any


DATA_ROOT = Path("/mnt/sda/rhyang/slm")
EXPERIMENT = "pvs_aabb_ray_mlp_formal_v1"
FORMAL_SEEDS = (20260801, 20260802, 20260803)
FORMAL_EPOCHS = 40
FORMAL_STEPS_PER_EPOCH = 900
SCAN_LEARNING_RATES = (2e-4, 1e-3)
SCAN_EPOCHS = 6
SCAN_STEPS_PER_EPOCH = 300
CALIBRATION_BOOTSTRAP_REPLICATES = 10000
WEIGHTED_RECALL_TARGET = 0.99

LOSS_CONFIG: dict[str, Any] = {
    "lossVariant": "pose_balanced_rvl_contrastive",
    "rvlRecallGuardWeight": 0.30,
    "rvlRecallTarget": 0.99,
    "rvlRecallTemperature": 0.05,
    "rvlPoseCvarFraction": 0.25,
    "rvlPoseCvarWeight": 0.25,
    "sharedTailSeparationWeight": 0.20,
    "tailRampFraction": 0.15,
    "tailPositiveMassFraction": 0.005,
    "tailPositiveCountCap": 64,
    "tailNegativeFraction": 0.01,
    "tailNegativeCountCap": 256,
    "tailMargin": 0.50,
    "tailTemperature": 0.25,
    "positiveImportanceFloor": 0.5,
    "positiveImportancePower": 0.5,
}

SCENES: dict[str, dict[str, str]] = {
    "hkust_v3": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
        "runtimeMeta": "hkust-v3/assets/runtimeVisibilityMeta.json",
        "glbIndex": "hkust-v3/assets/glbIndex.json",
        "glbRoot": "hkust-v3/assets",
    },
    "ifcbench_fantasy_metropolis": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "runtimeMeta": "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json",
        "glbIndex": "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json",
        "glbRoot": "ifcbench_fantasy_metropolis_instanced_v2/assets",
    },
    "sponza_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_sponza_standard_graphics_128k_fov66_v1",
        "runtimeMeta": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/runtimeVisibilityMeta.json",
        "glbIndex": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets/glbIndex.json",
        "glbRoot": "neural_instance_culling/dataset/out/standard_graphics_scenes/sponza_128k/assets",
    },
    "bigcity_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_bigcity_standard_graphics_128k_fov66_v1",
        "runtimeMeta": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/runtimeVisibilityMeta.json",
        "glbIndex": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets/glbIndex.json",
        "glbRoot": "neural_instance_culling/dataset/out/standard_graphics_scenes/bigcity_128k/assets",
    },
    "viking_village_128k": {
        "dataset": "neural_instance_culling/dataset/out/pose_csr_viking_village_standard_graphics_128k_fov66_v1",
        "runtimeMeta": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/runtimeVisibilityMeta.json",
        "glbIndex": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets/glbIndex.json",
        "glbRoot": "neural_instance_culling/dataset/out/standard_graphics_scenes/viking_village_128k/assets",
    },
}


def scene_paths(data_root: str | Path, scene: str) -> dict[str, Path]:
    if scene not in SCENES:
        raise KeyError(f"unknown registered scene {scene!r}; choose from {sorted(SCENES)}")
    root = Path(data_root).resolve()
    return {key: root / value for key, value in SCENES[scene].items()}


__all__ = [
    "CALIBRATION_BOOTSTRAP_REPLICATES",
    "DATA_ROOT",
    "EXPERIMENT",
    "FORMAL_EPOCHS",
    "FORMAL_SEEDS",
    "FORMAL_STEPS_PER_EPOCH",
    "LOSS_CONFIG",
    "SCAN_EPOCHS",
    "SCAN_LEARNING_RATES",
    "SCAN_STEPS_PER_EPOCH",
    "SCENES",
    "WEIGHTED_RECALL_TARGET",
    "scene_paths",
]
