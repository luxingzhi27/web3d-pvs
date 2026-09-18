"""Shared paths and fixed preprocessing contract for Connected-SAH scenes.

This module contains configuration only.  It deliberately does not inspect
the filesystem or start a preprocessing job when imported, so a training
runner can reuse the same paths without triggering a long task.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "pvs_v4_standard_graphics_connected_sah_128k_v1"
SEEDS = (20260801, 20260802, 20260803)
DEFAULT_SCENES = ("sponza_128k", "viking_village_128k", "bigcity_128k")

TARGET_UNIT_KIB = 128
POINTS_PER_GLB = 1024
MODEL_FOV_Y_DEG = 66.0
FRONTEND_RENDER_FOV_Y_DEG = 60.0
VIEWCELL_SHAPE = "horizontal_disk"
VIEWCELL_RADIUS_M = 0.75
SUBPOSES_PER_VIEWCELL = 32
REPRESENTATIVE_SUBPOSES_PER_VIEWCELL = 9
RELATION_K = 8
DEPTH_WIDTH = 320
DEPTH_HEIGHT = 180
DEPTH_MAX_LAYERS = 6

DATASET_OUT = ROOT / "neural_instance_culling" / "dataset" / "out"
OUTPUT_ROOT = DATASET_OUT / "standard_graphics_connected_sah_128k_v1"
GEOMETRY_ENCODER = DATASET_OUT / "fixed_geometry_features_hkust_v3" / "geometry_encoder.pt"
SAMPLING_V2_ROOT = DATASET_OUT / "standard_graphics_scene_sampling_v2"

_SPLIT_COUNTS = {
    "sponza_128k": {
        "train": 4800,
        "calibration": 528,
        "validation": 672,
        "test": 672,
        "guard": 0,
    },
    "viking_village_128k": {
        "train": 1944,
        "calibration": 216,
        "validation": 276,
        "test": 276,
        "guard": 0,
    },
    "bigcity_128k": {
        "train": 11580,
        "calibration": 1284,
        "validation": 1608,
        "test": 1608,
        "guard": 0,
    },
}

_SOURCE_FILES = {
    "sponza_128k": (
        ROOT
        / "neural_instance_culling"
        / "dataset"
        / "out"
        / "standard_graphics_sources"
        / "gltf_sample_assets"
        / "Models"
        / "Sponza"
        / "glTF"
        / "Sponza.gltf"
    ),
    "viking_village_128k": (
        ROOT
        / "neural_instance_culling"
        / "dataset"
        / "out"
        / "standard_graphics_sources"
        / "viking_village"
        / "VikingVillage.glb"
    ),
    "bigcity_128k": (
        ROOT
        / "neural_instance_culling"
        / "dataset"
        / "out"
        / "standard_graphics_sources"
        / "bigcity"
        / "scene.gltf"
    ),
}

_DEPTH_SHARD_COUNTS = {
    "sponza_128k": 32,
    "viking_village_128k": 16,
    "bigcity_128k": 64,
}


def _scene_config(
    scene: str,
    *,
    source_scene: str | None = None,
    output_root: Path = OUTPUT_ROOT,
    target_unit_kib: int = TARGET_UNIT_KIB,
    max_components_per_unit: int | None = None,
    experiment: str = EXPERIMENT,
) -> dict[str, object]:
    source_scene = source_scene or scene
    scene_root = output_root / scene
    assets = scene_root / "assets"
    dataset = scene_root / "pose_csr"
    depth_root = scene_root / "depth_manifest"
    split_counts = dict(_SPLIT_COUNTS[source_scene])
    return {
        "scene": scene,
        "source": _SOURCE_FILES[source_scene],
        "scene_root": scene_root,
        "assets": assets,
        "conversion_manifest": assets / "conversionManifest.json",
        "runtime_meta": assets / "runtimeVisibilityMeta.json",
        "glb_index": assets / "glbIndex.json",
        "glb_root": assets,
        "geometry_encoder": GEOMETRY_ENCODER,
        "glb_points": scene_root / "glb_points_1024.bin",
        "glb_points_meta": scene_root / "glb_points_1024_meta.json",
        "geometry": scene_root / "instance_geo_features_fp16.bin",
        "geometry_meta": scene_root / "instance_geo_features_fp16.json",
        "pose_plan": SAMPLING_V2_ROOT / source_scene / "viewcell_pose_plan.jsonl",
        "color_id": scene_root / "color_id",
        "dataset": dataset,
        "source_render_manifest": scene_root / "source_render_manifest.json",
        "depth_root": depth_root,
        "depth_manifest": depth_root / "manifest.json",
        "depth_manifests": depth_root / "manifests",
        "depth_cache": depth_root / "cache",
        "relation": scene_root / "relation_csr",
        "split_counts": split_counts,
        # ``splits`` follows the existing Full-runner convention.  Both names
        # describe the same fixed split contract; neither is inferred at run time.
        "splits": dict(split_counts),
        "depth_shard_count": _DEPTH_SHARD_COUNTS[source_scene],
        "color_id_shard_count": 16,
        "color_id_parallel": 4,
        "target_unit_kib": target_unit_kib,
        "max_components_per_unit": max_components_per_unit,
        "experiment": experiment,
        "dataset_experiment": f"{scene}_connected_sah_{target_unit_kib}k_fov66_sampling_v2",
    }


SCENES = {
    scene: _scene_config(scene)
    for scene in DEFAULT_SCENES
}
SCENES["bigcity_64k"] = _scene_config(
    "bigcity_64k",
    source_scene="bigcity_128k",
    output_root=DATASET_OUT / "standard_graphics_connected_sah_64k_v1",
    target_unit_kib=64,
    max_components_per_unit=128,
    experiment="pvs_v4_bigcity_connected_sah_64k_v1",
)
SCENES["sponza_64k"] = _scene_config(
    "sponza_64k",
    source_scene="sponza_128k",
    output_root=DATASET_OUT / "standard_graphics_connected_sah_64k_v1",
    target_unit_kib=64,
    max_components_per_unit=64,
    experiment="pvs_v4_sponza_connected_sah_64k_v1",
)
SCENES["viking_village_64k"] = _scene_config(
    "viking_village_64k",
    source_scene="viking_village_128k",
    output_root=DATASET_OUT / "standard_graphics_connected_sah_64k_v1",
    target_unit_kib=64,
    max_components_per_unit=128,
    experiment="pvs_v4_viking_connected_sah_64k_v1",
)
