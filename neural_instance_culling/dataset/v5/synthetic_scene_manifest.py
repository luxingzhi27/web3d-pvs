"""Deterministic 120-scene procedural catalog for the V5 training protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schemas import (
    SYNTHETIC_SCENE_SCHEMA,
    SchemaError,
    read_json_manifest,
    validate_synthetic_manifest,
    write_json_manifest,
)


STRUCTURE_FAMILIES = (
    "rooms_doorways_long_corridors",
    "multi_floor_campus_courtyards_colonnades",
    "city_street_canyons_dense_mixed_height",
    "industrial_pipes_equipment_beams_platforms",
    "repeated_and_unique_cluttered_units",
)
SPLIT_COUNTS = {"train": 96, "validation": 12, "diagnostic": 12}
SCENE_COUNT = 120
SCENES_PER_FAMILY = 24


def _scene_seed(base_seed: int, family_index: int, local_seed: int) -> int:
    return int(base_seed) + family_index * 100_000 + local_seed


def _scene_randomization(seed: int, family_index: int, local_seed: int) -> tuple[int, dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    unit_count = int(rng.integers(256, 4097))
    granularity = int(rng.choice(np.asarray([32, 64, 128], dtype=np.int64)))
    randomization = {
        "scaleRange": [round(float(rng.uniform(0.4, 1.2)), 6), round(float(rng.uniform(1.4, 5.0)), 6)],
        "density": round(float(rng.uniform(0.25, 1.0)), 6),
        "layers": int(rng.integers(1, 8)),
        "channelWidth": round(float(rng.uniform(1.5, 12.0)), 6),
        "occlusionDepth": round(float(rng.uniform(2.0, 80.0)), 6),
        "unitSizeDistribution": round(float(rng.uniform(0.1, 0.9)), 6),
        "repeatRate": round(float(rng.uniform(0.0, 0.95)), 6),
    }
    recipe = {
        "familyIndex": family_index,
        "localSeed": local_seed,
        "primitiveSet": "box_cylinder_beam_panel",
        "meshBank": "procedural_v5_primitives_only",
    }
    return unit_count, randomization | {"streamingGranularityKiB": granularity}, recipe


def generate_synthetic_scene_manifest(base_seed: int = 20260918) -> dict[str, Any]:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base_seed must be a non-negative integer")
    entries: list[dict[str, Any]] = []
    ranked: list[tuple[int, int, int]] = []
    for family_index, family in enumerate(STRUCTURE_FAMILIES):
        for local_seed in range(SCENES_PER_FAMILY):
            generator_seed = _scene_seed(base_seed, family_index, local_seed)
            ranked.append((_scene_seed(base_seed, family_index, local_seed), family_index, local_seed))
            unit_count, randomization, recipe = _scene_randomization(
                generator_seed, family_index, local_seed
            )
            entries.append(
                {
                    "sceneId": f"synthetic_v5_f{family_index:02d}_s{local_seed:02d}",
                    "structureFamily": family,
                    "localSeed": local_seed,
                    "generatorSeed": generator_seed,
                    "split": "",
                    "unitCount": unit_count,
                    "streamingGranularityKiB": randomization.pop("streamingGranularityKiB"),
                    "randomization": randomization,
                    "geometryRecipe": recipe,
                }
            )
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    split_by_key: dict[tuple[int, int], str] = {}
    for index, (_generator_seed, family_index, local_seed) in enumerate(ranked):
        split = "train" if index % 10 < 8 else "validation" if index % 10 == 8 else "diagnostic"
        split_by_key[(family_index, local_seed)] = split
    for entry in entries:
        family_index = int(entry["geometryRecipe"]["familyIndex"])
        entry["split"] = split_by_key[(family_index, int(entry["localSeed"]))]
    manifest = {
        "schema": SYNTHETIC_SCENE_SCHEMA,
        "version": 1,
        "method": "GCOF-PVS-V5",
        "baseSeed": base_seed,
        "generator": "procedural_primitive_mesh_bank_v5",
        "sceneCount": SCENE_COUNT,
        "scenesPerFamily": SCENES_PER_FAMILY,
        "structureFamilies": list(STRUCTURE_FAMILIES),
        "splitCounts": dict(SPLIT_COUNTS),
        "splitProtocol": {
            "strategy": "sorted_generator_seed_mod_10",
            "key": "ascending generatorSeed",
            "poseIndependent": True,
            "seedDisjoint": True,
            "assignmentOrder": ["train", "validation", "diagnostic"],
        },
        "scenes": entries,
    }
    validate_synthetic_manifest(manifest)
    return manifest


def materialize_scene_aabbs(scene: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic structured AABBs without browser rasterization."""

    required = {"sceneId", "structureFamily", "generatorSeed", "unitCount", "geometryRecipe"}
    if not required.issubset(scene):
        raise SchemaError("synthetic scene entry lacks geometry recipe fields")
    count = int(scene["unitCount"])
    seed = int(scene["generatorSeed"])
    family_index = int(scene["geometryRecipe"]["familyIndex"])
    rng = np.random.default_rng(seed)
    # A deterministic family-specific spatial scaffold gives the manifest
    # useful structured geometry while keeping the model input AABB-only.
    if family_index == 0:
        centers = rng.uniform([-60.0, -5.0, -20.0], [60.0, 25.0, 20.0], size=(count, 3))
        centers[:, 1] = np.floor(centers[:, 1] / 3.0) * 3.0
    elif family_index == 1:
        centers = rng.uniform([-70.0, 0.0, -70.0], [70.0, 45.0, 70.0], size=(count, 3))
        centers[:, 1] = np.floor(centers[:, 1] / 6.0) * 6.0
    elif family_index == 2:
        centers = rng.uniform([-100.0, 0.0, -100.0], [100.0, 80.0, 100.0], size=(count, 3))
        centers[:, [0, 2]] = np.round(centers[:, [0, 2]] / 8.0) * 8.0
    elif family_index == 3:
        centers = rng.uniform([-65.0, 0.0, -65.0], [65.0, 60.0, 65.0], size=(count, 3))
        centers[:, 1] = np.floor(centers[:, 1] / 4.0) * 4.0
    else:
        centers = rng.uniform([-50.0, -10.0, -50.0], [50.0, 50.0, 50.0], size=(count, 3))
    half_sizes = rng.uniform(0.08, 3.0, size=(count, 3))
    half_sizes *= rng.uniform(0.6, 1.8, size=(count, 1))
    aabbs = np.stack([centers - half_sizes, centers + half_sizes], axis=1).astype(np.float32)
    return np.arange(count, dtype=np.uint64), aabbs


def write_synthetic_scene_manifest(path: str | Path, base_seed: int = 20260918) -> Path:
    return write_json_manifest(path, generate_synthetic_scene_manifest(base_seed))


def load_synthetic_scene_manifest(path: str | Path) -> dict[str, Any]:
    return validate_synthetic_manifest(read_json_manifest(path))
