from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.dataset.v5.build_local_surface_points import build_local_surface_asset
from neural_instance_culling.dataset.v5.generate_synthetic_datasets import (
    PrimitiveScene,
    PrimitiveUnit,
    build_external_hit_probe_asset,
    build_primitive_scene,
    expand_probe_distance_grid,
    generate_scene_dataset,
    raycast_primitive_hits,
)
from neural_instance_culling.dataset.v5.synthetic_scene_manifest import generate_synthetic_scene_manifest
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset


def _small_entry(unit_count: int = 8) -> dict:
    return {
        "sceneId": "synthetic_test_scene",
        "structureFamily": "rooms_doorways_long_corridors",
        "localSeed": 0,
        "generatorSeed": 41,
        "split": "train",
        "unitCount": unit_count,
        "streamingGranularityKiB": 64,
        "randomization": {
            "scaleRange": [1.0, 1.0],
            "density": 0.5,
            "layers": 1,
            "channelWidth": 4.0,
            "occlusionDepth": 20.0,
            "unitSizeDistribution": 0.5,
            "repeatRate": 0.0,
        },
        "geometryRecipe": {
            "familyIndex": 0,
            "localSeed": 0,
            "primitiveSet": "box_cylinder_beam_panel",
            "meshBank": "procedural_v5_primitives_only",
        },
    }


def _box_unit(unit_id: int, x: float) -> PrimitiveUnit:
    from neural_instance_culling.dataset.v5.generate_synthetic_datasets import _box_mesh

    vertices, triangles = _box_mesh([2.0, 2.0, 2.0])
    center = np.asarray([x, 0.0, 0.0], dtype=np.float64)
    world = vertices + center
    return PrimitiveUnit(
        unit_id,
        "box",
        center,
        np.eye(3, dtype=np.float64),
        np.asarray([2.0, 2.0, 2.0], dtype=np.float64),
        vertices,
        triangles,
        unit_id,
        np.stack([world.min(axis=0), world.max(axis=0)]),
    )


class SyntheticDatasetGeneratorTest(unittest.TestCase):
    def test_catalog_split_is_seed_level_and_deterministic(self) -> None:
        first = generate_synthetic_scene_manifest(20260918)
        second = generate_synthetic_scene_manifest(20260918)
        self.assertEqual(first, second)
        self.assertEqual(first["splitCounts"], {"train": 96, "validation": 12, "diagnostic": 12})
        seeds_by_split = {name: set() for name in first["splitCounts"]}
        for scene in first["scenes"]:
            seeds_by_split[scene["split"]].add(scene["generatorSeed"])
        self.assertEqual({key: len(value) for key, value in seeds_by_split.items()}, {"train": 96, "validation": 12, "diagnostic": 12})
        self.assertEqual(sum(len(value) for value in seeds_by_split.values()), 120)

    def test_scene_contains_real_meshes_for_all_four_primitive_kinds(self) -> None:
        for family_index in range(5):
            entry = _small_entry()
            entry["sceneId"] = f"family_{family_index}"
            entry["geometryRecipe"]["familyIndex"] = family_index
            entry["structureFamily"] = generate_synthetic_scene_manifest()["structureFamilies"][family_index]
            scene = build_primitive_scene(entry, allow_small=True)
            self.assertEqual({unit.primitive_type for unit in scene.units}, {"box", "cylinder", "beam", "panel"})
            self.assertTrue(all(unit.local_vertices.shape[0] >= 8 for unit in scene.units))
            self.assertTrue(all(unit.local_triangles.shape[0] >= 12 for unit in scene.units))
            self.assertTrue(np.isfinite(scene.aabbs).all())

    def test_front_unit_wins_and_ignore_skips_the_entire_unit(self) -> None:
        scene = PrimitiveScene(
            "occlusion",
            (_box_unit(0, 3.0), _box_unit(1, 7.0)),
            np.asarray([[2.0, -1.0, -1.0], [8.0, 1.0, 1.0]], dtype=np.float64),
        )
        ids, distances = raycast_primitive_hits(scene, [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]])
        self.assertEqual(ids.tolist(), [0])
        self.assertAlmostEqual(float(distances[0]), 2.0, places=6)
        ids, distances = raycast_primitive_hits(
            scene,
            [[0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0]],
            ignore_unit_ids=[0],
        )
        self.assertEqual(ids.tolist(), [1])
        self.assertAlmostEqual(float(distances[0]), 6.0, places=6)

    def test_single_unit_probe_never_hits_its_own_surface_and_expands_grid(self) -> None:
        scene = PrimitiveScene(
            "probe",
            (_box_unit(0, 0.0),),
            np.asarray([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        )
        surface = build_local_surface_asset(scene.surface_inputs(), sampling_seed=7)
        probes = build_external_hit_probe_asset(scene, surface, scene_split="train", device="cpu")
        self.assertEqual(probes.manifest["rowCount"], 1 * 16 * 36)
        self.assertEqual(set(probes.manifest["files"]), {"unitIds", "directions", "hitDistances", "maxDistances", "startIds", "directionIds"})
        self.assertTrue(np.isnan(probes.hit_distances).all())
        expanded = expand_probe_distance_grid(probes, np.asarray([1.0], dtype=np.float32))
        self.assertEqual(expanded["events"].shape, (16 * 36, 13))
        self.assertFalse(bool(expanded["events"].any()))

    def test_output_is_posecsr_readable_and_visible_is_subset_of_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = generate_scene_dataset(
                _small_entry(),
                root,
                device="cpu",
                views_per_scene=2,
                render_width=16,
                render_height=9,
                allow_small_scene=True,
            )
            scene_root = root / "synthetic_test_scene"
            self.assertEqual(manifest["viewCells"]["count"], 2)
            self.assertEqual(set(manifest["viewCells"]["shapes"]), {"disk", "oriented_box"})
            self.assertTrue((scene_root / "compiled/surface").is_dir())
            self.assertTrue((scene_root / "compiled/relation").is_dir())
            self.assertTrue((scene_root / "compiled/probes").is_dir())
            dataset = PoseCSRDataset(scene_root / "pose_csr", 8)
            self.assertEqual(dataset.poses.size, 2)
            for pose_id in range(dataset.poses.size):
                visible, weights = dataset.visible_slice(pose_id)
                candidates = dataset.candidate_slice(pose_id)
                self.assertTrue(np.isfinite(weights).all())
                self.assertTrue(set(visible.tolist()).issubset(set(candidates.tolist())))
            probe_manifest = (scene_root / "compiled/probes/probe_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"rowCount": 4608', probe_manifest)
            surface_manifest = (scene_root / "compiled/surface/surface_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"pointsPerUnit": 256', surface_manifest)


if __name__ == "__main__":
    unittest.main()
