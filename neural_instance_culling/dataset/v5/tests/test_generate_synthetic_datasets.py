from __future__ import annotations

import json
import math
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
    build_viewcells,
    expand_probe_distance_grid,
    generate_scene_dataset,
    raycast_primitive_hits,
    select_catalog_scene_ids,
)
from neural_instance_culling.dataset.v5.permissions import make_universal_training_policy
from neural_instance_culling.dataset.v5.synthetic_scene_manifest import generate_synthetic_scene_manifest
from neural_instance_culling.model.v5.training_data import V5SceneTrainingData
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset, frustum_candidate_ids_for_pose


def _small_entry(unit_count: int = 64) -> dict:
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
            "channelWidth": 8.0,
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
    def test_scene_shards_are_disjoint_and_cover_the_frozen_catalog(self) -> None:
        catalog = generate_synthetic_scene_manifest(20260918)
        entries = {str(entry["sceneId"]): entry for entry in catalog["scenes"]}
        shards = [
            select_catalog_scene_ids(
                entries,
                all_scenes=True,
                requested_scene_ids=None,
                shard_index=index,
                shard_count=4,
            )
            for index in range(4)
        ]
        self.assertTrue(all(len(shard) == 30 for shard in shards))
        self.assertEqual(len(set().union(*map(set, shards))), 120)
        self.assertEqual(sum(len(shard) for shard in shards), 120)

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
            entry = _small_entry()
            manifest = generate_scene_dataset(
                entry,
                root,
                device="cpu",
                views_per_scene=4,
                render_width=16,
                render_height=9,
                allow_small_scene=True,
            )
            scene_root = root / "synthetic_test_scene"
            self.assertEqual(manifest["viewCells"]["count"], 4)
            self.assertEqual(set(manifest["viewCells"]["shapes"]), {"disk", "oriented_box"})
            self.assertTrue((scene_root / "compiled/surface").is_dir())
            self.assertTrue((scene_root / "compiled/relation").is_dir())
            self.assertTrue((scene_root / "compiled/probes").is_dir())
            dataset = PoseCSRDataset(scene_root / "pose_csr", int(entry["unitCount"]))
            self.assertEqual(dataset.poses.size, 4)
            candidate_counts = []
            scene = build_primitive_scene(entry, allow_small=True)
            viewcells = build_viewcells(scene, entry, views_per_scene=4)
            tan_y = math.tan(math.radians(66.0) * 0.5)
            tan_x = tan_y * 16.0 / 9.0
            for pose_id in range(dataset.poses.size):
                visible, weights = dataset.visible_slice(pose_id)
                candidates = dataset.candidate_slice(pose_id)
                self.assertTrue(np.isfinite(weights).all())
                self.assertTrue(set(visible.tolist()).issubset(set(candidates.tolist())))
                candidate_counts.append(int(candidates.size))
                cell = viewcells[pose_id]
                backed_positions = cell.subpose_positions - cell.subpose_forwards * cell.back_offset
                expected = np.unique(
                    np.concatenate(
                        [
                            frustum_candidate_ids_for_pose(
                                origin,
                                forward,
                                tan_x,
                                tan_y,
                                scene.aabbs.reshape(scene.unit_count, 6),
                            )
                            for origin, forward in zip(backed_positions, cell.subpose_forwards)
                        ]
                    )
                ).astype(np.uint32)
                np.testing.assert_array_equal(candidates, expected)
                np.testing.assert_allclose(
                    dataset.candidate_camera_world(pose_id),
                    cell.center - cell.forward * cell.back_offset,
                    atol=1.0e-5,
                )
            self.assertTrue(all(count < int(entry["unitCount"]) for count in candidate_counts))
            self.assertGreater(len(set(candidate_counts)), 1)
            pose_meta = (scene_root / "pose_csr/dataset_meta.json").read_text(encoding="utf-8")
            self.assertNotIn("candidateFallbackAllUnitsViewcells", pose_meta)
            probe_manifest = (scene_root / "compiled/probes/external_hit_probe_manifest.json").read_text(encoding="utf-8")
            self.assertFalse((scene_root / "compiled/probes/probe_manifest.json").exists())
            self.assertIn('"rowCount": 36864', probe_manifest)
            probe_data = json.loads(probe_manifest)
            self.assertEqual(probe_data["files"]["unitIds"], "probe_unit_ids_uint32.bin")
            self.assertEqual(probe_data["files"]["directions"], "probe_directions_fp32.bin")
            surface_manifest = (scene_root / "compiled/surface/surface_manifest.json").read_text(encoding="utf-8")
            self.assertIn('"pointsPerUnit": 256', surface_manifest)

            training_data = V5SceneTrainingData(
                scene_id=manifest["sceneId"],
                pose_dataset=scene_root / "pose_csr",
                runtime_meta=scene_root / "runtimeVisibilityMeta.json",
                compiled_dir=scene_root / "compiled",
                viewcell_shape="horizontal_disk",
                viewcell_half_extent_m=(1.0, 1.0, 1.0),
                camera_clip_m=(0.01, 1_000_000.0),
                probe_policy=make_universal_training_policy([manifest["sceneId"]]),
                region_manifest=scene_root / "scene_manifest.json",
            )
            pose_batch = training_data.sample_pose_batch(np.random.default_rng(7), pose_count=2)
            probe_batch = training_data.sample_probe_batch(np.random.default_rng(11), observation_count=32)
            self.assertGreater(pose_batch.candidate_ids.size, 0)
            self.assertEqual(pose_batch.query_geometry.shape[1], 16)
            self.assertEqual(probe_batch.unit_ids.shape, (32,))
            self.assertEqual(probe_batch.directions.shape, (32, 3))
            self.assertTrue(np.isfinite(probe_batch.distances).all())


if __name__ == "__main__":
    unittest.main()
