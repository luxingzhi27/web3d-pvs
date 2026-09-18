from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.dataset.v5.synthetic_scene_manifest import (
    generate_synthetic_scene_manifest,
    materialize_scene_aabbs,
)


class SyntheticSceneManifestTest(unittest.TestCase):
    def test_catalog_is_deterministic_and_has_fixed_family_split_counts(self) -> None:
        first = generate_synthetic_scene_manifest(20260918)
        second = generate_synthetic_scene_manifest(20260918)
        self.assertEqual(first, second)
        self.assertEqual(first["sceneCount"], 120)
        self.assertEqual(first["splitCounts"], {"train": 96, "validation": 12, "diagnostic": 12})
        families = {family: 0 for family in first["structureFamilies"]}
        for scene in first["scenes"]:
            families[scene["structureFamily"]] += 1
        self.assertEqual(families, {family: 24 for family in families})

    def test_geometry_materialization_is_seed_deterministic(self) -> None:
        scene = generate_synthetic_scene_manifest()["scenes"][37]
        ids_a, boxes_a = materialize_scene_aabbs(scene)
        ids_b, boxes_b = materialize_scene_aabbs(scene)
        self.assertTrue(np.array_equal(ids_a, ids_b))
        self.assertTrue(np.array_equal(boxes_a, boxes_b))
        self.assertGreaterEqual(boxes_a.shape[0], 256)
        self.assertLessEqual(boxes_a.shape[0], 4096)
        self.assertTrue(np.all(boxes_a[:, 0] < boxes_a[:, 1]))


if __name__ == "__main__":
    unittest.main()
