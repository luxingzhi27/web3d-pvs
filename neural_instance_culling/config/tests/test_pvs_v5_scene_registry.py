from __future__ import annotations

import unittest

from neural_instance_culling.config.pvs_v5_scene_registry import (
    build_loso_access,
    load_registry,
)


class PvsV5SceneRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()

    def test_registered_real_scenes_are_ready(self) -> None:
        self.assertEqual(len(self.registry["scenes"]), 5)
        self.assertEqual(
            {scene["viewCell"]["shape"] for scene in self.registry["scenes"]},
            {"horizontal_disk", "camera_aligned_box"},
        )

    def test_loso_held_out_permissions_are_geometry_only(self) -> None:
        access = build_loso_access(self.registry, "big_city_64k")
        held_out = next(scene for scene in access if scene.scene_id == "big_city_64k")
        sources = [scene for scene in access if scene.scene_id != "big_city_64k"]
        self.assertFalse(held_out.allow_train_labels)
        self.assertFalse(held_out.allow_field_probes)
        self.assertTrue(all(scene.allow_train_labels for scene in sources))
        self.assertTrue(all(scene.allow_field_probes for scene in sources))


if __name__ == "__main__":
    unittest.main()
