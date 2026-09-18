from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from neural_instance_culling.benchmark.run_standard_graphics_connected_sah_preprocessing import (
    CONNECTED_CONVERSION_SCHEMA,
    CONNECTED_PARTITION_SCHEMA,
    FORMAL_MODES,
    depth_manifest_shard_command,
    depth_shard_ranges,
    geometry_commands,
    pose_csr_command,
    relation_command,
    stage_commands,
    write_source_render_manifest,
)
from neural_instance_culling.benchmark.standard_graphics_connected_sah_config import (
    EXPERIMENT,
    DEFAULT_SCENES,
    OUTPUT_ROOT,
    POINTS_PER_GLB,
    RELATION_K,
    REPRESENTATIVE_SUBPOSES_PER_VIEWCELL,
    SCENES,
    SEEDS,
    SUBPOSES_PER_VIEWCELL,
)


class ConnectedSahPreprocessingTests(unittest.TestCase):
    def test_component_aware_packing_schema_is_current(self) -> None:
        self.assertEqual(
            CONNECTED_CONVERSION_SCHEMA,
            "pvs-standard-graphics-scene-connected-sah-pack-v2",
        )
        self.assertEqual(CONNECTED_PARTITION_SCHEMA, "connected-sah-pack-v2")

    def test_shared_config_registers_exact_experiment_and_three_sources(self) -> None:
        self.assertEqual(EXPERIMENT, "pvs_v4_standard_graphics_connected_sah_128k_v1")
        self.assertEqual(SEEDS, (20260801, 20260802, 20260803))
        self.assertEqual(OUTPUT_ROOT.name, "standard_graphics_connected_sah_128k_v1")
        self.assertEqual(DEFAULT_SCENES, ("sponza_128k", "viking_village_128k", "bigcity_128k"))
        self.assertEqual(
            set(SCENES),
            {*DEFAULT_SCENES, "bigcity_64k", "sponza_64k", "viking_village_64k"},
        )
        expected_source_names = {
            "sponza_128k": "Sponza.gltf",
            "viking_village_128k": "VikingVillage.glb",
            "bigcity_128k": "scene.gltf",
            "bigcity_64k": "scene.gltf",
            "sponza_64k": "Sponza.gltf",
            "viking_village_64k": "VikingVillage.glb",
        }
        for scene, config in SCENES.items():
            with self.subTest(scene=scene):
                for key in ("source", "assets", "geometry", "dataset", "relation", "pose_plan"):
                    self.assertIn(key, config)
                self.assertTrue(Path(config["source"]).is_file())
                source = Path(config["source"])
                self.assertEqual(source.name, expected_source_names[scene])
                self.assertIn("standard_graphics_sources", source.parts)
                self.assertEqual(Path(config["assets"]).parent.name, scene)
                self.assertTrue(Path(config["pose_plan"]).is_file())
                self.assertEqual(Path(config["pose_plan"]).name, "viewcell_pose_plan.jsonl")
                self.assertIn("sampling_v2", str(config["pose_plan"]))
                self.assertEqual(config["split_counts"], config["splits"])

    def test_fixed_v2_viewcell_and_split_contract_is_shared(self) -> None:
        expected = {
            "sponza_128k": {"train": 4800, "calibration": 528, "validation": 672, "test": 672, "guard": 0},
            "viking_village_128k": {"train": 1944, "calibration": 216, "validation": 276, "test": 276, "guard": 0},
            "bigcity_128k": {"train": 11580, "calibration": 1284, "validation": 1608, "test": 1608, "guard": 0},
            "bigcity_64k": {"train": 11580, "calibration": 1284, "validation": 1608, "test": 1608, "guard": 0},
            "sponza_64k": {"train": 4800, "calibration": 528, "validation": 672, "test": 672, "guard": 0},
            "viking_village_64k": {"train": 1944, "calibration": 216, "validation": 276, "test": 276, "guard": 0},
        }
        for scene, config in SCENES.items():
            with self.subTest(scene=scene):
                self.assertEqual(config["split_counts"], expected[scene])
        self.assertEqual(SUBPOSES_PER_VIEWCELL, 32)
        self.assertEqual(REPRESENTATIVE_SUBPOSES_PER_VIEWCELL, 9)

    def test_convert_and_geometry_commands_use_new_root_and_1024_points(self) -> None:
        for scene in SCENES:
            with self.subTest(scene=scene):
                convert = stage_commands(scene, "convert")[0][1]
                convert_text = " ".join(convert)
                self.assertIn("write_slm_scene.mjs", convert_text)
                self.assertIn(str(SCENES[scene]["source"]), convert_text)
                self.assertIn(str(SCENES[scene]["assets"]), convert_text)
                self.assertEqual(
                    convert[convert.index("--target-unit-kib") + 1],
                    str(SCENES[scene]["target_unit_kib"]),
                )
                for command in geometry_commands(scene):
                    self.assertEqual(command[command.index("--points-per-glb") + 1], str(POINTS_PER_GLB))
                self.assertIn("prepare_fixed_geometry_features.py", " ".join(geometry_commands(scene)[1]))
                self.assertEqual(geometry_commands(scene)[1][geometry_commands(scene)[1].index("--device") + 1], "cuda")
        bigcity_64k = stage_commands("bigcity_64k", "convert")[0][1]
        self.assertEqual(bigcity_64k[bigcity_64k.index("--max-components-per-unit") + 1], "128")
        viking_64k = stage_commands("viking_village_64k", "convert")[0][1]
        self.assertEqual(viking_64k[viking_64k.index("--max-components-per-unit") + 1], "128")

    def test_color_id_command_uses_existing_plan_and_formal_wrapper(self) -> None:
        command = stage_commands("bigcity_128k", "color-id")[0][1]
        command_text = " ".join(command)
        self.assertIn("run_scene_viewcell_colorid_sampling.mjs", command_text)
        self.assertIn(str(SCENES["bigcity_128k"]["pose_plan"]), command_text)
        self.assertEqual(command[command.index("--subposes-per-viewcell") + 1], "32")
        self.assertEqual(command[command.index("--fov-y") + 1], "66")
        wrapper_source = Path(command[1]).read_text(encoding="utf-8")
        self.assertIn("--require-hardware-gpu", wrapper_source)
        self.assertNotIn("--allow-software-gpu", command)

    def test_pose_csr_is_viewcell_color_id_union(self) -> None:
        command = pose_csr_command("sponza_128k")
        self.assertIn("build_rvc_viewcell_pose_csr.py", " ".join(command))
        self.assertEqual(command[command.index("--source-sampler") + 1], "three_color_id")
        self.assertEqual(command[command.index("--default-fov-y") + 1], "66")
        self.assertNotIn("--allow-candidate-visible-union", command)

    def test_depth_manifest_shards_cover_train_split_and_use_nine_representatives(self) -> None:
        for scene in SCENES:
            with self.subTest(scene=scene):
                ranges = depth_shard_ranges(scene)
                self.assertEqual(len(ranges), SCENES[scene]["depth_shard_count"])
                self.assertEqual(sum(count for _shard, _start, count in ranges), SCENES[scene]["split_counts"]["train"])
                self.assertEqual(ranges[0][1], 0)
                self.assertEqual(ranges[-1][1] + ranges[-1][2], SCENES[scene]["split_counts"]["train"])
                command = depth_manifest_shard_command(scene)
                self.assertIn("shard_triangle_depth_layer_manifest.py", " ".join(command))
                self.assertEqual(
                    command[command.index("--shards") + 1],
                    str(SCENES[scene]["depth_shard_count"]),
                )

    def test_relation_is_train_only_k8(self) -> None:
        command = relation_command("viking_village_128k")
        self.assertEqual(command[command.index("--splits") + 1], "train")
        self.assertEqual(command[command.index("--source-k") + 1], str(RELATION_K))
        self.assertEqual(RELATION_K, 8)
        self.assertNotIn("--surface-fallback-dir", command)

    def test_stage_modes_are_explicit_and_ordered(self) -> None:
        self.assertEqual(
            FORMAL_MODES,
            ("convert", "geometry", "color-id", "pose-csr", "depth-manifest", "shards", "relation"),
        )
        names = [name for name, _command in stage_commands("sponza_128k", "geometry")]
        self.assertEqual(names, ["glb_points_1024", "fixed_geometry_features_96d"])
        depth_names = [name for name, _command in stage_commands("sponza_128k", "depth-manifest")]
        self.assertEqual(depth_names, ["depth_manifest", "depth_manifest_shards"])

    def test_source_render_manifest_is_written_from_new_assets_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / "runtime.json"
            index = root / "glbIndex.json"
            assets = root / "assets"
            assets.mkdir()
            (assets / "unit.glb").write_bytes(b"fixture")
            runtime.write_text("{}", encoding="utf-8")
            index.write_text("{}", encoding="utf-8")
            config = SCENES["sponza_128k"]
            with mock.patch(
                "neural_instance_culling.benchmark.run_standard_graphics_connected_sah_preprocessing._path",
                side_effect=lambda _scene, key: {
                    "runtime_meta": runtime,
                    "glb_index": index,
                    "assets": assets,
                    "source_render_manifest": root / "source_render_manifest.json",
                }[key],
            ), mock.patch(
                "neural_instance_culling.benchmark.run_standard_graphics_connected_sah_preprocessing.build_instance_binding_preflight",
                return_value={"schema": "component-instance-binding-preflight-v1"},
            ) as build_binding:
                output = write_source_render_manifest("sponza_128k")
            self.assertTrue(output.is_file())
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["instanceBindings"]["schema"], "component-instance-binding-preflight-v1")
            build_binding.assert_called_once()
            self.assertEqual(config["scene"], "sponza_128k")


if __name__ == "__main__":
    unittest.main()
