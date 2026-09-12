from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import run_geometry_shell_hzb_paper as orchestrator  # noqa: E402


def _argument(command: tuple[str, ...], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _fixture_scene(root: Path, key: str = "hkust") -> orchestrator.SceneSpec:
    scene_names = {
        "hkust": "hkust-v3",
        "ifcbench": "ifcbench_fantasy_metropolis_instanced_v2",
        "sponza_128k": "sponza",
        "bigcity_128k": "neuralpvs_bigcity",
    }
    return orchestrator.SceneSpec(
        key=key,
        scene_name=scene_names[key],
        dataset_dir=root / f"{key}-dataset",
        region_dataset_dir=root / f"{key}-region",
        runtime_meta=root / f"{key}-runtime.json",
        glb_index=root / f"{key}-glb-index.json",
        glb_root=root / f"{key}-assets",
        shell_dirs={
            "lossless": root / f"{key}-lossless",
            "equal_asset": root / f"{key}-equal-asset",
        },
        timing_plan=root / f"{key}-timing.json",
        point60_plan_dir=root / f"{key}-point60-plans",
        representative_plan=root / f"{key}-representatives.jsonl",
        expected_splits=orchestrator.EXPECTED_SPLITS[key],
    )


class GeometryShellHZBPaperOrchestratorTests(unittest.TestCase):
    def test_region_preflight_accepts_current_pose_csr_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset_meta.json").write_text(
                json.dumps({
                    "schema": "pose-csr-explicit-four-way-split-v1",
                    "viewcellCount": 2,
                    "poseCount": 2,
                    "numInstances": 3,
                }),
                encoding="utf-8",
            )
            (root / "subpose_offsets.bin").write_bytes(struct.pack("<3Q", 0, 1, 2))
            (root / "subpose_camera_forward.bin").write_bytes(struct.pack("<6f", *([0.0] * 6)))
            (root / "subpose_camera_pos.bin").write_bytes(struct.pack("<6f", *([0.0] * 6)))
            (root / "subpose_pose_indices.bin").write_bytes(struct.pack("<2I", 0, 1))
            (root / "viewcell_centers.bin").write_bytes(struct.pack("<6f", *([0.0] * 6)))
            result = orchestrator._check_region_assets(root, 2, 3)
        self.assertEqual(result["schema"], "pose-csr-explicit-four-way-split-v1")
        self.assertEqual(result["subposeCount"], 2)

    def test_registered_matrix_has_the_requested_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hkust = _fixture_scene(root)
            ifcbench = _fixture_scene(root, "ifcbench")
            sponza = _fixture_scene(root, "sponza_128k")
            bigcity = _fixture_scene(root, "bigcity_128k")
            calibration = orchestrator.build_calibration_tasks(hkust, root / "run")
            self.assertEqual(len(calibration), 8)
            self.assertEqual(
                {(task.config["width"], task.config["height"], task.config["depthBiasM"]) for task in calibration},
                {
                    (512, 288, 0.01),
                    (512, 288, 1.0),
                    (512, 288, 10.0),
                    (512, 288, 100.0),
                    (1024, 576, 0.01),
                    (1024, 576, 1.0),
                    (1024, 576, 10.0),
                    (1024, 576, 100.0),
                },
            )
            self.assertTrue(all(task.config["calibrationInvocations"] == 1 for task in calibration))
            self.assertTrue(all("--timing-rounds" in task.commands[0][1] for task in calibration))
            self.assertTrue(all(task.config["runnerTimingRounds"] == 1 for task in calibration))
            self.assertTrue(all(_argument(task.commands[0][1], "--timing-rounds") == "1" for task in calibration))
            self.assertEqual(
                {(_argument(task.commands[0][1], "--width"), _argument(task.commands[0][1], "--height")) for task in calibration},
                {("512", "288"), ("1024", "576")},
            )
            self.assertTrue(all("--allow-software-gpu" not in task.commands[0][1] for task in calibration))

            selection = {
                "schema": orchestrator.SELECTION_SCHEMA,
                "selectionSplit": "calibration",
                "testRead": False,
                "selected": {
                    "assetVariant": "lossless",
                    "width": 512,
                    "height": 576,
                    "depthBiasM": 0.001,
                },
            }
            hkust_test = orchestrator.build_test_tasks(hkust, root / "run", selection)
            ifc_test = orchestrator.build_test_tasks(ifcbench, root / "run", selection)
            sponza_test = orchestrator.build_test_tasks(sponza, root / "run", selection)
            bigcity_test = orchestrator.build_test_tasks(bigcity, root / "run", selection)
            self.assertEqual(len(hkust_test), 8)
            self.assertEqual(len(ifc_test), 4)
            self.assertEqual(len(sponza_test), 4)
            self.assertEqual(len(bigcity_test), 4)
            self.assertEqual(
                {task.config["regionSampleCount"] for task in hkust_test},
                {1, 5, 9, 0},
            )
            self.assertEqual({task.config["regionSampleCount"] for task in ifc_test}, {1, 0})
            self.assertEqual({task.config["regionSampleCount"] for task in sponza_test}, {1, 0})
            self.assertEqual({task.config["regionSampleCount"] for task in bigcity_test}, {1, 0})
            self.assertTrue(all(task.config["testRead"] for task in hkust_test + ifc_test))
            self.assertTrue(all(task.config["runnerTimingRounds"] == 1 for task in hkust_test + ifc_test))
            self.assertTrue(
                all(_argument(task.commands[0][1], "--timing-rounds") == "1" for task in hkust_test + ifc_test)
            )

            timing = orchestrator.build_timing_tasks(hkust, root / "run", selection)
            self.assertEqual(len(timing), 2)
            self.assertTrue(all(task.config["poseCount"] == 120 for task in timing))
            self.assertTrue(all(task.config["timingRounds"] == 5 for task in timing))
            command = timing[0].commands[0][1]
            self.assertEqual(_argument(command, "--timing-rounds"), "5")
            self.assertEqual(_argument(command, "--pose-index-plan"), str(hkust.timing_plan.resolve()))
            self.assertIn("--require-hardware-gpu", command)

    def test_standard_graphics_scene_aliases_are_registered(self) -> None:
        self.assertEqual(
            orchestrator._parse_scene_keys("sponza,bigcity_128k"),
            ["sponza_128k", "bigcity_128k"],
        )

    def test_existing_complete_task_is_skipped_and_partial_task_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "task"
            task = orchestrator.TaskSpec(
                phase="timing",
                scene="hkust",
                name="fixture",
                directory=directory,
                commands=(),
                expected_artifacts=("result.json", "stdout.log", "stderr.log"),
                config={},
            )
            directory.mkdir()
            (directory / "result.json").write_text("{}", encoding="utf-8")
            (directory / "stdout.log").write_text("", encoding="utf-8")
            (directory / "stderr.log").write_text("", encoding="utf-8")
            self.assertEqual(orchestrator._task_state(task), "complete")

            partial = Path(temporary) / "partial"
            partial.mkdir()
            (partial / "stdout.log").write_text("started", encoding="utf-8")
            partial_task = task.__class__(
                phase=task.phase,
                scene=task.scene,
                name="partial",
                directory=partial,
                commands=(),
                expected_artifacts=task.expected_artifacts,
                config={},
            )
            with self.assertRaises(orchestrator.IncompleteArtifactError):
                orchestrator._task_state(partial_task)

    def test_formal_false_stops_before_evaluator_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "evaluator-ran"
            result = root / "task" / "result.json"
            write_result = (
                "import json; "
                f"json.dump({{'formalReady': False}}, open({str(result)!r}, 'w'))"
            )
            write_marker = f"open({str(marker)!r}, 'w').write('ran')"
            task = orchestrator.TaskSpec(
                phase="calibrate",
                scene="hkust",
                name="formal-gate",
                directory=root / "task",
                commands=(
                    ("browser-calibration", (sys.executable, "-c", write_result)),
                    ("evaluate-calibration", (sys.executable, "-c", write_marker)),
                ),
                expected_artifacts=("result.json", "stdout.log", "stderr.log"),
                config={},
            )
            with self.assertRaises(orchestrator.FormalArtifactError):
                orchestrator._execute_task(
                    task,
                    dry_run=False,
                    timeout_seconds=30,
                    after_step=lambda current, label: orchestrator._validate_formal_result(current)
                    if label == "browser-calibration" else None,
                )
            self.assertFalse(marker.exists())

    def test_direct_test_mode_requires_selection_before_creating_test_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = orchestrator.parse_args([
                "test",
                "--data-root", str(root),
                "--hzb-root", str(root / "hzb"),
                "--output-root", str(root / "run"),
                "--scenes", "hkust",
                "--dry-run",
            ])
            with self.assertRaises(FileNotFoundError):
                orchestrator.run(args)
            self.assertFalse((root / "run" / "test").exists())

    def test_point60_is_registered_as_nonblocking_raw_gt_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene = _fixture_scene(root)
            scene.point60_plan_dir.mkdir()
            plan = scene.point60_plan_dir / "point60_test.jsonl"
            plan.write_text("{}\n", encoding="utf-8")
            handoff = orchestrator._point60_handoff(
                scene,
                root / "run",
                point60_plan_builder=root / "builder.py",
                sampler=root / "sampler.mjs",
            )
            self.assertFalse(handoff["blocking"])
            self.assertEqual(handoff["status"], "pending-raw-gt")
            self.assertEqual(len(handoff["commands"]), 1)
            command = handoff["commands"][0]["command"]
            self.assertIn("--point60-gt", command)
            self.assertIn("--require-hardware-gpu", command)
            self.assertEqual(handoff["commands"][0]["plan"], str(plan.resolve()))

    def test_all_dry_run_is_cpu_only_and_enumerates_every_task(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary) / "dry-run-output"
            args = orchestrator.parse_args([
                "all",
                "--dry-run",
                "--output-root",
                str(output_root),
            ])
            with redirect_stdout(io.StringIO()):
                summary = orchestrator.run(args)
            self.assertTrue(summary["dryRun"])
            self.assertFalse(summary["formalGpuExecuted"])
            self.assertEqual(len(summary["records"]), 36)
            self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
