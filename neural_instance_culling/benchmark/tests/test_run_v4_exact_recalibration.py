from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural_instance_culling.benchmark import run_v4_exact_recalibration as runner


def _argument(command: list[str] | tuple[str, ...], name: str) -> str:
    return command[command.index(name) + 1]


def _write_fixture(root: Path) -> None:
    for scene in runner.SCENES.values():
        paths = runner.scene_paths(root, scene.key)
        paths["dataset"].mkdir(parents=True, exist_ok=True)
        files = {
            "poses": "poses.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "candidateOffsets": "candidate_offsets.bin",
            "candidateIds": "candidate_ids.bin",
            "queryCenterWorld": "query_center_world.bin",
            "candidateCameraWorld": "candidate_camera_world.bin",
            "viewcellRadiusM": "viewcell_radius_m.bin",
        }
        for name in files.values():
            (paths["dataset"] / name).write_bytes(b"\0")
        (paths["dataset"] / "dataset_meta.json").write_text(
            json.dumps(
                {
                    "schema": "pose-csr-explicit-four-way-split-v1",
                    "numInstances": 1,
                    "splitCounts": dict(scene.split_counts),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        paths["runtime_meta"].parent.mkdir(parents=True, exist_ok=True)
        paths["runtime_meta"].write_text(
            json.dumps(
                {
                    "instanceCount": 1,
                    "globalGlbCount": 1,
                    "componentRecords": [{"componentGlobalId": 0}],
                }
            ),
            encoding="utf-8",
        )
        paths["geometry_96d"].parent.mkdir(parents=True, exist_ok=True)
        paths["geometry_96d"].write_bytes(b"\0" * (runner.GEOMETRY_DIM * 2))

    for member in runner.registered_members(root):
        member.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        member.checkpoint.write_bytes(b"checkpoint")


class V4ExactRecalibrationRunnerTests(unittest.TestCase):
    def test_registration_contains_the_six_existing_full_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            members = runner.registered_members(Path(temporary))
        self.assertEqual(len(members), 6)
        self.assertEqual(
            [member.scene_key for member in members],
            ["hkust", "hkust", "hkust", "ifcbench", "ifcbench", "ifcbench"],
        )
        self.assertTrue(
            all(member.checkpoint.name == "best_safe.pt" for member in members)
        )
        self.assertTrue(
            all(member.geometry_96d.name == "instance_geo_features_fp16.bin" for member in members)
        )
        self.assertEqual(
            members[0].checkpoint.parent.name,
            "paper_full_seed20260801_e40",
        )
        self.assertEqual(
            members[3].checkpoint.parent.name,
            "full_seed20260801_e40",
        )

    def test_plan_has_four_test_free_steps_per_member_and_exact_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            plan = runner.build_plan(
                Path(temporary), Path(temporary) / "paper_results" / "exact_recalibration_v1"
            )
        self.assertEqual(plan["jobCount"], 24)
        self.assertEqual(
            set(plan["phases"]),
            {"score_calibration", "calibrate", "score_validation", "evaluate_validation"},
        )
        for jobs in plan["phases"].values():
            self.assertEqual(len(jobs), 6)
            for job in jobs:
                self.assertFalse(any(token.casefold() == "test" for token in job["command"]))
                self.assertFalse(job["testRead"])
        for job in plan["phases"]["calibrate"]:
            self.assertEqual(_argument(job["command"], "--bootstrap-replicates"), "10000")
            self.assertEqual(_argument(job["command"], "--bootstrap-seed"), "20260909")
        for job in plan["phases"]["evaluate_validation"]:
            self.assertEqual(_argument(job["command"], "--bootstrap-replicates"), "10000")
            self.assertEqual(_argument(job["command"], "--bootstrap-seed"), "20260910")

    def test_preflight_checks_exact_paths_without_opening_test_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_fixture(root)
            payload = runner.preflight(root, root / "paper_results" / "exact_recalibration_v1")
        self.assertEqual(payload["schema"], runner.PREFLIGHT_SCHEMA)
        self.assertEqual(len(payload["members"]), 6)
        self.assertEqual(
            [scene["splitCounts"]["validation"] for scene in payload["scenes"]],
            [730, 2712],
        )
        self.assertFalse(payload["testRead"])

    def test_scene_names_are_in_score_commands_and_test_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            members = runner.registered_members(Path(temporary))
            output = Path(temporary) / "output"
            hkust_command = runner.build_score_command(members[0], "calibration", output)
            ifcbench_command = runner.build_score_command(members[3], "validation", output)
        self.assertEqual(_argument(hkust_command, "--scene"), "HKUST")
        self.assertEqual(_argument(ifcbench_command, "--scene"), "IFCBench/Fantasy Metropolis")
        self.assertEqual(_argument(ifcbench_command, "--split"), "validation")
        with self.assertRaisesRegex(ValueError, "calibration and validation"):
            runner.build_score_command(members[0], "test", output)

    def test_output_cannot_be_nested_in_a_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            members = runner.registered_members(Path(temporary))
            with self.assertRaisesRegex(ValueError, "checkpoint directory"):
                runner.build_plan(Path(temporary), members[0].checkpoint.parent)

    def test_summary_contains_only_validation_safe_members_and_ranks_by_useful_cull(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "paper_results" / "exact_recalibration_v1"
            members = runner.registered_members(root)
            for index, member in enumerate(members):
                member_root = runner._member_root(output, member)
                calibration = {
                    "schema": runner.CALIBRATION_SCHEMA,
                    "scene": member.scene_name,
                    "split": "calibration",
                    "checkpoint": str(member.checkpoint.resolve()),
                    "status": "safe",
                    "testRead": False,
                }
                safe = index != 2
                metrics = {
                    "threshold": 0.5,
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.99 if not safe else 0.992,
                    "agg_useful_cull": 0.80 if index == 0 else 0.70,
                    "agg_balanced_accuracy": 0.50 if index == 0 else 0.99,
                    "candidate_normalized_occlusion_recall": 0.60,
                    "agg_specificity": 0.60,
                    "agg_precision": 0.50,
                    "avg_pred_count": 30.0,
                }
                validation = {
                    "schema": runner.VALIDATION_SCHEMA,
                    "scene": member.scene_name,
                    "split": "validation",
                    "checkpoint": str(member.checkpoint.resolve()),
                    "metrics": metrics,
                    "testRead": False,
                }
                runner._write_json(member_root / "exact_calibration.json", calibration)
                runner._write_json(member_root / "validation_frozen.json", validation)
            summary = runner.summarize(output, root)

        self.assertEqual(summary["registeredMemberCount"], 6)
        self.assertEqual(summary["validationSafeMemberCount"], 5)
        self.assertEqual(
            summary["sceneSelections"]["hkust"]["selected"]["member"],
            "hkust_paper_full_seed20260801_e40",
        )
        self.assertEqual(
            summary["sceneSelections"]["ifcbench"]["selected"]["scene"],
            "ifcbench",
        )
        self.assertNotIn("hkust_paper_full_seed20260803_e40", {
            row["member"] for row in summary["members"]
        })
        self.assertFalse(summary["testRead"])

    def test_repeated_gpu_ids_are_independent_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = runner.MemberSpec(
                key="fixture",
                scene_key="hkust",
                scene_name="HKUST",
                seed=1,
                checkpoint=root / "checkpoint.pt",
                dataset=root / "dataset",
                runtime_meta=root / "runtime.json",
                geometry_96d=root / "geometry.bin",
            )
            jobs = [
                runner.JobSpec(
                    member=member,
                    step=f"noop{index}",
                    command=(sys.executable, "-c", "pass"),
                    output=root / f"result{index}.json",
                    expected_artifact=root / f"result{index}.json",
                    bootstrap=None,
                )
                for index in range(2)
            ]
            with patch.object(runner, "subprocess") as subprocess_module:
                subprocess_module.run.return_value.returncode = 0
                results = runner.run_jobs(jobs, [2, 2])
                assigned = [
                    call.kwargs["env"]["CUDA_VISIBLE_DEVICES"]
                    for call in subprocess_module.run.call_args_list
                ]
        self.assertEqual(len(results), 2)
        self.assertEqual(assigned, ["2", "2"])


if __name__ == "__main__":
    unittest.main()
