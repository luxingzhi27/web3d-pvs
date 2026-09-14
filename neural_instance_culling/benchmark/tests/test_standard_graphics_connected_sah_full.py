from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from neural_instance_culling.benchmark.run_standard_graphics_connected_sah_full import (
    CALIBRATION_SUMMARY_SCHEMA,
    DATASET_SCHEMA,
    EPOCHS,
    GEOMETRY_DIM,
    GUARD_MIDDLE_END_FRACTION,
    GUARD_MIDDLE_SCALE,
    GUARD_WEIGHT,
    HARD_POSE_FRACTION,
    HARD_POSE_QUANTILE,
    LEARNING_RATE,
    MODES,
    POSES_PER_BATCH,
    RELATION_K,
    RELATION_SCHEMA,
    SCENE_ORDER,
    SNAPSHOT_EVERY,
    STEPS_PER_EPOCH,
    TAIL_RAMP_END_FRACTION,
    TAIL_SEPARATION_WEIGHT,
    TAIL_ZERO_FRACTION,
    GUARD_ZERO_FRACTION,
    is_valid_completed_calibration_summary,
    member_dir,
    pending_members,
    plan,
    preflight,
    run_round,
    train_command,
)
from neural_instance_culling.benchmark.standard_graphics_connected_sah_config import (
    EXPERIMENT,
    SCENES,
    SEEDS,
)
from neural_instance_culling.model.train_pvs import parse_args as parse_train_args


def _value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


class ConnectedSahFullRunnerTests(unittest.TestCase):
    def test_mode_surface_and_full_training_hyperparameters(self) -> None:
        self.assertEqual(MODES, ("plan", "preflight", "smoke", "train"))
        command = train_command("sponza_128k", Path("/tmp/member"), SEEDS[0])
        self.assertEqual(_value(command, "--epochs"), str(EPOCHS))
        self.assertEqual(_value(command, "--steps-per-epoch"), str(STEPS_PER_EPOCH))
        self.assertEqual(_value(command, "--poses-per-batch"), str(POSES_PER_BATCH))
        self.assertEqual(_value(command, "--learning-rate"), str(LEARNING_RATE))
        self.assertEqual(_value(command, "--pose-sampling"), "ambiguity_balanced")
        self.assertEqual(_value(command, "--hard-pose-fraction"), str(HARD_POSE_FRACTION))
        self.assertEqual(_value(command, "--hard-pose-quantile"), str(HARD_POSE_QUANTILE))
        self.assertEqual(_value(command, "--survival-rank"), "4")
        self.assertEqual(_value(command, "--relation-source"), "bounded_hierarchical")
        self.assertEqual(_value(command, "--spectral-mode"), "moment_envelope")
        self.assertEqual(_value(command, "--instance-calibration-mode"), "residual")
        self.assertEqual(_value(command, "--loss-variant"), "pose_balanced_rvl_contrastive")
        self.assertNotIn("--init-checkpoint", command)
        self.assertNotIn("evaluate_pvs.py", " ".join(command))
        self.assertNotIn("export_pvs.py", " ".join(command))
        self.assertNotIn("finalize", " ".join(command))
        self.assertNotIn("--test", command)
        parsed = parse_train_args(command[3:])
        self.assertEqual(parsed.epochs, EPOCHS)
        self.assertEqual(parsed.poses_per_batch, POSES_PER_BATCH)

    def test_curriculum_and_tail_parameters_are_explicit(self) -> None:
        command = train_command("bigcity_128k", Path("/tmp/member"), SEEDS[0])
        expected = {
            "--integrated-rvl-recall-guard-weight": GUARD_WEIGHT,
            "--integrated-separation-weight": TAIL_SEPARATION_WEIGHT,
            "--integrated-rvl-recall-guard-zero-fraction": GUARD_ZERO_FRACTION,
            "--integrated-rvl-recall-guard-middle-end-fraction": GUARD_MIDDLE_END_FRACTION,
            "--integrated-rvl-recall-guard-middle-scale": GUARD_MIDDLE_SCALE,
            "--integrated-tail-zero-fraction": TAIL_ZERO_FRACTION,
            "--integrated-tail-ramp-fraction": TAIL_RAMP_END_FRACTION,
            "--frontier-positive-mass-fraction": 0.01,
            "--frontier-negative-fraction": 0.02,
            "--frontier-positive-importance-floor": 0.25,
            "--frontier-positive-importance-power": 0.5,
        }
        for flag, expected_value in expected.items():
            with self.subTest(flag=flag):
                self.assertAlmostEqual(float(_value(command, flag)), expected_value)
        self.assertEqual(GUARD_ZERO_FRACTION, 0.30)
        self.assertEqual(TAIL_ZERO_FRACTION, 0.10)
        self.assertEqual(TAIL_RAMP_END_FRACTION, 0.30)
        self.assertEqual(GUARD_MIDDLE_END_FRACTION, 0.50)
        self.assertEqual(GUARD_MIDDLE_SCALE, 0.25)

    def test_plan_is_scene_first_and_test_free(self) -> None:
        payload = plan(SCENE_ORDER, Path("/tmp/connected-sah-model"))
        self.assertTrue(payload["sceneFirst"])
        self.assertEqual(payload["sceneOrder"], list(SCENE_ORDER))
        self.assertEqual(payload["seeds"], list(SEEDS))
        self.assertEqual([row["seed"] for row in payload["rounds"]], list(SEEDS))
        for row in payload["rounds"]:
            self.assertEqual([job["scene"] for job in row["jobs"]], list(SCENE_ORDER))
            self.assertTrue(all("--test" not in job["command"] for job in row["jobs"]))
        self.assertFalse(payload["testRead"])

    def test_preflight_checks_exact_dataset_relation_and_geometry_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_root = root / "scene"
            assets = scene_root / "assets"
            dataset = scene_root / "pose_csr"
            relation = scene_root / "relation"
            assets.mkdir(parents=True)
            dataset.mkdir()
            relation.mkdir()
            runtime_meta = assets / "runtimeVisibilityMeta.json"
            runtime_meta.write_text(json.dumps({"instanceCount": 2}), encoding="utf-8")
            geometry = scene_root / "instance_geo_features_fp16.bin"
            geometry.write_bytes(b"\0" * (2 * GEOMETRY_DIM * 2))
            (scene_root / "instance_geo_features_fp16.json").write_text(
                json.dumps({"shape": [2, GEOMETRY_DIM], "dtype": "float16"}),
                encoding="utf-8",
            )
            files = {
                name: f"{name}.bin"
                for name in (
                    "poses",
                    "queryCenterWorld",
                    "candidateCameraWorld",
                    "viewcellRadiusM",
                    "candidateIds",
                    "visibleIds",
                    "visibleWeights",
                )
            }
            for name in files.values():
                (dataset / name).write_bytes(b"\0")
            (dataset / "dataset_meta.json").write_text(
                json.dumps({
                    "schema": DATASET_SCHEMA,
                    "numInstances": 2,
                    "splitCounts": {"train": 4, "calibration": 1, "validation": 1, "test": 1, "guard": 0},
                    "modelInputFovYDeg": 66.0,
                    "frontendRenderFovYDeg": 60.0,
                    "files": files,
                }),
                encoding="utf-8",
            )
            (relation / "relation_csr_meta.json").write_text(
                json.dumps({
                    "schema": RELATION_SCHEMA,
                    "trainOnly": True,
                    "splitNames": ["train"],
                    "sourceTopK": RELATION_K,
                    "evidenceTopK": {"k": RELATION_K},
                    "numInstances": 2,
                }),
                encoding="utf-8",
            )
            fixture = {
                "source": root / "source.glb",
                "assets": assets,
                "runtime_meta": runtime_meta,
                "glb_index": assets / "glbIndex.json",
                "glb_root": assets,
                "geometry": geometry,
                "geometry_meta": scene_root / "instance_geo_features_fp16.json",
                "dataset": dataset,
                "relation": relation,
                "split_counts": {"train": 4, "calibration": 1, "validation": 1, "test": 1, "guard": 0},
            }
            fixture["source"].write_bytes(b"source")
            fixture["glb_index"].write_text("{}", encoding="utf-8")
            with mock.patch.dict(SCENES, {"fixture": fixture}, clear=False):
                report = preflight("fixture")
            self.assertEqual(report["datasetSchema"], DATASET_SCHEMA)
            self.assertEqual(report["relationSchema"], RELATION_SCHEMA)
            self.assertEqual(report["relationK"], RELATION_K)
            self.assertEqual(report["geometry"]["shape"], [2, GEOMETRY_DIM])
            self.assertFalse(report["testRead"])

    def test_only_valid_completed_calibration_summaries_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = member_dir(root, "sponza_128k", SEEDS[0])
            member.mkdir(parents=True)
            self.assertFalse(is_valid_completed_calibration_summary(member))
            self.assertEqual(pending_members(["sponza_128k"], SEEDS[0], root), [("sponza_128k", member)])
            (member / "calibration_ready_summary.json").write_text(
                json.dumps({
                    "schema": CALIBRATION_SUMMARY_SCHEMA,
                    "status": "safe",
                    "calibration": {},
                    "testRead": False,
                }),
                encoding="utf-8",
            )
            self.assertTrue(is_valid_completed_calibration_summary(member))
            self.assertEqual(pending_members(["sponza_128k"], SEEDS[0], root), [])
            (member / "calibration_ready_summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(pending_members(["sponza_128k"], SEEDS[0], root), [("sponza_128k", member)])

    def test_repeated_gpu_slots_map_one_job_per_slot_and_write_logs(self) -> None:
        active: list[str] = []
        seen: list[str] = []
        lock = threading.Lock()

        def fake_run(*_args, **kwargs):
            gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
            with lock:
                active.append(gpu)
                seen.append(gpu)
            time.sleep(0.03)
            with lock:
                active.remove(gpu)
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            with mock.patch(
                "neural_instance_culling.benchmark.run_standard_graphics_connected_sah_full.subprocess.run",
                side_effect=fake_run,
            ):
                skipped = run_round(
                    SCENE_ORDER,
                    SEEDS[0],
                    [1, 1, 2],
                    root / "models",
                    logs,
                )
            self.assertEqual(skipped, [])
            self.assertEqual(sorted(seen), ["1", "1", "2"])
            self.assertEqual(len(list(logs.rglob("*.stdout.log"))), 3)
            command_files = list(logs.rglob("*.command.json"))
            self.assertEqual(len(command_files), 3)
            for path in command_files:
                record = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn(record["gpuId"], (1, 2))
                self.assertEqual(record["cudaVisibleDevices"], str(record["gpuId"]))
                self.assertFalse(record["testRead"])


if __name__ == "__main__":
    unittest.main()
