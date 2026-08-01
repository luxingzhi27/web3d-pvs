from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BENCHMARK_DIR.parent / "model"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(MODEL_DIR))

import evaluate_frozen_test as frozen  # noqa: E402
from evaluate_viewcell_image_per import resolve_threshold  # noqa: E402
from model_runners import load_threshold  # noqa: E402


class _Split:
    pose_indices = np.asarray([7], dtype=np.int64)


class _Dataset:
    num_instances = 2
    split_ids = ("train", "validation", "calibration", "test")

    def __init__(self, _path: str, num_instances: int):
        if num_instances != self.num_instances:
            raise AssertionError("fixture instance count mismatch")

    def split(self, name: str):
        if name not in self.split_ids:
            raise AssertionError(name)
        return _Split()


class FrozenTestEntrypointTests(unittest.TestCase):
    def test_formal_summary_threshold_is_not_replaced_by_runner_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "eval_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "protocol": "frozen_calibration_one_shot_test",
                        "testEvaluationCount": 1,
                        "frozenThreshold": 0.64,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_threshold(summary, 0.5), 0.64)

            invalid = Path(temp_dir) / "invalid_summary.json"
            invalid.write_text(
                json.dumps(
                    {
                        "protocol": "frozen_calibration_one_shot_test",
                        "testEvaluationCount": 0,
                        "frozenThreshold": 0.64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                load_threshold(invalid, 0.5)

            pre_test = Path(temp_dir) / "calibration_ready_summary.json"
            pre_test.write_text(
                json.dumps(
                    {
                        "protocol": "calibration_ready_pre_test",
                        "testEvaluationCount": 0,
                        "frozenThreshold": 0.58,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_threshold(pre_test, 0.5), 0.58)

    def test_component_image_resolver_reads_formal_frozen_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "eval_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "protocol": "frozen_calibration_one_shot_test",
                        "testEvaluationCount": 1,
                        "frozenThreshold": 0.61,
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                threshold=None,
                threshold_policy="weighted_precision",
                target_weighted_recall=0.99,
            )
            threshold, source = resolve_threshold(
                args,
                {"eval_summary": str(summary)},
                SimpleNamespace(threshold=0.5),
            )
            self.assertEqual(threshold, 0.61)
            self.assertIn("no threshold scan", source["source"])

    def test_evaluate_uses_current_runner_contract_and_strict_single_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            runtime_meta = root / "runtime.json"
            glb_index = root / "glb.json"
            glb_root = root / "assets"
            glb_root.mkdir()
            runtime_meta.write_text("{}", encoding="utf-8")
            glb_index.write_text("{}", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": frozen.MANIFEST_SCHEMA,
                        "protocol": frozen.MANIFEST_PROTOCOL,
                        "datasetDir": str(dataset_dir.resolve()),
                        "testEvaluationCountBeforeThisRun": 0,
                        "targetWeightedRecall": 0.99,
                        "models": {
                            "fixture": {
                                "threshold": 0.5,
                                "preTestEvaluationCount": 0,
                                "protocolSplit": {"fixture": True},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            def fake_evaluate_runner(
                runner,
                split,
                thresholds,
                count_budgets,
                byte_budgets,
                time_costs,
                time_budgets_ms,
                glb_byte_cost,
                *,
                poses_per_batch,
                max_eval_poses,
                max_candidates_per_pose,
                seed,
                target_utility_recall,
                target_weighted_recall,
                score_modes,
                aggregations,
                aggregation_top_k,
                sample_with_replacement,
                progress,
            ):
                captured.update(
                    {
                        "thresholds": thresholds,
                        "count_budgets": count_budgets,
                        "byte_budgets": byte_budgets,
                        "time_costs": time_costs,
                        "time_budgets_ms": time_budgets_ms,
                        "glb_byte_cost": glb_byte_cost,
                        "poses_per_batch": poses_per_batch,
                        "max_eval_poses": max_eval_poses,
                        "max_candidates_per_pose": max_candidates_per_pose,
                        "target_utility_recall": target_utility_recall,
                        "target_weighted_recall": target_weighted_recall,
                        "score_modes": score_modes,
                        "aggregations": aggregations,
                        "aggregation_top_k": aggregation_top_k,
                        "sample_with_replacement": sample_with_replacement,
                        "progress": progress,
                    }
                )
                self.assertEqual(int(split.pose_indices.size), 1)
                return {
                    "thresholdRows": [
                        {
                            "threshold": float(thresholds[0]),
                            "eval_pose_count": 1,
                            "pose_precision": 1.0,
                            "pose_recall": 1.0,
                            "pose_weighted_recall": 1.0,
                            "pose_useful_cull_candidate_ratio": 0.0,
                            "pose_bad_cull_candidate_ratio": 0.0,
                            "avg_candidate_count": 2.0,
                            "avg_pred_count": 2.0,
                        }
                    ],
                    "workpoints": {},
                    "best": {},
                    "diagnosticBestSafetyAdjustedCull": {},
                }

            fake_runner = SimpleNamespace(
                world_aabbs=np.zeros((2, 6), dtype=np.float32),
                instance_to_glb=np.asarray([0, 1], dtype=np.int64),
            )

            args = frozen.build_parser().parse_args(
                [
                    "evaluate",
                    "--models",
                    "fixture",
                    "--manifest",
                    str(manifest),
                    "--dataset-dir",
                    str(dataset_dir),
                    "--runtime-meta",
                    str(runtime_meta),
                    "--glb-index",
                    str(glb_index),
                    "--glb-root",
                    str(glb_root),
                    "--output-dir",
                    str(root / "frozen-test"),
                    "--device",
                    "cpu",
                ]
            )

            with patch.object(
                frozen,
                "load_runtime_meta",
                lambda _path: (
                    np.zeros((2, 6), dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int64),
                    {},
                ),
            ), patch.object(frozen, "select_device", lambda _value: torch.device("cpu")), patch.object(
                frozen, "PoseCSRDataset", _Dataset
            ), patch.object(frozen, "validate_protocol_split", lambda _protocol, _dataset: {"frozenTestDigest": "fixture"}), patch.object(
                frozen, "preflight_candidates", lambda _dataset, _split: {"testPoseCount": 1}
            ), patch.object(frozen, "load_runner", lambda *args, **kwargs: fake_runner), patch.object(
                frozen,
                "load_glb_byte_costs",
                lambda *args, **kwargs: np.asarray([10.0, 20.0], dtype=np.float64),
            ), patch.object(frozen, "evaluate_runner", fake_evaluate_runner):
                frozen.evaluate_frozen(args)

            np.testing.assert_allclose(captured["thresholds"], [0.5])
            self.assertEqual(captured["byte_budgets"], ())
            self.assertIsNone(captured["time_costs"])
            self.assertEqual(captured["time_budgets_ms"], ())
            self.assertEqual(captured["score_modes"], ("visibility-only",))
            self.assertEqual(captured["aggregations"], ("max",))
            self.assertEqual(captured["aggregation_top_k"], 1)
            self.assertEqual(captured["max_eval_poses"], 0)
            self.assertEqual(captured["max_candidates_per_pose"], 0)
            self.assertFalse(captured["sample_with_replacement"])
            self.assertTrue(captured["progress"])
            claim = json.loads((root / "frozen-test" / "frozen_test_claim.json").read_text(encoding="utf-8"))
            self.assertEqual(claim["status"], "completed")
            self.assertEqual(claim["testEvaluationCount"], 1)


if __name__ == "__main__":
    unittest.main()
