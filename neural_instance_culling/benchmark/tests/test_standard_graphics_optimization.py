from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from neural_instance_culling.benchmark.run_standard_graphics_optimization import (
    optimized_train_command,
    relation_build_command,
    selection_key,
    train_only_relation_k,
    v2_relation_build_command,
    v2_train_command,
)
from neural_instance_culling.model.train_pvs import parse_args


class StandardGraphicsOptimizationTest(unittest.TestCase):
    def test_pilot_uses_train_only_ambiguity_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command = optimized_train_command(
                "sponza_128k",
                Path(temporary) / "pilot",
                20260801,
                16,
                pilot=True,
                init_checkpoint=Path(temporary) / "source.pt",
            )
            args = parse_args(command[2:])
        self.assertEqual(args.pose_sampling, "ambiguity_balanced")
        self.assertEqual(args.hard_pose_fraction, 0.5)
        self.assertEqual(args.hard_pose_quantile, 0.65)
        self.assertEqual(args.epochs, 8)
        self.assertNotIn("test", command)

    def test_relation_sweep_reuses_train_sparse_evidence(self) -> None:
        command = relation_build_command("bigcity_128k", 32)
        self.assertEqual(command[command.index("--splits") + 1], "train")
        self.assertEqual(command[command.index("--source-k") + 1], "32")
        self.assertIn("bigcity_triangle_depth_train/cache", " ".join(command))

    def test_pilot_selection_reads_flat_training_validation_metrics(self) -> None:
        def row(useful_cull: float) -> dict:
            return {
                "safe": True,
                "sourceTopK": 8,
                "validation": {
                    "agg_useful_cull": useful_cull,
                    "agg_balanced_accuracy": 0.7,
                    "agg_specificity": 0.6,
                    "agg_precision": 0.5,
                    "avg_pred_count": 10.0,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.995,
                },
            }

        self.assertGreater(selection_key(row(0.6)), selection_key(row(0.5)))

    def test_formal_k_uses_minimum_train_only_quality_member(self) -> None:
        payloads = [
            {"evidenceTopK": {"diagnostics": {"retainedQualityQuantiles": {"q01": 0.5}}}},
            {"evidenceTopK": {"diagnostics": {"retainedQualityQuantiles": {"q01": 0.81}}}},
            {"evidenceTopK": {"diagnostics": {"retainedQualityQuantiles": {"q01": 0.95}}}},
        ]
        with mock.patch(
            "neural_instance_culling.benchmark.run_standard_graphics_optimization.read_json",
            side_effect=payloads,
        ):
            selected, rows = train_only_relation_k("sponza_128k")
        self.assertEqual(selected, 16)
        self.assertEqual([row["sourceTopK"] for row in rows], [8, 16, 32])

    def test_v2_training_uses_new_dataset_and_keeps_test_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            command = v2_train_command(
                "viking_village_128k",
                Path(temporary) / "member",
                20260801,
                16,
            )
            args = parse_args(command[2:])
        self.assertIn("sampling_v2", str(args.dataset_dir))
        self.assertIn("sampling_v2_topk16", str(args.relation_dir))
        self.assertEqual(args.pose_sampling, "ambiguity_balanced")
        self.assertNotIn("test", command)

    def test_v2_relation_build_uses_new_train_sparse_cache(self) -> None:
        command = v2_relation_build_command("bigcity_128k", 32, Path("/results"))
        self.assertIn("pose_csr_bigcity_standard_graphics_128k_fov66_sampling_v2", " ".join(command))
        self.assertIn("/results/sampling_v2/bigcity_triangle_depth_train/cache", " ".join(command))
        self.assertEqual(command[command.index("--splits") + 1], "train")


if __name__ == "__main__":
    unittest.main()
