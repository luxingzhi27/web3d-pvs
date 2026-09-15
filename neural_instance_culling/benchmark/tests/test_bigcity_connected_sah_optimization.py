from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from neural_instance_culling.benchmark.run_bigcity_connected_sah_optimization import (
    PILOT_CONFIGS,
    relation_command,
    selection_key,
    train_command,
)
from neural_instance_culling.model.train_pvs import parse_args


class BigCityConnectedSahOptimizationTest(unittest.TestCase):
    def test_relation_build_is_train_only(self) -> None:
        command = relation_command(16)
        self.assertEqual(command[command.index("--splits") + 1], "train")
        self.assertEqual(command[command.index("--source-k") + 1], "16")

    def test_pilot_uses_registered_negative_only_strata(self) -> None:
        config = PILOT_CONFIGS[0]
        with tempfile.TemporaryDirectory() as directory:
            command = train_command(config, Path(directory), 20260801, full=False)
            args = parse_args(command[3:])
        self.assertEqual(args.epochs, 8)
        self.assertEqual(args.steps_per_epoch, 450)
        self.assertEqual(args.training_course_total_steps, 36000)
        self.assertEqual(args.negative_only_pose_fraction, 0.25)
        self.assertEqual(args.hard_pose_fraction, 0.375)
        self.assertEqual(args.integrated_rvl_recall_guard_zero_fraction, 0.10)
        self.assertNotEqual(args.validation_split, "test")

    def test_unsafe_selection_prioritizes_safety_lcb(self) -> None:
        low = {
            "safe": False,
            "validation": {
                "aggregateWeightedRecallLowerConfidenceBound": 0.98,
                "aggregateWeightedRecall": 0.99,
                "candidateNormalizedOcclusionRecall": 0.8,
            },
        }
        high = {
            "safe": False,
            "validation": {
                "aggregateWeightedRecallLowerConfidenceBound": 0.989,
                "aggregateWeightedRecall": 0.991,
                "candidateNormalizedOcclusionRecall": 0.2,
            },
        }
        self.assertGreater(selection_key(high), selection_key(low))


if __name__ == "__main__":
    unittest.main()
