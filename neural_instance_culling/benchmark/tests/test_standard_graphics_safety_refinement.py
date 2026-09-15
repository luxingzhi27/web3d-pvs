from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from neural_instance_culling.benchmark.run_standard_graphics_safety_refinement import (
    LOSS_CONFIGS,
    refinement_command,
    selection_key,
)
from neural_instance_culling.model.train_pvs import parse_args


class StandardGraphicsSafetyRefinementTest(unittest.TestCase):
    def test_pilot_uses_short_fresh_optimizer_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = refinement_command(
                "sponza_128k",
                LOSS_CONFIGS[1],
                Path(directory) / "output",
                Path(directory) / "source.pt",
                20260801,
                final=False,
            )
            args = parse_args(command[3:])
        self.assertEqual(args.epochs, 4)
        self.assertEqual(args.steps_per_epoch, 450)
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.integrated_separation_weight, 0.15)
        self.assertEqual(args.integrated_rvl_recall_guard_weight, 0.45)
        self.assertEqual(args.training_course_total_steps, 1800)
        self.assertEqual(args.init_checkpoint, Path(directory) / "source.pt")
        self.assertNotEqual(args.validation_split, "test")

    def test_diagnostic_selection_prioritizes_lcb(self) -> None:
        low = {"safe": False, "validation": {"aggregateWeightedRecallLowerConfidenceBound": 0.98, "agg_useful_cull": 0.8}}
        high = {"safe": False, "validation": {"aggregateWeightedRecallLowerConfidenceBound": 0.989, "agg_useful_cull": 0.1}}
        self.assertGreater(selection_key(high), selection_key(low))


if __name__ == "__main__":
    unittest.main()
