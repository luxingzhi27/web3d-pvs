from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from neural_instance_culling.benchmark.run_standard_graphics_mainline import (
    SCENES,
    VIKING_FINETUNE_MEMBERS,
    selection_members,
    validation_key,
    validation_safe,
)


class StandardGraphicsMainlineTest(unittest.TestCase):
    def test_registered_scenes_have_four_way_splits(self) -> None:
        for name in ("sponza_128k", "viking_village_128k", "bigcity_128k"):
            with self.subTest(scene=name):
                self.assertEqual(
                    set(SCENES[name]["splits"]),
                    {"train", "validation", "calibration", "test", "guard"},
                )

    def test_validation_gate_requires_weighted_recall_and_lcb(self) -> None:
        self.assertTrue(
            validation_safe(
                {
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                }
            )
        )
        self.assertFalse(
            validation_safe(
                {
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.989,
                }
            )
        )

    def test_safe_member_ranking_prioritizes_useful_cull(self) -> None:
        def payload(useful: float, balanced: float) -> dict:
            return {
                "aggregateWeightedRecallLowerConfidenceBound": 0.995,
                "aggregate": {
                    "usefulCull": useful,
                    "balancedAccuracy": balanced,
                    "specificity": 0.8,
                    "precision": 0.7,
                    "avgPredCount": 10.0,
                },
            }

        self.assertGreater(validation_key(payload(0.8, 0.7), 2), validation_key(payload(0.7, 0.9), 1))

    def test_viking_finetune_family_requires_all_three_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            finetunes = root / "finetunes"
            rows = selection_members("viking_village_128k", root, finetunes)
            self.assertEqual(len(rows), 3)

            for name in VIKING_FINETUNE_MEMBERS.values():
                member = finetunes / name
                member.mkdir(parents=True)
                (member / "calibration_ready_summary.json").write_text("{}", encoding="utf-8")
            rows = selection_members("viking_village_128k", root, finetunes)
            self.assertEqual(len(rows), 6)


if __name__ == "__main__":
    unittest.main()
