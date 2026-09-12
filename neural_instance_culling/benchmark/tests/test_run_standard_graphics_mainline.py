from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from neural_instance_culling.benchmark.run_standard_graphics_mainline import (
    BIGCITY_RECOVERY_MEMBERS,
    SCENES,
    VIKING_FINETUNE_MEMBERS,
    bigcity_recovery_command,
    selection_members,
    validation_key,
    validation_safe,
    viking_finetune_command,
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

    def test_bigcity_recovery_family_requires_both_members(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recoveries = root / "recoveries"
            rows = selection_members("bigcity_128k", root, bigcity_recovery_root=recoveries)
            self.assertEqual(len(rows), 3)

            for name in BIGCITY_RECOVERY_MEMBERS.values():
                member = recoveries / name
                member.mkdir(parents=True)
                (member / "calibration_ready_summary.json").write_text("{}", encoding="utf-8")
            rows = selection_members("bigcity_128k", root, bigcity_recovery_root=recoveries)
            self.assertEqual(len(rows), 5)

    def test_registered_recovery_commands_keep_test_closed(self) -> None:
        bigcity = bigcity_recovery_command("balanced")
        self.assertIn("--reset-runtime-heads", bigcity)
        self.assertNotIn("test", bigcity)
        viking = viking_finetune_command(20260801)
        self.assertIn("--init-checkpoint", viking)
        self.assertNotIn("test", viking)


if __name__ == "__main__":
    unittest.main()
