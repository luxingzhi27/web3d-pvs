from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neural_instance_culling.benchmark.run_pvs_joint_108d_query_tail_separator_from_scratch_v1 import (
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    VARIANT_FAMILIES,
    _jobs,
    _member_complete,
    build_train_command,
)


class JointQueryTailSeparatorFromScratchRunnerTest(unittest.TestCase):
    def test_formal_matrix_has_four_variants_and_three_seeds(self) -> None:
        jobs = _jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({name for name, _, _ in jobs}), 12)
        self.assertEqual({seed for _, _, seed in jobs}, set(FORMAL_SEEDS))

    def test_every_command_is_from_scratch_full_model_training(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "out"
            for variant, family in VARIANT_FAMILIES:
                command = build_train_command(
                    root,
                    model_root,
                    variant,
                    FORMAL_SEEDS[0],
                )
                joined = " ".join(command)
                self.assertNotIn("--initial-checkpoint", command)
                self.assertIn("--refinement-scope all", joined)
                self.assertIn(f"--epochs {FORMAL_EPOCHS}", joined)
                self.assertIn(f"--query-tail-separator-family {family}", joined)
                expected_weight = "0.0" if family == "disabled" else "0.30"
                self.assertIn(
                    f"--query-tail-separation-loss-weight {expected_weight}",
                    joined,
                )
                self.assertNotIn("test", [piece.lower() for piece in command])

    def test_member_completion_requires_epoch_80(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            member = Path(temporary)
            for name in (
                "last.pt",
                "model_meta.json",
                "calibration_ready_summary.json",
            ):
                (member / name).write_text("{}", encoding="utf-8")
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": 79}]), encoding="utf-8"
            )
            self.assertFalse(_member_complete(member))
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": 80}]), encoding="utf-8"
            )
            self.assertTrue(_member_complete(member))


if __name__ == "__main__":
    unittest.main()
