from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from neural_instance_culling.benchmark.run_pvs_train_owned_tail_nonlinear_posterior_longtrain import (
    FIXED_OPERATING_THRESHOLD_SEED,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    _member_complete,
    _selected_checkpoint,
    build_train_command,
)


class TrainOwnedNonlinearTailPosteriorLongtrainTest(unittest.TestCase):
    def test_registered_command_is_80_epoch_seed_owned_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = build_train_command(root, root / "model-out", FORMAL_SEEDS[1])
        joined = " ".join(command)
        self.assertIn(f"--epochs {FORMAL_EPOCHS}", joined)
        self.assertIn("--refinement-scope boundary_tail_residual", joined)
        self.assertIn("--boundary-tail-residual-centering pose_mean", joined)
        self.assertIn("--boundary-tail-residual-max-abs 0.5", joined)
        self.assertIn("--boundary-tail-residual-cross-pose-pair-weight 0.4", joined)
        self.assertIn("--boundary-tail-residual-global-tail-pair-weight 0.25", joined)
        self.assertIn("--boundary-tail-residual-positive-importance-power 0.5", joined)
        self.assertIn(f"--operating-threshold-seed {FIXED_OPERATING_THRESHOLD_SEED}", joined)
        self.assertIn(
            f"full_seed{FORMAL_SEEDS[1]}_e80/checkpoint_epoch_024.pt", joined
        )
        self.assertNotIn("test", [piece.lower() for piece in command])

    def test_member_complete_requires_epoch_80(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            member = Path(temporary)
            for name in ("last.pt", "model_meta.json", "calibration_ready_summary.json"):
                (member / name).write_text("{}", encoding="utf-8")
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": 79}]), encoding="utf-8"
            )
            self.assertFalse(_member_complete(member))
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": 80}]), encoding="utf-8"
            )
            self.assertTrue(_member_complete(member))

    def test_checkpoint_selection_prefers_declared_safe_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            member = Path(temporary)
            (member / "calibration_ready_summary.json").write_text(
                json.dumps({"status": "safe", "testRead": False}), encoding="utf-8"
            )
            (member / "best_safe.pt").write_bytes(b"safe")
            (member / "best_diagnostic.pt").write_bytes(b"diagnostic")
            checkpoint, safe = _selected_checkpoint(member)
            self.assertEqual(checkpoint.name, "best_safe.pt")
            self.assertTrue(safe)

    def test_checkpoint_selection_falls_back_without_claiming_safety(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            member = Path(temporary)
            (member / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {"status": "no_qualified_safety_workpoint", "testRead": False}
                ),
                encoding="utf-8",
            )
            (member / "best_diagnostic.pt").write_bytes(b"diagnostic")
            checkpoint, safe = _selected_checkpoint(member)
            self.assertEqual(checkpoint.name, "best_diagnostic.pt")
            self.assertFalse(safe)


if __name__ == "__main__":
    unittest.main()
