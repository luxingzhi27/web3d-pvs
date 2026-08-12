from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.safety_gradient_projection import project_auxiliary_gradients  # noqa: E402


class SafetyGradientProjectionTest(unittest.TestCase):
    def test_conflicting_auxiliary_gradient_is_projected_and_capped(self) -> None:
        safety = [torch.tensor([2.0, 0.0])]
        auxiliary = [torch.tensor([-4.0, 1.0])]
        projected, parts = project_auxiliary_gradients(
            safety, auxiliary, gradient_cap=0.25
        )
        self.assertEqual(parts["gradientProjectionApplied"], 1.0)
        self.assertLessEqual(
            float(projected[0].norm()),
            0.25 * float(safety[0].norm()) + 1e-6,
        )
        self.assertGreaterEqual(float(torch.dot(safety[0], projected[0])), -1e-6)

    def test_non_conflicting_gradient_keeps_direction_before_cap(self) -> None:
        safety = [torch.tensor([2.0, 0.0])]
        auxiliary = [torch.tensor([1.0, 1.0])]
        projected, parts = project_auxiliary_gradients(
            safety, auxiliary, gradient_cap=10.0
        )
        self.assertEqual(parts["gradientProjectionApplied"], 0.0)
        self.assertTrue(torch.allclose(projected[0], auxiliary[0]))

    def test_none_entries_and_length_mismatch_are_handled(self) -> None:
        projected, _parts = project_auxiliary_gradients(
            [None, torch.tensor([1.0])],
            [None, torch.tensor([-1.0])],
        )
        self.assertIsNone(projected[0])
        self.assertIsNotNone(projected[1])
        with self.assertRaises(ValueError):
            project_auxiliary_gradients([None], [None, None])


if __name__ == "__main__":
    unittest.main()
