from __future__ import annotations

import unittest

import torch

from neural_instance_culling.model.common.viewcell_exposure_supervision_loss import (
    viewcell_exposure_supervision_loss,
)


class ViewcellExposureSupervisionLossTests(unittest.TestCase):
    def test_matching_continuous_hit_rates_reduce_loss(self) -> None:
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        rates = torch.tensor([0.2, 0.0, 0.8, 0.0])
        offsets = torch.tensor([0, 2, 4])
        matched = torch.logit(rates.clamp(1e-4, 1.0 - 1e-4))
        wrong = -matched
        matching_loss, parts = viewcell_exposure_supervision_loss(
            matched, labels, rates, offsets
        )
        wrong_loss, _ = viewcell_exposure_supervision_loss(
            wrong, labels, rates, offsets
        )
        self.assertLess(float(matching_loss), float(wrong_loss))
        self.assertEqual(parts["viewcellExposureBoundaryCount"], 1.0)
        self.assertEqual(parts["viewcellExposureStableCount"], 1.0)

    def test_gradients_reach_positive_and_negative_rows(self) -> None:
        logits = torch.zeros(4, requires_grad=True)
        loss, _ = viewcell_exposure_supervision_loss(
            logits,
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([0.1, 0.0, 1.0, 0.0]),
            torch.tensor([0, 2, 4]),
        )
        loss.backward()
        self.assertTrue(bool((logits.grad != 0.0).all()))

    def test_semantic_mismatches_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "negative candidates"):
            viewcell_exposure_supervision_loss(
                torch.zeros(2),
                torch.tensor([1.0, 0.0]),
                torch.tensor([0.5, 0.1]),
                torch.tensor([0, 2]),
            )
        with self.assertRaisesRegex(ValueError, "visible union"):
            viewcell_exposure_supervision_loss(
                torch.zeros(2),
                torch.tensor([1.0, 0.0]),
                torch.zeros(2),
                torch.tensor([0, 2]),
            )


if __name__ == "__main__":
    unittest.main()
