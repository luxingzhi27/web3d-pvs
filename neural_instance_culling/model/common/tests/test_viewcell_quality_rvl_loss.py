from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.viewcell_quality_rvl_loss import quality_tail_rvl_loss  # noqa: E402


class QualityTailRvlLossTest(unittest.TestCase):
    @staticmethod
    def _inputs() -> tuple[torch.Tensor, ...]:
        # The candidate IDs are global scene IDs.  In particular, IDs 8 and
        # 10 are larger than the four rows in this batch.
        logits = torch.tensor([[1.8], [-0.7], [0.35], [1.1]], dtype=torch.float32)
        target = torch.tensor([[1.0], [0.0], [0.0], [1.0]], dtype=torch.float32)
        pose_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
        visible_weights = torch.tensor([[4.0], [0.5], [0.2], [3.0]], dtype=torch.float32)
        visible_hit_rates = torch.tensor([[0.8], [0.9], [0.4], [0.7]], dtype=torch.float32)
        instance_ids = torch.tensor([8, 2, 10, 5], dtype=torch.long)
        instance_to_glb = torch.tensor([0, 0, 1, 2, 2, 3, 3, 4, 5, 6, 7], dtype=torch.long)
        glb_cost_norm = torch.tensor(
            [1.0, 1.2, 0.8, 1.5, 2.0, 0.7, 1.1, 1.8], dtype=torch.float32
        )
        return (
            logits,
            target,
            pose_offsets,
            visible_weights,
            visible_hit_rates,
            instance_ids,
            instance_to_glb,
            glb_cost_norm,
        )

    @staticmethod
    def _loss_kwargs() -> dict[str, float]:
        return {
            "quality_weight": 0.5,
            "resource_weight": 0.05,
            "quality_temperature": 0.1,
            "tail_fraction": 0.5,
            "rare_positive_weight": 1.0,
        }

    def test_non_contiguous_global_instance_ids_match_manual_gather(self) -> None:
        (
            logits,
            target,
            pose_offsets,
            visible_weights,
            visible_hit_rates,
            instance_ids,
            instance_to_glb,
            glb_cost_norm,
        ) = self._inputs()

        actual_loss, actual_parts = quality_tail_rvl_loss(
            logits,
            target,
            pose_offsets,
            visible_weights,
            visible_hit_rates,
            instance_ids,
            instance_to_glb,
            glb_cost_norm,
            **self._loss_kwargs(),
        )

        # This reference explicitly performs the intended global lookup first,
        # then presents the gathered values as a batch-local mapping.
        gathered_glb = instance_to_glb[instance_ids]
        local_ids = torch.arange(instance_ids.numel(), dtype=torch.long)
        expected_loss, expected_parts = quality_tail_rvl_loss(
            logits,
            target,
            pose_offsets,
            visible_weights,
            visible_hit_rates,
            local_ids,
            gathered_glb,
            glb_cost_norm,
            **self._loss_kwargs(),
        )

        self.assertTrue(bool(torch.isfinite(actual_loss)))
        self.assertTrue(torch.allclose(actual_loss, expected_loss, atol=1e-7, rtol=1e-7))
        self.assertEqual(actual_parts.keys(), expected_parts.keys())
        for name in actual_parts:
            actual = torch.as_tensor(actual_parts[name])
            expected = torch.as_tensor(expected_parts[name])
            self.assertTrue(
                torch.allclose(actual, expected, atol=1e-7, rtol=1e-7),
                msg=f"mismatch in {name}: {actual} != {expected}",
            )

        self.assertEqual(float(actual_parts["resourceUnnecessaryGlbCount"]), 2.0)
        self.assertTrue(bool(torch.isfinite(actual_parts["lossResourceRepulsion"])))
        self.assertTrue(bool(torch.isfinite(actual_parts["lossQualityTail"])))
        self.assertTrue(bool(torch.isfinite(actual_parts["qualityRecall"])))

    def test_out_of_range_global_instance_id_is_rejected(self) -> None:
        inputs = list(self._inputs())
        inputs[5] = torch.tensor([8, 2, 11, 5], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "outside instance_to_glb"):
            quality_tail_rvl_loss(*inputs, **self._loss_kwargs())


if __name__ == "__main__":
    unittest.main()
