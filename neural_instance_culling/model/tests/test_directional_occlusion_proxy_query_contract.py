from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from directional_occlusion_proxy_encoder_model import DirectionalOcclusionProxyEncoderPVSModel


class DirectionalOcclusionProxyQueryContractTest(unittest.TestCase):
    def test_common_evaluator_viewcell_arguments_are_accepted(self) -> None:
        model = DirectionalOcclusionProxyEncoderPVSModel(
            num_instances=2,
            num_glbs=1,
            geo_dim=8,
            context_dim=4,
            proxy_dim=2,
            direction_bins=2,
            depth_shells=1,
            source_k=1,
            point_hidden_dim=8,
            pointnetpp_centers=2,
            pointnetpp_neighbors=1,
            graph_hidden_dim=8,
            graph_message_dim=4,
            ray_fourier_bands=1,
            ray_scalar_fourier_bands=1,
            mlp_hidden=8,
            interaction_dim=4,
        )
        model.set_instance_world_aabbs(
            torch.tensor(
                [
                    [-1.0, -1.0, 4.0, 1.0, 1.0, 6.0],
                    [1.0, -1.0, 8.0, 3.0, 1.0, 10.0],
                ]
            )
        )
        instance_ids = torch.tensor([0, 1], dtype=torch.long)
        runtime_features = torch.zeros((2, model.runtime_feature_dim))

        logits = model.compute_visibility_logits(
            torch.zeros((2, 3)),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]]).repeat(2, 1),
            torch.zeros((2, 3)),
            instance_ids,
            runtime_features=runtime_features,
            query_center_world=torch.zeros((2, 3)),
            viewcell_radius_m=torch.full((2, 1), 2.0),
        )

        self.assertEqual(tuple(logits.shape), (2, 1))
        self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
