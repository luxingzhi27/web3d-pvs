from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

MODEL_DIR = Path(__file__).resolve().parents[2] / "model"
sys.path.insert(0, str(MODEL_DIR))

from directional_occlusion_proxy_encoder_model import (  # noqa: E402
    DirectionalOcclusionProxyEncoderPVSModel,
)


class RuntimeFeatureAblationTests(unittest.TestCase):
    def _model(self, ablation: str) -> DirectionalOcclusionProxyEncoderPVSModel:
        model = DirectionalOcclusionProxyEncoderPVSModel(
            num_instances=3,
            num_glbs=2,
            geo_dim=4,
            context_dim=3,
            proxy_dim=2,
            direction_bins=2,
            depth_shells=2,
            source_k=1,
            point_hidden_dim=8,
            pointnetpp_centers=2,
            pointnetpp_neighbors=2,
            graph_hidden_dim=8,
            graph_message_dim=4,
            ray_fourier_bands=2,
            ray_scalar_fourier_bands=1,
            mlp_hidden=8,
            interaction_dim=4,
            scene_size_m=[10.0, 10.0, 10.0],
            runtime_feature_ablation=ablation,
        )
        model.set_instance_world_aabbs(torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]] * 3))
        model.set_instance_to_glb(torch.tensor([0, 0, 1]))
        return model

    def test_registered_ablations_zero_only_the_declared_inputs(self) -> None:
        runtime = torch.arange(3 * 15, dtype=torch.float32).reshape(3, 15)
        camera_view = torch.tensor([[0.0, 0.0, 1.0, 1.0, 1.0]] * 3)
        camera_world = torch.tensor([[0.0, 0.0, -2.0]] * 3)
        ids = torch.tensor([0, 1, 2])
        expected = {
            "none": (False, False, False),
            "proxy_zero": (False, False, True),
            "context_proxy_zero": (False, True, True),
            "geo_context_proxy_zero": (True, True, True),
        }
        for ablation, (geo_zero, context_zero, proxy_zero) in expected.items():
            model = self._model(ablation)
            _logits, aux = model.compute_logits_with_aux(
                torch.zeros((3, 3)),
                camera_view,
                camera_world,
                ids,
                runtime_features=runtime,
            )
            query = aux["query_features_for_ids"]
            geo = query[:, : model.geo_dim]
            context = query[:, model.geo_dim : model.geo_dim + model.context_dim]
            proxy = query[:, model.geo_dim + model.context_dim :]
            self.assertEqual(bool(torch.count_nonzero(geo).item() == 0), geo_zero, ablation)
            self.assertEqual(bool(torch.count_nonzero(context).item() == 0), context_zero, ablation)
            self.assertEqual(bool(torch.count_nonzero(proxy).item() == 0), proxy_zero, ablation)
            self.assertEqual(model.config["runtimeFeatureAblation"], ablation)


if __name__ == "__main__":
    unittest.main()
