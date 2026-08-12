from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from export_ray_context_survival_owrb import (  # noqa: E402
    DENSE_WEIGHT_NAMES,
    _query_input_layout,
    _serialize_weights,
)
from current_pvs_utils import fourier_features  # noqa: E402
from ray_context_survival_owrb_model import RayContextSurvivalOWRBModel  # noqa: E402


class RayContextSurvivalOWRBExportTest(unittest.TestCase):
    def test_online_weight_schema_matches_177_dim_model(self) -> None:
        model = RayContextSurvivalOWRBModel(
            num_instances=2,
            num_glbs=1,
            point_hidden_dim=16,
            pointnetpp_centers=2,
            pointnetpp_neighbors=2,
            relation_hidden_dim=8,
        )
        serialized = _serialize_weights(model.state_dict())
        self.assertNotIn("context_residual.0.weight", serialized)
        self.assertNotIn("context_residual_scale", serialized)
        self.assertNotIn("relation_residual_head.0.weight", serialized)
        self.assertIn("base_trunk.0.weight", serialized)
        self.assertEqual(serialized["base_trunk.0.weight"]["shape"], [64, 177])
        self.assertEqual(set(serialized), set(DENSE_WEIGHT_NAMES))

    def test_fourier117_schema_matches_model_ray_layout(self) -> None:
        model = RayContextSurvivalOWRBModel(
            num_instances=2,
            num_glbs=1,
            point_hidden_dim=16,
            pointnetpp_centers=2,
            pointnetpp_neighbors=2,
            relation_hidden_dim=8,
            ray_mode="fourier117",
        )
        layout = _query_input_layout(model.config)
        ray = layout["rayFeatures"]
        self.assertEqual(ray["mode"], "fourier117")
        self.assertEqual(ray["dim"], model.config["rayDim"])
        self.assertEqual(
            [(segment["name"], segment["offset"], segment["dim"], segment["sourceDim"], segment["bands"])
             for segment in ray["segments"]],
            [
                ("rayDirectionFourier", 0, 63, 3, 10),
                ("rayScalarsFourier", 63, 54, 6, 4),
            ],
        )
        self.assertEqual(layout["baseInput"]["dim"], 285)
        self.assertEqual(model.config["baseInputDim"], 285)

    def test_fourier117_values_match_direction_then_scalar_encoding(self) -> None:
        common = {
            "num_instances": 2,
            "num_glbs": 1,
            "point_hidden_dim": 16,
            "pointnetpp_centers": 2,
            "pointnetpp_neighbors": 2,
            "relation_hidden_dim": 8,
        }
        direct = RayContextSurvivalOWRBModel(**common, ray_mode="direct9")
        fourier = RayContextSurvivalOWRBModel(**common, ray_mode="fourier117")
        aabbs = torch.tensor([
            [-1.0, -1.0, 4.0, 1.0, 1.0, 6.0],
            [2.0, -0.5, 3.0, 3.0, 0.5, 5.0],
        ])
        camera_view = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.7]])
        camera_world = torch.tensor([[0.0, 0.0, 0.0]])
        instance_ids = torch.tensor([0])
        direct.set_instance_world_aabbs(aabbs)
        fourier.set_instance_world_aabbs(aabbs)
        direct_direction, direct_ray = direct.ray_features(camera_view, instance_ids, camera_world)
        fourier_direction, fourier_ray = fourier.ray_features(camera_view, instance_ids, camera_world)
        expected = torch.cat([
            fourier_features(direct_direction, 10),
            fourier_features(direct_ray[:, 3:], 4),
        ], dim=-1)
        self.assertTrue(torch.allclose(direct_direction, fourier_direction, atol=1e-7, rtol=1e-7))
        self.assertTrue(torch.allclose(fourier_ray, expected, atol=1e-7, rtol=1e-7))
        self.assertEqual(tuple(fourier_ray.shape), (1, 117))


if __name__ == "__main__":
    unittest.main()
