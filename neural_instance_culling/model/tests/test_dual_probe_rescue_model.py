from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pvs_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    DUAL_PROBE_RAW_QUERY_DIM,
    RUNTIME_FEATURE_DIM,
)


def _risk(risk: float) -> dict[str, object]:
    return {
        "sourceSplit": "train",
        "fitRowsOnly": True,
        "risk": risk,
        "observedRisk": 0.0,
        "registeredRiskUpperBound": risk,
        "riskWithinRegisteredBound": True,
    }


def _dual_probe_spec() -> dict[str, object]:
    zero = [0.0] * DUAL_PROBE_RAW_QUERY_DIM
    return {
        "enabled": True,
        "primary": {
            "mean": zero,
            "scale": [1.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "coefficients": [0.25] + [0.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "threshold": 0.25,
            "riskMetadata": _risk(0.01),
        },
        "coverage": {
            "mean": zero,
            "scale": [1.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "coefficients": [0.0] + [0.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "threshold": 0.05,
            "riskMetadata": _risk(0.05),
        },
        "alpha": 0.5,
        "temperature": 0.05,
        "coverageWeight": 2.0,
    }


class DualProbeRescueModelTests(unittest.TestCase):
    @staticmethod
    def _model(spec: dict[str, object] | None = None) -> BoundedRelationSurvivalMomentModel:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            spectral_mode="moment_envelope",
            depth_q01=0.0,
            depth_q99=4.0,
            dual_probe_rescue=_dual_probe_spec() if spec is None else spec,
        )
        model.set_instance_world_aabbs(
            torch.tensor(
                [
                    [-1.0, -1.0, 4.0, 1.0, 1.0, 6.0],
                    [2.0, -1.0, 7.0, 4.0, 1.0, 9.0],
                    [-4.0, -2.0, 9.0, -2.0, 2.0, 11.0],
                    [0.0, -1.0, 13.0, 2.0, 1.0, 15.0],
                ],
                dtype=torch.float32,
            )
        )
        return model

    @staticmethod
    def _inputs() -> dict[str, torch.Tensor]:
        torch.manual_seed(20260818)
        return {
            "camera": torch.zeros((4, 3), dtype=torch.float32),
            "camera_view": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 0.65]], dtype=torch.float32
            ).expand(4, -1),
            "candidate_camera": torch.tensor(
                [[0.0, 0.0, -3.0]], dtype=torch.float32
            ).expand(4, -1),
            "query_center": torch.zeros((4, 3), dtype=torch.float32),
            "radius": torch.full((4, 1), 2.0, dtype=torch.float32),
            "instance_ids": torch.arange(4, dtype=torch.long),
            "runtime": torch.randn((4, RUNTIME_FEATURE_DIM), dtype=torch.float32) * 0.05,
        }

    def test_raw_query_layout_and_extrema_follow_contract(self) -> None:
        center = torch.arange(18, dtype=torch.float32).reshape(2, 9)
        spectral = torch.zeros((2, 64), dtype=torch.float32)
        boundary = torch.zeros((2, 8), dtype=torch.float32)
        axes = torch.zeros((2, 9, 2), dtype=torch.float32)
        axes[0, :, 0] = 3.0
        axes[0, :, 1] = 4.0
        raw = self._model()._dual_probe_raw_query(center, spectral, boundary, axes)
        self.assertEqual(raw.shape, (2, 108))
        torch.testing.assert_close(raw[:, :9], center)
        torch.testing.assert_close(raw[:, 9:73], spectral)
        torch.testing.assert_close(raw[:, 73:81], boundary)
        extent = torch.linalg.vector_norm(axes, dim=-1)
        expected_extrema = torch.cat(
            [center - extent, center + extent, 2.0 * extent], dim=-1
        )
        torch.testing.assert_close(raw[:, 81:], expected_extrema)

    def test_initialization_reproduces_scanner_formula_and_is_non_negative(self) -> None:
        model = self._model()
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        primary_score = aux["dual_probe_primary_score"]
        coverage_score = aux["dual_probe_coverage_score"]
        primary_gate = torch.clamp(
            torch.tanh((0.25 - primary_score) / 0.05), min=0.0
        )
        coverage_gate = torch.clamp(
            torch.tanh((0.05 - coverage_score) / 0.05), min=0.0
        )
        fixed = 0.5 * torch.maximum(primary_gate, 2.0 * coverage_gate)
        torch.testing.assert_close(aux["dual_probe_fixed_residual"], fixed)
        torch.testing.assert_close(aux["dual_probe_residual"], fixed)
        torch.testing.assert_close(
            logits - aux["dual_probe_rescue_base_logit"],
            aux["dual_probe_residual"],
        )
        self.assertTrue(bool((aux["dual_probe_residual"] >= 0.0).all()))
        torch.testing.assert_close(
            aux["dual_probe_primary_attenuation"],
            torch.ones_like(aux["dual_probe_primary_attenuation"]),
        )
        torch.testing.assert_close(
            aux["dual_probe_coverage_attenuation"],
            torch.ones_like(aux["dual_probe_coverage_attenuation"]),
        )

    def test_attenuation_can_only_reduce_fixed_posterior_residual(self) -> None:
        model = self._model()
        assert model.dual_probe_rescue_attenuation_head is not None
        with torch.no_grad():
            model.dual_probe_rescue_attenuation_head[-1].bias.fill_(1.0)
        inputs = self._inputs()
        _logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertTrue(
            bool(
                (
                    aux["dual_probe_residual"]
                    <= aux["dual_probe_fixed_residual"] + 1e-6
                ).all()
            )
        )
        self.assertTrue(
            bool(
                (
                    aux["dual_probe_primary_attenuation"] < 1.0
                ).all()
            )
        )

    def test_config_and_state_dict_roundtrip_keep_frozen_probe_contract(self) -> None:
        model = self._model()
        config = model.config["dualProbeRescue"]
        self.assertEqual(config["rawQueryDim"], 108)
        self.assertEqual(config["primary"]["fitSplit"], "train")
        self.assertEqual(config["coverage"]["fitSplit"], "train")
        self.assertTrue(config["noPerInstanceAssets"])
        self.assertIn("max(primary_gate", config["fixedResidual"])
        self.assertIn("alpha * max", config["residual"])
        self.assertTrue(all(not value.requires_grad for value in model.buffers()))
        reconstructed = self._model(config)
        load_result = reconstructed.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(load_result.missing_keys)
        self.assertFalse(load_result.unexpected_keys)
        self.assertEqual(set(model.state_dict()), set(reconstructed.state_dict()))
        for name in (
            "dual_probe_primary_mean",
            "dual_probe_primary_scale",
            "dual_probe_primary_coefficients",
            "dual_probe_coverage_mean",
            "dual_probe_coverage_scale",
            "dual_probe_coverage_coefficients",
        ):
            torch.testing.assert_close(model.state_dict()[name], reconstructed.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
