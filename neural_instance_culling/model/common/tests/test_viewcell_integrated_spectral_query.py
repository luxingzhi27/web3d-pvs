from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch

MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.viewcell_integrated_spectral_query import (  # noqa: E402
    FREQUENCY_COUNT,
    HIGH_FREQUENCY_COUNT,
    LOW_FREQUENCY_COUNT,
    MIDDLE_FREQUENCY_COUNT,
    SPECTRAL_FEATURE_DIM,
    VIEWCELL_INTEGRATED_SPECTRAL_SCHEMA,
    ViewCellIntegratedSpectralQuery,
    analytic_viewcell_diagonal_variance,
)
from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    HierarchicalRelationSurvivalIntegratedModel,
)


class ViewCellIntegratedSpectralQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.query = ViewCellIntegratedSpectralQuery()
        self.center = torch.tensor([
            [0.31, -0.22, 0.92, -0.45, 0.88, 0.12, -0.07, -0.82, -0.76],
            [-0.40, 0.10, 0.91, 0.10, 0.71, -0.20, 0.16, -0.64, -0.51],
        ], dtype=torch.float32)

    def test_frequency_dimensions_and_export_schema_are_fixed(self) -> None:
        schema = self.query.export_schema()
        self.assertEqual(schema["schema"], VIEWCELL_INTEGRATED_SPECTRAL_SCHEMA)
        self.assertEqual(schema["frequency"]["count"], FREQUENCY_COUNT)
        self.assertEqual(schema["frequency"]["jointInputDim"], 9)
        self.assertEqual(schema["gaussianIntegration"]["componentsPerFrequency"], 3)
        self.assertEqual(schema["gaussianIntegration"]["spectralFeatureDim"], SPECTRAL_FEATURE_DIM)
        self.assertEqual(SPECTRAL_FEATURE_DIM, 9 + 16 * 3)
        self.assertEqual(schema["frequency"]["groups"]["low"]["count"], LOW_FREQUENCY_COUNT)
        self.assertEqual(schema["frequency"]["groups"]["middle"]["count"], MIDDLE_FREQUENCY_COUNT)
        self.assertEqual(schema["frequency"]["groups"]["high"]["count"], HIGH_FREQUENCY_COUNT)
        self.assertFalse(schema["runtime"]["subposeExpansion"])
        self.assertEqual(schema["relationSurvivalField"]["semanticDim"], 8)
        self.assertEqual(tuple(self.query.frequency_vectors.shape), (16, 9))
        norms = self.query.frequency_norms().detach()
        self.assertGreater(float(norms[:8].max()), float(norms[:8].min()))
        self.assertGreater(float(norms[12:].mean()), float(norms[:8].mean()))

    def test_zero_uncertainty_is_the_exact_center_spectral_query(self) -> None:
        zero = torch.zeros_like(self.center)
        result = self.query.encode(self.center, zero)
        frequencies = self.query.bounded_frequency_vectors().detach()
        phase = 2.0 * torch.pi * (self.center @ frequencies.T)
        expected_per_frequency = torch.stack([
            torch.sin(phase),
            torch.cos(phase),
            torch.zeros_like(phase),
        ], dim=-1)
        expected = torch.cat([
            self.center,
            expected_per_frequency.reshape(self.center.shape[0], -1),
        ], dim=-1)
        self.assertTrue(torch.equal(result.attenuation, torch.ones_like(result.attenuation)))
        self.assertTrue(torch.equal(result.unresolved_energy, torch.zeros_like(result.unresolved_energy)))
        self.assertTrue(torch.allclose(result.spectral_features, expected, atol=1e-6, rtol=1e-6))
        center_result = self.query.center_query(self.center)
        self.assertTrue(torch.allclose(center_result.spectral_features, result.spectral_features))

    def test_region_integration_damps_amplitude_and_exposes_unresolved_energy(self) -> None:
        small = self.query.encode(self.center, torch.full_like(self.center, 0.01))
        large = self.query.encode(self.center, torch.full_like(self.center, 0.8))
        self.assertTrue(bool((large.attenuation <= small.attenuation + 1e-6).all()))
        self.assertTrue(bool((large.unresolved_energy >= small.unresolved_energy - 1e-6).all()))
        self.assertTrue(bool(torch.isfinite(large.spectral_features).all()))

    def test_nonzero_region_variance_changes_the_integrated_query_output(self) -> None:
        relation = torch.tensor([
            [0.8, 0.2, 0.7, 0.4, 0.1, 0.5, 0.5, 0.3],
            [0.5, 0.5, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2],
        ], dtype=torch.float32)
        coefficients = torch.tensor([
            [
                [0.20, -0.10, 0.05, 0.15, -0.07, 0.11, 0.03],
                [-0.05, 0.08, 0.12, -0.09, 0.04, 0.06, -0.02],
                [0.07, 0.03, -0.11, 0.05, 0.09, -0.04, 0.10],
                [0.04, -0.06, 0.02, 0.08, -0.03, 0.05, 0.01],
            ],
            [
                [-0.12, 0.09, 0.04, -0.06, 0.10, 0.02, 0.05],
                [0.06, -0.03, 0.08, 0.11, -0.05, 0.07, -0.04],
                [0.03, 0.05, -0.07, 0.02, 0.06, -0.08, 0.09],
                [0.10, 0.01, 0.06, -0.04, 0.03, 0.05, -0.02],
            ],
        ], dtype=torch.float32)
        rho = torch.tensor([0.35, 0.70], dtype=torch.float32)
        center = self.query(
            self.center,
            torch.zeros_like(self.center),
            relation_condition=relation,
            relation_survival_coefficients=coefficients,
            rho=rho,
        )
        regional = self.query(
            self.center,
            torch.full_like(self.center, 0.20),
            relation_condition=relation,
            relation_survival_coefficients=coefficients,
            rho=rho,
        )

        self.assertTrue(bool((regional.attenuation < center.attenuation).any()))
        self.assertTrue(bool((regional.unresolved_energy > center.unresolved_energy).any()))
        self.assertGreater(
            float((regional.spectral_features - center.spectral_features).abs().max()),
            1e-5,
        )
        self.assertGreater(
            float((regional.query_basis - center.query_basis).abs().max()),
            1e-6,
        )
        self.assertGreater(
            float((regional.relation_survival - center.relation_survival).abs().max()),
            1e-6,
        )

    def test_integrated_forward_shapes_and_all_outputs_are_finite(self) -> None:
        relation = torch.full((self.center.shape[0], 8), 0.25, dtype=torch.float32)
        coefficients = torch.full((self.center.shape[0], 4, 7), 0.05, dtype=torch.float32)
        result = self.query(
            self.center,
            torch.full_like(self.center, 0.15),
            relation_condition=relation,
            relation_survival_coefficients=coefficients,
            rho=torch.tensor([0.25, 0.75]),
        )

        expected_shapes = {
            "center_view": (2, 9),
            "diagonal_variance": (2, 9),
            "frequencies": (16, 9),
            "phase": (2, 16),
            "attenuation": (2, 16),
            "integrated_sin_cos": (2, 16, 2),
            "unresolved_energy": (2, 16),
            "per_frequency": (2, 16, 3),
            "low_frequency": (2, 24),
            "middle_frequency": (2, 12),
            "high_frequency": (2, 12),
            "high_frequency_energy": (2, 2),
            "high_frequency_gate": (2, 1),
            "spectral_features": (2, 57),
            "query_basis": (2, 4),
            "relation_condition": (2, 8),
            "relation_survival": (2, 8),
        }
        for name, expected_shape in expected_shapes.items():
            value = getattr(result, name)
            self.assertIsNotNone(value, name)
            self.assertEqual(tuple(value.shape), expected_shape, name)
            self.assertTrue(bool(torch.isfinite(value).all()), name)

    def test_fourier117_preserves_the_unified_runtime_query_contract(self) -> None:
        batch = 3
        model = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=batch,
            relation_hidden_dim=8,
            hidden_dim=8,
            query_basis_dim=4,
            spectral_mode="fourier117",
        ).eval()
        geometry = torch.randn(batch, 96)
        coefficients = torch.randn(batch, 4, 7) * 0.1
        instance_ids = torch.arange(batch, dtype=torch.long)
        center_view = torch.tensor([
            [0.10, -0.20, 0.97, -0.35, 0.80, 0.05, -0.04, -0.75, -0.70],
            [-0.15, 0.08, 0.98, -0.10, 0.75, -0.08, 0.06, -0.65, -0.60],
            [0.03, 0.12, 0.99, 0.20, 0.70, 0.02, 0.09, -0.55, -0.50],
        ], dtype=torch.float32)
        diagonal_variance = torch.full((batch, 9), 0.20, dtype=torch.float32)
        rho = torch.tensor([[0.20], [0.50], [0.80]], dtype=torch.float32)

        with torch.no_grad():
            logits, aux = model.forward_batch(
                instance_ids,
                center_view,
                diagonal_variance,
                rho,
                {"geometry": geometry, "survival_coefficients": coefficients},
            )

        self.assertEqual(model.config["spectralMode"], "fourier117")
        self.assertEqual(model.config["runtimeInputDim"], 120)
        self.assertEqual(model.config["runtimeFeatureDim"], 124)
        self.assertEqual(tuple(aux["spectral_features"].shape), (batch, 117))
        self.assertEqual(tuple(aux["integrated_query_basis"].shape), (batch, 4))
        self.assertEqual(tuple(aux["survival_semantic"].shape), (batch, 8))
        self.assertEqual(tuple(logits.shape), (batch, 1))
        for name in ("spectral_features", "integrated_query_basis", "survival_semantic"):
            self.assertTrue(bool(torch.isfinite(aux[name]).all()), name)
        self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_analytic_uncertainty_is_zero_for_zero_radius_and_finite_for_large_inputs(self) -> None:
        zero = analytic_viewcell_diagonal_variance(self.center, 20.0, viewcell_radius_m=0.0)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        large = analytic_viewcell_diagonal_variance(
            self.center * 1e4,
            torch.tensor([1e-8, 1e8]),
            viewcell_radius_m=2.0,
        )
        self.assertTrue(bool(torch.isfinite(large).all()))
        self.assertTrue(bool((large >= 0.0).all()))

    def test_relation_condition_and_survival_field_are_single_query_operations(self) -> None:
        relation = torch.tensor([
            [0.8, 0.2, 0.7, 0.4, 0.1, 0.5, 0.5, 0.3],
            [0.5, 0.5, 0.4, 0.6, 0.2, 0.4, 0.6, 0.2],
        ], dtype=torch.float32)
        coefficients = torch.randn(2, 4, 7) * 0.2
        result = self.query(
            self.center,
            torch.full_like(self.center, 0.1),
            relation_condition=relation,
            relation_survival_coefficients=coefficients,
            rho=torch.tensor([0.3, 0.7]),
        )
        self.assertEqual(tuple(result.spectral_features.shape), (2, SPECTRAL_FEATURE_DIM))
        self.assertEqual(tuple(result.query_basis.shape), (2, 4))
        self.assertEqual(tuple(result.relation_survival.shape), (2, 8))
        self.assertTrue(bool(torch.isfinite(result.query_basis).all()))
        self.assertTrue(bool(torch.isfinite(result.relation_survival).all()))
        self.assertTrue(bool((result.relation_survival[:, 0] >= 0.0).all()))
        self.assertTrue(bool((result.relation_survival[:, 0] <= 1.0).all()))

    def test_extreme_finite_inputs_and_backward_remain_finite(self) -> None:
        center = torch.full((3, 9), 1e4, dtype=torch.float32)
        variance = torch.full((3, 9), 1e4, dtype=torch.float32)
        relation = torch.full((3, 8), 1e4, dtype=torch.float32)
        result = self.query(center, variance, relation_condition=relation)
        loss = result.query_basis.square().mean() + result.spectral_features.square().mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(result.spectral_features).all()))
        self.assertTrue(bool(torch.isfinite(result.query_basis).all()))
        gradients = [parameter.grad for parameter in self.query.parameters() if parameter.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
