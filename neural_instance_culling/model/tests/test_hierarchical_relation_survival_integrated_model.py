from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.train_observed_relation_csr import RELATION_FEATURE_DIM  # noqa: E402
from common.viewcell_integrated_spectral_query import build_viewcell_query  # noqa: E402
from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    HierarchicalRelationSurvivalIntegratedModel,
)


class HierarchicalRelationSurvivalIntegratedModelTest(unittest.TestCase):
    """Small contract tests for the offline-to-runtime unified model."""

    NUM_INSTANCES = 6
    GEO_DIM = 96
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.model = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=16,
            hidden_dim=16,
            query_basis_dim=4,
        )
        self.geo = torch.randn(self.NUM_INSTANCES, self.GEO_DIM)
        self.relation = {
            "source_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "target_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "direction_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            "depth_shell_ids": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "edge_features": torch.randn(4, RELATION_FEATURE_DIM) * 0.1,
        }
        self.local_group_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        self.structural_group_ids = torch.tensor([0, 0, 1], dtype=torch.long)
        self.center_view = torch.tensor([
            [0.00, 0.00, 1.00, -0.20, 0.90, -0.10, 0.05, -0.80, -0.75],
            [0.10, 0.00, 0.99, -0.15, 0.88, 0.05, 0.02, -0.78, -0.70],
            [-0.08, 0.04, 0.99, 0.10, 0.86, -0.04, 0.08, -0.72, -0.68],
            [0.05, -0.06, 0.99, 0.25, 0.84, 0.08, -0.06, -0.65, -0.60],
            [-0.12, -0.03, 0.99, 0.32, 0.82, -0.08, 0.03, -0.58, -0.55],
            [0.02, 0.09, 1.00, 0.40, 0.80, 0.01, 0.10, -0.50, -0.48],
        ], dtype=torch.float32)
        self.rho = torch.linspace(0.10, 0.90, self.NUM_INSTANCES).unsqueeze(-1)
        self.center_view, self.diagonal_variance = build_viewcell_query(
            self.center_view,
            torch.linspace(8.0, 13.0, self.NUM_INSTANCES),
        )

    def _offline_coefficients(self) -> torch.Tensor:
        return self.model.offline_encode_survival(
            self.geo,
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
        )

    def _forward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        coefficients = self._offline_coefficients()
        return self.model.forward_batch(
            torch.arange(self.NUM_INSTANCES),
            self.center_view,
            self.diagonal_variance,
            self.rho,
            {"geometry": self.geo, "survival_coefficients": coefficients},
        )

    @staticmethod
    def _field(result: object, *names: str) -> torch.Tensor:
        for name in names:
            if isinstance(result, dict) and name in result:
                value = result[name]
            elif hasattr(result, name):
                value = getattr(result, name)
            else:
                continue
            if not isinstance(value, torch.Tensor):
                raise AssertionError(f"model output {name!r} must be a tensor")
            return value
        raise AssertionError(f"model output is missing one of {names!r}")

    def test_forward_is_finite_and_has_unified_output_shapes(self) -> None:
        logits, result = self._forward()
        coefficients = result["survival_coefficients"]
        query_basis = result["integrated_query_basis"]
        survival = result["survival_semantic"]

        self.assertEqual(tuple(coefficients.shape), (self.NUM_INSTANCES, 4, 7))
        self.assertEqual(tuple(query_basis.shape), (self.NUM_INSTANCES, 4))
        self.assertEqual(tuple(survival.shape), (self.NUM_INSTANCES, 8))
        self.assertEqual(tuple(logits.shape), (self.NUM_INSTANCES, 1))
        self.assertEqual(tuple(result["utility_logits"].shape), (self.NUM_INSTANCES, 1))
        self.assertEqual(tuple(result["download_logits"].shape), (self.NUM_INSTANCES, 1))
        for value in (coefficients, query_basis, survival, logits):
            self.assertTrue(bool(torch.isfinite(value).all()))

    def test_backward_produces_finite_parameter_gradients(self) -> None:
        logits, result = self._forward()
        loss = (
            logits.square().mean()
            + result["survival_semantic"].square().mean()
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        gradients = [parameter.grad for parameter in self.model.parameters() if parameter.requires_grad]
        used_gradients = [gradient for gradient in gradients if gradient is not None]
        self.assertTrue(used_gradients)
        self.assertTrue(all(bool(torch.isfinite(gradient).all()) for gradient in used_gradients))
        self.assertTrue(any(name.startswith("offline_survival_encoder") for name, parameter in self.model.named_parameters() if parameter.grad is not None))
        self.assertTrue(any(name.startswith("integrated_query") for name, parameter in self.model.named_parameters() if parameter.grad is not None))
        self.assertTrue(any(name.startswith("visibility_head") for name, parameter in self.model.named_parameters() if parameter.grad is not None))

    def test_viewcell_is_evaluated_once_for_the_whole_candidate_batch(self) -> None:
        with mock.patch.object(
            self.model.integrated_query,
            "forward",
            wraps=self.model.integrated_query.forward,
        ) as query_forward:
            result = self._forward()

        self.assertEqual(query_forward.call_count, 1)
        self.assertEqual(
            tuple(result[1]["integrated_query_basis"].shape),
            (self.NUM_INSTANCES, 4),
        )

    def test_survival_probability_is_monotone_non_increasing_in_rho(self) -> None:
        _logits, result = self._forward()
        coefficients = result["survival_coefficients"]
        query_basis = result["integrated_query_basis"]
        query = self.model.integrated_query

        repeated_basis = query_basis[:1].expand(5, -1)
        repeated_coefficients = coefficients[:1].expand(5, -1, -1)
        rho = torch.tensor([[0.05], [0.25], [0.50], [0.75], [0.95]])
        survival = query.query_relation_survival(repeated_basis, rho, repeated_coefficients)
        self.assertTrue(bool(torch.all(survival[:-1, 0] >= survival[1:, 0] - 1e-6)))

    def test_runtime_schema_contains_no_online_graph_neighbor_or_subpose_contract(self) -> None:
        schema = self.model.export_schema()
        runtime = schema["runtime"]

        forbidden = ("graph", "neighbor", "subpose")

        def walk(value: object) -> list[str]:
            if isinstance(value, dict):
                values: list[str] = []
                for key, child in value.items():
                    values.append(str(key).lower())
                    values.extend(walk(child))
                return values
            if isinstance(value, (list, tuple)):
                values = []
                for child in value:
                    values.extend(walk(child))
                return values
            return [str(value).lower()] if isinstance(value, str) else []

        serialized_words = walk(runtime)
        self.assertFalse(
            any(forbidden_word in word for word in serialized_words for forbidden_word in forbidden),
            f"runtime schema exposes offline-only fields: {serialized_words}",
        )


if __name__ == "__main__":
    unittest.main()
