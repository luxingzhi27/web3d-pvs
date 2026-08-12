from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.owrb_loss import OWRBConfig, OWRBState, owrb_visibility_loss  # noqa: E402
from common.safety_constraint_utility_loss import (  # noqa: E402
    SafetyDualState,
    _soft_group_union,
    neutral_rvl_control_evidence,
    safety_constraint_utility_loss,
)
from ray_context_survival_owrb_model import RayContextSurvivalOWRBModel  # noqa: E402


class RayContextSurvivalOWRBTest(unittest.TestCase):
    def test_soft_group_union_counts_a_shared_glb_once(self) -> None:
        probabilities = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
        inverse = torch.tensor([0, 0, 1], dtype=torch.long)
        result = _soft_group_union(probabilities, inverse, 2)
        self.assertTrue(torch.allclose(result, torch.tensor([0.44, 0.4]), atol=1e-6))

    def test_safety_byte_term_uses_unique_glb_cost(self) -> None:
        # Both negative instances share GLB 0.  With p=0.5 each, the soft
        # request probability is 1-(1-.5)^2=.75, not 1.0 from double billing.
        logits = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
        loss, parts = safety_constraint_utility_loss(
            logits,
            torch.zeros((2, 1)),
            torch.tensor([0, 2]),
            torch.zeros((2, 1)),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.ones(1),
            gamma=1.0,
            dual_state=SafetyDualState(),
        )
        self.assertAlmostEqual(float(parts["lossExcessGlbBytes"]), 0.75, places=5)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_rvl_control_evidence_is_neutral(self) -> None:
        logits = torch.tensor([[2.0], [-1.0]], dtype=torch.float32)
        evidence = neutral_rvl_control_evidence(logits)
        self.assertEqual(tuple(evidence.shape), tuple(logits.shape))
        self.assertTrue(torch.equal(evidence, torch.zeros_like(logits)))

    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = RayContextSurvivalOWRBModel(
            num_instances=6,
            num_glbs=2,
            point_hidden_dim=24,
            pointnetpp_centers=4,
            pointnetpp_neighbors=4,
            relation_hidden_dim=16,
        )
        self.model.set_scene_bounds(torch.zeros(3), torch.ones(3) * 10.0)
        self.model.set_instance_world_aabbs(torch.tensor([
            [0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            [2.0, 0.0, 1.0, 3.0, 1.0, 2.0],
            [4.0, 0.0, 1.0, 5.0, 1.0, 2.0],
            [6.0, 0.0, 1.0, 7.0, 1.0, 2.0],
            [8.0, 0.0, 1.0, 9.0, 1.0, 2.0],
            [0.0, 2.0, 1.0, 1.0, 3.0, 2.0],
        ]))
        self.model.set_instance_to_glb(torch.tensor([0, 1, 0, 1, 0, 1]))
        source_ids = torch.zeros(6, 12, 3, 8, dtype=torch.long)
        source_ids[:, 0, 0, 0] = 1
        relation_stats = torch.zeros(6, 12, 3, 8, 6)
        relation_stats[:, 0, 0, 0, 0] = 0.8
        relation_stats[:, 0, 0, 0, 1:6] = 0.4
        strength = torch.zeros(6, 12, 3)
        strength[:, 0, 0] = 0.8
        self.model.set_relation_evidence(source_ids, relation_stats, strength, torch.ones(6, 12, 3, dtype=torch.long))

    def test_runtime_schema_and_formal_query_dimensions(self) -> None:
        geometry = torch.randn(6, 96)
        coefficients = self.model.compute_offline_context_coefficients(geometry, mode="directional", batch_size=2)
        runtime = self.model.runtime_features_from_offline(geometry, coefficients)
        logits, aux = self.model.compute_logits_with_aux(
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.7]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([0]),
            runtime,
        )
        self.assertEqual(tuple(coefficients.shape), (6, 4, 8))
        self.assertEqual(tuple(runtime.shape), (6, 156))
        self.assertEqual(tuple(aux["base_input_features"].shape), (1, 177))
        self.assertEqual(tuple(aux["context_query"].shape), (1, 8))
        self.assertEqual(tuple(aux["survival_semantic"].shape), (1, 8))
        expected_survival_visibility = torch.stack([
            aux["survival_semantic"][:, 0],
            aux["survival_semantic"][:, 2],
            aux["survival_semantic"][:, 3],
            aux["survival_semantic"][:, 4],
        ], dim=-1)
        self.assertTrue(torch.allclose(
            aux["base_input_features"][:, 173:],
            expected_survival_visibility,
            atol=1e-7,
            rtol=1e-7,
        ))
        self.assertTrue(torch.allclose(
            aux["survival_visibility_features"],
            expected_survival_visibility,
            atol=1e-7,
            rtol=1e-7,
        ))
        self.assertEqual(tuple(logits.shape), (1, 1))

    def test_visibility_input_uses_semantics_not_survival_basis(self) -> None:
        geometry = torch.randn(6, 96)
        coefficients = self.model.compute_offline_context_coefficients(
            geometry, mode="directional", batch_size=2
        )
        runtime = self.model.runtime_features_from_offline(geometry, coefficients)
        _logits, aux = self.model.compute_logits_with_aux(
            torch.zeros(1, 3),
            torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.7]]),
            torch.tensor([[0.0, 0.0, 0.0]]),
            torch.tensor([0]),
            runtime,
        )
        self.assertFalse(torch.allclose(
            aux["base_input_features"][:, 173:],
            aux["survival_basis"],
            atol=1e-7,
            rtol=1e-7,
        ))

    def test_semantic_mode_does_not_append_raw_survival_coefficients(self) -> None:
        self.assertEqual(self.model.config["survivalInputMode"], "semantic")
        self.assertEqual(self.model.config["baseInputDim"], 177)
        self.assertEqual(self.model.config["baseTrunk"], [177, 64, 32])
        self.assertEqual(self.model.config["visibilityHead"], [177, 64, 32, 1])

    def test_disabled_survival_factor_is_zero_filled_and_does_not_change_logits(self) -> None:
        disabled = RayContextSurvivalOWRBModel(
            num_instances=6,
            num_glbs=2,
            point_hidden_dim=24,
            pointnetpp_centers=4,
            pointnetpp_neighbors=4,
            relation_hidden_dim=16,
            survival_enabled=False,
        )
        disabled.set_scene_bounds(torch.zeros(3), torch.ones(3) * 10.0)
        disabled.set_instance_world_aabbs(self.model.instance_world_aabbs)
        disabled.set_instance_to_glb(self.model.instance_to_glb)
        disabled.set_relation_evidence(
            self.model.relation_source_ids,
            self.model.relation_stats,
            self.model.relation_strength,
            self.model.relation_count,
        )
        geometry = torch.randn(6, 96)
        context = torch.zeros(6, 4, 8)
        runtime = disabled.runtime_features_from_offline(geometry, context)
        self.assertTrue(torch.equal(runtime[:, 128:], torch.zeros_like(runtime[:, 128:])))
        camera_view = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.7]])
        camera_world = torch.tensor([[0.0, 0.0, 0.0]])
        ids = torch.tensor([0])
        first = disabled.compute_visibility_logits(
            torch.zeros(1, 3), camera_view, camera_world, ids, runtime_features=runtime
        )
        changed = runtime.clone()
        changed[:, 128:] = torch.randn_like(changed[:, 128:])
        second = disabled.compute_visibility_logits(
            torch.zeros(1, 3), camera_view, camera_world, ids, runtime_features=changed
        )
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_survival_probability_is_monotone(self) -> None:
        direction = torch.tensor([[0.0, 0.0, 1.0]]).repeat(5, 1)
        rho = torch.tensor([[0.1], [0.3], [0.5], [0.7], [0.9]])
        coefficients = self.model.survival_coefficients[:1].repeat(5, 1, 1)
        semantic, _ = self.model.survival_query_from_direction(direction, rho, coefficients)
        self.assertTrue(bool(torch.all(semantic[:-1, 0] >= semantic[1:, 0] - 1e-6)))

    def test_survival_component_permutation_preserves_semantics(self) -> None:
        direction = torch.tensor([[0.31, -0.22, 0.92]], dtype=torch.float32)
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
        rho = torch.tensor([[0.63]], dtype=torch.float32)
        coefficients = torch.tensor([[
            [0.2, -0.4, 0.7, 1.1, -0.8, -2.0, 1.5],
            [0.1, 0.3, -0.2, 0.4, 0.6, 0.2, -0.1],
            [-0.2, 0.1, 0.4, -0.3, 0.2, 0.5, 0.4],
            [0.3, -0.1, 0.2, 0.5, -0.4, 0.1, -0.6],
        ]], dtype=torch.float32)
        swapped = coefficients.clone()
        for left, right in ((1, 2), (3, 4), (5, 6)):
            swapped[:, :, left] = coefficients[:, :, right]
            swapped[:, :, right] = coefficients[:, :, left]
        first, _ = self.model.survival_query_from_direction(direction, rho, coefficients)
        second, _ = self.model.survival_query_from_direction(direction, rho, swapped)
        self.assertTrue(torch.allclose(first, second, atol=1e-6, rtol=1e-6))

    def test_owrb_gradient_is_finite(self) -> None:
        logits = torch.tensor([[2.0], [-1.0], [0.3], [1.0], [-2.0], [0.1]], requires_grad=True)
        target = torch.tensor([[1.0], [0.0], [1.0], [1.0], [0.0], [0.0]])
        weights = torch.tensor([[5.0], [0.0], [1.0], [3.0], [0.0], [0.0]])
        state = OWRBState()
        loss, parts = owrb_visibility_loss(
            logits, target, torch.tensor([0, 3, 6]), weights,
            torch.tensor([[0.1], [0.2], [0.4], [0.1], [0.9], [0.8]]),
            OWRBConfig(), state,
        )
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("lossOWRBHardNegative", parts)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(state.initialized)

    def test_registered_supplement_variants_have_distinct_query_shapes(self) -> None:
        geometry = torch.randn(6, 96)
        camera_view = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.7]])
        camera_world = torch.tensor([[0.0, 0.0, 0.0]])
        ids = torch.tensor([0])
        for kwargs, expected_runtime, expected_query in (
            ({"context_query_dim": 16}, 188, 217),
            ({"ray_mode": "fourier117"}, 156, 285),
            ({"survival_parameterization": "unconstrained28"}, 156, 177),
        ):
            variant = RayContextSurvivalOWRBModel(
                num_instances=6,
                num_glbs=2,
                point_hidden_dim=24,
                pointnetpp_centers=4,
                pointnetpp_neighbors=4,
                relation_hidden_dim=16,
                **kwargs,
            )
            variant.set_scene_bounds(torch.zeros(3), torch.ones(3) * 10.0)
            variant.set_instance_world_aabbs(self.model.instance_world_aabbs)
            variant.set_instance_to_glb(self.model.instance_to_glb)
            variant.set_relation_evidence(
                self.model.relation_source_ids,
                self.model.relation_stats,
                self.model.relation_strength,
                self.model.relation_count,
            )
            coefficients = variant.compute_offline_context_coefficients(
                geometry, mode="directional", batch_size=2
            )
            runtime = variant.runtime_features_from_offline(geometry, coefficients)
            logits, aux = variant.compute_logits_with_aux(
                torch.zeros(1, 3), camera_view, camera_world, ids, runtime_features=runtime
            )
            self.assertEqual(tuple(coefficients.shape), (6, 4, variant.context_query_dim))
            self.assertEqual(tuple(runtime.shape), (6, expected_runtime))
            self.assertEqual(tuple(aux["base_input_features"].shape), (1, expected_query))
            self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
