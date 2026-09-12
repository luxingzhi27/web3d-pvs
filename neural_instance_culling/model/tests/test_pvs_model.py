from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from pvs_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    GEO_DIM,
    MODEL_SCHEMA,
    SUPPORTED_SURVIVAL_RANKS,
)


def runtime_batch(model: BoundedRelationSurvivalMomentModel, count: int = 3):
    ids = torch.arange(count, dtype=torch.long)
    center = torch.zeros(count, 9)
    center[:, 2] = 1.0
    axes = torch.zeros(count, 9, 2)
    axes[:, 0, 0] = 0.1
    axes[:, 1, 1] = 0.1
    depth = torch.linspace(0.2, 0.8, count).reshape(-1, 1)
    runtime = torch.randn(model.num_instances, model.runtime_feature_dim)
    return ids, center, axes, depth, {"runtime_features": runtime}


class PvsModelTests(unittest.TestCase):
    def test_reset_runtime_heads_preserves_offline_encoder(self) -> None:
        torch.manual_seed(7)
        model = BoundedRelationSurvivalMomentModel(5, 2)
        offline_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith("offline_survival_encoder.")
        }
        runtime_before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith(("shared_trunk.", "visibility_head."))
        }

        model.reset_runtime_heads()

        for name, value in offline_before.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        self.assertTrue(
            any(
                not torch.equal(value, model.state_dict()[name])
                for name, value in runtime_before.items()
            )
        )

    def test_default_runtime_contract(self) -> None:
        model = BoundedRelationSurvivalMomentModel(5, 2)
        self.assertEqual(model.config["runtimeSchema"], MODEL_SCHEMA)
        self.assertEqual(model.runtime_feature_dim, 124)
        self.assertEqual(model.runtime_head_input_dim, 130)
        self.assertEqual(model.config["survivalCoefficientShape"], [4, 7])

    def test_registered_survival_ranks(self) -> None:
        for rank in SUPPORTED_SURVIVAL_RANKS:
            with self.subTest(rank=rank):
                model = BoundedRelationSurvivalMomentModel(
                    5, 2, survival_rank=rank, relation_source="geometry_only"
                )
                self.assertEqual(model.runtime_feature_dim, GEO_DIM + rank * 7)
                self.assertEqual(model.runtime_head_input_dim, 126 + rank)
                logits, _ = model.forward_batch(*runtime_batch(model))
                self.assertEqual(tuple(logits.shape), (3, 1))

    def test_registered_occlusion_controls(self) -> None:
        generic = BoundedRelationSurvivalMomentModel(
            5, 2, relation_source="none", occlusion_representation="generic28",
            instance_calibration_mode="disabled",
        )
        none = BoundedRelationSurvivalMomentModel(
            5, 2, relation_source="none", occlusion_representation="none",
            instance_calibration_mode="disabled",
        )
        self.assertEqual(generic.runtime_feature_dim, 124)
        self.assertEqual(none.runtime_feature_dim, 96)
        self.assertEqual(none.runtime_head_input_dim, 118)
        for model in (generic, none):
            logits, _ = model.forward_batch(*runtime_batch(model))
            self.assertTrue(bool(torch.isfinite(logits).all()))

    def test_modes_strictly_reload_their_state(self) -> None:
        configurations = (
            {},
            {"relation_source": "geometry_only"},
            {"relation_source": "none", "occlusion_representation": "generic28",
             "instance_calibration_mode": "disabled"},
            {"relation_source": "none", "occlusion_representation": "none",
             "instance_calibration_mode": "disabled"},
        )
        for values in configurations:
            source = BoundedRelationSurvivalMomentModel(5, 2, **values)
            target = BoundedRelationSurvivalMomentModel(5, 2, **values)
            target.load_state_dict(source.state_dict(), strict=True)

    def test_instance_calibration_changes_only_selected_row(self) -> None:
        model = BoundedRelationSurvivalMomentModel(5, 2, relation_source="geometry_only")
        model.set_instance_calibration_reliability(torch.ones(5))
        model.set_instance_calibration_blend(1.0)
        geometry = torch.randn(5, GEO_DIM)
        before = model.offline_encode_survival(geometry, None, None, None).detach()
        assert model.instance_calibration_residual_raw is not None
        with torch.no_grad():
            model.instance_calibration_residual_raw[2].fill_(0.5)
        after = model.offline_encode_survival(geometry, None, None, None).detach()
        self.assertTrue(torch.equal(before[:2], after[:2]))
        self.assertFalse(torch.equal(before[2], after[2]))
        self.assertTrue(torch.equal(before[3:], after[3:]))

    def test_survival_is_monotone_in_depth(self) -> None:
        model = BoundedRelationSurvivalMomentModel(2, 1)
        direction = torch.tensor([[1.0, 0.0, 0.0]]).expand(21, -1)
        coefficients = torch.randn(1, 4, 7).expand(21, -1, -1).clone()
        semantic, _ = model.query_survival_from_direction(
            direction, torch.linspace(0, 1, 21).reshape(-1, 1), coefficients
        )
        self.assertTrue(bool((semantic[1:, 0] <= semantic[:-1, 0] + 1e-7).all()))

    def test_point_control_removes_viewcell_extent(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            5, 2, relation_source="geometry_only", spectral_mode="point"
        )
        _, auxiliary = model.forward_batch(*runtime_batch(model))
        self.assertEqual(int(torch.count_nonzero(auxiliary["effective_disk_axes"])), 0)

    def test_runtime_rejects_offline_graph_tables(self) -> None:
        model = BoundedRelationSurvivalMomentModel(5, 2)
        values = runtime_batch(model)
        values[-1]["relation_csr"] = torch.zeros(1)
        with self.assertRaisesRegex(ValueError, "offline graph"):
            model.forward_batch(*values)

    def test_all_heads_have_finite_gradients(self) -> None:
        model = BoundedRelationSurvivalMomentModel(5, 2, relation_source="geometry_only")
        logits, auxiliary = model.forward_batch(*runtime_batch(model))
        (logits.mean() + auxiliary["utility_logits"].mean()
         + auxiliary["download_logits"].mean()).backward()
        for name in ("visibility_head", "utility_head", "download_head"):
            gradients = [value.grad for value in getattr(model, name).parameters()]
            self.assertTrue(all(value is not None for value in gradients), name)
            self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))

    def test_compute_logits_requires_viewcell_inputs(self) -> None:
        model = BoundedRelationSurvivalMomentModel(5, 2, relation_source="geometry_only")
        with self.assertRaisesRegex(ValueError, "query_center_world"):
            model.compute_logits_with_aux(
                torch.zeros(3), torch.zeros(5), torch.zeros(3), torch.tensor([0]),
                torch.zeros(5, model.runtime_feature_dim),
            )


if __name__ == "__main__":
    unittest.main()
