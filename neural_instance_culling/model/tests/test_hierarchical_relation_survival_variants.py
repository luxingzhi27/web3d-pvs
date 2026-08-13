from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
import sys
import unittest

import torch


MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from hierarchical_relation_survival_integrated_model import (  # noqa: E402
    HierarchicalRelationSurvivalIntegratedModel,
    GEO_DIM,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
)
from train_hierarchical_relation_survival_integrated import (  # noqa: E402
    _relation_variant,
)


class _UnreadableRelationMapping(Mapping[str, torch.Tensor]):
    """A relation mapping that fails if the free control inspects its graph."""

    def __getitem__(self, key: str) -> torch.Tensor:
        raise AssertionError(f"free relation variant read relation key {key!r}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("free relation variant iterated relation tensors")

    def __len__(self) -> int:
        raise AssertionError("free relation variant queried relation tensor count")


def _relation_fixture() -> dict[str, torch.Tensor]:
    return {
        "source_ids": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        "target_ids": torch.tensor([5, 6, 7, 0, 1], dtype=torch.long),
        "direction_ids": torch.tensor([0, 3, 3, 11, 0], dtype=torch.long),
        "depth_shell_ids": torch.tensor([0, 1, 2, 0, 2], dtype=torch.long),
        "edge_features": torch.cat(
            [
                torch.tensor(
                    [
                        [1.0, 0.0, 0.1],
                        [0.5, 0.2, 0.3],
                        [0.0, 1.0, 0.4],
                        [0.2, 0.8, 0.5],
                        [0.7, 0.3, 0.6],
                    ],
                    dtype=torch.float32,
                ),
                torch.zeros((5, 10), dtype=torch.float32),
            ],
            dim=1,
        ),
    }


class HierarchicalRelationSurvivalVariantsTest(unittest.TestCase):
    NUM_INSTANCES = 8

    def setUp(self) -> None:
        self.relation = _relation_fixture()
        self.local_group_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long)
        self.structural_group_ids = torch.tensor([0, 1, 1, 2], dtype=torch.long)

    def test_free_variant_does_not_read_relation_graph(self) -> None:
        relation, local_ids, structural_ids, metadata = _relation_variant(
            _UnreadableRelationMapping(),
            self.local_group_ids,
            self.structural_group_ids,
            "free",
            self.NUM_INSTANCES,
            torch.device("cpu"),
        )

        self.assertEqual(metadata["source"], "free")
        self.assertEqual(metadata["hierarchy"], "none_free_per_instance")
        self.assertFalse(metadata["shuffled"])
        self.assertEqual(tuple(local_ids.tolist()), tuple(range(self.NUM_INSTANCES)))
        self.assertEqual(tuple(structural_ids.tolist()), tuple(range(self.NUM_INSTANCES)))
        self.assertEqual(tuple(relation["source_ids"].shape), (0,))
        self.assertEqual(tuple(relation["target_ids"].shape), (0,))
        self.assertEqual(tuple(relation["edge_features"].shape), (0, 13))

    def test_single_scale_uses_identity_hierarchy(self) -> None:
        relation, local_ids, structural_ids, metadata = _relation_variant(
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
            "single_scale",
            self.NUM_INSTANCES,
            torch.device("cpu"),
        )

        self.assertEqual(metadata["source"], "single_scale")
        self.assertEqual(metadata["hierarchy"], "identity")
        self.assertFalse(metadata["shuffled"])
        self.assertEqual(tuple(local_ids.tolist()), tuple(range(self.NUM_INSTANCES)))
        self.assertEqual(tuple(structural_ids.tolist()), tuple(range(self.NUM_INSTANCES)))
        for key, value in self.relation.items():
            torch.testing.assert_close(relation[key], value)

    def test_shuffled_preserves_edges_direction_buckets_and_edge_weights(self) -> None:
        relation, local_ids, structural_ids, metadata = _relation_variant(
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
            "shuffled",
            self.NUM_INSTANCES,
            torch.device("cpu"),
        )

        self.assertEqual(metadata["source"], "shuffled")
        self.assertEqual(metadata["hierarchy"], "three_scale_source_shuffled")
        self.assertTrue(metadata["shuffled"])
        self.assertEqual(relation["source_ids"].numel(), self.relation["source_ids"].numel())
        self.assertEqual(relation["target_ids"].numel(), self.relation["target_ids"].numel())
        self.assertEqual(relation["direction_ids"].numel(), self.relation["direction_ids"].numel())
        self.assertEqual(relation["edge_features"].shape, self.relation["edge_features"].shape)
        torch.testing.assert_close(relation["target_ids"], self.relation["target_ids"])
        torch.testing.assert_close(relation["direction_ids"], self.relation["direction_ids"])
        torch.testing.assert_close(relation["depth_shell_ids"], self.relation["depth_shell_ids"])
        torch.testing.assert_close(relation["edge_features"], self.relation["edge_features"])
        self.assertEqual(
            torch.bincount(relation["direction_ids"], minlength=12).tolist(),
            torch.bincount(self.relation["direction_ids"], minlength=12).tolist(),
        )
        torch.testing.assert_close(local_ids, self.local_group_ids)
        torch.testing.assert_close(structural_ids, self.structural_group_ids)

    def test_integrated_and_fourier117_share_runtime_input_dimensions(self) -> None:
        integrated = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            query_basis_dim=4,
            spectral_mode="integrated",
        )
        fourier = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            query_basis_dim=4,
            spectral_mode="fourier117",
        )

        self.assertEqual(integrated.config["runtimeInputDim"], fourier.config["runtimeInputDim"])
        self.assertEqual(integrated.config["runtimeFeatureDim"], fourier.config["runtimeFeatureDim"])
        self.assertEqual(integrated.shared_trunk[0].in_features, fourier.shared_trunk[0].in_features)
        self.assertEqual(fourier.fourier_query[0].in_features, 117 + 8)
        self.assertEqual(integrated.integrated_query.spectral_feature_dim, 57)

        geometry = torch.randn(self.NUM_INSTANCES, 96)
        coefficients = torch.randn(self.NUM_INSTANCES, 4, 7)
        instance_ids = torch.arange(self.NUM_INSTANCES)
        center = torch.randn(self.NUM_INSTANCES, 9)
        variance = torch.rand(self.NUM_INSTANCES, 9)
        rho = torch.rand(self.NUM_INSTANCES, 1)
        tables = {"geometry": geometry, "survival_coefficients": coefficients}
        integrated_logits, integrated_aux = integrated.forward_batch(
            instance_ids, center, variance, rho, tables
        )
        fourier_logits, fourier_aux = fourier.forward_batch(
            instance_ids, center, variance, rho, tables
        )
        self.assertEqual(tuple(integrated_logits.shape), (self.NUM_INSTANCES, 1))
        self.assertEqual(tuple(fourier_logits.shape), (self.NUM_INSTANCES, 1))
        self.assertEqual(tuple(integrated_aux["integrated_query_basis"].shape), (self.NUM_INSTANCES, 4))
        self.assertEqual(tuple(fourier_aux["integrated_query_basis"].shape), (self.NUM_INSTANCES, 4))
        self.assertTrue(bool(torch.isfinite(integrated_logits).all()))
        self.assertTrue(bool(torch.isfinite(fourier_logits).all()))

    def test_geometry_only_matches_relation_encoder_capacity(self) -> None:
        """The geometry control must be a true, not padded, capacity match."""
        full = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            relation_source="hierarchical",
        )
        geometry_only = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            relation_source="geometry_only",
        )
        full_count = sum(
            parameter.numel()
            for parameter in full.offline_survival_encoder.parameters()
            if parameter.requires_grad
        )
        geometry_count = sum(
            parameter.numel()
            for parameter in geometry_only.geometry_only_survival_encoder.parameters()
            if parameter.requires_grad
        )
        self.assertLessEqual(abs(geometry_count - full_count) / full_count, 0.01)
        self.assertEqual(geometry_only.config["relationEncoderParameters"], None)
        self.assertEqual(geometry_only.config["geometryOnlyTargetParameters"], full_count)
        self.assertEqual(geometry_only.config["geometryOnlyParameters"], geometry_count)
        self.assertEqual(
            geometry_only.config["geometryOnlyWidths"],
            list(geometry_only.geometry_only_widths),
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in geometry_only.offline_survival_encoder.parameters()),
            0,
        )
        self.assertFalse(
            any(name.startswith("offline_survival_encoder.") for name, _ in geometry_only.named_parameters())
        )

    def test_geometry_only_output_schema_matches_full_encoder(self) -> None:
        geometry = torch.randn(self.NUM_INSTANCES, GEO_DIM)
        full = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            relation_source="hierarchical",
        )
        geometry_only = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            relation_source="geometry_only",
        )
        full_coefficients = full.offline_encode_survival(
            geometry,
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
        )
        geometry_coefficients = geometry_only.offline_encode_survival(
            geometry,
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
        )
        expected_shape = (self.NUM_INSTANCES, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
        self.assertEqual(tuple(full_coefficients.shape), expected_shape)
        self.assertEqual(tuple(geometry_coefficients.shape), expected_shape)
        self.assertTrue(bool(torch.isfinite(geometry_coefficients).all()))
        self.assertEqual(
            geometry_only.export_schema()["offline"]["output"]["survivalCoefficients"]["shape"],
            ["N", SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
        )

    def test_learned_point_query_removes_region_variance(self) -> None:
        model = HierarchicalRelationSurvivalIntegratedModel(
            num_instances=self.NUM_INSTANCES,
            relation_hidden_dim=8,
            hidden_dim=8,
            query_basis_dim=4,
            spectral_mode="learned_point",
        )
        geometry = torch.randn(self.NUM_INSTANCES, 96)
        coefficients = torch.randn(self.NUM_INSTANCES, 4, 7)
        center = torch.randn(self.NUM_INSTANCES, 9)
        variance = torch.rand(self.NUM_INSTANCES, 9) + 0.25
        rho = torch.rand(self.NUM_INSTANCES, 1)
        _logits, aux = model.forward_batch(
            torch.arange(self.NUM_INSTANCES),
            center,
            variance,
            rho,
            {"geometry": geometry, "survival_coefficients": coefficients},
        )
        torch.testing.assert_close(aux["variance_summary"], torch.zeros((self.NUM_INSTANCES, 2)))
        self.assertEqual(model.config["spectralMode"], "learned_point")


if __name__ == "__main__":
    unittest.main()
