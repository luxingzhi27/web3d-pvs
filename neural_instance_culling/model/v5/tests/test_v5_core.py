from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


MODEL_DIR = Path(__file__).resolve().parents[2]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from v5.core import (  # noqa: E402
    ANCHOR_COUNT,
    FIELD_SHAPE,
    FULL_HEAD_INPUT_DIM,
    GEOMETRY_RELATION_SCHEMA,
    GCOFPVSV5,
    QUERY_GEOMETRY_DIM,
    SingleLayerRelationFieldCompiler,
    VARIANTS,
    build_query_geometry,
    fixed_direction_projection,
    icosahedron_anchors,
)
from v5.geometry_encoder import GEOMETRY_DIM, POINT_COUNT, LocalSurfaceGeometryEncoder  # noqa: E402
from v5.survival import (  # noqa: E402
    build_oriented_box_support_points,
    double_truncated_logistic_survival,
)


def geometry_relation(
    source_ids: torch.Tensor,
    edge_features: torch.Tensor,
    *,
    target_ids: torch.Tensor | None = None,
    anchor_ids: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": GEOMETRY_RELATION_SCHEMA,
        "metadata": {
            "schema": GEOMETRY_RELATION_SCHEMA,
            "source": "geometry_only",
            "usesVisibilityLabels": False,
        },
        "source_ids": source_ids,
        "edge_features": edge_features,
    }
    if target_ids is not None:
        result["target_ids"] = target_ids
    if anchor_ids is not None:
        result["anchor_ids"] = anchor_ids
    if valid_mask is not None:
        result["valid_mask"] = valid_mask
    return result


class GeometryEncoderTests(unittest.TestCase):
    def test_fixed_256_point_encoder_shape(self) -> None:
        torch.manual_seed(4)
        encoder = LocalSurfaceGeometryEncoder()
        points = torch.randn(3, POINT_COUNT, 6)
        ratios = torch.tensor([[1.0, 0.5, 0.25]]).expand(3, -1)
        result = encoder(points, ratios)
        self.assertEqual(tuple(result.shape), (3, GEOMETRY_DIM))
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_wrong_point_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "256"):
            LocalSurfaceGeometryEncoder()(torch.zeros(1, POINT_COUNT - 1, 6), torch.ones(1, 3))


class SurvivalAndRegionTests(unittest.TestCase):
    def test_survival_is_one_at_zero_and_monotone(self) -> None:
        torch.manual_seed(5)
        coefficients = torch.randn(1, 4, 7).expand(17, -1, -1).clone()
        directions = torch.tensor([[0.3, -0.4, 0.5]]).expand(17, -1)
        distances = torch.linspace(0.0, 30.0, 17)
        survival = double_truncated_logistic_survival(
            coefficients, directions, distances, torch.ones(17)
        ).squeeze(1)
        self.assertTrue(torch.equal(survival[:1], torch.ones(1)))
        self.assertTrue(bool((survival[1:] <= survival[:-1] + 1.0e-7).all()))
        self.assertTrue(bool(((survival > 0.0) & (survival <= 1.0)).all()))

    def test_oriented_box_has_eight_real_unique_corners(self) -> None:
        center = torch.tensor([[1.0, 2.0, 3.0]])
        half_axes = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]])
        points = build_oriented_box_support_points(center, half_axes)
        self.assertEqual(tuple(points.shape), (1, 9, 3))
        corners = points[:, 1:]
        self.assertEqual(torch.unique(corners[0], dim=0).shape[0], 8)
        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 4.0, 0.0], [0.0, 4.0, 6.0],
             [2.0, 0.0, 0.0], [2.0, 0.0, 6.0], [2.0, 4.0, 0.0], [2.0, 4.0, 6.0]]
        )
        self.assertTrue(torch.equal(torch.sort(corners[0], dim=0).values, torch.sort(expected, dim=0).values))


class QueryGeometryTests(unittest.TestCase):
    def test_query16_matches_v6_1_channel_order(self) -> None:
        query = build_query_geometry(
            target_to_region_world_direction=torch.tensor([[1.0, 0.0, 0.0]]),
            camera_right_up_forward=torch.eye(3).unsqueeze(0),
            distance=torch.tensor([2.0]),
            target_radius=torch.tensor([1.0]),
            region_half_axes=torch.tensor([[0.2, 0.3, 0.4]]),
            fov_tangent_xy=torch.tensor([[0.5, 0.75]]),
            region_type=torch.tensor([1.0]),
            near=torch.tensor([0.1]),
            far=torch.tensor([3.0]),
        )
        expected = torch.tensor(
            [[
                1.0, 0.0, 0.0,
                -1.0, 0.0, 0.0,
                torch.log(torch.tensor(3.0)), 1.0 / 3.0,
                0.2 / 3.4, 0.3 / 3.4, 0.4 / 3.4,
                0.5, 0.75, 1.0, 0.1 / 3.0, torch.log(torch.tensor(2.0)),
            ]
        ])
        self.assertEqual(tuple(query.shape), (1, QUERY_GEOMETRY_DIM))
        self.assertTrue(torch.allclose(query, expected, atol=1.0e-6, rtol=0.0))


class CompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(8)
        self.geometry = torch.randn(7, GEOMETRY_DIM)
        self.compiler = SingleLayerRelationFieldCompiler()
        source = torch.tensor([1, 2, 3, 4, 5, 6])
        target = torch.tensor([0, 0, 2, 4, 5, 6])
        anchor = torch.tensor([0, 0, 1, 2, 3, 4])
        edge = torch.randn(6, 8)
        self.sparse = geometry_relation(
            source, edge, target_ids=target, anchor_ids=anchor
        )

    def test_icosahedron_and_projection_are_fixed_buffers(self) -> None:
        anchors = icosahedron_anchors()
        projection = fixed_direction_projection()
        self.assertEqual(tuple(anchors.shape), (ANCHOR_COUNT, 3))
        self.assertTrue(torch.allclose(anchors.norm(dim=-1), torch.ones(ANCHOR_COUNT)))
        self.assertEqual(tuple(projection.shape), (4, ANCHOR_COUNT))
        self.assertIn("direction_projection", dict(self.compiler.named_buffers()))
        self.assertNotIn("direction_projection", dict(self.compiler.named_parameters()))
        self.assertNotIn("anchors", dict(self.compiler.named_parameters()))

    def test_selective_sparse_compile_matches_full_rows(self) -> None:
        full = self.compiler(self.geometry, self.sparse)
        selected_ids = torch.tensor([5, 0, 2])
        selected = self.compiler(self.geometry, self.sparse, target_ids=selected_ids)
        self.assertEqual(tuple(selected.shape), (3, *FIELD_SHAPE))
        self.assertTrue(torch.allclose(selected, full[selected_ids], atol=1.0e-6, rtol=1.0e-6))

    def test_selective_dense_slice_matches_full_rows(self) -> None:
        n, k = self.geometry.shape[0], 2
        source = torch.full((n, ANCHOR_COUNT, k), -1, dtype=torch.long)
        edge = torch.zeros(n, ANCHOR_COUNT, k, 8)
        valid = torch.zeros(n, ANCHOR_COUNT, k, dtype=torch.bool)
        source[0, 0, 0], edge[0, 0, 0], valid[0, 0, 0] = 1, torch.ones(8), True
        source[2, 1, 0], edge[2, 1, 0], valid[2, 1, 0] = 3, torch.ones(8), True
        source[5, 4, 0], edge[5, 4, 0], valid[5, 4, 0] = 6, torch.ones(8), True
        relation = geometry_relation(source, edge, valid_mask=valid)
        full = self.compiler(self.geometry, relation)
        selected_ids = torch.tensor([5, 0, 2])
        selected_relation = geometry_relation(
            source[selected_ids], edge[selected_ids], valid_mask=valid[selected_ids]
        )
        selected = self.compiler(self.geometry, selected_relation, target_ids=selected_ids)
        self.assertTrue(torch.allclose(selected, full[selected_ids], atol=1.0e-6, rtol=1.0e-6))


class VariantTests(unittest.TestCase):
    def test_head_shapes_and_capacity_match(self) -> None:
        models = {variant: GCOFPVSV5(variant) for variant in VARIANTS}
        self.assertEqual(models["FULL"].config["headShape"], [52, 32, 1])
        self.assertEqual(models["GEOMETRY_FIELD"].config["headShape"], [52, 32, 1])
        self.assertEqual(models["GENERIC_RELATION_28"].config["headShape"], [76, 22, 1])
        full_count = models["FULL"].parameter_count
        for variant, model in models.items():
            self.assertLess(abs(model.parameter_count - full_count) / full_count, 0.05, variant)
            self.assertFalse(any("embedding" in name.lower() for name, _ in model.named_parameters()))
            self.assertFalse(any("residual" in name.lower() for name, _ in model.named_parameters()))

    def test_variant_outputs_have_expected_shapes(self) -> None:
        torch.manual_seed(9)
        geometry = torch.randn(4, GEOMETRY_DIM)
        query = torch.randn(4, QUERY_GEOMETRY_DIM)
        stats = torch.rand(4, 4)
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = GCOFPVSV5(variant)
                if variant == "GENERIC_RELATION_28":
                    latent = model.relation_latent(geometry)
                    self.assertEqual(tuple(latent.shape), (4, 28))
                    logits = model(geometry, query, generic_latent=latent)
                elif variant == "GEOMETRY_FIELD":
                    field = model.compile_field(geometry)
                    self.assertEqual(tuple(field.shape), (4, 4, 7))
                    logits = model(geometry, query, field=field, field_stats=stats)
                else:
                    field = model.compile_field(geometry)
                    self.assertEqual(tuple(field.shape), (4, 4, 7))
                    logits = model(geometry, query, field=field, field_stats=stats)
                self.assertEqual(tuple(logits.shape), (4, 1))
                self.assertTrue(bool(torch.isfinite(logits).all()))


if __name__ == "__main__":
    unittest.main()
