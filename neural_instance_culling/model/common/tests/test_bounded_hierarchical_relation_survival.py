from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODEL = Path(__file__).resolve().parents[2]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.bounded_hierarchical_relation_survival import (  # noqa: E402
    BoundedHierarchicalRelationSurvivalEncoder,
    SCHEMA_V3,
    RELATION_FEATURE_DIM_V3,
    deterministic_negative_edges,
    segment_attention,
    segment_softmax,
)


class BoundedHierarchicalRelationSurvivalTest(unittest.TestCase):
    NUM_INSTANCES = 5
    GEO_DIM = 8
    DIRECTION_BINS = 2
    DEPTH_SHELLS = 2

    def setUp(self) -> None:
        torch.manual_seed(20260814)
        self.model = BoundedHierarchicalRelationSurvivalEncoder(
            geo_dim=self.GEO_DIM,
            hidden_dim=16,
            direction_bins=self.DIRECTION_BINS,
            depth_shells=self.DEPTH_SHELLS,
        )
        self.geo = torch.randn(self.NUM_INSTANCES, self.GEO_DIM)
        centers = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 1.0]]
        )
        half_extent = torch.full_like(centers, 0.25)
        self.aabbs = torch.cat([centers - half_extent, centers + half_extent], dim=-1)
        features = torch.zeros(4, RELATION_FEATURE_DIM_V3)
        features[:, 0] = 2.0
        features[:, 1] = 0.5
        features[:, 2] = 4.0
        features[:, 3] = 0.5
        features[:, 4] = 0.5
        features[:, 5] = 0.1
        features[:, 6] = 0.25
        features[:, 7] = 0.05
        features[:, 8:11] = torch.tensor([
            [0.1, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.3],
            [0.2, 0.1, 0.0],
        ])
        features[:, 12] = torch.tensor([0.9, 0.2, 0.8, 0.5])
        features[:, 13] = 0.9
        features[:, 14] = 0.5
        features[:, 15] = 1.0
        features[:, 16] = 0.0
        features[:, 17] = 0.4
        features[:, 18] = 0.05
        features[:, 19] = 1.0
        self.relation = {
            "schema": SCHEMA_V3,
            "metadata": {"schema": SCHEMA_V3, "trainOnly": True, "splitNames": ["train"]},
            "source_ids": torch.tensor([0, 1, 3, 0]),
            "target_ids": torch.tensor([2, 2, 4, 4]),
            "direction_ids": torch.tensor([0, 0, 1, 1]),
            "depth_shell_ids": torch.tensor([0, 0, 1, 0]),
            "edge_features": features,
        }
        self.local_group_ids = torch.tensor([0, 0, 1, 1, 2])
        self.structural_group_ids = torch.tensor([0, 0, 1])

    def test_segment_attention_normalizes_non_empty_segments_and_zeros_empty_segments(self) -> None:
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        logits = torch.tensor([0.0, 1.0, -0.5])
        confidence = torch.tensor([1.0, 0.5, 1.0])
        pooled, weights, sums = segment_attention(
            values, logits, torch.tensor([0, 0, 3]), 4, confidence
        )
        self.assertTrue(torch.allclose(segment_softmax(
            logits, torch.tensor([0, 0, 3]), 4, confidence
        ), weights))
        self.assertTrue(torch.allclose(sums, torch.tensor([1.0, 0.0, 0.0, 1.0])))
        self.assertTrue(torch.allclose(pooled[1:3], torch.zeros(2, 2)))
        self.assertTrue(torch.allclose(weights[:2].sum(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(weights[2:], torch.tensor([1.0])))

    def test_forward_has_edge_grid_diagnostics_and_gate_delta_gradients(self) -> None:
        result = self.model(
            self.geo,
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
            return_diagnostics=True,
        )
        coefficients = result["survival_coefficients"]
        self.assertEqual(tuple(coefficients.shape), (self.NUM_INSTANCES, 4, 7))
        self.assertEqual(tuple(result["edge_attention_grid_sum"].shape), (5, 2, 2))
        grid = result["edge_attention_grid_sum"]
        self.assertTrue(torch.allclose(grid[2, 0, 0], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(grid[2, 1, 1], torch.tensor(0.0)))
        self.assertTrue(torch.allclose(grid[4, 1, 1], torch.tensor(1.0)))
        self.assertTrue(torch.allclose(grid[0], torch.zeros(2, 2)))
        loss = coefficients.sum() + result["relation_gate"].sum() + result["relation_delta"].square().mean()
        loss.backward()
        self.assertIsNotNone(self.model.relation_gate_head.weight.grad)
        self.assertIsNotNone(self.model.relation_delta_head.weight.grad)
        self.assertTrue(torch.isfinite(self.model.relation_gate_head.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.relation_delta_head.weight.grad).all())

    def test_removing_an_edge_changes_relation_coefficients(self) -> None:
        full = self.model(
            self.geo,
            self.relation,
            self.local_group_ids,
            self.structural_group_ids,
        )
        reduced = dict(self.relation)
        keep = torch.tensor([False, True, True, True])
        for key in ("source_ids", "target_ids", "direction_ids", "depth_shell_ids", "edge_features"):
            reduced[key] = self.relation[key][keep]
        changed = self.model(
            self.geo,
            reduced,
            self.local_group_ids,
            self.structural_group_ids,
        )
        self.assertFalse(torch.allclose(full, changed))

    def test_train_only_relation_loss_is_finite_and_negatives_are_deterministic(self) -> None:
        source = torch.tensor([0, 1, 2, 3])
        target = torch.tensor([4, 5, 6, 7])
        direction = torch.zeros(4, dtype=torch.long)
        shell = torch.zeros(4, dtype=torch.long)
        distance_bucket = torch.tensor([0, 0, 1, 1])
        first = deterministic_negative_edges(
            source,
            target,
            direction,
            shell,
            distance_bucket_ids=distance_bucket,
            seed=7,
        )
        second = deterministic_negative_edges(
            source,
            target,
            direction,
            shell,
            distance_bucket_ids=distance_bucket,
            seed=7,
        )
        for key in first:
            self.assertTrue(torch.equal(first[key], second[key]))
        self.assertEqual(first["source_ids"].numel(), 4)
        self.assertTrue(torch.equal(torch.sort(first["source_ids"]).values, source))
        self.assertTrue(torch.equal(torch.sort(first["target_ids"]).values, target))
        self.assertTrue(torch.equal(first["feature_row_ids"].sort().values, torch.arange(4)))
        self.assertTrue(torch.equal(first["source_ids"], source[first["feature_row_ids"]]))
        for row, shuffled_target in zip(
            first["feature_row_ids"].tolist(), first["target_ids"].tolist(), strict=True
        ):
            expected_targets = {4, 5} if int(distance_bucket[row]) == 0 else {6, 7}
            self.assertIn(shuffled_target, expected_targets)
        self.assertFalse(bool((first["source_ids"] == first["target_ids"]).any()))
        loss, parts = self.model.relation_consistency_loss(
            self.geo,
            self.relation,
            instance_world_aabbs=self.aabbs,
            seed=7,
        )
        self.assertTrue(torch.isfinite(loss))
        for value in parts.values():
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.isfinite(value).all())
        with self.assertRaises(ValueError):
            non_train = dict(self.relation)
            non_train["metadata"] = {"schema": SCHEMA_V3, "trainOnly": True, "splitNames": ["validation"]}
            self.model.relation_consistency_loss(
                self.geo,
                non_train,
                instance_world_aabbs=self.aabbs,
                seed=7,
            )

    def test_negative_evidence_gather_recomputes_source_target_geometry(self) -> None:
        features = self.relation["edge_features"]
        rows = self.model._negative_feature_rows(
            features,
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([3]),
            self.aabbs,
        )
        scene_size = self.aabbs[:, 3:].amax(dim=0) - self.aabbs[:, :3].amin(dim=0)
        centers = (self.aabbs[:, :3] + self.aabbs[:, 3:]) * 0.5
        expected_center = (centers[0] - centers[3]) / scene_size
        self.assertTrue(torch.allclose(rows[0, 8:11], expected_center))
        self.assertTrue(torch.allclose(rows[0, :8], features[0, :8]))
        self.assertTrue(torch.isfinite(rows).all())

    def test_edge_cap_is_seeded_reproducible_and_changes_coverage(self) -> None:
        edge_count = 32
        relation = {
            "source_ids": torch.arange(edge_count),
            "target_ids": torch.arange(edge_count) + edge_count,
            "direction_ids": torch.zeros(edge_count, dtype=torch.long),
            "depth_shell_ids": torch.zeros(edge_count, dtype=torch.long),
            "edge_features": torch.zeros(edge_count, RELATION_FEATURE_DIM_V3),
        }
        first = self.model._limit_relation_edges(relation, max_edges=8, seed=100)
        replay = self.model._limit_relation_edges(relation, max_edges=8, seed=100)
        next_seed = self.model._limit_relation_edges(relation, max_edges=8, seed=101)
        self.assertTrue(torch.equal(first["source_ids"], replay["source_ids"]))
        self.assertFalse(torch.equal(first["source_ids"], next_seed["source_ids"]))
        self.assertEqual(first["source_ids"].numel(), 8)


if __name__ == "__main__":
    unittest.main()
