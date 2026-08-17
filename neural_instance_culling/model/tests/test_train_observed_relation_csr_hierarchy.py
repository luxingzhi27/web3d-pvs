from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.hierarchical_relation_survival import (  # noqa: E402
    HierarchicalOcclusionSurvivalEncoder,
    identity_hierarchy,
    query_monotone_survival,
)
from common.train_observed_relation_csr import (  # noqa: E402
    RELATION_FEATURE_DIM,
    ObservedRelationCSR,
    RelationSchemaError,
    make_relation_metadata,
    summarize_candidate_csr,
    validate_candidate_summary,
    validate_relation_metadata,
    validate_survival_observations,
)


def _candidate_summary(ids: np.ndarray, poses: np.ndarray, offsets: np.ndarray, n: int) -> dict:
    return summarize_candidate_csr(ids, poses, offsets, n).to_dict()


def _metadata(num_instances: int = 4, directions: int = 3, shells: int = 2) -> dict:
    canonical_ids = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
    canonical_poses = np.asarray([10, 11], dtype=np.int64)
    canonical_offsets = np.asarray([0, 3, 6], dtype=np.uint64)
    render_ids = np.asarray([0, 1, 2, 0, 1, 2, 3], dtype=np.uint32)
    render_poses = np.asarray([10, 10, 11], dtype=np.int64)
    render_offsets = np.asarray([0, 3, 5, 7], dtype=np.uint64)
    return make_relation_metadata(
        num_instances=num_instances,
        direction_bins=directions,
        depth_shells=shells,
        canonical_candidate_summary=_candidate_summary(
            canonical_ids, canonical_poses, canonical_offsets, num_instances
        ),
        render_candidate_summary=_candidate_summary(
            render_ids, render_poses, render_offsets, num_instances
        ),
        input_sha256={"depthCache": "a" * 64, "candidateCsr": "b" * 64},
        cache_schema="triangle-depth-layer-cache-v2",
        max_layers=6,
        model_input_fov_y_deg=66.0,
        surface_fallback={"included": True, "relationCount": 2},
    )


def _features(count: int) -> np.ndarray:
    values = np.zeros((count, RELATION_FEATURE_DIM), dtype=np.float32)
    values[:, 0] = 1.0  # pose support count
    values[:, 1] = 1.0  # pose support rate
    values[:, 2] = 4.0  # pixel support
    values[:, 3] = 0.5  # pixel fraction
    values[:, 4] = 0.2  # strictly positive depth gap
    values[:, 5] = 0.01
    values[:, 6] = 0.1  # strictly positive relative depth gap
    values[:, 7] = 0.01
    values[:, 8:11] = np.asarray([0.1, -0.2, 0.3], dtype=np.float32)
    values[:, 11] = 0.2
    values[:, 12] = 1.0
    return values


def _relation() -> ObservedRelationCSR:
    # Intentionally unsorted: from_rows must build target/direction/shell CSR
    # rows while preserving every source identity.
    targets = np.asarray([2, 1, 2, 2], dtype=np.int64)
    directions = np.asarray([1, 0, 0, 0], dtype=np.int64)
    shells = np.asarray([1, 0, 0, 0], dtype=np.int64)
    sources = np.asarray([3, 0, 1, 0], dtype=np.uint32)
    return ObservedRelationCSR.from_rows(
        num_instances=4,
        direction_bins=3,
        depth_shells=2,
        target_ids=targets,
        direction_ids=directions,
        depth_shell_ids=shells,
        source_ids=sources,
        edge_features=_features(len(sources)),
        source_types=np.asarray([0, 0, 1, 2], dtype=np.uint8),
        metadata=_metadata(),
    )


class TrainObservedRelationCSRTests(unittest.TestCase):
    def test_candidate_summary_and_digest_are_order_sensitive(self) -> None:
        ids = np.asarray([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        poses = np.asarray([10, 11], dtype=np.int64)
        offsets = np.asarray([0, 3, 6], dtype=np.uint64)
        summary = summarize_candidate_csr(ids, poses, offsets, 4)
        self.assertEqual(summary.pose_count, 2)
        self.assertEqual(summary.candidate_reference_count, 6)
        self.assertEqual(summary.min_candidates, 3)
        self.assertEqual(summary.unique_candidate_count, 4)
        validate_candidate_summary(ids, poses, offsets, 4, summary.to_dict())
        with self.assertRaises(RelationSchemaError):
            validate_candidate_summary(ids[::-1], poses, offsets, 4, summary.to_dict())

    def test_csr_rows_are_target_direction_depth_ordered_without_source_topk(self) -> None:
        relation = _relation()
        target, direction, shell = relation.row_indices()
        self.assertEqual(relation.edge_count, 4)
        self.assertEqual(list(zip(target.tolist(), direction.tolist(), shell.tolist())), [
            (1, 0, 0), (2, 0, 0), (2, 0, 0), (2, 1, 1)
        ])
        self.assertEqual(relation.source_ids.tolist(), [0, 0, 1, 3])
        self.assertIsNone(relation.metadata["sourceTopK"])
        self.assertEqual(relation.metadata["sourceTruncation"], "none_within_observed_coverage")

    def test_train_only_and_provenance_are_hard_schema_gates(self) -> None:
        relation = _relation()
        validate_relation_metadata(relation.metadata, expected_num_instances=4)
        not_train = copy.deepcopy(relation.metadata)
        not_train["trainOnly"] = False
        with self.assertRaises(RelationSchemaError):
            validate_relation_metadata(not_train)
        wrong_split = copy.deepcopy(relation.metadata)
        wrong_split["splitNames"] = ["train", "validation"]
        with self.assertRaises(RelationSchemaError):
            validate_relation_metadata(wrong_split)
        topk = copy.deepcopy(relation.metadata)
        topk["sourceTopK"] = 8
        with self.assertRaises(RelationSchemaError):
            validate_relation_metadata(topk)

    def test_depth_gap_and_event_censoring_semantics_are_checked(self) -> None:
        bad = _relation()
        bad.edge_features[0, 4] = 0.0
        with self.assertRaises(RelationSchemaError):
            bad.validate()
        result = validate_survival_observations(
            {
                "instance": np.asarray([0, 1, 2]),
                "direction": np.asarray([0, 1, 2]),
                "rho": np.asarray([0.2, 0.4, 0.8]),
                "event": np.asarray([1, 0, 1]),
                "weight": np.asarray([1.0, 0.5, 2.0]),
            },
            num_instances=4,
            direction_bins=3,
        )
        self.assertEqual(result, {"observationCount": 3, "eventCount": 2, "rightCensoredCount": 1})
        with self.assertRaises(RelationSchemaError):
            validate_survival_observations(
                {
                    "instance": [0], "direction": [0], "rho": [0.2],
                    "event": [2], "weight": [1.0]
                },
                num_instances=4,
                direction_bins=3,
            )

    def test_binary_roundtrip_and_torch_conversion(self) -> None:
        relation = _relation()
        with tempfile.TemporaryDirectory() as temp:
            relation.save(temp)
            loaded = ObservedRelationCSR.load(temp, allow_legacy=True)
            np.testing.assert_array_equal(loaded.row_offsets, relation.row_offsets)
            np.testing.assert_array_equal(loaded.source_ids, relation.source_ids)
            np.testing.assert_allclose(loaded.edge_features, relation.edge_features)
            tensors = loaded.to_torch()
            self.assertEqual(tuple(tensors["edge_features"].shape), (4, RELATION_FEATURE_DIM))


class HierarchicalRelationSurvivalTests(unittest.TestCase):
    def test_direction_depth_segments_and_three_level_aggregation(self) -> None:
        torch.manual_seed(13)
        relation = _relation()
        encoder = HierarchicalOcclusionSurvivalEncoder(
            geo_dim=8, hidden_dim=12, direction_bins=3, depth_shells=2
        )
        local = torch.asarray([0, 0, 1, 1], dtype=torch.long)
        structural = torch.asarray([0, 0], dtype=torch.long)
        geo = torch.randn(4, 8)
        diagnostics = encoder(
            geo, relation, local, structural, return_diagnostics=True
        )
        coefficients = diagnostics["survival_coefficients"]
        self.assertEqual(tuple(coefficients.shape), (4, 4, 7))
        self.assertTrue(bool(torch.isfinite(coefficients).all()))
        shell_counts = diagnostics["shell_counts"]
        self.assertEqual(float(shell_counts[2, 0, 0, 0]), 2.0)
        self.assertEqual(float(shell_counts[2, 1, 1, 0]), 1.0)
        self.assertEqual(diagnostics["local_group_counts"].flatten().tolist(), [2.0, 2.0])
        self.assertEqual(diagnostics["structural_group_counts"].flatten().tolist(), [2.0])
        loss = coefficients.square().mean()
        loss.backward()
        self.assertIsNotNone(encoder.edge_message[0].weight.grad)
        self.assertTrue(bool(torch.isfinite(encoder.edge_message[0].weight.grad).all()))

    def test_reversing_source_and_target_changes_the_ordered_segment_owner(self) -> None:
        relation = _relation()
        reversed_relation = ObservedRelationCSR.from_rows(
            num_instances=4,
            direction_bins=3,
            depth_shells=2,
            target_ids=np.asarray([0], dtype=np.int64),
            direction_ids=np.asarray([0], dtype=np.int64),
            depth_shell_ids=np.asarray([0], dtype=np.int64),
            source_ids=np.asarray([2], dtype=np.uint32),
            edge_features=_features(1),
            source_types=np.asarray([0], dtype=np.uint8),
            metadata=_metadata(),
        )
        encoder = HierarchicalOcclusionSurvivalEncoder(
            geo_dim=8, hidden_dim=8, direction_bins=3, depth_shells=2
        )
        geo = torch.randn(4, 8)
        local = torch.asarray([0, 0, 1, 1], dtype=torch.long)
        structural = torch.asarray([0, 0], dtype=torch.long)
        first = encoder(geo, relation, local, structural, return_diagnostics=True)
        second = encoder(geo, reversed_relation, local, structural, return_diagnostics=True)
        self.assertEqual(float(first["shell_counts"][2, 0, 0, 0]), 2.0)
        self.assertEqual(float(second["shell_counts"][0, 0, 0, 0]), 1.0)
        self.assertEqual(float(second["shell_counts"][2].sum()), 0.0)

    def test_survival_query_is_monotone_in_depth(self) -> None:
        torch.manual_seed(5)
        coefficients = torch.randn(3, 4, 7)
        direction_basis = torch.randn(3, 4)
        rho = torch.asarray([0.1, 0.5, 0.9])
        survival = query_monotone_survival(coefficients, direction_basis, rho)
        # Each row uses a separate instance, so test the same field over a
        # repeated coefficient/direction query for the monotonicity contract.
        repeated = query_monotone_survival(
            coefficients[:1].expand(3, -1, -1),
            direction_basis[:1].expand(3, -1),
            rho,
        )
        self.assertTrue(bool(torch.isfinite(survival).all()))
        self.assertTrue(bool((repeated[:-1] >= repeated[1:]).all()))


if __name__ == "__main__":
    unittest.main()
