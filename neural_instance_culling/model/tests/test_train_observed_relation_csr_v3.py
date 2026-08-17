import sys
import unittest
from pathlib import Path

import numpy as np

MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.train_observed_relation_csr import (
    RelationSchemaError,
    bounded_hierarchy_ids,
    degree_preserving_directed_edge_swap,
    normalize_radius_relative_log_depth,
    recompute_relative_geometry,
    radius_relative_log_depth,
    summarize_bounded_hierarchy,
    train_depth_quantiles,
    validate_survival_observations_v3,
)


class TrainObservedRelationCSRv3Tests(unittest.TestCase):
    def test_bounded_hierarchy_caps_local_diameter_and_structural_fraction(self) -> None:
        n = 100
        source = np.asarray([i for i in range(1, n)], dtype=np.uint32)
        target = np.asarray([i - 1 for i in range(1, n)], dtype=np.uint32)
        confidence = np.ones(n - 1, dtype=np.float32)
        centers = np.column_stack((np.arange(n), np.zeros(n), np.zeros(n))).astype(np.float32)
        local, structural = bounded_hierarchy_ids(
            n, target, source, confidence, centers,
            local_max_size=32, local_diameter=31.0,
            structural_max_size=64, structural_diameter=50.0,
            max_structural_fraction=0.10,
        )
        summary = summarize_bounded_hierarchy(local, structural, centers)
        self.assertLessEqual(summary["maxLocalInstanceCount"], 32)
        self.assertLessEqual(summary["maxLocalDiameter"], 31.0)
        self.assertLessEqual(summary["maxStructuralLocalGroupCount"], 64)
        self.assertLessEqual(summary["maxStructuralInstanceFraction"], 0.10)
        self.assertLessEqual(summary["maxStructuralDiameter"], 50.0)
        self.assertEqual(np.unique(local).tolist(), list(range(int(local.max()) + 1)))
        self.assertEqual(np.unique(structural).tolist(), list(range(int(structural.max()) + 1)))

    def test_directed_swap_preserves_degrees_and_has_no_self_or_duplicate_edges(self) -> None:
        source = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.uint32)
        target = np.asarray([3, 4, 3, 5, 4, 5], dtype=np.uint32)
        direction = np.zeros(6, dtype=np.uint8)
        shell = np.zeros(6, dtype=np.uint8)
        swapped_source, swapped_target = degree_preserving_directed_edge_swap(
            source, target, direction, shell, seed=9, swap_fraction=1.0
        )
        self.assertEqual(np.bincount(swapped_source, minlength=6).tolist(), np.bincount(source, minlength=6).tolist())
        self.assertEqual(np.bincount(swapped_target, minlength=6).tolist(), np.bincount(target, minlength=6).tolist())
        self.assertTrue(np.all(swapped_source != swapped_target))
        self.assertEqual(np.unique(np.column_stack((swapped_source, swapped_target)), axis=0).shape[0], 6)

    def test_directed_swap_stays_inside_distance_strata(self) -> None:
        source = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint32)
        target = np.asarray([8, 9, 10, 11, 12, 13, 14, 15], dtype=np.uint32)
        direction = np.zeros(8, dtype=np.uint8)
        shell = np.zeros(8, dtype=np.uint8)
        bucket = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint8)
        _, shuffled_target = degree_preserving_directed_edge_swap(
            source,
            target,
            direction,
            shell,
            distance_bucket_ids=bucket,
            seed=4,
            swap_fraction=1.0,
        )
        self.assertEqual(sorted(shuffled_target[:4].tolist()), sorted(target[:4].tolist()))
        self.assertEqual(sorted(shuffled_target[4:].tolist()), sorted(target[4:].tolist()))

    def test_v3_observations_require_train_subpose_conservation(self) -> None:
        observations = {
            "instance": [1, 1], "direction": [0, 0], "normalizedDepth": [0.2, 0.2],
            "event": [0, 1], "weight": [0.4, 0.6], "subpose": [8, 8],
            "rawPixelCount": [4, 6],
        }
        result = validate_survival_observations_v3(observations, num_instances=3, direction_bins=1)
        self.assertEqual(result["eventCount"], 1)
        with self.assertRaises(RelationSchemaError):
            broken = dict(observations)
            broken["weight"] = [0.4, 0.5]
            validate_survival_observations_v3(broken, num_instances=3, direction_bins=1)

    def test_radius_relative_depth_normalization_is_monotone_and_train_only(self) -> None:
        raw = radius_relative_log_depth([1.0, 2.0, 8.0], [1.0, 1.0, 1.0])
        stats = train_depth_quantiles(raw)
        normalized = normalize_radius_relative_log_depth(raw, stats)
        self.assertTrue(np.all(np.diff(raw) > 0))
        self.assertTrue(np.all(np.diff(normalized) >= 0))
        self.assertEqual(stats["sourceSplit"], "train")

    def test_swapped_edges_require_recomputed_relative_geometry(self) -> None:
        geometry = recompute_relative_geometry(
            [0, 1], [2, 3],
            [[0, 0, 0], [1, 0, 0], [0, 0, 2], [2, 0, 0]],
            [1, 1, 2, 1], [0, 1, 3, 2],
            train_depth_quantiles([0.0, 1.0]),
        )
        np.testing.assert_allclose(geometry["relative_center"], [[0, 0, -2], [-1, 0, 0]])
        self.assertEqual(geometry["normalized_log_depth"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
