from __future__ import annotations

import sys
import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset"
MODEL = ROOT / "model"
if str(DATASET) not in sys.path:
    sys.path.insert(0, str(DATASET))
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from build_ray_context_relation_evidence import (  # noqa: E402
    _aggregate_rows,
    _copy_survival_observations,
    _soft_direction_neighbors,
    _surface_fallback_pose_rows,
)
from build_triangle_depth_layer_evidence import (  # noqa: E402
    SURFACE_FALLBACK_RELATION_DTYPE,
    spherical_direction_anchors,
)
from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402


class RayContextRelationEvidenceTest(unittest.TestCase):
    def test_repeated_render_source_poses_have_a_distinct_digest_from_canonical_split(self) -> None:
        class Dataset:
            def frustum_slice(self, pose_index: int) -> np.ndarray:
                return np.asarray([pose_index, pose_index + 10], dtype=np.uint32)

        dataset = Dataset()
        canonical = candidate_digest_for_pose_sequence(dataset, [0, 1])
        repeated_render = candidate_digest_for_pose_sequence(dataset, [0, 0, 1])
        self.assertEqual(len(canonical), 64)
        self.assertEqual(len(repeated_render), 64)
        self.assertNotEqual(canonical, repeated_render)

    def test_pose_source_duplicates_are_removed_and_support_is_pose_based(self) -> None:
        rows = np.asarray([
            [2, 1, 0, 0, 4, 0.20, 0.01, 10],
            [2, 1, 0, 0, 3, 0.30, 0.02, 10],
            [2, 3, 0, 0, 2, 0.40, 0.01, 10],
            [2, 1, 0, 0, 5, 0.20, 0.01, 11],
            [2, 3, 0, 0, 1, 0.50, 0.01, 11],
        ], dtype=np.float32)
        source_ids, stats, _strength, count, meta = _aggregate_rows(
            rows, num_instances=5, source_k=8, pixel_denominator=320 * 180
        )
        self.assertEqual(meta["deduplicatedPoseSourceRows"], 4)
        self.assertEqual(int(count[2, 0, 0]), 2)
        self.assertAlmostEqual(float(stats[2, 0, 0, 0, 2]), 1.0, places=6)
        valid = stats[2, 0, 0, :, 0] > 0
        self.assertEqual(int(np.unique(source_ids[2, 0, 0, valid]).size), int(valid.sum()))

    def test_soft_direction_weights_are_normalized(self) -> None:
        anchors = spherical_direction_anchors()
        directions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        ids, weights = _soft_direction_neighbors(directions, anchors, top_k=3, concentration=8.0)
        self.assertEqual(ids.shape, (2, 3))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertTrue(np.all(weights >= 0.0))

    def test_surface_fallback_rows_keep_only_candidate_relations_and_preserve_targets(self) -> None:
        rows = np.asarray([
            (7, 1, 3, 8, 0.25, 0.0),
            (7, 2, 3, 4, 0.10, 0.0),
        ], dtype=SURFACE_FALLBACK_RELATION_DTYPE)
        candidate_mask = np.asarray([False, True, True, True, False], dtype=bool)
        centers = np.asarray([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 3.0],
            [0.0, 0.0, 4.0],
        ], dtype=np.float32)
        converted, target_count = _surface_fallback_pose_rows(
            rows,
            candidate_mask,
            centers,
            np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            min_depth_gap=0.01,
            camera_far=10.0,
        )
        self.assertEqual(converted.shape, (2, 6))
        self.assertEqual(target_count, 1)
        self.assertTrue(np.all(converted[:, 0] == 3.0))
        self.assertTrue(np.all(converted[:, 1] != converted[:, 0]))
        # Fallback gaps are normalized depth-buffer differences.  With a
        # target-center depth of 0.3, 0.25 belongs to the far shell and 0.10
        # belongs to the middle shell; neither is divided by world meters.
        self.assertEqual(converted[0, 2], 2.0)
        self.assertEqual(converted[1, 2], 1.0)

    def test_surface_fallback_outside_candidate_is_rejected(self) -> None:
        rows = np.asarray([(7, 1, 4, 8, 0.25, 0.0)], dtype=SURFACE_FALLBACK_RELATION_DTYPE)
        with self.assertRaises(ValueError):
            _surface_fallback_pose_rows(
                rows,
                np.asarray([False, True, True, True, False], dtype=bool),
                np.zeros((5, 3), dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                min_depth_gap=0.01,
            )

    def test_stale_survival_evidence_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "survival"
            output = root / "relation"
            source.mkdir()
            output.mkdir()
            (source / "evidence_meta.json").write_text(json.dumps({
                "schema": "triangle-depth-layer-evidence-v1",
                "cacheSchema": "triangle-depth-layer-cache-v2",
                "modelInputFovYDeg": 66.0,
                "splits": ["train"],
                "candidateDigest": "stale-cache",
                "stats": {"poseCount": 1},
                "files": {},
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                _copy_survival_observations(
                    source,
                    output,
                    selected_pose_count=1,
                    expected_candidate_digest="current-cache",
                    expected_cache_schema="triangle-depth-layer-cache-v2",
                    expected_splits=["train"],
                )


if __name__ == "__main__":
    unittest.main()
