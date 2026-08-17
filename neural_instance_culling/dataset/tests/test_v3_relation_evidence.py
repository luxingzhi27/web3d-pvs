import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "dataset") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset"))
if str(ROOT / "model") not in sys.path:
    sys.path.insert(0, str(ROOT / "model"))

from build_triangle_depth_layer_evidence import (  # noqa: E402
    BACKGROUND_ID,
    build_survival_evidence_records,
    select_topk_evidence,
)
from build_observed_relation_csr import (  # noqa: E402
    derive_hierarchy_diameter_limits,
    summarize_topk_diagnostics_for_metadata,
    truncate_relation_topk,
)


class V3RelationEvidenceTests(unittest.TestCase):
    def test_event_and_right_censor_can_coexist_and_weights_conserve(self) -> None:
        ids = np.asarray([
            [[1, 1], [1, BACKGROUND_ID]],
            [[1, 1], [1, BACKGROUND_ID]],
            [[2, 2], [2, BACKGROUND_ID]],
            [[1, 1], [1, BACKGROUND_ID]],
        ], dtype=np.uint32)
        depths = np.asarray([
            [[0.1, 0.1], [0.1, 1.0]],
            [[0.4, 0.4], [0.4, 1.0]],
            [[0.8, 0.8], [0.8, 1.0]],
            [[0.9, 0.9], [0.9, 1.0]],
        ], dtype=np.float32)
        records, mass = build_survival_evidence_records(
            ids, depths, {1, 2, 3}, np.asarray([0, 0, 0, 0]),
            np.asarray([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]], dtype=np.float32),
            np.ones(4, dtype=np.float32), np.zeros(3, dtype=np.float32),
            1e-4, subpose_id=7,
        )
        target_one = [row for row in records if row["instance"] == 1]
        self.assertEqual({row["event"] for row in target_one}, {0, 1})
        self.assertAlmostEqual(sum(row["weight"] for row in records), 1.0, places=6)
        self.assertAlmostEqual(mass["weightSum"], 1.0, places=6)

    def test_topk_is_real_mass_sorted_and_reports_retained_quality(self) -> None:
        kept, total, quality, truncated = select_topk_evidence([
            {"source": 8, "pixelMass": 2.0},
            {"source": 3, "pixelMass": 9.0},
            {"source": 4, "pixelMass": 4.0},
        ], 2)
        self.assertEqual([row["source"] for row in kept], [3, 4])
        self.assertEqual(total, 15.0)
        self.assertAlmostEqual(quality, 13.0 / 15.0)
        self.assertTrue(truncated)

    def test_relation_topk_is_per_target_direction_shell(self) -> None:
        targets = np.asarray([0, 0, 0, 1], dtype=np.int64)
        directions = np.asarray([0, 0, 0, 0], dtype=np.int64)
        shells = np.asarray([0, 0, 0, 0], dtype=np.int64)
        sources = np.asarray([4, 2, 3, 5], dtype=np.uint32)
        features = np.zeros((4, 20), dtype=np.float32)
        features[:, 2] = [1.0, 8.0, 3.0, 2.0]
        features[:, 12] = 1.0
        result = truncate_relation_topk(targets, directions, shells, sources, features, k=2)
        kept_sources = result[3].tolist()
        self.assertEqual(kept_sources, [2, 3, 5])
        expected_quality = (np.log1p(8.0) + np.log1p(3.0)) / (
            np.log1p(1.0) + np.log1p(8.0) + np.log1p(3.0)
        )
        self.assertAlmostEqual(
            result[-1]["cells"]["0:0:0"]["retainedQuality"],
            expected_quality,
        )
        self.assertFalse(result[-1]["cells"]["1:0:0"]["truncated"])

        summary = summarize_topk_diagnostics_for_metadata(result[-1], worst_cell_limit=1)
        self.assertEqual(summary["cellCount"], 2)
        self.assertEqual(len(summary["worstCells"]), 1)
        self.assertAlmostEqual(
            summary["retainedQualityQuantiles"]["min"],
            expected_quality,
        )

    def test_hierarchy_diameters_are_frozen_from_train_observed_edges(self) -> None:
        centers = np.asarray([[0, 0, 0], [1, 0, 0], [4, 0, 0], [10, 0, 0]], dtype=np.float32)
        local, structural, meta = derive_hierarchy_diameter_limits(
            centers,
            np.asarray([0, 0, 0]),
            np.asarray([1, 2, 3]),
        )
        self.assertTrue(np.isfinite(local) and local > 0.0)
        self.assertTrue(np.isfinite(structural) and structural >= local)
        self.assertEqual(meta["source"], "train-only retained observed relation edge center distances")


if __name__ == "__main__":
    unittest.main()
