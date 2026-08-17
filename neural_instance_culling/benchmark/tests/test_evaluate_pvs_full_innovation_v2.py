from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import numpy as np

from evaluate_pvs_full_innovation_v2 import (  # noqa: E402
    _download_head_metrics,
    _ndcg,
    _ranked_budget_ids,
    self_test,
)
from evaluate_pvs_hierarchical_relation_survival_integrated import (  # noqa: E402
    _camel_metrics,
    _resource_metrics,
    _select_threshold,
)


class FullInnovationEvaluationTests(unittest.TestCase):
    def test_self_test_contract(self) -> None:
        self.assertEqual(self_test()["status"], "passed")

    def test_ndcg_is_one_for_ideal_order(self) -> None:
        self.assertAlmostEqual(_ndcg(np.asarray([2, 1, 3]), {1: 0.5, 2: 1.0}, 2), 1.0)

    def test_budget_skips_an_oversized_item_and_continues(self) -> None:
        selected = _ranked_budget_ids(
            np.asarray([0, 1, 2], dtype=np.int64),
            np.asarray([8.0, 100.0, 7.0], dtype=np.float64),
            15.0,
        )
        self.assertEqual(selected.tolist(), [0, 2])

    def test_download_head_is_separate_from_visibility_baseline(self) -> None:
        result = _download_head_metrics(
            candidate_ids=np.asarray([0, 1, 2, 3], dtype=np.int64),
            download_logits=np.asarray([4.0, 3.0, 1.0, 0.0], dtype=np.float32),
            utility_logits=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            visibility_logits=np.asarray([0.0, 0.0, 4.0, 3.0], dtype=np.float32),
            visible_ids=np.asarray([2, 3], dtype=np.uint32),
            visible_weights=np.asarray([2.0, 1.0], dtype=np.float32),
            instance_to_glb=np.asarray([0, 0, 1, 1], dtype=np.int64),
            glb_bytes=np.asarray([10.0, 10.0], dtype=np.float64),
        )
        self.assertNotEqual(
            result["downloadHead"]["topGlbIds"],
            result["visibilityAggregateBaseline"]["topGlbIds"] if "topGlbIds" in result["visibilityAggregateBaseline"] else [],
        )
        self.assertIn("ndcgAt20", result["downloadHead"])
        self.assertIn("fraction_20pct", result["downloadBudgetUtility"])

    def test_replay_reads_the_frozen_calibration_threshold_without_reselection(self) -> None:
        checkpoint = {"best": {"selection": {"threshold": 0.25}}}
        summary = {
            "schema": "pvs-hierarchical-relation-survival-integrated-calibration-v2",
            "protocol": "calibration_ready_pre_test",
            "testRead": False,
            "selected": {
                "threshold": 0.25,
                "aggregateWeightedRecall": 1.0,
                "aggregateWeightedRecallLowerConfidenceBound": 1.0,
            },
        }
        threshold, source = _select_threshold(checkpoint, False, summary)
        self.assertEqual(threshold, 0.25)
        self.assertEqual(source["source"], "calibration_ready_summary.selected")
        summary["selected"]["threshold"] = 0.5
        with self.assertRaises(ValueError):
            _select_threshold(checkpoint, False, summary)

    def test_resource_contract_produces_required_glb_count_and_bytes(self) -> None:
        result = _resource_metrics(
            np.asarray([0, 1, 2], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            {"targetIds": np.asarray([1, 2], dtype=np.uint32), "targetWeights": np.asarray([1.0, 1.0])},
            np.asarray([0, 1, 1], dtype=np.int64),
            np.asarray([10.0, 20.0], dtype=np.float64),
        )
        self.assertEqual(result["requiredGlbCount"], 1.0)
        self.assertEqual(result["requiredGlbBytes"], 20.0)

    def test_unavailable_image_metrics_are_not_zero_filled(self) -> None:
        metrics = _camel_metrics({}, {})
        self.assertNotIn("missPixelRate", metrics)
        self.assertNotIn("wrongIdPixelRate", metrics)
        self.assertNotIn("extraPixelRate", metrics)


if __name__ == "__main__":
    unittest.main()
