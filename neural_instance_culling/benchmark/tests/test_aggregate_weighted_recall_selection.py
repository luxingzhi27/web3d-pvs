from __future__ import annotations

import sys
from pathlib import Path
import unittest

MODEL = Path(__file__).resolve().parents[2] / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_recall_safe_rows,
    select_aggregate_weighted_cull_workpoint,
)


class AggregateWeightedRecallSelectionTest(unittest.TestCase):
    def test_macro_pass_cannot_rescue_aggregate_failure(self) -> None:
        rows = [{
            "threshold": 0.2,
            "poseMacroWeightedRecall": 0.998,
            "aggregateWeightedRecall": 0.982,
            "aggregateWeightedRecallLowerConfidenceBound": 0.975,
            "agg_useful_cull": 0.9,
            "agg_balanced_accuracy": 0.7,
            "agg_precision": 0.3,
            "avg_pred_count": 10,
        }]
        self.assertEqual(aggregate_weighted_recall_safe_rows(rows, 0.99, minimum_lower_confidence_bound=0.99), [])
        self.assertIsNone(select_aggregate_weighted_cull_workpoint(rows, 0.99, 0.99))

    def test_aggregate_pass_is_selected_even_when_macro_is_lower(self) -> None:
        rows = [{
            "threshold": 0.35,
            "poseMacroWeightedRecall": 0.988,
            "aggregateWeightedRecall": 0.994,
            "aggregateWeightedRecallLowerConfidenceBound": 0.991,
            "agg_useful_cull": 0.82,
            "agg_balanced_accuracy": 0.76,
            "agg_precision": 0.22,
            "avg_pred_count": 40,
        }]
        selected = select_aggregate_weighted_cull_workpoint(rows, 0.99, 0.99)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["threshold"], 0.35)

    def test_missing_canonical_lcb_is_not_treated_as_a_point_estimate(self) -> None:
        rows = [{
            "threshold": 0.4,
            "poseMacroWeightedRecall": 1.0,
            "aggregateWeightedRecall": 1.0,
            "agg_useful_cull": 0.1,
        }]
        self.assertIsNone(select_aggregate_weighted_cull_workpoint(rows, 0.99, 0.99))


if __name__ == "__main__":
    unittest.main()
