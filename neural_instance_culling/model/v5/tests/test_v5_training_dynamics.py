from __future__ import annotations

import unittest

from neural_instance_culling.model.v5.analyze_training_dynamics import (
    bin_metric_rows,
    summarize_groups,
)


class TrainingDynamicsTests(unittest.TestCase):
    def test_bins_by_dual_group_local_update(self) -> None:
        rows = []
        for update in range(1, 5):
            rows.append({
                "run": "pilot",
                "sourceKind": "synthetic",
                "dualGroupId": "synthetic_family:city",
                "dualGroupUpdate": update,
                "riskExtra": float(update),
                "riskCount": 1.0,
                "riskVisual": 2.0,
                "lambdaCount": 3.0,
                "lambdaVisual": 4.0,
            })
        binned = bin_metric_rows(rows, 2)
        self.assertEqual(len(binned), 2)
        self.assertEqual(binned[0]["sampleCount"], 2)
        self.assertEqual(binned[0]["riskExtra"], 1.5)
        self.assertEqual(binned[1]["riskExtra"], 3.5)
        summary = summarize_groups(binned, 2)
        group = summary["groups"]["pilot:synthetic_family:city"]
        self.assertEqual(group["first"]["riskExtra"], 1.5)
        self.assertEqual(group["last"]["riskExtra"], 3.5)


if __name__ == "__main__":
    unittest.main()
