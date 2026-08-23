from __future__ import annotations

from pathlib import Path
import unittest

from neural_instance_culling.benchmark.evaluate_pvs import (
    DIAGNOSTIC_RECALIBRATION_SCHEMA,
)
from neural_instance_culling.benchmark.reaudit_pvs import (
    PAIR_METRICS,
    RETAINED_EPOCHS,
    _corrected_calibration_summary,
    build_reaudit_command,
    select_checkpoint_reaudit,
)
from neural_instance_culling.benchmark.run_pvs import (
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
)


DATA_ROOT = Path("/mnt/sda/rhyang/slm")


def _payload(
    epoch: int,
    *,
    balanced_accuracy: float,
    precision: float,
    validation_lcb: float = 0.995,
    replicates: int = 10_000,
) -> dict:
    calibration = {
        "threshold": 0.4,
        "aggregateWeightedRecall": 0.997,
        "aggregateWeightedRecallLowerConfidenceBound": 0.994,
        "agg_useful_cull": 0.8,
    }
    validation = {
        "threshold": 0.4,
        "aggregateWeightedRecall": 0.998,
        "aggregateWeightedRecallLowerConfidenceBound": validation_lcb,
        "agg_balanced_accuracy": balanced_accuracy,
        "agg_precision": precision,
        "agg_accuracy": 0.9,
        "agg_useful_cull": 0.85,
        "avg_pred_count": 500.0,
    }
    return {
        "schema": DIAGNOSTIC_RECALIBRATION_SCHEMA,
        "status": "safe",
        "checkpoint": f"/tmp/checkpoint_epoch_{epoch:03d}.pt",
        "epoch": epoch,
        "seed": 20260801,
        "selectionRule": "calibration-only",
        "bootstrapReplicates": replicates,
        "selectedSafe": calibration,
        "diagnostic": calibration,
        "validationAtSelectedThreshold": validation,
        "calibrationThresholdRows": [calibration],
        "testRead": False,
    }


class IntegratedVisibilityProtocolFixTests(unittest.TestCase):
    def test_registered_matrix_is_five_variants_by_three_seeds(self) -> None:
        self.assertEqual(len(FORMAL_VARIANTS), 5)
        self.assertEqual(FORMAL_SEEDS, (20260801, 20260802, 20260803))
        self.assertIn("balancedAccuracy", PAIR_METRICS)
        self.assertIn("weightedRecall", PAIR_METRICS)
        self.assertIn("predictedGlbBytes", PAIR_METRICS)

    def test_reaudit_command_uses_separate_registered_bootstrap_seeds(self) -> None:
        command = build_reaudit_command(
            DATA_ROOT,
            Path("/tmp/checkpoint.pt"),
            Path("/tmp/output.json"),
            seed=20260801,
            replicates=10_000,
        )
        self.assertEqual(
            command[command.index("--seed") + 1],
            str(20260801 + 50_000),
        )
        self.assertEqual(
            command[command.index("--recalibration-validation-seed") + 1],
            str(20260801 + 60_000),
        )
        self.assertEqual(
            command[command.index("--recalibration-bootstrap-replicates") + 1],
            "10000",
        )
        self.assertNotIn("test", command)

    def test_checkpoint_selection_requires_all_snapshots_and_10000_replicates(self) -> None:
        payloads = [
            _payload(epoch, balanced_accuracy=0.8 + epoch / 1000.0, precision=0.4)
            for epoch in RETAINED_EPOCHS
        ]
        selected, safe = select_checkpoint_reaudit(payloads, replicates=10_000)
        self.assertTrue(safe)
        self.assertEqual(selected["epoch"], RETAINED_EPOCHS[-1])

        with self.assertRaisesRegex(ValueError, "10000"):
            select_checkpoint_reaudit(payloads, replicates=2_000)
        with self.assertRaisesRegex(ValueError, "every retained"):
            select_checkpoint_reaudit(payloads[:-1], replicates=10_000)

    def test_validation_unsafe_checkpoint_cannot_win_on_classification(self) -> None:
        payloads = [
            _payload(epoch, balanced_accuracy=0.8, precision=0.4)
            for epoch in RETAINED_EPOCHS
        ]
        payloads[-1] = _payload(
            RETAINED_EPOCHS[-1],
            balanced_accuracy=0.99,
            precision=0.99,
            validation_lcb=0.98,
        )
        selected, safe = select_checkpoint_reaudit(payloads, replicates=10_000)
        self.assertTrue(safe)
        self.assertNotEqual(selected["epoch"], RETAINED_EPOCHS[-1])

    def test_corrected_summary_keeps_calibration_threshold_and_test_unread(self) -> None:
        payload = _payload(24, balanced_accuracy=0.9, precision=0.5)
        summary, best = _corrected_calibration_summary(
            payload,
            safe=True,
            replicates=10_000,
        )
        self.assertEqual(summary["status"], "safe")
        self.assertEqual(summary["bootstrapReplicates"], 10_000)
        self.assertEqual(summary["calibration"]["bootstrapReplicates"], 10_000)
        self.assertEqual(best["epoch"], 24)
        self.assertEqual(best["threshold"], 0.4)
        self.assertIs(summary["testRead"], False)
        self.assertIs(summary["calibration"]["testRead"], False)


if __name__ == "__main__":
    unittest.main()
