from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from evaluate_unified_pvs_metrics import (  # noqa: E402
    load_frozen_thresholds,
    select_workpoints,
    validate_formal_test_specs,
)


class UnifiedPvsFormalTestTests(unittest.TestCase):
    def _fixture(self) -> tuple[dict[str, dict[str, str]], Path, dict[str, float]]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        checkpoint = root / "best.pt"
        calibration = root / "calibration.json"
        checkpoint.write_bytes(b"fixture")
        calibration.write_text(
            json.dumps(
                {
                    "schema": "pvs-aabb-ray-calibration-v1",
                    "status": "safe",
                    "selectionSplit": "calibration",
                    "testEvaluationCount": 0,
                    "testRead": False,
                    "bestSafe": {
                        "selection": {
                            "threshold": 0.7,
                            "aggregateWeightedRecall": 0.997,
                            "aggregateWeightedRecallLowerConfidenceBound": 0.995,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        registration = root / "thresholds.json"
        registration.write_text(
            json.dumps(
                {
                    "testRead": False,
                    "thresholds": {
                        "learned": {
                            "threshold": 0.7,
                            "selectionSplit": "calibration",
                            "checkpoint": str(checkpoint),
                            "calibration": str(calibration),
                            "testRead": False,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        specs = {
            "learned": {
                "kind": "learned_aabb_ray",
                "checkpoint": str(checkpoint),
                "eval_summary": str(calibration),
            }
        }
        return specs, registration, {"learned": 0.7}

    def test_formal_test_requires_checkpoint_calibration_and_matching_registration(self) -> None:
        specs, registration, thresholds = self._fixture()
        validate_formal_test_specs(specs, registration, thresholds)

        changed = copy.deepcopy(specs)
        changed["learned"].pop("eval_summary")
        with self.assertRaisesRegex(ValueError, "checkpoint and calibration"):
            validate_formal_test_specs(changed, registration, thresholds)

        changed_registration = registration.with_name("changed_thresholds.json")
        changed_registration.write_text(
            registration.read_text(encoding="utf-8").replace('"threshold": 0.7', '"threshold": 0.6'),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "threshold disagrees"):
            validate_formal_test_specs(specs, changed_registration, {"learned": 0.6})

    def test_test_tainted_registration_is_rejected(self) -> None:
        specs, registration, thresholds = self._fixture()
        payload = json.loads(registration.read_text(encoding="utf-8"))
        payload["thresholds"]["learned"]["testRead"] = True
        registration.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "test-tainted"):
            validate_formal_test_specs(specs, registration, thresholds)

    def test_frozen_threshold_loader_rejects_test_read_registration(self) -> None:
        _specs, registration, _thresholds = self._fixture()
        payload = json.loads(registration.read_text(encoding="utf-8"))
        payload["testRead"] = True
        registration.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "before test"):
            load_frozen_thresholds(registration, ["learned"])

    def test_primary_workpoint_uses_aggregate_weighted_safety(self) -> None:
        row = {
            "pose_weighted_recall": 0.999,
            "weighted_recall_lower_confidence_bound": 0.999,
            "aggregateWeightedRecall": 0.98,
            "aggregateWeightedRecallLowerConfidenceBound": 0.97,
            "pose_precision": 0.99,
            "pose_f1": 0.9,
            "pose_balanced_accuracy": 0.8,
            "pose_accuracy": 0.8,
            "avg_pred_count": 1.0,
            "pose_visual_utility_recall": 1.0,
            "pose_recall": 1.0,
            "safety_adjusted_cull_score": 0.5,
        }
        workpoints = select_workpoints([row], 0.95, 0.99, 0.98)
        self.assertIsNone(workpoints["primaryWeightedPrecision"])


if __name__ == "__main__":
    unittest.main()
