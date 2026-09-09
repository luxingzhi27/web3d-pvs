from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from generate_threshold_curves import (  # noqa: E402
    CURVE_SCHEMA,
    ResourceTable,
    SidecarData,
    _thresholds,
    _valid_sidecar,
    main,
    score_curve,
)
from score_sidecar import ScoreSidecarWriter  # noqa: E402


class ThresholdCurveTests(unittest.TestCase):
    def _data(self, scores, targets, weights, offsets, split="calibration") -> SidecarData:
        size = len(scores)
        return SidecarData(
            source=f"{split}-fixture",
            split=split,
            manifest={"schema": "pvs-typed-score-sidecar-v1", "checkpoint": "fixture.pt"},
            pose_indices=np.arange(len(offsets) - 1, dtype=np.int64),
            pose_offsets=np.asarray(offsets, dtype=np.int64),
            scores=np.asarray(scores, dtype=np.float32),
            targets=np.asarray(targets, dtype=np.uint8),
            weights=np.asarray(weights, dtype=np.float64),
            candidate_ids=np.arange(size, dtype=np.uint32),
        )

    @staticmethod
    def _resources(count: int) -> ResourceTable:
        return ResourceTable(
            instance_to_glb=np.arange(count, dtype=np.int64),
            glb_bytes=np.arange(1, count + 1, dtype=np.float64) * 10.0,
        )

    def test_calibration_change_points_are_deterministically_compressed(self) -> None:
        scores = np.linspace(0.0, 1.0, 1000, dtype=np.float32)
        points = _thresholds(scores, 16)
        self.assertEqual(points.size, 16)
        self.assertEqual(points[0], scores[0])
        self.assertEqual(points[-1], scores[-1])

    def test_equal_scores_enter_as_one_prediction_event(self) -> None:
        data = self._data([0.7, 0.7, 0.2], [1, 0, 0], [1, 0, 0], [0, 3])
        row = score_curve(
            data,
            np.asarray([0.7], dtype=np.float32),
            self._resources(3),
            bootstrap_replicates=0,
            bootstrap_seed=7,
            target_weighted_recall=0.99,
            threshold_source="fixture",
        )[0]
        self.assertEqual((row["tp"], row["fp"], row["fn"], row["tn"]), (1, 1, 0, 1))
        self.assertAlmostEqual(row["precision"], 0.5)

    def test_zero_gt_pose_is_counted_and_weighted_recall_is_neutral(self) -> None:
        data = self._data([0.8, 0.1, 0.9, 0.2], [0, 0, 1, 0], [0, 0, 1, 0], [0, 2, 4])
        row = score_curve(
            data,
            np.asarray([0.9], dtype=np.float32),
            self._resources(4),
            bootstrap_replicates=0,
            bootstrap_seed=7,
            target_weighted_recall=0.99,
            threshold_source="fixture",
        )[0]
        self.assertEqual(row["zero_gt_pose_count"], 1)
        self.assertEqual(row["weighted_recall"], 1.0)
        self.assertEqual(row["weighted_recall_lower_confidence_bound"], 1.0)
        self.assertEqual(row["bad_cull"], 0.0)

    def test_safety_requires_strictly_above_target_for_point_and_lcb(self) -> None:
        data = self._data([0.9, 0.1], [1, 1], [99, 1], [0, 2])
        row = score_curve(
            data,
            np.asarray([0.9], dtype=np.float32),
            self._resources(2),
            bootstrap_replicates=0,
            bootstrap_seed=7,
            target_weighted_recall=0.99,
            threshold_source="fixture",
        )[0]
        self.assertAlmostEqual(row["weighted_recall"], 0.99)
        self.assertAlmostEqual(row["weighted_recall_lower_confidence_bound"], 0.99)
        self.assertFalse(row["calibration_safe"])

    def test_validation_reuses_calibration_thresholds(self) -> None:
        calibration = self._data([0.7, 0.3, 0.1], [1, 0, 0], [1, 0, 0], [0, 3])
        validation = self._data([0.9, 0.2, 0.05], [1, 0, 0], [1, 0, 0], [0, 3], "validation")
        thresholds = _thresholds(calibration.scores, 256)
        calibration_rows = score_curve(
            calibration, thresholds, self._resources(3), bootstrap_replicates=0,
            bootstrap_seed=7, target_weighted_recall=0.99, threshold_source="calibration",
        )
        safe = {float(row["threshold"]): row["calibration_safe"] for row in calibration_rows}
        validation_rows = score_curve(
            validation, thresholds, self._resources(3), bootstrap_replicates=0,
            bootstrap_seed=7, target_weighted_recall=0.99, calibration_safe=safe,
            threshold_source="calibration_frozen",
        )
        np.testing.assert_array_equal(
            np.asarray([row["threshold"] for row in validation_rows], dtype=np.float32), thresholds
        )
        self.assertTrue(all(row["threshold_source"] == "calibration_frozen" for row in validation_rows))

    def test_only_typed_non_test_sidecars_are_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = ScoreSidecarWriter(root / "calibration", split="calibration", threshold=0.5)
            writer.append_pose(1, np.asarray([0]), np.asarray([0.8]), np.asarray([1]), np.asarray([1.0]))
            writer.close()
            loaded = _valid_sidecar(root / "calibration", "calibration")
            self.assertEqual(loaded.split, "calibration")

            test_writer = ScoreSidecarWriter(root / "test", split="test", threshold=0.5)
            test_writer.append_pose(1, np.asarray([0]), np.asarray([0.8]), np.asarray([1]), np.asarray([1.0]))
            test_writer.close()
            with self.assertRaisesRegex(ValueError, "testRead=false"):
                _valid_sidecar(root / "test", "calibration")

    def test_cli_writes_metrics_and_simple_plot_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sidecar = root / "calibration"
            writer = ScoreSidecarWriter(sidecar, split="calibration", threshold=0.5, checkpoint="fixture.pt")
            writer.append_pose(
                1,
                np.asarray([0, 1], dtype=np.uint32),
                np.asarray([0.8, 0.1], dtype=np.float32),
                np.asarray([1, 0], dtype=np.uint8),
                np.asarray([1, 0], dtype=np.float32),
            )
            writer.close()
            runtime_meta = root / "runtimeVisibilityMeta.json"
            runtime_meta.write_text(json.dumps({"componentRecords": [
                {"componentGlobalId": 0, "globalGlbId": 0, "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}},
                {"componentGlobalId": 1, "globalGlbId": 1, "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            ]}), encoding="utf-8")
            (root / "a.glb").write_bytes(b"a" * 10)
            (root / "b.glb").write_bytes(b"b" * 20)
            glb_index = root / "glbIndex.json"
            glb_index.write_text(json.dumps({"entries": [
                {"globalId": 0, "path": "a.glb"}, {"globalId": 1, "path": "b.glb"},
            ]}), encoding="utf-8")
            output_csv = root / "curve.csv"
            output_json = root / "curve.json"
            main([
                "--calibration-sidecar", str(sidecar),
                "--runtime-meta", str(runtime_meta),
                "--glb-index", str(glb_index),
                "--glb-root", str(root),
                "--output-csv", str(output_csv),
                "--output-json", str(output_json),
                "--max-thresholds", "2",
                "--bootstrap-replicates", "0",
            ])
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], CURVE_SCHEMA)
            self.assertFalse(payload["testRead"])
            self.assertTrue(Path(payload["plotSource"]).is_file())
            with output_csv.open(newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle))
            self.assertIn("weighted_recall", row)
            self.assertIn("avg_pred_glb_bytes", row)


if __name__ == "__main__":
    unittest.main()
