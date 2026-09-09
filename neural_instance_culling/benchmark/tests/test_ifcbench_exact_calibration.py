from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from ifcbench_exact_calibration import (  # noqa: E402
    BOOTSTRAP_SCHEMA,
    SCORE_SIDECAR_SCHEMA,
    _fixed_bootstrap,
    _load_sidecar,
    _metrics_at_threshold,
    _select_exact_threshold,
)


class IfcbenchExactCalibrationTest(unittest.TestCase):
    def _sidecar(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        scores = np.asarray([0.9, 0.4, 0.8, 0.7, 0.1], dtype="<f4")
        labels = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0], dtype="<f4")
        weights = np.asarray([1.0, 0.0, 2.0, 1.0, 0.0], dtype="<f4")
        scores.tofile(root / "scores_f32.bin")
        labels.tofile(root / "labels_f32.bin")
        weights.tofile(root / "weights_f32.bin")
        np.asarray([0, 2, 3, 5], dtype="<u8").tofile(root / "pose_offsets_u64.bin")
        np.asarray([10, 11, 12], dtype="<i8").tofile(root / "pose_indices_i64.bin")
        (root / "sidecar_manifest.json").write_text(
            json.dumps(
                {
                    "schema": SCORE_SIDECAR_SCHEMA,
                    "split": "calibration",
                    "checkpoint": "/source/checkpoint.pt",
                    "checkpointSeed": 20260802,
                    "checkpointEpoch": 4,
                    "poseCount": 3,
                    "scoreCount": 5,
                    "files": {
                        "scores": "scores_f32.bin",
                        "labels": "labels_f32.bin",
                        "weights": "weights_f32.bin",
                        "poseOffsets": "pose_offsets_u64.bin",
                        "poseIndices": "pose_indices_i64.bin",
                    },
                    "testRead": False,
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_exact_selection_uses_float32_change_points_and_next_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = _load_sidecar(self._sidecar(Path(temporary)))
            bootstrap, valid_rows, gt_mass, metadata = _fixed_bootstrap(
                sidecar, Path(temporary) / "bootstrap.bin", replicates=8, seed=7
            )
            selection = _select_exact_threshold(sidecar, bootstrap, valid_rows, gt_mass)
            row = _metrics_at_threshold(
                sidecar, selection["threshold"], bootstrap, valid_rows, gt_mass
            )
        self.assertEqual(metadata["schema"], BOOTSTRAP_SCHEMA)
        self.assertEqual(selection["status"], "safe")
        self.assertEqual(selection["candidateCount"], 5)
        self.assertEqual(selection["allScoreChangePointCount"], 5)
        self.assertAlmostEqual(selection["threshold"], 0.7, places=6)
        self.assertAlmostEqual(selection["nextHigherThreshold"], 0.8, places=6)
        self.assertFalse(selection["nextHigherSafety"]["safe"])
        self.assertEqual(row["thresholdIsScoreChangePoint"], True)
        self.assertAlmostEqual(row["aggregateWeightedRecall"], 1.0)
        self.assertAlmostEqual(row["aggregateWeightedRecallLowerConfidenceBound"], 1.0)

    def test_fixed_bootstrap_file_is_reused_without_regeneration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sidecar = _load_sidecar(self._sidecar(Path(temporary) / "sidecar"))
            path = Path(temporary) / "bootstrap.bin"
            first, first_rows, _first_gt, _first_meta = _fixed_bootstrap(
                sidecar, path, replicates=8, seed=7
            )
            first_values = np.asarray(first).copy()
            second, second_rows, _second_gt, _second_meta = _fixed_bootstrap(
                sidecar, path, replicates=8, seed=7
            )
        np.testing.assert_array_equal(first_rows, second_rows)
        np.testing.assert_array_equal(first_values, np.asarray(second))


if __name__ == "__main__":
    unittest.main()
