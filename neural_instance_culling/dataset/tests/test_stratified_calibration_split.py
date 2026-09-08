from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.dataset.build_stratified_calibration_split import (
    POSE_DTYPE,
    _visible_weight_sums,
    build_manifest,
    select_stratified_calibration,
)


class StratifiedCalibrationSplitTest(unittest.TestCase):
    def test_visible_weight_sums_support_empty_rows_at_end(self) -> None:
        offsets = np.asarray([0, 2, 2, 3, 3], dtype="<u8")
        weights = np.asarray([1.5, 2.5, 4.0], dtype="<f4")
        np.testing.assert_allclose(
            _visible_weight_sums(offsets, weights),
            np.asarray([4.0, 0.0, 4.0, 0.0]),
        )

    def test_selection_is_deterministic_and_exact(self) -> None:
        rows = 1000
        labels = np.stack(
            [
                np.arange(rows) % 5,
                5 + np.arange(rows) % 8,
                13 + np.arange(rows) % 4,
                17 + np.arange(rows) % 6,
                23 + np.arange(rows) % 7,
                30 + np.arange(rows) % 3,
            ],
            axis=1,
        ).astype(np.int64)
        first = select_stratified_calibration(labels, 100, seed=17, swap_iterations=5000)
        second = select_stratified_calibration(labels, 100, seed=17, swap_iterations=5000)
        self.assertEqual(first.size, 100)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(np.unique(first).size, 100)

    def test_each_marginal_stays_close_to_population_fraction(self) -> None:
        rows = 1200
        rng = np.random.default_rng(42)
        widths = (6, 12, 4, 8, 8, 8)
        offsets = np.cumsum((0, *widths[:-1]))
        labels = np.stack(
            [offset + rng.integers(0, width, size=rows) for offset, width in zip(offsets, widths)],
            axis=1,
        )
        selected = select_stratified_calibration(labels, 120, seed=9, swap_iterations=20000)
        for column in range(labels.shape[1]):
            population = np.bincount(labels[:, column])
            calibration = np.bincount(labels[selected, column], minlength=population.size)
            expected = population * 0.10
            active = population > 0
            self.assertLessEqual(float(np.max(np.abs(calibration[active] - expected[active]))), 1.5)

    def test_invalid_count_is_rejected(self) -> None:
        labels = np.zeros((5, 2), dtype=np.int64)
        with self.assertRaises(ValueError):
            select_stratified_calibration(labels, 0, seed=1, swap_iterations=0)
        with self.assertRaises(ValueError):
            select_stratified_calibration(labels, 5, seed=1, swap_iterations=0)

    def test_manifest_reads_historical_split_from_poses_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset"
            output = root / "split"
            dataset.mkdir()
            pose_count = 20
            poses = np.zeros((pose_count,), dtype=POSE_DTYPE)
            poses["camera_forward"][:, 2] = -1.0
            poses["split"][:16] = 0
            poses["split"][16:18] = 1
            poses["split"][18:] = 2
            poses.tofile(dataset / "poses.bin")
            np.arange(pose_count + 1, dtype="<u8").tofile(dataset / "candidate_offsets.bin")
            np.arange(pose_count, dtype="<u4").tofile(dataset / "candidate_ids.bin")
            np.arange(pose_count + 1, dtype="<u8").tofile(dataset / "visible_offsets.bin")
            np.arange(pose_count, dtype="<u4").tofile(dataset / "visible_ids.bin")
            np.ones((pose_count,), dtype="<f4").tofile(dataset / "visible_weights.bin")
            (dataset / "dataset_meta.json").write_text(
                json.dumps({"poseCount": pose_count}), encoding="utf-8"
            )
            manifest = build_manifest(
                argparse.Namespace(
                    dataset_dir=dataset,
                    source_split_ids=None,
                    output_dir=output,
                    calibration_fraction=0.10,
                    yaw_bins=2,
                    pitch_bins=2,
                    numeric_bins=2,
                    swap_iterations=20,
                    seed=7,
                )
            )
            self.assertEqual(manifest["sourceHistoricalSplitIds"], "poses.bin:split")
            self.assertEqual(
                manifest["poseCounts"],
                {"train": 14, "validation": 2, "calibration": 2, "test": 2, "guard": 0},
            )


if __name__ == "__main__":
    unittest.main()
