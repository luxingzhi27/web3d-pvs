from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.dataset.build_stratified_calibration_split import (
    select_stratified_calibration,
)


class StratifiedCalibrationSplitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
