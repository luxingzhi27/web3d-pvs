from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viewcell_representative_subposes import select_representative_subposes  # noqa: E402


class RepresentativeSubposeTest(unittest.TestCase):
    def test_center_plus_spatial_coverage_is_deterministic(self) -> None:
        positions = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
            [0.0, 0.0, 0.1],
        ], dtype=np.float32)
        selected = select_representative_subposes(np.zeros(3), positions, count=5)
        self.assertEqual(selected.size, 5)
        self.assertEqual(int(selected[0]), 0)
        self.assertEqual(len(set(selected.tolist())), 5)
        self.assertTrue({1, 2, 3, 4}.issubset(set(selected.tolist())))

    def test_count_is_capped_and_empty_is_supported(self) -> None:
        positions = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        self.assertEqual(select_representative_subposes(np.zeros(3), positions, count=5).tolist(), [0])
        self.assertEqual(select_representative_subposes(np.zeros(3), np.zeros((0, 3)), count=5).size, 0)


if __name__ == "__main__":
    unittest.main()
