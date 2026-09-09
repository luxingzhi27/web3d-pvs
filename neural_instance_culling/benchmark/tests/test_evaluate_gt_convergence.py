from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_gt_convergence.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gt_convergence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GtConvergenceTest(unittest.TestCase):
    def test_farthest_order_is_nested_and_starts_at_center(self) -> None:
        points = np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0.1, 0, 0]])
        order = MODULE.farthest_point_order(points, np.zeros(3))
        self.assertEqual(order[0], 0)
        self.assertEqual(set(order.tolist()), {0, 1, 2, 3})
        self.assertIn(order[1], (1, 2))

    def test_nested_union_uses_max_component_weight(self) -> None:
        visibility = {
            10: (np.asarray([1, 2]), np.asarray([0.2, 0.4])),
            11: (np.asarray([2, 3]), np.asarray([0.8, 0.1])),
        }
        ids, weights = MODULE.nested_union(np.asarray([10, 11]), visibility, 2)
        self.assertEqual(ids, {1, 2, 3})
        self.assertAlmostEqual(weights[2], 0.8)


if __name__ == "__main__":
    unittest.main()
