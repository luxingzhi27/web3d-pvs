from __future__ import annotations

import sys
from pathlib import Path
import unittest


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from run_pvs_hierarchical_relation_survival_integrated_matrix import (  # noqa: E402
    _mode_config,
)


class MatrixRunnerModeTests(unittest.TestCase):
    def test_module_recheck_is_full_calibration_two_seed_sixteen_epoch(self) -> None:
        variants, seeds, epochs, steps, max_eval_poses, allow_unsafe = _mode_config(
            "module_recheck16", None, None
        )
        self.assertEqual(
            set(variants),
            {
                "R1_single_scale_relation",
                "R3_hierarchical_shuffled_source",
                "S1_learned_spectral_point",
                "S2_integrated_spectral",
                "L1_quality_risk",
            },
        )
        self.assertEqual(seeds, [20260801, 20260802])
        self.assertEqual(epochs, 16)
        self.assertEqual(steps, 100)
        self.assertEqual(max_eval_poses, 0)
        self.assertFalse(allow_unsafe)

    def test_module_recheck_accepts_explicit_subset_without_changing_protocol(self) -> None:
        variants, seeds, epochs, steps, max_eval_poses, allow_unsafe = _mode_config(
            "module_recheck16",
            ["S2_integrated_spectral", "L1_quality_risk"],
            [20260802],
        )
        self.assertEqual(set(variants), {"S2_integrated_spectral", "L1_quality_risk"})
        self.assertEqual(seeds, [20260802])
        self.assertEqual((epochs, steps, max_eval_poses, allow_unsafe), (16, 100, 0, False))


if __name__ == "__main__":
    unittest.main()
