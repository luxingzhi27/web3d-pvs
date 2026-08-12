from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from run_ray_context_survival_owrb_supplement import (  # noqa: E402
    SUPPLEMENT_VARIANTS,
    _comparisons,
)


class RayContextSurvivalOWRBSupplementTest(unittest.TestCase):
    def test_registered_variants_change_one_supplement_factor(self) -> None:
        self.assertEqual(len(SUPPLEMENT_VARIANTS), 5)
        reference = SUPPLEMENT_VARIANTS["triangle_context32_direct9_monotone"]
        self.assertEqual(reference["evidence"], "triangle")
        self.assertEqual(reference["contextQueryDim"], 8)
        self.assertEqual(reference["rayMode"], "direct9")
        self.assertEqual(reference["survivalParameterization"], "monotone")
        self.assertEqual(SUPPLEMENT_VARIANTS["triangle_context64_direct9_monotone"]["contextQueryDim"], 16)
        self.assertEqual(SUPPLEMENT_VARIANTS["triangle_context32_fourier117_monotone"]["rayMode"], "fourier117")
        self.assertEqual(SUPPLEMENT_VARIANTS["triangle_context32_direct9_unconstrained28"]["survivalParameterization"], "unconstrained28")
        self.assertEqual(SUPPLEMENT_VARIANTS["aabb_context32_direct9_monotone"]["evidence"], "aabb")

    def test_comparison_contract_is_explicit_and_directional(self) -> None:
        comparisons = _comparisons(SUPPLEMENT_VARIANTS)
        self.assertEqual(
            [(item["name"], item["left"], item["right"]) for item in comparisons],
            [
                ("context64_minus_context32", "triangle_context64_direct9_monotone", "triangle_context32_direct9_monotone"),
                ("fourier117_minus_direct9", "triangle_context32_fourier117_monotone", "triangle_context32_direct9_monotone"),
                ("unconstrained28_minus_monotone", "triangle_context32_direct9_unconstrained28", "triangle_context32_direct9_monotone"),
                ("aabb_minus_triangle_relation", "aabb_context32_direct9_monotone", "triangle_context32_direct9_monotone"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
