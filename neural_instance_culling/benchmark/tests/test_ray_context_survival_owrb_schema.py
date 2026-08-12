import unittest
import sys
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from validate_ray_context_survival_owrb import self_test
from run_ray_context_survival_owrb_matrix import FORMAL_VARIANTS, _variant_specs


class RayContextSurvivalOWRBSchemaTest(unittest.TestCase):
    def test_rejects_test_read_summary(self) -> None:
        self.assertEqual(self_test()["status"], "passed")

    def test_formal_matrix_has_complete_three_factor_design(self) -> None:
        variants = _variant_specs("formal")
        self.assertEqual(set(variants), set(FORMAL_VARIANTS))
        self.assertEqual(len(variants), 8)
        self.assertEqual(
            {(spec["contextMode"], bool(spec.get("disableSurvival", False)), spec["lossMode"])
             for spec in variants.values()},
            {
                (context, survival_off, loss)
                for context in ("pooled", "directional")
                for survival_off in (False, True)
                for loss in ("rvl_strong_v2", "safety_constraint")
            },
        )


if __name__ == "__main__":
    unittest.main()
