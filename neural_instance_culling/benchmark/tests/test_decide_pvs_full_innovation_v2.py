from __future__ import annotations

from pathlib import Path
import sys
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import decide_pvs_full_innovation_v2 as route  # noqa: E402


class HistoricalFullInnovationV2RouteTests(unittest.TestCase):
    def test_historical_v2_self_test_stays_on_v2_schema(self) -> None:
        result = route.self_test()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["route"], "full_innovation_candidate")
        self.assertFalse(hasattr(route, "V3_FORMAL_VARIANTS"))

    def test_historical_v2_route_rejects_v3_summary(self) -> None:
        with self.assertRaises(ValueError):
            route.make_decision(
                {"schema": "pvs-bounded-relation-survival-moment-v3-validation-summary-v1"},
                Path("fixture.json"),
            )


if __name__ == "__main__":
    unittest.main()
