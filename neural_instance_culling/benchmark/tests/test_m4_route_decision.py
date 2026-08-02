import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from decide_m4_route import make_decision


def summary(delta: float, lower: float, bad_delta: float = 0.001, bad_upper: float = 0.0015) -> dict:
    variants = [
        "aabb_ray",
        "geometry_ray",
        "geometry_context_ray",
        "geometry_context_proxy_ray_no_inhibition",
        "full",
    ]
    return {
        "schema": "neuralstreamweb3d-formal-m4-matrix-summary-v1",
        "split": "validation",
        "status": "validation_summary_only; formal test not read",
        "referenceVariant": "geometry_context_ray",
        "variants": variants,
        "seeds": [20260801, 20260802, 20260803],
        "pairedComparisonsVariantMinusReference": {
            "full": {
                "useful_cull": {"mean_delta": delta, "ci95": [lower, delta + 0.01]},
                "bad_cull": {"mean_delta": bad_delta, "ci95": [bad_delta - 0.0005, bad_upper]},
            },
        },
    }


class M4RouteDecisionTests(unittest.TestCase):
    def write_summary(self, payload: dict) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "summary.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_selects_route_a_only_for_registered_gain(self) -> None:
        path = self.write_summary(summary(0.021, 0.001))
        decision = make_decision(json.loads(path.read_text()), path, 0.02, 0.0)
        self.assertEqual(decision["route"], "route_a_directional_proxy")
        self.assertTrue(decision["fullMinusReference"]["usefulCullGate"])

    def test_selects_route_b_when_confidence_interval_crosses_zero(self) -> None:
        path = self.write_summary(summary(0.03, -0.001))
        decision = make_decision(json.loads(path.read_text()), path, 0.02, 0.0)
        self.assertEqual(decision["route"], "route_b_system")
        self.assertFalse(decision["fullMinusReference"]["usefulCullGate"])

    def test_rejects_test_derived_summary(self) -> None:
        payload = summary(0.03, 0.001)
        payload["status"] = "test_summary"
        path = self.write_summary(payload)
        with self.assertRaises(ValueError):
            make_decision(json.loads(path.read_text()), path, 0.02, 0.0)

    def test_rejects_useful_gain_when_bad_cull_gate_fails(self) -> None:
        path = self.write_summary(summary(0.021, 0.001, bad_delta=0.0021, bad_upper=0.0021))
        decision = make_decision(json.loads(path.read_text()), path, 0.02, 0.0)
        self.assertEqual(decision["route"], "route_b_system")
        self.assertFalse(decision["fullMinusReference"]["badCullGate"])

    def test_rejects_summary_without_bad_cull_comparison(self) -> None:
        payload = summary(0.021, 0.001)
        del payload["pairedComparisonsVariantMinusReference"]["full"]["bad_cull"]
        path = self.write_summary(payload)
        with self.assertRaises(ValueError):
            make_decision(json.loads(path.read_text()), path, 0.02, 0.0)


if __name__ == "__main__":
    unittest.main()
