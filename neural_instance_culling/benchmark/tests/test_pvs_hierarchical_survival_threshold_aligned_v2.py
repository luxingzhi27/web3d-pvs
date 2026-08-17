from __future__ import annotations

import sys
import unittest
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_pvs_hierarchical_survival_threshold_aligned_v2 import (  # noqa: E402
    _aggregate_refine_groups,
    _frozen_from_refine,
    _mode_settings,
    _select_scan_configs,
    _select_refine_configuration,
    _summarize_stage_artifacts,
)


SEEDS = (20260801, 20260802)


def member(config_id: str, seed: int, hp: dict[str, float], *, safe: bool, useful: float, balanced: float, precision: float, predicted_bytes: float, predicted_count: float, variance_lcb: float = 0.995) -> dict:
    selected = {
        "aggregateWeightedRecall": 0.998 if safe else 0.985,
        "aggregateWeightedRecallLowerConfidenceBound": variance_lcb if safe else 0.980,
    }
    validation = {
        "agg_weighted_recall": 0.997 if safe else 0.982,
        "aggregateWeightedRecallLowerConfidenceBound": variance_lcb if safe else 0.978,
        "agg_balanced_accuracy": balanced,
        "agg_useful_cull": useful,
        "agg_precision": precision,
        "agg_accuracy": 0.90,
        "avg_pred_count": predicted_count,
        "avg_pred_glb_count": 12.0,
        "avg_pred_glb_bytes": predicted_bytes,
    }
    return {
        "returnCode": 0,
        "outputDir": f"/tmp/{config_id}_{seed}",
        "job": {
            "configId": config_id,
            "seed": seed,
            "member": f"{config_id}_seed{seed}",
            "hyperparameters": hp,
        },
        "calibration": {
            "status": "safe" if safe else "no_qualified_safety_workpoint",
            "selected": selected if safe else None,
            "diagnostic": selected,
            "validationAtCalibration": validation,
        },
    }


class PvsThresholdAlignedRunnerTests(unittest.TestCase):
    def test_groups_require_both_registered_seeds_and_keep_resources_and_variance(self) -> None:
        hp_a = {"lr": 1e-4, "weightDecay": 1e-5}
        hp_b = {"lr": 2e-4, "weightDecay": 1e-5}
        rows = [
            member("refine00", SEEDS[0], hp_a, safe=True, useful=0.60, balanced=0.80, precision=0.20, predicted_bytes=100.0, predicted_count=100.0),
            member("refine00", SEEDS[1], hp_a, safe=True, useful=0.70, balanced=0.82, precision=0.22, predicted_bytes=120.0, predicted_count=110.0),
            member("refine01", SEEDS[0], hp_b, safe=True, useful=0.90, balanced=0.90, precision=0.30, predicted_bytes=80.0, predicted_count=80.0),
        ]
        groups = _aggregate_refine_groups(rows)
        self.assertEqual(len(groups), 2)
        complete = next(group for group in groups if group["configId"] == "refine00")
        incomplete = next(group for group in groups if group["configId"] == "refine01")
        self.assertEqual(complete["seedCount"], 2)
        self.assertAlmostEqual(complete["mean"]["predictedGlbBytes"], 110.0)
        self.assertGreater(complete["variance"]["predictedGlbBytes"], 0.0)
        self.assertFalse(incomplete["completeTwoSeed"])

    def test_safe_pool_is_used_before_efficiency_ranking(self) -> None:
        hp_safe = {"lr": 1e-4, "weightDecay": 1e-5}
        hp_unsafe = {"lr": 3e-4, "weightDecay": 1e-6}
        rows = [
            member("safe", SEEDS[0], hp_safe, safe=True, useful=0.50, balanced=0.60, precision=0.10, predicted_bytes=500.0, predicted_count=500.0),
            member("safe", SEEDS[1], hp_safe, safe=True, useful=0.51, balanced=0.61, precision=0.11, predicted_bytes=510.0, predicted_count=510.0),
            member("unsafe", SEEDS[0], hp_unsafe, safe=False, useful=0.99, balanced=0.99, precision=0.99, predicted_bytes=1.0, predicted_count=1.0),
            member("unsafe", SEEDS[1], hp_unsafe, safe=False, useful=0.99, balanced=0.99, precision=0.99, predicted_bytes=1.0, predicted_count=1.0),
        ]
        selected = _select_refine_configuration(_aggregate_refine_groups(rows))
        self.assertEqual(selected["selectionPool"], "safe_candidate_pool")
        self.assertEqual(selected["configId"], "safe")
        self.assertTrue(selected["selection"]["safety"]["safeAcrossBothSeeds"])

    def test_safe_selection_is_pareto_filtered_then_deterministic(self) -> None:
        hp_a = {"lr": 1e-4, "weightDecay": 1e-5}
        hp_b = {"lr": 2e-4, "weightDecay": 1e-5}
        rows = [
            member("a", SEEDS[0], hp_a, safe=True, useful=0.70, balanced=0.80, precision=0.20, predicted_bytes=100.0, predicted_count=100.0),
            member("a", SEEDS[1], hp_a, safe=True, useful=0.70, balanced=0.80, precision=0.20, predicted_bytes=100.0, predicted_count=100.0),
            member("b", SEEDS[0], hp_b, safe=True, useful=0.70, balanced=0.80, precision=0.20, predicted_bytes=100.0, predicted_count=100.0),
            member("b", SEEDS[1], hp_b, safe=True, useful=0.70, balanced=0.80, precision=0.20, predicted_bytes=100.0, predicted_count=100.0),
        ]
        groups = _aggregate_refine_groups(rows)
        first = _select_refine_configuration(groups)
        second = _select_refine_configuration(list(reversed(groups)))
        self.assertEqual(first["configurationKey"], second["configurationKey"])
        self.assertEqual(first["configId"], "a")
        self.assertEqual(len(first["ranking"]), 2)

    def test_no_safe_pool_freezes_diagnostic_winner_without_claiming_safety(self) -> None:
        hp_a = {"lr": 1e-4}
        hp_b = {"lr": 2e-4}
        rows = [
            member("a", SEEDS[0], hp_a, safe=False, useful=0.4, balanced=0.7, precision=0.2, predicted_bytes=100.0, predicted_count=100.0),
            member("a", SEEDS[1], hp_a, safe=False, useful=0.4, balanced=0.7, precision=0.2, predicted_bytes=100.0, predicted_count=100.0),
            member("b", SEEDS[0], hp_b, safe=False, useful=0.4, balanced=0.7, precision=0.2, predicted_bytes=100.0, predicted_count=100.0),
            member("b", SEEDS[1], hp_b, safe=False, useful=0.4, balanced=0.7, precision=0.2, predicted_bytes=100.0, predicted_count=100.0),
        ]
        selected = _select_refine_configuration(_aggregate_refine_groups(rows))
        self.assertEqual(selected["selectionPool"], "diagnostic_candidate_pool")
        self.assertFalse(selected["selection"]["safety"]["safeAcrossBothSeeds"])
        self.assertEqual(selected["configId"], "a")

    def test_diagnostic_pool_uses_worst_seed_before_efficiency(self) -> None:
        hp_a = {"lr": 1e-4}
        hp_b = {"lr": 2e-4}
        rows = [
            member("a", SEEDS[0], hp_a, safe=False, useful=0.95, balanced=0.95, precision=0.95, predicted_bytes=1.0, predicted_count=1.0),
            member("a", SEEDS[1], hp_a, safe=False, useful=0.95, balanced=0.95, precision=0.95, predicted_bytes=1.0, predicted_count=1.0),
            member("b", SEEDS[0], hp_b, safe=False, useful=0.50, balanced=0.50, precision=0.50, predicted_bytes=100.0, predicted_count=100.0),
            member("b", SEEDS[1], hp_b, safe=False, useful=0.50, balanced=0.50, precision=0.50, predicted_bytes=100.0, predicted_count=100.0),
        ]
        # Give b a better aggregate recall only on one seed.  The diagnostic
        # rule must still use its worst seed, not an average that hides it.
        rows[2]["calibration"]["validationAtCalibration"]["agg_weighted_recall"] = 0.999
        rows[3]["calibration"]["validationAtCalibration"]["agg_weighted_recall"] = 0.970
        rows[2]["calibration"]["validationAtCalibration"]["aggregateWeightedRecallLowerConfidenceBound"] = 0.999
        rows[3]["calibration"]["validationAtCalibration"]["aggregateWeightedRecallLowerConfidenceBound"] = 0.960
        selected = _select_refine_configuration(_aggregate_refine_groups(rows))
        self.assertEqual(selected["selectionPool"], "diagnostic_candidate_pool")
        self.assertEqual(selected["configId"], "a")

    def test_frozen_from_refine_recomputes_old_placeholder(self) -> None:
        import json
        import tempfile

        hp = {"lr": 1e-4}
        rows = [
            member("a", SEEDS[0], hp, safe=True, useful=0.5, balanced=0.5, precision=0.5, predicted_bytes=100.0, predicted_count=100.0),
            member("a", SEEDS[1], hp, safe=True, useful=0.5, balanced=0.5, precision=0.5, predicted_bytes=100.0, predicted_count=100.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frozen_config.json").write_text(
                json.dumps({"schema": "pvs-full-innovation-frozen-config-v2", "selection": "pending_refine_summary_review", "hyperparameters": hp}),
                encoding="utf-8",
            )
            (root / "refine_member_summary.json").write_text(json.dumps({"members": rows}), encoding="utf-8")
            frozen = _frozen_from_refine(root)
        self.assertEqual(frozen["selectionPool"], "safe_candidate_pool")
        self.assertEqual(frozen["configId"], "a")
        self.assertIn("ranking", frozen)

    def test_reserved_inspection_is_read_only(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matrix_manifest.json").write_text(json.dumps({"mode": "refine24"}), encoding="utf-8")
            (root / "refine_member_summary.json").write_text(json.dumps({"members": []}), encoding="utf-8")
            result = _summarize_stage_artifacts(root, "evaluate")
            self.assertFalse(result["evaluationImplemented"])
            self.assertFalse(result["testRead"])

    def test_refine_mode_rejects_non_registered_seed_pair(self) -> None:
        with self.assertRaises(ValueError):
            _mode_settings("refine24", None, [20260801])
        with self.assertRaises(ValueError):
            _mode_settings("refine24", None, [20260801, 20260803])

    def test_scan_selection_uses_diagnostic_when_early_gate_is_unavailable(self) -> None:
        import json
        import tempfile

        rows = []
        for index in range(2):
            hp = {"lr": float(index + 1)}
            rows.append(
                {
                    "returnCode": 0,
                    "job": {"hyperparameters": hp},
                    "calibration": {
                        "selected": None,
                        "diagnostic": {
                            "aggregateWeightedRecall": 0.91 + index * 0.05,
                            "aggregateWeightedRecallLowerConfidenceBound": 0.90 + index * 0.05,
                            "agg_balanced_accuracy": 0.60 + index * 0.05,
                            "agg_useful_cull": 0.20 + index * 0.05,
                            "agg_precision": 0.10 + index * 0.05,
                            "avg_pred_count": 100.0 - index,
                            "avg_pred_glb_bytes": 1000.0 - index,
                        },
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scan_member_summary.json").write_text(json.dumps({"members": rows}), encoding="utf-8")
            selected = _select_scan_configs(root, limit=1)
        self.assertEqual(selected, [{"lr": 2.0}])


if __name__ == "__main__":
    unittest.main()
