from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from run_ifcbench_calibration_finetune import (  # noqa: E402
    CONFIRM_SEEDS,
    REFINE_CONFIGS,
    REFINE_CONFIRM_EPOCHS,
    REFINE_EXPERIMENT,
    REFINE_SCAN_EPOCHS,
    REFINE_SCAN_SEED,
    STEPS_PER_EPOCH,
    V1_BOUNDARY_REMOVED_CONFIG,
    _write_json,
    refine_confirmation_jobs,
    refine_confirmation_members,
    refine_confirmation_summary,
    refine_scan_jobs,
    refine_selection_payload,
    score_command,
    v1_confirm_checkpoint,
)


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def _validation_metrics(
    *,
    weighted_recall: float,
    lcb: float,
    useful_cull: float,
    predicted: float,
) -> dict[str, object]:
    return {
        "agg_precision": 0.40,
        "agg_recall": 0.995,
        "agg_weighted_recall": weighted_recall,
        "aggregateWeightedRecall": weighted_recall,
        "aggregateWeightedRecallLowerConfidenceBound": lcb,
        "agg_accuracy": 0.80,
        "agg_balanced_accuracy": 0.75,
        "agg_specificity": 0.50,
        "agg_useful_cull": useful_cull,
        "agg_bad_cull": 0.001,
        "avg_pred_count": predicted,
        "avg_gt_count": 10.0,
        "avg_candidate_count": 100.0,
        "candidate_reduction_ratio": 1.0 - predicted / 100.0,
        "positiveFraction": 0.1,
        "tp": 100,
        "fp": 10,
        "fn": 1,
        "tn": 889,
        "scoreDistribution": {"positive": {"q01": 0.2}},
    }


class IfcbenchCalibrationFinetuneRunnerTests(unittest.TestCase):
    def _write_scan_result(
        self,
        root: Path,
        config: str,
        *,
        weighted_recall: float,
        lcb: float,
        useful_cull: float,
        predicted: float,
        calibration_status: str = "safe",
    ) -> None:
        stage = root / "scan" / config
        calibration = {
            "schema": "pvs-ifcbench-v4-exact-calibration-v1",
            "checkpoint": f"/models/scan/{config}/last.pt",
            "status": calibration_status,
            "selection": {"threshold": 0.25},
            "selected": {"threshold": 0.25, "aggregateWeightedRecall": weighted_recall},
            "testRead": False,
        }
        validation = {
            "schema": "pvs-ifcbench-v4-frozen-threshold-validation-v1",
            "metrics": _validation_metrics(
                weighted_recall=weighted_recall,
                lcb=lcb,
                useful_cull=useful_cull,
                predicted=predicted,
            ),
            "testRead": False,
        }
        _write_json(stage / "exact_calibration.json", calibration)
        _write_json(stage / "validation_frozen.json", validation)

    def test_refine_matrix_and_scan_inherit_seed02_v1_confirm(self) -> None:
        self.assertEqual(len(REFINE_CONFIGS), 3)
        self.assertIn("boundary_half", " ".join(REFINE_CONFIGS))
        rvl_candidates = [values for values in REFINE_CONFIGS.values() if values["rvl"] == 0.35]
        self.assertEqual(len(rvl_candidates), 1)
        self.assertEqual(
            [values["lr"] for values in REFINE_CONFIGS.values()].count(1e-5), 1
        )
        self.assertEqual(REFINE_SCAN_EPOCHS, 2)
        self.assertEqual(STEPS_PER_EPOCH, 900)

        with tempfile.TemporaryDirectory() as temporary:
            data_root = Path(temporary) / "data"
            v1_root = Path(temporary) / "v1-models"
            jobs = refine_scan_jobs(data_root, Path(temporary) / "v2-models", v1_root)

        self.assertEqual(len(jobs), len(REFINE_CONFIGS))
        expected_init = v1_confirm_checkpoint(data_root, REFINE_SCAN_SEED, v1_root)
        for name, command in jobs:
            self.assertIn(name.removeprefix("refine_scan_"), REFINE_CONFIGS)
            self.assertEqual(_argument(command, "--epochs"), "2")
            self.assertEqual(_argument(command, "--steps-per-epoch"), "900")
            self.assertEqual(_argument(command, "--seed"), str(REFINE_SCAN_SEED))
            self.assertEqual(Path(_argument(command, "--init-checkpoint")), expected_init)
            self.assertEqual(_argument(command, "--experiment-name").split("_seed")[0], f"{REFINE_EXPERIMENT}_{name.removeprefix('refine_scan_')}")

        self.assertEqual(
            expected_init,
            v1_root
            / "confirm"
            / V1_BOUNDARY_REMOVED_CONFIG
            / f"seed{REFINE_SCAN_SEED}_e8"
            / "last.pt",
        )

    def test_refine_confirmation_reuses_each_v1_checkpoint_and_4x900(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            v1_root = root / "v1-models"
            v2_root = root / "v2-models"
            benchmark_root = root / "v2-benchmark"
            selected = next(iter(REFINE_CONFIGS))
            jobs = refine_confirmation_jobs(
                root / "data", v2_root, benchmark_root, selected, v1_root
            )
            members = refine_confirmation_members(
                root / "data", v2_root, benchmark_root, selected, v1_root
            )

        self.assertEqual([label for label, *_ in members], [f"seed{seed}" for seed in CONFIRM_SEEDS])
        self.assertEqual(len(jobs), 3)
        for (job_name, command), (label, member, _result, initial) in zip(jobs, members):
            self.assertIn(label, job_name)
            self.assertEqual(_argument(command, "--epochs"), str(REFINE_CONFIRM_EPOCHS))
            self.assertEqual(_argument(command, "--steps-per-epoch"), str(STEPS_PER_EPOCH))
            self.assertEqual(Path(_argument(command, "--init-checkpoint")), initial)
            self.assertTrue(str(member).endswith(f"seed{label.removeprefix('seed')}_e4"))

    def test_refine_selection_has_selected_config_and_lcb_relative_best_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configs = list(REFINE_CONFIGS)
            self._write_scan_result(
                root, configs[0], weighted_recall=0.997, lcb=0.994,
                useful_cull=0.55, predicted=45,
            )
            self._write_scan_result(
                root, configs[1], weighted_recall=0.996, lcb=0.992,
                useful_cull=0.65, predicted=35,
            )
            self._write_scan_result(
                root, configs[2], weighted_recall=0.998, lcb=0.995,
                useful_cull=0.60, predicted=40,
            )
            safe = refine_selection_payload(root)
            self.assertEqual(safe["selectedConfig"], configs[1])
            self.assertEqual(safe["selectionStatus"], "safe_member_selected")
            self.assertEqual(len(safe["rows"]), 3)
            self.assertFalse(safe["testRead"])

            for config, lcb in zip(configs, (0.980, 0.985, 0.975)):
                self._write_scan_result(
                    root, config, weighted_recall=0.995, lcb=lcb,
                    useful_cull=0.90, predicted=20,
                )
            fallback = refine_selection_payload(root)

        self.assertEqual(fallback["selectedConfig"], configs[1])
        self.assertEqual(
            fallback["selectionStatus"], "no_safe_member_selected_relative_best"
        )
        self.assertTrue(fallback["selectedConfig"])

    def test_refine_confirmation_summary_keeps_full_metrics_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            v1_root = root / "v1-models"
            v2_root = root / "v2-models"
            benchmark_root = root / "v2-benchmark"
            selected = next(iter(REFINE_CONFIGS))
            members = refine_confirmation_members(
                data_root, v2_root, benchmark_root, selected, v1_root
            )
            for seed_index, (seed, (_label, member, result_root, initial)) in enumerate(
                zip(CONFIRM_SEEDS, members)
            ):
                checkpoint = member / "last.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                checkpoint.touch()
                metrics = _validation_metrics(
                    weighted_recall=0.996 + seed_index * 0.0001,
                    lcb=0.993,
                    useful_cull=0.60,
                    predicted=40,
                )
                _write_json(
                    result_root / "exact_calibration.json",
                    {
                        "status": "safe",
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpointSeed": seed,
                        "checkpointEpoch": 4,
                        "selection": {"threshold": 0.24},
                        "selected": {"threshold": 0.24, "agg_precision": 0.4},
                        "testRead": False,
                    },
                )
                _write_json(
                    result_root / "validation_frozen.json",
                    {"metrics": metrics, "testRead": False},
                )
            summary = refine_confirmation_summary(
                data_root, v2_root, benchmark_root, selected, v1_root
            )

        self.assertEqual(summary["selectedConfig"], selected)
        self.assertEqual(len(summary["rows"]), 3)
        self.assertEqual(summary["training"]["epochs"], 4)
        self.assertEqual(summary["training"]["stepsPerEpoch"], 900)
        self.assertIn("agg_bad_cull", summary["rows"][0]["validation"])
        self.assertIn("scoreDistribution", summary["rows"][0]["validation"])
        self.assertIn("agg_bad_cull", summary["aggregate"])
        self.assertNotIn("scoreDistribution", summary["aggregate"])
        self.assertEqual(
            Path(summary["rows"][1]["initialCheckpoint"]),
            v1_confirm_checkpoint(data_root, CONFIRM_SEEDS[1], v1_root),
        )
        self.assertTrue(all(row["checkpoint"].endswith("/last.pt") for row in summary["rows"]))
        self.assertFalse(summary["testRead"])

    def test_score_command_rejects_test_split(self) -> None:
        with self.assertRaisesRegex(ValueError, "calibration and validation"):
            score_command(Path("/data"), Path("/checkpoint.pt"), "test", Path("/out"))


if __name__ == "__main__":
    unittest.main()
