from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_pvs_cross_pose_operating_exposure_representation_v1 as runner  # noqa: E402


class CrossPoseOperatingExposureRunnerTests(unittest.TestCase):
    def _args(self, mode: str = "scan8"):
        return runner.parse_args([mode, "--gpu-ids", "0", "1", "2", "3"])

    def test_registered_scan_and_factor_matrix(self) -> None:
        scan = runner._make_jobs("scan8")
        factor = runner._make_jobs("factor16", runner._scan_config("S2"))
        self.assertEqual([job["variant"] for job in scan], list(runner.SCAN_VARIANTS))
        self.assertEqual([job["variant"] for job in factor], list(runner.FACTOR_VARIANTS))
        self.assertEqual({int(job["seed"]) for job in scan}, {runner.SCAN_SEED})
        self.assertEqual({int(job["seed"]) for job in factor}, {runner.FACTOR_SEED})
        self.assertEqual({int(job["epochs"]) for job in scan}, {8})
        self.assertEqual({int(job["epochs"]) for job in factor}, {16})

    def test_cross_pose_and_exposure_command_contract(self) -> None:
        args = self._args("factor16")
        jobs = runner._make_jobs("factor16", runner._scan_config("S2"))
        mode = runner._mode_config("factor16")
        command_a = runner.build_train_command(args, jobs[0], Path("/tmp/member-a"), mode)
        command_d = runner.build_train_command(args, jobs[3], Path("/tmp/member-d"), mode)
        joined_a = " ".join(command_a)
        joined_d = " ".join(command_d)
        self.assertIn("--variant pose_balanced_frontier_without_exposure", joined_a)
        self.assertIn("--variant cross_pose_operating_with_exposure", joined_d)
        self.assertNotIn("--variant A_pose_frontier_no_exposure", joined_a)
        self.assertNotIn("--variant D_cross_pose_exposure", joined_d)
        self.assertIn("--loss-variant pose_balanced_frontier", joined_a)
        self.assertIn("--loss-variant cross_pose_operating", joined_d)
        self.assertIn("--exposure-supervision-hidden-dim 0", joined_a)
        self.assertIn("--exposure-supervision-hidden-dim 16", joined_d)
        self.assertIn("--exposure-supervision-loss-weight 0.05", joined_d)
        self.assertIn("--cross-pose-weighted-recall-target 0.995", joined_d)
        self.assertIn("--cross-pose-hard-negative-count-cap 512", joined_d)
        self.assertIn("--instance-calibration-warmup-fraction 0.25", joined_d)
        self.assertIn("--instance-calibration-ramp-fraction 0.5", joined_d)
        self.assertNotIn("--initial-checkpoint", command_d)

        smoke_job = runner._make_jobs("smoke")[0]
        smoke_command = runner.build_train_command(
            self._args("smoke"),
            smoke_job,
            Path("/tmp/member-smoke"),
            runner._mode_config("smoke"),
        )
        joined_smoke = " ".join(smoke_command)
        self.assertIn("--instance-calibration-warmup-fraction 0.0", joined_smoke)
        self.assertIn("--instance-calibration-ramp-fraction 1.0", joined_smoke)

    def test_scan_selection_uses_safe_pool_then_relative_diagnostic_order(self) -> None:
        def row(config_id: str, *, safe: bool, lcb: float, balanced: float, precision: float) -> dict:
            return {
                "configId": config_id,
                "eligibleSafe": safe,
                "aggregate": {
                    "weightedRecallLowerConfidenceBound": lcb,
                    "weightedRecall": lcb + 0.001,
                    "balancedAccuracy": balanced,
                    "precision": precision,
                    "usefulCull": 0.6,
                    "accuracy": 0.7,
                    "avgPredCount": 1000.0,
                },
            }

        safe_rows = [
            row("S1", safe=True, lcb=0.991, balanced=0.71, precision=0.11),
            row("S2", safe=False, lcb=0.999, balanced=0.95, precision=0.99),
            row("S3", safe=True, lcb=0.992, balanced=0.72, precision=0.10),
            row("S4", safe=False, lcb=0.998, balanced=0.94, precision=0.98),
        ]
        selected = runner.select_scan_configuration(safe_rows)
        self.assertEqual(selected["selectionPool"], "safe_candidate_pool")
        self.assertEqual(selected["selected"]["configId"], "S3")

        unsafe_rows = [
            row("S1", safe=False, lcb=0.989, balanced=0.71, precision=0.11),
            row("S2", safe=False, lcb=0.991, balanced=0.60, precision=0.10),
            row("S3", safe=False, lcb=0.990, balanced=0.80, precision=0.20),
            row("S4", safe=False, lcb=0.991, balanced=0.70, precision=0.30),
        ]
        selected = runner.select_scan_configuration(unsafe_rows)
        self.assertEqual(selected["selectionPool"], "diagnostic_candidate_pool")
        self.assertEqual(selected["selected"]["configId"], "S4")
        self.assertTrue(selected["factorStageMustRun"])

    def test_queue_uses_all_registered_gpus_and_keeps_results_separate(self) -> None:
        assignments: dict[str, int] = {}
        lock = threading.Lock()

        def fake_run(command, *, gpu, stdout_path, stderr_path):
            del command
            stdout_path.parent.mkdir(parents=True, exist_ok=True)
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("", encoding="utf-8")
            time.sleep(0.001)
            with lock:
                assignments[stdout_path.name] = int(gpu)
            return {
                "command": [],
                "gpu": int(gpu),
                "returnCode": 0,
                "elapsedSeconds": 0.001,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "testRead": False,
            }

        jobs = [
            (f"member-{index}", ["python", "-c", "pass"], Path(f"/tmp/member-{index}"))
            for index in range(8)
        ]
        with tempfile.TemporaryDirectory() as directory:
            original = runner._run_logged
            runner._run_logged = fake_run
            try:
                results = runner._run_queue(
                    jobs,
                    gpu_ids=[0, 1, 2, 3],
                    log_root=Path(directory) / "logs",
                )
            finally:
                runner._run_logged = original
        self.assertEqual(len(results), 8)
        self.assertEqual(set(assignments.values()), {0, 1, 2, 3})
        self.assertTrue(all(result["testRead"] is False for result in results))

    def test_runtime_contract_rejects_asset_or_query_growth(self) -> None:
        evaluation_path = Path("validation.json")
        valid = {
            "runtimeFeatureDim": runner.RUNTIME_FEATURE_DIM,
            "runtimeFeatureBytes": runner.HKUST_RUNTIME_FEATURE_BYTES,
            "inferenceInputDim": runner.RUNTIME_QUERY_INPUT_DIM,
        }
        runner._validate_runtime_contract(valid, evaluation_path)

        for field, value in (
            ("runtimeFeatureDim", runner.RUNTIME_FEATURE_DIM + 1),
            ("runtimeFeatureBytes", runner.HKUST_RUNTIME_FEATURE_BYTES + 2),
            ("inferenceInputDim", runner.RUNTIME_QUERY_INPUT_DIM + 1),
        ):
            invalid = dict(valid)
            invalid[field] = value
            with self.assertRaisesRegex(ValueError, "runtime"):
                runner._validate_runtime_contract(invalid, evaluation_path)


if __name__ == "__main__":
    unittest.main()
