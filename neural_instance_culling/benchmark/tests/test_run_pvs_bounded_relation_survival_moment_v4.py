from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_pvs_bounded_relation_survival_moment_v4 as runner  # noqa: E402


class V4RunnerTests(unittest.TestCase):
    def test_resume_preflight_preserves_shape_fields(self) -> None:
        value = runner._resume_preflight_view(
            {"geometry": {"path": "geo.bin", "shape": [18831, 96], "dtype": "float16", "geometrySha256": "legacy"}}
        )
        self.assertEqual(value["geometry"]["shape"], [18831, 96])
        self.assertNotIn("geometrySha256", value["geometry"])

    def test_formal_postprocess_uses_only_v3_named_entrypoints(self) -> None:
        self.assertEqual(runner.VALIDATOR_CLI.name, "validate_pvs_bounded_relation_survival_moment_v4.py")
        self.assertEqual(runner.EVALUATOR_CLI.name, "evaluate_pvs_bounded_relation_survival_moment_v4.py")
        self.assertEqual(runner.SUMMARIZER_CLI.name, "summarize_pvs_bounded_relation_survival_moment_v4.py")
        self.assertEqual(runner.EXPORTER_CLI.name, "export_bounded_relation_survival_moment.py")
        self.assertEqual(runner.ROUTE_CLI.name, "decide_pvs_bounded_relation_survival_moment_v4.py")
        self.assertNotIn("hierarchical_relation_survival_integrated", str(runner.VALIDATOR_CLI))
        self.assertNotIn("_v2", str(runner.EVALUATOR_CLI))
        self.assertNotIn("_v2", str(runner.SUMMARIZER_CLI))
        self.assertNotIn("_v2", str(runner.ROUTE_CLI))

    @staticmethod
    def _observation_batch_from_command(command: list[str]) -> int:
        index = command.index("--observation-batch-size")
        return int(command[index + 1])

    def test_smoke_manifest_and_command_use_full_observation_batch(self) -> None:
        args = runner.parse_args(["smoke", "--observation-batch-size", "8192", "--gpu-ids", "0"])
        config = runner._mode_config("smoke")
        jobs = runner._make_jobs(args, config)
        command = runner._train_command(args, jobs[0], Path("/tmp/v3-smoke-member"), config)
        self.assertEqual(config["observationBatchSize"], 32768)
        self.assertEqual(self._observation_batch_from_command(command), 32768)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = runner._build_manifest(
                args,
                config,
                {"candidateHashes": {}, "poseHashes": {}, "gtHashes": {}, "relationHashes": {}, "testRead": False},
                jobs,
                root / "out",
                root / "logs",
                root / "benchmark",
            )
        self.assertEqual(manifest["observationBatchSize"], 32768)
        self.assertEqual(
            self._observation_batch_from_command(manifest["plannedMembers"][0]["command"]),
            32768,
        )

    def test_non_smoke_modes_keep_cli_observation_batch_default(self) -> None:
        args = runner.parse_args(["scan8", "--gpu-ids", "0"])
        for mode in ("scan8", "formal80"):
            config = runner._mode_config(mode)
            jobs = runner._make_jobs(args, config)
            command = runner._train_command(args, jobs[0], Path(f"/tmp/{mode}-member"), config)
            self.assertEqual(self._observation_batch_from_command(command), 8192, mode)
            self.assertNotEqual(config.get("observationBatchSize"), 32768, mode)

    def test_formal_default_geometry_uses_registered_w042_path(self) -> None:
        self.assertTrue(str(runner.DEFAULT_GEO).endswith(
            "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin"
        ))

    def test_resume_preflight_ignores_only_obsolete_geometry_expectation(self) -> None:
        current = {
            "geometry": {"path": "geo.bin", "shape": [18831, 96], "dtype": "float16"},
            "splitPoseCounts": {"train": 2772, "validation": 213},
        }
        historical = json.loads(json.dumps(current))
        historical["geometry"]["expectedSha256"] = "legacy-metadata-only"
        self.assertEqual(
            runner._resume_preflight_view(current),
            runner._resume_preflight_view(historical),
        )
        changed = json.loads(json.dumps(historical))
        changed["geometry"]["shape"] = [18830, 96]
        self.assertNotEqual(
            runner._resume_preflight_view(current),
            runner._resume_preflight_view(changed),
        )

    def test_registered_member_counts_seed_epoch_and_persistent_gpu_queue(self) -> None:
        formal = runner._mode_config("formal80")
        self.assertEqual(formal["variants"], list(runner.FORMAL_VARIANTS))
        self.assertEqual(formal["seeds"], list(runner.FORMAL_SEEDS))
        self.assertEqual(formal["epochs"], 80)
        jobs = runner._make_jobs(
            type("Args", (), {"relation_dir": "native"})(),
            formal,
            {"configId": "frozen", "hyperparameters": dict(runner.DEFAULT_HYPERPARAMETERS)},
        )
        self.assertEqual(len(jobs), 15)
        self.assertEqual({int(job["seed"]) for job in jobs}, set(runner.FORMAL_SEEDS))
        self.assertTrue(all("_e80" in str(job["member"]) for job in jobs))

        scan = runner._mode_config("scan8")
        scan_jobs = runner._make_jobs(type("Args", (), {})(), scan)
        self.assertEqual(len(scan_jobs), 8)
        self.assertEqual({str(job["configId"]) for job in scan_jobs}, {f"s{index:02d}" for index in range(8)})
        residual_ablation = next(job for job in jobs if job["variant"] == "without_instance_calibration_residual")
        self.assertEqual(
            residual_ablation["variantSpec"]["instanceCalibrationMode"],
            "disabled",
        )
        self.assertEqual(scan["epochs"], 8)
        self.assertEqual(scan["stepsPerEpoch"], 50)
        self.assertEqual(scan["snapshotEvery"], 4)
        scan_command = runner._train_command(
            runner.parse_args(["scan8", "--gpu-ids", "0"]),
            scan_jobs[0],
            Path("/tmp/scan-member"),
            scan,
        )
        self.assertEqual(scan_command[scan_command.index("--snapshot-every") + 1], "4")

        assignments: dict[str, int] = {}
        lock = threading.Lock()

        def fake_run(_args, job, _output, _logs, _config, gpu):
            time.sleep(0.001 * (int(job["seed"]) % 3))
            with lock:
                assignments[str(job["member"])] = int(gpu)
            return {"member": str(job["member"]), "seed": int(job["seed"]), "variant": str(job["variant"]), "gpu": int(gpu), "returnCode": 0, "testRead": False}

        queue_jobs = [{"member": f"job{index}", "variant": "full", "seed": index} for index in range(9)]
        with tempfile.TemporaryDirectory() as directory:
            results = runner._dynamic_queue(
                type("Args", (), {})(),
                queue_jobs,
                Path(directory) / "out",
                Path(directory) / "logs",
                scan,
                [0, 1, 2, 3],
                run_one=fake_run,
            )
        self.assertEqual(len(results), len(queue_jobs))
        self.assertEqual({assignments[f"job{index}"] for index in range(4)}, {0, 1, 2, 3})
        self.assertTrue(all(value in {0, 1, 2, 3} for value in assignments.values()))
        self.assertFalse(any(result.get("testRead") is not False for result in results))

    def test_orphaned_completed_members_resume_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen_path = root / "frozen.json"
            frozen_payload = {
                "schema": runner.FROZEN_CONFIG_SCHEMA_V4,
                "configId": "s01",
                "hyperparameters": dict(runner.DEFAULT_HYPERPARAMETERS),
                "groups": [
                    {
                        "configId": str(config["configId"]),
                        "completeScanMember": True,
                        "metricContractComplete": True,
                    }
                    for config in runner.SCAN_CONFIGS
                ],
                "selected": {"configId": "s01"},
                "testRead": False,
            }
            frozen_path.write_text(json.dumps(frozen_payload), encoding="utf-8")
            args = runner.parse_args(
                [
                    "formal80",
                    "--frozen-config", str(frozen_path),
                    "--output-root", str(root / "model"),
                    "--log-root", str(root / "logs"),
                    "--benchmark-output-root", str(root / "benchmark"),
                    "--gpu-ids", "0", "1", "2", "3",
                    "--validator-cli", "",
                ]
            )
            config = runner._mode_config("formal80")
            jobs = runner._make_jobs(args, config, {"configId": "s01", "hyperparameters": dict(runner.DEFAULT_HYPERPARAMETERS)})
            preflight = {
                "candidateHashes": {"train": "train", "calibration": "calibration", "validation": "validation"},
                "relationHashes": {"native": {"artifactDigest": "native"}},
                "testRead": False,
            }
            output_root = Path(args.output_root)
            output_root.mkdir(parents=True)
            manifest = runner._build_manifest(args, config, preflight, jobs, output_root, Path(args.log_root), Path(args.benchmark_output_root))
            runner._write_json(output_root / "run_manifest.json", manifest)
            mtimes: dict[str, int] = {}
            for job in jobs:
                member_dir = output_root / str(job["member"])
                member_dir.mkdir()
                for name in ("last.pt", "model_meta.json", "calibration_ready_summary.json", "train_history.json"):
                    path = member_dir / name
                    if name == "train_history.json":
                        payload = [{"epoch": 80}]
                    elif name == "model_meta.json":
                        payload = {
                            "protocol": {
                                "variant": job["variant"],
                                "seed": job["seed"],
                                "experiment": f"fixture_{job['member']}",
                            }
                        }
                    else:
                        payload = {}
                    path.write_text(json.dumps(payload), encoding="utf-8")
                mtimes[str(member_dir)] = (member_dir / "last.pt").stat().st_mtime_ns
            runner._write_json(output_root / "pipeline_status.json", {"status": "training_complete_evaluation_incomplete", "testRead": False})

            postprocess_calls = []

            def failed_postprocess(*call_args):
                postprocess_calls.append(call_args)
                return {"status": "training_complete_evaluation_incomplete", "testRead": False}

            with mock.patch.object(runner, "_preflight", return_value=preflight), mock.patch.object(
                runner, "_dynamic_queue", side_effect=AssertionError("completed members must not be retrained")
            ), mock.patch.object(runner, "_formal_postprocess", side_effect=failed_postprocess):
                result = runner.run_experiment(args)

            self.assertEqual(result["status"], "training_complete_evaluation_incomplete")
            self.assertEqual(len(postprocess_calls), 1)
            self.assertTrue(all((Path(member) / "last.pt").stat().st_mtime_ns == mtime for member, mtime in mtimes.items()))
            self.assertFalse(any(result.get("testRead") is not False for result in postprocess_calls[0][2]))

    def test_resume_does_not_treat_incomplete_history_as_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            member = Path(directory) / "member"
            member.mkdir()
            for name in ("last.pt", "model_meta.json", "calibration_ready_summary.json"):
                (member / name).write_text("{}", encoding="utf-8")
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": 7}]),
                encoding="utf-8",
            )
            self.assertFalse(runner._training_artifacts_complete(member, expected_epochs=8))
            self.assertIsNone(runner._completed_member_dir(Path(directory), "member", expected_epochs=8))

    def test_manifest_and_planned_commands_are_explicitly_test_free(self) -> None:
        args = runner.parse_args(["scan8", "--gpu-ids", "0", "1", "2", "3"])
        config = runner._mode_config("scan8")
        jobs = runner._make_jobs(args, config)
        preflight = {"candidateHashes": {}, "poseHashes": {}, "gtHashes": {}, "relationHashes": {}, "testRead": False}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = runner._build_manifest(args, config, preflight, jobs, root / "out", root / "logs", root / "benchmark")
        self.assertFalse(manifest["testRead"])
        self.assertEqual(manifest["plannedMemberCount"], 8)
        self.assertTrue(all(member["status"] == "planned" for member in manifest["plannedMembers"]))
        self.assertTrue(all(member["testRead"] is False for member in manifest["plannedMembers"]))
        self.assertTrue(all("--seed" in member["command"] for member in manifest["plannedMembers"]))
        self.assertTrue(all("test" not in member["command"] for member in manifest["plannedMembers"]))

    def test_manifest_carries_fixed_geometry_schema_without_file_gate(self) -> None:
        args = runner.parse_args(["formal80", "--dry-run", "--gpu-ids", "0", "1", "2", "3"])
        config = runner._mode_config("formal80")
        jobs = runner._make_jobs(args, config, {"configId": "frozen", "hyperparameters": dict(runner.DEFAULT_HYPERPARAMETERS)})
        preflight = {
            "geometry": {
                "path": str(runner.DEFAULT_GEO),
                "sha256": "recorded-by-the-runner",
            },
            "candidateHashes": {}, "poseHashes": {}, "gtHashes": {}, "relationHashes": {}, "testRead": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = runner._build_manifest(args, config, preflight, jobs, root / "out", root / "logs", root / "benchmark")
        self.assertEqual(manifest["fixedGeometry"]["path"], str(runner.DEFAULT_GEO))
        self.assertNotIn("sha256", manifest["fixedGeometry"])
        self.assertNotIn("expectedSha256", manifest["preflight"]["geometry"])

    def test_external_validator_command_carries_preflight_identity(self) -> None:
        args = runner.parse_args(["formal80", "--validator-cli", "/tmp/validator.py"])
        command = runner._validator_command(
            args,
            Path("/tmp/member/best.pt"),
            expected_variant="full",
            expected_seed=20260801,
        )
        assert command is not None
        self.assertIn("--expected-variant", command)
        self.assertIn("--expected-seed", command)
        self.assertNotIn("--expected-relation-digest", command)
        self.assertNotIn("--expected-geometry-sha256", command)

    def test_scan_safe_pool_prefers_classification_and_resource_quality_over_higher_recall(self) -> None:
        configs = (
            {"configId": "quality", "hyperparameters": {"learningRate": 1e-4}},
            {"configId": "overpredict", "hyperparameters": {"learningRate": 2e-4}},
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "SCAN_CONFIGS", configs
        ):
            root = Path(directory)
            training_results = []
            values = {
                "quality": {
                    "aggregateWeightedRecall": 0.992,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                    "eval_pose_count": 213,
                    "agg_balanced_accuracy": 0.81,
                    "agg_useful_cull": 0.42,
                    "agg_precision": 0.63,
                    "avg_pred_glb_bytes": 100.0,
                    "avg_pred_count": 200.0,
                },
                "overpredict": {
                    "aggregateWeightedRecall": 0.999,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.998,
                    "eval_pose_count": 213,
                    "agg_balanced_accuracy": 0.61,
                    "agg_useful_cull": 0.20,
                    "agg_precision": 0.31,
                    "avg_pred_glb_bytes": 180.0,
                    "avg_pred_count": 360.0,
                },
            }
            for config in configs:
                for seed in (runner.SCAN_SEED,):
                    member = root / f"{config['configId']}_seed{seed}"
                    member.mkdir()
                    (member / "calibration_ready_summary.json").write_text(
                        json.dumps({
                            "status": "safe",
                            "validationAtFrozenThreshold": values[config["configId"]],
                            "bestSafe": {"selection": {"eval_pose_count": 168}},
                        }),
                        encoding="utf-8",
                    )
                    training_results.append({
                        "configId": config["configId"],
                        "seed": seed,
                        "returnCode": 0,
                        "outputDir": str(member),
                    })
            result = runner._select_scan_configuration(root, training_results)
        self.assertEqual(result["selectionPool"], "safe_candidate_pool")
        self.assertEqual(result["configId"], "quality")
        self.assertIn("balanced accuracy", result["selectionRule"])

    def test_scan_rejects_incomplete_validation_metric_contract(self) -> None:
        configs = ({"configId": "incomplete", "hyperparameters": {"learningRate": 1e-4}},)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            runner, "SCAN_CONFIGS", configs
        ):
            root = Path(directory)
            training_results = []
            for seed in (runner.SCAN_SEED,):
                member = root / f"incomplete_seed{seed}"
                member.mkdir()
                (member / "calibration_ready_summary.json").write_text(
                    json.dumps(
                        {
                            "status": "safe",
                            "validationAtFrozenThreshold": {
                                "aggregateWeightedRecall": 0.995,
                                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                                "agg_balanced_accuracy": 0.80,
                                "agg_useful_cull": 0.40,
                                # precision is intentionally absent.
                                "avg_pred_glb_bytes": 100.0,
                                "avg_pred_count": 200.0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                training_results.append(
                    {
                        "configId": "incomplete",
                        "seed": seed,
                        "returnCode": 0,
                        "outputDir": str(member),
                    }
                )
            with self.assertRaisesRegex(RuntimeError, "classification/resource metrics"):
                runner._select_scan_configuration(root, training_results)

    def test_runtime_bundle_rejects_fixed_table_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            table = bundle / "instance_runtime_features_fp16.bin"
            table.write_bytes(b"checkpoint-specific-table")
            meta = {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4",
                "testRead": False,
                "fixedTable": {"file": "wrong.bin", "dtype": "float16", "shape": [1, 124]},
                "runtimeFeatureSource": {
                    "source": "checkpoint.geometryFeatures_plus_survivalCoefficients",
                    "geometry": {"shape": [1, 96], "dtype": "float16"},
                },
                "threshold": 0.25,
            }
            (bundle / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
            member = {
                "checkpointInfo": {
                    "geometry": {"shape": [1, 96], "dtype": "float16"},
                    "threshold": 0.25,
                    "safeWorkpoint": True,
                }
            }
            with self.assertRaisesRegex(ValueError, "fixed-table file"):
                runner._validate_runtime_bundle(member, bundle)

    def test_checkpoint_selection_follows_calibration_role_and_never_uses_last(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "safe"
            safe.mkdir()
            payload = b"safe-checkpoint"
            (safe / "best_safe.pt").write_bytes(payload)
            (safe / "best.pt").write_bytes(payload)
            (safe / "last.pt").write_bytes(b"last-checkpoint")
            (safe / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {
                        "schema": runner.CALIBRATION_SUMMARY_SCHEMA_V4,
                        "status": "safe",
                        "safeCheckpoint": str(safe / "best_safe.pt"),
                        "diagnosticCheckpoint": None,
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(runner._selected_checkpoint(safe), safe / "best_safe.pt")

            diagnostic = root / "diagnostic"
            diagnostic.mkdir()
            (diagnostic / "best_diagnostic.pt").write_bytes(b"diagnostic-checkpoint")
            (diagnostic / "last.pt").write_bytes(b"last-checkpoint")
            (diagnostic / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {
                        "schema": runner.CALIBRATION_SUMMARY_SCHEMA_V4,
                        "status": "no_qualified_safety_workpoint",
                        "safeCheckpoint": None,
                        "diagnosticCheckpoint": str(diagnostic / "best_diagnostic.pt"),
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                runner._selected_checkpoint(diagnostic), diagnostic / "best_diagnostic.pt"
            )

            last_only = root / "last_only"
            last_only.mkdir()
            (last_only / "last.pt").write_bytes(b"last-checkpoint")
            (last_only / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {
                        "schema": runner.CALIBRATION_SUMMARY_SCHEMA_V4,
                        "status": "no_qualified_safety_workpoint",
                        "safeCheckpoint": None,
                        "diagnosticCheckpoint": str(last_only / "best_diagnostic.pt"),
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileNotFoundError, "best_diagnostic.pt"):
                runner._selected_checkpoint(last_only)

    def test_safe_checkpoint_selection_does_not_depend_on_alias_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            member = Path(directory)
            (member / "best_safe.pt").write_bytes(b"safe-checkpoint")
            (member / "best.pt").write_bytes(b"different-checkpoint")
            (member / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {
                        "schema": runner.CALIBRATION_SUMMARY_SCHEMA_V4,
                        "status": "safe",
                        "safeCheckpoint": str(member / "best_safe.pt"),
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(runner._selected_checkpoint(member), member / "best_safe.pt")

    def test_formal_postprocess_requires_image_and_resource_clis(self) -> None:
        args = runner.parse_args(["formal80", "--image-cli", "", "--resource-cli", "/tmp/resource.py"])
        with self.assertRaisesRegex(RuntimeError, "--image-cli"):
            runner._require_formal_evaluation_clis(args)

        args = runner.parse_args(["formal80", "--image-cli", "/tmp/image.py", "--resource-cli", ""])
        with self.assertRaisesRegex(RuntimeError, "--resource-cli"):
            runner._require_formal_evaluation_clis(args)

    def test_formal_image_command_is_single_member_and_uses_its_bundle(self) -> None:
        args = runner.parse_args(["formal80", "--image-cli", "/tmp/image.py"])
        member = {
            "variant": "full",
            "seed": 20260801,
            "checkpointInfo": {
                "checkpoint": "/tmp/member/best.pt",
                "calibration": "/tmp/member/calibration_ready_summary.json",
                "threshold": 0.25,
            },
            "runtimeBundle": {
                "runtimeFeatures": "/tmp/bundle/instance_runtime_features_fp16.bin",
            },
        }
        command = runner._image_command(args, member, Path("/tmp/image/full_seed20260801"))
        assert command is not None
        self.assertEqual(command.count("--model-spec"), 1)
        spec = command[command.index("--model-spec") + 1]
        self.assertIn("|bounded_relation_survival_moment_v4|", spec)
        self.assertIn("/tmp/bundle/instance_runtime_features_fp16.bin", spec)
        self.assertEqual(command[command.index("--threshold") + 1], "0.25")

    def test_formal_report_contains_member_table_and_route_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "benchmark" / "formal_validation"
            evaluation.mkdir(parents=True)
            summary = evaluation / "validation_summary.json"
            route = evaluation / "route_decision.json"
            route_report = evaluation / "route_report.md"
            summary.write_text(
                json.dumps(
                    {
                        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-validation-summary-v1",
                        "split": "validation",
                        "testRead": False,
                        "bootstrap": {"replicates": 10000, "paired": True},
                        "members": {
                            "full:20260801": {
                                "calibrationSafeWorkpoint": True,
                                "aggregate": {
                                    "aggregateWeightedRecall": 0.995,
                                    "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                                    "precision": 0.2,
                                    "balancedAccuracy": 0.7,
                                    "usefulCull": 0.3,
                                    "badCull": 0.001,
                                    "avgPredCount": 10,
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            route.write_text(
                json.dumps(
                    {
                        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-route-decision-v1",
                        "route": "baseline_system",
                        "routeReason": "fixture",
                    }
                ),
                encoding="utf-8",
            )
            route_report.write_text("# Route fixture\n", encoding="utf-8")
            with mock.patch.object(runner, "ROOT", root), mock.patch.object(
                runner, "DEFAULT_FORMAL_VALIDATION", evaluation
            ):
                report = runner._write_formal_validation_report(summary, route, route_report, evaluation)
            content = report.read_text(encoding="utf-8")
            self.assertIn("full", content)
            self.assertIn("validation_summary.json", content)
            self.assertIn("Route fixture", content)


if __name__ == "__main__":
    unittest.main()
