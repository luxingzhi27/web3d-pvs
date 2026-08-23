from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.run_pvs_v4_integrated_visibility_mainline_v1 import (
    EVALUATION_SCHEMA,
    EXPECTED_SPLITS,
    FORMAL_ABLATIONS,
    FORMAL_ABLATION_STAGE,
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_EPOCHS,
    FORMAL_FULL_STAGE,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
    FORMAL_VARIANTS,
    SCAN_CONFIGS,
    SCAN_EPOCHS,
    SCAN_SEED,
    UPDATE_BUDGET_CHECK_CONFIGS,
    UPDATE_BUDGET_CHECK_EPOCHS,
    UPDATE_BUDGET_CHECK_STAGE,
    UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH,
    _evaluation_matches_member,
    _evaluation_path,
    _member_complete,
    _member_contract,
    _member_for_spec,
    _require_formal_bootstrap_protocol,
    _specs,
    _validate_scan_summary,
    build_train_command,
    preflight,
)


DATA_ROOT = Path("/mnt/sda/rhyang/slm")


def _argument(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


class IntegratedVisibilityMainlineRunnerTests(unittest.TestCase):
    def test_preflight_fixes_main_split_and_from_scratch_contract(self) -> None:
        contract = preflight(DATA_ROOT)

        self.assertEqual(contract["splitCounts"], EXPECTED_SPLITS)
        self.assertIsNone(contract["initialCheckpoint"])
        self.assertEqual(contract["relationSource"], "bounded_hierarchical")
        self.assertEqual(contract["spectralMode"], "moment_envelope")
        self.assertEqual(contract["instanceCalibrationMode"], "residual")
        self.assertEqual(contract["lossVariant"], "pose_balanced_rvl_contrastive")
        self.assertFalse(contract["testRead"])

    def test_full_command_contains_only_registered_visibility_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_train_command(
                DATA_ROOT,
                Path(directory) / "member",
                SCAN_CONFIGS[2],
                "full",
                seed=FORMAL_SEEDS[0],
                epochs=FORMAL_EPOCHS,
            )

        self.assertNotIn("--initial-checkpoint", command)
        self.assertEqual(_argument(command, "--variant"), "full_integrated_visibility_mainline")
        self.assertEqual(_argument(command, "--relation-source"), "bounded_hierarchical")
        self.assertEqual(_argument(command, "--spectral-mode"), "moment_envelope")
        self.assertEqual(_argument(command, "--loss-variant"), "pose_balanced_rvl_contrastive")
        self.assertEqual(_argument(command, "--utility-loss-weight"), "0.0")
        self.assertEqual(_argument(command, "--download-loss-weight"), "0.0")
        self.assertEqual(_argument(command, "--glb-resource-weight"), "0.0")
        self.assertEqual(_argument(command, "--rvl-count-weight"), "0.0")
        self.assertEqual(_argument(command, "--rvl-rank-weight"), "0.0")

    def test_formal_matrix_is_three_seed_forty_epoch_and_has_core_ablations(self) -> None:
        self.assertEqual(FORMAL_SEEDS, (20260801, 20260802, 20260803))
        self.assertEqual(FORMAL_EPOCHS, 40)
        self.assertEqual(FORMAL_STEPS_PER_EPOCH, 900)
        self.assertEqual(
            set(FORMAL_VARIANTS),
            {
                "full",
                "without_hierarchical_relation_prior",
                "without_viewcell_moment_envelope",
                "without_rvl_recall_guard",
                "without_contrastive_separation",
            },
        )

    def test_update_budget_check_retrains_s01_and_s02_for_twelve_by_three_hundred(self) -> None:
        self.assertEqual(
            [config["name"] for config in UPDATE_BUDGET_CHECK_CONFIGS],
            [SCAN_CONFIGS[1]["name"], SCAN_CONFIGS[2]["name"]],
        )
        self.assertEqual(UPDATE_BUDGET_CHECK_EPOCHS, 12)
        self.assertEqual(UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH, 300)
        spec = (
            UPDATE_BUDGET_CHECK_STAGE,
            "full",
            UPDATE_BUDGET_CHECK_CONFIGS[0],
            SCAN_SEED,
            UPDATE_BUDGET_CHECK_EPOCHS,
        )
        contract = _member_contract(Path("/tmp/member"), spec)
        command = build_train_command(
            DATA_ROOT,
            Path("/tmp/member"),
            spec[2],
            spec[1],
            seed=spec[3],
            epochs=spec[4],
            steps_per_epoch=UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH,
            eval_every=2,
        )

        self.assertEqual(contract["steps_per_epoch"], 300)
        self.assertEqual(_argument(command, "--epochs"), "12")
        self.assertEqual(_argument(command, "--steps-per-epoch"), "300")
        self.assertEqual(_argument(command, "--eval-every"), "2")
        self.assertEqual(_argument(command, "--snapshot-every"), "2")

    def test_s02_formal_longtrain_is_full_model_three_seed_forty_epoch(self) -> None:
        specs = _specs(
            FORMAL_FULL_STAGE,
            ("full",),
            (SCAN_CONFIGS[2],),
            FORMAL_SEEDS,
            FORMAL_EPOCHS,
        )

        self.assertEqual(len(specs), 3)
        self.assertEqual({spec[1] for spec in specs}, {"full"})
        self.assertEqual({spec[2]["name"] for spec in specs}, {SCAN_CONFIGS[2]["name"]})
        self.assertEqual({spec[3] for spec in specs}, set(FORMAL_SEEDS))
        self.assertEqual({spec[4] for spec in specs}, {40})
        contract = _member_contract(Path("/tmp/member"), specs[0])
        command = build_train_command(
            DATA_ROOT,
            Path("/tmp/member"),
            specs[0][2],
            specs[0][1],
            seed=specs[0][3],
            epochs=specs[0][4],
            steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
        )
        self.assertEqual(contract["steps_per_epoch"], 900)
        self.assertEqual(_argument(command, "--steps-per-epoch"), "900")
        self.assertEqual(
            _argument(command, "--calibration-bootstrap-replicates"), "10000"
        )

    def test_s02_formal_ablation_matrix_has_four_variants_and_three_seeds(self) -> None:
        specs = _specs(
            FORMAL_ABLATION_STAGE,
            FORMAL_ABLATIONS,
            (SCAN_CONFIGS[2],),
            FORMAL_SEEDS,
            FORMAL_EPOCHS,
        )

        self.assertEqual(len(specs), 12)
        self.assertNotIn("full", FORMAL_ABLATIONS)
        self.assertEqual(set(FORMAL_ABLATIONS), set(FORMAL_VARIANTS) - {"full"})
        self.assertEqual({spec[3] for spec in specs}, set(FORMAL_SEEDS))
        self.assertEqual({spec[4] for spec in specs}, {40})
        for spec in specs:
            contract = _member_contract(Path("/tmp/member"), spec)
            self.assertEqual(contract["steps_per_epoch"], 900)

    def test_formal_runner_rejects_historical_2000_replicate_calibration(self) -> None:
        spec = (
            FORMAL_FULL_STAGE,
            "full",
            SCAN_CONFIGS[2],
            FORMAL_SEEDS[0],
            FORMAL_EPOCHS,
        )
        with tempfile.TemporaryDirectory() as directory:
            member = Path(directory)
            summary = {
                "calibration": {"bootstrapReplicates": 2_000},
                "testRead": False,
            }
            (member / "calibration_ready_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "reaudit"):
                _require_formal_bootstrap_protocol(member, spec)

            summary["calibration"]["bootstrapReplicates"] = (
                FORMAL_BOOTSTRAP_REPLICATES
            )
            (member / "calibration_ready_summary.json").write_text(
                json.dumps(summary), encoding="utf-8"
            )
            _require_formal_bootstrap_protocol(member, spec)

    def test_loss_ablations_change_only_the_registered_loss_component(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            no_guard = build_train_command(
                DATA_ROOT,
                root / "no_guard",
                SCAN_CONFIGS[2],
                "without_rvl_recall_guard",
                seed=FORMAL_SEEDS[0],
                epochs=FORMAL_EPOCHS,
            )
            no_contrastive = build_train_command(
                DATA_ROOT,
                root / "no_contrastive",
                SCAN_CONFIGS[2],
                "without_contrastive_separation",
                seed=FORMAL_SEEDS[0],
                epochs=FORMAL_EPOCHS,
            )

        self.assertEqual(
            _argument(no_guard, "--integrated-rvl-recall-guard-weight"),
            "0.0",
        )
        self.assertEqual(
            _argument(no_contrastive, "--integrated-contrastive-mix"),
            "0.0",
        )
        for command in (no_guard, no_contrastive):
            self.assertEqual(_argument(command, "--relation-source"), "bounded_hierarchical")
            self.assertEqual(_argument(command, "--spectral-mode"), "moment_envelope")

    def test_member_completion_rejects_a_mismatched_run_contract(self) -> None:
        spec = ("scan", "full", SCAN_CONFIGS[0], SCAN_SEED, SCAN_EPOCHS)
        with tempfile.TemporaryDirectory() as directory:
            member = _member_for_spec(Path(directory), spec)
            member.mkdir(parents=True)
            for name in ("last.pt", "model_meta.json", "calibration_ready_summary.json"):
                (member / name).write_text("{}", encoding="utf-8")
            (member / "train_history.json").write_text(
                json.dumps([{"epoch": epoch} for epoch in range(1, SCAN_EPOCHS + 1)]),
                encoding="utf-8",
            )
            arguments = _member_contract(member, spec)
            (member / "run_manifest.json").write_text(
                json.dumps({"arguments": arguments, "testRead": False}),
                encoding="utf-8",
            )
            self.assertTrue(_member_complete(member, spec))

            arguments["seed"] = SCAN_SEED + 1
            (member / "run_manifest.json").write_text(
                json.dumps({"arguments": arguments, "testRead": False}),
                encoding="utf-8",
            )
            self.assertFalse(_member_complete(member, spec))

    def test_validation_cache_must_reference_the_selected_checkpoint(self) -> None:
        spec = ("scan", "full", SCAN_CONFIGS[0], SCAN_SEED, SCAN_EPOCHS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = _member_for_spec(root / "models", spec)
            member.mkdir(parents=True)
            checkpoint = member / "best_safe.pt"
            checkpoint.write_bytes(b"checkpoint")
            (member / "model_meta.json").write_text("{}", encoding="utf-8")
            (member / "calibration_ready_summary.json").write_text(
                json.dumps(
                    {
                        "status": "safe",
                        "bestSafe": {"epoch": 8, "threshold": 0.2},
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            output = _evaluation_path(root / "benchmarks", member)
            output.parent.mkdir(parents=True)
            payload = {
                "schema": EVALUATION_SCHEMA,
                "split": "validation",
                "testRead": False,
                "checkpoint": str(checkpoint.resolve()),
                "modelMeta": str((member / "model_meta.json").resolve()),
                "calibrationSummary": str(
                    (member / "calibration_ready_summary.json").resolve()
                ),
                "checkpointSeed": SCAN_SEED,
                "epoch": 8,
                "threshold": 0.2,
            }
            output.write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(_evaluation_matches_member(output, member, spec))

            payload["checkpoint"] = str((member / "stale.pt").resolve())
            output.write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_evaluation_matches_member(output, member, spec))

    def test_scan_validation_rejects_an_incomplete_summary(self) -> None:
        specs = _specs(
            "scan", ("full",), SCAN_CONFIGS, (SCAN_SEED,), SCAN_EPOCHS
        )
        summary = {
            "schema": "pvs-v4-integrated-visibility-mainline-summary-v1",
            "selectionSplit": "validation",
            "selected": {},
            "rows": [],
            "testRead": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "complete registered scan matrix"):
                _validate_scan_summary(summary, root / "models", root / "benchmarks", specs)


if __name__ == "__main__":
    unittest.main()
