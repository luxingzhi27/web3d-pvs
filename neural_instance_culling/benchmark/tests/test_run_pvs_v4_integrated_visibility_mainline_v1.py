from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.run_pvs_v4_integrated_visibility_mainline_v1 import (
    EVALUATION_SCHEMA,
    EXPECTED_SPLITS,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    SCAN_CONFIGS,
    SCAN_EPOCHS,
    SCAN_SEED,
    _evaluation_matches_member,
    _evaluation_path,
    _member_complete,
    _member_contract,
    _member_for_spec,
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

    def test_formal_matrix_is_three_seed_eighty_epoch_and_has_core_ablations(self) -> None:
        self.assertEqual(FORMAL_SEEDS, (20260801, 20260802, 20260803))
        self.assertEqual(FORMAL_EPOCHS, 80)
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

    def test_formal80_rejects_an_incomplete_scan_summary(self) -> None:
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
