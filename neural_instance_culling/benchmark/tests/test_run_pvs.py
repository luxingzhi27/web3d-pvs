from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neural_instance_culling.benchmark.run_pvs import (
    CORE_VARIANTS,
    EXPECTED_SPLITS,
    FORMAL_BOOTSTRAP_REPLICATES,
    FORMAL_EPOCHS,
    FORMAL_SEEDS,
    FORMAL_STEPS_PER_EPOCH,
    MAINLINE_CONFIG,
    PAPER_VARIANTS,
    _member_dir,
    _specs,
    build_evaluate_command,
    build_train_command,
    preflight,
)


DATA_ROOT = Path("/mnt/sda/rhyang/slm")


def _argument(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


class PaperMainlineRunnerTests(unittest.TestCase):
    def test_preflight_fixes_data_and_training_contract(self) -> None:
        contract = preflight(DATA_ROOT)
        self.assertEqual(contract["splitCounts"], EXPECTED_SPLITS)
        self.assertIsNone(contract["initialCheckpoint"])
        self.assertEqual(contract["lossVariant"], "pose_balanced_rvl_contrastive")
        self.assertEqual(contract["paperVariants"], list(PAPER_VARIANTS))
        self.assertFalse(contract["testRead"])

    def test_full_command_contains_only_current_visibility_objective(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = build_train_command(
                DATA_ROOT,
                Path(directory) / "member",
                MAINLINE_CONFIG,
                "full",
                seed=FORMAL_SEEDS[0],
            )
        self.assertNotIn("--initial-checkpoint", command)
        self.assertNotIn("--refinement-scope", command)
        self.assertNotIn("--integrated-contrastive-mix", command)
        self.assertEqual(_argument(command, "--loss-variant"), "pose_balanced_rvl_contrastive")
        self.assertEqual(_argument(command, "--epochs"), str(FORMAL_EPOCHS))
        self.assertEqual(
            _argument(command, "--steps-per-epoch"), str(FORMAL_STEPS_PER_EPOCH)
        )
        self.assertEqual(
            _argument(command, "--calibration-bootstrap-replicates"),
            str(FORMAL_BOOTSTRAP_REPLICATES),
        )

    def test_matrix_is_full_plus_six_ablations_by_three_seeds(self) -> None:
        self.assertEqual(len(CORE_VARIANTS), 6)
        self.assertEqual(len(_specs(tuple(PAPER_VARIANTS))), 21)
        self.assertEqual(set(FORMAL_SEEDS), {20260801, 20260802, 20260803})

    def test_non_survival_controls_remove_relation_supervision(self) -> None:
        for variant, representation in (("generic28", "generic28"), ("no_survival", "none")):
            command = build_train_command(
                DATA_ROOT,
                Path("/tmp") / variant,
                MAINLINE_CONFIG,
                variant,
                seed=FORMAL_SEEDS[0],
            )
            self.assertEqual(_argument(command, "--occlusion-representation"), representation)
            self.assertEqual(_argument(command, "--relation-source"), "none")
            self.assertEqual(_argument(command, "--survival-loss-weight"), "0.0")

    def test_smoke_has_one_member_and_one_update(self) -> None:
        specs = _specs(("full",), smoke=True)
        self.assertEqual(len(specs), 1)
        member = _member_dir(Path("/tmp/models"), specs[0])
        command = build_train_command(
            DATA_ROOT,
            member,
            MAINLINE_CONFIG,
            "full",
            seed=specs[0][2],
            epochs=specs[0][3],
            smoke=True,
        )
        self.assertEqual(_argument(command, "--epochs"), "1")
        self.assertEqual(_argument(command, "--steps-per-epoch"), "1")

    def test_unsafe_member_evaluation_is_explicitly_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            member = Path(directory)
            (member / "best_diagnostic.pt").touch()
            (member / "model_meta.json").write_text("{}", encoding="utf-8")
            (member / "calibration_ready_summary.json").write_text(
                json.dumps({"status": "no_safe_workpoint", "bestSafe": None}),
                encoding="utf-8",
            )
            command = build_evaluate_command(
                DATA_ROOT, member, member / "evaluation.json", seed=FORMAL_SEEDS[0]
            )
        self.assertIn("--allow-unsafe-diagnostic", command)


if __name__ == "__main__":
    unittest.main()
