from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from finalize_ifcbench_mainline import (  # noqa: E402
    CONFIRMATION_SCHEMA,
    SEEDS,
    FinalMember,
    _assert_new_targets,
    _run_job,
    build_export_command,
    freeze_decision,
    load_confirmation_summary,
    prepare_plan,
    select_runtime_member,
    validation_safety,
)


def _confirmation_row(seed: int, *, recall: float = 0.995, lower: float = 0.992) -> dict:
    return {
        "seed": seed,
        "checkpoint": f"/tmp/confirm/seed{seed}/last.pt",
        "validation": {
            "agg_weighted_recall": recall,
            "aggregateWeightedRecallLowerConfidenceBound": lower,
            "agg_useful_cull": 0.5,
        },
        "testRead": False,
    }


class FinalizeIfcbenchMainlineTests(unittest.TestCase):
    def _summary(self, *, unsafe_seed: int | None = None) -> dict:
        rows = []
        for seed in SEEDS:
            rows.append(
                _confirmation_row(
                    seed,
                    lower=0.99 if seed == unsafe_seed else 0.992,
                )
            )
        return {
            "schema": CONFIRMATION_SCHEMA,
            "selectedConfig": "boundary_removed_rvl030_boundary000_margin050_temp025_lr2e-5",
            "rows": rows,
            "testRead": False,
        }

    def test_promotion_requires_all_three_strict_validation_gates(self) -> None:
        promoted = freeze_decision(self._summary())
        self.assertEqual(promoted["selectedFamily"], "finetune")
        self.assertTrue(promoted["promotedFineTune"])

        fallback = freeze_decision(self._summary(unsafe_seed=SEEDS[1]))
        self.assertEqual(fallback["selectedFamily"], "full_v4")
        self.assertFalse(fallback["promotedFineTune"])

    def test_decision_rejects_a_non_registered_confirmation_seed(self) -> None:
        payload = self._summary()
        payload["rows"][0]["seed"] = 123
        with self.assertRaisesRegex(ValueError, "invalid or duplicate seed"):
            freeze_decision(payload)

    def test_exact_boundary_is_not_safe(self) -> None:
        self.assertFalse(
            validation_safety(
                {
                    "aggregateWeightedRecall": 0.9900001,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.99,
                }
            )["safe"]
        )

    def test_confirmation_loader_rejects_missing_seed_and_test_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "confirmation_summary.json"
            payload = self._summary()
            payload["rows"] = payload["rows"][:-1]
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 3 rows"):
                load_confirmation_summary(path)

            payload = self._summary()
            payload["testRead"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "testRead=false"):
                load_confirmation_summary(path)

    def test_runtime_selection_uses_useful_cull_before_lcb(self) -> None:
        members = []
        for seed, useful, lower in (
            (SEEDS[0], 0.70, 0.991),
            (SEEDS[1], 0.60, 0.999),
            (SEEDS[2], 0.50, 0.995),
        ):
            members.append(
                FinalMember(
                    family="full_v4",
                    method_name="full_v4",
                    seed=seed,
                    checkpoint=Path(f"/tmp/{seed}.pt"),
                    calibration=Path(f"/tmp/{seed}.json"),
                    calibration_kind="member",
                    model_meta=Path(f"/tmp/{seed}_meta.json"),
                    validation={
                        "aggregateWeightedRecall": 0.995,
                        "aggregateWeightedRecallLowerConfidenceBound": lower,
                        "usefulCull": useful,
                    },
                )
            )
        selected, pool = select_runtime_member(members)
        self.assertEqual(selected.seed, SEEDS[0])
        self.assertEqual([member.seed for member in pool], list(SEEDS))

    def test_finetune_export_has_explicit_external_calibration(self) -> None:
        member = FinalMember(
            family="finetune",
            method_name="finetune_boundary_removed",
            seed=SEEDS[0],
            checkpoint=Path("/tmp/last.pt"),
            calibration=Path("/tmp/exact_calibration.json"),
            calibration_kind="exact",
            model_meta=Path("/tmp/model_meta.json"),
            validation={
                "aggregateWeightedRecall": 0.995,
                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                "usefulCull": 0.5,
            },
        )
        command = build_export_command(member, Path("/tmp/slm"), output_dir=Path("/tmp/runtime"))
        self.assertIn("--calibration", command)
        self.assertEqual(command[command.index("--calibration") + 1], "/tmp/exact_calibration.json")

    def test_full_export_uses_checkpoint_owned_calibration(self) -> None:
        member = FinalMember(
            family="full_v4",
            method_name="full_v4",
            seed=SEEDS[0],
            checkpoint=Path("/tmp/best_safe.pt"),
            calibration=Path("/tmp/calibration_ready_summary.json"),
            calibration_kind="member",
            model_meta=Path("/tmp/model_meta.json"),
            validation={
                "aggregateWeightedRecall": 0.995,
                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                "usefulCull": 0.5,
            },
        )
        command = build_export_command(member, Path("/tmp/slm"), output_dir=Path("/tmp/runtime"))
        self.assertNotIn("--calibration", command)

    def test_promoted_plan_uses_each_confirmation_exact_calibration(self) -> None:
        config = "boundary_removed_rvl030_boundary000_margin050_temp025_lr2e-5"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_root = root / "model"
            benchmark_root = root / "benchmark"
            for seed in SEEDS:
                member = model_root / "confirm" / config / f"seed{seed}_e8"
                member.mkdir(parents=True)
                checkpoint = member / "last.pt"
                checkpoint.write_bytes(b"checkpoint")
                (member / "model_meta.json").write_text("{}", encoding="utf-8")
                result_root = benchmark_root / "confirm" / config / f"seed{seed}"
                result_root.mkdir(parents=True)
                (result_root / "exact_calibration.json").write_text(
                    json.dumps(
                        {
                            "schema": "pvs-ifcbench-v4-exact-calibration-v1",
                            "split": "calibration",
                            "checkpoint": str(checkpoint),
                            "status": "safe",
                            "selection": {"threshold": 0.7},
                            "selected": {
                                "threshold": 0.7,
                                "aggregateWeightedRecall": 0.995,
                                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                            },
                            "testRead": False,
                        }
                    ),
                    encoding="utf-8",
                )
            summary_path = root / "confirmation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        **self._summary(),
                        "rows": [
                            {
                                **row,
                                "checkpoint": str(
                                    model_root
                                    / "confirm"
                                    / config
                                    / f"seed{row['seed']}_e8"
                                    / "last.pt"
                                ),
                            }
                            for row in self._summary()["rows"]
                        ],
                    }
                ),
                encoding="utf-8",
            )
            contract = {
                "schema": "pvs-ifcbench-finalize-preflight-v1",
                "scene": "ifcbench_fantasy_metropolis",
                "splitCounts": {
                    "train": 19647,
                    "validation": 2712,
                    "calibration": 2183,
                    "test": 2710,
                    "guard": 0,
                },
                "testRead": False,
            }
            with patch(
                "finalize_ifcbench_mainline.preflight_inputs",
                return_value=contract,
            ):
                plan = prepare_plan(
                    data_root=root,
                    confirmation_summary_path=summary_path,
                    full_model_root=root / "unused-full-model",
                    full_benchmark_root=root / "unused-full-benchmark",
                    finetune_model_root=model_root,
                    finetune_benchmark_root=benchmark_root,
                    paper_results_root=root / "paper-results",
                    runtime_output_dir=root / "runtime-output",
                    logs_dir=root / "logs",
                    device="cpu",
                )
        self.assertEqual(plan["decision"]["selectedFamily"], "finetune")
        self.assertEqual(len(plan["testJobs"]), 3)
        self.assertTrue(all("--calibration" in job["command"] for job in plan["testJobs"]))
        self.assertIn("--calibration", plan["exportJob"]["command"])

    def test_existing_target_is_rejected_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "existing"
            target.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                _assert_new_targets([target])

    def test_run_job_accepts_string_log_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outcome = _run_job(
                {
                    "job": "path-smoke",
                    "command": [sys.executable, "-c", "print('ok')"],
                    "stdout": str(root / "stdout.log"),
                    "stderr": str(root / "stderr.log"),
                }
            )
            self.assertEqual(outcome["returnCode"], 0)
            self.assertEqual((root / "stdout.log").read_text(encoding="utf-8"), "ok\n")
            self.assertEqual((root / "stderr.log").read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
