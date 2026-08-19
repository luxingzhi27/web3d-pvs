from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


BENCHMARK_DIR = Path(__file__).resolve().parents[1]
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import run_pvs_direction_conditioned_counterfactual_occlusion_v1 as runner  # noqa: E402


class DirectionConditionedCounterfactualRunnerTests(unittest.TestCase):
    def test_registered_matrix_is_complete_and_fixed_budget(self) -> None:
        jobs = runner._jobs()
        self.assertEqual(len(jobs), 4)
        self.assertEqual({int(job["seed"]) for job in jobs}, {runner.SEED})
        self.assertEqual({int(job["epochs"]) for job in jobs}, {8})
        self.assertEqual(
            {str(job["relationFeatureMode"]) for job in jobs},
            {"basis", "gated_contrast"},
        )
        self.assertEqual(
            {float(job["counterfactualWeight"]) for job in jobs}, {0.0, 0.05}
        )

    def test_commands_train_from_scratch_and_keep_runtime_shape(self) -> None:
        args = runner.parse_args(["--dry-run"])
        commands = [
            runner.build_train_command(
                args, job, Path("/tmp") / str(job["member"])
            )
            for job in runner._jobs()
        ]
        joined = [" ".join(command) for command in commands]
        self.assertTrue(all("--initial-checkpoint" not in row for row in joined))
        self.assertTrue(
            any("--runtime-relation-feature-mode gated_contrast" in row for row in joined)
        )
        self.assertTrue(
            any("--exposure-supervision-source relation_contrast" in row for row in joined)
        )
        self.assertTrue(
            any("--counterfactual-view-rank-weight 0.05" in row for row in joined)
        )
        self.assertEqual(runner.base.RUNTIME_FEATURE_DIM, 124)
        self.assertEqual(runner.base.RUNTIME_QUERY_INPUT_DIM, 130)
        self.assertEqual(runner.base.HKUST_RUNTIME_FEATURE_BYTES, 4_670_088)

    def test_selection_prioritizes_resource_efficiency_inside_safe_pool(self) -> None:
        def row(
            name: str,
            *,
            safe: bool,
            useful_cull: float,
            glb_reduction: float,
            balanced_accuracy: float,
        ) -> dict:
            return {
                "member": name,
                "eligibleSafe": safe,
                "aggregate": {
                    "weightedRecall": 0.995,
                    "weightedRecallLowerConfidenceBound": 0.992,
                    "usefulCull": useful_cull,
                    "glbByteReduction": glb_reduction,
                    "balancedAccuracy": balanced_accuracy,
                    "precision": 0.2,
                    "accuracy": 0.8,
                    "avgPredCount": 500.0,
                },
            }

        result = runner._selection(
            [
                row(
                    "efficient",
                    safe=True,
                    useful_cull=0.90,
                    glb_reduction=0.40,
                    balanced_accuracy=0.75,
                ),
                row(
                    "slightly_higher_balanced_accuracy",
                    safe=True,
                    useful_cull=0.65,
                    glb_reduction=0.08,
                    balanced_accuracy=0.76,
                ),
                row(
                    "unsafe",
                    safe=False,
                    useful_cull=0.99,
                    glb_reduction=0.80,
                    balanced_accuracy=0.95,
                ),
            ]
        )

        self.assertEqual(result["selectionPool"], "safe_candidate_pool")
        self.assertEqual(result["selected"]["member"], "efficient")
        self.assertIn("GLB byte reduction", result["selectionRule"])

    def test_completed_stage_record_is_reused_only_when_complete(self) -> None:
        schema = "registered-stage-v1"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stage.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": schema,
                        "results": [
                            {"member": "member-b", "returnCode": 0},
                            {"member": "member-a", "returnCode": 0},
                        ],
                        "testRead": False,
                    }
                ),
                encoding="utf-8",
            )
            rows = runner._load_completed_stage_results(
                path,
                schema=schema,
                expected_members={"member-a", "member-b"},
            )
            self.assertEqual(
                [row["member"] for row in rows or []],
                ["member-a", "member-b"],
            )

            with self.assertRaises(RuntimeError):
                runner._load_completed_stage_results(
                    path,
                    schema=schema,
                    expected_members={"member-a", "member-c"},
                )

if __name__ == "__main__":
    unittest.main()
