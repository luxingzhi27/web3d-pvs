from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import summarize_pvs_bounded_relation_survival_moment_v4 as summary  # noqa: E402


def _row(pose: int, pred: int, candidate: int, gt: int) -> dict:
    tp = min(pred, gt)
    fp = pred - tp
    fn = gt - tp
    tn = candidate - tp - fp - fn
    return {
        "poseIndex": pose,
        "candidateDigest": f"{pose:064d}",
        "metrics": {
            **{metric: 0.5 for metric in summary.METRIC_NAMES},
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "weightedTp": float(tp),
            "weightedGt": float(gt),
            "candidateCount": candidate,
            "candidateGlbCount": 2.0,
            "predictedGlbCount": 1.0,
            "candidateGlbBytes": 20.0,
            "predictedGlbBytes": 10.0,
            "requiredGlbCount": 1.0,
            "requiredGlbBytes": 10.0,
        },
    }


class BoundedRelationSurvivalMomentV4SummarizerTests(unittest.TestCase):
    def test_pooled_counts_keep_average_count_semantics(self) -> None:
        result = summary._summarize_rows([_row(0, 2, 10, 1), _row(1, 6, 20, 2)])
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["avgPredCount"], 4.0)
        self.assertEqual(aggregate["avgCandidateCount"], 15.0)
        self.assertEqual(aggregate["avgGtCount"], 1.5)
        self.assertEqual(aggregate["tp"], 1.5)

    def test_seed_clustered_gather_has_expected_shape(self) -> None:
        arrays = [{"value": np.arange(5.0)}, {"value": np.arange(5.0) + 10.0}]
        selected_seed = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        pose_indices = np.asarray(
            [[[0, 2, 4], [1, 3, 0]], [[4, 3, 2], [0, 1, 2]]],
            dtype=np.int64,
        )
        gathered = summary._gather_pose_field(arrays, "value", selected_seed, pose_indices)
        self.assertEqual(gathered.shape, (2, 2, 3))
        np.testing.assert_array_equal(gathered[0, 0], [0.0, 2.0, 4.0])
        np.testing.assert_array_equal(gathered[0, 1], [11.0, 13.0, 10.0])

    def test_threshold_provenance_is_checkpoint_owned_and_test_free(self) -> None:
        payload = {
            "threshold": 0.125,
            "thresholdSource": {
                "protocol": "checkpoint_own_calibration_only",
                "selectionSplit": "calibration",
                "selectedFromTest": False,
                "testEvaluationCount": 0,
                "selectedThreshold": 0.125,
                "safeWorkpoint": True,
            },
        }
        summary._validate_threshold_provenance(payload, Path("v4_fixture.json"))
        payload["thresholdSource"]["selectionSplit"] = "validation"
        with self.assertRaises(ValueError):
            summary._validate_threshold_provenance(payload, Path("v4_fixture.json"))

    def test_formal_comparisons_are_named_full_minus_ablation(self) -> None:
        pairs = summary._comparison_pairs(list(summary.V4_FORMAL_VARIANTS))
        self.assertEqual(len(summary.V4_FORMAL_VARIANTS), 5)
        self.assertEqual(len(summary.V4_FORMAL_VARIANTS) * len(summary.V4_SEEDS), 15)
        self.assertEqual(len(pairs), 4)
        self.assertEqual(
            len([name for name, _left, _right in pairs if name.startswith("full_minus_")]),
            4,
        )
        self.assertEqual(pairs[:2], [
            ("full_minus_without_bounded_relation", "without_bounded_relation", "full"),
            ("full_minus_without_viewcell_moment_envelope", "without_viewcell_moment_envelope", "full"),
        ])
        self.assertIn(
            (
                "full_minus_without_instance_calibration_residual",
                "without_instance_calibration_residual",
                "full",
            ),
            pairs,
        )

    def test_summarizer_rejects_a_legacy_matrix_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "matrix_manifest.json").write_text(
                '{"schema":"pvs-hierarchical-relation-survival-integrated-validation-summary-v1","testRead":false}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                summary.build_summary(root, 10000, 10000)

    def test_image_metric_summary_and_paired_bootstrap_keep_mean_and_p95(self) -> None:
        def sample(sample_id: str, pose: int, value: float) -> dict:
            return {
                "sampleId": sample_id,
                "viewcellRow": pose,
                "poseIndex": pose,
                "imageMetrics": {
                    "totalPixels": 100,
                    "validReferencePixels": 80,
                    "errorPixels": int(value * 80),
                    "missPixels": int(value * 80),
                    "wrongInstancePixels": 0,
                    "extraPixels": int(value * 100),
                    "missPixelRate": value,
                    "wrongInstancePixelRate": 0.0,
                    "extraPixelRateOverImage": value,
                },
            }

        left_rows = [sample("p0s0", 0, 0.20), sample("p1s0", 1, 0.20)]
        right_rows = [sample("p0s0", 0, 0.10), sample("p1s0", 1, 0.10)]
        image_members = {}
        for variant, rows in (("left", left_rows), ("right", right_rows)):
            for seed in summary.V4_SEEDS:
                image_members[(variant, seed)] = {
                    "summary": summary._image_metric_summary(rows),
                    "poseArrays": summary._image_pose_arrays(rows, [0, 1]),
                    "poseIndices": [0, 1],
                }
        effects = summary._paired_image_bootstrap(
            image_members, "left", "right", 10000, 7
        )
        self.assertAlmostEqual(effects["meanMissPixelRate"]["meanDelta"], -0.10)
        self.assertLess(effects["p95MissPixelRate"]["ci95"][1], 0.0)

    def test_image_backfill_rejects_a_software_gpu_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_root = Path(directory)
            members = {}
            manifest_members = []
            for variant in summary.V4_FORMAL_VARIANTS:
                for seed in summary.V4_SEEDS:
                    name = f"{variant}_seed{seed}"
                    checkpoint = image_root / f"{name}.pt"
                    runtime = image_root / f"{name}.bin"
                    members[(variant, seed)] = {
                        "payload": {"threshold": 0.25, "checkpoint": str(checkpoint)}
                    }
                    manifest_members.append({
                        "variant": variant,
                        "seed": seed,
                        "runtimeBundle": {
                            "runtimeFeatures": str(runtime),
                            "runtimeFeaturesSha256": "f" * 64,
                        },
                    })
                    output = image_root / name
                    render_root = output / "true_glb_render"
                    render_root.mkdir(parents=True)
                    sample_rows = [{
                        "sampleId": "vc00000_sp00_pose0",
                        "viewcellRow": 0,
                        "poseIndex": 0,
                        "imageMetrics": {
                            "totalPixels": 100,
                            "validReferencePixels": 80,
                            "errorPixels": 8,
                            "missPixels": 4,
                            "wrongInstancePixels": 4,
                            "extraPixels": 2,
                            "missPixelRate": 0.05,
                            "wrongInstancePixelRate": 0.05,
                            "extraPixelRateOverImage": 0.02,
                        },
                    }]
                    computed = summary._image_metric_summary(sample_rows)
                    hardware = not (variant == "full" and seed == summary.V4_SEEDS[0])
                    (output / "summary.json").write_text(json.dumps({
                        "schema": summary.IMAGE_SUMMARY_SCHEMA,
                        "modelName": name,
                        "split": "validation",
                        "formalImageEvaluationReady": True,
                        "threshold": 0.25,
                        "modelSpec": {
                            "checkpoint": str(checkpoint),
                            "runtime_features": str(runtime),
                        },
                    }), encoding="utf-8")
                    (render_root / "render_summary.json").write_text(json.dumps({
                        "schema": summary.IMAGE_RENDER_SCHEMA,
                        "formalImageEvaluationReady": True,
                        "gpuGate": {"hardware": hardware},
                        "imageMetrics": computed,
                    }), encoding="utf-8")
                    (render_root / "sample_image_metrics.json").write_text(
                        json.dumps(sample_rows), encoding="utf-8"
                    )
            manifest = {
                "validationPoseIndices": [0],
                "members": manifest_members,
            }
            with self.assertRaisesRegex(ValueError, "hardware GPU gate"):
                summary._load_image_members(image_root, members, manifest)

    def test_resource_backfill_requires_all_validation_members_and_binds_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = {}
            summaries = []
            for variant in summary.V4_FORMAL_VARIANTS:
                for seed in summary.V4_SEEDS:
                    name = f"{variant}_seed{seed}"
                    members[(variant, seed)] = {"payload": {"threshold": 0.25}}
                    summaries.append({
                        "name": name,
                        "thresholdRows": [{
                            "threshold": 0.25,
                            "avg_candidate_glb_count": 10.0,
                            "avg_pred_glb_count": 4.0,
                            "avg_candidate_glb_bytes": 100.0,
                            "avg_pred_glb_bytes": 40.0,
                        }],
                        "workpoints": {},
                        "scoreModes": {},
                    })
            (root / "summary.json").write_text(json.dumps({
                "meta": {
                    "requestedSplit": "validation",
                    "split": "validation",
                    "evaluatedPoses": 213,
                    "testEvaluationCount": None,
                },
                "summaries": summaries,
            }), encoding="utf-8")
            result = summary._load_resource_evaluation(root, members)
            self.assertEqual(result["memberCount"], 15)
            self.assertEqual(result["requestedSplit"], "validation")
            self.assertAlmostEqual(
                result["members"]["full_seed20260801"]["glbByteReduction"],
                0.60,
            )

            broken = json.loads((root / "summary.json").read_text(encoding="utf-8"))
            broken["meta"]["requestedSplit"] = "test"
            (root / "summary.json").write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "validation-only"):
                summary._load_resource_evaluation(root, members)


if __name__ == "__main__":
    unittest.main()
