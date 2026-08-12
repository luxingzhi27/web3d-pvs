from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from run_ray_context_survival_owrb_supplement import _comparisons  # noqa: E402
from summarize_ray_context_survival_owrb import ALL_METRICS  # noqa: E402
from validate_ray_context_survival_owrb_supplement import (  # noqa: E402
    EXPECTED_RUNTIME_DIMS,
    SEEDS,
    SUPPLEMENT_ORDER,
    validate_summary,
)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class RayContextSurvivalOWRBSupplementValidatorTest(unittest.TestCase):
    def _fixture(self, directory: Path) -> tuple[dict, dict[int, Path]]:
        pose_count = 213
        pose_indices = list(range(pose_count))
        candidate_digest = "b" * 64
        row_hashes = {pose: _sha256_text(f"candidate-{pose}") for pose in pose_indices}
        members: dict[str, dict[str, dict]] = {variant: {} for variant in SUPPLEMENT_ORDER}
        paths: dict[int, Path] = {}
        for seed in SEEDS:
            for variant in SUPPLEMENT_ORDER:
                table_dim, input_dim, ray_dim = EXPECTED_RUNTIME_DIMS[variant]
                rows = [
                    {
                        "poseIndex": pose,
                        "candidateIdSha256": row_hashes[pose],
                        "imageMetrics": {"PER": 0.0},
                    }
                    for pose in pose_indices
                ]
                evaluation = {
                    "schema": "ray-context-survival-owrb-evaluation-v1",
                    "split": "validation",
                    "testRead": False,
                    "candidateDigest": candidate_digest,
                    "threshold": 0.5,
                    "thresholdSource": {
                        "protocol": "calibration_ready_pre_test",
                        "testEvaluationCount": 0,
                        "safeWorkpoint": True,
                        "selection": {
                            "pose_weighted_recall": 0.995,
                            "weighted_recall_lower_confidence_bound": 0.992,
                        },
                    },
                    "safeWorkpoint": True,
                    "imageMetrics": {"status": "formal_ready", "perPoseAvailable": True},
                    "perPose": rows,
                }
                path = directory / f"{variant}_seed{seed}.json"
                path.write_text(json.dumps(evaluation), encoding="utf-8")
                paths[seed] = path
                members[variant][str(seed)] = {
                    "path": str(path),
                    "checkpoint": str(directory / f"{variant}_seed{seed}.pt"),
                    "threshold": 0.5,
                    "runtimeFeatureBytes": table_dim * 1000,
                    "runtimeFeatureTableDim": table_dim,
                    "runtimeInputFeatureDim": input_dim,
                    "visibilityHeadInputDim": input_dim,
                    "rayQueryDim": ray_dim,
                    "safeWorkpoint": True,
                    "poseMacro": {
                        "precision": 0.5,
                        "recall": 0.9,
                        "weightedRecall": 0.995,
                        "accuracy": 0.9,
                        "balancedAccuracy": 0.9,
                        "specificity": 0.9,
                        "f1": 0.6,
                        "jaccard": 0.5,
                        "usefulCull": 0.8,
                        "badCull": 0.01,
                    },
                    "aggregate": {
                        "precision": 0.5,
                        "recall": 0.9,
                        "weightedRecall": 0.995,
                        "accuracy": 0.9,
                        "balancedAccuracy": 0.9,
                        "specificity": 0.9,
                        "f1": 0.6,
                        "jaccard": 0.5,
                        "usefulCull": 0.8,
                        "badCull": 0.01,
                    },
                    "runtimeLatency": {"meanMs": 1.0, "p50Ms": 1.0, "p95Ms": 2.0, "p99Ms": 3.0},
                    "imageMetrics": {"status": "formal_ready", "perPoseAvailable": True},
                }

        comparisons = {}
        for item in _comparisons({variant: {} for variant in SUPPLEMENT_ORDER}):
            comparisons[item["name"]] = {
                "left": item["left"],
                "right": item["right"],
                "metrics": {
                    metric: {
                        "meanDelta": 0.0,
                        "ci95": [-0.1, 0.1],
                        "direction": "zero",
                        "crossesZero": True,
                        "bootstrapReplicates": 10000,
                        "clusterUnit": "outer seed cluster, inner pose resampling",
                    }
                    for metric in ALL_METRICS
                },
            }
        # The production summary computes this digest before JSON encoding,
        # with integer seed/pose keys.  Keep the fixture's key types identical
        # because lexical string ordering differs from numeric ordering for
        # poses such as 2 and 10.
        reference_hashes = {seed: dict(row_hashes) for seed in SEEDS}
        summary = {
            "schema": "ray-context-survival-owrb-matrix-summary-v1",
            "mode": "formal",
            "supplement": True,
            "testRead": False,
            "variants": list(SUPPLEMENT_ORDER),
            "seeds": list(SEEDS),
            "poseCount": pose_count,
            "poseIndices": pose_indices,
            "candidateHashDigest": _sha256_text(json.dumps(reference_hashes, sort_keys=True)),
            "members": members,
            "comparisons": comparisons,
            "factorEffects": {},
            "metricNames": list(ALL_METRICS),
            "bootstrap": {
                "replicates": 10000,
                "paired": True,
                "clusterUnit": "outer seed cluster, inner pose resampling",
            },
            "matrixDesign": {
                "type": "registered_supplement_mechanism_audit",
                "variantCount": 5,
                "derivedContrastCount": 0,
            },
            "imageEvaluation": {"status": "formal_ready", "available": True},
            "webgpuParity": {"status": "not_registered"},
        }
        return summary, paths

    def test_accepts_registered_five_variants_and_shared_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            summary, _paths = self._fixture(Path(raw))
            result = validate_summary(summary)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["variantCount"], 5)
        self.assertEqual(result["seedCount"], 3)
        self.assertEqual(result["poseCount"], 213)
        self.assertEqual(result["pairwiseComparisonCount"], 4)
        self.assertFalse(result["testRead"])

    def test_rejects_candidate_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            summary, _paths = self._fixture(Path(raw))
            summary["candidateHashDigest"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "candidateHashDigest"):
                validate_summary(summary)

    def test_rejects_unfrozen_threshold_or_nonformal_image(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            summary, paths = self._fixture(Path(raw))
            payload = json.loads(paths[SEEDS[0]].read_text(encoding="utf-8"))
            payload["thresholdSource"]["protocol"] = "validation_selected"
            paths[SEEDS[0]].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "calibration-frozen"):
                validate_summary(summary)

            summary, paths = self._fixture(Path(raw))
            # The member table is nested under ``members``; keep this mutation
            # explicit so the test guards the formal image gate itself.
            summary["members"][SUPPLEMENT_ORDER[0]][str(SEEDS[0])]["imageMetrics"] = {"status": "not_ready"}
            with self.assertRaisesRegex(ValueError, "formal image metrics"):
                validate_summary(summary)

    def test_rejects_insufficient_bootstrap_or_test_read(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            summary, _paths = self._fixture(Path(raw))
            summary["bootstrap"]["replicates"] = 9999
            with self.assertRaisesRegex(ValueError, "10000"):
                validate_summary(summary)

            summary, _paths = self._fixture(Path(raw))
            summary["testRead"] = True
            with self.assertRaisesRegex(ValueError, "test was read"):
                validate_summary(summary)

    def test_wrong_variant_count_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            summary, _paths = self._fixture(Path(raw))
            summary["variants"] = list(SUPPLEMENT_ORDER[:-1])
            with self.assertRaisesRegex(ValueError, "five-member"):
                validate_summary(summary)


if __name__ == "__main__":
    unittest.main()
