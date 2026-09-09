from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from glb_streaming import (  # noqa: E402
    BANDWIDTHS_MBPS,
    FIXED_RANDOM_METHODS,
    GlbAsset,
    PoseRecord,
    RANKING_METHODS,
    StreamingContractError,
    simulate_ranked_pose,
    simulate_threshold_filter_pose,
    summarize_filter_results,
    summarize_ranking_results,
)
from build_reference_frontmost_histogram import build_reference_histogram  # noqa: E402
from generate_streaming_paper_outputs import scene_display_name, summary_row  # noqa: E402


def assets() -> dict[int, GlbAsset]:
    return {
        0: GlbAsset(0, 10, 0, np.zeros(6, dtype=np.float32)),
        1: GlbAsset(1, 100, 1, np.ones(6, dtype=np.float32)),
        2: GlbAsset(2, 20, 2, np.full(6, 2, dtype=np.float32)),
    }


class GlbStreamingContractTests(unittest.TestCase):
    def test_paper_figure_uses_public_scene_names(self) -> None:
        self.assertEqual(scene_display_name("hkust-v3"), "HKUST")
        self.assertEqual(
            scene_display_name("ifcbench_fantasy_metropolis_instanced_v2"),
            "Metropolis",
        )

    def test_complete_glb_arrival_and_bytes_targets(self) -> None:
        pose = PoseRecord(
            pose_id=7,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(1, 2),
            utility_by_glb={1: 9.0, 2: 1.0},
            rank_scores={"full": {0: 0.9, 1: 0.8, 2: 0.7}},
        )
        result = simulate_ranked_pose(pose, assets(), "full")
        self.assertEqual(result["decisionMode"], "threshold_free_ranking")
        self.assertFalse(result["thresholdApplied"])
        self.assertEqual(result["bytesAtCoverage"]["95"], 130)
        self.assertEqual(result["bytesAtCoverage"]["99"], 130)
        self.assertEqual(result["bytesAtCoverage"]["99.9"], 130)
        self.assertEqual(result["bytesAtCoverage"]["100"], 130)
        self.assertEqual(result["requiredRank"], 3)
        self.assertEqual(result["wasteBefore99Bytes"], 10)
        self.assertAlmostEqual(result["timeSecondsAtCoverage"]["99"]["25"], 130 * 8 / 25_000_000)
        # No partial resource can contribute: rank 2 has only 90% utility,
        # even though the next GLB has already started in a real transport.
        self.assertEqual(result["rankAtCoverage"]["95"], 3)

    def test_oracle_is_utility_per_byte_and_not_global_optimum_claim(self) -> None:
        pose = PoseRecord(
            pose_id=1,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(1, 2),
            utility_by_glb={1: 9.0, 2: 1.0},
        )
        result = simulate_ranked_pose(pose, assets(), "gt_utility_per_byte_oracle")
        self.assertEqual(result["bytesAtCoverage"]["95"], 120)
        self.assertEqual(result["bytesAtCoverage"]["99"], 120)
        self.assertEqual(result["wasteBefore99Bytes"], 0)
        self.assertEqual(result["requiredRank"], 2)

    def test_fixed_random_methods_are_deterministic_and_keep_the_same_set(self) -> None:
        pose = PoseRecord(
            pose_id=13,
            ordinal=4,
            candidate_glb_ids=(2, 0, 1),
            gt_glb_ids=(1,),
        )
        first = [simulate_ranked_pose(pose, assets(), method) for method in FIXED_RANDOM_METHODS]
        second = [simulate_ranked_pose(pose, assets(), method) for method in FIXED_RANDOM_METHODS]
        self.assertEqual(len(FIXED_RANDOM_METHODS), 20)
        for left, right in zip(first, second):
            self.assertEqual(left, right)
            self.assertEqual(left["candidateGlbCount"], 3)
        self.assertEqual(len(RANKING_METHODS), 28)

    def test_missing_gt_resource_reports_unreachable_without_candidate_repair(self) -> None:
        pose = PoseRecord(
            pose_id=2,
            ordinal=0,
            candidate_glb_ids=(0, 1),
            gt_glb_ids=(1, 2),
            utility_by_glb={1: 1.0, 2: 1.0},
            rank_scores={"full": {0: 0.5, 1: 0.4}},
        )
        result = simulate_ranked_pose(pose, assets(), "full")
        self.assertEqual(result["coverageCeiling"], 0.5)
        self.assertTrue(result["coverageUnreachable"]["95"])
        self.assertIsNone(result["bytesAtCoverage"]["100"])
        self.assertIsNone(result["requiredRank"])
        self.assertEqual(result["missingGtGlbIds"], [2])

    def test_filtering_is_separate_and_reports_selected_subset(self) -> None:
        pose = PoseRecord(
            pose_id=3,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(1, 2),
            utility_by_glb={1: 9.0, 2: 1.0},
        )
        filtered = simulate_threshold_filter_pose(
            pose,
            assets(),
            method="full",
            instance_ids=(10, 11, 12),
            instance_scores=(0.1, 0.9, 0.2),
            instance_to_glb={10: 0, 11: 1, 12: 2},
            threshold=0.5,
        )
        self.assertEqual(filtered["decisionMode"], "threshold_filtering")
        self.assertTrue(filtered["thresholdApplied"])
        self.assertEqual(filtered["predictedGlbIds"], [1])
        self.assertEqual(filtered["predictedGlbBytes"], 100)
        self.assertAlmostEqual(filtered["coverageCeiling"], 0.9)
        self.assertTrue(filtered["coverageUnreachable"]["99"])
        ranking = simulate_ranked_pose(pose, assets(), "original")
        ranking_summary = summarize_ranking_results([{"methods": {"original": ranking}}], methods=("original",))
        filter_summary = summarize_filter_results([{"methods": {"full": filtered}}], methods=("full",))
        self.assertEqual(ranking_summary["decisionMode"], "threshold_free_ranking")
        self.assertEqual(filter_summary["decisionMode"], "threshold_filtering")
        self.assertNotIn("full", ranking_summary["methods"])
        self.assertNotIn("original", filter_summary["methods"])

    def test_invalid_score_method_is_rejected(self) -> None:
        pose = PoseRecord(
            pose_id=4,
            ordinal=0,
            candidate_glb_ids=(0,),
            gt_glb_ids=(),
        )
        with self.assertRaises(StreamingContractError):
            simulate_ranked_pose(pose, assets(), "aabb")

    def test_reference_frontmost_histogram_maps_component_ids_and_sums_samples(self) -> None:
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer_dir = root / "buffers"
            buffer_dir.mkdir()
            runtime_path = root / "runtime.json"
            manifest_path = root / "manifest.json"
            runtime_path.write_text(
                json.dumps(
                    {
                        "componentRecords": [
                            {"componentGlobalId": 0, "globalGlbId": 0},
                            {"componentGlobalId": 1, "globalGlbId": 0},
                            {"componentGlobalId": 2, "globalGlbId": 1},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "width": 2,
                        "height": 2,
                        "samples": [
                            {"sampleId": "a", "viewcellRow": 7},
                            {"sampleId": "b", "viewcellRow": 7},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            np.asarray([0, 1, 1, 3], dtype="<u4").tofile(buffer_dir / "a_reference_u32.bin")
            np.asarray([2, 3, 0, 0], dtype="<u4").tofile(buffer_dir / "b_reference_u32.bin")

            rows, metadata = build_reference_histogram(manifest_path, buffer_dir, runtime_path)

        self.assertEqual(metadata["histogramSchema"], "reference-frontmost-pixel-histogram-v1")
        self.assertEqual(rows[0]["poseId"], 7)
        self.assertEqual(rows[0]["sampleCount"], 2)
        self.assertEqual(rows[0]["histogram"], {"0": 3, "1": 2})
        self.assertEqual(rows[0]["frontmostPixelCount"], 5)

    def test_filter_table_keeps_predicted_bytes_separate_from_ranking_bytes(self) -> None:
        row = summary_row(
            {"coverageSource": "gt_glb_presence"},
            Path("fixture/filtering_summary.json"),
            "full",
            {
                "label": "Full",
                "status": "available",
                "decisionMode": "threshold_filtering",
                "poseCount": 2,
                "candidateGlbCount": {"mean": 3},
                "candidateGlbBytes": {"mean": 300},
                "predictedGlbCount": {"mean": 1},
                "predictedGlbBytes": {"mean": 100},
                "predictedGlbByteRatio": {"mean": 1 / 3},
                "coverageUpperBound": {"mean": 0.99},
                "unreachable": {"99": {"ratio": 0.5}},
            },
        )
        self.assertEqual(row["predicted_glb_mean"], 1)
        self.assertEqual(row["predicted_bytes_mean"], 100)
        self.assertAlmostEqual(row["predicted_byte_ratio_mean"], 1 / 3)
        self.assertIsNone(row["bytes_at_99_mean"])


if __name__ == "__main__":
    unittest.main()
