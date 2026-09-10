from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCHMARK_DIR))

from glb_streaming import (  # noqa: E402
    BANDWIDTHS_MBPS,
    DECISION_MODE_SCHEDULER_REPLAY,
    NEURAL_COST_ALPHA_CANDIDATES,
    NEURAL_COST_METHOD,
    FIXED_RANDOM_METHODS,
    GlbAsset,
    PoseRecord,
    RANKING_METHODS,
    StreamingContractError,
    apply_neural_cost_alpha,
    select_neural_cost_alpha,
    simulate_ranked_pose,
    simulate_scheduler_replay_pose,
    simulate_threshold_filter_pose,
    summarize_filter_results,
    summarize_ranking_results,
)
from build_reference_frontmost_histogram import build_reference_histogram  # noqa: E402
from export_glb_streaming_scores import load_formal_aabb_test_sidecar  # noqa: E402
from export_glb_streaming_scores import _model_sources  # noqa: E402
from generate_streaming_paper_outputs import (  # noqa: E402
    scene_display_name,
    summary_row,
    validate_scheduler_replay_summary,
    validate_paper_summary,
)
from score_sidecar import ScoreSidecarWriter  # noqa: E402
from glb_streaming_io import (  # noqa: E402
    LoadedPose,
    attach_geometry_scores,
    load_neural_cost_manifest,
    write_neural_cost_manifest,
)
from simulate_glb_streaming import (  # noqa: E402
    attach_formal_region66_visible_glbs,
    hzb_ordering_metadata,
    load_formal_region66_test_result,
    ordered_hzb_glb_ids_for_pose,
    require_complete_test_pose_selection,
    simulate_hzb_ranked_pose,
)


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

    def test_neural_glb_probability_and_frozen_cost_order(self) -> None:
        pose = PoseRecord(
            pose_id=8,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(0, 1, 2),
            utility_by_glb={0: 1.0, 1: 1.0, 2: 1.0},
            rank_scores={"neural_probability": {0: 0.6, 1: 0.9, 2: 0.8}},
            neural_cost_alpha=0.5,
            coverage_source="visible_weights",
            coverage_semantics="fixture visible weights",
            coverage_unit="fixture weight",
        )
        pure = simulate_ranked_pose(pose, assets(), "neural_probability")
        cost = simulate_ranked_pose(pose, assets(), NEURAL_COST_METHOD)
        self.assertEqual(pure["rankingInput"]["formula"], "p_g=max_i(p_i)")
        self.assertEqual(cost["rankingInput"]["costExponent"], 0.5)
        self.assertEqual(cost["rankingInput"]["instanceProbabilityAggregation"], "max")
        self.assertEqual(cost["rankingInput"]["bytesSource"], "complete GLB file bytes")
        self.assertEqual(cost["visibleWeightCoverageCeiling"], 1.0)

    def test_neural_cost_alpha_selection_is_validation_only_and_manifest_is_frozen(self) -> None:
        poses = [
            PoseRecord(
                pose_id=9,
                ordinal=0,
                candidate_glb_ids=(0, 1, 2),
                gt_glb_ids=(1, 2),
                utility_by_glb={1: 9.0, 2: 1.0},
                rank_scores={"full": {0: 0.1, 1: 0.9, 2: 0.8}},
                coverage_source="visible_weights",
                coverage_semantics="fixture visible weights",
                coverage_unit="fixture weight",
            )
        ]
        manifest = select_neural_cost_alpha(poses, assets(), selection_split="validation")
        self.assertTrue(manifest["frozen"])
        self.assertEqual(manifest["selectionSplit"], "validation")
        self.assertEqual(manifest["candidateAlphas"], list(NEURAL_COST_ALPHA_CANDIDATES))
        self.assertIn(manifest["selectedAlpha"], NEURAL_COST_ALPHA_CANDIDATES)
        self.assertEqual(manifest["coverageSource"], poses[0].coverage_source)
        self.assertEqual(manifest["coverageUnit"], poses[0].coverage_unit)
        self.assertEqual(len(manifest["candidates"]), 3)
        with self.assertRaisesRegex(StreamingContractError, "only on the validation"):
            select_neural_cost_alpha(poses, assets(), selection_split="test")

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = write_neural_cost_manifest(Path(directory) / "alpha.json", manifest)
            loaded = load_neural_cost_manifest(path)
        self.assertEqual(loaded["selectedAlpha"], manifest["selectedAlpha"])
        self.assertEqual(loaded["path"], str(path))

    def test_neural_cost_manifest_rejects_unregistered_alpha(self) -> None:
        manifest = {
            "schema": "pvs-glb-streaming-neural-cost-manifest-v1",
            "frozen": True,
            "eligibleForTest": True,
            "selectionSplit": "validation",
            "candidateAlphas": [0.0, 0.5, 1.0],
            "selectedAlpha": 0.25,
            "formula": "p_g / bytes_g^alpha",
            "instanceProbabilityAggregation": "max",
            "bytesSource": "complete GLB file bytes",
            "coverageSource": "visible_weight_coverage",
            "coverageUnit": "rvcServer_component_weight",
            "coverageSemantics": "visible weights grouped by GLB",
        }
        from glb_streaming_io import validate_neural_cost_manifest

        with self.assertRaisesRegex(StreamingContractError, "selectedAlpha"):
            validate_neural_cost_manifest(manifest)

    def test_geometry_heuristics_aggregate_candidate_instance_aabbs(self) -> None:
        pose = PoseRecord(
            pose_id=11,
            ordinal=0,
            candidate_glb_ids=(0, 1),
            gt_glb_ids=(0,),
        )
        loaded = LoadedPose(
            record=pose,
            candidate_instance_ids=np.asarray([0, 1, 2], dtype=np.uint32),
            visible_instance_ids=np.asarray([0], dtype=np.uint32),
            camera_world=np.zeros(3, dtype=np.float32),
            camera_forward=np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            camera_view=np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            mvp=np.eye(4, dtype=np.float32).reshape(-1),
            candidate_instance_aabbs=np.asarray(
                [
                    [10.0, 10.0, 1.0, 10.1, 10.1, 1.1],
                    [0.0, 0.0, 1.0, 1.0, 1.0, 1.1],
                    [0.0, 0.0, 2.0, 0.1, 0.1, 2.1],
                ],
                dtype=np.float32,
            ),
        )
        instance_to_glb = {0: 0, 1: 0, 2: 1}
        result = attach_geometry_scores(
            [loaded],
            {
                0: GlbAsset(0, 10, 0, np.asarray([-100, -100, -100, 100, 100, 100], dtype=np.float32)),
                1: GlbAsset(1, 20, 1, np.asarray([-100, -100, -100, 100, 100, 100], dtype=np.float32)),
            },
            {},
            instance_to_glb,
        )[0]
        self.assertAlmostEqual(result.rank_scores["distance"][0], 0.5)
        self.assertAlmostEqual(result.rank_scores["distance"][1], 1.0 / 3.0)
        self.assertGreater(result.rank_scores["projected_area"][0], result.rank_scores["projected_area"][1])
        self.assertEqual(result.rank_scores["projected_area_per_byte"][0], result.rank_scores["projected_area"][0] / 10.0)

    def test_geometry_heuristics_do_not_fallback_to_merged_asset_aabb(self) -> None:
        pose = PoseRecord(
            pose_id=12,
            ordinal=0,
            candidate_glb_ids=(0,),
            gt_glb_ids=(),
        )
        loaded = SimpleNamespace(
            record=pose,
            candidate_instance_ids=np.asarray([0], dtype=np.uint32),
            camera_world=np.zeros(3, dtype=np.float32),
            candidate_instance_aabbs=None,
        )
        result = attach_geometry_scores(
            [loaded],
            {0: GlbAsset(0, 10, 0, np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float32))},
            {},
            {0: 0},
        )[0]
        self.assertNotIn("distance", result.rank_scores)
        self.assertNotIn("projected_area", result.rank_scores)

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
        self.assertEqual(len(RANKING_METHODS), 29)

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

    def test_visible_weight_coverage_is_explicit_and_not_pixel_coverage(self) -> None:
        pose = PoseRecord(
            pose_id=14,
            ordinal=0,
            candidate_glb_ids=(0, 1),
            gt_glb_ids=(1,),
            utility_by_glb={1: 7.0},
            rank_scores={"full": {0: 0.99, 1: 0.5}},
            coverage_source="visible_weights",
            coverage_semantics="rvcServer visible component weight",
            coverage_unit="rvcServer_component_weight",
        )
        result = simulate_ranked_pose(pose, assets(), "full")
        self.assertEqual(result["coverageSource"], "visible_weight_coverage")
        self.assertEqual(result["coverageMetric"], "visible_weight_coverage")
        self.assertIn("visibleWeightCoverageCurve", result)
        self.assertIn("visibleWeightCoverage", result["coverageCurve"][0])
        self.assertNotIn("pixelCoverage", result)
        self.assertNotIn("pixel_coverage", result)

    def test_formal_test_pose_selection_rejects_limits_and_subsets(self) -> None:
        class DatasetFixture:
            def split(self, name: str) -> SimpleNamespace:
                self.assert_name = name
                return SimpleNamespace(pose_indices=np.asarray([3, 7, 9], dtype=np.int64))

        dataset = DatasetFixture()
        with self.assertRaisesRegex(StreamingContractError, "pose_limit"):
            require_complete_test_pose_selection(dataset, [3, 7, 9], pose_limit=1)
        with self.assertRaisesRegex(StreamingContractError, "complete test split"):
            require_complete_test_pose_selection(dataset, [3, 7])
        self.assertEqual(require_complete_test_pose_selection(dataset, [3, 7, 9]), [3, 7, 9])

    def test_scheduler_replay_is_not_a_threshold_free_ranking_result(self) -> None:
        pose = PoseRecord(
            pose_id=15,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(1,),
        )
        result = simulate_scheduler_replay_pose(
            pose,
            assets(),
            method="neural_scheduler",
            tiers={"urgent": [1], "warm": [2], "speculative": [0]},
            threshold_applied=True,
        )
        self.assertEqual(result["decisionMode"], DECISION_MODE_SCHEDULER_REPLAY)
        self.assertTrue(result["thresholdApplied"])
        self.assertEqual(result["schedulerReplay"]["arrivalOrder"], [1, 2, 0])

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

    def test_formal_aabb_sidecar_is_loaded_as_continuous_candidate_aligned_scores(self) -> None:
        class DatasetFixture:
            def candidate_slice(self, pose_id: int) -> np.ndarray:
                return np.asarray({10: [0, 2]}[pose_id], dtype=np.uint32)

        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "aabb_test.sidecar"
            with ScoreSidecarWriter(sidecar, split="test", threshold=0.73) as writer:
                writer.append_pose(
                    10,
                    np.asarray([0, 2], dtype=np.uint32),
                    np.asarray([0.21, 0.89], dtype=np.float32),
                    np.asarray([0, 1], dtype=np.uint8),
                    np.asarray([0.0, 1.0], dtype=np.float32),
                )
            scores, source = load_formal_aabb_test_sidecar(
                sidecar,
                DatasetFixture(),
                np.asarray([10], dtype=np.int64),
            )

        np.testing.assert_allclose(scores[10], [0.21, 0.89])
        self.assertEqual(source["kind"], "formal_aabb_test_sidecar")
        self.assertTrue(source["testRead"])
        self.assertEqual(source["continuousScoreField"], "scores")

    def test_formal_region66_result_maps_instances_and_uses_projected_area_per_byte(self) -> None:
        import json
        import tempfile

        payload = {
            "schema": "geometry-shell-hzb-browser-result-v2",
            "mode": "Region66",
            "formalReady": True,
            "executionClass": "formal-hardware-gpu",
            "gpuGate": {"required": True, "hardware": True},
            "workload": {
                "schema": "geometry-shell-hzb-browser-workload-v1",
                "scene": "fixture-scene",
                "split": "test",
                "poseCount": 1,
                "candidateCount": 3,
                "candidateFile": "geometry_shell_hzb_candidates_uint32.bin",
                "candidateDtype": "uint32-little-endian",
                "fovYDeg": 66,
                "poseSelection": {
                    "split": "test",
                    "selectedPoseIndices": [10],
                    "selectedPoseCount": 1,
                    "limit": 0,
                    "representative": True,
                },
                "provenance": {
                    "configuration": {"fovYDeg": 60, "regionFovYDeg": 66},
                },
            },
            "samples": [
                {"poseId": 10, "candidateCount": 3, "visibleInstanceIds": [0, 2]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "region66.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            np.asarray([0, 1, 2], dtype="<u4").tofile(
                result_path.parent / "geometry_shell_hzb_candidates_uint32.bin"
            )
            visible, source = load_formal_region66_test_result(
                result_path,
                [10],
                expected_scene="fixture-scene",
                expected_candidate_ids={10: [0, 1, 2]},
            )

        self.assertEqual(visible, {10: (0, 2)})
        self.assertEqual(source["kind"], "formal_region66_test_result")
        self.assertFalse(source["continuousScore"])

        pose = PoseRecord(
            pose_id=10,
            ordinal=0,
            candidate_glb_ids=(0, 1, 2),
            gt_glb_ids=(0, 2),
            rank_scores={"projected_area_per_byte": {0: 0.2, 1: 0.99, 2: 0.5}},
        )
        loaded = SimpleNamespace(
            record=SimpleNamespace(pose_id=10),
            candidate_instance_ids=np.asarray([0, 1, 2], dtype=np.uint32),
        )
        mapped = attach_formal_region66_visible_glbs(
            [pose], [loaded], visible, {0: 0, 1: 1, 2: 2}
        )[0]
        self.assertNotIn("hzb_visible_first", mapped.rank_scores)
        self.assertEqual(ordered_hzb_glb_ids_for_pose(mapped, assets()), (2, 0, 1))
        ranking = simulate_hzb_ranked_pose(mapped, assets())
        self.assertEqual(ranking["method"], "hzb_visible_first")
        self.assertEqual(ranking["rankingInput"], hzb_ordering_metadata())
        self.assertFalse(ranking["rankingInput"]["continuousScore"])

    def test_formal_region66_result_rejects_candidate_count_mismatch(self) -> None:
        import json
        import tempfile

        payload = {
            "schema": "geometry-shell-hzb-browser-result-v2",
            "mode": "Region66",
            "formalReady": True,
            "executionClass": "formal-hardware-gpu",
            "gpuGate": {"required": True, "hardware": True},
            "workload": {
                "schema": "geometry-shell-hzb-browser-workload-v1",
                "scene": "fixture-scene",
                "split": "test",
                "poseCount": 1,
                "candidateCount": 2,
                "candidateFile": "geometry_shell_hzb_candidates_uint32.bin",
                "candidateDtype": "uint32-little-endian",
                "fovYDeg": 66,
                "poseSelection": {
                    "split": "test",
                    "selectedPoseIndices": [10],
                    "selectedPoseCount": 1,
                    "limit": 0,
                    "representative": True,
                },
                "provenance": {
                    "configuration": {"fovYDeg": 60, "regionFovYDeg": 66},
                },
            },
            "samples": [
                {"poseId": 10, "candidateCount": 2, "visibleInstanceIds": [0]},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "region66.json"
            result_path.write_text(json.dumps(payload), encoding="utf-8")
            np.asarray([0, 1], dtype="<u4").tofile(
                result_path.parent / "geometry_shell_hzb_candidates_uint32.bin"
            )
            with self.assertRaisesRegex(StreamingContractError, "does not match CSR row length"):
                load_formal_region66_test_result(
                    result_path,
                    [10],
                    expected_scene="fixture-scene",
                    expected_candidate_ids={10: [0, 1, 2]},
                )

    def test_legacy_aabb_summary_is_unavailable_in_paper_outputs(self) -> None:
        row = summary_row(
            {
                "coverageSource": "gt_glb_presence",
                "methods": {},
            },
            Path("fixture/ranking_summary.json"),
            "aabb",
            {
                "status": "available",
                "decisionMode": "threshold_free_ranking",
                "poseCount": 1,
            },
        )
        self.assertEqual(row["status"], "unavailable")
        self.assertIn("formal test score sidecar", row["availability_reason"])

    def test_paper_output_rejects_non_test_or_partial_summary(self) -> None:
        ranking = {
            "schema": "pvs-glb-streaming-ranking-summary-v1",
            "decisionMode": "threshold_free_ranking",
            "thresholdApplied": False,
            "cacheMode": "strict_cold_cache_per_pose",
            "split": "validation",
            "testRead": False,
            "poseCount": 2,
            "methods": {
                "full": {
                    "status": "available",
                    "decisionMode": "threshold_free_ranking",
                    "poseCount": 2,
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "frozen test summaries"):
            validate_paper_summary(ranking, Path("ranking.json"), "threshold_free_ranking")
        ranking["split"] = "test"
        ranking["testRead"] = True
        ranking["methods"]["full"]["poseCount"] = 1
        with self.assertRaisesRegex(ValueError, "complete test summary"):
            validate_paper_summary(ranking, Path("ranking.json"), "threshold_free_ranking")

    def test_paper_output_accepts_complete_formal_filter_summary(self) -> None:
        filtering = {
            "schema": "pvs-glb-streaming-filter-summary-v1",
            "decisionMode": "threshold_filtering",
            "thresholdApplied": True,
            "cacheMode": "strict_cold_cache_per_pose",
            "split": "test",
            "testRead": True,
            "poseCount": 2,
            "testPoseCount": 2,
            "poseSelection": {
                "split": "test",
                "selectedPoseCount": 2,
                "limit": 0,
                "completeTest": True,
            },
            "methods": {
                "full": {
                    "status": "available",
                    "decisionMode": "threshold_filtering",
                    "poseCount": 2,
                }
            },
        }
        validate_paper_summary(filtering, Path("filtering.json"), "threshold_filtering")

    def test_scheduler_summary_has_a_separate_contract(self) -> None:
        scheduler = {
            "schema": "pvs-real-scheduler-streaming-summary-v1",
            "scheduler": "GlbResourceScheduler",
            "schedulerTiers": ["urgent", "warm", "speculative"],
            "split": "test",
            "testRead": True,
            "poseCount": 12,
            "poseIds": list(range(12)),
            "startup100Enabled": False,
            "startupTierUsed": False,
            "cacheMode": "strict_cold_cache_per_pose",
        }
        validate_scheduler_replay_summary(scheduler, Path("scheduler.json"))
        validate_paper_summary(scheduler, Path("scheduler.json"), DECISION_MODE_SCHEDULER_REPLAY)

    def test_aabb_runner_spec_never_turns_into_a_fallback_source(self) -> None:
        runner_specs, sidecar, requested = _model_sources(
            [("aabb", {"kind": "aabb_ray", "checkpoint": "-"})],
            None,
        )
        self.assertEqual(runner_specs, [])
        self.assertIsNone(sidecar)
        self.assertTrue(requested)


if __name__ == "__main__":
    unittest.main()
