from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from neural_instance_culling.benchmark.evaluate_download_trajectory import (
    REPLAY_SCHEMA,
    TRAJECTORY_SCHEMA,
    PosePlan,
    ReplayConfig,
    build_pose_plans,
    load_trajectory,
    simulate_download_trajectory,
)


class _FakeDataset:
    def __init__(self, candidates: list[list[int]], visible: list[list[int]], weights: list[list[float]]) -> None:
        self.poses = np.zeros(
            (len(candidates),),
            dtype=np.dtype(
                [
                    ("camera_norm", "<f4", (3,)),
                    ("camera_world", "<f4", (3,)),
                ]
            ),
        )
        self.poses["camera_norm"] = 0.5
        self.poses["camera_world"] = 0.0
        self.mvp = None
        self._candidates = [np.asarray(row, dtype=np.uint32) for row in candidates]
        self._visible = [np.asarray(row, dtype=np.uint32) for row in visible]
        self._weights = [np.asarray(row, dtype=np.float32) for row in weights]

    def frustum_slice(self, pose_index: int) -> np.ndarray:
        return self._candidates[pose_index]

    def visible_slice(self, pose_index: int) -> tuple[np.ndarray, np.ndarray]:
        return self._visible[pose_index], self._weights[pose_index]

    def camera_view(self, pose_index: int) -> np.ndarray:
        del pose_index
        return np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32)


class _FakeRunner:
    name = "fake_runner"
    kind = "fixture"
    threshold = 0.5
    information_level = "fixture"
    decision_mode = "ranking"

    def __init__(self) -> None:
        self.instance_to_glb = np.asarray([0, 0, 1], dtype=np.int64)

    def score_arrays(self, camera, camera_world, camera_view, instance_ids, mvp=None):
        del camera, camera_world, camera_view, mvp
        ids = np.asarray(instance_ids, dtype=np.int64)
        # Instance 1 gets the higher score within GLB 0; the returned order
        # remains exactly the candidate order supplied by the dataset.
        values = np.asarray([0.2, 0.9, 0.4], dtype=np.float32)[ids]
        return SimpleNamespace(scores=values, utility_scores=None, download_scores=None)


def _fixture_plans() -> list[PosePlan]:
    return [
        PosePlan(
            sequence=0,
            pose_index=0,
            time_ms=0.0,
            candidate_count=3,
            candidate_glb_count=3,
            ranked_glbs=np.asarray([0, 2, 1], dtype=np.int64),
            ranked_scores=np.asarray([0.9, 0.8, 0.7], dtype=np.float64),
            utility_by_glb={0: 1.0},
        ),
        PosePlan(
            sequence=1,
            pose_index=1,
            time_ms=300.0,
            candidate_count=3,
            candidate_glb_count=3,
            ranked_glbs=np.asarray([1, 0, 2], dtype=np.int64),
            ranked_scores=np.asarray([0.95, 0.6, 0.1], dtype=np.float64),
            utility_by_glb={1: 1.0},
        ),
    ]


class DownloadTrajectoryTests(unittest.TestCase):
    def test_pose_index_loader_and_runner_glb_aggregation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trajectory.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": TRAJECTORY_SCHEMA,
                        "poseIndices": [0, 1],
                        "stepMs": 125,
                        "network": {"bandwidthBytesPerSec": 1000},
                        "cache": {"mode": "cold"},
                    }
                ),
                encoding="utf-8",
            )
            trajectory = load_trajectory(path)
        self.assertEqual([row.pose_index for row in trajectory.poses], [0, 1])
        self.assertEqual([row.time_ms for row in trajectory.poses], [0.0, 125.0])

        dataset = _FakeDataset([[0, 1, 2], [2]], [[1], [2]], [[3.0], [1.0]])
        plans, status = build_pose_plans(
            _FakeRunner(),
            dataset,
            trajectory,
            score_mode="visibility-only",
            aggregation="max",
        )
        self.assertEqual(status["status"], "implemented")
        self.assertEqual(plans[0].ranked_glbs.tolist(), [0, 1])
        self.assertAlmostEqual(plans[0].ranked_scores[0], 0.9)
        self.assertAlmostEqual(plans[0].utility_by_glb[0], np.log1p(3.0))
        self.assertEqual(plans[1].ranked_glbs.tolist(), [1])

    def test_visible_outside_candidate_is_rejected_without_repair(self) -> None:
        trajectory = SimpleNamespace(
            poses=(
                SimpleNamespace(sequence=0, pose_index=0, time_ms=0.0),
            )
        )
        dataset = _FakeDataset([[0]], [[2]], [[1.0]])
        with self.assertRaisesRegex(ValueError, "outside the stored candidate set"):
            build_pose_plans(
                _FakeRunner(),
                dataset,
                trajectory,
                score_mode="visibility-only",
                aggregation="max",
            )

    def test_cold_replay_tracks_completion_and_invalid_bytes(self) -> None:
        result = simulate_download_trajectory(
            _fixture_plans(),
            np.asarray([100.0, 200.0, 50.0]),
            np.asarray([20.0, 10.0, 5.0]),
            ReplayConfig(
                bandwidth_bytes_per_sec=1000.0,
                request_latency_ms=0.0,
                max_concurrent_downloads=2,
                max_concurrent_decode_uploads=1,
                initial_cache_glb_ids=frozenset(),
                cache_mode="cold",
                drain_after_last_pose_ms=500.0,
            ),
            byte_budgets=(100.0, 350.0),
            time_budgets_ms=(220.0, 1000.0),
        )
        accounting = result["resourceAccounting"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual(accounting["downloadedBytes"], 350.0)
        self.assertEqual(accounting["invalidDownloadBytes"], 50.0)
        self.assertEqual(result["firstUsefulFrame"]["timeMs"], 220.0)
        self.assertEqual(result["utilityAtBytes"]["budgets"]["350.0"]["utilityRecall"], 1.0)
        self.assertGreater(result["missingUtility"]["missingUtilityRatioIntegralMs"], 0.0)
        self.assertEqual(len(result["events"]["downloads"]), 3)
        self.assertEqual(len(result["events"]["decodeUploads"]), 3)

    def test_warm_cache_produces_useful_frame_without_network_bytes(self) -> None:
        result = simulate_download_trajectory(
            _fixture_plans()[:1],
            np.asarray([100.0, 200.0, 50.0]),
            np.asarray([20.0, 10.0, 5.0]),
            ReplayConfig(
                bandwidth_bytes_per_sec=1000.0,
                request_latency_ms=0.0,
                max_concurrent_downloads=1,
                max_concurrent_decode_uploads=1,
                initial_cache_glb_ids=frozenset({0}),
                cache_mode="warm",
                drain_after_last_pose_ms=0.0,
            ),
        )
        self.assertEqual(result["firstUsefulFrame"]["timeMs"], 0.0)
        self.assertEqual(result["cache"]["initialGlbIds"], [0])
        self.assertEqual(result["utilityAtBytes"]["curve"][1]["downloadedBytes"], 0.0)

    def test_replay_schema_is_explicit(self) -> None:
        self.assertEqual(REPLAY_SCHEMA, "neuralstreamweb3d-download-trajectory-replay-v1")


if __name__ == "__main__":
    unittest.main()
