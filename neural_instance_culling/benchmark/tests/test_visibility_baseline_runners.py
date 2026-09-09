from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BENCHMARK_DIR.parent / "model"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(MODEL_DIR))

from model_runners import (  # noqa: E402
    PvsV4Runner,
    load_runner,
    selected_default_specs,
)
from pose_csr_dataset import DIRECTIONAL_POSE_DTYPE  # noqa: E402
from train_aabb_ray_baseline import AabbRayMLP  # noqa: E402


def write_runtime_meta(path: Path) -> None:
    records = []
    aabbs = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 1.0, 0.25, 0.25, 2.0],
            [10.0, 0.0, 10.0, 11.0, 1.0, 11.0],
        ],
        dtype=np.float32,
    )
    for instance_id, aabb in enumerate(aabbs):
        records.append(
            {
                "componentGlobalId": instance_id,
                "globalGlbId": instance_id,
                "bounds": {"min": aabb[:3].tolist(), "max": aabb[3:].tolist()},
            }
        )
    path.write_text(
        json.dumps(
            {
                "sceneBounds": {"min": [-1.0, -1.0, 0.0], "max": [12.0, 2.0, 12.0]},
                "componentRecords": records,
            }
        ),
        encoding="utf-8",
    )


def write_synthetic_csr(path: Path, missing_train_candidate: bool = False) -> None:
    path.mkdir(parents=True, exist_ok=True)
    meta = {
        "splitIds": {"train": 0, "val": 1, "test": 2},
        "poseStrideBytes": DIRECTIONAL_POSE_DTYPE.itemsize,
        "visibleWeightDtype": "float32",
        "visibleWeightSemantics": "synthetic unit weight",
    }
    (path / "dataset_meta.json").write_text(json.dumps(meta), encoding="utf-8")

    poses = np.zeros((3,), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_norm"] = 0.5
    poses["camera_world"] = 0.0
    poses["camera_forward"] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    poses["camera_view"] = np.asarray([1.0, 1.0], dtype=np.float32)
    poses["split"] = np.asarray([0, 0, 2], dtype=np.uint8)
    poses.tofile(path / "poses.bin")

    visible_ids = np.asarray([2 if missing_train_candidate else 0, 1, 2], dtype=np.uint32)
    np.asarray([0, 1, 2, 3], dtype=np.uint64).tofile(path / "visible_offsets.bin")
    visible_ids.tofile(path / "visible_ids.bin")
    np.ones((3,), dtype=np.float32).tofile(path / "visible_weights.bin")

    # Train candidates are {0, 1} and {1, 2}; the test row is {0, 2}.
    np.asarray([0, 2, 4, 6], dtype=np.uint64).tofile(path / "candidate_offsets.bin")
    np.asarray([0, 1, 1, 2, 0, 2], dtype=np.uint32).tofile(path / "candidate_ids.bin")


def make_batch() -> dict[str, np.ndarray]:
    identity_mvp = np.eye(4, dtype=np.float32).reshape(-1)
    instance_ids = np.asarray([2, 0, 1, 1, 0], dtype=np.int64)
    return {
        "camera": np.full((5, 3), 0.5, dtype=np.float32),
        "camera_world": np.zeros((5, 3), dtype=np.float32),
        "camera_view": np.tile(np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32), (5, 1)),
        "instance": instance_ids,
        "target": np.asarray([1, 0, 0, 0, 1], dtype=np.float32),
        "visible_weights": np.asarray([1, 0, 0, 0, 1], dtype=np.float32),
        "pose_offsets": np.asarray([0, 3, 5], dtype=np.int64),
        "mvp": np.tile(identity_mvp, (5, 1)),
    }



class VisibilityBaselineRunnerTests(unittest.TestCase):
    def test_v4_runner_accepts_empty_viewcell_candidates_without_model_call(self) -> None:
        class RejectingModel:
            def compute_logits_with_aux(self, *_args, **_kwargs):
                raise AssertionError("the model must not run for an empty candidate set")

        runner = PvsV4Runner(
            "empty-viewcell",
            "bounded-relation-survival-moment-v4",
            RejectingModel(),
            np.zeros((3, 6), dtype=np.float32),
            np.arange(3, dtype=np.int64),
            {"sceneBounds": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            {},
            0.5,
            torch.device("cpu"),
            runtime_features=torch.zeros((3, 124), dtype=torch.float32),
        )

        predicted, result = runner.predict_viewcell_ids(
            np.zeros((3,), dtype=np.float32),
            np.zeros((3,), dtype=np.float32),
            np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
            np.empty((0,), dtype=np.uint32),
            query_center_world=np.zeros((3,), dtype=np.float32),
            viewcell_radius_m=1.0,
        )

        self.assertEqual(predicted.dtype, np.uint32)
        self.assertEqual(predicted.size, 0)
        self.assertEqual(result.scores.size, 0)
        self.assertEqual(result.forward_ms, 0.0)
        self.assertEqual(result.total_ms, 0.0)

    def test_bounded_relation_runner_preserves_pose_offsets(self) -> None:
        class RecordingModel:
            def __init__(self) -> None:
                self.pose_offsets = None

            def compute_logits_with_aux(self, camera, *_args, **kwargs):
                self.pose_offsets = kwargs.get("pose_offsets")
                count = int(camera.shape[0])
                zero = torch.zeros((count, 1), dtype=torch.float32)
                return zero, {
                    "utility_logits": zero,
                    "download_logits": zero,
                    "survival_occlusion_probability": zero,
                }

        model = RecordingModel()
        runner = PvsV4Runner(
            "tail-residual",
            "bounded-relation-survival-moment-v4",
            model,
            np.zeros((3, 6), dtype=np.float32),
            np.arange(3, dtype=np.int64),
            {"sceneBounds": {"min": [0, 0, 0], "max": [1, 1, 1]}},
            {},
            0.5,
            torch.device("cpu"),
            runtime_features=torch.zeros((3, 124), dtype=torch.float32),
        )
        batch = make_batch()
        batch["query_center_world"] = np.zeros((5, 3), dtype=np.float32)
        batch["viewcell_radius_m"] = np.ones((5,), dtype=np.float32)

        result = runner.score_batch(batch)

        self.assertEqual(result.scores.shape, (5,))
        self.assertIsNotNone(model.pose_offsets)
        assert model.pose_offsets is not None
        self.assertEqual(model.pose_offsets.cpu().tolist(), [0, 3, 5])

    def test_registered_rules_preserve_candidate_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            dataset_dir = root / "dataset"
            write_runtime_meta(runtime_meta)
            write_synthetic_csr(dataset_dir)

            names = [
                "baseline_keep_all",
                "baseline_static_frequency_train",
                "baseline_camera_distance",
                "baseline_projected_aabb_area",
                "baseline_aabb_ray",
                "baseline_viewcell_bitset_train",
            ]
            self.assertEqual(set(names), set(selected_default_specs(",".join(names))))
            batch = make_batch()
            runners = [
                load_runner(
                    name,
                    selected_default_specs(name)[name],
                    runtime_meta,
                    torch.device("cpu"),
                    dataset_dir=dataset_dir,
                )
                for name in names
            ]

            results = [runner.score_batch(batch) for runner in runners]
            for result in results:
                self.assertEqual(result.scores.shape, batch["instance"].shape)
                self.assertTrue(np.isfinite(result.scores).all())

            np.testing.assert_array_equal(results[0].scores, np.ones((5,), dtype=np.float32))
            # Candidate order is [2, 0, 1, 1, 0]; the score lookup must not sort it.
            np.testing.assert_allclose(results[1].scores, [0.0, 1.0, 0.5, 0.5, 1.0])
            self.assertGreater(results[2].scores[1], results[2].scores[0])
            self.assertGreater(results[3].scores[1], results[3].scores[2])
            # The AABB-ray rule is candidate-preserving and uses the pose's
            # camera/MVP metadata.  The nearby projected boxes score above the
            # clearly off-screen third box without consulting GT fields.
            self.assertGreater(results[4].scores[1], results[4].scores[0])
            self.assertGreater(results[4].scores[2], results[4].scores[0])
            # The nearest train row is the first train row, whose bitset only
            # contains instance 0.  The test-only visible instance 2 is not
            # allowed to enter the baseline prediction.
            np.testing.assert_allclose(results[5].scores, [0.0, 1.0, 0.0, 0.0, 1.0])
            self.assertEqual(runners[5].train_pose_count, 2)
            self.assertEqual(runners[5].bitset_bytes, 2)

            aabb_ray_batch = {
                key: value
                for key, value in batch.items()
                if key not in {"target", "visible_weights"}
            }
            ray_result = runners[4].score_batch(aabb_ray_batch)
            np.testing.assert_allclose(ray_result.scores, results[4].scores)

            candidate_ids = np.asarray([2, 0, 1], dtype=np.uint32)
            predicted, _result = runners[0].predict_ids(
                np.zeros((3,), dtype=np.float32),
                np.zeros((3,), dtype=np.float32),
                np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),
                candidate_ids,
                threshold=0.5,
            )
            np.testing.assert_array_equal(predicted, candidate_ids)

    def test_frequency_reads_train_only_and_rejects_candidate_repairs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            write_runtime_meta(runtime_meta)

            valid_dataset = root / "valid"
            write_synthetic_csr(valid_dataset)
            spec = selected_default_specs("baseline_static_frequency_train")["baseline_static_frequency_train"]
            runner = load_runner(
                "baseline_static_frequency_train",
                spec,
                runtime_meta,
                torch.device("cpu"),
                dataset_dir=valid_dataset,
            )
            # Instance 2 is visible only in the test row, so its train-only
            # frequency remains zero.
            np.testing.assert_allclose(runner.train_visibility_frequency, [1.0, 0.5, 0.0])
            self.assertEqual(runner.train_pose_count, 2)

            invalid_dataset = root / "invalid"
            write_synthetic_csr(invalid_dataset, missing_train_candidate=True)
            with self.assertRaisesRegex(ValueError, "refuses to repair"):
                load_runner(
                    "baseline_static_frequency_train",
                    spec,
                    runtime_meta,
                    torch.device("cpu"),
                    dataset_dir=invalid_dataset,
                )

    def test_projected_area_requires_mvp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            write_runtime_meta(runtime_meta)
            spec = selected_default_specs("baseline_projected_aabb_area")["baseline_projected_aabb_area"]
            runner = load_runner("baseline_projected_aabb_area", spec, runtime_meta, torch.device("cpu"))
            batch = make_batch()
            batch.pop("mvp")
            with self.assertRaisesRegex(RuntimeError, "requires mvp"):
                runner.score_batch(batch)

    def test_aabb_ray_requires_mvp_and_preserves_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            write_runtime_meta(runtime_meta)
            spec = selected_default_specs("baseline_aabb_ray")["baseline_aabb_ray"]
            runner = load_runner("baseline_aabb_ray", spec, runtime_meta, torch.device("cpu"))
            self.assertTrue(runner.requires_mvp)
            self.assertEqual(runner.information_level, "L0_metadata_cold_start")
            self.assertIn("mvp_aabb_area", runner.score_formula)

            batch = make_batch()
            result = runner.score_batch(batch)
            self.assertEqual(result.scores.shape, batch["instance"].shape)
            self.assertTrue(np.isfinite(result.scores).all())

            no_mvp = dict(batch)
            no_mvp.pop("mvp")
            with self.assertRaisesRegex(RuntimeError, "requires mvp"):
                runner.score_batch(no_mvp)

            # A different forward ray changes the score while the candidate
            # order remains unchanged; no target/visible fields are required.
            changed = {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in batch.items()
                if key not in {"target", "visible_weights"}
            }
            changed["camera_view"][:, :3] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            changed_result = runner.score_batch(changed)
            self.assertEqual(changed_result.scores.shape, result.scores.shape)
            self.assertFalse(np.allclose(changed_result.scores, result.scores))

    def test_learned_aabb_ray_checkpoint_uses_shared_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            write_runtime_meta(runtime_meta)
            checkpoint_path = root / "learned_aabb_ray.pt"
            model = AabbRayMLP()
            torch.save(
                {
                    "schema": "neuralstreamweb3d-learned-aabb-ray-v1",
                    "config": {"featureDim": 18, "numInstances": 3, "numGlbs": 3, "sceneDiagonal": 20.0},
                    "model": model.state_dict(),
                },
                checkpoint_path,
            )
            runner = load_runner(
                "learned_aabb_ray",
                {"kind": "learned_aabb_ray", "checkpoint": str(checkpoint_path)},
                runtime_meta,
                torch.device("cpu"),
            )
            self.assertTrue(runner.requires_mvp)
            self.assertEqual(runner.information_level, "L0_metadata_cold_start_learned")
            result = runner.score_batch(make_batch())
            self.assertEqual(result.scores.shape, (5,))
            self.assertTrue(np.isfinite(result.scores).all())


if __name__ == "__main__":
    unittest.main()
