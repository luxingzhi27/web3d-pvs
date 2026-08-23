from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BENCHMARK_DIR.parent / "model"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(MODEL_DIR))

import model_runners as model_runners_module  # noqa: E402

from model_runners import (  # noqa: E402
    BoundedRelationSurvivalMomentV3Runner,
    _v4_optional_head_constructor_values,
    load_bounded_relation_survival_moment_v4_runner,
    load_runner,
    selected_default_specs,
)
from pose_csr_dataset import DIRECTIONAL_POSE_DTYPE  # noqa: E402
from train_aabb_ray_baseline import AabbRayMLP  # noqa: E402
from triangle_hzb import (  # noqa: E402
    CACHE_SCHEMA,
    DEPTH_ENCODING,
    build_hzb_levels,
    flatten_hzb_levels,
)


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
    np.asarray([0, 2, 4, 6], dtype=np.uint64).tofile(path / "frustum_offsets.bin")
    np.asarray([0, 1, 1, 2, 0, 2], dtype=np.uint32).tofile(path / "frustum_ids.bin")


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


def write_triangle_hzb_cache(path: Path) -> None:
    depth = np.full((4, 4), 0.2, dtype=np.float32)
    levels = build_hzb_levels(depth, width=4, height=4)
    values, descriptors = flatten_hzb_levels(levels)
    path.write_bytes(values.tobytes())
    metadata = {
        "schema": CACHE_SCHEMA,
        "depthEncoding": DEPTH_ENCODING,
        "cameraFar": 10.0,
        "levelDescriptors": descriptors,
        "valueCount": int(values.size),
        "poses": [
            {
                "poseIndex": 0,
                "cameraWorld": [0.0, 0.0, 0.0],
                "cameraForward": [0.0, 0.0, -1.0],
                "valueOffset": 0,
                "valueCount": int(values.size),
            },
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def write_v4_runner_fixture(root: Path, bundle_patch: dict | None = None) -> tuple[Path, Path, Path, Path, dict]:
    runtime_meta = root / "runtimeVisibilityMeta.json"
    write_runtime_meta(runtime_meta)
    config = {
        "runtimeSchema": "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4",
        "numInstances": 3,
        "numGlbs": 3,
        "geometryDim": 96,
        "runtimeFeatureDim": 124,
        "runtimeHeadInputDim": 130,
        "boundarySummaryDim": 8,
        "lowRankSummaryDim": 4,
        "hiddenDim": 11,
        "relationHiddenDim": 7,
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_extrema",
        "instanceCalibration": {
            "mode": "residual",
            "shape": [4, 7],
            "maximumAbsoluteResidual": 4.0,
            "sparseInstancePenalty": 3.0,
            "runtimeExport": "fused coefficients only",
        },
        "depthNormalization": {
            "definition": "depth",
            "q01": 0.0,
            "q99": 1.0,
            "epsilon": 1e-6,
            "sourceSplit": "train",
        },
        "frequency": {"count": 16, "units": "cycles", "maxNormCycles": 8.0},
        "viewcellRegionConditionedVisibility": {
            "enabled": True,
            "regionInputDim": 27,
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": 17,
            "hiddenInputDim": 11,
            "projectionDim": 16,
            "fusionDim": 48,
            "headHiddenDim": 16,
            "activation": "SiLU",
            "fusion": (
                "concat(region_projection, hidden_projection, "
                "region_projection * hidden_projection)"
            ),
            "output": "unbounded additive main visibility logit",
            "outputInitialization": "zero weight and bias",
            "centering": "pose_mean",
            "poseReduction": "candidate_mean_per_pose",
            "runtimeReduction": "candidate_mean_per_pose",
            "boundedCorrection": False,
        },
    }
    geometry = {"path": "checkpoint-geometry.bin", "shape": [3, 96], "dtype": "float16"}
    checkpoint = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
        "runtimeSchema": config["runtimeSchema"],
        "testRead": False,
        "protocol": {
            "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4",
            "testRead": False,
        },
        "modelConfig": config,
        "geometry": geometry,
        "modelState": {},
        "best": {"selection": {"threshold": 0.25}},
    }
    checkpoint_path = root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)

    bundle = root / "bundle"
    bundle.mkdir()
    table_path = bundle / "instance_runtime_features_fp16.bin"
    np.zeros((3, 124), dtype="<f2").tofile(table_path)
    bundle_config = {
        key: config[key]
        for key in (
            "runtimeSchema",
            "numInstances",
            "numGlbs",
            "geometryDim",
            "runtimeFeatureDim",
            "runtimeHeadInputDim",
            "boundarySummaryDim",
            "lowRankSummaryDim",
            "hiddenDim",
            "relationSource",
            "spectralMode",
        )
    }
    bundle_config["instanceCalibration"] = dict(config["instanceCalibration"])
    bundle_config["depthNormalization"] = dict(config["depthNormalization"])
    bundle_config["frequency"] = dict(config["frequency"])
    bundle_meta = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4",
        "testRead": False,
        "checkpointSchema": checkpoint["schema"],
        "modelSchema": config["runtimeSchema"],
        "numInstances": 3,
        "numGlbs": 3,
        "modelConfig": bundle_config,
        "fixedTable": {
            "file": table_path.name,
            "dtype": "float16",
            "shape": [3, 124],
            "byteLength": int(table_path.stat().st_size),
        },
        "runtimeFeatureSource": {
            "source": "checkpoint.geometryFeatures_plus_survivalCoefficients",
            "geometry": {"shape": [3, 96], "dtype": "float16"},
        },
        "threshold": 0.25,
    }
    if bundle_patch:
        bundle_meta.update(bundle_patch)
    (bundle / "model_meta.json").write_text(json.dumps(bundle_meta), encoding="utf-8")

    calibration_path = root / "calibration_ready_summary.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4",
                "status": "safe",
                "bestSafe": {"selection": {"threshold": 0.25}},
                "testRead": False,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path, runtime_meta, table_path, calibration_path, config


class VisibilityBaselineRunnerTests(unittest.TestCase):
    def test_v4_runner_reconstructs_checkpoint_widths_and_region_head_from_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            (
                checkpoint_path,
                runtime_meta_path,
                table_path,
                calibration_path,
                _config,
            ) = write_v4_runner_fixture(Path(temp_dir))
            captured: dict[str, object] = {}

            class CapturingModel:
                def __init__(self, **kwargs: object) -> None:
                    captured.update(kwargs)

                def to(self, _device: object) -> "CapturingModel":
                    return self

                def set_instance_world_aabbs(self, _value: object) -> None:
                    return None

                def set_instance_to_glb(self, _value: object) -> None:
                    return None

                def load_state_dict(self, _state: object, strict: bool = True) -> None:
                    self.strict = strict

                def eval(self) -> "CapturingModel":
                    return self

            with mock.patch.object(
                model_runners_module,
                "BoundedRelationSurvivalMomentModel",
                CapturingModel,
            ):
                runner = load_bounded_relation_survival_moment_v4_runner(
                    "synthetic-v4",
                    checkpoint_path,
                    runtime_meta_path,
                    table_path,
                    calibration_path,
                    torch.device("cpu"),
                )

            self.assertEqual(captured["relation_hidden_dim"], 7)
            self.assertEqual(captured["hidden_dim"], 11)
            self.assertTrue(captured["viewcell_region_conditioned_visibility_enabled"])
            self.assertEqual(
                captured["viewcell_region_conditioned_visibility_centering"],
                "pose_mean",
            )
            self.assertEqual(tuple(runner.runtime_features.shape), (3, 124))
            self.assertEqual(runner.runtime_bundle_meta["fixedTable"]["shape"], [3, 124])

    def test_v4_runner_rejects_non_test_free_or_mismatched_bundle_manifest(self) -> None:
        cases = (
            ({"testRead": True}, "not test-free"),
            ({"numGlbs": 2}, "GLB count"),
            ({"modelConfig": {"hiddenDim": 99}}, "modelConfig field disagrees"),
        )
        for bundle_patch, message in cases:
            with self.subTest(bundle_patch=bundle_patch), tempfile.TemporaryDirectory() as temp_dir:
                (
                    checkpoint_path,
                    runtime_meta_path,
                    table_path,
                    calibration_path,
                    _config,
                ) = write_v4_runner_fixture(Path(temp_dir), bundle_patch=bundle_patch)
                with self.assertRaisesRegex(ValueError, message):
                    load_bounded_relation_survival_moment_v4_runner(
                        "synthetic-v4",
                        checkpoint_path,
                        runtime_meta_path,
                        table_path,
                        calibration_path,
                        torch.device("cpu"),
                    )

    def test_v4_runner_old_checkpoint_defaults_viewcell_extreme_visibility_off(self) -> None:
        values = _v4_optional_head_constructor_values({})
        self.assertFalse(values["viewcell_extreme_visibility_enabled"])
        self.assertIsNone(values["dual_probe_rescue"])

    def test_v4_runner_forwards_enabled_dual_probe_rescue_config(self) -> None:
        config = {
            "enabled": True,
            "primary": {"threshold": 0.1},
            "coverage": {"threshold": 0.2},
        }
        values = _v4_optional_head_constructor_values(
            {"dualProbeRescue": config}
        )
        self.assertEqual(values["dual_probe_rescue"], config)
        self.assertIsNot(values["dual_probe_rescue"], config)

    def test_v4_runner_rejects_non_object_dual_probe_rescue_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "dual-probe-rescue"):
            _v4_optional_head_constructor_values({"dualProbeRescue": []})

    def test_v4_runner_reconstructs_enabled_viewcell_extreme_visibility(self) -> None:
        values = _v4_optional_head_constructor_values(
            {
                "viewcellExtremeVisibility": {
                    "enabled": True,
                    "inputDim": 27,
                    "queryAuxKey": "viewcell_extreme_features",
                    "queryAuxFeatureDim": 17,
                    "projectionDim": 16,
                    "activation": "SiLU",
                    "poseReduction": "none",
                    "boundedCorrection": False,
                }
            }
        )
        self.assertTrue(values["viewcell_extreme_visibility_enabled"])

    def test_v4_runner_rejects_invalid_enabled_viewcell_extreme_visibility(self) -> None:
        with self.assertRaisesRegex(ValueError, "viewcell-extreme-visibility"):
            _v4_optional_head_constructor_values(
                {
                    "viewcellExtremeVisibility": {
                        "enabled": True,
                        "inputDim": 26,
                        "queryAuxKey": "viewcell_extreme_features",
                        "queryAuxFeatureDim": 17,
                        "projectionDim": 16,
                        "activation": "SiLU",
                        "poseReduction": "none",
                        "boundedCorrection": False,
                    }
                }
            )

    def test_v4_runner_old_checkpoint_defaults_region_conditioned_visibility_off(self) -> None:
        values = _v4_optional_head_constructor_values({})
        self.assertFalse(
            values["viewcell_region_conditioned_visibility_enabled"]
        )
        self.assertEqual(
            values["viewcell_region_conditioned_visibility_centering"], "none"
        )

    def test_v4_runner_reconstructs_enabled_region_conditioned_visibility(self) -> None:
        values = _v4_optional_head_constructor_values(
            {
                "hiddenDim": 64,
                "viewcellRegionConditionedVisibility": {
                    "enabled": True,
                    "regionInputDim": 27,
                    "queryAuxKey": "viewcell_extreme_features",
                    "queryAuxFeatureDim": 17,
                    "hiddenInputDim": 64,
                    "projectionDim": 16,
                    "fusionDim": 48,
                    "headHiddenDim": 16,
                    "activation": "SiLU",
                    "fusion": (
                        "concat(region_projection, hidden_projection, "
                        "region_projection * hidden_projection)"
                    ),
                    "output": "unbounded additive main visibility logit",
                    "outputInitialization": "zero weight and bias",
                    "centering": "pose_mean",
                    "poseReduction": "candidate_mean_per_pose",
                    "runtimeReduction": "candidate_mean_per_pose",
                    "boundedCorrection": False,
                },
            }
        )
        self.assertTrue(
            values["viewcell_region_conditioned_visibility_enabled"]
        )
        self.assertEqual(
            values["viewcell_region_conditioned_visibility_centering"],
            "pose_mean",
        )

    def test_v4_runner_rejects_invalid_region_conditioned_visibility_schema(self) -> None:
        base = {
            "enabled": True,
            "regionInputDim": 27,
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": 17,
            "hiddenInputDim": 64,
            "projectionDim": 16,
            "fusionDim": 48,
            "headHiddenDim": 16,
            "activation": "SiLU",
            "fusion": (
                "concat(region_projection, hidden_projection, "
                "region_projection * hidden_projection)"
            ),
            "output": "unbounded additive main visibility logit",
            "outputInitialization": "zero weight and bias",
            "centering": "pose_mean",
            "poseReduction": "candidate_mean_per_pose",
            "runtimeReduction": "candidate_mean_per_pose",
            "boundedCorrection": False,
        }
        for field, value in (
            ("regionInputDim", 26),
            ("centering", "batch_mean"),
            ("poseReduction", "none"),
            ("runtimeReduction", "none"),
        ):
            with self.subTest(field=field):
                config = dict(base)
                config[field] = value
                with self.assertRaisesRegex(
                    ValueError, "viewcell-region-conditioned-visibility"
                ):
                    _v4_optional_head_constructor_values(
                        {"viewcellRegionConditionedVisibility": config}
                    )

    def test_v4_runner_reconstructs_enabled_tail_residual_head(self) -> None:
        values = _v4_optional_head_constructor_values(
            {
                "boundaryTailResidual": {
                    "enabled": True,
                    "hiddenDim": 32,
                    "projectionDim": 24,
                    "maximumAbsoluteResidual": 1.5,
                    "centering": "none",
                    "shortcut": "region_linear",
                    "fusionMode": "affine_region",
                    "outputInitializationStd": 0.002,
                }
            }
        )
        self.assertEqual(values["boundary_tail_residual_hidden_dim"], 32)
        self.assertEqual(values["boundary_tail_residual_projection_dim"], 24)
        self.assertEqual(values["boundary_tail_residual_max_abs"], 1.5)
        self.assertEqual(values["boundary_tail_residual_centering"], "none")
        self.assertEqual(
            values["boundary_tail_residual_shortcut"], "region_linear"
        )
        self.assertEqual(
            values["boundary_tail_residual_fusion"], "affine_region"
        )
        self.assertEqual(
            values["boundary_tail_residual_output_init_std"], 0.002
        )
        self.assertEqual(values["boundary_opportunity_hidden_dim"], 0)

    def test_v4_runner_rejects_invalid_enabled_optional_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "boundary-tail-residual"):
            _v4_optional_head_constructor_values(
                {
                    "boundaryTailResidual": {
                        "enabled": True,
                        "hiddenDim": 0,
                        "projectionDim": 24,
                        "maximumAbsoluteResidual": 1.0,
                    }
                }
            )

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
        runner = BoundedRelationSurvivalMomentV3Runner(
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
                "baseline_aabb_hzb",
                "baseline_viewcell_bitset_train",
            ]
            self.assertTrue(set(names).issubset(selected_default_specs(",".join(names))))
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
            self.assertEqual(runners[5].decision_mode, "threshold")
            self.assertTrue(np.isfinite(results[5].scores).all())
            # The nearest train row is the first train row, whose bitset only
            # contains instance 0.  The test-only visible instance 2 is not
            # allowed to enter the baseline prediction.
            np.testing.assert_allclose(results[6].scores, [0.0, 1.0, 0.0, 0.0, 1.0])
            self.assertEqual(runners[6].train_pose_count, 2)
            self.assertEqual(runners[6].bitset_bytes, 2)

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

    def test_triangle_hzb_cache_uses_triangle_depth_and_exact_pose_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_meta = root / "runtimeVisibilityMeta.json"
            write_runtime_meta(runtime_meta)
            cache_path = root / "triangle_hzb_values.bin"
            write_triangle_hzb_cache(cache_path)
            runner = load_runner(
                "baseline_triangle_hzb",
                {"kind": "triangle_hzb", "cache": str(cache_path)},
                runtime_meta,
                torch.device("cpu"),
            )
            batch = {
                "camera_world": np.zeros((2, 3), dtype=np.float32),
                "camera_view": np.tile(np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32), (2, 1)),
                "instance": np.asarray([0, 2], dtype=np.int64),
                "pose_offsets": np.asarray([0, 2], dtype=np.int64),
                "pose_indices": np.asarray([0], dtype=np.int64),
            }
            result = runner.score_batch(batch)
            # Instance 0 is in front of the cached depth and instance 2 is
            # behind it.  The score is a geometric HZB query, not a GT lookup.
            self.assertEqual(result.scores.shape, (2,))
            self.assertGreater(result.scores[0], 0.99)
            self.assertLess(result.scores[1], 0.01)
            self.assertEqual(runner.information_level, "L2_warm_triangle_depth")

            missing_indices = dict(batch)
            missing_indices.pop("pose_indices")
            with self.assertRaisesRegex(RuntimeError, "exact pose_indices"):
                runner.score_batch(missing_indices)


if __name__ == "__main__":
    unittest.main()
