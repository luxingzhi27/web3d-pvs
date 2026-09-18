from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from neural_instance_culling.benchmark.v5.calibration import calibrate_target
from neural_instance_culling.benchmark.v5.contracts import (
    ALL_VARIANTS,
    LOSO_VARIANTS,
    V5Run,
    validate_matrix,
)
from neural_instance_culling.benchmark.v5.inference import (
    COMPILED_GEOMETRY_SCHEMA,
    InferenceSceneSpec,
    compile_geometry,
    export_score_bundle,
    infer_split,
    load_checkpoint,
    load_compiled_geometry,
)
from neural_instance_culling.benchmark.v5.metrics import evaluate_scene
from neural_instance_culling.benchmark.v5.scan_selection import (
    select_scan_candidates,
    summarize_single_shared_run,
)
from neural_instance_culling.benchmark.v5.score_bundle import (
    BUNDLE_SCHEMA,
    ColumnarScoreSidecar,
    ColumnarScoreSidecarWriter,
    FrozenPoseCSR,
    SceneScores,
)
from neural_instance_culling.model.pose_csr_dataset import DIRECTIONAL_POSE_DTYPE
from neural_instance_culling.model.v5.core import GCOFPVSV5
from neural_instance_culling.model.v5.train import CHECKPOINT_SCHEMA


SURFACE_HEADER = struct.Struct("<4sHHII")


def _write_pose_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    poses = np.zeros((4,), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_world"] = np.asarray([[0.0, 0.0, 2.0]] * 4, dtype=np.float32)
    poses["camera_norm"] = poses["camera_world"]
    poses["camera_forward"] = np.asarray([[0.0, 0.0, -1.0]] * 4, dtype=np.float32)
    poses["camera_view"] = np.asarray([[1.0, 1.0]] * 4, dtype=np.float32)
    poses["split"] = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    poses.tofile(root / "poses.bin")
    np.asarray([0, 1, 2, 3, 4], dtype=np.uint64).tofile(root / "visible_offsets.bin")
    np.asarray([0, 0, 0, 1], dtype=np.uint32).tofile(root / "visible_ids.bin")
    np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32).tofile(root / "visible_weights.bin")
    np.asarray([0, 2, 4, 6, 8], dtype=np.uint64).tofile(root / "candidate_offsets.bin")
    np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.uint32).tofile(root / "candidate_ids.bin")
    np.asarray([[0.0, 0.0, 0.0]] * 4, dtype=np.float32).tofile(root / "query_center_world.bin")
    np.asarray([[0.0, 0.0, 2.0]] * 4, dtype=np.float32).tofile(root / "candidate_camera_world.bin")
    np.asarray([1.0] * 4, dtype=np.float32).tofile(root / "viewcell_radius_m.bin")
    (root / "dataset_meta.json").write_text(
        json.dumps(
            {
                "schema": "pose-csr-explicit-four-way-split-v1",
                "numInstances": 2,
                "poseStrideBytes": 64,
                "splitIds": {"train": 0, "calibration": 1, "validation": 2, "test": 3},
                "splitCounts": {"train": 1, "calibration": 1, "validation": 1, "test": 1},
                "visibleWeightDtype": "float32",
                "visibleWeightSemantics": "fixture weight",
            }
        ),
        encoding="utf-8",
    )


def _write_runtime_and_compiled_assets(root: Path) -> tuple[Path, Path]:
    runtime = root / "runtimeVisibilityMeta.json"
    runtime.write_text(
        json.dumps(
            {
                "instanceCount": 2,
                "componentRecords": [
                    {"componentGlobalId": 0, "globalGlbId": 0, "bounds": {"min": [-1, -1, -1], "max": [0, 0, 0]}},
                    {"componentGlobalId": 1, "globalGlbId": 1, "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]}},
                ],
            }
        ),
        encoding="utf-8",
    )
    compiled = root / "compiled"
    surface = compiled / "surface"
    relation = compiled / "relation"
    surface.mkdir(parents=True)
    relation.mkdir(parents=True)
    points = np.zeros((2, 256, 6), dtype=np.float32)
    points[..., 0] = 0.25
    points[..., 3] = 1.0
    points[..., 4] = 0.0
    points[..., 5] = 0.0
    with (surface / "surface_points_fp32.bin").open("wb") as stream:
        stream.write(SURFACE_HEADER.pack(b"GPV5", 1, 6, 2, 256))
        points.tofile(stream)
    ratios = np.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    with (surface / "size_ratios_fp32.bin").open("wb") as stream:
        stream.write(SURFACE_HEADER.pack(b"GPV5", 1, 3, 2, 1))
        ratios.tofile(stream)
    np.asarray([[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0]], dtype=np.float32).tofile(surface / "aabb_min_fp32.npy")
    np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32).tofile(surface / "aabb_max_fp32.npy")
    (surface / "surface_manifest.json").write_text(
        json.dumps(
            {
                "schema": "pvs-v5-local-surface-samples-v1",
                "numUnits": 2,
                "degenerateUnitIds": [],
                "files": {"points": "surface_points_fp32.bin", "sizeRatios": "size_ratios_fp32.bin"},
            }
        ),
        encoding="utf-8",
    )
    source = np.full((2, 12, 8), -1, dtype=np.int64)
    valid = np.zeros((2, 12, 8), dtype=bool)
    edge = np.zeros((2, 12, 8, 8), dtype=np.float32)
    np.save(relation / "source_ids_int64.npy", source)
    np.save(relation / "valid_mask_bool.npy", valid)
    np.save(relation / "edge_features_fp32.npy", edge)
    (relation / "relation_manifest.json").write_text(
        json.dumps(
            {
                "schema": "pvs-geometry-proxy-relation-csr-v1",
                "numUnits": 2,
                "usesVisibilityLabels": False,
                "source": "geometry_only",
            }
        ),
        encoding="utf-8",
    )
    return runtime, compiled


def _checkpoint(path: Path) -> Path:
    model = GCOFPVSV5("FULL")
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "modelConfig": model.config,
        "model": model.state_dict(),
        "config": {
            "protocol": "shared",
            "variant": "FULL",
            "seed": 3,
            "realSceneIds": ["fixture"],
            "heldOutScene": None,
        },
        "testRead": False,
    }
    torch.save(payload, path)
    return path


class ColumnarScoreBundleTests(unittest.TestCase):
    def test_parameter_scan_uses_registered_lexicographic_order(self) -> None:
        def summary(name: str, strict: int, cnor: float) -> dict:
            rows = []
            for index in range(5):
                qualification = "strict_lcb_target" if index < strict else "mean_target"
                rows.append({
                    "protocol": "shared",
                    "variant": "FULL",
                    "seed": 0,
                    "scene": f"scene_{index}",
                    "split": "validation",
                    "threshold_mode": "target_calibrated",
                    "qualification": qualification,
                    "selection_split": "calibration",
                    "selection_test_read": False,
                    "test_read": False,
                    "metrics": {
                        "weighted_recall_lcb": 0.99 + index * 1e-4,
                        "cnor": cnor,
                        "useful_cull": 0.5,
                        "pred_over_gt": 1.2,
                    },
                })
            value = summarize_single_shared_run(rows)
            value["name"] = name
            return value

        matrix = {
            "schema": "gcof-pvs-v5-parameter-scan-matrix-v1",
            "phase": "pilot",
            "selectionRule": ["strict_lcb_scene_count", "mean_target_scene_count"],
            "testRead": False,
            "runs": [{"name": "a"}, {"name": "b"}, {"name": "c"}],
        }
        selected = select_scan_candidates(
            matrix,
            {
                "a": summary("a", 4, 0.9),
                "b": summary("b", 5, 0.1),
                "c": summary("c", 4, 0.95),
            },
            top_k=2,
        )
        self.assertEqual(selected["selectedConfigurations"], ["b", "c"])
        self.assertFalse(selected["testRead"])

    def test_non_renderable_units_are_excluded_from_offsets_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose_dataset = root / "pose_csr"
            _write_pose_dataset(pose_dataset)
            writer = ColumnarScoreSidecarWriter(
                root / "scores",
                scene="fixture",
                split="validation",
                pose_dataset=pose_dataset,
                num_instances=2,
                excluded_unit_ids=[1],
            )
            writer.append_pose(2, np.asarray([0.25], dtype=np.float32), candidate_count=1)
            manifest = writer.close()
            sidecar = ColumnarScoreSidecar(manifest)
            csr = FrozenPoseCSR(pose_dataset, scene="fixture", num_instances=2)
            validation = sidecar.validate_against_pose_csr(csr)
            self.assertEqual(validation.candidate_count, 1)
            record = next(iter(SceneScores(
                scene="fixture",
                pose_dataset=pose_dataset,
                num_instances=2,
                sidecars={"validation": manifest},
            ).records("validation")))
            self.assertEqual(record.candidate_ids.tolist(), [0])
            self.assertEqual(record.targets.tolist(), [1])

    def test_checkpoint_geometry_inference_and_streaming_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose_dataset = root / "pose_csr"
            _write_pose_dataset(pose_dataset)
            runtime, compiled_root = _write_runtime_and_compiled_assets(root)
            checkpoint_path = _checkpoint(root / "checkpoint.pt")
            checkpoint = load_checkpoint(checkpoint_path, device="cpu")
            spec = InferenceSceneSpec(
                scene_id="fixture",
                pose_dataset=pose_dataset,
                runtime_meta=runtime,
                compiled_dir=compiled_root,
                viewcell_shape="horizontal_disk",
                viewcell_half_extent_m=(0.5, 0.5, 0.0),
                camera_clip_m=(0.01, 1000.0),
            )
            geometry_manifest = compile_geometry(
                checkpoint,
                spec,
                root / "geometry",
                device="cpu",
                geometry_chunk_size=1,
            )
            geometry_payload = json.loads(geometry_manifest.read_text(encoding="utf-8"))
            self.assertEqual(geometry_payload["schema"], COMPILED_GEOMETRY_SCHEMA)
            compiled = load_compiled_geometry(root / "geometry", expected_scene="fixture", expected_variant="FULL")
            validation_sidecar = infer_split(
                checkpoint,
                spec,
                compiled,
                root / "scores" / "validation",
                split="validation",
                device="cpu",
                pose_chunk_size=1,
            )
            self.assertTrue(validation_sidecar.is_file())
            scene = SceneScores(
                scene="fixture",
                pose_dataset=pose_dataset,
                num_instances=2,
                sidecars={"validation": validation_sidecar},
            )
            records = scene.records("validation")
            metrics = evaluate_scene(records, threshold=0.0, bootstrap_replicates=0)
            self.assertEqual(metrics["pose_count"], 1)
            self.assertEqual(metrics["candidate_count"], 2)
            selection = calibrate_target(
                SceneScores(
                    scene="fixture",
                    pose_dataset=pose_dataset,
                    num_instances=2,
                    sidecars={"calibration": infer_split(
                        checkpoint,
                        spec,
                        compiled,
                        root / "scores" / "calibration",
                        split="calibration",
                        device="cpu",
                        pose_chunk_size=1,
                    )},
                ).records("calibration"),
                scene="fixture",
                bootstrap_replicates=0,
            )
            self.assertEqual(selection.selection_split, "calibration")
            with mock.patch(
                "neural_instance_culling.benchmark.v5.inference.registered_scene_specs",
                return_value=(spec,),
            ):
                bundle_manifest = export_score_bundle(
                    checkpoint_path,
                    root / "bundle",
                    registry_path=root / "unused_registry.json",
                    device="cpu",
                    geometry_chunk_size=1,
                    pose_chunk_size=1,
                )
            bundle = V5Run.from_manifest(bundle_manifest)
            self.assertEqual(bundle.records("fixture", "validation").__class__.__name__, "_PoseScoreStream")

    def test_sidecar_is_float32_only_and_rejects_pose_csr_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose_dataset = root / "pose_csr"
            _write_pose_dataset(pose_dataset)
            sidecar_root = root / "sidecar"
            writer = ColumnarScoreSidecarWriter(
                sidecar_root,
                scene="fixture",
                split="validation",
                pose_dataset=pose_dataset,
                num_instances=2,
            )
            writer.append_pose(2, np.asarray([0.2, -0.3], dtype=np.float32), candidate_count=2)
            manifest = writer.close()
            self.assertFalse((sidecar_root / "candidate_ids.bin").exists())
            self.assertFalse((sidecar_root / "targets_uint8.bin").exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["dtypes"]["scores"], "<f4")
            self.assertEqual(payload["shapes"]["scores"], [2])
            sidecar = ColumnarScoreSidecar(manifest)
            csr = FrozenPoseCSR(pose_dataset, scene="fixture", num_instances=2)
            sidecar.validate_against_pose_csr(csr)
            candidate_path = pose_dataset / "candidate_ids.bin"
            np.asarray([0, 1, 0, 1, 0, 0, 0, 1], dtype=np.uint32).tofile(candidate_path)
            with self.assertRaisesRegex(ValueError, "duplicates"):
                sidecar.validate_against_pose_csr(csr)

    def test_test_sidecar_requires_explicit_final_test_permission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(PermissionError):
                ColumnarScoreSidecarWriter(
                    root / "test",
                    scene="fixture",
                    split="test",
                    pose_dataset=root / "pose_csr",
                    num_instances=2,
                )
            writer = ColumnarScoreSidecarWriter(
                root / "test_final",
                scene="fixture",
                split="test",
                pose_dataset=root / "pose_csr",
                num_instances=2,
                allow_test=True,
            )
            writer.append_pose(0, np.asarray([0.0], dtype=np.float32), candidate_count=1)
            manifest = writer.close()
            with self.assertRaises(PermissionError):
                ColumnarScoreSidecar(manifest)
            loaded = ColumnarScoreSidecar(manifest, allow_test=True)
            self.assertTrue(bool(loaded.manifest["testRead"]))

    def test_large_column_stays_memory_mapped_without_json_expansion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "large_fixture"
            writer = ColumnarScoreSidecarWriter(
                root,
                scene="fixture",
                split="validation",
                pose_dataset=Path(directory) / "pose_csr",
                num_instances=2048,
            )
            scores = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
            for pose_index in range(128):
                writer.append_pose(pose_index, scores, candidate_count=1024)
            manifest = writer.close()
            sidecar = ColumnarScoreSidecar(manifest)
            self.assertIsInstance(sidecar.scores, np.memmap)
            self.assertEqual(sidecar.pose_count, 128)
            self.assertEqual(sidecar.candidate_count, 128 * 1024)
            self.assertLess(manifest.stat().st_size, 4096)

    def test_embedded_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": BUNDLE_SCHEMA,
                        "version": 1,
                        "protocol": "shared",
                        "variant": "FULL",
                        "seed": 0,
                        "testRead": False,
                        "checkpoint": {
                            "path": "checkpoint.pt",
                            "schema": CHECKPOINT_SCHEMA,
                            "protocol": "shared",
                            "variant": "FULL",
                            "seed": 0,
                            "testRead": False,
                        },
                        "scoreSplits": ["calibration", "validation"],
                        "scenes": {"fixture": {"calibration": [{"poseId": "0"}]}},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "embedded pose rows"):
                V5Run.from_manifest(path)


class V5MatrixContractTests(unittest.TestCase):
    scenes = ("hkust", "ifcbench", "sponza", "viking", "big_city")

    @staticmethod
    def _scene(scene: str) -> SceneScores:
        return SceneScores(
            scene=scene,
            pose_dataset=Path("/fixture/pose_csr"),
            num_instances=2,
            sidecars={"calibration": Path("/fixture/calibration"), "validation": Path("/fixture/validation")},
        )

    def test_shared_matrix_contract(self) -> None:
        runs = [
            V5Run(
                protocol="shared",
                variant=variant,
                seed=seed,
                scenes={scene: self._scene(scene) for scene in self.scenes},
            )
            for variant in ALL_VARIANTS
            for seed in (1, 2, 3)
        ]
        self.assertEqual(len(validate_matrix(runs, protocol="shared", expected_scenes=self.scenes)), 12)

    def test_loso_matrix_requires_all_folds(self) -> None:
        runs = []
        for variant in LOSO_VARIANTS:
            for seed in (1, 2, 3):
                for held_out in self.scenes:
                    runs.append(
                        V5Run(
                            protocol="loso",
                            variant=variant,
                            seed=seed,
                            scenes={scene: self._scene(scene) for scene in self.scenes},
                            held_out_scene=held_out,
                            source_scenes=tuple(scene for scene in self.scenes if scene != held_out),
                        )
                    )
        self.assertEqual(len(validate_matrix(runs, protocol="loso", expected_scenes=self.scenes)), 45)


if __name__ == "__main__":
    unittest.main()
