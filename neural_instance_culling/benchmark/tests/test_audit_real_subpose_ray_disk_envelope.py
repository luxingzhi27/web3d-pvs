from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_instance_culling.benchmark.audit_real_subpose_ray_disk_envelope import (
    audit_real_subpose_ray_disk_envelope,
    compute_nonlinear_ray_features,
)
from neural_instance_culling.model.pose_csr_dataset import DIRECTIONAL_POSE_DTYPE


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source_viewcell"
    pose_csr = root / "pose_csr"
    source.mkdir()
    pose_csr.mkdir()

    subpose_positions = np.asarray(
        [[0.0, 0.0, 0.0]] * 19 + [[0.6, 0.0, 0.0]],
        dtype="<f4",
    )
    source_meta = {
        "schema": "proxy-viewcell-pvs-dataset-v1",
        "viewcellCount": 1,
        "numInstances": 3,
        "files": {
            "viewcellCenters": "viewcell_centers.bin",
            "viewcellParams": "viewcell_params.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "subposeCameraPos": "subpose_camera_pos.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleHitCounts": "visible_hit_counts.bin",
        },
    }
    (source / "dataset_meta.json").write_text(json.dumps(source_meta), encoding="utf-8")
    np.asarray([[0.0, 0.0, 0.0]], dtype="<f4").tofile(source / "viewcell_centers.bin")
    np.asarray([[1.0, 66.0, 66.0, 66.0, 0.0, 0.0, 0.0, 20.0]], dtype="<f4").tofile(
        source / "viewcell_params.bin"
    )
    np.asarray([0, 20], dtype="<u8").tofile(source / "subpose_offsets.bin")
    subpose_positions.tofile(source / "subpose_camera_pos.bin")
    np.asarray([0, 2], dtype="<u8").tofile(source / "visible_offsets.bin")
    np.asarray([0, 1], dtype="<u4").tofile(source / "visible_ids.bin")
    # Instance 0 is a rare positive (1/20), instance 1 is regular (2/20).
    np.asarray([1, 2], dtype="<u2").tofile(source / "visible_hit_counts.bin")

    poses = np.zeros((1,), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_world"] = [[0.0, 0.0, 0.0]]
    poses["camera_forward"] = [[0.0, 0.0, -1.0]]
    poses["camera_view"] = [[1.0, 1.0]]
    poses["split"] = [0]
    poses.tofile(pose_csr / "poses.bin")
    pose_meta = {
        "schema": "pose-csr-test-fixture",
        "poseCount": 1,
        "numInstances": 3,
        "poseStrideBytes": 64,
        "splitIds": {"train": 0, "calibration": 1, "validation": 2, "test": 3},
        "files": {
            "queryCenterWorld": "query_center_world.bin",
            "viewcellRadiusM": "viewcell_radius_m.bin",
        },
    }
    (pose_csr / "dataset_meta.json").write_text(json.dumps(pose_meta), encoding="utf-8")
    np.asarray([0, 2], dtype="<u8").tofile(pose_csr / "visible_offsets.bin")
    np.asarray([0, 1], dtype="<u4").tofile(pose_csr / "visible_ids.bin")
    np.asarray([1.0, 1.0], dtype="<f4").tofile(pose_csr / "visible_weights.bin")
    np.asarray([0, 3], dtype="<u8").tofile(pose_csr / "frustum_offsets.bin")
    np.asarray([0, 1, 2], dtype="<u4").tofile(pose_csr / "frustum_ids.bin")
    np.asarray([[0.0, 0.0, 0.0]], dtype="<f4").tofile(pose_csr / "query_center_world.bin")
    np.asarray([1.0], dtype="<f4").tofile(pose_csr / "viewcell_radius_m.bin")

    runtime = root / "runtimeVisibilityMeta.json"
    runtime.write_text(
        json.dumps(
            {
                "sceneName": "unit-test",
                "componentRecords": [
                    {
                        "componentGlobalId": instance_id,
                        "globalGlbId": instance_id,
                        "bounds": {"min": [-0.5, -0.5, -10.5], "max": [0.5, 0.5, -9.5]},
                    }
                    for instance_id in range(3)
                ],
            }
        ),
        encoding="utf-8",
    )
    return pose_csr, source, runtime


class RealSubposeRayDiskEnvelopeAuditTests(unittest.TestCase):
    def test_audit_reports_three_semantic_categories_and_never_reads_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pose_csr, source, runtime = _write_fixture(Path(temporary))
            result = audit_real_subpose_ray_disk_envelope(
                pose_csr,
                source,
                runtime,
                "train",
                max_candidates=0,
                seed=7,
            )

        self.assertEqual(result["schema"], "pvs-real-subpose-ray-disk-envelope-audit-v1")
        self.assertIs(result["testRead"], False)
        self.assertEqual(result["sampling"]["selectedPoseCount"], 1)
        self.assertEqual(result["sampling"]["candidateReferenceCount"], 3)
        self.assertEqual(result["sampling"]["gtReferenceCount"], 2)
        self.assertEqual(result["categories"]["rarePositive"]["pairCount"], 1)
        self.assertEqual(result["categories"]["regularPositive"]["pairCount"], 1)
        self.assertEqual(result["categories"]["negative"]["pairCount"], 1)
        self.assertEqual(result["categories"]["all"]["subposeSampleCount"], 60)
        self.assertEqual(len(result["categories"]["rarePositive"]["dimensions"]), 9)
        self.assertEqual(result["sourceViewcell"]["hitCountSource"], "visible_hit_counts.bin")
        self.assertGreater(max(result["categories"]["all"]["maxExcessByDimension"]), 0.0)
        self.assertIn('"testRead": false', json.dumps(result, allow_nan=False))

    def test_candidate_cap_preserves_all_positive_instances(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pose_csr, source, runtime = _write_fixture(Path(temporary))
            with self.assertRaises(ValueError):
                # The fixture has two visible instances, so a cap of one cannot
                # satisfy the audit's explicit positive-preservation contract.
                audit_real_subpose_ray_disk_envelope(
                    pose_csr,
                    source,
                    runtime,
                    "train",
                    max_candidates=1,
                )

    def test_test_split_is_rejected_before_running_the_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            pose_csr, source, runtime = _write_fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "test is refused"):
                audit_real_subpose_ray_disk_envelope(
                    pose_csr,
                    source,
                    runtime,
                    "test",
                )

    def test_nonlinear_feature_helper_is_finite_and_nine_dimensional(self) -> None:
        aabb = np.asarray(
            [-0.5, -0.5, -10.5, 0.5, 0.5, -9.5],
            dtype=np.float32,
        )
        features = compute_nonlinear_ray_features(
            np.asarray([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=np.float32),
            np.asarray([[0.0, 0.0, -1.0, 1.0, 1.0]] * 2, dtype=np.float32),
            np.repeat(aabb[None, :], 2, axis=0),
        )
        self.assertEqual(features.shape, (2, 9))
        self.assertTrue(np.isfinite(features).all())


if __name__ == "__main__":
    unittest.main()
