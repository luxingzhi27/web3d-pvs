from __future__ import annotations

import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BENCHMARK_DIR.parent / "model"
sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(MODEL_DIR))

from evaluate_viewcell_image_per import ViewcellDataset, resolve_threshold  # noqa: E402
from pose_csr_dataset import DIRECTIONAL_POSE_DTYPE, PoseCSRDataset  # noqa: E402


def _write_fixture(root: Path, *, canonical_center: bool = False) -> tuple[Path, Path]:
    viewcell = root / "viewcell_source"
    pose_csr = root / "pose_csr"
    viewcell.mkdir()
    pose_csr.mkdir()

    count = 3
    forward = np.asarray(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    camera_world = np.asarray(
        [[0.0, 2.0, 3.0], [5.0, 2.0, 0.0], [0.0, 2.0, -4.0]],
        dtype=np.float32,
    )
    centers = camera_world if canonical_center else camera_world + forward * np.float32(3.4641016)

    def write(name: str, values: np.ndarray) -> None:
        values.tofile(viewcell / name)

    write("viewcell_ids.bin", np.arange(count, dtype="<u4"))
    write("viewcell_centers.bin", centers.astype("<f4"))
    write("viewcell_forwards.bin", forward.astype("<f4"))
    write("viewcell_params.bin", np.zeros((count, 8), dtype="<f4"))
    write("viewcell_category_ids.bin", np.zeros((count,), dtype="u1"))
    write("viewcell_split_ids.bin", np.asarray([0, 1, 2], dtype="u1"))
    write("visible_offsets.bin", np.arange(count + 1, dtype="<u8"))
    write("visible_ids.bin", np.arange(count, dtype="<u4"))
    write("visible_weights.bin", np.ones((count,), dtype="<f4"))
    write("subpose_offsets.bin", np.arange(count + 1, dtype="<u8"))
    write("subpose_pose_indices.bin", np.arange(count, dtype="<u4"))
    write("subpose_camera_pos.bin", camera_world.astype("<f4"))
    write("subpose_camera_forward.bin", forward.astype("<f4"))
    write("subpose_params.bin", np.zeros((count, 4), dtype="<f4"))
    (viewcell / "dataset_meta.json").write_text(
        json.dumps({"viewcellCount": count, "splitIds": {"train": 0, "val": 1, "test": 2}}),
        encoding="utf-8",
    )

    poses = np.zeros((count,), dtype=DIRECTIONAL_POSE_DTYPE)
    poses["camera_world"] = camera_world
    poses["camera_forward"] = forward
    poses["camera_norm"] = forward
    poses["camera_view"] = np.asarray([[1.0, 1.0]] * count, dtype=np.float32)
    poses["split"] = np.asarray([0, 2, 3], dtype=np.uint8)
    poses.tofile(pose_csr / "poses.bin")
    np.asarray([0, 1, 2, 3], dtype="<u8").tofile(pose_csr / "visible_offsets.bin")
    np.arange(count, dtype="<u4").tofile(pose_csr / "visible_ids.bin")
    np.ones((count,), dtype="<f4").tofile(pose_csr / "visible_weights.bin")
    np.asarray([0, 1, 2, 3], dtype="<u8").tofile(pose_csr / "candidate_offsets.bin")
    np.arange(count, dtype="<u4").tofile(pose_csr / "candidate_ids.bin")
    pose_meta = {
        "poseCount": count,
        "poseStrideBytes": 64,
        "numInstances": count,
        "splitIds": {"train": 0, "validation": 1, "calibration": 2, "test": 3},
    }
    if canonical_center:
        pose_meta["cameraSemantics"] = "camera_world is the canonical viewcell plan center"
    (pose_csr / "dataset_meta.json").write_text(
        json.dumps(pose_meta),
        encoding="utf-8",
    )
    return viewcell, pose_csr


class ViewcellSplitSourceTests(unittest.TestCase):
    def test_current_v4_calibration_summary_resolves_frozen_safe_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            summary = Path(temp_dir) / "calibration.json"
            summary.write_text(
                json.dumps(
                    {
                        "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4",
                        "testRead": False,
                        "weightedRecallFloor": 0.99,
                        "weightedRecallLowerConfidenceBoundFloor": 0.99,
                        "bestSafe": {
                            "safe": True,
                            "threshold": 0.68,
                            "selection": {
                                "threshold": 0.68,
                                "agg_weighted_recall": 0.997,
                                "aggregateWeightedRecallLowerConfidenceBound": 0.994,
                                "pose_recall": 0.97,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                threshold=None,
                threshold_policy="weighted_precision",
                target_weighted_recall=0.99,
                minimum_pose_recall=None,
            )
            threshold, source = resolve_threshold(
                args,
                {"eval_summary": str(summary)},
                SimpleNamespace(threshold=0.1),
            )
            self.assertAlmostEqual(threshold, 0.68)
            self.assertEqual(source["selectionSplit"], "calibration")
            self.assertFalse(source["testRead"])

    def test_pose_csr_split_labels_are_aligned_and_support_formal_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewcell_dir, pose_dir = _write_fixture(Path(temp_dir))
            viewcells = ViewcellDataset(viewcell_dir)
            pose = PoseCSRDataset(pose_dir, num_instances=3)

            calibration = viewcells.split_indices(
                "calibration", 0, split_source="pose_csr", pose_dataset=pose
            )
            self.assertEqual(calibration.tolist(), [1])
            self.assertTrue(viewcells.split_alignment["validated"])
            self.assertLess(viewcells.split_alignment["maxCenterResidual"], 1e-3)

            exploratory_val = viewcells.split_indices("val", 0, split_source="viewcell")
            self.assertEqual(exploratory_val.tolist(), [1])

    def test_pose_csr_alignment_rejects_mismatched_forward_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewcell_dir, pose_dir = _write_fixture(Path(temp_dir))
            pose_path = pose_dir / "poses.bin"
            poses = np.fromfile(pose_path, dtype=DIRECTIONAL_POSE_DTYPE)
            poses[2]["camera_forward"] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
            poses.tofile(pose_path)

            viewcells = ViewcellDataset(viewcell_dir)
            pose = PoseCSRDataset(pose_dir, num_instances=3)
            with self.assertRaisesRegex(ValueError, "camera_forward differs"):
                viewcells.split_indices("test", 0, split_source="pose_csr", pose_dataset=pose)

    def test_canonical_viewcell_center_semantics_are_explicitly_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            viewcell_dir, pose_dir = _write_fixture(Path(temp_dir), canonical_center=True)
            viewcells = ViewcellDataset(viewcell_dir)
            pose = PoseCSRDataset(pose_dir, num_instances=3)

            selected = viewcells.split_indices(
                "test", 0, split_source="pose_csr", pose_dataset=pose
            )
            self.assertEqual(selected.tolist(), [2])
            self.assertEqual(
                viewcells.split_alignment["centerRelation"],
                "source view-cell center equals canonical camera_world",
            )


if __name__ == "__main__":
    unittest.main()
