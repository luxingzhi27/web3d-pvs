from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest

import numpy as np

from neural_instance_culling.dataset.build_rvc_viewcell_pose_csr import (
    SPLIT_IDS,
    aggregate_viewcells,
    backed_candidate_camera,
    link_or_copy,
)


def _args() -> argparse.Namespace:
    return argparse.Namespace(
        subpose_groups_per_viewcell=1,
        min_success_subposes=1,
        default_fov_y=66.0,
        default_aspect=16.0 / 9.0,
    )


def _row(split: str | None, subpose_id: int = 0) -> dict[str, object]:
    return {
        "viewcell_id": 7,
        "subpose_id": subpose_id,
        "split": split,
        "viewcell_shape": "horizontal_disk",
        "viewcell_center": [0.0, 0.0, 0.0],
        "viewcell_radius": 2.0,
        "viewcell_forward": [0.0, 0.0, -1.0],
        "camera_pos": [float(subpose_id), 0.0, 0.0],
        "pvs_back_offset": 3.464101615137755,
        "pvs_fov_y": 66.0,
        "aspect": 16.0 / 9.0,
        "visible_component_ids": [1],
        "component_weights": [1.0],
    }


class BuildRvcViewcellPoseCsrTest(unittest.TestCase):
    def test_uses_current_four_way_split_ids(self) -> None:
        self.assertEqual(
            SPLIT_IDS,
            {
                "train": 0,
                "validation": 1,
                "calibration": 2,
                "test": 3,
                "guard": 254,
                "unknown": 255,
            },
        )

    def test_preserves_explicit_formal_split(self) -> None:
        viewcells = aggregate_viewcells([_row("calibration")], _args())
        self.assertEqual(viewcells[0]["split"], "calibration")
        self.assertEqual(viewcells[0]["subpose_pose_indices"].tolist(), [0])
        self.assertEqual(viewcells[0]["subpose_forwards"].shape, (1, 3))
        self.assertEqual(viewcells[0]["subpose_params"].shape, (1, 4))

    def test_rejects_missing_or_obsolete_split_name(self) -> None:
        for split in (None, "val", "unknown"):
            with self.subTest(split=split), self.assertRaises(ValueError):
                aggregate_viewcells([_row(split)], _args())

    def test_rejects_conflicting_splits_within_viewcell(self) -> None:
        rows = [_row("train", 0), _row("test", 1)]
        with self.assertRaises(ValueError):
            aggregate_viewcells(rows, _args())

    def test_candidate_camera_is_backed_from_query_center(self) -> None:
        camera = backed_candidate_camera(
            np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -2.0], dtype=np.float32),
            0.5,
        )
        np.testing.assert_allclose(camera, [1.0, 2.0, 3.5])
        with self.assertRaises(ValueError):
            backed_candidate_camera(np.zeros(3), np.asarray([0.0, 0.0, -1.0]), 0.0)

    def test_candidate_alias_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate.bin"
            target = Path(temporary) / "raw_candidate.bin"
            source.write_bytes(b"candidate")
            link_or_copy(source, target)
            link_or_copy(source, target)
            self.assertEqual(target.read_bytes(), b"candidate")


if __name__ == "__main__":
    unittest.main()
