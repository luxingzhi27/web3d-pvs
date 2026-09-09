from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np


BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

import build_hzb_point60_gt_plan as builder  # noqa: E402


class Point60GTPlanTests(unittest.TestCase):
    @staticmethod
    def _write_fixture(root: Path) -> tuple[Path, Path]:
        dataset = root / "pose_csr"
        dataset.mkdir()
        poses = np.zeros((5,), dtype=builder.DIRECTIONAL_POSE_DTYPE)
        poses["split"] = np.asarray([0, 3, 3, 3, 1], dtype=np.uint8)
        poses.tofile(dataset / "poses.bin")
        (dataset / "dataset_meta.json").write_text(
            json.dumps({
                "schema": "pose-csr-explicit-four-way-split-v1",
                "poseCount": 5,
                "poseStrideBytes": 64,
                "splitIds": {"train": 0, "validation": 1, "calibration": 2, "test": 3},
                "splitCounts": {"train": 1, "validation": 1, "calibration": 0, "test": 3},
            }),
            encoding="utf-8",
        )
        representatives = []
        viewports = [
            (512, 288, 512 / 288),
            (461, 288, 1.6),
            (461, 288, 1.6),
            (512, 288, 512 / 288),
            (288, 288, 1.0),
        ]
        for index, (width, height, aspect) in enumerate(viewports):
            representatives.append({
                "pose_index": index,
                "camera_pos": [float(index), 2.0, -float(index)],
                "camera_forward": [0.0, 0.0, -2.0],
                "aspect": aspect,
                "width": width,
                "height": height,
                "sample_category": "street_gap",
            })
        representative_plan = root / "representatives.jsonl"
        representative_plan.write_text(
            "".join(json.dumps(row) + "\n" for row in representatives),
            encoding="utf-8",
        )
        return dataset, representative_plan

    def test_selects_test_rows_and_groups_exact_viewports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset, representative_plan = self._write_fixture(Path(temporary))
            output_dir = Path(temporary) / "point60_plans"
            summary = builder.build_point60_plan(dataset, representative_plan, output_dir, scene="fixture")

            self.assertEqual(summary["selectedPoseCount"], 3)
            self.assertEqual(summary["selectedPoseIndices"], [1, 2, 3])
            self.assertEqual(summary["canonicalSubposeId"], 0)
            self.assertEqual(summary["fovYDeg"], 60.0)
            self.assertEqual(summary["groupCount"], 2)
            self.assertEqual(
                [group["sourcePoseIndices"] for group in summary["groups"]],
                [[1, 2], [3]],
            )

            for group in summary["groups"]:
                rows = [
                    json.loads(line)
                    for line in (output_dir / group["file"]).read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual([row["pose_index"] for row in rows], group["sourcePoseIndices"])
                for row in rows:
                    self.assertEqual(row["viewcell_id"], row["pose_index"])
                    self.assertEqual(row["subpose_id"], 0)
                    self.assertEqual(row["fov_y"], 60.0)
                    self.assertEqual(row["render_fov_y"], 60.0)
                    self.assertEqual(row["pvs_back_offset"], 0.0)
                    self.assertEqual(row["camera_pos"], [float(row["pose_index"]), 2.0, -float(row["pose_index"])])

    def test_missing_aspect_comes_from_explicit_dimensions_not_a_global_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, representative_plan = self._write_fixture(root)
            rows = [json.loads(line) for line in representative_plan.read_text(encoding="utf-8").splitlines()]
            rows[1].pop("aspect")
            representative_plan.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            summary = builder.build_point60_plan(dataset, representative_plan, root / "plans")
            row = json.loads(
                (root / "plans" / "point60_test_w461_h288_a1p6006944444444444.jsonl")
                .read_text(encoding="utf-8").splitlines()[0]
            )
            self.assertEqual(row["point60_aspect_source"], "width_div_height")
            self.assertAlmostEqual(row["aspect"], 461 / 288)
            self.assertEqual(summary["aspectSources"], ["representative_plan", "width_div_height"])

    def test_refuses_misaligned_representative_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, representative_plan = self._write_fixture(root)
            rows = [json.loads(line) for line in representative_plan.read_text(encoding="utf-8").splitlines()]
            rows[2]["pose_index"] = 99
            representative_plan.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "row-aligned"):
                builder.build_point60_plan(dataset, representative_plan, root / "plans")

    def test_refuses_nonempty_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, representative_plan = self._write_fixture(root)
            output_dir = root / "plans"
            output_dir.mkdir()
            (output_dir / "old.jsonl").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "non-empty"):
                builder.build_point60_plan(dataset, representative_plan, output_dir)


if __name__ == "__main__":
    unittest.main()
