from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "evaluate_gt_convergence.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gt_convergence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_plan_and_raw(viewcell_count: int = 2, subpose_count: int = 4) -> tuple[list[dict], list[dict]]:
    plan: list[dict] = []
    raw: list[dict] = []
    for viewcell_id in range(viewcell_count):
        source_pose_index = 100 + viewcell_id
        for subpose_id in range(subpose_count):
            pose_index = viewcell_id * subpose_count + subpose_id
            plan.append({
                "pose_index": pose_index,
                "viewcell_id": viewcell_id,
                "subpose_id": subpose_id,
                "source_pose_index": source_pose_index,
            })
            if viewcell_id == 0:
                visible = [1] if subpose_id == 0 else ([2] if subpose_id == 1 else [3])
                weights = [2 + subpose_id]
            else:
                visible = [10 + subpose_id]
                weights = [1]
            raw.append({
                **plan[-1],
                "visible_component_ids": visible,
                "component_weights": weights,
            })
    return plan, raw


class GtConvergenceTest(unittest.TestCase):
    def test_nested_union_uses_max_component_weight(self) -> None:
        visibility = {
            10: (np.asarray([1, 2]), np.asarray([0.2, 0.4])),
            11: (np.asarray([2, 3]), np.asarray([0.8, 0.1])),
        }
        ids, weights = MODULE.nested_union(np.asarray([10, 11]), visibility, 2)
        self.assertEqual(ids, {1, 2, 3})
        self.assertAlmostEqual(weights[2], 0.8)

    def test_load_plan_requires_the_exact_nested_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.jsonl"
            plan, _raw = make_plan_and_raw()
            write_jsonl(path, plan)
            loaded = MODULE.load_plan(path, viewcell_count=2, subpose_count=4)
            self.assertEqual(
                [entry["key"] for entry in loaded[1]],
                [(1, 0, 101), (1, 1, 101), (1, 2, 101), (1, 3, 101)],
            )

            missing = plan[:-1]
            write_jsonl(path, missing)
            with self.assertRaisesRegex(ValueError, "exactly 2x4=8"):
                MODULE.load_plan(path, viewcell_count=2, subpose_count=4)

    def test_raw_is_joined_by_viewcell_subpose_and_source_pose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, raw = make_plan_and_raw()
            plan_path = root / "plan.jsonl"
            raw_path = root / "raw.jsonl"
            write_jsonl(plan_path, plan)
            write_jsonl(raw_path, list(reversed(raw)))
            loaded_plan = MODULE.load_plan(plan_path, viewcell_count=2, subpose_count=4)
            visibility = MODULE.load_raw(raw_path, loaded_plan)
            self.assertEqual(visibility[(0, 2, 100)][0].tolist(), [3])

            bad = list(raw)
            bad[0] = {**bad[0], "source_pose_index": 999}
            write_jsonl(raw_path, bad)
            with self.assertRaisesRegex(ValueError, "does not match the supplied plan"):
                MODULE.load_raw(raw_path, loaded_plan)

    def test_convergence_is_relative_to_the_final_128_union(self) -> None:
        plan, raw = make_plan_and_raw(viewcell_count=1, subpose_count=4)
        loaded_plan = {0: [
            {"key": (row["viewcell_id"], row["subpose_id"], row["source_pose_index"]), "poseIndex": row["pose_index"]}
            for row in plan
        ]}
        visibility = {
            (row["viewcell_id"], row["subpose_id"], row["source_pose_index"]): (
                np.asarray(row["visible_component_ids"], dtype=np.uint32),
                np.asarray(row["component_weights"], dtype=np.float64),
            )
            for row in raw
        }
        rows = MODULE.build_convergence_rows("test", loaded_plan, visibility, [1, 2, 4])
        by_count = {row["sampleCount"]: row for row in rows}
        self.assertEqual(by_count[4]["final128VisibleCount"], 3)
        self.assertAlmostEqual(by_count[1]["newInstanceRate"], 1 / 3)
        self.assertAlmostEqual(by_count[2]["newInstanceRate"], 1 / 3)
        self.assertAlmostEqual(by_count[4]["newInstanceRate"], 1 / 3)
        self.assertAlmostEqual(by_count[2]["weightedConvergence"], 1 / 2)
        self.assertEqual(by_count[4]["remainingInstanceRate"], 0.0)

    def test_evaluate_writes_all_formal_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan, raw = make_plan_and_raw(viewcell_count=100, subpose_count=128)
            plan_path = root / "plan.jsonl"
            raw_path = root / "raw.jsonl"
            write_jsonl(plan_path, plan)
            write_jsonl(raw_path, raw)
            manifest = MODULE.evaluate(
                scene="synthetic",
                plan_path=plan_path,
                raw_path=raw_path,
                output_dir=root / "evaluation",
            )
            self.assertEqual(manifest["viewcellCount"], 100)
            self.assertEqual(manifest["subposesPerViewcell"], 128)
            self.assertEqual(
                [row["sampleCount"] for row in manifest["summary"]],
                [1, 2, 4, 8, 16, 32, 64, 128],
            )
            self.assertTrue((root / "evaluation" / "per_viewcell.csv").is_file())
            self.assertEqual(len(manifest["summary"]), 8)
            self.assertTrue(all(row["viewcellCount"] == 100 for row in manifest["summary"]))


if __name__ == "__main__":
    unittest.main()
