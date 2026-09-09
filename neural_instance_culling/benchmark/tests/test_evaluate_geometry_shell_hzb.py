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

import evaluate_geometry_shell_hzb as evaluator  # noqa: E402


class GeometryShellHZBEvaluatorTests(unittest.TestCase):
    def test_aggregate_counts_keep_repeated_pose_references(self) -> None:
        first = evaluator.set_metrics(
            np.asarray([1], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
            np.asarray([1, 2], dtype=np.uint32),
        )
        second = evaluator.set_metrics(
            np.asarray([1], dtype=np.uint32),
            np.asarray([1], dtype=np.uint32),
            np.asarray([1, 2], dtype=np.uint32),
        )
        totals = {
            key: int(first[key]) + int(second[key])
            for key in ("candidateCount", "predictedCount", "gtCount", "tp")
        }
        aggregate = evaluator.metrics_from_counts(
            totals["candidateCount"],
            totals["predictedCount"],
            totals["gtCount"],
            totals["tp"],
        )
        self.assertEqual(aggregate["candidateCount"], 4)
        self.assertEqual(aggregate["predictedCount"], 2)
        self.assertEqual(aggregate["gtCount"], 2)
        self.assertEqual(aggregate["tp"], 2)
        self.assertEqual(aggregate["tn"], 2)

    def test_glb_metrics_are_per_pose_and_can_report_source_bytes(self) -> None:
        first = evaluator.glb_metrics({1, 2}, {2, 3}, {1: 10, 2: 20, 3: 30})
        second = evaluator.glb_metrics({2}, {2}, {1: 10, 2: 20, 3: 30})
        macro = evaluator.macro_glb_metrics([first, second])
        self.assertEqual(first["predictedCount"], 2)
        self.assertEqual(first["truthCount"], 2)
        self.assertEqual(first["intersectionCount"], 1)
        self.assertEqual(first["predictedBytes"], 30)
        self.assertEqual(first["truthBytes"], 50)
        self.assertEqual(first["intersectionBytes"], 20)
        self.assertAlmostEqual(float(macro["predictedCount"]), 1.5)
        self.assertAlmostEqual(float(macro["predictedBytes"]), 25.0)
        self.assertAlmostEqual(float(macro["byteRecall"]), (20 / 50 + 1.0) / 2)

    def test_glb_union_is_explicitly_diagnostic(self) -> None:
        first = evaluator.glb_metrics({1}, {1, 2})
        second = evaluator.glb_metrics({2}, {1, 2})
        union = evaluator.glb_metrics({1, 2}, {1, 2})
        self.assertEqual(evaluator.macro_glb_metrics([first, second])["predictedCount"], 1.0)
        self.assertEqual(union["predictedCount"], 2)

    def test_glb_index_and_root_supply_file_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "task-0").mkdir()
            glb_path = root / "task-0" / "sub_0.glb"
            glb_path.write_bytes(b"glb-payload")
            index_path = root / "glbIndex.json"
            index_path.write_text(json.dumps({
                "total": 1,
                "entries": [{"globalId": 0, "path": "task-0/sub_0.glb"}],
            }), encoding="utf-8")
            sizes, source = evaluator.load_glb_sizes(index_path, root)
            self.assertEqual(sizes, {0: len(b"glb-payload")})
            self.assertEqual(source["sizeUnit"], "source_glb_file_bytes")

if __name__ == "__main__":
    unittest.main()
