from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

BENCHMARK = Path(__file__).resolve().parents[1]
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))

from generate_paper_tables import collect_test_metric_rows, generate_tables  # noqa: E402


class PaperTableGeneratorTests(unittest.TestCase):
    def test_only_frozen_test_payloads_enter_table_two(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "scene_statistics.csv"
            with stats.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["scene", "instance_count"])
                writer.writeheader()
                writer.writerow({"scene": "fixture", "instance_count": 2})
            metrics = root / "metrics"
            metrics.mkdir()
            (metrics / "validation.json").write_text(
                json.dumps({"testRead": False, "split": "validation"}),
                encoding="utf-8",
            )
            (metrics / "test.json").write_text(
                json.dumps(
                    {
                        "testRead": True,
                        "split": "test",
                        "testEvaluationCount": 1,
                        "scene": "fixture",
                        "method": "keep_all",
                        "poseCount": 1,
                        "aggregate": {
                            "precision": 1.0,
                            "recall": 1.0,
                            "weightedRecall": 1.0,
                            "weightedRecallLowerConfidenceBound": 1.0,
                        },
                        "poseMacro": {},
                    }
                ),
                encoding="utf-8",
            )
            rows = collect_test_metric_rows(metrics, ["fixture"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["method"], "keep_all")
            result = generate_tables(stats, metrics, root / "tables")
            self.assertEqual(result["table1RowCount"], 1)
            self.assertEqual(result["table2RowCount"], 1)

    def test_non_test_split_cannot_be_marked_as_test_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            path = metrics / "invalid.json"
            path.write_text(
                json.dumps({"testRead": True, "split": "validation"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not a test split"):
                collect_test_metric_rows(metrics, ["fixture"])


if __name__ == "__main__":
    unittest.main()
