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
from generate_paper_tables import collect_artifact_registry, collect_test_metric_rows, generate_tables, summarize_test_rows  # noqa: E402


def _test_payload(seed: int, precision: float) -> dict[str, object]:
    return {
        "testRead": True, "split": "test", "testEvaluationCount": 1,
        "scene": "fixture", "method": f"full_seed{seed}", "poseCount": 3,
        "aggregate": {
            "precision": precision, "recall": 0.9, "weightedRecall": 0.95,
            "weightedRecallLowerConfidenceBound": 0.9, "averagePrecision": 0.5,
            "balancedAccuracy": 0.8, "specificity": 0.7,
            "usefulCull": 0.3, "badCull": 0.1,
            "avgPredCount": 2.0, "glbByteReduction": 0.2,
        },
        "poseMacro": {
            "precision": precision, "recall": 0.8, "weightedRecall": 0.95,
            "balancedAccuracy": 0.75, "specificity": 0.7,
            "averagePrecision": 0.4, "positiveRate": 0.2, "apLift": 2.0,
        },
    }


class PaperTableGeneratorTests(unittest.TestCase):
    def test_table_two_uses_three_seed_mean_and_sample_std(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / "scene_statistics.csv"
            stats.write_text("scene,instance_count\nfixture,2\n", encoding="utf-8")
            metrics = root / "metrics"
            metrics.mkdir()
            for seed, precision in enumerate((1.0, 2.0, 3.0), 1):
                (metrics / f"seed{seed}.json").write_text(json.dumps(_test_payload(seed, precision)), encoding="utf-8")
            raw = collect_test_metric_rows(metrics, ["fixture"])
            summary = summarize_test_rows(raw)
            self.assertEqual(len(raw), 3)
            self.assertEqual(summary[0]["seed_count"], 3)
            self.assertEqual(summary[0]["aggregate_precision_mean"], 2.0)
            self.assertAlmostEqual(summary[0]["aggregate_precision_std"], 1.0)
            self.assertEqual(summary[0]["pose_average_precision_mean"], 0.4)
            self.assertAlmostEqual(summary[0]["pose_false_occlusion_rate_mean"], 0.2)
            result = generate_tables(stats, metrics, root / "tables")
            self.assertEqual(result["table2RowCount"], 1)
            with (root / "tables/table2_test_visibility.csv").open(newline="") as handle:
                self.assertEqual(next(csv.DictReader(handle))["seed_count"], "3")

    def test_non_test_split_cannot_be_marked_as_test_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics"
            metrics.mkdir()
            (metrics / "invalid.json").write_text(json.dumps({"testRead": True, "split": "validation"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a test split"):
                collect_test_metric_rows(metrics, ["fixture"])

    def test_registry_keeps_explicit_unavailable_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "bundle_manifest.json"
            manifest.write_text(json.dumps({"artifactRegistry": {"schema": "fixture", "sections": ["hzb"], "artifacts": [
                {"section": "hzb", "artifactId": "hzb", "path": "/missing", "status": "unavailable", "reason": "not supplied"},
            ]}}), encoding="utf-8")
            rows = collect_artifact_registry(manifest)
            self.assertEqual(rows[0]["status"], "unavailable")
            self.assertEqual(rows[0]["reason"], "not supplied")


if __name__ == "__main__":
    unittest.main()
