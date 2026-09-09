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
    def test_point_gt_uses_only_canonical_raw_subpose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            rows = [
                {
                    "viewcell_id": 7,
                    "subpose_id": 1,
                    "pose_index": 8,
                    "visible_component_ids": [3],
                    "component_weights": [99],
                },
                {
                    "viewcell_id": 7,
                    "subpose_id": 0,
                    "pose_index": 7,
                    "visible_component_ids": [2, 1],
                    "component_weights": [4, 9],
                },
            ]
            (raw_dir / "samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            found, summary = evaluator.load_point_ground_truth(raw_dir, {7})
            self.assertEqual(summary["subposeId"], 0)
            self.assertEqual(summary["selection"], "fixed-subpose-id-0-per-viewcell")
            self.assertEqual(found[7]["visibleIds"].tolist(), [1, 2])
            self.assertEqual(found[7]["visibleWeights"].tolist(), [9.0, 4.0])
            self.assertEqual(found[7]["poseIndex"], 7)

    def test_point_gt_requires_every_canonical_subpose(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = Path(temporary)
            (raw_dir / "samples.jsonl").write_text(
                json.dumps({
                    "viewcell_id": 7,
                    "subpose_id": 1,
                    "visible_component_ids": [],
                    "component_weights": [],
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "canonical subpose_id=0"):
                evaluator.load_point_ground_truth(raw_dir, {7})

    def test_point_evaluation_rejects_union_gt_without_raw_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "dataset_meta.json").write_text(
                json.dumps({"poseCount": 1, "numInstances": 2}), encoding="utf-8"
            )
            np.asarray([0, 2], dtype="<u8").tofile(dataset_dir / "candidate_offsets.bin")
            np.asarray([0, 1], dtype="<u4").tofile(dataset_dir / "candidate_ids.bin")
            runtime_meta = root / "runtime.json"
            runtime_meta.write_text(json.dumps({
                "componentRecords": [
                    {"componentGlobalId": 0, "globalGlbId": 0},
                    {"componentGlobalId": 1, "globalGlbId": 1},
                ]
            }), encoding="utf-8")
            result = root / "point.json"
            result.write_text(json.dumps({
                "schema": evaluator.RESULT_SCHEMA,
                "mode": "Point60",
                "samples": [{"poseId": 0, "visibleInstanceIds": [0]}],
            }), encoding="utf-8")
            args = type("Args", (), {
                "result": result,
                "dataset_dir": dataset_dir,
                "runtime_meta": runtime_meta,
                "output": root / "metrics.json",
                "split": None,
                "bootstrap_seed": 1,
                "glb_index": None,
                "glb_root": None,
                "point_gt_raw_dir": None,
            })()
            with self.assertRaisesRegex(ValueError, "--point-gt-raw-dir"):
                evaluator.evaluate(args)

    def test_point_evaluation_uses_raw_ids_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "dataset_meta.json").write_text(
                json.dumps({"poseCount": 1, "numInstances": 3}), encoding="utf-8"
            )
            np.asarray([0, 3], dtype="<u8").tofile(dataset_dir / "candidate_offsets.bin")
            np.asarray([0, 1, 2], dtype="<u4").tofile(dataset_dir / "candidate_ids.bin")
            raw_dir = root / "raw"
            raw_dir.mkdir()
            (raw_dir / "samples.jsonl").write_text("\n".join([
                json.dumps({
                    "viewcell_id": 0,
                    "subpose_id": 1,
                    "visible_component_ids": [2],
                    "component_weights": [100],
                }),
                json.dumps({
                    "viewcell_id": 0,
                    "subpose_id": 0,
                    "visible_component_ids": [1],
                    "component_weights": [7],
                }),
            ]) + "\n", encoding="utf-8")
            runtime_meta = root / "runtime.json"
            runtime_meta.write_text(json.dumps({
                "componentRecords": [
                    {"componentGlobalId": 0, "globalGlbId": 0},
                    {"componentGlobalId": 1, "globalGlbId": 1},
                    {"componentGlobalId": 2, "globalGlbId": 2},
                ]
            }), encoding="utf-8")
            result = root / "point.json"
            result.write_text(json.dumps({
                "schema": evaluator.RESULT_SCHEMA,
                "mode": "Point60",
                "workload": {"split": "test"},
                "samples": [{"poseId": 0, "visibleInstanceIds": [1]}],
            }), encoding="utf-8")
            args = type("Args", (), {
                "result": result,
                "dataset_dir": dataset_dir,
                "runtime_meta": runtime_meta,
                "output": root / "metrics.json",
                "split": None,
                "bootstrap_seed": 1,
                "glb_index": None,
                "glb_root": None,
                "point_gt_raw_dir": raw_dir,
            })()
            output = evaluator.evaluate(args)
            self.assertEqual(output["groundTruth"]["mode"], "canonical-subpose")
            self.assertEqual(output["groundTruth"]["subposeId"], 0)
            self.assertEqual(output["groundTruth"]["source"], "raw_three_color_id_jsonl")
            self.assertEqual(output["aggregate"]["gtCount"], 1)
            self.assertEqual(output["aggregate"]["tp"], 1)
            self.assertEqual(output["weightedRecallWeightSource"], "raw_three_color_id_component_weights")

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
