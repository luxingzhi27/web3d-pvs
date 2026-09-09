#!/usr/bin/env python3
"""Generate the paper Table 1 scene statistics and Table 2 test metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "neural_instance_culling/benchmark/out/paper_results"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"cannot write an empty CSV: {output}")
    fields = list(values[0])
    for row in values[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in values:
            writer.writerow({field: row.get(field, "") for field in fields})
    temporary.replace(output)
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"scene statistics CSV is empty: {path}")
    return [dict(row) for row in rows]


def _value(container: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in container:
            return container[key]
    return default


def _metric_row(
    payload: Mapping[str, Any],
    source: Path,
    *,
    scene: str,
    method: str,
    threshold_row: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = threshold_row or payload.get("aggregate") or payload.get("metrics") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"test metric payload has no metric mapping: {source}")
    pose = payload.get("poseMacro") or {}
    if not isinstance(pose, Mapping):
        pose = {}

    def metric(aggregate_key: str, row_key: str, default: Any = None) -> Any:
        return _value(raw, aggregate_key, row_key, default=default)

    aggregate_weighted_recall = metric("weightedRecall", "agg_weighted_recall")
    lower = metric(
        "weightedRecallLowerConfidenceBound",
        "aggregateWeightedRecallLowerConfidenceBound",
        "aggregate_weighted_recall_lower_confidence_bound",
    )
    if lower is None and threshold_row is not None:
        lower = _value(threshold_row, "aggregateWeightedRecallLowerConfidenceBound", "aggregate_weighted_recall_lower_confidence_bound")
    result = {
        "scene": scene,
        "method": method,
        "test_pose_count": _value(payload, "poseCount", "evaluatedPoseCount", default=_value(summary or {}, "evaluatedPoses", default=None)),
        "zero_gt_pose_count": _value(payload, "zeroGtPoseCount", default=_value(summary or {}, "zeroGtPoseCount", default=0)),
        "threshold": _value(payload, "threshold", default=_value(threshold_row or {}, "threshold")),
        "threshold_source": _value(payload, "thresholdSource", default=""),
        "checkpoint": _value(payload, "checkpoint", default=_value(summary or {}, "checkpoint", default="")),
        "calibration": _value(
            payload,
            "calibration",
            "calibrationSummary",
            default=_value(summary or {}, "calibration", "calibrationSummary", default=""),
        ),
        "aggregate_precision": metric("precision", "agg_precision"),
        "aggregate_recall": metric("recall", "agg_recall"),
        "aggregate_weighted_recall": aggregate_weighted_recall,
        "aggregate_weighted_recall_lcb": lower,
        "aggregate_f1": metric("f1", "agg_f1"),
        "aggregate_accuracy": metric("accuracy", "agg_accuracy"),
        "aggregate_balanced_accuracy": metric("balancedAccuracy", "agg_balanced_accuracy"),
        "aggregate_specificity": metric("specificity", "agg_specificity"),
        "aggregate_average_precision": metric("averagePrecision", "aggregateAveragePrecision"),
        "aggregate_positive_rate": metric("positiveRate", "aggregatePositiveRate"),
        "aggregate_ap_lift": metric("apLift", "aggregateApLift"),
        "pose_precision": _value(pose, "precision", default=_value(threshold_row or {}, "pose_precision")),
        "pose_recall": _value(pose, "recall", default=_value(threshold_row or {}, "pose_recall")),
        "pose_weighted_recall": _value(pose, "weightedRecall", default=_value(threshold_row or {}, "pose_weighted_recall")),
        "pose_f1": _value(pose, "f1", default=_value(threshold_row or {}, "pose_f1")),
        "pose_accuracy": _value(pose, "accuracy", default=_value(threshold_row or {}, "pose_accuracy")),
        "pose_balanced_accuracy": _value(pose, "balancedAccuracy", default=_value(threshold_row or {}, "pose_balanced_accuracy")),
        "pose_specificity": _value(pose, "specificity", default=_value(threshold_row or {}, "pose_specificity")),
        "pose_average_precision": _value(pose, "averagePrecision", default=_value(threshold_row or {}, "poseMacroAveragePrecision")),
        "pose_positive_rate": _value(pose, "positiveRate", default=_value(threshold_row or {}, "poseMacroPositiveRate")),
        "pose_ap_lift": _value(pose, "apLift", default=_value(threshold_row or {}, "poseMacroApLift")),
        "useful_cull": metric("usefulCull", "agg_useful_cull"),
        "bad_cull": metric("badCull", "agg_bad_cull"),
        "avg_candidate_count": metric("avgCandidateCount", "avg_candidate_count"),
        "avg_gt_count": metric("avgGtCount", "avg_gt_count"),
        "avg_pred_count": metric("avgPredCount", "avg_pred_count"),
        "predicted_glb_count": metric("predictedGlbCount", "avg_pred_glb_count"),
        "predicted_glb_bytes": metric("predictedGlbBytes", "avg_pred_glb_bytes"),
        "glb_byte_reduction": metric("glbByteReduction", "pose_candidate_byte_reduction_ratio"),
        "score_sidecar": _value(payload, "scoreSidecar", default=_value(summary or {}, "scoreSidecar", default="")),
        "testRead": True,
        "source": str(source.resolve()),
    }
    return result


def _scene_from_path(path: Path, known_scenes: set[str]) -> str | None:
    for part in reversed(path.parts):
        if part in known_scenes:
            return part
    return None


def collect_test_metric_rows(test_metrics_dir: str | Path, scenes: Sequence[str]) -> list[dict[str, Any]]:
    root = Path(test_metrics_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"test metrics directory does not exist: {root}")
    known = set(str(scene) for scene in scenes)
    result: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        if payload.get("testRead") is not True:
            continue
        if payload.get("split") not in (None, "test"):
            raise ValueError(f"testRead=true payload is not a test split: {path}")
        payload_test_count = payload.get("testEvaluationCount")
        if payload_test_count is None and isinstance(payload.get("meta"), Mapping):
            payload_test_count = payload["meta"].get("testEvaluationCount")
        if payload_test_count is not None and int(payload_test_count) != 1:
            raise ValueError(f"formal test payload must record exactly one test evaluation: {path}")
        fallback_scene = _scene_from_path(path, known)
        if isinstance(payload.get("summaries"), list):
            for summary in payload["summaries"]:
                if not isinstance(summary, Mapping) or summary.get("testRead") is not True:
                    continue
                rows = summary.get("thresholdRows")
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError(f"formal test summary must contain one frozen threshold row: {path}")
                result.append(
                    _metric_row(
                        payload,
                        path,
                        scene=str(payload.get("scene") or fallback_scene or "unknown"),
                        method=str(summary.get("name") or "unknown"),
                        threshold_row=rows[0],
                        summary=summary,
                    )
                )
            continue
        if isinstance(payload.get("aggregate"), Mapping) or isinstance(payload.get("metrics"), Mapping):
            result.append(
                _metric_row(
                    payload,
                    path,
                    scene=str(payload.get("scene") or fallback_scene or "unknown"),
                    method=str(payload.get("method") or path.stem),
                )
            )
    if not result:
        raise ValueError(f"no testRead=true metric payloads found below {root}")
    return sorted(result, key=lambda row: (str(row["scene"]), str(row["method"]), str(row["source"])))


def _markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> Path:
    lines = [f"# {title}", "", "| " + " | ".join(label for _key, label in columns) + " |", "|" + "|".join("---" for _key, _label in columns) + "|"]
    for row in rows:
        values = []
        for key, _label in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def generate_tables(scene_statistics: Path, test_metrics_dir: Path, output_dir: Path) -> dict[str, Any]:
    stats = _read_csv(scene_statistics)
    scenes = [str(row.get("scene", "")) for row in stats]
    test_rows = collect_test_metric_rows(test_metrics_dir, scenes)
    output_dir = output_dir.resolve()
    table1 = _write_csv(output_dir / "table1_scene_statistics.csv", stats)
    table2 = _write_csv(output_dir / "table2_test_visibility.csv", test_rows)
    _markdown_table(
        output_dir / "table1_scene_statistics.md",
        "Table 1. Scene Statistics",
        stats,
        (
            ("scene", "Scene"),
            ("instance_count", "Instances"),
            ("glb_count", "GLBs"),
            ("prototype_triangle_count", "Prototype triangles"),
            ("expanded_triangle_count", "Expanded triangles"),
            ("instance_glb_reuse_factor", "Instance/GLB reuse"),
            ("glb_bytes_total", "GLB bytes"),
            ("test_pose_count", "Test poses"),
            ("test_candidate_count_mean", "Test candidates/pose"),
            ("test_gt_count_mean", "Test GT/pose"),
            ("test_zero_gt_pose_count", "Zero-GT test poses"),
        ),
    )
    _markdown_table(
        output_dir / "table2_test_visibility.md",
        "Table 2. Frozen Test Visibility",
        test_rows,
        (
            ("scene", "Scene"),
            ("method", "Method"),
            ("test_pose_count", "Test poses"),
            ("aggregate_precision", "Precision"),
            ("aggregate_recall", "Recall"),
            ("aggregate_weighted_recall", "Weighted recall"),
            ("aggregate_weighted_recall_lcb", "Weighted recall LCB"),
            ("aggregate_average_precision", "Aggregate AP"),
            ("pose_average_precision", "Pose AP"),
            ("useful_cull", "Useful cull"),
            ("bad_cull", "Bad cull"),
            ("avg_pred_count", "Predicted/pose"),
            ("glb_byte_reduction", "GLB byte reduction"),
        ),
    )
    return {
        "table1": str(table1),
        "table2": str(table2),
        "table1RowCount": len(stats),
        "table2RowCount": len(test_rows),
        "testRead": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-statistics", type=Path, default=DEFAULT_OUTPUT_DIR / "scene_statistics.csv")
    parser.add_argument("--test-metrics-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "test_metrics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(generate_tables(args.scene_statistics, args.test_metrics_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
