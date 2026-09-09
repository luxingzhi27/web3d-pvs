#!/usr/bin/env python3
"""Generate Table 1, three-seed Table 2, and the artifact registry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "neural_instance_culling/benchmark/out/paper_results"
UNAVAILABLE = "unavailable"
TABLE2_METRICS = (
    "aggregate_precision", "aggregate_recall", "aggregate_weighted_recall",
    "aggregate_weighted_recall_lcb", "aggregate_average_precision",
    "pose_average_precision", "useful_cull", "bad_cull", "avg_pred_count",
    "glb_byte_reduction",
)
ARTIFACT_FIELDS = ("section", "artifactId", "path", "status", "reason")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str] | None = None) -> Path:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    if not values:
        raise ValueError(f"cannot write an empty CSV: {output}")
    columns = list(fields or values[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in columns} for row in values)
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
    raw: Mapping[str, Any] | None = None,
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = raw or payload.get("aggregate") or payload.get("metrics") or {}
    if not isinstance(metrics, Mapping):
        raise ValueError(f"test metric payload has no metric mapping: {source}")
    pose = payload.get("poseMacro") or {}
    if not isinstance(pose, Mapping):
        pose = {}
    summary = summary or {}

    def metric(*names: str) -> Any:
        return _value(metrics, *names)

    return {
        "scene": scene, "method": method,
        "test_pose_count": _value(payload, "poseCount", "evaluatedPoseCount", default=summary.get("evaluatedPoses")),
        "zero_gt_pose_count": _value(payload, "zeroGtPoseCount", default=summary.get("zeroGtPoseCount", 0)),
        "threshold": _value(payload, "threshold", default=_value(metrics, "threshold")),
        "threshold_source": _value(payload, "thresholdSource", default=""),
        "checkpoint": _value(payload, "checkpoint", default=summary.get("checkpoint", "")),
        "calibration": _value(payload, "calibration", "calibrationSummary", default=summary.get("calibration", "")),
        "aggregate_precision": metric("precision", "agg_precision"),
        "aggregate_recall": metric("recall", "agg_recall"),
        "aggregate_weighted_recall": metric("weightedRecall", "aggregateWeightedRecall", "agg_weighted_recall"),
        "aggregate_weighted_recall_lcb": metric("weightedRecallLowerConfidenceBound", "aggregateWeightedRecallLowerConfidenceBound", "aggregate_weighted_recall_lower_confidence_bound"),
        "aggregate_f1": metric("f1", "agg_f1"),
        "aggregate_accuracy": metric("accuracy", "agg_accuracy"),
        "aggregate_balanced_accuracy": metric("balancedAccuracy", "agg_balanced_accuracy"),
        "aggregate_specificity": metric("specificity", "agg_specificity"),
        "aggregate_average_precision": metric("averagePrecision", "aggregateAveragePrecision"),
        "aggregate_positive_rate": metric("positiveRate", "aggregatePositiveRate"),
        "aggregate_ap_lift": metric("apLift", "aggregateApLift"),
        "pose_precision": _value(pose, "precision"),
        "pose_recall": _value(pose, "recall"),
        "pose_weighted_recall": _value(pose, "weightedRecall"),
        "pose_f1": _value(pose, "f1"),
        "pose_accuracy": _value(pose, "accuracy"),
        "pose_balanced_accuracy": _value(pose, "balancedAccuracy"),
        "pose_specificity": _value(pose, "specificity"),
        "pose_average_precision": _value(pose, "averagePrecision", "poseMacroAveragePrecision"),
        "pose_positive_rate": _value(pose, "positiveRate", "poseMacroPositiveRate"),
        "pose_ap_lift": _value(pose, "apLift", "poseMacroApLift"),
        "useful_cull": metric("usefulCull", "agg_useful_cull"),
        "bad_cull": metric("badCull", "agg_bad_cull"),
        "avg_candidate_count": metric("avgCandidateCount", "avg_candidate_count"),
        "avg_gt_count": metric("avgGtCount", "avg_gt_count"),
        "avg_pred_count": metric("avgPredCount", "avg_pred_count"),
        "predicted_glb_count": metric("predictedGlbCount", "avg_pred_glb_count"),
        "predicted_glb_bytes": metric("predictedGlbBytes", "avg_pred_glb_bytes"),
        "glb_byte_reduction": metric("glbByteReduction", "pose_candidate_byte_reduction_ratio"),
        "testRead": True, "source": str(source.resolve()),
    }


def _base_method(method: str) -> str:
    return re.sub(r"_seed\d+$", "", method)


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
        count = payload.get("testEvaluationCount")
        if count is None and isinstance(payload.get("meta"), Mapping):
            count = payload["meta"].get("testEvaluationCount")
        if count is not None and int(count) != 1:
            raise ValueError(f"formal test payload must record exactly one test evaluation: {path}")
        scene = str(payload.get("scene") or next((part for part in reversed(path.parts) if part in known), "unknown"))
        summaries = payload.get("summaries")
        if isinstance(summaries, list):
            for summary in summaries:
                if not isinstance(summary, Mapping) or summary.get("testRead") is not True:
                    continue
                rows = summary.get("thresholdRows")
                if not isinstance(rows, list) or len(rows) != 1:
                    raise ValueError(f"formal test summary must contain one frozen threshold row: {path}")
                result.append(_metric_row(payload, path, scene=scene, method=str(summary.get("name", "unknown")), raw=rows[0], summary=summary))
        elif isinstance(payload.get("aggregate"), Mapping) or isinstance(payload.get("metrics"), Mapping):
            result.append(_metric_row(payload, path, scene=scene, method=str(payload.get("method") or path.stem)))
    if not result:
        raise ValueError(f"no testRead=true metric payloads found below {root}")
    return sorted(result, key=lambda row: (str(row["scene"]), str(row["method"]), str(row["source"])))


def summarize_test_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["scene"]), _base_method(str(row["method"]))), []).append(row)
    result: list[dict[str, Any]] = []
    for (scene, method), members in sorted(groups.items()):
        row: dict[str, Any] = {
            "scene": scene, "method": method, "seed_count": len(members),
            "test_pose_count": members[0].get("test_pose_count", UNAVAILABLE),
        }
        for field in TABLE2_METRICS:
            values = [float(item[field]) for item in members] if all(item.get(field) is not None for item in members) else []
            row[f"{field}_mean"] = mean(values) if values else UNAVAILABLE
            row[f"{field}_std"] = stdev(values) if len(values) > 1 else 0.0 if values else UNAVAILABLE
            row[f"{field}_display"] = f"{row[f'{field}_mean']:.6g} +/- {row[f'{field}_std']:.6g}" if values else UNAVAILABLE
        result.append(row)
    return result


def _markdown_table(path: Path, title: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> Path:
    lines = [f"# {title}", "", "| " + " | ".join(label for _key, label in columns) + " |", "|" + "|".join("---" for _key, _label in columns) + "|"]
    for row in rows:
        values = [f"{row.get(key):.6g}" if isinstance(row.get(key), float) else str(row.get(key, "")) for key, _label in columns]
        lines.append("| " + " | ".join(values) + " |")
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def collect_artifact_registry(bundle_manifest: str | Path | Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = bundle_manifest if isinstance(bundle_manifest, Mapping) else _load_json(Path(bundle_manifest).expanduser().resolve())
    registry = payload.get("artifactRegistry")
    if not isinstance(registry, Mapping) or not isinstance(registry.get("artifacts"), list):
        raise ValueError("bundle manifest has no artifact registry")
    return [{field: item.get(field, UNAVAILABLE) for field in ARTIFACT_FIELDS} for item in registry["artifacts"] if isinstance(item, Mapping)]


def _write_artifact_registry(bundle_manifest: Path, output_dir: Path) -> dict[str, Any]:
    rows = collect_artifact_registry(bundle_manifest)
    fields = ("section", "artifactId", "path", "status", "reason")
    csv_path = _write_csv(output_dir / "artifact_registry.csv", rows, fields)
    _markdown_table(output_dir / "artifact_registry.md", "Paper Artifact Registry", rows, tuple((field, field) for field in fields))
    registry = _load_json(bundle_manifest).get("artifactRegistry", {})
    return {"schema": registry.get("schema", UNAVAILABLE), "csv": str(csv_path), "markdown": str((output_dir / "artifact_registry.md").resolve()), "rowCount": len(rows), "sections": registry.get("sections", [])}


def generate_tables(scene_statistics: Path, test_metrics_dir: Path, output_dir: Path, *, bundle_manifest: Path | None = None) -> dict[str, Any]:
    stats = _read_csv(scene_statistics)
    raw_rows = collect_test_metric_rows(test_metrics_dir, [str(row.get("scene", "")) for row in stats])
    table2_rows = summarize_test_rows(raw_rows)
    output_dir = output_dir.resolve()
    table1 = _write_csv(output_dir / "table1_scene_statistics.csv", stats)
    table2_fields = ["scene", "method", "seed_count", "test_pose_count"] + [f"{field}_{suffix}" for field in TABLE2_METRICS for suffix in ("mean", "std")]
    table2 = _write_csv(output_dir / "table2_test_visibility.csv", table2_rows, table2_fields)
    _markdown_table(output_dir / "table1_scene_statistics.md", "Table 1. Scene Statistics", stats, (("scene", "Scene"), ("instance_count", "Instances"), ("glb_count", "GLBs"), ("prototype_triangle_count", "Prototype triangles"), ("expanded_triangle_count", "Expanded triangles"), ("instance_glb_reuse_factor", "Instance/GLB reuse"), ("glb_bytes_total", "GLB bytes"), ("test_pose_count", "Test poses"), ("test_candidate_count_mean", "Test candidates/pose"), ("test_gt_count_mean", "Test GT/pose"), ("test_zero_gt_pose_count", "Zero-GT test poses")))
    _markdown_table(output_dir / "table2_test_visibility.md", "Table 2. Frozen Test Visibility (mean +/- std)", table2_rows, (("scene", "Scene"), ("method", "Method"), ("seed_count", "Seeds"), ("test_pose_count", "Test poses")) + tuple((f"{field}_display", field) for field in TABLE2_METRICS))
    manifest = bundle_manifest.resolve() if bundle_manifest is not None else output_dir / "bundle_manifest.json"
    if bundle_manifest is not None and not manifest.is_file():
        raise FileNotFoundError(f"bundle manifest does not exist: {manifest}")
    result: dict[str, Any] = {"table1": str(table1), "table2": str(table2), "table1RowCount": len(stats), "table2RowCount": len(table2_rows), "testRead": True}
    if manifest.is_file():
        result["artifactRegistry"] = _write_artifact_registry(manifest, output_dir)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-statistics", type=Path, default=DEFAULT_OUTPUT_DIR / "scene_statistics.csv")
    parser.add_argument("--test-metrics-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "test_metrics")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bundle-manifest", type=Path, default=None)
    args = parser.parse_args(argv)
    print(json.dumps(generate_tables(args.scene_statistics, args.test_metrics_dir, args.output_dir, bundle_manifest=args.bundle_manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
