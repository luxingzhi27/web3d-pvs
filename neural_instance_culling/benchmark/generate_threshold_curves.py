#!/usr/bin/env python3
"""Build calibration/validation weighted-recall efficiency curves."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

BENCHMARK = Path(__file__).resolve().parent
MODEL = BENCHMARK.parent / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from pvs_threshold_metrics import weighted_recall_lower_confidence_bound  # noqa: E402

try:
    from .score_sidecar import SIDECAR_SCHEMA, read_score_sidecar  # noqa: E402
except ImportError:  # Direct script execution.
    from score_sidecar import SIDECAR_SCHEMA, read_score_sidecar  # noqa: E402


CURVE_SCHEMA = "pvs-weighted-recall-efficiency-curves-v2"
PLOT_SCHEMA = "pvs-weighted-recall-efficiency-plot-source-v1"
TARGET_WEIGHTED_RECALL = 0.99
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260909

CSV_FIELDS = [
    "source", "split", "threshold", "threshold_source", "calibration_safe",
    "weighted_recall", "weighted_recall_lower_confidence_bound", "precision",
    "recall", "accuracy", "balanced_accuracy", "specificity", "avg_pred",
    "useful_cull", "bad_cull", "avg_candidate_count", "avg_gt_count",
    "avg_pred_glb_bytes", "avg_candidate_glb_bytes", "glb_byte_reduction",
    "pose_weighted_recall", "pose_precision", "pose_recall", "pose_accuracy",
    "pose_balanced_accuracy", "pose_specificity", "pose_useful_cull",
    "pose_bad_cull", "tp", "fp", "fn", "tn", "eval_pose_count",
    "zero_gt_pose_count", "glb_bytes_available",
]


@dataclass
class SidecarData:
    source: str
    split: str
    manifest: dict[str, Any]
    pose_indices: np.ndarray
    pose_offsets: np.ndarray
    scores: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    candidate_ids: np.ndarray


@dataclass(frozen=True)
class ResourceTable:
    instance_to_glb: np.ndarray
    glb_bytes: np.ndarray


def _manifest_path(path: str | Path) -> Path:
    value = Path(path).resolve()
    if value.is_dir():
        value = value / "manifest.json"
    if not value.is_file() or value.name != "manifest.json":
        raise ValueError("the input must be a typed sidecar directory or its manifest.json")
    return value


def _valid_sidecar(path: str | Path, split: str) -> SidecarData:
    manifest_path = _manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"expected canonical typed sidecar {SIDECAR_SCHEMA}: {manifest_path}")
    if manifest.get("split") != split or manifest.get("testRead") is not False:
        raise ValueError(f"sidecar must be split={split!r} with testRead=false: {manifest_path}")
    arrays = read_score_sidecar(manifest_path)
    poses = np.asarray(arrays["poseIndices"], dtype=np.int64)
    offsets = np.asarray(arrays["poseOffsets"], dtype=np.int64)
    scores = np.asarray(arrays["scores"], dtype=np.float32)
    targets = np.asarray(arrays["targets"], dtype=np.uint8)
    weights = np.asarray(arrays["visibleWeights"], dtype=np.float64)
    candidates = np.asarray(arrays["candidateIds"], dtype=np.uint32)
    if poses.size == 0 or offsets.size != poses.size + 1 or int(offsets[0]) != 0:
        raise ValueError("typed sidecar has invalid pose offsets")
    if int(offsets[-1]) != scores.size or bool(np.any(np.diff(offsets) < 0)):
        raise ValueError("typed sidecar offsets do not cover scores")
    if not (scores.size == targets.size == weights.size == candidates.size):
        raise ValueError("typed sidecar arrays are not candidate aligned")
    if not bool(np.isfinite(scores).all()) or bool(np.any((scores < 0.0) | (scores > 1.0))):
        raise ValueError("typed sidecar scores must be finite probabilities")
    if bool(np.any(targets > 1)) or not bool(np.isfinite(weights).all()) or bool(np.any(weights < 0.0)):
        raise ValueError("typed sidecar targets or weights are invalid")
    if bool(np.any(np.diff(poses) <= 0)):
        raise ValueError("typed sidecar pose indices must be strictly increasing")
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        values = candidates[int(start) : int(end)]
        if values.size != np.unique(values).size:
            raise ValueError("typed sidecar candidate IDs must be unique within each pose")
    return SidecarData(
        source=str(manifest_path.parent),
        split=split,
        manifest=manifest,
        pose_indices=poses,
        pose_offsets=offsets,
        scores=scores,
        targets=targets,
        weights=weights,
        candidate_ids=candidates,
    )


def _thresholds(scores: np.ndarray, maximum: int) -> np.ndarray:
    points = np.unique(np.asarray(scores, dtype=np.float32))
    if points.size == 0:
        return np.asarray([0.0], dtype=np.float32)
    if points.size <= maximum:
        return points
    indices = np.linspace(0, points.size - 1, maximum, dtype=np.int64)
    return points[np.unique(indices)]


def _div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator > 0.0 else float(default)


def load_resource_table(runtime_meta: str | Path, glb_index: str | Path, glb_root: str | Path | None = None) -> ResourceTable:
    """Load the existing instance-to-GLB mapping and real GLB file sizes."""
    _world_aabbs, instance_to_glb, _meta = load_runtime_meta(Path(runtime_meta).resolve())
    index_path = Path(glb_index).resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"GLB index has no entries: {index_path}")
    count = int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0
    costs = np.zeros((count,), dtype=np.float64)
    root = Path(glb_root).resolve() if glb_root is not None else index_path.parent
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        gid = int(entry.get("globalId", -1))
        file_path = root / str(entry.get("path", ""))
        if 0 <= gid < count and file_path.is_file() and file_path.stat().st_size > 0:
            costs[gid] = float(file_path.stat().st_size)
    used = np.unique(np.asarray(instance_to_glb, dtype=np.int64))
    if used.size and (int(used.min()) < 0 or int(used.max()) >= count or bool(np.any(costs[used] <= 0.0))):
        raise FileNotFoundError("GLB index does not provide a positive byte cost for every runtime GLB")
    return ResourceTable(np.asarray(instance_to_glb, dtype=np.int64), costs)


def _resource_arrays(data: SidecarData, resources: ResourceTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(data.candidate_ids, dtype=np.int64)
    if candidates.size and (int(candidates.min()) < 0 or int(candidates.max()) >= resources.instance_to_glb.size):
        raise ValueError("typed sidecar candidate ID is outside runtime metadata")
    glb_ids = resources.instance_to_glb[candidates]
    pose_count = data.pose_indices.size
    candidate_glb_counts = np.zeros(pose_count, dtype=np.float64)
    candidate_bytes = np.zeros(pose_count, dtype=np.float64)
    for pose, (start, end) in enumerate(zip(data.pose_offsets[:-1], data.pose_offsets[1:], strict=True)):
        unique = np.unique(glb_ids[int(start) : int(end)])
        candidate_glb_counts[pose] = unique.size
        candidate_bytes[pose] = resources.glb_bytes[unique].sum() if unique.size else 0.0
    return glb_ids, candidate_glb_counts, candidate_bytes


def _safe(weighted_recall: float, lower: float, target: float) -> bool:
    return bool(weighted_recall > target and lower > target)


def score_curve(
    data: SidecarData,
    thresholds: np.ndarray,
    resources: ResourceTable,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    target_weighted_recall: float,
    calibration_safe: Mapping[float, bool] | None = None,
    threshold_source: str,
) -> list[dict[str, Any]]:
    """Compute one row per threshold; tied scores enter in one batch."""
    pose_count = int(data.pose_indices.size)
    row_ids = np.repeat(np.arange(pose_count, dtype=np.int32), np.diff(data.pose_offsets))
    positive = data.targets.astype(bool, copy=False)
    candidate_counts = np.diff(data.pose_offsets).astype(np.float64)
    gt_counts = np.bincount(row_ids, weights=positive.astype(np.float64), minlength=pose_count)
    negative_counts = candidate_counts - gt_counts
    gt_mass = np.bincount(
        row_ids,
        weights=np.where(positive, data.weights, 0.0),
        minlength=pose_count,
    )
    glb_ids, candidate_glb_counts, candidate_bytes = _resource_arrays(data, resources)
    predicted_glbs = [set() for _ in range(pose_count)]
    predicted_glb_counts = np.zeros(pose_count, dtype=np.float64)
    predicted_bytes = np.zeros(pose_count, dtype=np.float64)

    order = np.argsort(-data.scores, kind="mergesort") if data.scores.size else np.zeros(0, dtype=np.int64)
    sorted_scores = data.scores[order]
    starts = (
        np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]
        if sorted_scores.size else np.zeros(0, dtype=np.int64)
    )
    ends = np.r_[starts[1:], sorted_scores.size] if starts.size else np.zeros(0, dtype=np.int64)
    tp_pose = np.zeros(pose_count, dtype=np.float64)
    fp_pose = np.zeros(pose_count, dtype=np.float64)
    fn_pose = gt_counts.copy()
    tn_pose = negative_counts.copy()
    weighted_tp_pose = np.zeros(pose_count, dtype=np.float64)
    rows_desc: list[dict[str, Any]] = []
    cursor = 0
    points = np.unique(np.asarray(thresholds, dtype=np.float32))
    for reverse_index, threshold in enumerate(points[::-1]):
        threshold = np.float32(threshold)
        while cursor < starts.size and sorted_scores[int(starts[cursor])] >= threshold:
            indices = order[int(starts[cursor]) : int(ends[cursor])]
            poses = row_ids[indices]
            pos_indices = indices[positive[indices]]
            neg_indices = indices[~positive[indices]]
            if pos_indices.size:
                counts = np.bincount(row_ids[pos_indices], minlength=pose_count)
                tp_pose += counts
                fn_pose -= counts
                weighted_tp_pose += np.bincount(
                    row_ids[pos_indices], weights=data.weights[pos_indices], minlength=pose_count
                )
            if neg_indices.size:
                counts = np.bincount(row_ids[neg_indices], minlength=pose_count)
                fp_pose += counts
                tn_pose -= counts
            for row_index, pose in zip(indices.tolist(), poses.tolist(), strict=True):
                gid = int(glb_ids[row_index])
                if gid not in predicted_glbs[pose]:
                    predicted_glbs[pose].add(gid)
                    predicted_glb_counts[pose] += 1.0
                    predicted_bytes[pose] += resources.glb_bytes[gid]
            cursor += 1

        pred_pose = tp_pose + fp_pose
        pose_precision = np.divide(tp_pose, pred_pose, out=np.ones(pose_count), where=pred_pose > 0.0)
        pose_recall = np.divide(tp_pose, gt_counts, out=np.ones(pose_count), where=gt_counts > 0.0)
        pose_specificity = np.divide(tn_pose, tn_pose + fp_pose, out=np.ones(pose_count), where=tn_pose + fp_pose > 0.0)
        pose_accuracy = np.divide(tp_pose + tn_pose, candidate_counts, out=np.ones(pose_count), where=candidate_counts > 0.0)
        pose_weighted = np.divide(weighted_tp_pose, gt_mass, out=np.ones(pose_count), where=gt_mass > 1e-12)
        pose_useful = np.divide(tn_pose, candidate_counts, out=np.ones(pose_count), where=candidate_counts > 0.0)
        pose_bad = np.divide(fn_pose, candidate_counts, out=np.zeros(pose_count), where=candidate_counts > 0.0)
        tp, fp, fn, tn = (float(value.sum()) for value in (tp_pose, fp_pose, fn_pose, tn_pose))
        candidate_count = float(candidate_counts.sum())
        gt_count = float(gt_counts.sum())
        predicted_count = float(pred_pose.sum())
        weighted_tp = float(weighted_tp_pose.sum())
        weighted_gt = float(gt_mass.sum())
        weighted_recall = _div(weighted_tp, weighted_gt, 1.0)
        lcb = float(weighted_recall_lower_confidence_bound(
            weighted_tp_pose, gt_mass, replicates=bootstrap_replicates,
            seed=bootstrap_seed + len(points) - 1 - reverse_index,
        ))
        safe_workpoint = (
            _safe(weighted_recall, lcb, target_weighted_recall)
            if data.split == "calibration"
            else None if calibration_safe is None
            else calibration_safe.get(float(threshold))
        )
        row: dict[str, Any] = {
            "source": data.source,
            "split": data.split,
            "threshold": float(threshold),
            "threshold_source": threshold_source,
            "calibration_safe": safe_workpoint,
            "weighted_recall": weighted_recall,
            "weighted_recall_lower_confidence_bound": lcb,
            "precision": _div(tp, tp + fp, 1.0),
            "recall": _div(tp, gt_count, 1.0),
            "accuracy": _div(tp + tn, candidate_count, 1.0),
            "balanced_accuracy": 0.5 * (_div(tp, gt_count, 1.0) + _div(tn, tn + fp, 1.0)),
            "specificity": _div(tn, tn + fp, 1.0),
            "avg_pred": _div(predicted_count, pose_count),
            "useful_cull": _div(tn, candidate_count, 1.0),
            "bad_cull": _div(fn, candidate_count),
            "avg_candidate_count": _div(candidate_count, pose_count),
            "avg_gt_count": _div(gt_count, pose_count),
            "avg_pred_glb_bytes": float(predicted_bytes.mean()),
            "avg_candidate_glb_bytes": float(candidate_bytes.mean()),
            "glb_byte_reduction": 1.0 - _div(float(predicted_bytes.sum()), float(candidate_bytes.sum()), 0.0),
            "pose_weighted_recall": float(pose_weighted.mean()),
            "pose_precision": float(pose_precision.mean()),
            "pose_recall": float(pose_recall.mean()),
            "pose_accuracy": float(pose_accuracy.mean()),
            "pose_balanced_accuracy": float((0.5 * (pose_recall + pose_specificity)).mean()),
            "pose_specificity": float(pose_specificity.mean()),
            "pose_useful_cull": float(pose_useful.mean()),
            "pose_bad_cull": float(pose_bad.mean()),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "eval_pose_count": pose_count,
            "zero_gt_pose_count": int(np.count_nonzero(gt_counts == 0.0)),
            "glb_bytes_available": True,
        }
        rows_desc.append(row)
    return list(reversed(rows_desc))


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _plot_source(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = ("split", "threshold", "weighted_recall", "weighted_recall_lower_confidence_bound")
    useful = [{**{key: row[key] for key in keys}, "useful_cull": row["useful_cull"], "bad_cull": row["bad_cull"]} for row in rows]
    glb = [{**{key: row[key] for key in keys}, "avg_pred_glb_bytes": row["avg_pred_glb_bytes"], "avg_candidate_glb_bytes": row["avg_candidate_glb_bytes"]} for row in rows]
    return {"schema": PLOT_SCHEMA, "testRead": False, "x": "weighted_recall", "curves": {"useful_cull": useful, "glb_bytes": glb}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-sidecar", type=Path, required=True)
    parser.add_argument("--validation-sidecar", type=Path, default=None)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-plot-json", type=Path, default=None)
    parser.add_argument("--max-thresholds", type=int, default=256)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--target-weighted-recall", type=float, default=TARGET_WEIGHTED_RECALL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_thresholds <= 0 or args.bootstrap_replicates < 0:
        raise ValueError("max thresholds must be positive and bootstrap replicates non-negative")
    if not 0.0 <= args.target_weighted_recall <= 1.0:
        raise ValueError("target weighted recall must lie in [0, 1]")
    calibration = _valid_sidecar(args.calibration_sidecar, "calibration")
    validation = None if args.validation_sidecar is None else _valid_sidecar(args.validation_sidecar, "validation")
    if validation is not None and calibration.manifest.get("checkpoint") != validation.manifest.get("checkpoint"):
        raise ValueError("calibration and validation sidecars must belong to the same checkpoint")
    resources = load_resource_table(args.runtime_meta, args.glb_index, args.glb_root)
    thresholds = _thresholds(calibration.scores, args.max_thresholds)
    calibration_rows = score_curve(
        calibration, thresholds, resources,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        target_weighted_recall=args.target_weighted_recall,
        threshold_source="calibration_float32_change_points",
    )
    safe_by_threshold = {float(row["threshold"]): bool(row["calibration_safe"]) for row in calibration_rows}
    rows = list(calibration_rows)
    if validation is not None:
        rows.extend(score_curve(
            validation, thresholds, resources,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
            target_weighted_recall=args.target_weighted_recall,
            calibration_safe=safe_by_threshold,
            threshold_source="calibration_frozen_thresholds",
        ))
    plot_path = (args.output_plot_json or args.output_json.with_name(args.output_json.stem + ".plot.json")).resolve()
    plot = _plot_source(rows)
    payload = {
        "schema": CURVE_SCHEMA,
        "version": 2,
        "testRead": False,
        "calibrationSidecar": str(_manifest_path(args.calibration_sidecar)),
        "validationSidecar": None if validation is None else str(_manifest_path(args.validation_sidecar)),
        "thresholds": {
            "source": "calibration scores only",
            "predictionRule": "score >= threshold",
            "changePointCount": int(np.unique(calibration.scores).size),
            "writtenCount": len(thresholds),
            "maxThresholds": int(args.max_thresholds),
        },
        "safety": {
            "metric": "aggregate weighted recall",
            "target": float(args.target_weighted_recall),
            "strict": True,
            "lowerConfidenceBound": "one-sided 95th-percentile pose bootstrap",
            "selectionSplit": "calibration",
            "validationDiagnosticOnly": True,
        },
        "plotSource": str(plot_path),
        "rows": rows,
    }
    _write_csv(args.output_csv.resolve(), rows)
    _write_json(plot_path, plot)
    _write_json(args.output_json.resolve(), payload)
    print(json.dumps({"csv": str(args.output_csv.resolve()), "json": str(args.output_json.resolve()), "plot": str(plot_path), "rowCount": len(rows), "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
