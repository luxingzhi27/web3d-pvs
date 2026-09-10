#!/usr/bin/env python3
"""Evaluate browser Geometry-shell HZB IDs against an existing CSR dataset.

This is intentionally independent from evaluate_pvs.py.  It consumes the HZB
runner's final component IDs, so the browser path never reads candidate depths
or per-candidate debug values back to Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v2"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_u32(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype="<u4")
    return values


def read_u64(path: Path) -> np.ndarray:
    return np.fromfile(path, dtype="<u8")


def safe_ratio(numerator: float, denominator: float, empty: float = 1.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def metrics_from_counts(
    candidate_count: int,
    predicted_count: int,
    gt_count: int,
    tp: int,
) -> dict[str, float | int]:
    fp = int(predicted_count - tp)
    fn = int(gt_count - tp)
    tn = int(candidate_count - tp - fp - fn)
    if min(candidate_count, predicted_count, gt_count, tp, fp, fn) < 0:
        raise ValueError("classification counts must be non-negative")
    if tn < 0:
        raise ValueError("candidate set is smaller than the prediction/GT union")
    recall = safe_ratio(tp, tp + fn)
    specificity = safe_ratio(tn, tn + fp)
    precision = safe_ratio(tp, tp + fp, empty=1.0 if fn == 0 else 0.0)
    f1 = safe_ratio(2 * precision * recall, precision + recall, empty=0.0)
    return {
        "candidateCount": int(candidate_count),
        "predictedCount": int(predicted_count),
        "gtCount": int(gt_count),
        "tp": int(tp),
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": safe_ratio(tp, tp + fp + fn),
        "specificity": specificity,
        "accuracy": safe_ratio(tp + tn, candidate_count),
        "balancedAccuracy": (recall + specificity) * 0.5,
        "usefulCull": safe_ratio(tn, candidate_count),
        "badCull": safe_ratio(fn, candidate_count, empty=0.0),
        "badCullOverGt": safe_ratio(fn, gt_count, empty=0.0),
        "predictedOverCandidate": safe_ratio(predicted_count, candidate_count, empty=0.0),
        "predictedOverGt": safe_ratio(predicted_count, gt_count, empty=0.0),
    }


def set_metrics(predicted: np.ndarray, truth: np.ndarray, candidates: np.ndarray) -> dict[str, float | int]:
    pred = np.unique(np.asarray(predicted, dtype=np.uint32))
    gt = np.unique(np.asarray(truth, dtype=np.uint32))
    candidate = np.unique(np.asarray(candidates, dtype=np.uint32))
    if pred.size and not np.isin(pred, candidate).all():
        raise ValueError("HZB predicted an instance outside its candidate set")
    if gt.size and not np.isin(gt, candidate).all():
        raise ValueError("dataset GT contains an instance outside its candidate set")
    tp = int(np.intersect1d(pred, gt, assume_unique=True).size)
    return metrics_from_counts(int(candidate.size), int(pred.size), int(gt.size), tp)


def iter_jsonl(path_or_dir: Path):
    files = sorted(path_or_dir.glob("*.jsonl")) if path_or_dir.is_dir() else [path_or_dir]
    for path in files:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if line.strip():
                    yield path, line_number, json.loads(line)


def _raw_visible_ids_and_weights(row: dict[str, Any], source: str) -> tuple[np.ndarray, np.ndarray, str]:
    raw_ids = row.get("visible_component_ids") or []
    if not isinstance(raw_ids, list):
        raise ValueError(f"{source} visible_component_ids must be a list")
    ids_list = []
    for value in raw_ids:
        try:
            component_id = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source} contains a non-integer visible component ID") from error
        if component_id < 0 or component_id >= 2**32:
            raise ValueError(f"{source} contains an out-of-range visible component ID")
        ids_list.append(component_id)

    raw_weights = row.get("component_weights")
    if raw_weights is None:
        weights = np.ones((len(ids_list),), dtype=np.float32)
        weight_source = "uniform-visible-instance-fallback"
    else:
        if not isinstance(raw_weights, list) or len(raw_weights) != len(ids_list):
            raise ValueError(f"{source} component_weights must align one-to-one with visible_component_ids")
        weights = np.asarray(raw_weights, dtype=np.float32)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"{source} component_weights must be finite and non-negative")
        weight_source = "raw_three_color_id_component_weights"

    by_id: dict[int, float] = {}
    for component_id, weight in zip(ids_list, weights.tolist()):
        by_id[component_id] = max(by_id.get(component_id, 0.0), float(weight))
    sorted_ids = np.asarray(sorted(by_id), dtype=np.uint32)
    sorted_weights = np.asarray([by_id[int(value)] for value in sorted_ids.tolist()], dtype=np.float32)
    return sorted_ids, sorted_weights, weight_source


def load_point_ground_truth(
    raw_dir: Path,
    needed_viewcell_ids: set[int],
    canonical_subpose_id: int = 0,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    """Load one deterministic raw Color-ID subpose per Point60 view-cell.

    The view-cell CSR intentionally stores a visibility union.  Point60 must
    use a single raw source row instead; the source row is selected by its
    explicit ``subpose_id`` and never by its visible labels.
    """
    if not raw_dir.is_dir():
        raise ValueError(f"Point60 requires an existing raw Color-ID directory: {raw_dir}")
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise ValueError(f"Point60 raw GT directory has no top-level JSONL files: {raw_dir}")
    if canonical_subpose_id != 0:
        raise ValueError("Point60 canonical subpose selection is fixed to subpose_id=0")

    needed = {int(value) for value in needed_viewcell_ids}
    found: dict[int, dict[str, Any]] = {}
    rows_scanned = 0
    rows_without_weights = 0
    for path, line_number, row in iter_jsonl(raw_dir):
        rows_scanned += 1
        try:
            viewcell_id = int(row["viewcell_id"])
            subpose_id = int(row["subpose_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Point60 raw GT row {path}:{line_number} lacks integer viewcell_id/subpose_id") from error
        if viewcell_id not in needed or subpose_id != canonical_subpose_id:
            continue
        if viewcell_id in found:
            raise ValueError(f"Point60 raw GT has duplicate canonical row for viewcell_id {viewcell_id}")
        ids, weights, weight_source = _raw_visible_ids_and_weights(row, f"{path}:{line_number}")
        if weight_source == "uniform-visible-instance-fallback":
            rows_without_weights += 1
        found[viewcell_id] = {
            "visibleIds": ids,
            "visibleWeights": weights,
            "viewcellId": viewcell_id,
            "subposeId": subpose_id,
            "poseIndex": int(row["pose_index"]) if row.get("pose_index") is not None else None,
            "cameraPos": row.get("camera_pos"),
            "viewcellCenter": row.get("viewcell_center"),
        }

    missing = sorted(needed.difference(found))
    if missing:
        raise ValueError(
            "Point60 raw GT is missing canonical subpose_id=0 rows for view-cells: "
            f"{missing[:16]}" + (" ..." if len(missing) > 16 else "")
        )
    summary = {
        "schema": "geometry-shell-hzb-point-gt-v1",
        "enabled": True,
        "source": "raw_three_color_id_jsonl",
        "rawDir": str(raw_dir.resolve()),
        "rawFileCount": len(files),
        "rowsScanned": rows_scanned,
        "requestedViewcellCount": len(needed),
        "matchedViewcellCount": len(found),
        "subposeId": canonical_subpose_id,
        "selection": "fixed-subpose-id-0-per-viewcell",
        "rowsWithoutComponentWeights": rows_without_weights,
        "weightSource": (
            "uniform-visible-instance-fallback"
            if rows_without_weights else "raw_three_color_id_component_weights"
        ),
    }
    return found, summary


def load_glb_sizes(glb_index_path: Path, glb_root: Path) -> tuple[dict[int, int], dict[str, Any]]:
    """Resolve source GLB paths and read their file sizes without loading payloads."""
    index = read_json(glb_index_path)
    entries = index.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("glbIndex.json must contain a non-empty entries array")
    root = glb_root.resolve()
    sizes: dict[int, int] = {}
    for entry in entries:
        try:
            global_id = int(entry["globalId"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("glbIndex entries must contain integer globalId values") from error
        if global_id < 0 or global_id in sizes:
            raise ValueError("glbIndex globalId values must be unique and non-negative")
        relative_path = entry.get("path")
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).is_absolute():
            raise ValueError(f"glbIndex entry {global_id} has an invalid relative path")
        glb_path = (root / relative_path).resolve()
        try:
            glb_path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"glbIndex entry {global_id} escapes --glb-root") from error
        if not glb_path.is_file():
            raise ValueError(f"GLB file for globalId {global_id} does not exist: {glb_path}")
        sizes[global_id] = int(glb_path.stat().st_size)
    return sizes, {
        "glbIndex": str(glb_index_path.resolve()),
        "glbRoot": str(root),
        "entryCount": len(entries),
        "sizeUnit": "source_glb_file_bytes",
    }


def glb_metrics(
    predicted: set[int],
    truth: set[int],
    sizes: dict[int, int] | None = None,
) -> dict[str, float | int]:
    """Return one pose's GLB count metrics, optionally including source bytes."""
    intersection = predicted.intersection(truth)
    result: dict[str, float | int] = {
        "predictedCount": len(predicted),
        "truthCount": len(truth),
        "intersectionCount": len(intersection),
        "countRecall": safe_ratio(len(intersection), len(truth)),
        "predictedOverTruth": safe_ratio(len(predicted), len(truth), empty=0.0),
    }
    if sizes is not None:
        referenced = predicted.union(truth)
        missing = sorted(global_id for global_id in referenced if global_id not in sizes)
        if missing:
            raise ValueError(f"glbIndex is missing referenced global IDs: {missing[:16]}")
        predicted_bytes = sum(sizes[global_id] for global_id in predicted)
        truth_bytes = sum(sizes[global_id] for global_id in truth)
        intersection_bytes = sum(sizes[global_id] for global_id in intersection)
        result.update({
            "predictedBytes": predicted_bytes,
            "truthBytes": truth_bytes,
            "intersectionBytes": intersection_bytes,
            "byteRecall": safe_ratio(intersection_bytes, truth_bytes),
            "predictedOverTruthBytes": safe_ratio(predicted_bytes, truth_bytes, empty=0.0),
        })
    return result


def macro_glb_metrics(per_pose: list[dict[str, float | int]]) -> dict[str, float]:
    """Average GLB metrics per evaluated pose; never substitutes a pose union."""
    if not per_pose:
        raise ValueError("per-pose GLB metrics are required")
    keys = [
        "predictedCount", "truthCount", "intersectionCount", "countRecall", "predictedOverTruth",
        "predictedBytes", "truthBytes", "intersectionBytes", "byteRecall", "predictedOverTruthBytes",
    ]
    return {
        key: float(np.mean([float(item[key]) for item in per_pose]))
        for key in keys
        if key in per_pose[0]
    }


def bootstrap_lower(values: np.ndarray, totals: np.ndarray, seed: int, samples: int = 10_000) -> float:
    if values.size == 0:
        return 1.0
    if values.size == 1:
        return float(values[0] / max(totals[0], 1e-12))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    hit = values[indices].sum(axis=1)
    total = totals[indices].sum(axis=1)
    recalls = hit / np.maximum(total, 1e-12)
    return float(np.quantile(recalls, 0.05, method="linear"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Geometry-shell HZB browser IDs.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=20260909)
    parser.add_argument("--glb-index", type=Path, default=None)
    parser.add_argument("--glb-root", type=Path, default=None)
    parser.add_argument(
        "--point-gt-raw-dir",
        type=Path,
        default=None,
        help="Top-level Color-ID JSONL directory for Point60 canonical subpose GT.",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    result = read_json(args.result)
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError(f"unsupported HZB result schema: {result.get('schema')!r}")
    mode = result.get("mode")
    if mode not in {"Point60", "Region66"}:
        raise ValueError(f"unsupported HZB evaluation mode: {mode!r}")
    samples = result.get("samples") or []
    if not samples:
        raise ValueError("HZB result has no samples")
    point_gt_raw_dir = getattr(args, "point_gt_raw_dir", None)
    if mode == "Point60" and point_gt_raw_dir is None:
        raise ValueError(
            "Point60 accuracy evaluation requires --point-gt-raw-dir; "
            "the dataset visible_ids.bin is a Region66 view-cell union"
        )
    if mode == "Region66" and point_gt_raw_dir is not None:
        raise ValueError("--point-gt-raw-dir is only valid for Point60 evaluation")
    dataset_meta = read_json(args.dataset_dir / "dataset_meta.json")
    runtime_meta = read_json(args.runtime_meta)
    glb_sizes = None
    glb_source = None
    if (args.glb_index is None) != (args.glb_root is None):
        raise ValueError("--glb-index and --glb-root must be provided together")
    if args.glb_index is not None and args.glb_root is not None:
        glb_sizes, glb_source = load_glb_sizes(args.glb_index, args.glb_root)
    pose_count = int(dataset_meta.get("poseCount", dataset_meta.get("viewcellCount", 0)))
    offsets = read_u64(args.dataset_dir / "candidate_offsets.bin")
    candidates = read_u32(args.dataset_dir / "candidate_ids.bin")
    if offsets.size != pose_count + 1:
        raise ValueError("dataset CSR offset lengths do not match pose count")
    if int(offsets[-1]) != candidates.size:
        raise ValueError("dataset candidate CSR offsets do not match data lengths")
    visible_offsets = None
    visible_ids = None
    visible_weights = None
    point_ground_truth = None
    point_ground_truth_summary = None
    if mode == "Region66":
        visible_offsets = read_u64(args.dataset_dir / "visible_offsets.bin")
        visible_ids = read_u32(args.dataset_dir / "visible_ids.bin")
        visible_weights_path = args.dataset_dir / "visible_weights.bin"
        visible_weights = np.fromfile(visible_weights_path, dtype="<f4") if visible_weights_path.exists() else None
        if visible_offsets.size != pose_count + 1:
            raise ValueError("dataset visible CSR offset lengths do not match pose count")
        if int(visible_offsets[-1]) != visible_ids.size:
            raise ValueError("dataset visible CSR offsets do not match data lengths")
        if visible_weights is not None and visible_weights.size != visible_ids.size:
            raise ValueError("visible_weights.bin length does not match visible_ids.bin")
        if visible_weights is not None and (
            not np.isfinite(visible_weights).all() or (visible_weights < 0).any()
        ):
            raise ValueError("visible_weights.bin must contain finite non-negative values")
    records = runtime_meta.get("componentRecords", [])
    mapping = np.full(int(dataset_meta.get("numInstances", 0)), -1, dtype=np.int64)
    for record in records:
        component_id = int(record["componentGlobalId"])
        if component_id < 0 or component_id >= mapping.size or mapping[component_id] >= 0:
            raise ValueError("runtime component IDs must be unique and dense")
        mapping[component_id] = int(record["globalGlbId"])
    if mapping.size == 0 or (mapping < 0).any():
        raise ValueError("runtime metadata has no dense component-to-GLB mapping")
    if mode == "Point60":
        sample_pose_ids = {int(sample["poseId"]) for sample in samples}
        if any(pose_id < 0 or pose_id >= pose_count for pose_id in sample_pose_ids):
            raise ValueError("Point60 result contains a poseId outside the dataset")
        point_ground_truth, point_ground_truth_summary = load_point_ground_truth(
            point_gt_raw_dir,
            sample_pose_ids,
        )
    split = args.split if args.split is not None else result.get("workload", {}).get("split")
    if split is None:
        split = "unknown"
    per_pose: list[dict[str, Any]] = []
    weighted_hits: list[float] = []
    weighted_totals: list[float] = []
    totals: dict[str, int] = {key: 0 for key in ("candidateCount", "predictedCount", "gtCount", "tp", "fp", "fn", "tn")}
    glb_per_pose: list[dict[str, float | int]] = []
    glb_predicted_union: set[int] = set()
    glb_truth_union: set[int] = set()
    for sample in samples:
        pose_id = int(sample["poseId"])
        if pose_id < 0 or pose_id >= pose_count:
            raise ValueError(f"HZB result poseId {pose_id} is out of range")
        candidate = candidates[int(offsets[pose_id]): int(offsets[pose_id + 1])]
        if mode == "Point60":
            point_truth = point_ground_truth[pose_id]
            truth = point_truth["visibleIds"]
            truth_weights = point_truth["visibleWeights"]
        else:
            truth = visible_ids[int(visible_offsets[pose_id]): int(visible_offsets[pose_id + 1])]
            truth_weights = (
                visible_weights[int(visible_offsets[pose_id]): int(visible_offsets[pose_id + 1])]
                if visible_weights is not None else None
            )
        predicted = np.asarray(sample.get("visibleInstanceIds", []), dtype=np.uint32)
        metrics = set_metrics(predicted, truth, candidate)
        metrics["poseId"] = pose_id
        metrics["uncertainCount"] = int(sample.get("uncertainCount", 0))
        per_pose.append(metrics)
        for key in totals:
            totals[key] += int(metrics[key])
        if truth_weights is None:
            weighted_hits.append(float(metrics["tp"]))
            weighted_totals.append(float(metrics["gtCount"]))
        else:
            predicted_set = set(int(value) for value in predicted.tolist())
            weighted_hits.append(float(sum(float(weight) for value, weight in zip(truth.tolist(), truth_weights.tolist()) if int(value) in predicted_set)))
            weighted_totals.append(float(truth_weights.sum()))
        if predicted.size and (predicted >= mapping.size).any():
            raise ValueError(f"HZB result poseId {pose_id} contains an out-of-range instance")
        if truth.size and (truth >= mapping.size).any():
            raise ValueError(f"dataset GT poseId {pose_id} contains an out-of-range instance")
        predicted_glb = {int(mapping[int(value)]) for value in predicted.tolist()}
        truth_glb = {int(mapping[int(value)]) for value in truth.tolist()}
        pose_glb = glb_metrics(predicted_glb, truth_glb, glb_sizes)
        pose_glb["poseId"] = pose_id
        glb_per_pose.append(pose_glb)
        glb_predicted_union.update(predicted_glb)
        glb_truth_union.update(truth_glb)
    aggregate = metrics_from_counts(
        totals["candidateCount"],
        totals["predictedCount"],
        totals["gtCount"],
        totals["tp"],
    )
    weighted_hits_array = np.asarray(weighted_hits, dtype=np.float64)
    weighted_totals_array = np.asarray(weighted_totals, dtype=np.float64)
    weighted_recall = safe_ratio(weighted_hits_array.sum(), weighted_totals_array.sum())
    if mode == "Point60":
        ground_truth = {
            "mode": "canonical-subpose",
            "source": "raw_three_color_id_jsonl",
            "subposeId": int(point_ground_truth_summary["subposeId"]),
            "selection": point_ground_truth_summary["selection"],
            "rawDir": point_ground_truth_summary["rawDir"],
            "summary": point_ground_truth_summary,
        }
        weight_source = (
            "raw_three_color_id_component_weights"
            if point_ground_truth_summary["weightSource"] == "raw_three_color_id_component_weights"
            else "raw-three-color-id-uniform-visible-instance-fallback"
        )
    else:
        ground_truth = {
            "mode": "viewcell-union",
            "source": "dataset_visible_ids_bin",
            "semantics": dataset_meta.get("gtSemantics"),
        }
        weight_source = "visible_weights.bin" if visible_weights is not None else "uniform-visible-instance-fallback"
    glb_output: dict[str, Any] = {
        "poseMacro": macro_glb_metrics(glb_per_pose),
        "perPose": glb_per_pose,
        "unionDiagnostic": glb_metrics(glb_predicted_union, glb_truth_union, glb_sizes),
    }
    if glb_source is not None:
        glb_output["byteSource"] = glb_source
    output = {
        "schema": "geometry-shell-hzb-metrics-v2",
        "sourceResult": args.result.name,
        "dataset": args.dataset_dir.name,
        "split": split,
        "mode": result.get("mode"),
        "evaluatedPoseCount": len(samples),
        "datasetPoseCount": pose_count,
        "aggregate": aggregate,
        "poseMacro": {
            key: float(np.mean([float(item[key]) for item in per_pose]))
            for key in ("precision", "recall", "f1", "jaccard", "specificity", "accuracy", "balancedAccuracy", "usefulCull", "badCull", "badCullOverGt", "predictedOverCandidate")
        },
        "weightedRecall": weighted_recall,
        "weightedRecallLower95": bootstrap_lower(weighted_hits_array, weighted_totals_array, int(args.bootstrap_seed)),
        "weightedRecallWeightSource": weight_source,
        "groundTruth": ground_truth,
        "glb": glb_output,
        "totals": totals,
        "perPose": per_pose,
        "timing": result.get("summary"),
        "gpuBackend": result.get("gpuBackend"),
        "adapterInfo": result.get("adapterInfo"),
        "webglInfo": result.get("webglInfo"),
        "gpuGate": result.get("gpuGate"),
        "formalReady": bool(result.get("formalReady", False)),
        "executionClass": result.get("executionClass"),
        "gpuConcurrency": result.get("gpuConcurrency"),
    }
    return output


def main() -> None:
    args = parse_args()
    output = evaluate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "evaluatedPoseCount": output["evaluatedPoseCount"],
        "aggregate": output["aggregate"],
        "glbPoseMacro": output["glb"]["poseMacro"],
        "weightedRecall": output["weightedRecall"],
        "weightedRecallLower95": output["weightedRecallLower95"],
        "formalReady": output["formalReady"],
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
