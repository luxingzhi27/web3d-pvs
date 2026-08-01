#!/usr/bin/env python3
"""Summarize the M12 PyTorch/FP16/WGSL parity capture.

The browser capture contains the production packed scores plus two opt-in raw
logits.  This script keeps the comparison separate from training and reports
both numeric parity and the decision-level consequences at the frozen model
threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _float_or_none(value: Any) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _percentile(values: list[float] | np.ndarray, percentile: float) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return _float_or_none(np.percentile(array, percentile))


def _mean(values: list[float] | np.ndarray) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return None
    return _float_or_none(np.mean(array))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _decode_capture(raw_output: list[int]) -> dict[str, np.ndarray]:
    raw = np.asarray(raw_output, dtype=np.uint32)
    if raw.size % 4 != 0:
        raise ValueError(f"M12 raw output has {raw.size} words; expected a multiple of 4")
    words = raw.reshape(-1, 4)
    visibility_q16 = words[:, 0] >> np.uint32(1)
    download_q16 = words[:, 1]
    return {
        "visibilityProbability": visibility_q16.astype(np.float64) / 65535.0,
        "downloadProbability": download_q16.astype(np.float64) / 65535.0,
        "visibilityLogit": words[:, 2].view(np.float32).astype(np.float64),
        "downloadLogit": words[:, 3].view(np.float32).astype(np.float64),
        "visibleFlag": (words[:, 0] & np.uint32(1)) != 0,
    }


def _safe_set_jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _binary_metrics(predicted: set[int], visible: set[int], candidate_count: int) -> dict[str, Any]:
    tp = len(predicted & visible)
    fp = len(predicted - visible)
    fn = len(visible - predicted)
    tn = max(0, int(candidate_count) - tp - fp - fn)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    accuracy = (tp + tn) / candidate_count if candidate_count else 1.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "jaccard": _safe_set_jaccard(predicted, visible),
    }


def _weighted_recall(predicted: set[int], visible_ids: list[int], visible_weights: list[float]) -> float:
    weights = np.asarray(visible_weights, dtype=np.float64)
    ids = np.asarray(visible_ids, dtype=np.int64)
    if ids.size == 0 or weights.size != ids.size:
        return 1.0 if ids.size == 0 else float("nan")
    total = float(np.sum(weights))
    if total <= 0.0:
        return 1.0
    return float(np.sum(weights[np.isin(ids, list(predicted))]) / total)


def _rank_map(candidate_ids: list[int], scores: np.ndarray, instance_to_glb: list[int]) -> dict[int, float]:
    grouped: dict[int, float] = {}
    for instance_id, score in zip(candidate_ids, scores.tolist()):
        if instance_id < 0 or instance_id >= len(instance_to_glb):
            continue
        glb_id = int(instance_to_glb[instance_id])
        grouped[glb_id] = max(grouped.get(glb_id, -np.inf), float(score))
    return grouped


def _rank_vector(values: dict[int, float], keys: list[int]) -> np.ndarray:
    ordered = sorted(keys, key=lambda key: (-values[key], key))
    ranks = {key: rank for rank, key in enumerate(ordered)}
    return np.asarray([ranks[key] for key in keys], dtype=np.float64)


def _spearman(left: dict[int, float], right: dict[int, float]) -> float | None:
    keys = sorted(set(left) & set(right))
    if len(keys) < 2:
        return None
    x = _rank_vector(left, keys)
    y = _rank_vector(right, keys)
    x -= np.mean(x)
    y -= np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return None if denominator <= 0.0 else _float_or_none(float(np.dot(x, y) / denominator))


def _top_k_overlap(left: dict[int, float], right: dict[int, float], fraction: float = 0.1) -> float | None:
    keys = sorted(set(left) & set(right))
    if not keys:
        return None
    count = max(1, int(np.ceil(len(keys) * fraction)))
    top_left = {key for key in sorted(keys, key=lambda key: (-left[key], key))[:count]}
    top_right = {key for key in sorted(keys, key=lambda key: (-right[key], key))[:count]}
    return _safe_set_jaccard(top_left, top_right)


def _error_summary(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | None]:
    error = np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    return _absolute_error_summary(error)


def _absolute_error_summary(error: np.ndarray | list[float]) -> dict[str, float | None]:
    error = np.asarray(error, dtype=np.float64)
    return {
        "meanAbs": _mean(error),
        "p95Abs": _percentile(error, 95),
        "p99Abs": _percentile(error, 99),
        "maxAbs": _float_or_none(np.max(error)) if error.size else None,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    cases_payload = _load_json(Path(args.cases))
    reference_payload = _load_json(Path(args.reference))
    capture_payload = _load_json(Path(args.capture))
    model_meta = _load_json(Path(args.model_meta))
    instance_to_glb = [int(value) for value in model_meta.get("instanceToGlobalGlb", [])]
    if not instance_to_glb:
        raise ValueError("model metadata has no instanceToGlobalGlb mapping")

    specs = {int(item["caseId"]): item for item in cases_payload.get("cases", [])}
    fp32 = {int(item["caseId"]): item for item in reference_payload.get("cases", [])}
    fp16 = {int(item["caseId"]): item for item in reference_payload.get("fp16RoundedCases", [])}
    captured = {int(item["caseId"]): item for item in capture_payload.get("probe", {}).get("results", [])}
    case_ids = sorted(set(specs) & set(fp32) & set(fp16) & set(captured))
    missing = {
        "cases": sorted(set(specs) - set(captured)),
        "fp32": sorted(set(specs) - set(fp32)),
        "fp16": sorted(set(specs) - set(fp16)),
    }
    threshold = float(args.threshold if args.threshold is not None else cases_payload.get("threshold", 0.02))
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {
        "visibilityLogitVsFp32": [],
        "visibilityLogitVsFp16": [],
        "downloadLogitVsFp32": [],
        "downloadLogitVsFp16": [],
        "visibilityProbabilityVsFp32": [],
        "visibilityProbabilityVsFp16": [],
        "downloadProbabilityVsFp32": [],
        "downloadProbabilityVsFp16": [],
    }
    total_flip_fp32 = total_flip_fp16 = total_values = 0
    weighted_values = {"wgsl": [], "fp32": [], "fp16": []}
    valid = not missing["cases"] and not missing["fp32"] and not missing["fp16"]

    for case_id in case_ids:
        spec = specs[case_id]
        ref32 = fp32[case_id]
        ref16 = fp16[case_id]
        result = captured[case_id]
        candidate_ids = [int(value) for value in spec.get("candidateIds", [])]
        captured_ids = [int(value) for value in result.get("candidateIds", [])]
        ids_match = candidate_ids == captured_ids
        if not ids_match:
            valid = False
        actual = _decode_capture(result.get("rawOutput", []))
        n = len(candidate_ids)
        if actual["visibilityLogit"].size != n:
            valid = False
            continue
        ref32_vis_logit = np.asarray(ref32["visibilityLogit"], dtype=np.float64)
        ref16_vis_logit = np.asarray(ref16["visibilityLogit"], dtype=np.float64)
        ref32_dl_logit = np.asarray(ref32["downloadLogit"], dtype=np.float64)
        ref16_dl_logit = np.asarray(ref16["downloadLogit"], dtype=np.float64)
        ref32_vis_prob = np.asarray(ref32["visibilityProbability"], dtype=np.float64)
        ref16_vis_prob = np.asarray(ref16["visibilityProbability"], dtype=np.float64)
        ref32_dl_prob = np.asarray(ref32["downloadProbability"], dtype=np.float64)
        ref16_dl_prob = np.asarray(ref16["downloadProbability"], dtype=np.float64)
        arrays_match = all(array.size == n for array in [ref32_vis_logit, ref16_vis_logit, ref32_dl_logit, ref16_dl_logit, ref32_vis_prob, ref16_vis_prob, ref32_dl_prob, ref16_dl_prob])
        if not arrays_match:
            valid = False
            continue

        for name, actual_key, ref_key, reference in [
            ("visibilityLogitVsFp32", "visibilityLogit", "visibilityLogit", ref32_vis_logit),
            ("visibilityLogitVsFp16", "visibilityLogit", "visibilityLogit", ref16_vis_logit),
            ("downloadLogitVsFp32", "downloadLogit", "downloadLogit", ref32_dl_logit),
            ("downloadLogitVsFp16", "downloadLogit", "downloadLogit", ref16_dl_logit),
            ("visibilityProbabilityVsFp32", "visibilityProbability", "visibilityProbability", ref32_vis_prob),
            ("visibilityProbabilityVsFp16", "visibilityProbability", "visibilityProbability", ref16_vis_prob),
            ("downloadProbabilityVsFp32", "downloadProbability", "downloadProbability", ref32_dl_prob),
            ("downloadProbabilityVsFp16", "downloadProbability", "downloadProbability", ref16_dl_prob),
        ]:
            del ref_key
            errors[name].extend(np.abs(actual[actual_key] - reference).tolist())

        wgsl_pred = {candidate_ids[index] for index, flag in enumerate(actual["visibleFlag"]) if flag}
        fp32_pred = {candidate_ids[index] for index, score in enumerate(ref32_vis_prob) if score >= threshold}
        fp16_pred = {candidate_ids[index] for index, score in enumerate(ref16_vis_prob) if score >= threshold}
        visible_ids = [int(value) for value in spec.get("visibleIds", [])]
        visible_weights = [float(value) for value in spec.get("visibleWeights", [])]
        visible_set = set(visible_ids)
        wgsl_weighted = _weighted_recall(wgsl_pred, visible_ids, visible_weights)
        fp32_weighted = _weighted_recall(fp32_pred, visible_ids, visible_weights)
        fp16_weighted = _weighted_recall(fp16_pred, visible_ids, visible_weights)
        weighted_values["wgsl"].append(wgsl_weighted)
        weighted_values["fp32"].append(fp32_weighted)
        weighted_values["fp16"].append(fp16_weighted)
        total_flip_fp32 += int(np.count_nonzero(actual["visibleFlag"] != (ref32_vis_prob >= threshold)))
        total_flip_fp16 += int(np.count_nonzero(actual["visibleFlag"] != (ref16_vis_prob >= threshold)))
        total_values += n

        wgsl_rank = _rank_map(candidate_ids, actual["downloadProbability"], instance_to_glb)
        fp32_rank = _rank_map(candidate_ids, ref32_dl_prob, instance_to_glb)
        fp16_rank = _rank_map(candidate_ids, ref16_dl_prob, instance_to_glb)
        rows.append({
            "caseId": case_id,
            "poseIndex": int(spec.get("poseIndex", -1)),
            "candidateCount": n,
            "gtVisibleCount": len(visible_set),
            "candidateIdsMatch": ids_match,
            "visibility": {
                "wgsl": _binary_metrics(wgsl_pred, visible_set, n),
                "fp32": _binary_metrics(fp32_pred, visible_set, n),
                "fp16": _binary_metrics(fp16_pred, visible_set, n),
                "wgslWeightedRecall": _float_or_none(wgsl_weighted),
                "fp32WeightedRecall": _float_or_none(fp32_weighted),
                "fp16WeightedRecall": _float_or_none(fp16_weighted),
                "jaccardWgslVsFp32": _safe_set_jaccard(wgsl_pred, fp32_pred),
                "jaccardWgslVsFp16": _safe_set_jaccard(wgsl_pred, fp16_pred),
                "thresholdFlipCountVsFp32": int(np.count_nonzero(actual["visibleFlag"] != (ref32_vis_prob >= threshold))),
                "thresholdFlipCountVsFp16": int(np.count_nonzero(actual["visibleFlag"] != (ref16_vis_prob >= threshold))),
            },
            "downloadRanking": {
                "glbCountWgsl": len(wgsl_rank),
                "glbCountFp32": len(fp32_rank),
                "glbCountFp16": len(fp16_rank),
                "spearmanWgslVsFp32": _spearman(wgsl_rank, fp32_rank),
                "spearmanWgslVsFp16": _spearman(wgsl_rank, fp16_rank),
                "top10PercentJaccardWgslVsFp32": _top_k_overlap(wgsl_rank, fp32_rank),
                "top10PercentJaccardWgslVsFp16": _top_k_overlap(wgsl_rank, fp16_rank),
            },
            "timings": result.get("timings"),
        })

    aggregate: dict[str, Any] = {
        "caseCount": len(rows),
        "candidateValueCount": total_values,
        "threshold": threshold,
        "thresholdFlipRateVsFp32": total_flip_fp32 / total_values if total_values else None,
        "thresholdFlipRateVsFp16": total_flip_fp16 / total_values if total_values else None,
        "weightedRecallMean": {key: _mean(value) for key, value in weighted_values.items()},
        "weightedRecallMin": {key: _float_or_none(np.min(value)) if value else None for key, value in weighted_values.items()},
        "visibilityLogitError": {},
        "downloadLogitError": {},
        "visibilityProbabilityError": {},
        "downloadProbabilityError": {},
        "downloadRanking": {
            "spearmanMeanWgslVsFp32": _mean([row["downloadRanking"]["spearmanWgslVsFp32"] for row in rows if row["downloadRanking"]["spearmanWgslVsFp32"] is not None]),
            "spearmanMeanWgslVsFp16": _mean([row["downloadRanking"]["spearmanWgslVsFp16"] for row in rows if row["downloadRanking"]["spearmanWgslVsFp16"] is not None]),
            "top10PercentJaccardMeanWgslVsFp32": _mean([row["downloadRanking"]["top10PercentJaccardWgslVsFp32"] for row in rows if row["downloadRanking"]["top10PercentJaccardWgslVsFp32"] is not None]),
            "top10PercentJaccardMeanWgslVsFp16": _mean([row["downloadRanking"]["top10PercentJaccardWgslVsFp16"] for row in rows if row["downloadRanking"]["top10PercentJaccardWgslVsFp16"] is not None]),
        },
    }
    for metric_name, target in [
        ("visibilityLogit", "visibilityLogitError"),
        ("downloadLogit", "downloadLogitError"),
        ("visibilityProbability", "visibilityProbabilityError"),
        ("downloadProbability", "downloadProbabilityError"),
    ]:
        for reference_name in ("Fp32", "Fp16"):
            aggregate[target][reference_name] = _absolute_error_summary(errors[f"{metric_name}Vs{reference_name}"])

    output = {
        "schema": "m12-webgpu-parity-summary-v1",
        "valid": bool(valid and capture_payload.get("probe", {}).get("backend") == "webgpu" and not capture_payload.get("pageErrors")),
        "cases": str(Path(args.cases)),
        "reference": str(Path(args.reference)),
        "capture": str(Path(args.capture)),
        "modelMeta": str(Path(args.model_meta)),
        "browser": capture_payload.get("browser"),
        "adapterInfo": capture_payload.get("adapterInfo"),
        "frontendReady": capture_payload.get("frontendReady"),
        "probeBackend": capture_payload.get("probe", {}).get("backend"),
        "pageErrors": capture_payload.get("pageErrors", []),
        "missing": missing,
        "aggregate": aggregate,
        "rows": rows,
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize M12 PyTorch/FP16/WGSL parity output")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--model-meta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    summary = summarize(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "valid": summary["valid"],
        "probeBackend": summary["probeBackend"],
        "caseCount": summary["aggregate"]["caseCount"],
        "thresholdFlipRateVsFp16": summary["aggregate"]["thresholdFlipRateVsFp16"],
        "weightedRecallMeanWgsl": summary["aggregate"]["weightedRecallMean"]["wgsl"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
