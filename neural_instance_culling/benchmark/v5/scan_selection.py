"""Validation-only ranking for the frozen V5 optimizer parameter scan."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SINGLE_RUN_SCHEMA = "gcof-pvs-v5-single-run-validation-v1"
SCAN_SELECTION_SCHEMA = "gcof-pvs-v5-parameter-scan-selection-v1"


def summarize_single_shared_run(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    if len(materialized) != 5 or len({str(row.get("scene")) for row in materialized}) != 5:
        raise ValueError("single-run scan evaluation requires exactly five scene rows")
    identity = {
        (str(row.get("protocol")), str(row.get("variant")), int(row.get("seed", -1)))
        for row in materialized
    }
    if len(identity) != 1 or next(iter(identity))[0] != "shared":
        raise ValueError("scan rows must belong to one shared checkpoint")
    if any(
        row.get("split") != "validation"
        or row.get("selection_split") != "calibration"
        or row.get("selection_test_read") is not False
        or row.get("test_read") is not False
        for row in materialized
    ):
        raise ValueError("scan ranking must be calibration-selected validation-only")
    qualifications = [str(row.get("qualification")) for row in materialized]
    allowed = {"strict_lcb_target", "mean_target", "diagnostic"}
    if any(value not in allowed for value in qualifications):
        raise ValueError("scan row has an unknown safety qualification")

    def metric(name: str) -> list[float | None]:
        values: list[float | None] = []
        for row in materialized:
            metrics = row.get("metrics")
            if not isinstance(metrics, Mapping):
                raise ValueError("scan row has no metrics")
            value = metrics.get(name)
            if value is None:
                values.append(None)
                continue
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"scan metric {name} is non-finite")
            values.append(numeric)
        return values

    wr_lcb = metric("weighted_recall_lcb")
    cnor = metric("cnor")
    useful = metric("useful_cull")
    pred_over_gt = metric("pred_over_gt")
    if any(value is None for value in cnor + useful + pred_over_gt):
        raise ValueError("scan ranking metrics CNOR/Useful Cull/predicted-over-GT are required")
    protocol, variant, seed = next(iter(identity))
    return {
        "schema": SINGLE_RUN_SCHEMA,
        "protocol": protocol,
        "variant": variant,
        "seed": seed,
        "split": "validation",
        "selectionSplit": "calibration",
        "testRead": False,
        "sceneCount": 5,
        "strictLcbSceneCount": sum(value == "strict_lcb_target" for value in qualifications),
        "meanTargetSceneCount": sum(value in {"strict_lcb_target", "mean_target"} for value in qualifications),
        "sceneEqualWeightedRecallLcb": (
            None if any(value is None for value in wr_lcb)
            else float(np.mean([float(value) for value in wr_lcb]))
        ),
        "sceneEqualCnor": float(np.mean([float(value) for value in cnor])),
        "sceneEqualUsefulCull": float(np.mean([float(value) for value in useful])),
        "sceneEqualPredictedOverGt": float(np.mean([float(value) for value in pred_over_gt])),
        "rows": materialized,
    }


def scan_rank_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    if summary.get("schema") != SINGLE_RUN_SCHEMA or summary.get("testRead") is not False:
        raise ValueError("scan ranking requires a validation-only single-run summary")
    lcb = summary.get("sceneEqualWeightedRecallLcb")
    return (
        float(summary["strictLcbSceneCount"]),
        float(summary["meanTargetSceneCount"]),
        -np.inf if lcb is None else float(lcb),
        float(summary["sceneEqualCnor"]),
        float(summary["sceneEqualUsefulCull"]),
        -float(summary["sceneEqualPredictedOverGt"]),
    )


def select_scan_candidates(
    scan_matrix: Mapping[str, Any],
    summaries: Mapping[str, Mapping[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    if scan_matrix.get("schema") != "gcof-pvs-v5-parameter-scan-matrix-v1":
        raise ValueError("unsupported V5 scan matrix")
    if scan_matrix.get("testRead") is not False:
        raise ValueError("test-tainted scan matrix")
    rows = scan_matrix.get("runs")
    if not isinstance(rows, list) or not rows:
        raise ValueError("scan matrix has no runs")
    names = [str(row.get("name")) for row in rows]
    if len(names) != len(set(names)) or set(names) != set(summaries):
        raise ValueError("scan summaries must cover every planned configuration exactly once")
    if top_k <= 0 or top_k > len(names):
        raise ValueError("top_k is outside the scan matrix")
    ranking = sorted(
        names,
        key=lambda name: (scan_rank_key(summaries[name]), name),
        reverse=True,
    )
    return {
        "schema": SCAN_SELECTION_SCHEMA,
        "phase": scan_matrix.get("phase"),
        "selectionSplit": "validation",
        "thresholdSplit": "calibration",
        "testRead": False,
        "selectionRule": list(scan_matrix.get("selectionRule", [])),
        "selectedConfigurations": ranking[:top_k],
        "ranking": [
            {"rank": index + 1, "name": name, "summary": dict(summaries[name])}
            for index, name in enumerate(ranking)
        ],
    }


def read_json(path: str | Path) -> Mapping[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


__all__ = [
    "SCAN_SELECTION_SCHEMA",
    "SINGLE_RUN_SCHEMA",
    "scan_rank_key",
    "select_scan_candidates",
    "summarize_single_shared_run",
]
