#!/usr/bin/env python3
"""Audit per-epoch training history for finite and skipped optimization steps.

The training loop records non-finite loss and gradient skips per epoch.  This
tool sums those counters across the complete history so a final clean epoch
cannot hide an earlier unstable step.  It does not alter checkpoints or
declare a model suitable for the paper; it only produces provenance for the
experiment report.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _history_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = payload.get("history") or payload.get("epochs") or []
    else:
        rows = []
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"training history is not a list of objects: {path}")
    if not rows:
        raise ValueError(f"training history is empty: {path}")
    return rows


def _nonfinite_paths(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [] if math.isfinite(float(value)) else [prefix or "value"]
    if isinstance(value, dict):
        result: list[str] = []
        for key, child in value.items():
            result.extend(_nonfinite_paths(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            result.extend(_nonfinite_paths(child, f"{prefix}[{index}]"))
        return result
    return []


def audit_history(path: Path, expected_epochs: int | None = None) -> dict[str, Any]:
    rows = _history_rows(path)
    epochs = [int(row["epoch"]) for row in rows if "epoch" in row]
    if len(epochs) != len(rows) or epochs != sorted(set(epochs)):
        raise ValueError(f"epoch sequence is missing or not strictly increasing: {path}")
    if expected_epochs is not None and len(rows) != expected_epochs:
        raise ValueError(f"expected {expected_epochs} epochs, found {len(rows)}: {path}")

    loss_skips = sum(int(row.get("trainSkippedNonFiniteLoss", 0) or 0) for row in rows)
    grad_skips = sum(int(row.get("trainSkippedNonFiniteGrad", 0) or 0) for row in rows)
    steps = sum(int(row.get("stepsPerEpoch", 0) or 0) for row in rows)
    nonfinite = []
    for row in rows:
        nonfinite.extend(_nonfinite_paths(row))

    if nonfinite or loss_skips:
        status = "fail"
    elif grad_skips:
        status = "warning"
    else:
        status = "pass"
    return {
        "history": str(path.resolve()),
        "epochCount": len(rows),
        "firstEpoch": epochs[0],
        "lastEpoch": epochs[-1],
        "stepsPerEpoch": sorted({int(row.get("stepsPerEpoch", 0) or 0) for row in rows}),
        "totalRecordedSteps": steps,
        "trainSkippedNonFiniteLossTotal": loss_skips,
        "trainSkippedNonFiniteGradTotal": grad_skips,
        "nonFiniteMetricCount": len(nonfinite),
        "nonFiniteMetricExamples": nonfinite[:20],
        "gradientSkipRate": grad_skips / steps if steps else None,
        "lossSkipRate": loss_skips / steps if steps else None,
        "status": status,
        "interpretation": {
            "pass": "No non-finite loss/gradient skip was recorded.",
            "warning": "Optimization remained finite after skipped gradient steps; use a separate stable rerun for a clean claim.",
            "fail": "Non-finite loss or metric was recorded; do not use this run as the clean training evidence.",
        }[status],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="append", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [audit_history(path, args.expected_epochs) for path in args.history]
    payload = {
        "schema": "training-stability-audit-v1",
        "status": "fail" if any(row["status"] == "fail" for row in reports) else (
            "warning" if any(row["status"] == "warning" for row in reports) else "pass"
        ),
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
