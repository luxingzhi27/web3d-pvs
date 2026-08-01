#!/usr/bin/env python3
"""Finalize checkpoint provenance after a training-side one-shot evaluation.

The training process writes the final calibration/bootstrap result and the
one-shot test summary after it has selected ``best.pt``.  Older processes did
not copy that final calibration record back into the checkpoint.  This tool
repairs only checkpoint metadata; it never loads a CSR split or performs
inference, so it cannot create a second test evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_close(left: Any, right: Any) -> bool:
    return abs(float(left) - float(right)) <= 1e-7


def finalize(checkpoint_path: Path, summary_path: Path, output_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("protocol") != "frozen_calibration_one_shot_test":
        raise ValueError("The summary is not a frozen calibration one-shot test summary.")
    if int(summary.get("testEvaluationCount", 0)) != 1:
        raise ValueError("The summary does not record exactly one test evaluation.")
    protocol = checkpoint.get("protocolSplit")
    summary_protocol = summary.get("protocolSplit")
    if not isinstance(protocol, dict) or not isinstance(summary_protocol, dict):
        raise ValueError("Checkpoint and summary must both contain protocolSplit provenance.")
    for key in ("fixedValidationDigest", "calibrationDigest", "frozenTestDigest"):
        if protocol.get(key) != summary_protocol.get(key):
            raise ValueError(f"Protocol digest mismatch for {key}.")

    calibration = summary.get("calibration")
    validation = summary.get("validationAtFrozenThreshold")
    threshold = summary.get("frozenThreshold")
    if not isinstance(calibration, dict) or not isinstance(validation, dict) or threshold is None:
        raise ValueError("Summary lacks final calibration, frozen validation, or threshold records.")
    selected = calibration.get("selected")
    bootstrap = calibration.get("bootstrap")
    if not isinstance(selected, dict) or not isinstance(bootstrap, dict):
        raise ValueError("Summary lacks selected calibration/bootstrap provenance.")
    if int(bootstrap.get("replicates", 0)) <= 0:
        raise ValueError("Final calibration has no bootstrap replicates.")
    lcb_floor = bootstrap.get("lowerBoundFloor")
    selected_lcb = selected.get("weighted_recall_lower_confidence_bound")
    if lcb_floor is None or selected_lcb is None or float(selected_lcb) <= float(lcb_floor):
        raise ValueError("Final calibration does not satisfy its registered lower-bound floor.")
    point_floor = float(checkpoint.get("args", {}).get("calibration_point_floor", 0.9925))
    if float(selected.get("pose_weighted_recall", -1.0)) < point_floor:
        raise ValueError("Final calibration does not satisfy its point-estimate floor.")
    if not threshold_close(threshold, selected.get("threshold")):
        raise ValueError("Summary frozenThreshold differs from selected calibration threshold.")
    if not threshold_close(threshold, validation.get("threshold")):
        raise ValueError("Validation record is not evaluated at the frozen threshold.")

    old_best = checkpoint.get("best")
    if not isinstance(old_best, dict):
        raise ValueError("Checkpoint has no validation-selected best record.")
    checkpoint_selection = dict(checkpoint.get("checkpointSelection") or old_best)
    if int(checkpoint_selection.get("epoch", -1)) < 0:
        raise ValueError("Checkpoint selection has no epoch.")
    if int(validation.get("eval_pose_count", -1)) != int(protocol.get("fixedValidationCount", -2)):
        raise ValueError("Final frozen validation does not cover the complete validation split.")
    if int(selected.get("eval_pose_count", -1)) != int(protocol.get("calibrationCount", -2)):
        raise ValueError("Final calibration does not cover the complete calibration split.")

    checkpoint["checkpointSelection"] = checkpoint_selection
    checkpoint["best"] = {
        "epoch": int(checkpoint_selection["epoch"]),
        "threshold": float(threshold),
        **validation,
        "selectionMetrics": checkpoint_selection,
    }
    checkpoint["workpoints"] = {
        "calibration": calibration,
        "validationAtFrozenThreshold": validation,
        "frozenThreshold": float(threshold),
        "selectionRule": summary.get("selectionRule"),
    }
    checkpoint["finalCalibration"] = calibration
    checkpoint["frozenSummary"] = {
        "summaryPath": str(summary_path.resolve()),
        "summarySha256": sha256_file(summary_path),
        "protocol": summary["protocol"],
        "testEvaluationCount": 1,
        "frozenThreshold": float(threshold),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "output": str(output_path),
        "summary": str(summary_path),
        "summarySha256": sha256_file(summary_path),
        "selectedEpoch": int(checkpoint_selection["epoch"]),
        "frozenThreshold": float(threshold),
        "testEvaluationCount": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(finalize(args.checkpoint, args.summary, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
