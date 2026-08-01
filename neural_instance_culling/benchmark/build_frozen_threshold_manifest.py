#!/usr/bin/env python3
"""Build the immutable threshold manifest consumed by formal test evaluation.

Each input summary must come from the training/export path's independent
calibration and one-shot test protocol.  Historical summaries that only
contain a threshold scan are rejected instead of being silently reused.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_model_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--model-summary must have the form model_name=/path/eval_summary.json")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    path = Path(raw_path.strip())
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("--model-summary requires a non-empty model name and path")
    return name, path


def read_frozen_threshold(name: str, path: Path) -> tuple[float, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = str(payload.get("protocol", ""))
    threshold = payload.get("frozenThreshold")
    if protocol != "frozen_calibration_one_shot_test" or threshold is None:
        raise ValueError(
            f"{path} is not a frozen calibration summary for {name!r}; "
            "refusing to import a historical/test-scanned threshold."
        )
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Frozen threshold for {name!r} is outside [0, 1]: {threshold}")
    split = payload.get("protocolSplit") or {}
    if int(payload.get("testEvaluationCount", 0)) != 1:
        raise ValueError(f"{path} does not record exactly one test evaluation.")
    if not split.get("frozenTestDigest"):
        raise ValueError(f"{path} has no frozen test digest.")
    return threshold, {
        "summary": str(path.as_posix()),
        "protocol": protocol,
        "testEvaluationCount": 1,
        "frozenTestDigest": split["frozenTestDigest"],
        "calibrationDigest": split.get("calibrationDigest"),
        "selectionRule": payload.get("selectionRule"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a formal frozen-threshold manifest.")
    parser.add_argument("--model-summary", action="append", type=parse_model_summary, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    thresholds: dict[str, float] = {}
    provenance: dict[str, dict] = {}
    for name, path in args.model_summary:
        if name in thresholds:
            raise ValueError(f"Duplicate model name: {name}")
        threshold, details = read_frozen_threshold(name, path)
        thresholds[name] = threshold
        provenance[name] = details

    payload = {
        "schema": "neuralstreamweb3d-frozen-threshold-manifest-v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "selection": "thresholds selected on calibration only; test is evaluated once at the frozen value",
        "thresholds": thresholds,
        "provenance": provenance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": sorted(thresholds)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
