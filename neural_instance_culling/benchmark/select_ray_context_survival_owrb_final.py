#!/usr/bin/env python3
"""Select one validation-authorized OWRB member for the one-shot test.

This selector reads only the test-free matrix summary.  It requires every
registered seed of a candidate variant to have a calibration-qualified safety
workpoint, then ranks variants by validation useful cull, balanced accuracy,
precision, and predicted count.  The selected checkpoint's threshold remains
frozen when the separate test command is run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _mean(records: list[dict[str, Any]], section: str, field: str) -> float:
    values = [float(record[section][field]) for record in records]
    return sum(values) / max(1, len(values))


def select_final_member(summary: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    if summary.get("schema") != "ray-context-survival-owrb-matrix-summary-v1":
        raise ValueError("final selection requires the OWRB matrix summary schema")
    if summary.get("testRead") is not False:
        raise ValueError("final selection requires a test-free validation summary")
    variants = list(summary.get("variants", []))
    seeds = [int(value) for value in summary.get("seeds", [])]
    eligible: list[dict[str, Any]] = []
    for variant in variants:
        records = [(seed, summary.get("members", {}).get(variant, {}).get(str(seed))) for seed in seeds]
        if any(record is None for _seed, record in records):
            continue
        record_values = [record for _seed, record in records if record is not None]
        if not all(bool(record.get("safeWorkpoint", False)) for record in record_values):
            continue
        # This is a validation diagnostic guard in addition to the calibration
        # lower-bound gate.  It does not read or optimize on test rows.
        if not all(
            float(record["poseMacro"]["weightedRecall"]) > 0.99
            and float(record["aggregate"]["weightedRecall"]) > 0.99
            for record in record_values
        ):
            continue
        eligible.append({
            "variant": variant,
            "meanUsefulCull": _mean(record_values, "poseMacro", "usefulCull"),
            "meanBalancedAccuracy": _mean(record_values, "poseMacro", "balancedAccuracy"),
            "meanPrecision": _mean(record_values, "poseMacro", "precision"),
            "meanPredictedCount": _mean(record_values, "aggregate", "avgPredCount"),
            "seedRecords": records,
        })
    if not eligible:
        return {
            "schema": "ray-context-survival-owrb-final-selection-v1",
            "split": "validation",
            "testRead": False,
            "status": "no_validation_safe_final_member",
            "selection": None,
            "eligibleVariants": [],
            "sourceSummary": str(summary_path.resolve()),
            "selectionRule": "all seeds calibration-qualified with validation pose and aggregate weighted recall > 0.99; maximize validation pose useful cull, then balanced accuracy and precision, then minimize predicted count",
        }
    chosen_variant = max(
        eligible,
        key=lambda item: (
            item["meanUsefulCull"],
            item["meanBalancedAccuracy"],
            item["meanPrecision"],
            -item["meanPredictedCount"],
        ),
    )
    chosen_seed, chosen_record = max(
        chosen_variant["seedRecords"],
        key=lambda item: (
            float(item[1]["poseMacro"]["usefulCull"]),
            float(item[1]["poseMacro"]["balancedAccuracy"]),
            float(item[1]["poseMacro"]["precision"]),
            -float(item[1]["aggregate"]["avgPredCount"]),
        ),
    )
    selection = {
        "variant": chosen_variant["variant"],
        "seed": int(chosen_seed),
        "memberKey": f"{chosen_variant['variant']}:seed{int(chosen_seed)}",
        "checkpoint": chosen_record.get("checkpoint"),
        "runtimeFeatures": chosen_record.get("runtimeFeatures"),
        "threshold": float(chosen_record["threshold"]),
        "validationPoseMacro": chosen_record["poseMacro"],
        "validationAggregate": chosen_record["aggregate"],
        "thresholdSource": chosen_record.get("thresholdSource"),
        "testAuthorization": "one-shot test only; threshold is frozen from this checkpoint calibration",
    }
    return {
        "schema": "ray-context-survival-owrb-final-selection-v1",
        "split": "validation",
        "testRead": False,
        "status": "selected_validation_safe_final_member",
        "selection": selection,
        "eligibleVariants": [
            {key: value for key, value in item.items() if key != "seedRecords"}
            for item in eligible
        ],
        "sourceSummary": str(summary_path.resolve()),
        "selectionRule": "all seeds calibration-qualified with validation pose and aggregate weighted recall > 0.99; maximize validation pose useful cull, then balanced accuracy and precision, then minimize predicted count",
    }


def self_test() -> dict[str, Any]:
    summary = {
        "schema": "ray-context-survival-owrb-matrix-summary-v1",
        "testRead": False,
        "variants": ["a", "b"],
        "seeds": [1, 2, 3],
        "members": {"a": {}, "b": {}},
    }
    for variant in summary["variants"]:
        for seed in summary["seeds"]:
            summary["members"][variant][str(seed)] = {
                "safeWorkpoint": True,
                "threshold": 0.5,
                "checkpoint": f"{variant}-{seed}.pt",
                "runtimeFeatures": f"{variant}-{seed}.bin",
                "thresholdSource": {"safeWorkpoint": True},
                "poseMacro": {
                    "weightedRecall": 0.999,
                    "usefulCull": 0.2 if variant == "a" else 0.1,
                    "balancedAccuracy": 0.8,
                    "precision": 0.7,
                },
                "aggregate": {"avgPredCount": 10.0, "weightedRecall": 0.999},
            }
    result = select_final_member(summary, Path("fixture-summary.json"))
    assert result["status"] == "selected_validation_safe_final_member"
    assert result["selection"]["variant"] == "a"
    return {"status": "passed", "selected": result["selection"]["memberKey"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.summary is None or args.output is None:
        parser.error("--summary and --output are required unless --self-test is used")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = select_final_member(summary, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output.resolve()), "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
