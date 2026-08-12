#!/usr/bin/env python3
"""Validate captured WebGPU outputs against the PyTorch parity cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np


OUTPUT_NAMES = (
    "visibilityLogits",
    "visibilityScores",
    "utilityLogits",
    "utilityScores",
    "downloadLogits",
    "downloadScores",
)


def _bits_to_float32(values: list[int]) -> np.ndarray:
    words = np.asarray(values, dtype="<u4")
    return words.view("<f4")


def validate(capture_path: Path, cases_path: Path, atol: float, rtol: float, require_formal_gpu: bool) -> dict[str, Any]:
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    cases_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if capture.get("schema") != "ray-context-survival-owrb-webgpu-parity-capture-v1":
        raise ValueError("unexpected WebGPU capture schema")
    if capture.get("backend") != "webgpu":
        raise ValueError("capture backend is not WebGPU")
    if require_formal_gpu and capture.get("formalReady") is not True:
        raise ValueError("WebGPU capture lacks formal hardware evidence")
    expected_cases = {int(item["caseId"]): item for item in cases_payload.get("cases", [])}
    actual_cases = {int(item["caseId"]): item for item in capture.get("results", [])}
    if set(expected_cases) != set(actual_cases):
        raise ValueError(f"case ID mismatch: expected={sorted(expected_cases)} actual={sorted(actual_cases)}")
    case_reports: list[dict[str, Any]] = []
    max_abs = 0.0
    max_rel = 0.0
    all_values_passed = True
    for case_id in sorted(expected_cases):
        expected = expected_cases[case_id]
        actual = actual_cases[case_id]
        if list(map(int, actual.get("candidateIds", []))) != list(map(int, expected.get("candidateIds", []))):
            raise ValueError(f"candidate ID mismatch in case {case_id}")
        candidate_count = len(expected["candidateIds"])
        raw = _bits_to_float32([int(value) for value in actual.get("rawOutput", [])])
        if raw.size != candidate_count * len(OUTPUT_NAMES):
            raise ValueError(f"case {case_id} returned {raw.size} floats, expected {candidate_count * len(OUTPUT_NAMES)}")
        raw = raw.reshape(candidate_count, len(OUTPUT_NAMES))
        case_max_abs = 0.0
        case_max_rel = 0.0
        metric_reports: dict[str, Any] = {}
        for index, name in enumerate(OUTPUT_NAMES):
            reference = np.asarray(expected["expected"][name], dtype=np.float32)
            observed = raw[:, index]
            if reference.shape != observed.shape:
                raise ValueError(f"shape mismatch for case {case_id}, {name}")
            if not np.isfinite(reference).all() or not np.isfinite(observed).all():
                raise ValueError(f"non-finite output in case {case_id}, {name}")
            difference = np.abs(observed - reference)
            relative = difference / np.maximum(np.abs(reference), 1e-5)
            tolerance = float(atol) + float(rtol) * np.abs(reference)
            all_values_passed = all_values_passed and bool(np.all(difference <= tolerance))
            metric_abs = float(np.max(difference)) if difference.size else 0.0
            metric_rel = float(np.max(relative)) if relative.size else 0.0
            metric_reports[name] = {"maxAbs": metric_abs, "maxRelative": metric_rel}
            case_max_abs = max(case_max_abs, metric_abs)
            case_max_rel = max(case_max_rel, metric_rel)
        max_abs = max(max_abs, case_max_abs)
        max_rel = max(max_rel, case_max_rel)
        case_reports.append({"caseId": case_id, "candidateCount": candidate_count, "forwardMs": actual.get("forwardMs"), "metrics": metric_reports, "maxAbs": case_max_abs, "maxRelative": case_max_rel})
    passed = bool(all_values_passed)
    result = {
        "schema": "ray-context-survival-owrb-webgpu-parity-validation-v1",
        "capture": str(capture_path.resolve()),
        "cases": str(cases_path.resolve()),
        "caseCount": len(case_reports),
        "maxAbs": max_abs,
        "maxRelative": max_rel,
        "atol": float(atol),
        "rtol": float(rtol),
        "formalGpuRequired": bool(require_formal_gpu),
        "gpuGate": capture.get("gpuGate"),
        "formalReady": capture.get("formalReady"),
        "passed": passed,
        "caseReports": case_reports,
    }
    if not passed:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--allow-nonformal-gpu", action="store_true")
    args = parser.parse_args()
    result = validate(args.capture, args.cases, args.atol, args.rtol, not args.allow_nonformal_gpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "output": str(args.output.resolve()), "maxAbs": result["maxAbs"], "maxRelative": result["maxRelative"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
