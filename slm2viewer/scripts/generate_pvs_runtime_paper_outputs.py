#!/usr/bin/env python3
"""Build paper tables and figures from browser runtime uploads."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


RESULT_SCHEMA = "pvs-v4-browser-runtime-result-v1"
FORMAL_SESSION_COUNT = 5
FORMAL_WARMUP_COUNT = 50
SCENES = ("hkust", "ifcbench")
BACKENDS = ("webgpu-v4", "wasm-simd-v4")
CANDIDATE_BINS = (
    ("0-1k", 0, 1_000),
    ("1-2k", 1_000, 2_000),
    ("2-5k", 2_000, 5_000),
    ("5-10k", 5_000, 10_000),
    ("10-15k", 10_000, 15_000),
    ("15k+", 15_000, math.inf),
)

SUMMARY_FIELDS = [
    "status", "scene", "device", "deviceModel", "backend", "adapter", "timingSource",
    "formalUploadCount", "formalSessions", "sampleCount", "poseCount", "candidateTotal",
    "candidateMean", "candidateP95", "candidateMin", "candidateMax", "runtimeAssetMiB",
    "latencyMeanMs", "latencyMeanCi95LowMs", "latencyMeanCi95HighMs",
    "latencyP50Ms", "latencyP50Ci95LowMs", "latencyP50Ci95HighMs",
    "latencyP95Ms", "latencyP95Ci95LowMs", "latencyP95Ci95HighMs",
    "fitInterceptMs", "fitSlopeMsPerCandidate", "fitRmseMs", "fitMaeMs",
    "fitR2", "fitMaxAbsErrorMs", "reason",
]

BIN_FIELDS = [
    "status", "scene", "device", "deviceModel", "backend", "adapter", "timingSource",
    "candidateBin", "formalSessions", "binSessionCount", "sampleCount", "candidateMean",
    "candidateP95", "candidateMin", "candidateMax", "latencyMeanMs",
    "latencyMeanCi95LowMs", "latencyMeanCi95HighMs", "latencyP50Ms",
    "latencyP50Ci95LowMs", "latencyP50Ci95HighMs", "latencyP95Ms",
    "latencyP95Ci95LowMs", "latencyP95Ci95HighMs", "reason",
]

SOURCE_FIELDS = [
    "sourceFile", "scene", "device", "deviceModel", "backend", "timingSource", "round",
    "poseId", "ordinal", "candidateBin", "candidateCount", "modelInferenceMs",
    "gpuKernelMs", "submitCompletionMs",
]


def canonical_scene(value: Any) -> str:
    scene = str(value or "").strip().lower()
    return "hkust" if scene == "hkust-v3" else scene


def canonical_backend(value: Any) -> str:
    return str(value or "").strip().lower()


def canonical_device(payload: dict[str, Any]) -> str:
    label = str((payload.get("device") or {}).get("label") or "").strip()
    model = str((payload.get("device") or {}).get("model") or "").strip()
    identity = f"{label} {model}".lower()
    if "a6000" in identity:
        return "RTX A6000"
    if "macbook" in identity and "m2" in identity:
        return "MacBook Air M2"
    if "vivo" in identity and "x200" in identity:
        return "vivo X200 Pro mini"
    return label or model or "unknown device"


def adapter_name(payload: dict[str, Any]) -> str:
    adapter = (payload.get("environment") or {}).get("adapter") or {}
    return " ".join(
        str(adapter.get(key) or "").strip()
        for key in ("vendor", "architecture", "device", "description")
        if str(adapter.get(key) or "").strip()
    )


def timing_source(payload: dict[str, Any]) -> str:
    source = str((payload.get("timingDefinition") or {}).get("primary") or "").strip()
    if source:
        return source
    sessions = payload.get("sessions") or []
    samples = sessions[0].get("samples") if sessions else []
    return str((samples[0] if samples else {}).get("timingSource") or "").strip()


def group_key(payload: dict[str, Any]) -> tuple[str, str, str]:
    return (
        canonical_device(payload),
        canonical_scene((payload.get("workload") or {}).get("scene")),
        canonical_backend((payload.get("environment") or {}).get("backend")),
    )


def formal_exclusion_reason(path: Path, payload: dict[str, Any]) -> str | None:
    label = str((payload.get("device") or {}).get("label") or "")
    if "smoke" in f"{path.name} {label}".lower():
        return "smoke-excluded: filename or device label is marked smoke"
    workload = payload.get("workload") or {}
    environment = payload.get("environment") or {}
    backend = canonical_backend(environment.get("backend"))
    if workload.get("split") != "test":
        return "unavailable: workload is not the frozen test split"
    if environment.get("secureContext") is not True:
        return "unavailable: browser was not in a secure context"
    if int(environment.get("visibilityViolations") or 0) != 0:
        return "unavailable: browser visibility violation"
    if payload.get("serverReceipt") is not None and (
        payload.get("serverReceipt") or {}
    ).get("formalReady") is not True:
        return "unavailable: server formal gate failed"
    sessions = payload.get("sessions") or []
    if len(sessions) != FORMAL_SESSION_COUNT:
        return "smoke-excluded: formal result requires exactly five sessions"
    pose_count = int(workload.get("poseCount") or 0)
    if pose_count <= 0:
        return "unavailable: invalid pose count"
    for session in sessions:
        if session.get("warmupCount") != FORMAL_WARMUP_COUNT:
            return "unavailable: formal session does not contain 50 warmup poses"
        if len(session.get("samples") or []) != pose_count:
            return "unavailable: session does not cover the complete test workload"
    if backend == "webgpu-v4":
        if (environment.get("hardwareGate") or {}).get("hardware") is not True:
            return "unavailable: WebGPU hardware gate failed"
    elif backend == "wasm-simd-v4":
        if environment.get("wasmSimd") is not True:
            return "unavailable: WASM SIMD gate failed"
    else:
        return f"unavailable: unsupported backend {backend or 'unknown'}"
    return None


def read_uploads(input_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted({path for directory in input_dirs for path in directory.glob("*.json")})
    if not paths:
        joined = ", ".join(str(path) for path in input_dirs)
        raise ValueError(f"no browser upload JSON files found in {joined}")
    records = []
    excluded = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != RESULT_SCHEMA:
            continue
        key = group_key(payload)
        reason = formal_exclusion_reason(path, payload)
        record = {"path": path, "payload": payload, "key": key, "reason": reason}
        records.append(record)
        if reason is not None:
            excluded.append({
                "file": path.name,
                "device": key[0],
                "scene": key[1],
                "backend": key[2],
                "reason": reason,
            })
    return records, excluded


def session_arrays(records: list[dict[str, Any]]) -> list[tuple[np.ndarray, np.ndarray]]:
    result = []
    for record in records:
        for session in record["payload"]["sessions"]:
            candidates = np.asarray(
                [int(sample["candidateCount"]) for sample in session["samples"]],
                dtype=np.float64,
            )
            latency = np.asarray(
                [float(sample["modelInferenceMs"]) for sample in session["samples"]],
                dtype=np.float64,
            )
            result.append((candidates, latency))
    return result


def percentile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile))


def bootstrap_ci(
    sessions: list[np.ndarray], statistic: Callable[[np.ndarray], float],
    repetitions: int, rng: np.random.Generator,
) -> list[float]:
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, len(sessions), size=len(sessions))
        resampled = []
        for selected_index in selected:
            session = sessions[int(selected_index)]
            resampled.append(session[rng.integers(0, session.size, size=session.size)])
        estimates[index] = statistic(np.concatenate(resampled))
    return [
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    ]


def latency_stats(
    sessions: list[np.ndarray], repetitions: int, rng: np.random.Generator,
) -> dict[str, float]:
    values = np.concatenate(sessions)
    estimators = (
        ("Mean", np.mean),
        ("P50", lambda array: np.quantile(array, 0.50)),
        ("P95", lambda array: np.quantile(array, 0.95)),
    )
    result: dict[str, float] = {}
    for name, estimator in estimators:
        value = float(estimator(values))
        interval = bootstrap_ci(sessions, estimator, repetitions, rng)
        result[f"latency{name}Ms"] = value
        result[f"latency{name}Ci95LowMs"] = interval[0]
        result[f"latency{name}Ci95HighMs"] = interval[1]
    return result


def linear_fit(candidates: np.ndarray, latency: np.ndarray) -> dict[str, float | None]:
    centered = candidates - candidates.mean()
    denominator = float(np.dot(centered, centered))
    if denominator == 0:
        return {
            "fitInterceptMs": None,
            "fitSlopeMsPerCandidate": None,
            "fitRmseMs": None,
            "fitMaeMs": None,
            "fitR2": None,
            "fitMaxAbsErrorMs": None,
        }
    slope = float(np.dot(centered, latency - latency.mean()) / denominator)
    intercept = float(latency.mean() - slope * candidates.mean())
    fitted = intercept + slope * candidates
    residual = latency - fitted
    squared_total = float(np.dot(latency - latency.mean(), latency - latency.mean()))
    return {
        "fitInterceptMs": intercept,
        "fitSlopeMsPerCandidate": slope,
        "fitRmseMs": float(np.sqrt(np.mean(residual * residual))),
        "fitMaeMs": float(np.mean(np.abs(residual))),
        "fitR2": 1.0 - float(np.dot(residual, residual)) / squared_total
        if squared_total > 0 else None,
        "fitMaxAbsErrorMs": float(np.max(np.abs(residual))),
    }


def candidate_bin(candidate: int) -> str:
    for name, lower, upper in CANDIDATE_BINS:
        if lower <= candidate < upper:
            return name
    raise ValueError(f"candidate count is outside the defined bins: {candidate}")


def representative(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "deviceModel": "",
            "adapter": "",
            "timingSource": "",
            "poseCount": None,
            "candidateTotal": None,
            "runtimeAssetMiB": None,
        }
    payload = records[0]["payload"]
    device = payload.get("device") or {}
    workload = payload.get("workload") or {}
    model = payload.get("model") or {}
    return {
        "deviceModel": str(device.get("model") or "").strip(),
        "adapter": adapter_name(payload),
        "timingSource": timing_source(payload),
        "poseCount": int(workload.get("poseCount") or 0),
        "candidateTotal": int(workload.get("candidateCount") or 0),
        "runtimeAssetMiB": float(model.get("runtimeAssetBytes") or 0) / (1024 ** 2),
    }


def empty_group_row(
    key: tuple[str, str, str], records: list[dict[str, Any]], reason: str,
) -> dict[str, Any]:
    info = representative(records)
    return {
        "status": "unavailable",
        "scene": key[1],
        "device": key[0],
        "deviceModel": info["deviceModel"],
        "backend": key[2],
        "adapter": info["adapter"],
        "timingSource": info["timingSource"],
        "formalUploadCount": 0,
        "formalSessions": 0,
        "sampleCount": 0,
        "poseCount": info["poseCount"],
        "candidateTotal": info["candidateTotal"],
        "runtimeAssetMiB": None,
        "reason": reason,
    }


def formal_group_row(
    key: tuple[str, str, str], records: list[dict[str, Any]],
    bootstrap_repetitions: int, rng: np.random.Generator,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    sessions = session_arrays(records)
    candidates = np.concatenate([item[0] for item in sessions])
    latency = np.concatenate([item[1] for item in sessions])
    info = representative(records)
    row: dict[str, Any] = {
        "status": "formal",
        "scene": key[1],
        "device": key[0],
        "deviceModel": info["deviceModel"],
        "backend": key[2],
        "adapter": info["adapter"],
        "timingSource": info["timingSource"],
        "formalUploadCount": len(records),
        "formalSessions": len(sessions),
        "sampleCount": int(latency.size),
        "poseCount": info["poseCount"],
        "candidateTotal": info["candidateTotal"],
        "candidateMean": float(candidates.mean()),
        "candidateP95": percentile(candidates, 0.95),
        "candidateMin": float(candidates.min()),
        "candidateMax": float(candidates.max()),
        "runtimeAssetMiB": info["runtimeAssetMiB"],
        "reason": None,
    }
    row.update(latency_stats([item[1] for item in sessions], bootstrap_repetitions, rng))
    row.update(linear_fit(candidates, latency))
    source_rows = []
    round_number = 0
    for record in records:
        payload = record["payload"]
        for session in payload["sessions"]:
            round_number += 1
            for sample in session["samples"]:
                source_rows.append({
                    "sourceFile": record["path"].name,
                    "scene": key[1],
                    "device": key[0],
                    "deviceModel": info["deviceModel"],
                    "backend": key[2],
                    "timingSource": info["timingSource"],
                    "round": round_number,
                    "poseId": sample.get("poseId"),
                    "ordinal": sample.get("ordinal"),
                    "candidateBin": candidate_bin(int(sample["candidateCount"])),
                    "candidateCount": int(sample["candidateCount"]),
                    "modelInferenceMs": float(sample["modelInferenceMs"]),
                    "gpuKernelMs": sample.get("gpuKernelMs"),
                    "submitCompletionMs": sample.get("submitCompletionMs"),
                })
    return row, source_rows, sessions


def bin_rows(
    group_row: dict[str, Any], sessions: list[tuple[np.ndarray, np.ndarray]],
    bootstrap_repetitions: int, rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows = []
    for name, lower, upper in CANDIDATE_BINS:
        selected = []
        for candidates, latency in sessions:
            mask = (candidates >= lower) & (candidates < upper)
            if mask.any():
                selected.append((candidates[mask], latency[mask]))
        if selected:
            candidates = np.concatenate([item[0] for item in selected])
            latency_sessions = [item[1] for item in selected]
            latency = np.concatenate(latency_sessions)
            row = {
                "status": "formal",
                "scene": group_row["scene"],
                "device": group_row["device"],
                "deviceModel": group_row["deviceModel"],
                "backend": group_row["backend"],
                "adapter": group_row["adapter"],
                "timingSource": group_row["timingSource"],
                "candidateBin": name,
                "formalSessions": group_row["formalSessions"],
                "binSessionCount": len(selected),
                "sampleCount": int(latency.size),
                "candidateMean": float(candidates.mean()),
                "candidateP95": percentile(candidates, 0.95),
                "candidateMin": float(candidates.min()),
                "candidateMax": float(candidates.max()),
                "reason": None,
            }
            row.update(latency_stats(latency_sessions, bootstrap_repetitions, rng))
        else:
            row = {
                "status": "unavailable",
                "scene": group_row["scene"],
                "device": group_row["device"],
                "deviceModel": group_row["deviceModel"],
                "backend": group_row["backend"],
                "adapter": group_row["adapter"],
                "timingSource": group_row["timingSource"],
                "candidateBin": name,
                "formalSessions": group_row["formalSessions"],
                "binSessionCount": 0,
                "sampleCount": 0,
                "reason": "unavailable: no formal test pose in candidate bucket",
            }
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    output_dir: Path, group_rows: list[dict[str, Any]], bin_rows_list: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    formal_groups = [row for row in group_rows if row["status"] == "formal"]
    colors = plt.get_cmap("tab10").colors
    fig, (bucket_axis, fit_axis) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(CANDIDATE_BINS))
    for index, group in enumerate(formal_groups):
        color = colors[index % len(colors)]
        label = f"{group['device']} / {group['scene']} / {group['backend']}"
        group_bins = [
            row for row in bin_rows_list
            if row["device"] == group["device"]
            and row["scene"] == group["scene"]
            and row["backend"] == group["backend"]
        ]
        p50 = [row.get("latencyP50Ms", np.nan) for row in group_bins]
        p95 = [row.get("latencyP95Ms", np.nan) for row in group_bins]
        bucket_axis.plot(x, p50, marker="o", color=color, label=f"{label} p50")
        bucket_axis.plot(x, p95, marker="x", linestyle="--", color=color, label=f"{label} p95")

        samples = [
            row for row in source_rows
            if row["device"] == group["device"]
            and row["scene"] == group["scene"]
            and row["backend"] == group["backend"]
        ]
        candidates = np.asarray([row["candidateCount"] for row in samples], dtype=np.float64)
        latency = np.asarray([row["modelInferenceMs"] for row in samples], dtype=np.float64)
        fit_axis.scatter(candidates, latency, s=5, alpha=0.18, color=color, label=label)
        if group.get("fitSlopeMsPerCandidate") is not None:
            line_x = np.asarray([candidates.min(), candidates.max()])
            line_y = group["fitInterceptMs"] + group["fitSlopeMsPerCandidate"] * line_x
            fit_axis.plot(line_x, line_y, color=color, linewidth=2)

    bucket_axis.set_xticks(x)
    bucket_axis.set_xticklabels([item[0] for item in CANDIDATE_BINS])
    bucket_axis.set_xlabel("Candidate count bucket")
    bucket_axis.set_ylabel("Model forward latency (ms)")
    bucket_axis.set_title("Formal latency by bucket")
    bucket_axis.grid(axis="y", alpha=0.25)
    bucket_axis.legend(fontsize=7)
    fit_axis.set_xlabel("Candidate count")
    fit_axis.set_ylabel("Model forward latency (ms)")
    fit_axis.set_title("T(N) = a + bN")
    fit_axis.grid(alpha=0.25)
    fit_axis.legend(fontsize=7)
    fig.suptitle("PVS V4 model-forward latency (five-session formal uploads)")
    fig.tight_layout()
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_dir / f"runtime_paper_latency.{suffix}", dpi=180)
    plt.close(fig)


def build_outputs(
    input_dirs: list[Path], output_dir: Path, bootstrap_repetitions: int = 10_000,
    seed: int = 20260909,
) -> dict[str, Any]:
    records, excluded = read_uploads(input_dirs)
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    formal_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_key[record["key"]].append(record)
        if record["reason"] is None:
            formal_by_key[record["key"]].append(record)
    if not formal_by_key:
        raise ValueError("no five-session formal browser uploads found")

    devices = sorted(by_key)
    device_names = sorted({key[0] for key in devices})
    keys = {
        (device, scene, backend)
        for device in device_names
        for scene in SCENES
        for backend in BACKENDS
    }
    keys.update(formal_by_key)
    keys = sorted(keys)
    excluded_reasons: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for item in excluded:
        excluded_reasons[(item["device"], item["scene"], item["backend"])].append(item["reason"])

    rng = np.random.default_rng(seed)
    group_rows = []
    candidate_rows = []
    source_rows = []
    for key in keys:
        formal_records = formal_by_key.get(key, [])
        if formal_records:
            row, source, sessions = formal_group_row(key, formal_records, bootstrap_repetitions, rng)
            group_rows.append(row)
            candidate_rows.extend(bin_rows(row, sessions, bootstrap_repetitions, rng))
            source_rows.extend(source)
            continue
        reasons = sorted(set(excluded_reasons.get(key, [])))
        reason = "; ".join(reasons) if reasons else "unavailable: no five-session formal upload"
        row = empty_group_row(key, by_key.get(key, []), reason)
        group_rows.append(row)
        for name, _, _ in CANDIDATE_BINS:
            candidate_rows.append({
                "status": "unavailable",
                "scene": key[1],
                "device": key[0],
                "deviceModel": row["deviceModel"],
                "backend": key[2],
                "adapter": row["adapter"],
                "timingSource": row["timingSource"],
                "candidateBin": name,
                "formalSessions": 0,
                "binSessionCount": 0,
                "sampleCount": 0,
                "reason": reason,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "runtime_paper_summary.csv", group_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "runtime_paper_candidate_bins.csv", candidate_rows, BIN_FIELDS)
    write_csv(output_dir / "runtime_paper_source.csv", source_rows, SOURCE_FIELDS)
    make_figure(output_dir, group_rows, candidate_rows, source_rows)
    summary = {
        "schema": "pvs-runtime-paper-output-v1",
        "formalRule": "exactly five complete test sessions, 50 warmups per session, and backend gate passed",
        "bootstrapRepetitions": bootstrap_repetitions,
        "bootstrapUnit": "session, then pose within selected session",
        "candidateBins": [name for name, _, _ in CANDIDATE_BINS],
        "inputDirectories": [str(path.resolve()) for path in input_dirs],
        "formalInputs": [
            record["path"].name
            for record in records
            if record["reason"] is None
        ],
        "excludedInputs": excluded,
        "groups": group_rows,
        "candidateBuckets": candidate_rows,
        "outputs": [
            "runtime_paper_source.csv",
            "runtime_paper_summary.csv",
            "runtime_paper_candidate_bins.csv",
            "runtime_paper_latency.pdf",
            "runtime_paper_latency.svg",
            "runtime_paper_latency.png",
        ],
    }
    (output_dir / "runtime_paper_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", dest="input_dirs", action="append", type=Path,
        help="browser upload directory; repeat for multiple result roots",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "neural_instance_culling/benchmark/out/paper_results/mobile_runtime",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 100:
        raise SystemExit("--bootstrap must be at least 100")
    root = Path(__file__).resolve().parents[2]
    input_dirs = args.input_dirs or [
        root / "neural_instance_culling/benchmark/out/paper_results/mobile_runtime/browser_uploads",
    ]
    summary = build_outputs(input_dirs, args.output_dir, args.bootstrap, args.seed)
    formal = [row for row in summary["groups"] if row["status"] == "formal"]
    print(json.dumps({
        "outputDir": str(args.output_dir.resolve()),
        "formalGroups": len(formal),
        "formalInputs": len(summary["formalInputs"]),
        "excludedInputs": len(summary["excludedInputs"]),
    }, indent=2))


if __name__ == "__main__":
    main()
