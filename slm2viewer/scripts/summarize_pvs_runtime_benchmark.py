#!/usr/bin/env python3
"""Summarize uploaded model-only WebGPU and WASM SIMD timing sessions."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1/browser_uploads",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "neural_instance_culling/benchmark/out/pvs_v4_frontend_inference_latency_v1",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def clustered_interval(
    sessions: list[np.ndarray], q: float, repetitions: int, rng: np.random.Generator
) -> list[float]:
    estimates = np.empty(repetitions, dtype=np.float64)
    for iteration in range(repetitions):
        selected = rng.integers(0, len(sessions), size=len(sessions))
        samples = []
        for session_index in selected:
            session = sessions[int(session_index)]
            samples.append(session[rng.integers(0, session.size, size=session.size)])
        estimates[iteration] = np.quantile(np.concatenate(samples), q)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def group_key(payload: dict) -> tuple[str, str, str]:
    device = payload.get("device") or {}
    adapter = (payload.get("environment") or {}).get("adapter") or {}
    adapter_name = " ".join(
        str(adapter.get(key) or "").strip()
        for key in ("vendor", "architecture", "device", "description")
    ).strip()
    backend = str((payload.get("environment") or {}).get("backend") or "unknown")
    scene = str((payload.get("workload") or {}).get("scene") or "unknown")
    hardware = adapter_name or ("CPU" if "wasm" in backend else "unknown adapter")
    return scene, backend, f"{device.get('label', 'unknown')} | {hardware}"


def main() -> None:
    args = parse_args()
    files = sorted(args.input_dir.glob("*.json"))
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "pvs-v4-browser-runtime-result-v1":
            continue
        groups[group_key(payload)].append(payload)
    if not groups:
        raise SystemExit(f"No browser runtime results found in {args.input_dir}")

    rng = np.random.default_rng(args.seed)
    rows = []
    scaling_rows = []
    for key, payloads in sorted(groups.items()):
        scene, backend, device_key = key
        timing_sessions: list[np.ndarray] = []
        candidate_sessions: list[np.ndarray] = []
        for payload in payloads:
            for session in payload["sessions"]:
                timing_sessions.append(
                    np.asarray([sample["modelInferenceMs"] for sample in session["samples"]], dtype=np.float64)
                )
                candidate_sessions.append(
                    np.asarray([sample["candidateCount"] for sample in session["samples"]], dtype=np.int64)
                )
        timings = np.concatenate(timing_sessions)
        candidates = np.concatenate(candidate_sessions)
        near_10k = timings[(candidates >= 9000) & (candidates <= 11000)]
        slope, intercept = np.polyfit(candidates.astype(np.float64), timings, 1)
        fitted = intercept + slope * candidates
        residual = float(np.square(timings - fitted).sum())
        total = float(np.square(timings - timings.mean()).sum())
        payload = payloads[0]
        row = {
            "scene": scene,
            "device": device_key,
            "backend": backend,
            "timingSource": (payload.get("timingDefinition") or {}).get("primary"),
            "sessionCount": len(timing_sessions),
            "sampleCount": int(timings.size),
            "runtimeAssetMiB": float((payload.get("model") or {}).get("runtimeAssetBytes", 0)) / (1024**2),
            "candidateMean": float(candidates.mean()),
            "candidateP95": percentile(candidates, 0.95),
            "modelInferenceP50Ms": percentile(timings, 0.50),
            "modelInferenceP50Ci95": clustered_interval(timing_sessions, 0.50, args.bootstrap, rng),
            "modelInferenceP95Ms": percentile(timings, 0.95),
            "modelInferenceP95Ci95": clustered_interval(timing_sessions, 0.95, args.bootstrap, rng),
            "near10kSampleCount": int(near_10k.size),
            "near10kP95Ms": percentile(near_10k, 0.95) if near_10k.size else None,
            "fixedOverheadMs": float(intercept),
            "perCandidateMs": float(slope),
            "linearFitR2": 1.0 - residual / total if total > 0 else None,
        }
        rows.append(row)
        bins = ((0, 1000), (1000, 2000), (2000, 5000), (5000, 10000),
                (10000, 15000), (15000, np.inf))
        for lower, upper in bins:
            mask = (candidates >= lower) & (candidates < upper)
            values = timings[mask]
            scaling_rows.append({
                "scene": scene,
                "device": device_key,
                "backend": backend,
                "candidateBin": f"{lower // 1000:g}k+" if np.isinf(upper)
                else f"{lower // 1000:g}-{upper // 1000:g}k",
                "sampleCount": int(values.size),
                "candidateMean": float(candidates[mask].mean()) if values.size else None,
                "latencyMeanMs": float(values.mean()) if values.size else None,
                "latencyP50Ms": percentile(values, 0.50) if values.size else None,
                "latencyP95Ms": percentile(values, 0.95) if values.size else None,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "pvs-v4-browser-runtime-summary-v1",
                "bootstrapRepetitions": args.bootstrap,
                "groups": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "paper_table.csv").open("w", encoding="utf-8", newline="") as file:
        columns = [
            "scene", "device", "backend", "timingSource", "sessionCount", "sampleCount", "runtimeAssetMiB",
            "candidateMean", "candidateP95", "modelInferenceP50Ms", "modelInferenceP50Ci95",
            "modelInferenceP95Ms", "modelInferenceP95Ci95", "near10kSampleCount", "near10kP95Ms",
            "fixedOverheadMs", "perCandidateMs", "linearFitR2",
        ]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "modelInferenceP50Ci95": json.dumps(row["modelInferenceP50Ci95"]),
                             "modelInferenceP95Ci95": json.dumps(row["modelInferenceP95Ci95"])})
    with (args.output_dir / "candidate_scaling.csv").open("w", encoding="utf-8", newline="") as file:
        columns = list(scaling_rows[0])
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(scaling_rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
