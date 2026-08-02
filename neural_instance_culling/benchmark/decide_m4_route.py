#!/usr/bin/env python3
"""Decide the registered M4 contribution route from validation evidence only.

The decision is intentionally conservative.  It never opens a test split and
does not recalibrate a checkpoint.  Route A requires the pre-registered
useful-cull gain over ``geometry_context_ray``; the alternative image/byte
gate is reported as unavailable when the input summary does not contain that
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REFERENCE = "geometry_context_ray"
FULL = "full"
REQUIRED_VARIANTS = (
    "aabb_ray",
    "geometry_ray",
    "geometry_context_ray",
    "geometry_context_proxy_ray_no_inhibition",
    "full",
)
REQUIRED_METRIC = "useful_cull"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--useful-cull-gain", type=float, default=0.02)
    parser.add_argument("--ci-lower-floor", type=float, default=0.0)
    return parser.parse_args()


def finite_number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def validate_summary(payload: dict[str, Any], source: Path) -> None:
    if payload.get("schema") != "neuralstreamweb3d-formal-m4-matrix-summary-v1":
        raise ValueError(f"unexpected M4 summary schema: {source}")
    if payload.get("split") != "validation":
        raise ValueError(f"M4 route decision requires validation evidence: {source}")
    if payload.get("status") != "validation_summary_only; formal test not read":
        raise ValueError(f"M4 summary is not explicitly test-free: {source}")
    if payload.get("referenceVariant") != REFERENCE:
        raise ValueError(f"unexpected M4 reference variant: {source}")
    variants = tuple(payload.get("variants") or ())
    missing = [variant for variant in REQUIRED_VARIANTS if variant not in variants]
    if missing:
        raise ValueError(f"M4 summary is incomplete; missing variants: {missing}")
    seeds = payload.get("seeds") or []
    if len(seeds) != 3:
        raise ValueError(f"M4 route decision requires three seeds, got {len(seeds)}")
    comparisons = payload.get("pairedComparisonsVariantMinusReference") or {}
    full = comparisons.get(FULL)
    if not isinstance(full, dict) or REQUIRED_METRIC not in full:
        raise ValueError("M4 summary lacks the full-vs-reference useful-cull comparison")
    useful = full[REQUIRED_METRIC]
    if not isinstance(useful, dict):
        raise ValueError("M4 useful-cull comparison is not structured")
    finite_number(useful.get("mean_delta"), "useful-cull mean_delta")
    ci = useful.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("M4 useful-cull comparison lacks a two-sided ci95")
    finite_number(ci[0], "useful-cull ci95 lower")
    finite_number(ci[1], "useful-cull ci95 upper")


def make_decision(payload: dict[str, Any], source: Path, gain_floor: float, ci_floor: float) -> dict[str, Any]:
    validate_summary(payload, source)
    useful = payload["pairedComparisonsVariantMinusReference"][FULL][REQUIRED_METRIC]
    mean_delta = finite_number(useful["mean_delta"], "useful-cull mean_delta")
    ci95 = [finite_number(value, "useful-cull ci95") for value in useful["ci95"]]
    useful_gate = mean_delta >= gain_floor and ci95[0] > ci_floor
    reasons: list[str] = []
    if not useful_gate:
        reasons.append(
            f"full useful-cull gain {mean_delta:.6f} with CI lower {ci95[0]:.6f} "
            f"does not satisfy gain >= {gain_floor:.6f} and CI lower > {ci_floor:.6f}"
        )
    reasons.append("the image/byte alternative gate is not present in the M4 validation summary")
    route = "route_a_directional_proxy" if useful_gate else "route_b_system"
    return {
        "schema": "neuralstreamweb3d-m4-route-decision-v1",
        "sourceSummary": str(source),
        "sourceSummarySha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": "validation",
        "testRead": False,
        "referenceVariant": REFERENCE,
        "fullVariant": FULL,
        "criterion": {
            "usefulCullGainFloor": float(gain_floor),
            "usefulCullCiLowerFloor": float(ci_floor),
            "imageByteAlternativeAvailable": False,
        },
        "fullMinusReference": {
            "usefulCullMeanDelta": mean_delta,
            "usefulCullCi95": ci95,
            "usefulCullGate": useful_gate,
        },
        "route": route,
        "reasons": reasons,
        "status": "route_selected_from_validation_only",
    }


def render_markdown(decision: dict[str, Any]) -> str:
    delta = decision["fullMinusReference"]
    lines = [
        "# M4 Contribution Route Decision",
        "",
        "This decision uses the complete validation matrix only; the sealed test split was not read.",
        "",
        f"- Route: `{decision['route']}`",
        f"- Full minus geometry+context+ray useful-cull delta: `{delta['usefulCullMeanDelta']:.6f}`",
        f"- Paired bootstrap 95% CI: `[{delta['usefulCullCi95'][0]:.6f}, {delta['usefulCullCi95'][1]:.6f}]`",
        f"- Test read: `{decision['testRead']}`",
        "",
        "Reasons:",
    ]
    lines.extend(f"- {reason}" for reason in decision["reasons"])
    lines.extend([
        "",
        "The image/byte alternative requires a separate image-safety and download-trajectory result; it is not inferred from this matrix.",
        f"Source SHA-256: `{decision['sourceSummarySha256']}`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.useful_cull_gain < 0:
        raise ValueError("--useful-cull-gain must be non-negative")
    payload = json.loads(args.summary.read_text(encoding="utf-8"))
    decision = make_decision(payload, args.summary, args.useful_cull_gain, args.ci_lower_floor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(decision), encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
