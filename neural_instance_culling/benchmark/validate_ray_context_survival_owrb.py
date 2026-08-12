#!/usr/bin/env python3
"""Validate the formal ray-context/survival/OWRB result schema.

This validator is independent from training and from the summary writer.  It
checks that a formal result really contains the registered 2x2x2 matrix, three
seeds, the complete validation pose sequence, calibration-frozen thresholds,
common candidate identity, image metrics and 10,000-draw paired contrasts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_ray_context_survival_owrb_matrix import FORMAL_VARIANTS, _comparisons
from summarize_ray_context_survival_owrb import ALL_METRICS, _factor_contrasts


SEEDS = (20260801, 20260802, 20260803)
EXPECTED_FACTOR_EFFECTS = {
    "context_main_effect",
    "survival_main_effect",
    "loss_main_effect",
    "context_survival_interaction_rvl",
    "context_survival_interaction_safety",
    "context_loss_interaction_off",
    "context_loss_interaction_on",
    "survival_loss_interaction_off",
    "survival_loss_interaction_on",
    "context_survival_loss_three_way_interaction",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_webgpu_parity(
    parity: Any,
    *,
    require_artifact_files: bool,
) -> dict[str, Any]:
    """Require an explicit PyTorch/WebGPU parity result without inventing hardware evidence.

    A formal result must prove that the browser path was numerically checked.
    The browser may still be running through SwiftShader on a machine whose
    WebGL renderer is hardware accelerated, so ``formalReady`` is retained as
    a separate hardware claim rather than being conflated with numerical pass.
    """
    _require(isinstance(parity, dict), "formal summary is missing webgpuParity")
    _require(
        parity.get("schema") == "ray-context-survival-owrb-webgpu-parity-validation-v1",
        "formal summary has an unexpected WebGPU parity schema",
    )
    _require(
        parity.get("status") in {"software_numeric_parity_passed", "hardware_numeric_parity_passed"},
        "formal summary does not contain a passed WebGPU numeric parity result",
    )
    _require(parity.get("passed") is True, "formal WebGPU parity result did not pass")
    _require(parity.get("backend") == "webgpu", "formal WebGPU parity backend is not WebGPU")
    _require(_finite(parity.get("maxAbs")), "formal WebGPU parity has a non-finite maxAbs")
    _require(_finite(parity.get("maxRelative")), "formal WebGPU parity has a non-finite maxRelative")
    _require(int(parity.get("caseCount", 0)) > 0, "formal WebGPU parity contains no cases")
    _require(isinstance(parity.get("formalReady"), bool), "formal WebGPU parity is missing formalReady")
    if parity.get("status") == "hardware_numeric_parity_passed":
        _require(parity.get("formalReady") is True, "hardware WebGPU parity is not marked formalReady")
    else:
        _require(
            parity.get("hardwareClaim") == "software WebGPU numeric parity only; no hardware WebGPU claim",
            "software WebGPU parity must explicitly disclaim a hardware claim",
        )
    if require_artifact_files:
        _require(Path(str(parity.get("capture", ""))).is_file(), "WebGPU parity capture file is missing")
        _require(Path(str(parity.get("cases", ""))).is_file(), "WebGPU parity cases file is missing")
    return {
        "status": parity["status"],
        "formalReady": bool(parity["formalReady"]),
        "caseCount": int(parity["caseCount"]),
        "maxAbs": float(parity["maxAbs"]),
        "maxRelative": float(parity["maxRelative"]),
    }


def _effect_record_ok(record: dict[str, Any], name: str, metric: str) -> None:
    _require(_finite(record.get("meanDelta")), f"{name}/{metric}: non-finite meanDelta")
    interval = record.get("ci95")
    _require(isinstance(interval, list) and len(interval) == 2, f"{name}/{metric}: invalid ci95")
    _require(all(_finite(item) for item in interval), f"{name}/{metric}: non-finite ci95")
    _require(float(interval[0]) <= float(interval[1]), f"{name}/{metric}: reversed ci95")
    _require(record.get("direction") in {"positive", "negative", "zero"}, f"{name}/{metric}: invalid direction")
    _require(isinstance(record.get("crossesZero"), bool), f"{name}/{metric}: missing crossesZero")
    _require(int(record.get("bootstrapReplicates", 0)) >= 10000, f"{name}/{metric}: fewer than 10000 bootstrap draws")
    _require(record.get("clusterUnit") == "outer seed cluster, inner pose resampling", f"{name}/{metric}: wrong cluster unit")


def validate_summary(
    summary: dict[str, Any],
    *,
    expected_pose_count: int | None = 213,
    require_artifact_files: bool = False,
) -> dict[str, Any]:
    _require(summary.get("schema") == "ray-context-survival-owrb-matrix-summary-v1", "unexpected summary schema")
    _require(summary.get("testRead") is False, "formal summary claims test was read")
    _require(tuple(summary.get("variants", ())) == tuple(FORMAL_VARIANTS), "variant order/count does not match the registered 2x2x2 matrix")
    _require(tuple(int(seed) for seed in summary.get("seeds", ())) == SEEDS, "seed order/count does not match the registered three seeds")
    pose_count = int(summary.get("poseCount", 0))
    _require(pose_count > 0, "summary has no validation poses")
    if expected_pose_count is not None:
        _require(pose_count == int(expected_pose_count), f"expected {expected_pose_count} validation poses, got {pose_count}")
    _require(len(summary.get("poseIndices", [])) == pose_count, "summary poseIndices length mismatch")
    _require(summary.get("bootstrap", {}).get("paired") is True, "bootstrap is not paired")
    _require(int(summary.get("bootstrap", {}).get("replicates", 0)) >= 10000, "fewer than 10000 bootstrap draws")
    _require(summary.get("bootstrap", {}).get("clusterUnit") == "outer seed cluster, inner pose resampling", "wrong bootstrap cluster unit")
    _require(set(summary.get("metricNames", [])) >= set(ALL_METRICS), "summary is missing required metrics")

    members = summary.get("members")
    _require(isinstance(members, dict), "missing members")
    _require(set(members) == set(FORMAL_VARIANTS), "formal summary does not contain exactly eight variants")

    reference_digest: str | None = None
    reference_hashes: dict[int, dict[int, str]] = {}
    for variant in FORMAL_VARIANTS:
        by_seed = members[variant]
        _require(set(by_seed) == {str(seed) for seed in SEEDS}, f"{variant}: seed set mismatch")
        for seed in SEEDS:
            member = by_seed[str(seed)]
            path = Path(str(member.get("path", "")))
            _require(path.is_file(), f"{variant}/seed{seed}: validation file is missing")
            payload = json.loads(path.read_text(encoding="utf-8"))
            _require(payload.get("schema") == "ray-context-survival-owrb-evaluation-v1", f"{variant}/seed{seed}: wrong evaluation schema")
            _require(payload.get("split") == "validation", f"{variant}/seed{seed}: evaluation is not validation")
            _require(payload.get("thresholdSource", {}).get("protocol") == "calibration_ready_pre_test", f"{variant}/seed{seed}: threshold was not calibration-frozen")
            _require(int(payload.get("thresholdSource", {}).get("testEvaluationCount", -1)) == 0, f"{variant}/seed{seed}: threshold source read test")
            _require(str(payload.get("candidateDigest", "")) and len(str(payload["candidateDigest"])) == 64, f"{variant}/seed{seed}: missing candidate digest")
            if reference_digest is None:
                reference_digest = str(payload["candidateDigest"])
            _require(str(payload["candidateDigest"]) == reference_digest, f"{variant}/seed{seed}: candidate digest mismatch")
            rows = payload.get("perPose", [])
            _require(len(rows) == pose_count, f"{variant}/seed{seed}: pose count mismatch")
            _require([int(row["poseIndex"]) for row in rows] == [int(value) for value in summary["poseIndices"]], f"{variant}/seed{seed}: pose order mismatch")
            hashes = {int(row["poseIndex"]): str(row["candidateIdSha256"]) for row in rows}
            if int(seed) not in reference_hashes:
                reference_hashes[int(seed)] = hashes
            else:
                _require(hashes == reference_hashes[int(seed)], f"{variant}/seed{seed}: candidate row hash mismatch")
            _require(_finite(member.get("threshold")) and 0.0 <= float(member["threshold"]) <= 1.0, f"{variant}/seed{seed}: invalid threshold")
            _require(int(member.get("runtimeFeatureBytes", 0)) > 0, f"{variant}/seed{seed}: missing runtime feature byte size")
            _require(int(member.get("runtimeFeatureTableDim", member.get("runtimeFeatureDim", 0))) == 156, f"{variant}/seed{seed}: unexpected fixed runtime feature dimension")
            _require(int(member.get("runtimeInputFeatureDim", 0)) == 177, f"{variant}/seed{seed}: unexpected visibility-head input dimension")
            _require(int(member.get("visibilityHeadInputDim", 0)) == 177, f"{variant}/seed{seed}: missing visibility-head input dimension")
            _require(int(member.get("rayQueryDim", 0)) == 9, f"{variant}/seed{seed}: unexpected ray query dimension")
            for section in ("poseMacro", "aggregate"):
                values = member.get(section)
                _require(isinstance(values, dict), f"{variant}/seed{seed}: missing {section}")
                for metric in ("precision", "recall", "weightedRecall", "accuracy", "balancedAccuracy", "specificity", "f1", "jaccard", "usefulCull", "badCull"):
                    _require(_finite(values.get(metric)), f"{variant}/seed{seed}/{section}: missing {metric}")
            latency = member.get("runtimeLatency", {})
            _require(all(_finite(latency.get(metric)) for metric in ("meanMs", "p50Ms", "p95Ms", "p99Ms")), f"{variant}/seed{seed}: invalid runtime latency")
            image = member.get("imageMetrics") or {}
            _require(image.get("status") == "formal_ready" and image.get("perPoseAvailable") is True, f"{variant}/seed{seed}: missing formal image metrics")
            if require_artifact_files:
                _require(Path(str(member.get("checkpoint", ""))).is_file(), f"{variant}/seed{seed}: checkpoint is missing")

    _require(reference_hashes, "no candidate hashes were loaded")
    expected_hash_digest = _sha256_text(json.dumps(reference_hashes, sort_keys=True))
    _require(summary.get("candidateHashDigest") == expected_hash_digest, "summary candidateHashDigest does not match member rows")

    webgpu_parity = _validate_webgpu_parity(
        summary.get("webgpuParity"),
        require_artifact_files=require_artifact_files,
    )

    comparisons = summary.get("comparisons", {})
    expected_comparisons = {item["name"] for item in _comparisons(FORMAL_VARIANTS)}
    _require(set(comparisons) == expected_comparisons, "pairwise comparison set does not match the registered matrix")
    for name, effect in comparisons.items():
        _require(isinstance(effect.get("metrics"), dict), f"{name}: missing metrics")
        for metric in ALL_METRICS:
            _effect_record_ok(effect["metrics"][metric], name, metric)

    factor_effects = summary.get("factorEffects", {})
    _require(set(factor_effects) == EXPECTED_FACTOR_EFFECTS, "factor effect set does not match the registered 2x2x2 design")
    _require(int(summary.get("matrixDesign", {}).get("derivedContrastCount", 0)) == len(EXPECTED_FACTOR_EFFECTS), "derived contrast count mismatch")
    for name, effect in factor_effects.items():
        for metric in ALL_METRICS:
            _effect_record_ok(effect["metrics"][metric], name, metric)
    return {
        "status": "passed",
        "schema": summary["schema"],
        "variantCount": 8,
        "seedCount": 3,
        "poseCount": pose_count,
        "candidateDigest": reference_digest,
        "pairwiseComparisonCount": len(comparisons),
        "factorEffectCount": len(factor_effects),
        "bootstrapReplicates": int(summary["bootstrap"]["replicates"]),
        "formalImageEvaluation": True,
        "webgpuParity": webgpu_parity,
        "testRead": False,
    }


def self_test() -> dict[str, Any]:
    bad = {
        "schema": "ray-context-survival-owrb-matrix-summary-v1",
        "testRead": True,
        "variants": list(FORMAL_VARIANTS),
        "seeds": list(SEEDS),
    }
    try:
        validate_summary(bad, expected_pose_count=None)
    except ValueError:
        return {"status": "passed"}
    raise AssertionError("validator accepted a test-reading summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-pose-count", type=int, default=213)
    parser.add_argument("--allow-any-pose-count", action="store_true")
    parser.add_argument("--require-artifact-files", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.summary is None:
        parser.error("--summary is required unless --self-test is used")
    result = validate_summary(
        json.loads(args.summary.read_text(encoding="utf-8")),
        expected_pose_count=None if args.allow_any_pose_count else args.expected_pose_count,
        require_artifact_files=args.require_artifact_files,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "validated", "output": str(args.output) if args.output else None, **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
