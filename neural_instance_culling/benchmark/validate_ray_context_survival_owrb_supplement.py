#!/usr/bin/env python3
"""Validate the independent five-member OWRB supplement matrix.

The core validator intentionally accepts only the registered 2x2x2 matrix.
This validator has a separate contract for the supplement, whose members
change one implementation choice at a time (context width, ray encoding,
survival parameterization, or relation evidence).  It checks the shared
validation protocol without pretending that the supplement is a factorial
main-effect experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from run_ray_context_survival_owrb_supplement import SUPPLEMENT_VARIANTS, _comparisons
from summarize_ray_context_survival_owrb import ALL_METRICS
from validate_ray_context_survival_owrb import _effect_record_ok


SEEDS = (20260801, 20260802, 20260803)
SUPPLEMENT_ORDER = tuple(SUPPLEMENT_VARIANTS)
EXPECTED_RUNTIME_DIMS: dict[str, tuple[int, int, int]] = {
    # fixed table dimension, visibility-head input dimension, ray query dimension
    "triangle_context32_direct9_monotone": (156, 177, 9),
    "triangle_context64_direct9_monotone": (188, 217, 9),
    "triangle_context32_fourier117_monotone": (156, 285, 117),
    "triangle_context32_direct9_unconstrained28": (156, 177, 9),
    "aabb_context32_direct9_monotone": (156, 177, 9),
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


def _validate_member(
    variant: str,
    seed: int,
    member: dict[str, Any],
    *,
    pose_count: int,
    pose_indices: list[int],
    expected_digest: str | None,
    reference_hashes: dict[int, dict[int, str]],
    require_artifact_files: bool,
) -> tuple[str, dict[int, str]]:
    expected_table_dim, expected_input_dim, expected_ray_dim = EXPECTED_RUNTIME_DIMS[variant]
    path = Path(str(member.get("path", "")))
    _require(path.is_file(), f"{variant}/seed{seed}: validation file is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema") == "ray-context-survival-owrb-evaluation-v1", f"{variant}/seed{seed}: wrong evaluation schema")
    _require(payload.get("split") == "validation", f"{variant}/seed{seed}: evaluation is not validation")
    _require(payload.get("testRead") is False, f"{variant}/seed{seed}: evaluation claims test was read")

    threshold_source = payload.get("thresholdSource") or {}
    _require(threshold_source.get("protocol") == "calibration_ready_pre_test", f"{variant}/seed{seed}: threshold was not calibration-frozen")
    _require(int(threshold_source.get("testEvaluationCount", -1)) == 0, f"{variant}/seed{seed}: threshold source read test")
    _require(threshold_source.get("safeWorkpoint") is True, f"{variant}/seed{seed}: calibration did not select a safe workpoint")
    selection = threshold_source.get("selection") or {}
    _require(_finite(selection.get("pose_weighted_recall")) and float(selection["pose_weighted_recall"]) > 0.99, f"{variant}/seed{seed}: calibration weighted recall is not above 0.99")
    _require(
        _finite(selection.get("weighted_recall_lower_confidence_bound"))
        and float(selection["weighted_recall_lower_confidence_bound"]) > 0.99,
        f"{variant}/seed{seed}: calibration weighted-recall lower bound is not above 0.99",
    )
    _require(payload.get("safeWorkpoint") is True, f"{variant}/seed{seed}: evaluation is not marked safe")

    candidate_digest = str(payload.get("candidateDigest", ""))
    _require(len(candidate_digest) == 64, f"{variant}/seed{seed}: missing candidate digest")
    if expected_digest is not None:
        _require(candidate_digest == expected_digest, f"{variant}/seed{seed}: candidate digest mismatch")

    rows = payload.get("perPose") or []
    _require(len(rows) == pose_count, f"{variant}/seed{seed}: pose count mismatch")
    indices = [int(row["poseIndex"]) for row in rows]
    _require(indices == pose_indices, f"{variant}/seed{seed}: pose order mismatch")
    hashes: dict[int, str] = {}
    for row in rows:
        pose = int(row["poseIndex"])
        digest = str(row.get("candidateIdSha256", ""))
        _require(len(digest) == 64, f"{variant}/seed{seed}/pose{pose}: missing candidate row hash")
        hashes[pose] = digest
        _require(isinstance(row.get("imageMetrics"), dict), f"{variant}/seed{seed}/pose{pose}: missing image metrics")
    if seed in reference_hashes:
        _require(hashes == reference_hashes[seed], f"{variant}/seed{seed}: candidate row hash mismatch")
    else:
        reference_hashes[seed] = hashes

    dims = (
        int(member.get("runtimeFeatureTableDim", member.get("runtimeFeatureDim", 0))),
        int(member.get("visibilityHeadInputDim", member.get("runtimeInputFeatureDim", 0))),
        int(member.get("rayQueryDim", 0)),
    )
    _require(dims == (expected_table_dim, expected_input_dim, expected_ray_dim), f"{variant}/seed{seed}: runtime dimensions {dims} do not match {EXPECTED_RUNTIME_DIMS[variant]}")
    _require(int(member.get("runtimeFeatureBytes", 0)) > 0, f"{variant}/seed{seed}: missing runtime feature byte size")
    _require(_finite(member.get("threshold")) and 0.0 <= float(member["threshold"]) <= 1.0, f"{variant}/seed{seed}: invalid threshold")
    for section in ("poseMacro", "aggregate"):
        values = member.get(section)
        _require(isinstance(values, dict), f"{variant}/seed{seed}: missing {section}")
        for metric in ("precision", "recall", "weightedRecall", "accuracy", "balancedAccuracy", "specificity", "f1", "jaccard", "usefulCull", "badCull"):
            _require(_finite(values.get(metric)), f"{variant}/seed{seed}/{section}: missing {metric}")
    latency = member.get("runtimeLatency") or {}
    _require(all(_finite(latency.get(metric)) for metric in ("meanMs", "p50Ms", "p95Ms", "p99Ms")), f"{variant}/seed{seed}: invalid runtime latency")
    image = member.get("imageMetrics") or {}
    _require(image.get("status") == "formal_ready" and image.get("perPoseAvailable") is True, f"{variant}/seed{seed}: missing formal image metrics")
    if require_artifact_files:
        _require(Path(str(member.get("checkpoint", ""))).is_file(), f"{variant}/seed{seed}: checkpoint is missing")
    return candidate_digest, hashes


def _validate_webgpu_parity_if_registered(parity: Any) -> None:
    """Supplement parity is optional because non-default variants are offline-only.

    When a parity record is attached, reject malformed records rather than
    silently accepting an unsupported hardware claim.  The shared default
    direct-nine path is covered by the core formal parity bundle.
    """
    if not isinstance(parity, dict) or parity.get("status") == "not_registered":
        return
    _require(parity.get("passed") is True, "registered supplement WebGPU parity did not pass")
    _require(parity.get("backend") == "webgpu", "registered supplement WebGPU parity is not WebGPU")
    _require(parity.get("status") in {"software_numeric_parity_passed", "hardware_numeric_parity_passed"}, "invalid registered supplement WebGPU parity status")
    _require(_finite(parity.get("maxAbs")) and _finite(parity.get("maxRelative")), "invalid supplement WebGPU parity error")


def validate_summary(
    summary: dict[str, Any],
    *,
    expected_pose_count: int | None = 213,
    require_artifact_files: bool = False,
) -> dict[str, Any]:
    _require(summary.get("schema") == "ray-context-survival-owrb-matrix-summary-v1", "unexpected summary schema")
    _require(summary.get("mode") == "formal" and summary.get("supplement") is True, "summary is not a formal supplement")
    _require(summary.get("testRead") is False, "formal supplement summary claims test was read")
    _require(tuple(summary.get("variants", ())) == SUPPLEMENT_ORDER, "variant order/count does not match the registered five-member supplement")
    _require(tuple(int(seed) for seed in summary.get("seeds", ())) == SEEDS, "seed order/count does not match the registered three seeds")
    pose_count = int(summary.get("poseCount", 0))
    _require(pose_count > 0, "summary has no validation poses")
    if expected_pose_count is not None:
        _require(pose_count == int(expected_pose_count), f"expected {expected_pose_count} validation poses, got {pose_count}")
    pose_indices = [int(value) for value in summary.get("poseIndices", [])]
    _require(len(pose_indices) == pose_count, "summary poseIndices length mismatch")
    _require(summary.get("bootstrap", {}).get("paired") is True, "bootstrap is not paired")
    _require(int(summary.get("bootstrap", {}).get("replicates", 0)) >= 10000, "fewer than 10000 bootstrap draws")
    _require(summary.get("bootstrap", {}).get("clusterUnit") == "outer seed cluster, inner pose resampling", "wrong bootstrap cluster unit")
    _require(set(summary.get("metricNames", [])) >= set(ALL_METRICS), "summary is missing required metrics")
    _require(summary.get("imageEvaluation", {}).get("status") == "formal_ready", "supplement image evaluation is not formal-ready")

    design = summary.get("matrixDesign") or {}
    _require(design.get("type") == "registered_supplement_mechanism_audit", "supplement summary has the wrong matrix design type")
    _require(int(design.get("variantCount", 0)) == len(SUPPLEMENT_ORDER), "supplement variant count mismatch")
    _require(int(design.get("derivedContrastCount", -1)) == 0, "supplement must not emit core factorial effects")
    _require(summary.get("factorEffects") == {}, "supplement unexpectedly contains factorial factor effects")
    _validate_webgpu_parity_if_registered(summary.get("webgpuParity"))

    members = summary.get("members")
    _require(isinstance(members, dict) and tuple(members) == SUPPLEMENT_ORDER, "summary members do not match the registered supplement")
    reference_digest: str | None = None
    reference_hashes: dict[int, dict[int, str]] = {}
    for variant in SUPPLEMENT_ORDER:
        by_seed = members[variant]
        _require(set(by_seed) == {str(seed) for seed in SEEDS}, f"{variant}: seed set mismatch")
        for seed in SEEDS:
            digest, _hashes = _validate_member(
                variant,
                seed,
                by_seed[str(seed)],
                pose_count=pose_count,
                pose_indices=pose_indices,
                expected_digest=reference_digest,
                reference_hashes=reference_hashes,
                require_artifact_files=require_artifact_files,
            )
            if reference_digest is None:
                reference_digest = digest

    _require(reference_digest is not None, "no candidate digest was loaded")
    _require(len(reference_hashes) == len(SEEDS), "candidate hashes are missing for one or more seeds")
    canonical_hashes = reference_hashes[SEEDS[0]]
    for seed in SEEDS[1:]:
        _require(reference_hashes[seed] == canonical_hashes, f"candidate row hashes differ across seeds at seed {seed}")
    expected_hash_digest = _sha256_text(json.dumps(reference_hashes, sort_keys=True))
    _require(summary.get("candidateHashDigest") == expected_hash_digest, "summary candidateHashDigest does not match member rows")

    expected_comparisons = {item["name"] for item in _comparisons(SUPPLEMENT_VARIANTS)}
    comparisons = summary.get("comparisons", {})
    _require(set(comparisons) == expected_comparisons, "supplement comparison set does not match the registered mechanism comparisons")
    for name, effect in comparisons.items():
        _require(isinstance(effect.get("metrics"), dict), f"{name}: missing metrics")
        for metric in ALL_METRICS:
            _effect_record_ok(effect["metrics"][metric], name, metric)
    return {
        "status": "passed",
        "schema": summary["schema"],
        "matrix": "supplement",
        "variantCount": len(SUPPLEMENT_ORDER),
        "seedCount": len(SEEDS),
        "poseCount": pose_count,
        "candidateDigest": reference_digest,
        "pairwiseComparisonCount": len(comparisons),
        "factorEffectCount": 0,
        "bootstrapReplicates": int(summary["bootstrap"]["replicates"]),
        "formalImageEvaluation": True,
        "testRead": False,
    }


def self_test() -> dict[str, Any]:
    bad = {
        "schema": "ray-context-survival-owrb-matrix-summary-v1",
        "mode": "formal",
        "supplement": True,
        "testRead": True,
        "variants": list(SUPPLEMENT_ORDER),
        "seeds": list(SEEDS),
    }
    try:
        validate_summary(bad, expected_pose_count=None)
    except ValueError:
        return {"status": "passed"}
    raise AssertionError("supplement validator accepted a test-reading summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
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
