#!/usr/bin/env python3
"""Build the independent M4-v2 validation matrix summary.

The summary is validation-only.  Existing B/C/D interventions are consumed as
immutable inputs; the new A intervention is expected under the v2 output
directory.  Thresholds are checked against the calibration-only workpoint
manifest before any metric or bootstrap result is written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from m4_formal_matrix_v2_utils import (
    ALL_VARIANTS,
    ALL_EFFECTS,
    CORE_METRICS,
    DIAGNOSTIC_METRICS,
    FACTOR_EFFECTS,
    FACTOR_VARIANTS,
    SEEDS,
    STAGED_VARIANTS,
    STAGED_EFFECTS,
    UNAVAILABLE_METRICS,
    bootstrap_all_effects_by_scope,
    candidate_digest,
    candidate_identity,
    old_intervention_path,
    read_member,
    summarize_rows,
    validate_candidate_matrix,
    v2_intervention_path,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out")
    parser.add_argument("--v2-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2")
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--calibration-workpoints", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--expected-pose-count", type=int, default=664)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def member_path(args: argparse.Namespace, variant: str, seed: int) -> Path:
    if variant == "geometry_context_ray_no_inhibition":
        return v2_intervention_path(args.v2_root, variant, seed)
    return old_intervention_path(args.benchmark_root, variant, seed)


def load_workpoints(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "neuralstreamweb3d-m4-v2-calibration-workpoints-v1":
        raise ValueError(f"unexpected calibration workpoint schema: {path}")
    if bool(payload.get("testRead", True)):
        raise ValueError(f"calibration workpoint manifest is not test-free: {path}")
    return payload


def check_threshold_workpoint(member: dict[str, Any], workpoints: dict[str, Any] | None) -> dict[str, Any] | None:
    if workpoints is None:
        return None
    key = f"{member['variant']}:{member['seed']}"
    record = workpoints.get("members", {}).get(key)
    if not isinstance(record, dict):
        raise ValueError(f"missing calibration workpoint for {key}")
    threshold = float(member["payload"]["threshold"])
    selected = float(record["threshold"])
    if abs(threshold - selected) > 1e-7:
        raise ValueError(f"{key} uses threshold {threshold}, expected calibration threshold {selected}")
    source = member["payload"].get("thresholdSource", {})
    provenance_path = str(record.get("source", ""))
    if str(source.get("protocol")) != "calibration_ready_pre_test":
        raise ValueError(f"{key} threshold is not calibration-ready")
    if int(source.get("testEvaluationCount", -1)) != 0:
        raise ValueError(f"{key} threshold has test provenance")
    selected_record = source.get("m4v2CalibrationSelection")
    if isinstance(selected_record, dict) and str(selected_record.get("source", "")) != provenance_path:
        # This is a diagnostic mismatch only when a caller copied the source
        # file.  The threshold and no-test provenance remain authoritative.
        pass
    return record


def load_matrix(args: argparse.Namespace) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any] | None]:
    if args.expected_pose_count <= 0:
        raise ValueError("expected pose count must be positive")
    workpoints = load_workpoints(args.calibration_workpoints)
    members: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in ALL_VARIANTS:
        for seed in SEEDS:
            path = member_path(args, variant, seed)
            member = read_member(path, variant, seed, args.repo_root, args.expected_pose_count)
            check_threshold_workpoint(member, workpoints)
            members[(variant, seed)] = member
    validate_candidate_matrix(members)
    return members, workpoints


def member_record(member: dict[str, Any], workpoint: dict[str, Any] | None, args: argparse.Namespace) -> dict[str, Any]:
    summary = summarize_rows(member["rows"].values(), lcb_replicates=args.bootstrap_replicates, lcb_seed=member["seed"] + 7001)
    payload = member["payload"]
    source = payload.get("thresholdSource", {})
    return {
        "variant": member["variant"],
        "factorCode": member.get("factorCode"),
        "seed": member["seed"],
        "input": member["path"],
        "inputSha256": member["artifactSha256"],
        "checkpoint": member["checkpoint"],
        "checkpointSha256": member["checkpointSha256"],
        "checkpointBytes": member["checkpointBytes"],
        "runtimeFeatures": member["runtimeFeatures"],
        "runtimeFeaturesSha256": member["runtimeFeaturesSha256"],
        "runtimeFeatureBytes": member["runtimeFeaturesBytes"],
        "threshold": float(payload["threshold"]),
        "thresholdSource": source,
        "calibrationWorkpoint": workpoint,
        "poseCount": summary["poseCount"],
        "poseMacro": summary["poseMacro"],
        "aggregate": summary["aggregate"],
        "safety": summary["safety"],
        "runtime": {
            "forwardLatencyMs": {"status": "not_available", "reason": UNAVAILABLE_METRICS["forward_latency_ms"]},
            "fixedFeatureTableBytes": member["runtimeFeaturesBytes"],
            "inferenceInput": "fixed per-instance geometry/context/proxy table plus ray features; exact candidate tensor timing not stored",
        },
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    members, workpoints = load_matrix(args)
    identity = validate_candidate_matrix(members)
    member_summaries: dict[str, Any] = {}
    for key, member in sorted(members.items()):
        workpoint = workpoints.get("members", {}).get(f"{key[0]}:{key[1]}") if workpoints else None
        member_summaries[f"{key[0]}:{key[1]}"] = member_record(member, workpoint, args)

    bootstrap_metrics = DIAGNOSTIC_METRICS
    effects = bootstrap_all_effects_by_scope(
        members,
        ALL_EFFECTS,
        bootstrap_metrics,
        replicates=args.bootstrap_replicates,
        seed=20260803,
    )

    return {
        "schema": "neuralstreamweb3d-formal-m4-matrix-summary-v2",
        "version": 2,
        "split": "validation",
        "testRead": False,
        "status": "validation_summary_only; formal test not read",
        "scene": "hkust-v3",
        "fov": {"modelAndBackCameraDegrees": 66, "renderCameraDegrees": 60},
        "candidateSemantics": "stored back-camera candidates; no GT union, candidate cap, or frontend whitelist",
        "gtSemantics": "stored validation visible_ids; visible_weights are importance evidence, not exact pixel coverage",
        "expectedPoseCount": args.expected_pose_count,
        "poseCount": identity["poseCount"],
        "seeds": list(SEEDS),
        "candidateIdentity": identity,
        "variants": {
            "staged": list(STAGED_VARIANTS),
            "factorial": list(FACTOR_VARIANTS),
            "all": list(ALL_VARIANTS),
        },
        "factorDefinition": {
            "A": "geometry_context_ray_no_inhibition",
            "B": "geometry_context_proxy_ray_no_inhibition",
            "C": "geometry_context_ray",
            "D": "full",
            "directionProxyMainEffectWithoutInhibition": "B-A",
            "explicitInhibitionMainEffectWithoutProxy": "C-A",
            "directionProxyEffectWithInhibition": "D-C",
            "explicitInhibitionEffectWithProxy": "D-B",
            "interaction": "D-B-C+A",
        },
        "thresholdPolicy": {
            "source": "each checkpoint's calibration split only",
            "poseRecallFloor": 0.95,
            "weightedRecallFloor": 0.99,
            "weightedRecallLowerConfidenceBoundFloor": 0.99,
            "safetyWorkpoint": "maximize useful cull after all three safety constraints; no validation threshold scan",
            "diagnosticWorkpoints": "best F1, maximum precision, and old/fixed thresholds are diagnostic only",
        },
        "members": member_summaries,
        "stagedEffects": {name: effects[name] for name in STAGED_EFFECTS},
        "factorEffects": {name: effects[name] for name in FACTOR_EFFECTS},
        "bootstrap": {
            "replicates": args.bootstrap_replicates,
            "confidence": 0.95,
            "clusterUnit": "seed, then validation pose within seed",
            "pairedPoseRequirement": True,
        },
        "unavailableMetrics": UNAVAILABLE_METRICS,
        "metricDefinitions": {
            "TP": "P intersect G",
            "FP": "P minus G",
            "FN": "G minus P",
            "TN": "C minus (P union G)",
            "poseMacro": "compute each pose metric first, then arithmetic mean over poses",
            "aggregate": "sum counts and visible weights over all poses before deriving ratios",
            "usefulCull": "TN / candidate; correct removal of invisible candidates only",
            "badCull": "FN / candidate; incorrect removal of visible candidates",
            "instanceAccuracy": "(TP + TN) / candidate",
            "balancedAccuracy": "(recall + specificity) / 2",
            "specificity": "TN / (TN + FP)",
            "safetyFactor": "min(1, pose_recall / 0.95) * min(1, weighted_recall / 0.99)",
        },
    }


def self_test() -> dict[str, Any]:
    from m4_formal_matrix_v2_utils import bootstrap_effects_by_scope

    def row(pose: int, tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
        candidate = tp + fp + fn + tn
        gt = tp + fn
        recall = tp / max(1, gt)
        precision = tp / max(1, tp + fp)
        specificity = tn / max(1, tn + fp)
        return {
            "pose_index": pose,
            "candidate_id_sha256": f"{pose:064x}",
            "candidate_count": candidate,
            "gt_count": gt,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "weighted_recall": recall,
            "f1": 2 * precision * recall / max(1e-8, precision + recall),
            "jaccard": tp / max(1, tp + fp + fn),
            "accuracy": (tp + tn) / candidate,
            "balanced_accuracy": (recall + specificity) / 2,
            "specificity": specificity,
            "useful_cull": tn / candidate,
            "bad_cull": fn / candidate,
            "avg_pred_count": tp + fp,
            "weighted_tp": float(tp),
            "weighted_gt": float(gt),
        }

    members: dict[tuple[str, int], dict[str, Any]] = {}
    for variant in FACTOR_VARIANTS:
        for seed in SEEDS:
            members[(variant, seed)] = {"rows": {0: row(0, 1, 1, 0, 2), 1: row(1, 1, 0, 1, 2)}}
    summary = summarize_rows(members[("full", SEEDS[0])]["rows"].values(), lcb_replicates=200, lcb_seed=1)
    effects = bootstrap_effects_by_scope(members, FACTOR_EFFECTS["B_minus_A"], ("recall", "useful_cull"), replicates=10000, seed=2)
    assert summary["aggregate"]["accuracy"] == 0.75
    assert effects["pose_macro"]["recall"]["bootstrap_replicates"] == 10000
    return {"status": "passed", "scopes": sorted(effects), "metricCount": 2}


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.bootstrap_replicates < 10000:
        raise ValueError("--bootstrap-replicates must be at least 10000")
    if args.output is None:
        raise ValueError("--output is required unless --self-test is used")
    payload = build_summary(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "poseCount": payload["poseCount"], "testRead": False}, indent=2))


if __name__ == "__main__":
    main()
