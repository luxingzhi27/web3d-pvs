#!/usr/bin/env python3
"""Audit the immutable M4 artifacts before the M4-v2 factor evaluation.

The auditor is deliberately read-only with respect to the existing M4
outputs.  It verifies that every stored validation intervention contains the
per-pose confusion counts and that all members share the same poses and
back-camera candidates.  It also records the missing fourth factor member so
the v2 runner can train only the artifact that is genuinely absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SEEDS = (20260801, 20260802, 20260803)
OLD_VARIANTS = (
    "aabb_ray",
    "geometry_ray",
    "geometry_context_ray",
    "geometry_context_proxy_ray_no_inhibition",
    "full",
)
FACTOR_VARIANTS = (
    "geometry_context_ray_no_inhibition",
    "geometry_context_proxy_ray_no_inhibition",
    "geometry_context_ray",
    "full",
)
FACTOR_CODES = {
    "geometry_context_ray_no_inhibition": "A",
    "geometry_context_proxy_ray_no_inhibition": "B",
    "geometry_context_ray": "C",
    "full": "D",
}
REQUIRED_ROW_FIELDS = (
    "pose_index",
    "candidate_id_sha256",
    "candidate_count",
    "gt_count",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
    "jaccard",
    "weighted_recall",
    "accuracy",
    "balanced_accuracy",
    "specificity",
    "useful_cull",
    "bad_cull",
    "avg_pred_count",
    "weighted_tp",
    "weighted_gt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old-root",
        type=Path,
        default=Path("neural_instance_culling/benchmark/out"),
        help="Root containing the immutable m4_formal_baseline_* directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neural_instance_culling/benchmark/out/m4_formal_matrix_validation_v2/input_manifest.json"),
    )
    parser.add_argument("--expected-pose-count", type=int, default=664)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def intervention_path(root: Path, variant: str, seed: int) -> Path:
    experiment = f"pvs_m4_ablation_{variant}_rvl_strong_v2_hkust_spatial_fov66_seed{seed}_full40"
    return root / f"m4_formal_baseline_{experiment}_validation" / "interventions.json"


def validate_threshold_source(payload: dict[str, Any], path: Path) -> None:
    source = payload.get("thresholdSource")
    if not isinstance(source, dict):
        raise ValueError(f"missing structured thresholdSource: {path}")
    if source.get("protocol") != "calibration_ready_pre_test":
        raise ValueError(f"non-calibration threshold protocol in {path}: {source.get('protocol')!r}")
    if int(source.get("testEvaluationCount", -1)) != 0:
        raise ValueError(f"test-derived threshold provenance in {path}")
    if bool(source.get("testThresholdOverride", False)):
        raise ValueError(f"test threshold override in {path}")


def validate_row(row: dict[str, Any], path: Path) -> None:
    missing = [field for field in REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{path} row {row.get('pose_index')} missing fields: {missing}")
    pose = int(row["pose_index"])
    candidate = int(row["candidate_count"])
    gt = int(row["gt_count"])
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    tn = int(row["tn"])
    if pose < 0 or candidate < 0 or gt < 0 or min(tp, fp, fn, tn) < 0:
        raise ValueError(f"negative count in {path}, pose {pose}")
    if tp + fp + fn + tn != candidate:
        raise ValueError(f"candidate confusion counts do not close in {path}, pose {pose}")
    if tp + fn != gt:
        raise ValueError(f"GT confusion counts do not close in {path}, pose {pose}")
    if gt > candidate:
        raise ValueError(f"GT is not a subset of candidates in {path}, pose {pose}")
    if not isinstance(row["candidate_id_sha256"], str) or len(row["candidate_id_sha256"]) != 64:
        raise ValueError(f"invalid candidate hash in {path}, pose {pose}")
    for key in REQUIRED_ROW_FIELDS:
        if key not in {"pose_index", "candidate_id_sha256"}:
            finite(row[key], f"{path}:{pose}:{key}")


def read_member(path: Path, variant: str, seed: int, expected_pose_count: int) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "pvs-proxy-intervention-v2":
        raise ValueError(f"unexpected intervention schema in {path}: {payload.get('schema')!r}")
    if payload.get("split") != "validation":
        raise ValueError(f"non-validation intervention in {path}")
    validate_threshold_source(payload, path)
    evaluation_seed = int(payload.get("seed", -1))
    if evaluation_seed < 0:
        raise ValueError(f"missing evaluation seed in {path}")
    if int(payload.get("posePlan", {}).get("poseCount", -1)) != expected_pose_count:
        raise ValueError(f"pose count mismatch in {path}")
    rows = payload.get("perPose", {}).get("baseline")
    if not isinstance(rows, list) or len(rows) != expected_pose_count:
        raise ValueError(f"baseline rows mismatch in {path}")
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"non-object pose row in {path}")
        validate_row(row, path)
        pose = int(row["pose_index"])
        if pose in indexed:
            raise ValueError(f"duplicate pose {pose} in {path}")
        indexed[pose] = row
    if len(indexed) != expected_pose_count:
        raise ValueError(f"unique pose count mismatch in {path}")
    aggregate = payload.get("interventions", {}).get("baseline", {}).get("aggregate")
    if not isinstance(aggregate, dict):
        raise ValueError(f"missing aggregate baseline in {path}")
    for key in ("precision", "recall", "weighted_recall", "accuracy", "balanced_accuracy", "specificity", "useful_cull", "bad_cull"):
        finite(aggregate.get(key), f"{path}:aggregate:{key}")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    runtime_features = Path(str(payload.get("runtimeFeatures", "")))
    if not checkpoint.is_file() or not runtime_features.is_file():
        raise FileNotFoundError(f"member dependencies missing for {path}: {checkpoint}, {runtime_features}")
    checkpoint_sha = sha256_file(checkpoint)
    feature_sha = sha256_file(runtime_features)
    if checkpoint_sha != payload.get("checkpointSha256"):
        raise ValueError(f"checkpoint SHA mismatch in {path}")
    if feature_sha != payload.get("runtimeFeaturesSha256"):
        raise ValueError(f"runtime feature SHA mismatch in {path}")
    return {
        "variant": variant,
        "factorCode": FACTOR_CODES.get(variant),
        "seed": seed,
        "evaluationSeed": evaluation_seed,
        "path": str(path.resolve()),
        "artifactSha256": sha256_file(path),
        "checkpoint": str(checkpoint),
        "checkpointSha256": checkpoint_sha,
        "runtimeFeatures": str(runtime_features),
        "runtimeFeaturesSha256": feature_sha,
        "datasetDir": payload.get("datasetDir"),
        "runtimeMeta": payload.get("runtimeMeta"),
        "threshold": finite(payload.get("threshold"), f"{path}:threshold"),
        "thresholdSource": payload["thresholdSource"],
        "posePlan": payload["posePlan"],
        "rowCount": len(indexed),
        "rows": indexed,
        "aggregate": aggregate,
    }


def candidate_identity(member: dict[str, Any]) -> dict[int, tuple[str, int]]:
    return {
        pose: (str(row["candidate_id_sha256"]), int(row["candidate_count"]))
        for pose, row in member["rows"].items()
    }


def audit_matrix(old_root: Path, expected_pose_count: int) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for variant in OLD_VARIANTS:
        for seed in SEEDS:
            path = intervention_path(old_root, variant, seed)
            if not path.is_file():
                missing.append({"variant": variant, "seed": seed, "path": str(path)})
                continue
            members.append(read_member(path, variant, seed, expected_pose_count))
    if missing:
        raise FileNotFoundError(f"missing old M4 artifacts: {missing}")
    if len(members) != len(OLD_VARIANTS) * len(SEEDS):
        raise ValueError(f"expected 15 old members, found {len(members)}")
    reference = candidate_identity(members[0])
    for member in members[1:]:
        current = candidate_identity(member)
        if current != reference:
            for pose in sorted(set(reference) | set(current)):
                if reference.get(pose) != current.get(pose):
                    raise ValueError(
                        f"candidate identity mismatch at pose {pose}: "
                        f"{members[0]['variant']}/{members[0]['seed']} vs {member['variant']}/{member['seed']}"
                    )
    factor_members = {
        f"{variant}:{seed}": next(
            member for member in members if member["variant"] == variant and member["seed"] == seed
        )
        for variant in FACTOR_VARIANTS
        for seed in SEEDS
        if any(member["variant"] == variant and member["seed"] == seed for member in members)
    }
    missing_factor = [
        {"variant": variant, "factorCode": FACTOR_CODES[variant], "seed": seed}
        for variant in FACTOR_VARIANTS
        for seed in SEEDS
        if f"{variant}:{seed}" not in factor_members
    ]
    for member in members:
        member.pop("rows", None)
    return {
        "schema": "neuralstreamweb3d-m4-v2-input-audit-v1",
        "status": "passed" if not missing_factor else "passed_with_missing_factor_members",
        "oldArtifactsReadOnly": True,
        "split": "validation",
        "testRead": False,
        "expectedPoseCount": expected_pose_count,
        "oldVariants": list(OLD_VARIANTS),
        "factorVariants": list(FACTOR_VARIANTS),
        "seeds": list(SEEDS),
        "validatedMemberCount": len(members),
        "candidateIdentity": {
            "poseCount": len(reference),
            "candidateDigest": hashlib.sha256(
                json.dumps(reference, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "source": "stored back-camera candidate hash and count from each pose; no GT union or candidate cap",
        },
        "missingFactorMembers": missing_factor,
        "members": members,
        "unavailableMetrics": {
            "visualUtilityRecall": "not_available_in_interventions_json; requires unified visual utility evaluation",
            "imagePixelMetrics": "not_available_in_interventions_json; requires same-pose 60-degree instance-ID rendering",
            "glbCountBytes": "not_available_in_interventions_json; requires strict GLB byte index evaluation",
            "forwardLatency": "not_available_in_interventions_json; requires runner timing",
        },
    }


def self_test() -> dict[str, Any]:
    row = {
        "pose_index": 0,
        "candidate_id_sha256": "0" * 64,
        "candidate_count": 4,
        "gt_count": 2,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "tn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "jaccard": 1 / 3,
        "weighted_recall": 0.5,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "specificity": 0.5,
        "useful_cull": 0.25,
        "bad_cull": 0.25,
        "avg_pred_count": 2,
        "weighted_tp": 1,
        "weighted_gt": 2,
    }
    validate_row(row, Path("<fixture>"))
    return {"status": "passed", "requiredRowFields": len(REQUIRED_ROW_FIELDS)}


def main() -> None:
    args = parse_args()
    if args.expected_pose_count <= 0:
        raise ValueError("expected pose count must be positive")
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    payload = audit_matrix(args.old_root, args.expected_pose_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "status": payload["status"], "missingFactorMembers": payload["missingFactorMembers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
