#!/usr/bin/env python3
"""Validate the M4-v2 summary/route schema and matrix completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from m4_formal_matrix_v2_utils import ALL_VARIANTS, DIAGNOSTIC_METRICS, FACTOR_EFFECTS, FACTOR_VARIANTS, SEEDS, STAGED_EFFECTS


EXPECTED_CANDIDATE_DIGEST = "8bd3e6a840c7624e2de459ef8057b24380c91936383c29f2368d93801f4c17bf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--route", type=Path, default=None)
    parser.add_argument("--expected-pose-count", type=int, default=664)
    parser.add_argument("--expected-candidate-digest", default=EXPECTED_CANDIDATE_DIGEST)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def validate_summary(
    payload: dict,
    expected_pose_count: int = 664,
    expected_candidate_digest: str = EXPECTED_CANDIDATE_DIGEST,
) -> None:
    if payload.get("schema") != "neuralstreamweb3d-formal-m4-matrix-summary-v2":
        raise ValueError("unexpected M4-v2 summary schema")
    if payload.get("split") != "validation" or payload.get("testRead") is not False:
        raise ValueError("summary is not validation-only")
    if int(payload.get("poseCount", -1)) != expected_pose_count:
        raise ValueError("unexpected pose count")
    candidate_identity = payload.get("candidateIdentity", {})
    if candidate_identity.get("candidateDigest") != expected_candidate_digest:
        raise ValueError(
            "candidate digest drift: "
            f"expected {expected_candidate_digest}, got {candidate_identity.get('candidateDigest')}"
        )
    if sorted(payload.get("seeds", [])) != list(SEEDS):
        raise ValueError("registered seed set is incomplete")
    if payload.get("variants", {}).get("all") != list(ALL_VARIANTS):
        raise ValueError("variant list is not the registered staged-plus-factor matrix")
    members = payload.get("members")
    expected_keys = {f"{variant}:{seed}" for variant in ALL_VARIANTS for seed in SEEDS}
    if not isinstance(members, dict) or set(members) != expected_keys:
        raise ValueError("summary does not contain exactly 6 variants x 3 seeds")
    for key, record in members.items():
        if int(record.get("poseCount", -1)) != expected_pose_count:
            raise ValueError(f"member {key} has the wrong pose count")
        if not Path(record["input"]).is_file():
            raise FileNotFoundError(record["input"])
        if not record.get("candidateIdentity", payload.get("candidateIdentity")):
            raise ValueError(f"member {key} lacks candidate identity provenance")
    effects = payload.get("factorEffects")
    if not isinstance(effects, dict) or set(effects) != set(FACTOR_EFFECTS):
        raise ValueError("factor effect set is incomplete")
    for name in FACTOR_EFFECTS:
        comparisons = effects[name].get("comparisons", {})
        for scope in ("pose_macro", "aggregate"):
            if set(comparisons.get(scope, {})) != set(DIAGNOSTIC_METRICS):
                raise ValueError(f"{name}/{scope} metric set is incomplete")
            for metric, record in comparisons[scope].items():
                ci = record.get("ci95")
                if int(record.get("bootstrap_replicates", 0)) < 10000 or not isinstance(ci, list) or len(ci) != 2:
                    raise ValueError(f"invalid bootstrap record {name}/{scope}/{metric}")
    staged = payload.get("stagedEffects")
    if not isinstance(staged, dict) or set(staged) != set(STAGED_EFFECTS):
        raise ValueError("staged input effects are incomplete")
    for name in STAGED_EFFECTS:
        comparisons = staged[name].get("comparisons", {})
        for scope in ("pose_macro", "aggregate"):
            if set(comparisons.get(scope, {})) != set(DIAGNOSTIC_METRICS):
                raise ValueError(f"{name}/{scope} staged metric set is incomplete")
            for metric, record in comparisons[scope].items():
                ci = record.get("ci95")
                if int(record.get("bootstrap_replicates", 0)) < 10000 or not isinstance(ci, list) or len(ci) != 2:
                    raise ValueError(f"invalid staged bootstrap record {name}/{scope}/{metric}")


def validate_route(payload: dict, summary_path: Path) -> None:
    if payload.get("schema") != "neuralstreamweb3d-formal-m4-route-decision-v2":
        raise ValueError("unexpected M4-v2 route schema")
    if payload.get("testRead") is not False or payload.get("split") != "validation":
        raise ValueError("route is not validation-only")
    if Path(payload.get("sourceSummary", "")).resolve() != summary_path.resolve():
        raise ValueError("route does not point to the supplied summary")
    if payload.get("route") not in {"route_a_directional_proxy", "route_b_system"}:
        raise ValueError("unknown M4-v2 route")


def self_test() -> dict:
    return {"status": "passed", "requiredVariants": len(ALL_VARIANTS), "requiredSeeds": len(SEEDS)}


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.summary is None:
        raise ValueError("--summary is required unless --self-test is used")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    validate_summary(summary, args.expected_pose_count, args.expected_candidate_digest)
    if args.route is not None:
        validate_route(json.loads(args.route.read_text(encoding="utf-8")), args.summary)
    print(json.dumps({"status": "passed", "summary": str(args.summary), "route": str(args.route) if args.route else None}, indent=2))


if __name__ == "__main__":
    main()
