#!/usr/bin/env python3
"""Freeze M4-v2 thresholds from each member's calibration rows only.

The older M4 checkpoints were selected with a weighted-recall rule that did
not require pose recall.  M4-v2 records a stricter calibration workpoint for
the same checkpoint: pose recall >= 0.95, weighted recall > 0.99, and the
available one-sided weighted-recall lower bound > 0.99.  Validation is never
read by this tool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260801, 20260802, 20260803)
VARIANTS = (
    "aabb_ray",
    "geometry_ray",
    "geometry_context_ray_no_inhibition",
    "geometry_context_proxy_ray_no_inhibition",
    "geometry_context_ray",
    "full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=ROOT / "neural_instance_culling/model/out")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def experiment_name(variant: str, seed: int) -> str:
    prefix = "pvs_m4_v2_ablation" if variant == "geometry_context_ray_no_inhibition" else "pvs_m4_ablation"
    return f"{prefix}_{variant}_rvl_strong_v2_hkust_spatial_fov66_seed{seed}_full40"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def useful_cull(row: dict[str, Any]) -> float:
    candidate = float(row["avg_candidate_count"])
    total = float(row["eval_pose_count"])
    tp = float(row["tp"])
    fp = float(row["fp"])
    fn = float(row["fn"])
    tn = candidate * total - tp - fp - fn
    return tn / max(1.0, candidate * total)


def select_from_summary(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("protocol") != "calibration_ready_pre_test":
        raise ValueError(f"M4-v2 calibration source is not pre-test: {summary_path}")
    if int(payload.get("testEvaluationCount", -1)) != 0:
        raise ValueError(f"M4-v2 calibration source has test evaluations: {summary_path}")
    rows = payload.get("calibration", {}).get("thresholdRows") or payload.get("thresholdRows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"calibration threshold rows are missing: {summary_path}")
    eligible: list[dict[str, Any]] = []
    best_f1 = max(rows, key=lambda row: float(row.get("pose_f1", -1.0)))
    highest_precision = max(rows, key=lambda row: float(row.get("pose_precision", -1.0)))
    original_threshold = payload.get("frozenThreshold")
    fixed_row = None
    if original_threshold is not None:
        fixed_row = min(rows, key=lambda row: abs(float(row.get("threshold", 0.0)) - float(original_threshold)))
    for row in rows:
        if float(row.get("pose_recall", -1.0)) < 0.95:
            continue
        if float(row.get("pose_weighted_recall", -1.0)) <= 0.99:
            continue
        if float(row.get("weighted_recall_lower_confidence_bound", -1.0)) <= 0.99:
            continue
        if not all(key in row for key in ("avg_candidate_count", "eval_pose_count", "tp", "fp", "fn")):
            raise ValueError(f"calibration row lacks count fields needed for useful cull: {summary_path}")
        candidate = dict(row)
        candidate["calibration_useful_cull"] = useful_cull(candidate)
        eligible.append(candidate)
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                float(row["calibration_useful_cull"]),
                float(row.get("pose_precision", 0.0)),
                float(row.get("pose_f1", 0.0)),
                -float(row.get("avg_pred_count", 0.0)),
                float(row.get("threshold", 0.0)),
            ),
        )
        status = "safe"
    else:
        # Keep a calibration-derived diagnostic threshold in the matrix even
        # when no row passes the registered safety gate.  Such a member can
        # never rank as a safe workpoint in the route decision.
        selected = dict(payload.get("calibration", {}).get("selected") or {})
        if "threshold" not in selected:
            selected["threshold"] = payload.get("frozenThreshold")
        status = "no_qualified_safety_workpoint"
    if selected.get("threshold") is None:
        raise ValueError(f"calibration source has no fallback threshold: {summary_path}")
    return {
        "threshold": float(selected["threshold"]),
        "status": status,
        "source": str(summary_path.resolve()),
        "sourceSha256": sha256_file(summary_path),
        "protocol": "calibration_ready_pre_test",
        "testEvaluationCount": 0,
        "testThresholdOverride": False,
        "selectionRule": "calibration only: pose recall >= 0.95, weighted recall > 0.99, one-sided LCB > 0.99; maximize useful cull, then precision, F1, minimize predicted count",
        "eligibleCalibrationRowCount": len(eligible),
        "selectedCalibrationRow": selected,
        "originalFrozenThreshold": payload.get("frozenThreshold"),
        "diagnosticWorkpoints": {
            "bestF1": best_f1,
            "highestPrecision": highest_precision,
            "originalFrozenThreshold": fixed_row,
        },
    }


def build_manifest(model_root: Path) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            path = model_root / experiment_name(variant, seed) / "calibration_ready_summary.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            members[f"{variant}:{seed}"] = select_from_summary(path)
    return {
        "schema": "neuralstreamweb3d-m4-v2-calibration-workpoints-v1",
        "split": "calibration_only_threshold_selection",
        "testRead": False,
        "poseRecallFloor": 0.95,
        "weightedRecallFloor": 0.99,
        "weightedRecallLowerConfidenceBoundFloor": 0.99,
        "members": members,
    }


def write_member_provenance(manifest: dict[str, Any], output: Path, model_root: Path) -> None:
    workpoint_root = output.parent / "workpoints"
    workpoint_root.mkdir(parents=True, exist_ok=True)
    for key, value in manifest["members"].items():
        variant, seed = key.split(":", 1)
        payload = dict(value)
        payload.update(
            {
                "schema": "neuralstreamweb3d-m4-v2-member-calibration-workpoint-v1",
                "member": key,
                "checkpoint": str((model_root / experiment_name(variant, int(seed)) / "best.pt").resolve()),
            }
        )
        (workpoint_root / f"{variant}_seed{seed}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def self_test() -> dict[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        row = {
            "threshold": 0.2,
            "pose_recall": 0.96,
            "pose_weighted_recall": 0.995,
            "weighted_recall_lower_confidence_bound": 0.992,
            "pose_precision": 0.4,
            "pose_f1": 0.55,
            "avg_candidate_count": 100.0,
            "eval_pose_count": 10,
            "tp": 90,
            "fp": 10,
            "fn": 4,
            "avg_pred_count": 100.0,
        }
        path.write_text(json.dumps({"protocol": "calibration_ready_pre_test", "testEvaluationCount": 0, "calibration": {"thresholdRows": [row]}}, indent=2), encoding="utf-8")
        result = select_from_summary(path)
        assert result["status"] == "safe"
        assert result["threshold"] == 0.2
    return {"status": "passed"}


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.output is None:
        raise ValueError("--output is required unless --self-test is used")
    payload = build_manifest(args.model_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_member_provenance(payload, args.output, args.model_root)
    print(json.dumps({"output": str(args.output), "memberCount": len(payload["members"]), "testRead": False}, indent=2))


if __name__ == "__main__":
    main()
