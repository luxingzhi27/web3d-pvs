#!/usr/bin/env python3
"""Re-audit historical hierarchical checkpoints under the v2 safety schema.

This is an explicit bridge for the retained 16-epoch confirmation artifacts.
It reruns the model on the complete calibration and validation splits, so old
pose-macro rows are never renamed in place or accepted as aggregate evidence.
It never reads the test split and refuses to write into an existing output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from common.runtime_meta import load_runtime_meta  # noqa: E402
from current_pvs_utils import evaluate_thresholds, threshold_grid  # noqa: E402
from evaluate_pvs_hierarchical_relation_survival_integrated import (  # noqa: E402
    _build_model,
    _load_checkpoint,
    _load_geometry,
    _runtime_features,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


OUT_SCHEMA = "pvs-hierarchical-relation-survival-integrated-calibration-reaudit-v1"
CALIBRATION_SCHEMA = "pvs-hierarchical-relation-survival-integrated-calibration-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sanitize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if key == "_pose_weighted_recall_values":
            result["poseMacroWeightedRecallValues"] = [float(item) for item in value]
        elif key == "_per_pose":
            continue
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    result["schema"] = CALIBRATION_SCHEMA
    result["aggregateWeightedRecall"] = float(result["aggregateWeightedRecall"])
    result["poseMacroWeightedRecall"] = float(result["poseMacroWeightedRecall"])
    result["aggregateWeightedRecallLowerConfidenceBound"] = float(
        result["aggregateWeightedRecallLowerConfidenceBound"]
    )
    result["poseMacroWeightedRecallLowerConfidenceBound"] = float(
        result["poseMacroWeightedRecallLowerConfidenceBound"]
    )
    return result


def _member_name(checkpoint: Path) -> str:
    return checkpoint.parent.name


def _run_one(
    checkpoint_path: Path,
    *,
    dataset: PoseCSRDataset,
    world_aabbs: np.ndarray,
    instance_to_glb: np.ndarray,
    runtime_meta_path: Path,
    geometry_path: Path,
    device: torch.device,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if bool(checkpoint.get("testRead", True)):
        raise ValueError(f"checkpoint claims test was read: {checkpoint_path}")
    model = _build_model(checkpoint, world_aabbs, instance_to_glb, device)
    geometry = _load_geometry(geometry_path, int(world_aabbs.shape[0]))
    runtime_features = _runtime_features(checkpoint, geometry, device)
    calibration_split = dataset.split("calibration")
    validation_split = dataset.split("validation")
    seed = int(checkpoint.get("args", {}).get("seed", 0))
    calibration_rows_raw = evaluate_thresholds(
        model,
        calibration_split,
        runtime_features,
        world_aabbs,
        device,
        poses_per_batch=2,
        max_steps=None,
        max_candidates_per_pose=0,
        seed=seed,
        thresholds=threshold_grid(),
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=int(bootstrap_replicates),
        collect_score_stats=True,
        collect_per_pose=False,
    )
    calibration_rows = [_sanitize_row(row) for row in calibration_rows_raw]
    selected = select_aggregate_weighted_cull_workpoint(
        calibration_rows,
        target_weighted_recall=0.99,
        minimum_lower_confidence_bound=0.99,
    )
    diagnostic = max(
        calibration_rows,
        key=lambda row: (
            float(row.get("aggregateWeightedRecall", -1.0)),
            float(row.get("agg_balanced_accuracy", 0.0)),
            float(row.get("agg_precision", 0.0)),
            -float(row.get("avg_pred_count", 0.0)),
        ),
    ) if calibration_rows else None
    chosen = selected or diagnostic
    validation_row = None
    if chosen is not None:
        validation_rows = evaluate_thresholds(
            model,
            validation_split,
            runtime_features,
            world_aabbs,
            device,
            poses_per_batch=2,
            max_steps=None,
            max_candidates_per_pose=0,
            seed=seed + 1,
            thresholds=np.asarray([float(chosen["threshold"])], dtype=np.float32),
            collect_pose_stats=True,
            allow_candidate_visible_union=False,
            bootstrap_replicates=int(bootstrap_replicates),
            collect_score_stats=False,
            collect_per_pose=False,
        )
        validation_row = _sanitize_row(validation_rows[0])
    summary = {
        "schema": CALIBRATION_SCHEMA,
        "protocol": "calibration_ready_pre_test",
        "testRead": False,
        "primarySafetyMetric": "aggregateWeightedRecall",
        "diagnosticSafetyMetric": "poseMacroWeightedRecall",
        "selectionRule": aggregate_weighted_cull_selection_rule(0.99, 0.99),
        "weightedRecallFloor": 0.99,
        "weightedRecallLowerConfidenceBoundFloor": 0.99,
        "bootstrapReplicates": int(bootstrap_replicates),
        "selected": selected,
        "diagnostic": diagnostic,
        "diagnosticThreshold": None if chosen is None else float(chosen["threshold"]),
        "thresholdRows": calibration_rows,
        "validationAtCalibration": validation_row,
        "status": "safe" if selected is not None else "no_qualified_safety_workpoint",
        "sourceCheckpoint": str(checkpoint_path.resolve()),
        "sourceCheckpointSha256": _sha256(checkpoint_path),
        "sourceCheckpointEpoch": int(checkpoint.get("epoch", -1)),
        "seed": seed,
        "splitPoseCounts": {
            "calibration": int(calibration_split.pose_indices.size),
            "validation": int(validation_split.pose_indices.size),
        },
        "candidateDigests": {
            "calibration": candidate_digest_for_pose_sequence(dataset, calibration_split.pose_indices),
            "validation": candidate_digest_for_pose_sequence(dataset, validation_split.pose_indices),
        },
        "runtimeMeta": {"path": str(runtime_meta_path.resolve()), "sha256": _sha256(runtime_meta_path)},
        "testRead": False,
    }
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bootstrap_replicates < 10000:
        raise ValueError("formal calibration re-audit requires at least 10000 bootstrap replicates")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = args.dataset_dir.resolve()
    runtime_meta_path = args.runtime_meta.resolve()
    geometry_path = args.initial_geo_features.resolve()
    world_aabbs, instance_to_glb, _runtime = load_runtime_meta(runtime_meta_path)
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(world_aabbs.shape[0]))
    if set(dataset.split_ids) & {"test"}:
        # The dataset may contain a test split, but this entry point must not
        # even instantiate it. The check only records the fixed split policy.
        pass
    device = torch.device(args.device)
    checkpoints = sorted(args.checkpoint_root.resolve().glob("*_seed*_e16/best.pt"))
    if not checkpoints:
        raise FileNotFoundError("no retained e16 best.pt checkpoints found")
    expected_members = {
        "R0_free_survival_seed20260801_e16",
        "R0_free_survival_seed20260802_e16",
        "R2_hierarchical_relation_seed20260801_e16",
        "R2_hierarchical_relation_seed20260802_e16",
    }
    actual_members = {checkpoint.parent.name for checkpoint in checkpoints}
    if actual_members != expected_members:
        raise ValueError(f"reaudit checkpoint set mismatch: expected {sorted(expected_members)}, got {sorted(actual_members)}")
    members: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        summary = _run_one(
            checkpoint,
            dataset=dataset,
            world_aabbs=world_aabbs,
            instance_to_glb=instance_to_glb,
            runtime_meta_path=runtime_meta_path,
            geometry_path=geometry_path,
            device=device,
            bootstrap_replicates=int(args.bootstrap_replicates),
        )
        member_dir = output_root / _member_name(checkpoint)
        _json(member_dir / "calibration_ready_summary.json", summary)
        members.append({
            "variant": str(summary.get("sourceCheckpoint", checkpoint).split("/")[-2]).split("_seed", 1)[0],
            "seed": int(summary["seed"]),
            "checkpoint": str(checkpoint.resolve()),
            "summary": str((member_dir / "calibration_ready_summary.json").resolve()),
            "status": summary["status"],
            "selectedThreshold": None if summary["selected"] is None else summary["selected"]["threshold"],
            "testRead": False,
        })
    manifest = {
        "schema": OUT_SCHEMA,
        "source": "retained 16-epoch confirmation checkpoints; rerun on complete calibration and validation",
        "checkpointRoot": str(args.checkpoint_root.resolve()),
        "datasetDir": str(dataset_dir),
        "runtimeMeta": str(runtime_meta_path),
        "geometry": str(geometry_path),
        "bootstrapReplicates": int(args.bootstrap_replicates),
        "splitPoseCounts": {
            "train": int(dataset.split("train").pose_indices.size),
            "calibration": int(dataset.split("calibration").pose_indices.size),
            "validation": int(dataset.split("validation").pose_indices.size),
        },
        "candidateDigests": {
            name: candidate_digest_for_pose_sequence(dataset, dataset.split(name).pose_indices)
            for name in ("train", "calibration", "validation")
        },
        "members": members,
        "testRead": False,
    }
    _json(output_root / "reaudit_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
