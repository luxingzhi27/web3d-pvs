#!/usr/bin/env python3
"""Repair the formal Full bootstrap protocol and summarize the five variants.

The original Full run used 2,000 bootstrap replicates during checkpoint-owned
calibration and validation, while the four ablations used 10,000.  This entry
replays every retained four-epoch Full snapshot on calibration and validation,
selects each seed's checkpoint under the registered rule, and then performs a
10,000-replicate paired seed/pose bootstrap against the existing ablations.
It never reads test and never modifies the source checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "neural_instance_culling" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from evaluate_pvs import (  # noqa: E402
    CALIBRATION_SCHEMA,
    DIAGNOSTIC_RECALIBRATION_SCHEMA,
)
from run_pvs import (  # noqa: E402
    EVALUATE,
    EVALUATION_SCHEMA,
    FORMAL_ABLATIONS,
    FORMAL_ABLATION_STAGE,
    FORMAL_EPOCHS,
    FORMAL_FULL_STAGE,
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    SCAN_CONFIGS,
    WEIGHTED_RECALL_FLOOR,
    _member_for_spec,
    _paths,
    _run_queue,
    _specs,
    preflight,
)
from summarize_pvs import (  # noqa: E402
    BOOTSTRAP_CLUSTER_UNIT,
    _paired_bootstrap,
    _summarize_rows,
)


SCHEMA = "pvs-v4-integrated-visibility-mainline-protocolfix-10000-v1"
SUMMARY_SCHEMA = "pvs-v4-integrated-visibility-mainline-paired-summary-10000-v1"
PROTOCOLFIX_DIR = "formal40_s02_protocolfix_10000"
RETAINED_EPOCHS = tuple(range(4, FORMAL_EPOCHS + 1, 4))
PAIR_METRICS = (
    "precision",
    "recall",
    "weightedRecall",
    "accuracy",
    "balancedAccuracy",
    "f1",
    "specificity",
    "usefulCull",
    "badCull",
    "avgPredCount",
    "predictedGlbBytes",
    "glbByteReduction",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _full_specs() -> list[tuple[str, str, Mapping[str, Any], int, int]]:
    return _specs(
        FORMAL_FULL_STAGE,
        ("full",),
        (SCAN_CONFIGS[2],),
        FORMAL_SEEDS,
        FORMAL_EPOCHS,
    )


def _ablation_specs() -> list[tuple[str, str, Mapping[str, Any], int, int]]:
    return _specs(
        FORMAL_ABLATION_STAGE,
        FORMAL_ABLATIONS,
        (SCAN_CONFIGS[2],),
        FORMAL_SEEDS,
        FORMAL_EPOCHS,
    )


def _snapshot_path(member: Path, epoch: int) -> Path:
    return member / f"checkpoint_epoch_{int(epoch):03d}.pt"


def _diagnostic_path(protocol_root: Path, member: Path, epoch: int) -> Path:
    return protocol_root / "checkpoint_reaudit" / member.name / f"epoch_{int(epoch):03d}.json"


def _selected_member_dir(protocol_root: Path, member: Path) -> Path:
    return protocol_root / "selected_full" / member.name


def _diagnostic_is_complete(
    path: Path,
    checkpoint: Path,
    *,
    epoch: int,
    seed: int,
    replicates: int,
) -> bool:
    try:
        payload = _load_json(path)
        seeds = payload.get("bootstrapSeeds")
        return bool(
            payload.get("schema") == DIAGNOSTIC_RECALIBRATION_SCHEMA
            and payload.get("testRead") is False
            and int(payload.get("epoch", -1)) == int(epoch)
            and int(payload.get("seed", -1)) == int(seed)
            and int(payload.get("bootstrapReplicates", 0)) >= int(replicates)
            and isinstance(seeds, Mapping)
            and int(seeds.get("calibration", -1)) == int(seed) + 50_000
            and int(seeds.get("validation", -1)) == int(seed) + 60_000
            and Path(str(payload.get("checkpoint"))).resolve() == checkpoint.resolve()
            and int(payload.get("calibrationPoseCount", -1)) == 659
            and int(payload.get("validationPoseCount", -1)) == 730
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def build_reaudit_command(
    data_root: Path,
    checkpoint: Path,
    output: Path,
    *,
    seed: int,
    replicates: int,
) -> list[str]:
    paths = _paths(data_root)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--dataset-dir",
        str(paths["dataset"]),
        "--runtime-meta",
        str(paths["runtime_meta"]),
        "--initial-geo-features",
        str(paths["geometry"]),
        "--glb-index",
        str(paths["glb_index"]),
        "--glb-root",
        str(paths["glb_root"]),
        "--relation-dir",
        str(paths["relation"]),
        "--output",
        str(output.resolve()),
        "--poses-per-batch",
        "2",
        "--seed",
        str(int(seed) + 50_000),
        "--device",
        "cuda",
        "--diagnostic-recalibrate",
        "--recalibration-bootstrap-replicates",
        str(int(replicates)),
        "--recalibration-validation-seed",
        str(int(seed) + 60_000),
    ]


def _validation_is_safe(payload: Mapping[str, Any]) -> bool:
    selected = payload.get("selectedSafe")
    validation = payload.get("validationAtSelectedThreshold")
    if not isinstance(selected, Mapping) or not isinstance(validation, Mapping):
        return False
    return bool(
        float(validation.get("aggregateWeightedRecall", -1.0))
        > WEIGHTED_RECALL_FLOOR
        and float(
            validation.get("aggregateWeightedRecallLowerConfidenceBound", -1.0)
        )
        > WEIGHTED_RECALL_FLOOR
    )


def _safe_rank(payload: Mapping[str, Any]) -> tuple[float, ...]:
    validation = payload["validationAtSelectedThreshold"]
    calibration = payload["selectedSafe"]
    return (
        float(validation.get("agg_balanced_accuracy") or 0.0),
        float(validation.get("agg_precision") or 0.0),
        float(validation.get("agg_accuracy") or 0.0),
        float(validation.get("agg_useful_cull") or 0.0),
        -float(validation.get("avg_pred_count") or 0.0),
        float(calibration.get("agg_useful_cull") or 0.0),
    )


def _diagnostic_rank(payload: Mapping[str, Any]) -> tuple[float, ...]:
    validation = payload.get("validationAtSelectedThreshold") or {}
    calibration = payload.get("diagnostic") or {}
    return (
        float(validation.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
        float(validation.get("aggregateWeightedRecall") or -1.0),
        float(calibration.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
        float(calibration.get("aggregateWeightedRecall") or -1.0),
        float(calibration.get("agg_balanced_accuracy") or 0.0),
        float(calibration.get("agg_useful_cull") or 0.0),
        float(calibration.get("agg_precision") or 0.0),
        -float(calibration.get("avg_pred_count") or 0.0),
    )


def select_checkpoint_reaudit(
    payloads: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
) -> tuple[Mapping[str, Any], bool]:
    if int(replicates) < 10_000:
        raise ValueError("formal Full re-audit requires at least 10000 replicates")
    if len(payloads) != len(RETAINED_EPOCHS):
        raise ValueError("Full re-audit requires every retained four-epoch checkpoint")
    epochs = {int(payload.get("epoch", -1)) for payload in payloads}
    if epochs != set(RETAINED_EPOCHS):
        raise ValueError("Full re-audit checkpoint epochs are incomplete")
    for payload in payloads:
        if (
            payload.get("schema") != DIAGNOSTIC_RECALIBRATION_SCHEMA
            or payload.get("testRead") is not False
            or int(payload.get("bootstrapReplicates", 0)) < int(replicates)
        ):
            raise ValueError("Full re-audit payload violates the formal protocol")
    safe = [payload for payload in payloads if _validation_is_safe(payload)]
    if safe:
        return max(safe, key=_safe_rank), True
    return max(payloads, key=_diagnostic_rank), False


def _corrected_calibration_summary(
    selected_payload: Mapping[str, Any],
    *,
    safe: bool,
    replicates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workpoint = (
        selected_payload.get("selectedSafe")
        if safe
        else selected_payload.get("diagnostic")
    )
    validation = selected_payload.get("validationAtSelectedThreshold")
    if not isinstance(workpoint, Mapping) or not isinstance(validation, Mapping):
        raise ValueError("selected re-audit checkpoint has no frozen workpoint")
    epoch = int(selected_payload["epoch"])
    best = {
        "epoch": epoch,
        "threshold": float(workpoint["threshold"]),
        "safe": bool(safe),
        "selection": dict(workpoint),
        "validationSelection": dict(validation),
        "validationSafetyPassed": bool(safe),
    }
    calibration = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-v4",
        "thresholdSource": "this-checkpoint-calibration-only-protocolfix-10000",
        "selectedSafe": selected_payload.get("selectedSafe"),
        "diagnostic": selected_payload.get("diagnostic"),
        "thresholdRows": selected_payload.get("calibrationThresholdRows", []),
        "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": WEIGHTED_RECALL_FLOOR,
        "selectionRule": selected_payload.get("selectionRule"),
        "bootstrapReplicates": int(replicates),
        "validationBootstrapReplicates": int(replicates),
        "validationSafetyRequiredForBestCheckpoint": True,
        "validationSafetyPassed": bool(safe),
        "testRead": False,
    }
    summary = {
        "schema": CALIBRATION_SCHEMA,
        "status": "safe" if safe else "no_qualified_safety_workpoint",
        "primarySafetyMetric": "aggregateWeightedRecall",
        "weightedRecallFloor": WEIGHTED_RECALL_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": WEIGHTED_RECALL_FLOOR,
        "validationSafetyRequiredForBestCheckpoint": True,
        "bestSafe": best if safe else None,
        "bestDiagnostic": best,
        "calibration": calibration,
        "validationAtFrozenThreshold": dict(validation),
        "sourceCheckpoint": str(selected_payload["checkpoint"]),
        "sourceCheckpointEpoch": epoch,
        "bootstrapReplicates": int(replicates),
        "protocolFix": {
            "reason": "Full originally used 2000 replicates while ablations used 10000",
            "retainedCheckpointEpochs": list(RETAINED_EPOCHS),
            "testRead": False,
        },
        "testRead": False,
    }
    return summary, best


def materialize_selected_checkpoint(
    selected_payload: Mapping[str, Any],
    output_dir: Path,
    *,
    safe: bool,
    replicates: int,
) -> dict[str, Any]:
    source_checkpoint = Path(str(selected_payload["checkpoint"])).resolve()
    source_member = source_checkpoint.parent
    source_model_meta = source_member / "model_meta.json"
    if not source_checkpoint.is_file() or not source_model_meta.is_file():
        raise FileNotFoundError("selected checkpoint or model_meta is missing")
    summary, best = _corrected_calibration_summary(
        selected_payload,
        safe=safe,
        replicates=replicates,
    )
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("testRead") is not False:
        raise ValueError("selected source checkpoint claims test was read")
    if int(checkpoint.get("epoch", -1)) != int(selected_payload["epoch"]):
        raise ValueError("selected source checkpoint epoch disagrees with re-audit")
    checkpoint["calibration"] = summary["calibration"]
    checkpoint["validationAtCalibration"] = summary["validationAtFrozenThreshold"]
    checkpoint["best"] = best
    checkpoint["testRead"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "selected_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    shutil.copyfile(source_model_meta, output_dir / "model_meta.json")
    _write_json(output_dir / "calibration_ready_summary.json", summary)
    selection = {
        "schema": SCHEMA,
        "seed": int(selected_payload["seed"]),
        "epoch": int(selected_payload["epoch"]),
        "threshold": float(best["threshold"]),
        "safe": bool(safe),
        "sourceCheckpoint": str(source_checkpoint),
        "derivedCheckpoint": str(checkpoint_path.resolve()),
        "bootstrapReplicates": int(replicates),
        "testRead": False,
    }
    _write_json(output_dir / "selection.json", selection)
    return selection


def build_selected_evaluate_command(
    data_root: Path,
    selected_dir: Path,
    output: Path,
    *,
    seed: int,
) -> list[str]:
    paths = _paths(data_root)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint",
        str((selected_dir / "selected_checkpoint.pt").resolve()),
        "--dataset-dir",
        str(paths["dataset"]),
        "--runtime-meta",
        str(paths["runtime_meta"]),
        "--initial-geo-features",
        str(paths["geometry"]),
        "--glb-index",
        str(paths["glb_index"]),
        "--glb-root",
        str(paths["glb_root"]),
        "--relation-dir",
        str(paths["relation"]),
        "--model-meta",
        str(selected_dir / "model_meta.json"),
        "--calibration",
        str(selected_dir / "calibration_ready_summary.json"),
        "--output",
        str(output.resolve()),
        "--split",
        "validation",
        "--poses-per-batch",
        "2",
        "--seed",
        str(int(seed)),
        "--device",
        "cuda",
        "--allow-unsafe-diagnostic",
        "--persist-ids",
    ]


def _evaluation_member(payload: Mapping[str, Any]) -> dict[str, Any]:
    if (
        payload.get("schema") != EVALUATION_SCHEMA
        or payload.get("split") != "validation"
        or payload.get("testRead") is not False
    ):
        raise ValueError("formal evaluation payload has the wrong schema or split")
    pose_indices = [int(value) for value in payload.get("poseIndices", [])]
    rows = payload.get("perPose")
    if not isinstance(rows, list) or len(rows) != 730 or len(pose_indices) != 730:
        raise ValueError("formal evaluation must contain all 730 validation poses")
    if [int(row["poseIndex"]) for row in rows] != pose_indices:
        raise ValueError("formal evaluation per-pose order is inconsistent")
    candidate_counts = [int(row["candidateCount"]) for row in rows]
    return {
        "payload": payload,
        "rows": rows,
        "poseIndices": pose_indices,
        "candidateReferenceCount": candidate_counts,
    }


def _source_calibration_replicates(evaluation: Mapping[str, Any]) -> int:
    path = Path(str(evaluation.get("calibrationSummary"))).resolve()
    calibration = _load_json(path)
    nested = calibration.get("calibration")
    if not isinstance(nested, Mapping):
        return 0
    return int(nested.get("bootstrapReplicates", 0))


def build_paired_summary(
    model_root: Path,
    benchmark_root: Path,
    protocol_root: Path,
    *,
    replicates: int,
) -> dict[str, Any]:
    if int(replicates) < 10_000:
        raise ValueError("formal paired summary requires at least 10000 replicates")
    members: dict[tuple[str, int], dict[str, Any]] = {}
    source_paths: dict[str, str] = {}
    full_selections: list[dict[str, Any]] = []
    for spec in _full_specs():
        member = _member_for_spec(model_root, spec)
        seed = int(spec[3])
        selected_dir = _selected_member_dir(protocol_root, member)
        selection = _load_json(selected_dir / "selection.json")
        if (
            selection.get("schema") != SCHEMA
            or selection.get("testRead") is not False
            or int(selection.get("seed", -1)) != seed
            or int(selection.get("bootstrapReplicates", 0)) < int(replicates)
        ):
            raise ValueError("corrected Full selection violates the formal protocol")
        full_selections.append(selection)
        path = selected_dir / "validation_evaluation.json"
        payload = _load_json(path)
        if _source_calibration_replicates(payload) < int(replicates):
            raise ValueError("corrected Full member has fewer than 10000 calibration replicates")
        members[("full", seed)] = _evaluation_member(payload)
        source_paths[f"full/seed{seed}"] = str(path.resolve())
    for spec in _ablation_specs():
        member = _member_for_spec(model_root, spec)
        variant = str(spec[1])
        seed = int(spec[3])
        path = benchmark_root / "members" / member.name / "validation_evaluation.json"
        payload = _load_json(path)
        if _source_calibration_replicates(payload) < int(replicates):
            raise ValueError(f"ablation member has fewer than 10000 calibration replicates: {member.name}")
        members[(variant, seed)] = _evaluation_member(payload)
        source_paths[f"{variant}/seed{seed}"] = str(path.resolve())
    expected = {
        (variant, seed) for variant in FORMAL_VARIANTS for seed in FORMAL_SEEDS
    }
    if set(members) != expected:
        raise ValueError("formal protocol-fix matrix is not the complete five-by-three design")
    reference = members[("full", FORMAL_SEEDS[0])]
    for member in members.values():
        if (
            member["poseIndices"] != reference["poseIndices"]
            or member["candidateReferenceCount"]
            != reference["candidateReferenceCount"]
        ):
            raise ValueError("formal members do not share validation poses and candidates")
    member_summaries: dict[str, Any] = {}
    for (variant, seed), member in sorted(members.items()):
        summary = _summarize_rows(
            member["rows"],
            lcb_replicates=int(replicates),
            lcb_seed=int(seed),
        )
        aggregate = summary["aggregate"]
        member_summaries[f"{variant}/seed{seed}"] = {
            "variant": variant,
            "seed": seed,
            "threshold": float(member["payload"]["threshold"]),
            "epoch": int(member["payload"]["epoch"]),
            "calibrationSafe": bool(
                (member["payload"].get("thresholdSource") or {}).get("safeWorkpoint")
            ),
            "validationSafe": bool(
                float(aggregate["weightedRecall"]) > WEIGHTED_RECALL_FLOOR
                and float(aggregate["weightedRecallLowerConfidenceBound"])
                > WEIGHTED_RECALL_FLOOR
            ),
            "poseMacro": summary["poseMacro"],
            "aggregate": aggregate,
            "testRead": False,
        }
    comparisons: dict[str, Any] = {}
    for index, ablation in enumerate(FORMAL_ABLATIONS):
        comparisons[f"full_minus_{ablation}"] = {
            "left": ablation,
            "right": "full",
            "deltaDefinition": "full minus ablation",
            "metrics": _paired_bootstrap(
                members,
                ablation,
                "full",
                list(FORMAL_SEEDS),
                PAIR_METRICS,
                int(replicates),
                20260823 + index,
                metric_names=PAIR_METRICS,
            ),
        }
    return {
        "schema": SUMMARY_SCHEMA,
        "split": "validation",
        "testRead": False,
        "variants": list(FORMAL_VARIANTS),
        "seeds": list(FORMAL_SEEDS),
        "poseCount": len(reference["poseIndices"]),
        "candidateSemantics": "stored native back-camera candidates; no GT union",
        "thresholdSource": "each checkpoint's own calibration split",
        "fullProtocolRepair": {
            "sourceReplicates": 2000,
            "correctedReplicates": int(replicates),
            "retainedCheckpointEpochs": list(RETAINED_EPOCHS),
            "checkpointWeightsRetrained": False,
            "testRead": False,
        },
        "fullSelections": full_selections,
        "members": member_summaries,
        "comparisons": comparisons,
        "bootstrap": {
            "replicates": int(replicates),
            "confidence": 0.95,
            "paired": True,
            "clusterUnit": BOOTSTRAP_CLUSTER_UNIT,
            "outerUnit": "seed cluster resampled with replacement",
            "innerUnit": "same validation pose IDs resampled within selected seed",
        },
        "sourceEvaluations": source_paths,
        "imageMetrics": {
            "status": "not_available",
            "reason": "protocol repair covers calibration and instance-level validation only",
            "testRead": False,
        },
    }


def _preflight_inputs(model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    snapshots: list[str] = []
    for spec in _full_specs():
        member = _member_for_spec(model_root, spec)
        for epoch in RETAINED_EPOCHS:
            checkpoint = _snapshot_path(member, epoch)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            snapshots.append(str(checkpoint.resolve()))
    ablations: list[str] = []
    for spec in _ablation_specs():
        member = _member_for_spec(model_root, spec)
        evaluation = benchmark_root / "members" / member.name / "validation_evaluation.json"
        if not evaluation.is_file():
            raise FileNotFoundError(evaluation)
        payload = _load_json(evaluation)
        if payload.get("testRead") is not False or _source_calibration_replicates(payload) < 10_000:
            raise ValueError(f"ablation evaluation violates formal protocol: {member.name}")
        ablations.append(str(evaluation.resolve()))
    return {
        "schema": SCHEMA,
        "fullSnapshotCount": len(snapshots),
        "fullSnapshotEpochs": list(RETAINED_EPOCHS),
        "ablationMemberCount": len(ablations),
        "seeds": list(FORMAL_SEEDS),
        "testRead": False,
    }


def run_protocol_fix(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.bootstrap_replicates) < 10_000:
        raise ValueError("formal protocol repair requires at least 10000 bootstrap replicates")
    preflight(args.data_root)
    protocol_root = args.benchmark_root.resolve() / PROTOCOLFIX_DIR
    contract = _preflight_inputs(args.model_root.resolve(), args.benchmark_root.resolve())
    _write_json(protocol_root / "preflight.json", contract)
    jobs: list[tuple[str, Sequence[str]]] = []
    for spec in _full_specs():
        member = _member_for_spec(args.model_root, spec)
        seed = int(spec[3])
        for epoch in RETAINED_EPOCHS:
            checkpoint = _snapshot_path(member, epoch)
            output = _diagnostic_path(protocol_root, member, epoch)
            if not _diagnostic_is_complete(
                output,
                checkpoint,
                epoch=epoch,
                seed=seed,
                replicates=int(args.bootstrap_replicates),
            ):
                output.parent.mkdir(parents=True, exist_ok=True)
                jobs.append(
                    (
                        f"reaudit_{member.name}_epoch{epoch:03d}",
                        build_reaudit_command(
                            args.data_root,
                            checkpoint,
                            output,
                            seed=seed,
                            replicates=int(args.bootstrap_replicates),
                        ),
                    )
                )
    if args.dry_run:
        return {"commands": [list(command) for _, command in jobs], **contract}
    if jobs:
        _run_queue(jobs, args.gpu_ids, protocol_root / "logs" / "checkpoint_reaudit")
    selections: list[dict[str, Any]] = []
    final_jobs: list[tuple[str, Sequence[str]]] = []
    for spec in _full_specs():
        member = _member_for_spec(args.model_root, spec)
        seed = int(spec[3])
        payloads = [
            _load_json(_diagnostic_path(protocol_root, member, epoch))
            for epoch in RETAINED_EPOCHS
        ]
        selected, safe = select_checkpoint_reaudit(
            payloads,
            replicates=int(args.bootstrap_replicates),
        )
        selected_dir = _selected_member_dir(protocol_root, member)
        selection = materialize_selected_checkpoint(
            selected,
            selected_dir,
            safe=safe,
            replicates=int(args.bootstrap_replicates),
        )
        selections.append(selection)
        output = selected_dir / "validation_evaluation.json"
        final_jobs.append(
            (
                f"evaluate_selected_{member.name}",
                build_selected_evaluate_command(
                    args.data_root,
                    selected_dir,
                    output,
                    seed=seed,
                ),
            )
        )
    _run_queue(final_jobs, args.gpu_ids, protocol_root / "logs" / "selected_validation")
    summary = build_paired_summary(
        args.model_root.resolve(),
        args.benchmark_root.resolve(),
        protocol_root,
        replicates=int(args.bootstrap_replicates),
    )
    if summary["fullSelections"] != selections:
        raise ValueError("paired summary Full selections disagree with the current re-audit")
    output = args.benchmark_root.resolve() / "formal40_s02_paired_bootstrap_10000_summary.json"
    _write_json(output, summary)
    full_summary = {
        "schema": SCHEMA,
        "split": "validation",
        "bootstrapReplicates": int(args.bootstrap_replicates),
        "selections": selections,
        "members": {
            key: value
            for key, value in summary["members"].items()
            if key.startswith("full/")
        },
        "pairedSummary": str(output.resolve()),
        "testRead": False,
    }
    _write_json(
        args.benchmark_root.resolve()
        / "formal40_s02_full_protocolfix_10000_summary.json",
        full_summary,
    )
    _write_json(
        protocol_root / "protocolfix_manifest.json",
        {
            **contract,
            "bootstrapReplicates": int(args.bootstrap_replicates),
            "fullSelections": selections,
            "pairedSummary": str(output.resolve()),
            "testRead": False,
        },
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "summarize"))
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/sda/rhyang/slm"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT
        / "neural_instance_culling/model/out/pvs_v4_integrated_visibility_mainline_v1_20260821",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT
        / "neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1_20260821",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if int(args.bootstrap_replicates) < 10_000:
        raise ValueError("formal protocol repair requires at least 10000 bootstrap replicates")
    protocol_root = args.benchmark_root.resolve() / PROTOCOLFIX_DIR
    if args.mode == "preflight":
        preflight(args.data_root)
        payload = _preflight_inputs(args.model_root.resolve(), args.benchmark_root.resolve())
    elif args.mode == "summarize":
        payload = build_paired_summary(
            args.model_root.resolve(),
            args.benchmark_root.resolve(),
            protocol_root,
            replicates=int(args.bootstrap_replicates),
        )
        _write_json(
            args.benchmark_root.resolve()
            / "formal40_s02_paired_bootstrap_10000_summary.json",
            payload,
        )
    else:
        payload = run_protocol_fix(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
