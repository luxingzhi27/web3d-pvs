#!/usr/bin/env python3
"""Strict M0 frozen-test protocol entry point.

The command has two deliberately separate phases:

``prepare``
    Read a selected checkpoint and its calibration/validation provenance.  It
    writes an immutable manifest containing one threshold and the hashes of
    the checkpoint and runtime feature table.  It never reads the test split.

``evaluate``
    Read that manifest and evaluate the complete native test split exactly
    once.  The command has no threshold-grid, candidate-cap, or GT-union
    option.  A fresh output directory is claimed atomically before inference;
    an existing directory is always rejected, including one from a failed
    attempt.

The actual scoring loop is shared with ``evaluate_unified_pvs_metrics`` so
that metric definitions stay aligned.  This entry point owns the formal
protocol guards and passes exactly one threshold with strict stored-candidate
semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = ROOT / "benchmark"
MODEL_DIR = ROOT / "model"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from evaluate_visual_utility_metrics import (  # noqa: E402
    evaluate_runner,
    load_glb_byte_costs,
    parse_int_list,
)
from model_runners import (  # noqa: E402
    load_runner,
    load_runtime_meta,
    selected_default_specs,
    select_device,
)
from pose_csr_dataset import PoseCSRDataset, PoseCSRSplit  # noqa: E402


MANIFEST_SCHEMA = "neuralstreamweb3d-frozen-test-manifest-v2"
MANIFEST_PROTOCOL = "fixed_validation_independent_calibration_frozen_test"
CLAIM_SCHEMA = "neuralstreamweb3d-frozen-test-claim-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose_index_digest(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected name=/path/to/file")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("name and path must both be non-empty")
    return name, Path(raw_path)


def resolve_recorded_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _split_names(dataset: PoseCSRDataset) -> tuple[str, str, str, str]:
    validation = "validation" if "validation" in dataset.split_ids else "val"
    required = ("train", validation, "calibration", "test")
    missing = [name for name in required if name not in dataset.split_ids]
    if missing:
        raise ValueError(
            "Formal M0 evaluation requires native train/validation/calibration/test split IDs; "
            f"missing {missing}. A legacy three-way split cannot be promoted to formal test."
        )
    return required


def expected_protocol_split(dataset: PoseCSRDataset) -> dict[str, Any]:
    train_name, validation_name, calibration_name, test_name = _split_names(dataset)
    train = dataset.split(train_name)
    validation = dataset.split(validation_name)
    calibration = dataset.split(calibration_name)
    test = dataset.split(test_name)
    return {
        "trainFitCount": int(train.pose_indices.size),
        "fixedValidationCount": int(validation.pose_indices.size),
        "calibrationCount": int(calibration.pose_indices.size),
        "frozenTestCount": int(test.pose_indices.size),
        "trainFitDigest": pose_index_digest(train.pose_indices),
        "fixedValidationDigest": pose_index_digest(validation.pose_indices),
        "calibrationDigest": pose_index_digest(calibration.pose_indices),
        "frozenTestDigest": pose_index_digest(test.pose_indices),
    }


def validate_protocol_split(protocol: dict[str, Any], dataset: PoseCSRDataset) -> dict[str, Any]:
    """Check that recorded four-way provenance describes this exact dataset."""
    if not isinstance(protocol, dict):
        raise ValueError("Frozen manifest protocolSplit must be an object.")
    required_keys = (
        "trainFitCount",
        "fixedValidationCount",
        "calibrationCount",
        "frozenTestCount",
        "trainFitDigest",
        "fixedValidationDigest",
        "calibrationDigest",
        "frozenTestDigest",
    )
    missing = [key for key in required_keys if key not in protocol]
    if missing:
        raise ValueError(f"Frozen manifest protocolSplit is missing {missing}.")
    actual = expected_protocol_split(dataset)
    mismatches = {
        key: {"recorded": protocol[key], "actual": actual[key]}
        for key in required_keys
        if str(protocol[key]) != str(actual[key])
    }
    if mismatches:
        raise ValueError(f"Frozen split provenance does not match dataset: {json.dumps(mismatches)}")
    validation_semantics = str(protocol.get("validationSemantics", "")).lower()
    calibration_semantics = str(protocol.get("calibrationSemantics", "")).lower()
    test_semantics = str(protocol.get("testSemantics", "")).lower()
    if "complete" not in validation_semantics or "no random" not in validation_semantics:
        raise ValueError("Frozen protocol does not prove that validation is the complete fixed set.")
    if "calibration" not in calibration_semantics or "not used" not in calibration_semantics:
        raise ValueError("Frozen protocol does not prove that calibration was held out from updates.")
    if "frozen" not in test_semantics or "once" not in test_semantics:
        raise ValueError("Frozen protocol does not describe a one-shot frozen test split.")

    names = _split_names(dataset)
    sets = [set(dataset.split(name).pose_indices.tolist()) for name in names]
    for left_index, left in enumerate(sets):
        for right_index in range(left_index + 1, len(sets)):
            overlap = left.intersection(sets[right_index])
            if overlap:
                raise ValueError(
                    f"Formal split overlap between {names[left_index]} and {names[right_index]}: "
                    f"{len(overlap)} pose indices."
                )
    return actual


def _threshold_close(left: Any, right: Any) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-7))


def _selection_key(row: dict[str, Any], target_weighted_recall: float) -> tuple[float, float, float, float] | None:
    if float(row.get("pose_weighted_recall", -1.0)) <= float(target_weighted_recall):
        return None
    return (
        float(row.get("pose_precision", 0.0)),
        float(row.get("pose_f1", 0.0)),
        float(row.get("pose_weighted_recall", 0.0)),
        -float(row.get("avg_pred_count", 0.0)),
    )


def audit_checkpoint_selection(
    checkpoint: dict[str, Any],
    history: list[dict[str, Any]],
    target_weighted_recall: float,
    expected_split: dict[str, Any],
) -> dict[str, Any]:
    """Prove checkpoint selection used complete validation, not test data."""
    protocol = checkpoint.get("protocolSplit")
    if not isinstance(protocol, dict):
        raise ValueError("Checkpoint has no protocolSplit provenance.")
    best = checkpoint.get("best")
    # Formal checkpoints keep the validation-selected checkpoint record
    # separate from the final calibration threshold. Older checkpoints used
    # one record for both decisions and remain supported for legacy auditing.
    checkpoint_selection = checkpoint.get("checkpointSelection") or best
    workpoints = checkpoint.get("workpoints")
    if not isinstance(best, dict) or not isinstance(checkpoint_selection, dict) or not isinstance(workpoints, dict):
        raise ValueError("Checkpoint lacks best/workpoints selection records.")
    calibration = workpoints.get("calibration")
    validation = workpoints.get("validationAtFrozenThreshold")
    selected = calibration.get("selected") if isinstance(calibration, dict) else None
    if not isinstance(selected, dict) or not isinstance(validation, dict):
        raise ValueError("Checkpoint lacks independent calibration and frozen validation records.")
    threshold = selected.get("threshold")
    if threshold is None or not _threshold_close(threshold, best.get("threshold")):
        raise ValueError("Checkpoint final best threshold does not equal the selected calibration threshold.")
    if not _threshold_close(threshold, validation.get("threshold")):
        raise ValueError("Validation record is not evaluated at the frozen calibration threshold.")
    if int(selected.get("eval_pose_count", -1)) != int(expected_split["calibrationCount"]):
        raise ValueError("Calibration threshold was not selected on the complete calibration split.")
    if int(validation.get("eval_pose_count", -1)) != int(expected_split["fixedValidationCount"]):
        raise ValueError("Checkpoint validation record is not from the complete validation split.")
    if int(best.get("eval_pose_count", -1)) != int(expected_split["fixedValidationCount"]):
        raise ValueError("Checkpoint final best record does not cover the complete validation split.")

    threshold_rows = calibration.get("thresholdRows") if isinstance(calibration, dict) else None
    if not isinstance(threshold_rows, list) or not threshold_rows:
        raise ValueError("Calibration provenance has no threshold rows.")
    selected_rows = [row for row in threshold_rows if _threshold_close(row.get("threshold"), threshold)]
    if len(selected_rows) != 1:
        raise ValueError("Calibration selected threshold is not uniquely present in calibration rows.")
    selected_row = selected_rows[0]
    bootstrap = calibration.get("bootstrap") if isinstance(calibration, dict) else None
    if not isinstance(bootstrap, dict) or int(bootstrap.get("replicates", 0)) <= 0:
        raise ValueError(
            "Calibration provenance has no view-cell bootstrap. The formal safety margin cannot be audited."
        )
    lcb_floor = bootstrap.get("lowerBoundFloor")
    if lcb_floor is None:
        raise ValueError("Calibration provenance has no one-sided lower-bound floor.")
    lcb = selected_row.get("weighted_recall_lower_confidence_bound")
    if lcb is None or float(lcb) <= float(lcb_floor):
        raise ValueError("Selected calibration threshold does not pass its recorded lower-confidence bound.")
    point_floor = float(checkpoint.get("args", {}).get("calibration_point_floor", 0.9925))
    if float(selected_row.get("pose_weighted_recall", -1.0)) < point_floor:
        raise ValueError("Selected calibration threshold fails the recorded point-estimate floor.")
    if float(selected_row.get("pose_weighted_recall", -1.0)) <= float(target_weighted_recall):
        raise ValueError("Selected calibration threshold does not strictly exceed weighted-recall target.")

    eval_rows = [row for row in history if isinstance(row, dict) and isinstance(row.get("val"), dict)]
    if not eval_rows:
        raise ValueError("Training history has no complete validation records for checkpoint selection audit.")
    for row in eval_rows:
        validation_row = row["val"]
        if int(validation_row.get("eval_pose_count", -1)) != int(expected_split["fixedValidationCount"]):
            raise ValueError(
                f"Validation at epoch {row.get('epoch')} was truncated; formal checkpoint selection is invalid."
            )
    safe_history = []
    for row in eval_rows:
        key = _selection_key(row["val"], target_weighted_recall)
        if key is not None:
            safe_history.append((key, int(row.get("epoch", -1)), row["val"]))
    if not safe_history:
        raise ValueError("No recorded validation checkpoint satisfies the weighted-recall safety rule.")
    selected_history = max(safe_history, key=lambda item: item[0])
    if int(checkpoint_selection.get("epoch", -1)) != selected_history[1]:
        raise ValueError(
            "best.pt is not the highest-precision safe checkpoint over the complete validation history: "
            f"recorded epoch={checkpoint_selection.get('epoch')}, expected epoch={selected_history[1]}."
        )
    if _selection_key(checkpoint_selection, target_weighted_recall) != selected_history[0]:
        raise ValueError("Checkpoint selection metrics do not match the selected validation history row.")
    if int(best.get("epoch", -1)) != int(checkpoint_selection.get("epoch", -1)):
        raise ValueError("Final calibration record changed the selected checkpoint epoch.")
    return {
        "selectedEpoch": int(checkpoint_selection["epoch"]),
        "threshold": float(threshold),
        "validationCount": int(expected_split["fixedValidationCount"]),
        "calibrationCount": int(expected_split["calibrationCount"]),
        "calibrationBootstrapReplicates": int(bootstrap["replicates"]),
        "calibrationLowerBound": float(lcb),
        "calibrationLowerBoundFloor": float(lcb_floor),
        "calibrationPointFloor": point_floor,
    }


def infer_runtime_features(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "instance_runtime_features_fp16.bin"


def prepare_one_model(
    name: str,
    checkpoint_path: Path,
    runtime_features_path: Path,
    history_path: Path,
    dataset_dir: Path,
    target_weighted_recall: float | None,
) -> tuple[str, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not runtime_features_path.is_file():
        raise FileNotFoundError(f"Runtime feature table does not exist: {runtime_features_path}")
    if not history_path.is_file():
        raise FileNotFoundError(f"Training history does not exist: {history_path}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    try:
        specs = selected_default_specs(name)
        canonical_name = next(iter(specs))
        model_kind = specs[canonical_name].get("kind")
    except KeyError:
        # Formal experiments use stable, experiment-specific names that are
        # intentionally not added to the global default model list.  The
        # checkpoint schema is sufficient to establish the learned runner
        # kind here; evaluate() will take the immutable paths from the
        # resulting manifest.
        canonical_name = str(name)
        model_kind = "directional_occlusion_proxy_encoder"
    if model_kind != "directional_occlusion_proxy_encoder":
        raise ValueError("prepare currently requires a learned directional PVS checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    protocol = checkpoint.get("protocolSplit")
    if not isinstance(protocol, dict):
        raise ValueError("Checkpoint has no protocolSplit; it cannot be used for formal M0 test.")
    expected = {
        key: protocol.get(key)
        for key in (
            "trainFitCount",
            "fixedValidationCount",
            "calibrationCount",
            "frozenTestCount",
            "trainFitDigest",
            "fixedValidationDigest",
            "calibrationDigest",
            "frozenTestDigest",
        )
    }
    if any(value is None for value in expected.values()):
        raise ValueError("Checkpoint protocolSplit is incomplete.")
    args = checkpoint.get("args", {})
    if int(args.get("max_candidates_per_pose", 0)) != 0:
        raise ValueError("Formal M0 manifest refuses a checkpoint trained/evaluated with candidate cropping.")
    if bool(args.get("allow_invalid_resource_semantics", False)):
        raise ValueError("Formal M0 manifest refuses a checkpoint with exploratory resource semantics enabled.")
    target = float(
        target_weighted_recall
        if target_weighted_recall is not None
        else args.get("target_weighted_recall", 0.99)
    )
    audit = audit_checkpoint_selection(
        checkpoint,
        json.loads(history_path.read_text(encoding="utf-8")),
        target,
        expected,
    )
    manifest_entry = {
        "threshold": audit["threshold"],
        "thresholdSource": "checkpoint.workpoints.calibration.selected; no test data",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpointSha256": sha256_file(checkpoint_path),
        "runtimeFeatures": str(runtime_features_path.resolve()),
        "runtimeFeaturesSha256": sha256_file(runtime_features_path),
        "protocolSplit": protocol,
        "selectionAudit": audit,
        "preTestEvaluationCount": 0,
    }
    return canonical_name, manifest_entry


def prepare_manifest(args: argparse.Namespace) -> None:
    pairs = [parse_name_path(value) for value in args.model_checkpoint]
    if not pairs:
        raise ValueError("At least one --model-checkpoint name=/path/to/best.pt is required.")
    entries: dict[str, dict[str, Any]] = {}
    protocol: dict[str, Any] | None = None
    for supplied_name, checkpoint_path in pairs:
        runtime_path = dict(parse_name_path(value) for value in args.runtime_features).get(supplied_name)
        checkpoint_path = checkpoint_path.resolve()
        runtime_path = (runtime_path or infer_runtime_features(checkpoint_path)).resolve()
        history_path = (args.history or (checkpoint_path.parent / "train_history.json")).resolve()
        canonical_name, entry = prepare_one_model(
            supplied_name,
            checkpoint_path,
            runtime_path,
            history_path,
            Path(args.dataset_dir).resolve(),
            args.target_weighted_recall,
        )
        if canonical_name in entries:
            raise ValueError(f"Duplicate canonical model name: {canonical_name}")
        if protocol is None:
            protocol = entry["protocolSplit"]
        elif entry["protocolSplit"] != protocol:
            raise ValueError("All models in one frozen manifest must use the same split provenance.")
        entries[canonical_name] = entry
    payload = {
        "schema": MANIFEST_SCHEMA,
        "protocol": MANIFEST_PROTOCOL,
        "created": utc_now(),
        "datasetDir": str(Path(args.dataset_dir).resolve()),
        "thresholdSelection": "independent calibration only; checkpoint selected on complete validation",
        "testPolicy": "frozen scalar threshold, complete unique test split, one shot",
        "testEvaluationCountBeforeThisRun": 0,
        "targetWeightedRecall": float(
            args.target_weighted_recall if args.target_weighted_recall is not None else 0.99
        ),
        "models": entries,
    }
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable frozen manifest: {output}")
    atomic_write_json(output, payload)
    print(json.dumps({"manifest": str(output), "models": sorted(entries)}, ensure_ascii=False, indent=2))


def load_manifest(path: Path, model_names: list[str]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA or payload.get("protocol") != MANIFEST_PROTOCOL:
        raise ValueError(
            f"{path} is not a strict {MANIFEST_SCHEMA} manifest; refusing legacy/test-derived thresholds."
        )
    if int(payload.get("testEvaluationCountBeforeThisRun", -1)) != 0:
        raise ValueError("Frozen manifest has already recorded a test evaluation before this run.")
    entries = payload.get("models")
    if not isinstance(entries, dict):
        raise ValueError("Frozen manifest has no models object.")
    missing = [name for name in model_names if name not in entries]
    if missing:
        raise ValueError(f"Frozen manifest has no entry for models: {missing}")
    for name in model_names:
        entry = entries[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Frozen manifest entry for {name} is not an object.")
        threshold = float(entry.get("threshold", -1.0))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Frozen threshold for {name} is outside [0, 1]: {threshold}")
        if int(entry.get("preTestEvaluationCount", -1)) != 0:
            raise ValueError(f"Frozen manifest entry for {name} is not pre-test.")
        if not isinstance(entry.get("protocolSplit"), dict):
            raise ValueError(f"Frozen manifest entry for {name} has no protocolSplit.")
        checkpoint_raw = entry.get("checkpoint")
        if checkpoint_raw:
            checkpoint = resolve_recorded_path(checkpoint_raw, path)
            recorded_hash = str(entry.get("checkpointSha256", ""))
            if not checkpoint.is_file() or not recorded_hash or sha256_file(checkpoint) != recorded_hash:
                raise ValueError(f"Checkpoint hash/path mismatch for frozen model {name}: {checkpoint}")
        runtime_raw = entry.get("runtimeFeatures")
        if runtime_raw:
            runtime_features = resolve_recorded_path(runtime_raw, path)
            recorded_hash = str(entry.get("runtimeFeaturesSha256", ""))
            if not runtime_features.is_file() or not recorded_hash or sha256_file(runtime_features) != recorded_hash:
                raise ValueError(f"Runtime feature hash/path mismatch for frozen model {name}: {runtime_features}")
    return payload


def preflight_candidates(dataset: PoseCSRDataset, split: PoseCSRSplit) -> dict[str, int]:
    """Check raw stored candidates without changing them or using GT repair."""
    visible_refs = 0
    candidate_refs = 0
    for pose_index in split.pose_indices.tolist():
        visible_ids, weights = dataset.visible_slice(int(pose_index))
        candidates = dataset.frustum_slice(int(pose_index))
        visible = np.unique(np.asarray(visible_ids, dtype=np.uint32))
        candidates = np.asarray(candidates, dtype=np.uint32)
        if candidates.size != np.unique(candidates).size:
            raise ValueError(f"Duplicate stored candidate IDs at test pose {pose_index}.")
        if candidates.size and int(candidates.max()) >= int(dataset.num_instances):
            raise ValueError(f"Candidate ID out of range at test pose {pose_index}.")
        if visible.size and int(visible.max()) >= int(dataset.num_instances):
            raise ValueError(f"Visible ID out of range at test pose {pose_index}.")
        missing = np.setdiff1d(visible, candidates, assume_unique=True)
        if missing.size:
            raise ValueError(
                f"Test pose {pose_index} has {missing.size} GT-visible IDs outside the stored candidate set; "
                "formal test refuses GT union or candidate repair."
            )
        weights = np.asarray(weights)
        if weights.size != np.asarray(visible_ids).size or not np.all(np.isfinite(weights)):
            raise ValueError(f"Invalid visible weights at test pose {pose_index}.")
        if np.any(weights < 0):
            raise ValueError(f"Negative visible weights at test pose {pose_index}.")
        visible_refs += int(visible.size)
        candidate_refs += int(candidates.size)
    return {
        "testPoseCount": int(split.pose_indices.size),
        "visibleReferenceCount": visible_refs,
        "candidateReferenceCount": candidate_refs,
    }


def claim_output_directory(output_dir: Path, manifest: Path, model_names: list[str]) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to rerun frozen test: output directory already exists: {output_dir}"
        ) from exc
    claim = {
        "schema": CLAIM_SCHEMA,
        "status": "started",
        "startedAt": utc_now(),
        "manifest": str(manifest.resolve()),
        "models": model_names,
        "testEvaluationCount": 0,
    }
    atomic_write_json(output_dir / "frozen_test_claim.json", claim)
    return claim


def update_claim(output_dir: Path, claim: dict[str, Any], status: str, **extra: Any) -> None:
    updated = dict(claim)
    updated.update(extra)
    updated["status"] = status
    updated["updatedAt"] = utc_now()
    atomic_write_json(output_dir / "frozen_test_claim.json", updated)


def write_summary_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines = [
        "# Frozen M0 Test",
        "",
        f"- Dataset: `{meta['datasetDir']}`",
        f"- Split: `{meta['split']}`",
        f"- Test poses: `{meta['evaluatedPoses']}`",
        f"- Threshold mode: `{meta['thresholdMode']}`",
        f"- Candidate semantics: `{meta['candidateSemanticsMode']}`",
        f"- Test evaluation count: `{meta['testEvaluationCount']}`",
        "",
        "The threshold is read from the pre-test calibration manifest. This report does not select a workpoint.",
        "",
        "| Model | Frozen threshold | Pose precision | Pose recall | Weighted recall | Useful cull | Bad cull | Avg candidate | Avg prediction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["results"].items():
        row = result["frozenResult"]
        lines.append(
            f"| {name} | {row['threshold']:.6f} | {row.get('pose_precision', 0.0):.6f} | "
            f"{row.get('pose_recall', 0.0):.6f} | {row.get('pose_weighted_recall', 0.0):.6f} | "
            f"{row.get('pose_useful_cull_candidate_ratio', 0.0):.6f} | "
            f"{row.get('pose_bad_cull_candidate_ratio', 0.0):.6f} | "
            f"{row.get('avg_candidate_count', 0.0):.2f} | {row.get('avg_pred_count', 0.0):.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_frozen(args: argparse.Namespace) -> None:
    requested_names = [part.strip() for part in args.models.split(",") if part.strip()]
    if not requested_names:
        raise ValueError("--models must contain at least one model name.")
    specs: dict[str, dict[str, str]] = {}
    for requested_name in requested_names:
        try:
            specs.update(selected_default_specs(requested_name))
        except KeyError:
            specs[requested_name] = {"kind": "directional_occlusion_proxy_encoder"}
    canonical_names = list(specs)
    manifest_path = Path(args.manifest).resolve()
    manifest = load_manifest(manifest_path, canonical_names)
    dataset_dir = Path(args.dataset_dir).resolve()
    recorded_dataset = manifest.get("datasetDir")
    if recorded_dataset and Path(recorded_dataset).resolve() != dataset_dir:
        raise ValueError(
            f"Dataset path differs from frozen manifest: recorded={recorded_dataset}, requested={dataset_dir}"
        )
    runtime_meta_path = Path(args.runtime_meta).resolve()
    device = select_device(args.device)
    world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(runtime_meta_path)
    checkpoint_entries = [manifest["models"][name].get("checkpoint") for name in canonical_names]
    checkpoint_entries = [value for value in checkpoint_entries if value]
    if checkpoint_entries:
        first_checkpoint = resolve_recorded_path(checkpoint_entries[0], manifest_path)
        checkpoint = torch.load(first_checkpoint, map_location="cpu")
        num_instances = int(checkpoint["config"]["numInstances"])
    else:
        num_instances = int(world_aabbs.shape[0])
    if int(world_aabbs.shape[0]) != num_instances:
        raise ValueError("Runtime metadata and frozen checkpoint disagree on instance count.")
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    protocol = manifest["models"][canonical_names[0]]["protocolSplit"]
    actual_split = validate_protocol_split(protocol, dataset)
    for name in canonical_names[1:]:
        if manifest["models"][name]["protocolSplit"] != protocol:
            raise ValueError(f"Model {name} does not use the same frozen split provenance.")
    test_split = dataset.split("test")
    candidate_stats = preflight_candidates(dataset, test_split)

    budgets = parse_int_list(args.budgets)
    output_dir = Path(args.output_dir).resolve()
    claim = claim_output_directory(output_dir, manifest_path, canonical_names)
    try:
        results: dict[str, Any] = {}
        for name in canonical_names:
            entry = manifest["models"][name]
            spec = dict(specs[name])
            if entry.get("checkpoint"):
                spec["checkpoint"] = str(resolve_recorded_path(entry["checkpoint"], manifest_path))
                runtime_features = entry.get("runtimeFeatures")
                if not runtime_features:
                    raise ValueError(f"Learned model {name} has no frozen runtime feature path.")
                spec["runtime_features"] = str(resolve_recorded_path(runtime_features, manifest_path))
                # The threshold is supplied by this manifest, never re-read from a legacy summary.
                spec["eval_summary"] = ""
            runner = load_runner(
                name,
                spec,
                runtime_meta_path,
                device,
                fallback_threshold=float(entry["threshold"]),
                dataset_dir=dataset_dir,
            )
            if int(runner.world_aabbs.shape[0]) != int(dataset.num_instances):
                raise ValueError(f"Runner {name} disagrees with dataset instance count.")
            glb_count = int(np.max(runner.instance_to_glb)) + 1 if runner.instance_to_glb.size else 0
            glb_cost = load_glb_byte_costs(
                args.glb_index,
                args.glb_root,
                glb_count,
                runner.instance_to_glb,
                allow_missing=False,
            )
            threshold = np.asarray([float(entry["threshold"])], dtype=np.float32)
            summary = evaluate_runner(
                runner,
                test_split,
                threshold,
                budgets,
                (),
                None,
                (),
                glb_cost,
                poses_per_batch=int(args.poses_per_batch),
                max_eval_poses=0,
                max_candidates_per_pose=0,
                seed=int(args.seed),
                target_utility_recall=float(args.target_utility_recall),
                target_weighted_recall=float(manifest.get("targetWeightedRecall", 0.99)),
                score_modes=("visibility-only",),
                aggregations=("max",),
                aggregation_top_k=1,
                sample_with_replacement=False,
                progress=True,
            )
            rows = summary.get("thresholdRows") or []
            if len(rows) != 1 or not _threshold_close(rows[0].get("threshold"), entry["threshold"]):
                raise RuntimeError(f"Frozen runner {name} did not produce exactly one requested threshold row.")
            if int(rows[0].get("eval_pose_count", -1)) != int(test_split.pose_indices.size):
                raise RuntimeError(
                    f"Frozen runner {name} evaluated {rows[0].get('eval_pose_count')} poses; "
                    f"expected all {test_split.pose_indices.size} unique test poses."
                )
            summary.pop("workpoints", None)
            summary.pop("best", None)
            summary.pop("diagnosticBestSafetyAdjustedCull", None)
            results[name] = {
                "threshold": float(entry["threshold"]),
                "frozenResult": rows[0],
                "runner": summary,
            }
        payload = {
            "meta": {
                "created": utc_now(),
                "manifest": str(manifest_path),
                "datasetDir": str(dataset_dir),
                "runtimeMeta": str(runtime_meta_path),
                "split": "test",
                "testPoseDigest": actual_split["frozenTestDigest"],
                "evaluatedPoses": int(test_split.pose_indices.size),
                "evalMode": "all_unique_test_poses_without_replacement",
                "thresholdMode": "frozen_single_value_no_selection",
                "candidateSemanticsMode": "stored_candidate_set_strict",
                "maxCandidatesPerPose": 0,
                "allowCandidateVisibleUnion": False,
                "testEvaluationCount": 1,
                "targetRecall": float(args.target_recall),
                "targetWeightedRecall": float(manifest.get("targetWeightedRecall", 0.99)),
                "candidateStats": candidate_stats,
                "device": str(device),
            },
            "protocol": MANIFEST_PROTOCOL,
            "results": results,
        }
        atomic_write_json(output_dir / "summary.json", payload)
        write_summary_markdown(output_dir / "summary.md", payload)
        update_claim(
            output_dir,
            claim,
            "completed",
            completedAt=utc_now(),
            testEvaluationCount=1,
            evaluatedPoses=int(test_split.pose_indices.size),
        )
        print(json.dumps({"outputDir": str(output_dir), "evaluatedPoses": int(test_split.pose_indices.size)}, ensure_ascii=False, indent=2))
    except BaseException as exc:
        update_claim(output_dir, claim, "failed", error=f"{type(exc).__name__}: {exc}", testEvaluationCount=0)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and execute the strict M0 frozen PVS test protocol.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a pre-test manifest from calibration/validation provenance.")
    prepare.add_argument("--model-checkpoint", action="append", required=True, help="name=/path/to/best.pt")
    prepare.add_argument("--runtime-features", action="append", default=[], help="name=/path/to/instance_runtime_features_fp16.bin")
    prepare.add_argument("--history", type=Path, default=None, help="Training history JSON; defaults beside checkpoint.")
    prepare.add_argument("--dataset-dir", required=True)
    prepare.add_argument("--target-weighted-recall", type=float, default=None)
    prepare.add_argument("--output", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Run the complete frozen test exactly once.")
    evaluate.add_argument("--models", required=True, help="Comma-separated retained model names.")
    evaluate.add_argument("--manifest", required=True, type=Path)
    evaluate.add_argument("--dataset-dir", required=True)
    evaluate.add_argument("--runtime-meta", required=True)
    evaluate.add_argument("--glb-index", required=True)
    evaluate.add_argument("--glb-root", required=True)
    evaluate.add_argument("--output-dir", required=True, type=Path)
    evaluate.add_argument("--poses-per-batch", type=int, default=4)
    evaluate.add_argument("--budgets", default="50,100,200,384")
    evaluate.add_argument("--target-recall", type=float, default=0.95)
    evaluate.add_argument("--target-utility-recall", type=float, default=0.98)
    evaluate.add_argument("--seed", type=int, default=20260801)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "prepare":
        prepare_manifest(args)
    elif args.command == "evaluate":
        if int(args.poses_per_batch) <= 0:
            parser.error("--poses-per-batch must be positive")
        evaluate_frozen(args)
    else:  # pragma: no cover - argparse enforces subcommands
        parser.error("a command is required")


if __name__ == "__main__":
    main()
