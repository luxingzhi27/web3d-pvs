#!/usr/bin/env python3
"""Freeze the IFCBench mainline, run formal test once, and export runtime assets.

The confirmation summary is the only source used to promote the fine-tune
family.  Promotion requires all three registered confirmation seeds to pass
the validation weighted-recall and lower-confidence-bound gates.  Otherwise
the original three-seed Full V4 family remains selected.

This orchestrator deliberately does not train, rescore calibration, or change
the evaluator/exporter.  ``preflight`` validates the inputs and target paths,
``dry-run`` prints the exact commands and log paths, and ``finalize`` executes
the three test jobs followed by one runtime export.  Every target is required
to be new so a rerun cannot overwrite a test result, sidecar, log, or asset.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVALUATE = ROOT / "neural_instance_culling" / "benchmark" / "evaluate_pvs.py"
EXPORT = ROOT / "neural_instance_culling" / "model" / "export_pvs.py"

SCENE = "ifcbench_fantasy_metropolis"
FULL_EXPERIMENT = "pvs_mainline_v4_ifcbench_fantasy_metropolis_v1"
FINETUNE_EXPERIMENT = "pvs_ifcbench_v4_calibration_finetune_v1"
FULL_EPOCHS = 40
FINETUNE_EPOCHS = 8
SEEDS = (20260801, 20260802, 20260803)
EXPECTED_SPLITS = {
    "train": 19647,
    "validation": 2712,
    "calibration": 2183,
    "test": 2710,
    "guard": 0,
}
WEIGHTED_RECALL_FLOOR = 0.99
TEST_BOOTSTRAP_REPLICATES = 10_000

CONFIRMATION_SCHEMA = "pvs-ifcbench-finetune-confirmation-summary-v1"
DECISION_SCHEMA = "pvs-ifcbench-final-freeze-decision-v1"
PREFLIGHT_SCHEMA = "pvs-ifcbench-finalize-preflight-v1"
PLAN_SCHEMA = "pvs-ifcbench-finalize-plan-v1"
FINALIZE_SCHEMA = "pvs-ifcbench-finalize-manifest-v1"
EXACT_CALIBRATION_SCHEMA = "pvs-ifcbench-v4-exact-calibration-v1"
RUNTIME_EXPORT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4"


@dataclass(frozen=True)
class FinalMember:
    """One checkpoint plus its frozen calibration and validation evidence."""

    family: str
    method_name: str
    seed: int
    checkpoint: Path
    calibration: Path
    calibration_kind: str
    model_meta: Path
    validation: Mapping[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing JSON file: {resolved}")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {resolved}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {resolved}")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a new manifest without replacing an existing user or run file."""
    resolved = Path(os.path.abspath(Path(path).expanduser()))
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing manifest: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary manifest already exists: {temporary}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _seed(value: Any, name: str = "seed") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if result != value and isinstance(value, float):
        raise ValueError(f"{name} must be an integer")
    return result


def paths(data_root: Path) -> dict[str, Path]:
    """Return the registered IFCBench data inputs used by both commands."""
    root = Path(data_root).expanduser().resolve()
    return {
        "dataset": root
        / "neural_instance_culling/dataset/out"
        / "pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "relation": root
        / "neural_instance_culling/dataset/out"
        / "ifcbench_fantasy_metropolis_v4_bounded_relation_csr_v1",
        "runtime_meta": root
        / "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json",
        "geometry": root
        / "neural_instance_culling/dataset/out"
        / "fixed_geometry_features_metropolis_v2/instance_geo_features_fp16.bin",
        "glb_index": root
        / "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json",
        "glb_root": root / "ifcbench_fantasy_metropolis_instanced_v2/assets",
    }


def _resolve_path(value: Any, *, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve()


def preflight_inputs(data_root: Path) -> dict[str, Any]:
    """Validate data/schema inputs without opening any split samples."""
    registered = paths(data_root)
    required_files = [
        registered["dataset"] / "dataset_meta.json",
        registered["relation"] / "relation_csr_meta.json",
        registered["runtime_meta"],
        registered["geometry"],
        registered["glb_index"],
        EVALUATE,
        EXPORT,
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if not registered["glb_root"].is_dir():
        missing.append(str(registered["glb_root"]))
    if missing:
        raise FileNotFoundError(f"missing IFCBench finalization input(s): {missing}")

    dataset_meta = _read_json(registered["dataset"] / "dataset_meta.json")
    split_counts = {
        str(name): _seed(count, f"splitCounts.{name}")
        for name, count in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"IFCBench split changed: expected={EXPECTED_SPLITS}, actual={split_counts}"
        )
    num_instances = _seed(dataset_meta.get("numInstances"), "dataset_meta.numInstances")
    files = dataset_meta.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("IFCBench dataset metadata has no files mapping")
    for name in (
        "poses",
        "mvp",
        "candidateIds",
        "visibleIds",
        "visibleWeights",
        "queryCenterWorld",
        "candidateCameraWorld",
        "viewcellRadiusM",
    ):
        declared = files.get(name)
        if not isinstance(declared, str) or not (registered["dataset"] / declared).is_file():
            raise ValueError(f"IFCBench dataset is missing {name}")
    if dataset_meta.get("candidateVisibleUnionAllowed") is not False:
        raise ValueError("IFCBench finalization requires candidateVisibleUnionAllowed=false")
    stats = dataset_meta.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError("IFCBench dataset metadata has no stats mapping")
    for name in (
        "candidateMissVisible",
        "candidateMissVisibleViewcells",
        "visibleSubsetFailures",
        "candidateVisibleUnionAdded",
    ):
        if _seed(stats.get(name), f"dataset_meta.stats.{name}") != 0:
            raise ValueError(f"IFCBench dataset violates candidate/GT contract: {name}")

    runtime_meta = _read_json(registered["runtime_meta"])
    if _seed(runtime_meta.get("instanceCount"), "runtime_meta.instanceCount") != num_instances:
        raise ValueError("runtime metadata instanceCount disagrees with dataset metadata")
    records = runtime_meta.get("componentRecords")
    if not isinstance(records, list) or len(records) != num_instances:
        raise ValueError("runtime metadata componentRecords disagree with dataset metadata")
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"runtime metadata component record {index} is invalid")
        bounds = record.get("bounds")
        if not isinstance(bounds, Mapping):
            raise ValueError(f"runtime metadata component record {index} has invalid bounds")
        center = bounds.get("center")
        size = bounds.get("size")
        if not isinstance(center, list) or len(center) != 3:
            raise ValueError(f"runtime metadata component record {index} has invalid center")
        if not isinstance(size, list) or len(size) != 3:
            raise ValueError(f"runtime metadata component record {index} has invalid size")
        for value in (*center, *size):
            _finite_float(value, f"runtime metadata component record {index} bounds")
        global_glb_id = record.get("globalGlbId")
        if not isinstance(global_glb_id, (int, float)) or isinstance(global_glb_id, bool):
            raise ValueError(f"runtime metadata component record {index} has no globalGlbId")
        if _seed(global_glb_id, f"runtime metadata component record {index}.globalGlbId") < 0:
            raise ValueError(f"runtime metadata component record {index} has invalid globalGlbId")

    relation_meta = _read_json(registered["relation"] / "relation_csr_meta.json")
    if (
        relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3"
        or relation_meta.get("trainOnly") is not True
    ):
        raise ValueError("IFCBench relation artifact has the wrong schema")
    if _seed(relation_meta.get("numInstances"), "relation_meta.numInstances") != num_instances:
        raise ValueError("relation metadata numInstances disagrees with dataset metadata")
    expected_geometry_bytes = num_instances * 96 * 2
    if registered["geometry"].stat().st_size != expected_geometry_bytes:
        raise ValueError(
            "fixed geometry feature table has the wrong size: "
            f"expected={expected_geometry_bytes}, actual={registered['geometry'].stat().st_size}"
        )
    return {
        "schema": PREFLIGHT_SCHEMA,
        "scene": SCENE,
        "dataRoot": str(Path(data_root).expanduser().resolve()),
        "splitCounts": split_counts,
        "numInstances": num_instances,
        "paths": {name: str(path) for name, path in registered.items()},
        "testRead": False,
    }


def _metric_containers(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = [value]
    for key in ("aggregate", "metrics", "validation", "selected", "selection"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and nested not in containers:
            containers.append(nested)
    return containers


def _metric(value: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for container in _metric_containers(value):
        for name in names:
            if name in container and container[name] is not None:
                return _finite_float(container[name], name)
    return None


WR_NAMES = (
    "aggregateWeightedRecall",
    "weightedRecall",
    "agg_weighted_recall",
    "aggregate_weighted_recall",
)
LCB_NAMES = (
    "aggregateWeightedRecallLowerConfidenceBound",
    "weightedRecallLowerConfidenceBound",
    "weighted_recall_lower_confidence_bound",
    "aggregate_weighted_recall_lower_confidence_bound",
)
USEFUL_NAMES = ("usefulCull", "agg_useful_cull", "useful_cull")
BALANCED_NAMES = ("balancedAccuracy", "agg_balanced_accuracy", "balanced_accuracy")
SPECIFICITY_NAMES = ("specificity", "agg_specificity")
PRECISION_NAMES = ("precision", "agg_precision")
PREDICTED_NAMES = ("avgPredCount", "avg_pred_count")


def validation_safety(validation: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict validation safety gate and its two source values."""
    weighted_recall = _metric(validation, WR_NAMES)
    lower_bound = _metric(validation, LCB_NAMES)
    safe = (
        weighted_recall is not None
        and lower_bound is not None
        and weighted_recall > WEIGHTED_RECALL_FLOOR
        and lower_bound > WEIGHTED_RECALL_FLOOR
    )
    return {
        "weightedRecall": weighted_recall,
        "weightedRecallLowerConfidenceBound": lower_bound,
        "safe": bool(safe),
    }


def load_confirmation_summary(path: Path) -> dict[str, Any]:
    """Load and structurally validate the test-free confirmation summary."""
    resolved = Path(path).expanduser().resolve()
    payload = _read_json(resolved)
    if payload.get("schema") != CONFIRMATION_SCHEMA:
        raise ValueError(
            f"confirmation summary schema is invalid: expected={CONFIRMATION_SCHEMA!r}"
        )
    if payload.get("testRead") is not False:
        raise ValueError("confirmation summary must declare testRead=false")
    selected_config = payload.get("selectedConfig")
    if not isinstance(selected_config, str) or not selected_config:
        raise ValueError("confirmation summary has no selectedConfig")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError(f"confirmation summary must contain exactly {len(SEEDS)} rows")
    by_seed: dict[int, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            raise ValueError(f"confirmation row {index} is not an object")
        seed = _seed(raw.get("seed"), f"confirmation row {index}.seed")
        if seed in by_seed:
            raise ValueError(f"confirmation summary contains duplicate seed {seed}")
        if seed not in SEEDS:
            raise ValueError(f"confirmation summary contains unregistered seed {seed}")
        validation = raw.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError(f"confirmation row {seed} has no validation metrics")
        gate = validation_safety(validation)
        if gate["weightedRecall"] is None or gate["weightedRecallLowerConfidenceBound"] is None:
            raise ValueError(f"confirmation row {seed} has no validation WR/LCB")
        by_seed[seed] = {
            **raw,
            "seed": seed,
            "validationSafety": gate,
        }
    if set(by_seed) != set(SEEDS):
        raise ValueError(
            f"confirmation seeds changed: expected={list(SEEDS)}, actual={sorted(by_seed)}"
        )
    return {
        **payload,
        "confirmationSummaryPath": str(resolved),
        "rows": [by_seed[seed] for seed in SEEDS],
    }


def freeze_decision(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the all-three-seed confirmation gate without aggregate shortcuts."""
    if summary.get("testRead") is not False:
        raise ValueError("confirmation summary must declare testRead=false")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError("confirmation summary rows are incomplete")
    decisions: list[dict[str, Any]] = []
    seen_seeds: set[int] = set()
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise ValueError("confirmation summary contains an invalid row")
        seed = _seed(raw.get("seed"))
        if seed in seen_seeds or seed not in SEEDS:
            raise ValueError(f"confirmation summary contains invalid or duplicate seed {seed}")
        seen_seeds.add(seed)
        validation = raw.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError(f"confirmation row {seed} has no validation metrics")
        gate = validation_safety(validation)
        decisions.append(
            {
                "seed": seed,
                "weightedRecall": gate["weightedRecall"],
                "weightedRecallLowerConfidenceBound": gate[
                    "weightedRecallLowerConfidenceBound"
                ],
                "validationSafe": gate["safe"],
            }
        )
    if seen_seeds != set(SEEDS):
        raise ValueError(
            f"confirmation seeds changed: expected={list(SEEDS)}, actual={sorted(seen_seeds)}"
        )
    decisions.sort(key=lambda row: int(row["seed"]))
    promoted = len(decisions) == len(SEEDS) and all(
        bool(row["validationSafe"]) for row in decisions
    )
    selected_family = "finetune" if promoted else "full_v4"
    return {
        "schema": DECISION_SCHEMA,
        "scene": SCENE,
        "confirmationSummary": str(summary.get("confirmationSummaryPath", "")),
        "selectedFamily": selected_family,
        "promotedFineTune": promoted,
        "selectedConfig": summary.get("selectedConfig"),
        "rule": (
            "promote fine-tune only when all three confirmation validation WR and "
            "LCB are strictly greater than 0.99"
        ),
        "fallback": "original three-seed Full V4" if not promoted else None,
        "confirmationRows": decisions,
        "testRead": False,
    }


def _default_exact_calibration(
    benchmark_root: Path, selected_config: str, seed: int
) -> Path:
    return (
        Path(benchmark_root).expanduser().resolve()
        / "confirm"
        / selected_config
        / f"seed{seed}"
        / "exact_calibration.json"
    )


def _validate_calibration(member: FinalMember) -> dict[str, Any]:
    payload = _read_json(member.calibration)
    if payload.get("testRead") is not False:
        raise ValueError(f"calibration is not test-free: {member.calibration}")
    if payload.get("testEvaluationCount", 0) not in (0, None):
        raise ValueError(f"calibration already read test: {member.calibration}")
    if member.calibration_kind == "exact":
        if payload.get("schema") != EXACT_CALIBRATION_SCHEMA:
            raise ValueError(f"fine-tune calibration schema is invalid: {member.calibration}")
        if payload.get("split") != "calibration" or payload.get("status") != "safe":
            raise ValueError(f"fine-tune exact calibration is not safe: {member.calibration}")
        declared = payload.get("checkpoint")
        if declared is None or Path(str(declared)).expanduser().resolve() != member.checkpoint:
            raise ValueError("fine-tune exact calibration belongs to a different checkpoint")
        selection = payload.get("selection")
        selected = payload.get("selected")
        if not isinstance(selection, Mapping) or not isinstance(selected, Mapping):
            raise ValueError("fine-tune exact calibration has no frozen selection")
        threshold = _finite_float(selection.get("threshold"), "exact calibration threshold")
        selected_threshold = _finite_float(
            selected.get("threshold"), "exact calibration selected threshold"
        )
        if not math.isclose(threshold, selected_threshold, rel_tol=0.0, abs_tol=1e-7):
            raise ValueError("fine-tune exact calibration threshold fields disagree")
        gate = validation_safety(selected)
        if not gate["safe"]:
            raise ValueError("fine-tune exact calibration fails the weighted-recall safety gate")
    else:
        if payload.get("status") != "safe" or not isinstance(payload.get("bestSafe"), Mapping):
            raise ValueError(f"Full V4 member calibration is not safe: {member.calibration}")
    return payload


def _member_method_name(family: str, selected_config: str | None = None) -> str:
    if family == "full_v4":
        return "full_v4"
    if family == "finetune" and selected_config:
        if any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for character in selected_config
        ):
            raise ValueError(f"selectedConfig contains unsafe filename characters: {selected_config!r}")
        return f"finetune_{selected_config}"
    raise ValueError(f"cannot construct method name for family={family!r}")


def _fine_tune_members(
    summary: Mapping[str, Any],
    *,
    model_root: Path,
    benchmark_root: Path,
) -> list[FinalMember]:
    selected_config = summary.get("selectedConfig")
    if not isinstance(selected_config, str) or not selected_config:
        raise ValueError("fine-tune confirmation has no selectedConfig")
    method_name = _member_method_name("finetune", selected_config)
    summary_path = Path(str(summary.get("confirmationSummaryPath", "."))).expanduser().resolve()
    result: list[FinalMember] = []
    for raw in summary["rows"]:
        seed = _seed(raw["seed"])
        fallback_checkpoint = (
            Path(model_root).expanduser().resolve()
            / "confirm"
            / selected_config
            / f"seed{seed}_e{FINETUNE_EPOCHS}"
            / "last.pt"
        )
        checkpoint = (
            _resolve_path(
                raw["checkpoint"],
                base=summary_path.parent,
                name="confirmation checkpoint",
            )
            if raw.get("checkpoint") is not None
            else fallback_checkpoint
        )
        exact_value = raw.get("exactCalibration", raw.get("calibrationPath"))
        calibration = (
            _resolve_path(
                exact_value,
                base=summary_path.parent,
                name="exact calibration",
            )
            if exact_value is not None
            else _default_exact_calibration(benchmark_root, selected_config, seed)
        )
        model_meta_value = raw.get("modelMeta")
        model_meta = (
            _resolve_path(
                model_meta_value,
                base=summary_path.parent,
                name="fine-tune model metadata",
            )
            if model_meta_value is not None
            else checkpoint.parent / "model_meta.json"
        )
        validation = raw.get("validation")
        if not isinstance(validation, Mapping):
            raise ValueError(f"fine-tune confirmation row {seed} has no validation metrics")
        result.append(
            FinalMember(
                family="finetune",
                method_name=method_name,
                seed=seed,
                checkpoint=checkpoint,
                calibration=calibration,
                calibration_kind="exact",
                model_meta=model_meta,
                validation=validation,
            )
        )
    return result


def _full_members(
    *,
    model_root: Path,
    benchmark_root: Path,
) -> list[FinalMember]:
    model_root = Path(model_root).expanduser().resolve()
    benchmark_root = Path(benchmark_root).expanduser().resolve()
    result: list[FinalMember] = []
    for seed in SEEDS:
        member_dir = model_root / f"full_seed{seed}_e{FULL_EPOCHS}"
        validation_path = benchmark_root / "members" / f"seed{seed}_validation.json"
        validation_payload = _read_json(validation_path)
        if validation_payload.get("split") != "validation" or validation_payload.get("testRead") is not False:
            raise ValueError(f"Full V4 validation payload is not test-free: {validation_path}")
        validation = validation_payload.get("aggregate", validation_payload.get("metrics"))
        if not isinstance(validation, Mapping):
            raise ValueError(f"Full V4 validation payload has no aggregate metrics: {validation_path}")
        result.append(
            FinalMember(
                family="full_v4",
                method_name="full_v4",
                seed=seed,
                checkpoint=member_dir / "best_safe.pt",
                calibration=member_dir / "calibration_ready_summary.json",
                calibration_kind="member",
                model_meta=member_dir / "model_meta.json",
                validation=validation,
            )
        )
    return result


def build_members(
    decision: Mapping[str, Any],
    confirmation_summary: Mapping[str, Any],
    *,
    full_model_root: Path,
    full_benchmark_root: Path,
    finetune_model_root: Path,
    finetune_benchmark_root: Path,
) -> list[FinalMember]:
    if decision.get("selectedFamily") == "finetune":
        members = _fine_tune_members(
            confirmation_summary,
            model_root=finetune_model_root,
            benchmark_root=finetune_benchmark_root,
        )
    elif decision.get("selectedFamily") == "full_v4":
        members = _full_members(
            model_root=full_model_root,
            benchmark_root=full_benchmark_root,
        )
    else:
        raise ValueError(f"unknown frozen family: {decision.get('selectedFamily')!r}")
    if [member.seed for member in members] != list(SEEDS):
        raise ValueError("frozen family does not contain the registered three seeds in order")
    for member in members:
        if not member.checkpoint.is_file():
            raise FileNotFoundError(f"missing frozen checkpoint: {member.checkpoint}")
        if not member.model_meta.is_file():
            raise FileNotFoundError(f"missing frozen model metadata: {member.model_meta}")
        _validate_calibration(member)
    return members


def _cull_metric(validation: Mapping[str, Any]) -> float:
    useful = _metric(validation, USEFUL_NAMES)
    if useful is not None:
        return useful
    containers = _metric_containers(validation)
    counts: list[float] = []
    for name in ("tp", "fp", "fn", "tn"):
        value = None
        for container in containers:
            if name in container:
                value = _finite_float(container[name], name)
                break
        if value is None:
            raise ValueError("validation metrics have no useful cull or confusion counts")
        counts.append(value)
    total = sum(counts)
    return 0.0 if total <= 0.0 else counts[3] / total


def _rank_value(validation: Mapping[str, Any], names: Sequence[str], default: float) -> float:
    value = _metric(validation, names)
    return default if value is None else value


def validation_rank_key(member: FinalMember) -> tuple[float, ...]:
    """Rank an already-safe validation pool with useful cull first."""
    gate = validation_safety(member.validation)
    if not gate["safe"]:
        raise ValueError(f"cannot rank an unsafe validation member: seed {member.seed}")
    lower = gate["weightedRecallLowerConfidenceBound"]
    assert lower is not None
    return (
        _cull_metric(member.validation),
        _rank_value(member.validation, BALANCED_NAMES, -math.inf),
        _rank_value(member.validation, SPECIFICITY_NAMES, -math.inf),
        _rank_value(member.validation, PRECISION_NAMES, -math.inf),
        lower,
        -_rank_value(member.validation, PREDICTED_NAMES, math.inf),
        -float(member.seed),
    )


def select_runtime_member(members: Sequence[FinalMember]) -> tuple[FinalMember, list[FinalMember]]:
    """Select from the validation-safe pool, prioritizing useful cull."""
    safe_pool = [member for member in members if validation_safety(member.validation)["safe"]]
    if not safe_pool:
        raise ValueError("frozen family has no validation-safe member for runtime export")
    selected = max(safe_pool, key=validation_rank_key)
    return selected, safe_pool


def _member_record(member: FinalMember) -> dict[str, Any]:
    gate = validation_safety(member.validation)
    return {
        "family": member.family,
        "method": member.method_name,
        "seed": member.seed,
        "checkpoint": str(member.checkpoint),
        "calibration": str(member.calibration),
        "calibrationKind": member.calibration_kind,
        "modelMeta": str(member.model_meta),
        "validationSafety": gate,
        "validationUsefulCull": _cull_metric(member.validation),
    }


def _new_target(path: Path) -> str:
    return os.path.abspath(str(Path(path).expanduser()))


def _target_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _assert_new_targets(targets: Sequence[Path]) -> None:
    seen: set[Path] = set()
    existing: list[str] = []
    duplicate: list[str] = []
    for raw in targets:
        path = Path(os.path.abspath(str(Path(raw).expanduser())))
        if path in seen:
            duplicate.append(str(path))
        seen.add(path)
        if _target_exists(path):
            existing.append(str(path))
    if duplicate:
        raise ValueError(f"finalization targets overlap: {duplicate}")
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing IFCBench test/asset/log target(s): "
            f"{existing}"
        )


def _paths_overlap(left: Path, right: Path) -> bool:
    left_path = Path(os.path.abspath(str(Path(left).expanduser())))
    right_path = Path(os.path.abspath(str(Path(right).expanduser())))
    return (
        left_path == right_path
        or left_path in right_path.parents
        or right_path in left_path.parents
    )


def build_evaluate_command(
    member: FinalMember,
    data_root: Path,
    *,
    output: Path,
    sidecar_dir: Path,
    device: str = "cuda",
    poses_per_batch: int = 2,
) -> list[str]:
    registered = paths(data_root)
    if poses_per_batch <= 0:
        raise ValueError("poses_per_batch must be positive")
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint",
        str(member.checkpoint),
        "--dataset-dir",
        str(registered["dataset"]),
        "--runtime-meta",
        str(registered["runtime_meta"]),
        "--initial-geo-features",
        str(registered["geometry"]),
        "--glb-index",
        str(registered["glb_index"]),
        "--glb-root",
        str(registered["glb_root"]),
        "--relation-dir",
        str(registered["relation"]),
        "--model-meta",
        str(member.model_meta),
        "--calibration",
        str(member.calibration),
        "--output",
        str(Path(output).expanduser().resolve()),
        "--split",
        "test",
        "--poses-per-batch",
        str(poses_per_batch),
        "--seed",
        str(member.seed),
        "--device",
        device,
        "--test-bootstrap-replicates",
        str(TEST_BOOTSTRAP_REPLICATES),
        "--sidecar-dir",
        str(Path(sidecar_dir).expanduser().resolve()),
        "--scene-name",
        SCENE,
        "--method-name",
        f"{member.method_name}_seed{member.seed}",
    ]


def build_export_command(
    member: FinalMember,
    data_root: Path,
    *,
    output_dir: Path,
) -> list[str]:
    registered = paths(data_root)
    command = [
        sys.executable,
        str(EXPORT),
        "--checkpoint",
        str(member.checkpoint),
        "--runtime-meta",
        str(registered["runtime_meta"]),
        "--output-dir",
        str(Path(output_dir).expanduser().resolve()),
    ]
    # Native Full V4 reads its member calibration from the checkpoint.  A
    # warm-start checkpoint has no native safe summary, so its exact external
    # calibration must be passed explicitly to export_pvs.py.
    if member.calibration_kind == "exact":
        command.extend(["--calibration", str(member.calibration)])
    return command


def _default_manifest(paper_results_root: Path) -> Path:
    return (
        Path(paper_results_root).expanduser().resolve()
        / "finalize"
        / SCENE
        / "finalize_manifest.json"
    )


def _default_runtime_dir(member: FinalMember, model_root: Path) -> Path:
    return (
        Path(model_root).expanduser().resolve()
        / f"runtime_final_{member.method_name}_seed{member.seed}_v1"
    )


def prepare_plan(
    *,
    data_root: Path,
    confirmation_summary_path: Path,
    full_model_root: Path,
    full_benchmark_root: Path,
    finetune_model_root: Path,
    finetune_benchmark_root: Path,
    paper_results_root: Path,
    runtime_output_dir: Path | None = None,
    manifest_path: Path | None = None,
    logs_dir: Path | None = None,
    device: str = "cuda",
    poses_per_batch: int = 2,
    gpu_ids: Sequence[int] = (0, 1, 2),
) -> dict[str, Any]:
    contract = preflight_inputs(data_root)
    summary = load_confirmation_summary(confirmation_summary_path)
    decision = freeze_decision(summary)
    members = build_members(
        decision,
        summary,
        full_model_root=full_model_root,
        full_benchmark_root=full_benchmark_root,
        finetune_model_root=finetune_model_root,
        finetune_benchmark_root=finetune_benchmark_root,
    )
    runtime_member, safe_pool = select_runtime_member(members)

    paper_root = Path(paper_results_root).expanduser().resolve()
    test_root = paper_root / "test_metrics" / SCENE
    logs_root = (
        Path(logs_dir).expanduser().resolve()
        if logs_dir is not None
        else paper_root / "logs" / "finalize_ifcbench_mainline"
    )
    selected_model_root = (
        finetune_model_root if decision["selectedFamily"] == "finetune" else full_model_root
    )
    runtime_dir = (
        Path(runtime_output_dir).expanduser().resolve()
        if runtime_output_dir is not None
        else _default_runtime_dir(runtime_member, selected_model_root)
    )
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else _default_manifest(paper_root)
    )
    registered = paths(data_root)
    if runtime_dir in {
        runtime_member.checkpoint.parent,
        registered["runtime_meta"].parent,
        registered["glb_root"],
    }:
        raise ValueError(
            "runtime output directory must be independent of source asset directories"
        )
    for name, target in (
        ("test output", test_root),
        ("logs", logs_root),
        ("manifest", manifest),
    ):
        if _paths_overlap(runtime_dir, target):
            raise ValueError(f"runtime output directory overlaps the {name} target")
    if _paths_overlap(test_root, logs_root):
        raise ValueError("test output directory overlaps the logs directory")
    test_jobs: list[dict[str, Any]] = []
    targets: list[Path] = [manifest, runtime_dir]
    for member in members:
        stem = f"{member.method_name}_seed{member.seed}"
        output = test_root / f"{stem}.json"
        sidecar = test_root / f"{stem}.sidecar"
        stdout = logs_root / f"{stem}.stdout.log"
        stderr = logs_root / f"{stem}.stderr.log"
        command = build_evaluate_command(
            member,
            data_root,
            output=output,
            sidecar_dir=sidecar,
            device=device,
            poses_per_batch=poses_per_batch,
        )
        test_jobs.append(
            {
                "job": f"test_{stem}",
                "seed": member.seed,
                "checkpoint": str(member.checkpoint),
                "calibration": str(member.calibration),
                "output": _new_target(output),
                "sidecar": _new_target(sidecar),
                "stdout": _new_target(stdout),
                "stderr": _new_target(stderr),
                "command": command,
                "commandString": shlex.join(command),
            }
        )
        targets.extend((output, sidecar, stdout, stderr))
    export_stdout = (
        logs_root
        / f"export_{runtime_member.method_name}_seed{runtime_member.seed}.stdout.log"
    )
    export_stderr = (
        logs_root
        / f"export_{runtime_member.method_name}_seed{runtime_member.seed}.stderr.log"
    )
    export_command = build_export_command(
        runtime_member,
        data_root,
        output_dir=runtime_dir,
    )
    export_job = {
        "job": f"export_{runtime_member.method_name}_seed{runtime_member.seed}",
        "seed": runtime_member.seed,
        "checkpoint": str(runtime_member.checkpoint),
        "calibration": str(runtime_member.calibration),
        "outputDir": _new_target(runtime_dir),
        "stdout": _new_target(export_stdout),
        "stderr": _new_target(export_stderr),
        "command": export_command,
        "commandString": shlex.join(export_command),
        "explicitCalibrationArgument": member_calibration_argument(runtime_member),
    }
    targets.extend((export_stdout, export_stderr))
    _assert_new_targets(targets)
    if device == "cuda" and len(gpu_ids) < len(members):
        raise ValueError("three-seed CUDA finalization requires at least three GPU IDs")
    return {
        "schema": PLAN_SCHEMA,
        "scene": SCENE,
        "dataRoot": str(Path(data_root).expanduser().resolve()),
        "contract": contract,
        "decision": decision,
        "members": [_member_record(member) for member in members],
        "runtimeSelection": {
            "seed": runtime_member.seed,
            "family": runtime_member.family,
            "method": runtime_member.method_name,
            "rule": (
                "validation-safe pool, useful cull first, then balanced accuracy, "
                "specificity, precision, LCB, and fewer predictions"
            ),
            "safePoolSeeds": [member.seed for member in safe_pool],
            "usefulCull": _cull_metric(runtime_member.validation),
        },
        "testJobs": test_jobs,
        "exportJob": export_job,
        "gpuIds": [int(value) for value in gpu_ids],
        "targets": {
            "testRoot": _new_target(test_root),
            "logsRoot": _new_target(logs_root),
            "runtimeDir": _new_target(runtime_dir),
            "manifest": _new_target(manifest),
        },
        "testRead": False,
    }


def member_calibration_argument(member: FinalMember) -> bool:
    return member.calibration_kind == "exact"


def _run_job(job: Mapping[str, Any], gpu_id: int | None = None) -> dict[str, Any]:
    stdout_path = Path(os.path.abspath(str(Path(str(job["stdout"])).expanduser())))
    stderr_path = Path(os.path.abspath(str(Path(str(job["stderr"])).expanduser())))
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if _target_exists(stdout_path) or _target_exists(stderr_path):
        raise FileExistsError(f"job log target already exists: {stdout_path} or {stderr_path}")
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if gpu_id is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
        "x", encoding="utf-8"
    ) as stderr:
        result = subprocess.run(
            [str(value) for value in job["command"]],
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    elapsed = time.perf_counter() - started
    outcome = {
        "job": str(job["job"]),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(elapsed),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    print(json.dumps(outcome, ensure_ascii=False), flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"IFCBench finalization job failed: {job['job']}; see {stdout_path}")
    return outcome


def _verify_test_output(job: Mapping[str, Any]) -> dict[str, Any]:
    output_path = Path(str(job["output"])).resolve()
    payload = _read_json(output_path)
    if payload.get("split") != "test" or payload.get("testRead") is not True:
        raise ValueError(f"formal evaluator did not produce a test payload: {output_path}")
    if _seed(payload.get("testEvaluationCount"), "testEvaluationCount") != 1:
        raise ValueError(f"formal test was not read exactly once: {output_path}")
    if _seed(payload.get("poseCount"), "test poseCount") != EXPECTED_SPLITS["test"]:
        raise ValueError(f"formal test pose count is wrong: {output_path}")
    if Path(str(payload.get("checkpoint"))).expanduser().resolve() != Path(
        str(job["checkpoint"])
    ).resolve():
        raise ValueError(f"test result checkpoint disagrees with plan: {output_path}")
    if Path(str(payload.get("calibrationSummary"))).expanduser().resolve() != Path(
        str(job["calibration"])
    ).resolve():
        raise ValueError(f"test result calibration disagrees with plan: {output_path}")
    sidecar_manifest = Path(str(job["sidecar"])).resolve() / "manifest.json"
    sidecar = _read_json(sidecar_manifest)
    if sidecar.get("schema") != "pvs-typed-score-sidecar-v1":
        raise ValueError(f"formal test sidecar schema is invalid: {sidecar_manifest}")
    if sidecar.get("split") != "test" or sidecar.get("testRead") is not True:
        raise ValueError(f"formal test sidecar provenance is invalid: {sidecar_manifest}")
    if _seed(sidecar.get("poseCount"), "sidecar poseCount") != EXPECTED_SPLITS["test"]:
        raise ValueError(f"formal test sidecar pose count is wrong: {sidecar_manifest}")
    return {
        "seed": _seed(job["seed"]),
        "output": str(output_path),
        "sidecar": str(sidecar_manifest),
        "testRead": True,
        "testEvaluationCount": 1,
        "poseCount": EXPECTED_SPLITS["test"],
        "threshold": payload.get("threshold"),
        "aggregate": payload.get("aggregate", payload.get("metrics")),
        "runtime": payload.get("runtime"),
    }


def _verify_export(job: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(str(job["outputDir"])).resolve()
    if not output_dir.is_dir():
        raise FileNotFoundError(f"runtime export did not create its directory: {output_dir}")
    expected = {
        "model_meta.json",
        "instance_runtime_features_fp16.bin",
        "instance_aabb_fp32.bin",
        "instance_to_glb_uint32.bin",
        "query_weights_fp16.bin",
        "frequency_cycles_fp32.bin",
        "chi_table_fp32.bin",
    }
    missing = [name for name in sorted(expected) if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"runtime export is missing files: {missing}")
    meta_path = output_dir / "model_meta.json"
    meta = _read_json(meta_path)
    if meta.get("schema") != RUNTIME_EXPORT_SCHEMA or meta.get("testRead") is not False:
        raise ValueError(f"runtime export metadata schema/provenance is invalid: {meta_path}")
    return {
        "outputDir": str(output_dir),
        "metadata": str(meta_path),
        "schema": meta.get("schema"),
        "calibrationFrozenThreshold": meta.get("calibrationFrozenThreshold"),
        "testRead": False,
    }


def execute_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Run all test jobs once, then export the selected runtime member."""
    job_targets: list[Path] = [Path(str(plan["targets"]["manifest"]))]
    for job in plan["testJobs"]:
        job_targets.extend(
            Path(str(job[name])) for name in ("output", "sidecar", "stdout", "stderr")
        )
    export_job = plan["exportJob"]
    job_targets.extend(
        Path(str(export_job[name])) for name in ("outputDir", "stdout", "stderr")
    )
    _assert_new_targets(job_targets)
    test_jobs = list(plan["testJobs"])
    gpu_ids = [int(value) for value in plan.get("gpuIds") or []]
    job_outcomes = []
    with ThreadPoolExecutor(max_workers=len(test_jobs)) as executor:
        futures = {
            executor.submit(
                _run_job,
                job,
                gpu_ids[index] if gpu_ids else None,
            ): job
            for index, job in enumerate(test_jobs)
        }
        for future in as_completed(futures):
            future.result()
            job_outcomes.append(_verify_test_output(futures[future]))
    job_outcomes.sort(key=lambda row: int(row["seed"]))
    _run_job(export_job)
    export_outcome = _verify_export(export_job)
    manifest = {
        "schema": FINALIZE_SCHEMA,
        "scene": SCENE,
        "decision": plan["decision"],
        "runtimeSelection": plan["runtimeSelection"],
        "testResults": job_outcomes,
        "runtimeExport": export_outcome,
        "formalTestEvaluationCount": len(job_outcomes),
        "plan": plan,
        "testRead": True,
        "testEvaluationCount": len(job_outcomes),
    }
    manifest_path = Path(
        os.path.abspath(str(Path(str(plan["targets"]["manifest"])).expanduser()))
    )
    _write_new_json(manifest_path, manifest)
    return manifest


def _preflight_view(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREFLIGHT_SCHEMA,
        "scene": plan["scene"],
        "contract": plan["contract"],
        "decision": plan["decision"],
        "members": plan["members"],
        "runtimeSelection": plan["runtimeSelection"],
        "targets": plan["targets"],
        "testRead": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "dry-run", "finalize"))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument(
        "--confirmation-summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--full-model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / FULL_EXPERIMENT,
    )
    parser.add_argument(
        "--full-benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / FULL_EXPERIMENT,
    )
    parser.add_argument(
        "--finetune-model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / FINETUNE_EXPERIMENT,
    )
    parser.add_argument(
        "--finetune-benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / FINETUNE_EXPERIMENT,
    )
    parser.add_argument(
        "--paper-results-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out/paper_results",
    )
    parser.add_argument("--runtime-output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--logs-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args(argv)


def _plan_from_args(args: argparse.Namespace) -> dict[str, Any]:
    confirmation_summary = (
        args.confirmation_summary
        if args.confirmation_summary is not None
        else args.finetune_benchmark_root / "confirmation_summary.json"
    )
    return prepare_plan(
        data_root=args.data_root,
        confirmation_summary_path=confirmation_summary,
        full_model_root=args.full_model_root,
        full_benchmark_root=args.full_benchmark_root,
        finetune_model_root=args.finetune_model_root,
        finetune_benchmark_root=args.finetune_benchmark_root,
        paper_results_root=args.paper_results_root,
        runtime_output_dir=args.runtime_output_dir,
        manifest_path=args.manifest,
        logs_dir=args.logs_dir,
        device=args.device,
        poses_per_batch=args.poses_per_batch,
        gpu_ids=args.gpu_ids,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    plan = _plan_from_args(args)
    if args.mode == "preflight":
        print(json.dumps(_preflight_view(plan), ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.mode == "dry-run":
        print(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False))
        return
    result = execute_plan(plan)
    print(
        json.dumps(
            {
                "manifest": str(plan["targets"]["manifest"]),
                "selectedFamily": result["decision"]["selectedFamily"],
                "runtimeSeed": result["runtimeSelection"]["seed"],
                "formalTestEvaluationCount": result["formalTestEvaluationCount"],
                "testRead": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
