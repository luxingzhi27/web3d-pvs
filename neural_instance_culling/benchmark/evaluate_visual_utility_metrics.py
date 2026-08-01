#!/usr/bin/env python3
"""Evaluate utility-preserving visibility and GLB download schedules.

The evaluator deliberately separates three concerns:

* instance visibility is evaluated from the stored back-camera candidate set;
* a score mode supplies one instance-level download ranking score;
* an explicit GLB aggregation and a strict resource budget turn that ranking
  into a download schedule.

The only utility target available in the current CSR data is ``visible_weights``.
Those values are weak importance evidence, not pixel coverage.  This script
never presents them as pixels and never fabricates an independent ranker or a
download-time cost table when those resources are absent.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runners import load_runner, load_runtime_meta, selected_default_specs, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    select_weighted_precision_workpoint,
    weighted_precision_selection_rule,
    weighted_recall_safe_rows,
)


UTILITY_TEACHER = "weak_log1p_visible_weights_not_pixel_coverage"
SCORE_MODES = ("visibility-only", "independent-utility", "current-cascade", "visibility-gated")
GLB_AGGREGATIONS = ("max", "sum", "top-k", "noisy-or")


def threshold_grid() -> np.ndarray:
    """Return the registered threshold grid used only on validation/calibration."""

    low = np.asarray(
        [0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2],
        dtype=np.float32,
    )
    high = np.linspace(0.22, 0.9, 35, dtype=np.float32)
    return np.unique(np.concatenate([low, high]))


def parse_int_list(value: str) -> tuple[int, ...]:
    values = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if any(v < 0 for v in values):
        raise ValueError(f"Integer budgets must be non-negative, got {values}.")
    return values


def parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(v.strip()) for v in value.split(",") if v.strip())
    if any(not np.isfinite(v) or v < 0.0 for v in values):
        raise ValueError(f"Float budgets must be finite and non-negative, got {values}.")
    return values


def parse_choice_list(value: str, allowed: Iterable[str], option_name: str) -> tuple[str, ...]:
    allowed_set = set(allowed)
    values = tuple(v.strip() for v in value.split(",") if v.strip())
    unknown = [v for v in values if v not in allowed_set]
    if unknown:
        raise ValueError(f"Unknown {option_name}: {unknown}; allowed values are {sorted(allowed_set)}.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {option_name} values are not allowed: {values}.")
    return values


def parse_learned_model_specs(values: Iterable[str]) -> dict[str, dict[str, str]]:
    """Parse explicit dynamic learned runners without changing global defaults.

    Format: ``name|checkpoint|runtime_features|calibration_or_test_summary``.
    The pipe delimiter keeps ordinary Linux paths readable and makes the
    checkpoint/feature/threshold provenance visible in the command line.
    """

    specs: dict[str, dict[str, str]] = {}
    for raw in values:
        parts = str(raw).split("|")
        if len(parts) != 4 or any(not part.strip() for part in parts):
            raise ValueError(
                "--learned-model-spec must use name|checkpoint|runtime_features|summary"
            )
        name, checkpoint, runtime_features, eval_summary = (part.strip() for part in parts)
        if name in specs:
            raise ValueError(f"Duplicate learned model spec: {name!r}")
        specs[name] = {
            "kind": "directional_occlusion_proxy_encoder",
            "checkpoint": checkpoint,
            "runtime_features": runtime_features,
            "eval_summary": eval_summary,
        }
    return specs


def parse_learned_aabb_ray_specs(values: Iterable[str]) -> dict[str, dict[str, str]]:
    """Parse explicit M6 learned AABB-ray runners.

    Format: ``name|checkpoint|training_summary``.  This is separate from the
    deployed directional model registration so the M6 baseline cannot be
    mistaken for the current mainline architecture.
    """

    specs: dict[str, dict[str, str]] = {}
    for raw in values:
        parts = str(raw).split("|")
        if len(parts) not in (2, 3) or any(not part.strip() for part in parts[:2]):
            raise ValueError("--learned-aabb-ray-spec must use name|checkpoint[|summary]")
        name, checkpoint = (part.strip() for part in parts[:2])
        summary = parts[2].strip() if len(parts) == 3 else ""
        if name in specs:
            raise ValueError(f"Duplicate learned AABB-ray model spec: {name!r}")
        spec = {"kind": "learned_aabb_ray", "checkpoint": checkpoint}
        if summary:
            spec["eval_summary"] = summary
        specs[name] = spec
    return specs


def parse_independent_ranker_specs(values: Iterable[str]) -> dict[str, dict[str, str]]:
    """Parse explicit train-only independent utility ranker checkpoints."""

    specs: dict[str, dict[str, str]] = {}
    for raw in values:
        parts = str(raw).split("|")
        if len(parts) != 2 or any(not part.strip() for part in parts):
            raise ValueError("--independent-ranker-spec must use name|checkpoint")
        name, checkpoint = (part.strip() for part in parts)
        if name in specs:
            raise ValueError(f"Duplicate independent ranker spec: {name!r}")
        specs[name] = {"kind": "independent_utility_ranker", "checkpoint": checkpoint}
    return specs


def load_frozen_thresholds(
    path: str | Path,
    model_names: Iterable[str],
    require_manifest: bool = False,
    dataset_dir: str | Path | None = None,
) -> dict[str, float]:
    """Load one registered threshold for every requested model.

    Validation/calibration fixtures may use a direct model-to-threshold mapping.
    Formal test requires the manifest generated by
    ``build_frozen_threshold_manifest.py`` and checks its one-shot provenance.
    Missing models are fatal; a default threshold would make a one-shot test
    unregistered.
    """

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frozen threshold file {manifest_path} must contain a JSON object.")
    values: dict[str, Any] | None = None
    if require_manifest:
        schema = payload.get("schema")
        if schema == "neuralstreamweb3d-frozen-test-manifest-v2":
            if payload.get("protocol") != "fixed_validation_independent_calibration_frozen_test":
                raise ValueError(f"{manifest_path} has an unexpected frozen-test protocol.")
            if int(payload.get("testEvaluationCountBeforeThisRun", -1)) != 0:
                raise ValueError(f"{manifest_path} is not a pre-test immutable manifest.")
            if dataset_dir is not None:
                recorded = payload.get("datasetDir")
                if not recorded or Path(recorded).resolve() != Path(dataset_dir).resolve():
                    raise ValueError(
                        f"Frozen manifest dataset mismatch: recorded={recorded!r}, requested={str(dataset_dir)!r}."
                    )
            entries = payload.get("models")
            if not isinstance(entries, dict):
                raise ValueError(f"Frozen threshold manifest {manifest_path} has no models object.")
            values = {}
            for name in model_names:
                details = entries.get(name)
                if not isinstance(details, dict):
                    raise ValueError(f"Frozen threshold manifest {manifest_path} has no model entry for {name!r}.")
                if int(details.get("preTestEvaluationCount", -1)) != 0:
                    raise ValueError(f"Model {name!r} in {manifest_path} is not pre-test.")
                values[name] = details.get("threshold")
        else:
            expected_schema = "neuralstreamweb3d-frozen-threshold-manifest-v1"
            if schema != expected_schema:
                raise ValueError(
                    f"Formal test requires {expected_schema!r} or the strict frozen-test manifest; "
                    f"{manifest_path} is not a supported frozen threshold manifest."
                )
            selection = str(payload.get("selection", "")).lower()
            if "calibration" not in selection or "test" not in selection:
                raise ValueError(
                    f"Frozen threshold manifest {manifest_path} does not record calibration-only selection and one-shot test."
                )
            provenance = payload.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError(f"Frozen threshold manifest {manifest_path} has no per-model provenance.")
            for name in model_names:
                details = provenance.get(name)
                if not isinstance(details, dict):
                    raise ValueError(f"Frozen threshold manifest {manifest_path} has no provenance for model {name!r}.")
                if details.get("protocol") != "frozen_calibration_one_shot_test":
                    raise ValueError(f"Model {name!r} in {manifest_path} is not marked frozen_calibration_one_shot_test.")
                if int(details.get("testEvaluationCount", 0)) != 1 or not details.get("frozenTestDigest"):
                    raise ValueError(f"Model {name!r} in {manifest_path} has incomplete one-shot test provenance.")
    if values is None:
        values = payload.get("thresholds", payload) if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise ValueError(f"Frozen threshold file {manifest_path} must contain a model-to-threshold mapping.")
    result: dict[str, float] = {}
    for name in model_names:
        if name not in values:
            raise ValueError(f"Frozen threshold file {manifest_path} has no threshold for model {name!r}.")
        threshold = float(values[name])
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Frozen threshold for {name!r} must be in [0, 1], got {threshold}.")
        result[name] = threshold
    return result


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b > 0.0 else float(default)


def load_glb_byte_costs(
    glb_index: str | Path | None,
    glb_root: str | Path | None,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    allow_missing: bool = False,
) -> np.ndarray:
    """Read actual filesystem bytes for every GLB used by the scene.

    Formal evaluation refuses median/count imputation.  The opt-in fallback is
    retained only for a clearly exploratory run and is recorded by the caller.
    """

    if num_glbs <= 0:
        raise ValueError("The scene must contain at least one GLB.")
    costs = np.zeros((num_glbs,), dtype=np.float64)
    if glb_index:
        path = Path(glb_index)
        if path.exists():
            root = Path(glb_root or path.parent)
            obj = json.loads(path.read_text(encoding="utf-8"))
            for entry in obj.get("entries", []):
                gid = int(entry.get("globalId", -1))
                if gid < 0 or gid >= num_glbs:
                    continue
                glb_path = root / str(entry.get("path", ""))
                if glb_path.exists() and glb_path.stat().st_size > 0:
                    costs[gid] = float(glb_path.stat().st_size)

    mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
    used = np.unique(mapping)
    invalid = used[(used < 0) | (used >= num_glbs)]
    if invalid.size:
        raise ValueError(f"instance_to_glb contains invalid GLB ids: {invalid[:20].tolist()}")
    missing = used[costs[used] <= 0.0]
    if missing.size and not allow_missing:
        raise FileNotFoundError(
            "Missing filesystem byte cost for runtime GLB ids "
            f"{missing[:20].tolist()}" + (" ..." if missing.size > 20 else "")
            + ". Formal utility@bytes evaluation refuses imputation."
        )
    if missing.size:
        present = costs[costs > 0.0]
        if not present.size:
            counts = np.bincount(mapping, minlength=num_glbs).astype(np.float64)
            costs = np.maximum(counts, 1.0)
        else:
            costs[missing] = max(1.0, float(np.median(present)))
    return costs


def _time_value_from_entry(entry: dict[str, Any]) -> float | None:
    def read_nonnegative(key: str) -> float | None:
        if key not in entry or entry[key] is None:
            return None
        value = float(entry[key])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid non-finite or negative GLB time cost in field {key!r}: {entry[key]!r}")
        return value

    total = read_nonnegative("totalDecodeUploadMs")
    decode = next(
        (value for key in ("decodeMs", "decodeTimeMs") if (value := read_nonnegative(key)) is not None),
        None,
    )
    upload = read_nonnegative("uploadMs")
    if decode is not None or upload is not None:
        component_total = float(decode or 0.0) + float(upload or 0.0)
        if total is not None and not np.isclose(total, component_total, rtol=1e-5, atol=1e-3):
            raise ValueError(
                "GLB time entry has inconsistent totalDecodeUploadMs: "
                f"total={total}, decode+upload={component_total}"
            )
        return total if total is not None else component_total
    if total is not None:
        return total
    for key in ("loadMs", "timeMs", "durationMs"):
        value = read_nonnegative(key)
        if value is not None:
            return value
    return None


def load_glb_time_costs(
    time_index: str | Path | None,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    allow_missing: bool = False,
) -> np.ndarray | None:
    """Load explicit per-GLB milliseconds, or return ``None`` when absent.

    A byte count is never converted to time.  The accepted JSON forms are a
    ``{"times": {"0": 1.2}}`` mapping or an ``entries`` list containing
    ``totalDecodeUploadMs`` or decode/upload component fields. When both
    decode and upload are present, their sum is used and an explicit total is
    checked against that sum.
    """

    if not time_index:
        return None
    path = Path(time_index)
    if not path.exists():
        raise FileNotFoundError(f"GLB time index does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema") not in (None, "neuralstreamweb3d-glb-cost-index-v1"):
        raise ValueError(
            f"Unsupported GLB cost index schema {payload.get('schema')!r}; "
            "expected neuralstreamweb3d-glb-cost-index-v1."
        )
    costs = np.zeros((num_glbs,), dtype=np.float64)
    if isinstance(payload, dict) and isinstance(payload.get("times"), dict):
        for raw_gid, raw_value in payload["times"].items():
            gid = int(raw_gid)
            if 0 <= gid < num_glbs:
                value = float(raw_value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"Invalid time cost for GLB {gid}: {raw_value!r}")
                costs[gid] = value
    elif isinstance(payload, dict) and isinstance(payload.get("costs"), dict):
        for raw_gid, raw_value in payload["costs"].items():
            gid = int(raw_gid)
            if 0 <= gid < num_glbs:
                value = float(raw_value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"Invalid time cost for GLB {gid}: {raw_value!r}")
                costs[gid] = value
    elif isinstance(payload, dict) and isinstance(payload.get("entries"), list):
        seen: set[int] = set()
        for entry in payload["entries"]:
            gid = int(entry.get("globalId", -1))
            if 0 <= gid < num_glbs:
                if gid in seen:
                    raise ValueError(f"GLB time index contains duplicate entry for runtime id {gid}.")
                seen.add(gid)
                value = _time_value_from_entry(entry)
                if value is not None:
                    costs[gid] = value
    else:
        raise ValueError(
            f"Unsupported GLB time index {path}; use a 'times'/'costs' mapping or entries with *Ms fields."
        )

    mapping = np.asarray(instance_to_glb, dtype=np.int64).reshape(-1)
    used = np.unique(mapping)
    missing = used[costs[used] <= 0.0]
    if missing.size and not allow_missing:
        raise ValueError(
            "GLB time index is incomplete for runtime ids "
            f"{missing[:20].tolist()}" + (" ..." if missing.size > 20 else "")
        )
    if missing.size:
        present = costs[costs > 0.0]
        costs[missing] = max(1e-6, float(np.median(present))) if present.size else 1.0
    return costs


def _probability_scores(values: np.ndarray) -> np.ndarray:
    """Normalize logits to [0, 1] without double-sigmoid-ing probabilities."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("Download ranking scores contain non-finite values.")
    if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
        clipped = np.clip(array, -60.0, 60.0)
        array = 1.0 / (1.0 + np.exp(-clipped))
    return np.clip(array, 0.0, 1.0)


def _aggregate_glb_scores(
    inverse: np.ndarray,
    instance_scores: np.ndarray,
    aggregation: str,
    top_k: int,
) -> np.ndarray:
    group_count = int(inverse.max()) + 1 if inverse.size else 0
    values = np.asarray(instance_scores, dtype=np.float64).reshape(-1)
    result = np.zeros((group_count,), dtype=np.float64)
    if aggregation == "max":
        result.fill(-np.inf)
        np.maximum.at(result, inverse, values)
        return result
    if aggregation == "sum":
        np.add.at(result, inverse, values)
        return result
    if aggregation == "top-k":
        k = max(1, int(top_k))
        for group in range(group_count):
            group_values = np.sort(values[inverse == group])[::-1]
            result[group] = float(group_values[:k].sum())
        return result
    if aggregation == "noisy-or":
        result.fill(1.0)
        for group in range(group_count):
            group_values = np.clip(values[inverse == group], 0.0, 1.0)
            result[group] = 1.0 - float(np.prod(1.0 - group_values))
        return result
    raise ValueError(f"Unsupported GLB aggregation: {aggregation}")


def _select_strict_prefix(order: np.ndarray, costs: np.ndarray, budget: float) -> np.ndarray:
    """Select a ranked prefix without ever exceeding a resource budget."""

    selected: list[int] = []
    consumed = 0.0
    for index in order.tolist():
        cost = max(0.0, float(costs[index]))
        if consumed + cost > float(budget) + 1e-9:
            break
        selected.append(int(index))
        consumed += cost
    return np.asarray(selected, dtype=np.int64)


def _budget_row(
    selected: np.ndarray,
    costs: np.ndarray,
    time_costs: np.ndarray | None,
    utility: np.ndarray,
    required: np.ndarray,
    candidate_bytes: float,
    candidate_time_ms: float | None,
) -> dict[str, Any]:
    selected_utility = float(utility[selected].sum()) if selected.size else 0.0
    selected_required = int(required[selected].sum()) if selected.size else 0
    total_utility = float(utility.sum())
    required_count = int(required.sum())
    utility_valid = total_utility > 0.0
    required_valid = required_count > 0
    row: dict[str, Any] = {
        "utilityRecall": selected_utility / total_utility if utility_valid else None,
        "requiredRecall": selected_required / required_count if required_valid else None,
        "utilityStatus": "available" if utility_valid else "not_applicable",
        "requiredStatus": "available" if required_valid else "not_applicable",
        "precision": selected_required / max(1, int(selected.size)),
        "selectedGlbCount": float(selected.size),
        "selectedBytes": float(costs[selected].sum()) if selected.size else 0.0,
        "byteReductionVsCandidate": safe_div(
            candidate_bytes - (float(costs[selected].sum()) if selected.size else 0.0),
            candidate_bytes,
            default=1.0,
        ),
    }
    if time_costs is not None:
        row["selectedTimeMs"] = float(time_costs[selected].sum()) if selected.size else 0.0
        row["timeReductionVsCandidate"] = safe_div(
            (candidate_time_ms or 0.0) - row["selectedTimeMs"],
            candidate_time_ms or 0.0,
            default=1.0,
        )
    return row


def _empty_budget_rows(
    budgets: tuple[int, ...],
    byte_budgets: tuple[float, ...],
    time_budgets_ms: tuple[float, ...],
    time_available: bool,
    time_reason: str,
) -> dict[str, Any]:
    def empty_row() -> dict[str, Any]:
        return {
            "utilityRecall": None,
            "requiredRecall": None,
            "utilityStatus": "not_applicable",
            "requiredStatus": "not_applicable",
            "precision": 1.0,
            "selectedGlbCount": 0.0,
            "selectedBytes": 0.0,
            "byteReductionVsCandidate": 1.0,
        }

    return {
        "countBudget": {str(b): empty_row() for b in budgets},
        "utilityAtBytes": {
            "status": "available",
            "budgetUnit": "bytes",
            "budgets": [float(b) for b in byte_budgets],
            "rows": {str(b): empty_row() for b in byte_budgets},
        },
        "utilityAtTime": {
            "status": "available" if time_available else "not_available",
            "budgetUnit": "milliseconds",
            "budgets": [float(b) for b in time_budgets_ms] if time_available else [],
            "rows": {str(b): empty_row() for b in time_budgets_ms} if time_available else {},
            **({} if time_available else {"reason": time_reason}),
        },
    }


def glb_budget_stats(
    glbs: np.ndarray,
    rank_scores: np.ndarray,
    utility: np.ndarray,
    glb_byte_cost: np.ndarray,
    count_budgets: tuple[int, ...],
    byte_budgets: tuple[float, ...],
    time_costs: np.ndarray | None,
    time_budgets_ms: tuple[float, ...],
    aggregation: str,
    top_k: int,
    target_utility_recall: float,
) -> dict[str, Any]:
    """Compute count, strict byte/time budget and target-utility rows for one pose."""

    glb_values = np.asarray(glbs, dtype=np.int64).reshape(-1)
    scores = np.asarray(rank_scores, dtype=np.float64).reshape(-1)
    weak_utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    if not (glb_values.size == scores.size == weak_utility.size):
        raise ValueError("GLB ids, ranking scores, and utility targets must have equal lengths.")
    unique_glbs, inverse = np.unique(glb_values, return_inverse=True)
    if unique_glbs.size == 0:
        return {
            "aggregation": aggregation,
            "topK": int(top_k),
            **_empty_budget_rows(
                count_budgets,
                byte_budgets,
                time_budgets_ms,
                time_costs is not None,
                "No explicit per-GLB decode/upload time index was supplied.",
            ),
            "bytePrefixAtUtilityTarget": {
                "selectedGlbCount": 0.0,
                "selectedBytes": 0.0,
                "byteReductionVsCandidate": 1.0,
                "utilityRecall": None,
                "utilityStatus": "not_applicable",
                "candidateGlbCount": 0.0,
                "candidateBytes": 0.0,
            },
        }

    glb_scores = _aggregate_glb_scores(inverse, scores, aggregation, top_k)
    glb_utility = np.zeros((unique_glbs.size,), dtype=np.float64)
    np.add.at(glb_utility, inverse, weak_utility)
    required = glb_utility > 0.0
    byte_costs = np.asarray(glb_byte_cost[unique_glbs], dtype=np.float64)
    if np.any(~np.isfinite(byte_costs)) or np.any(byte_costs <= 0.0):
        raise ValueError("Formal GLB byte costs must be finite and strictly positive.")
    time_cost_subset = None if time_costs is None else np.asarray(time_costs[unique_glbs], dtype=np.float64)
    if time_cost_subset is not None and (np.any(~np.isfinite(time_cost_subset)) or np.any(time_cost_subset <= 0.0)):
        raise ValueError("Formal GLB time costs must be finite and strictly positive.")
    order = np.lexsort((unique_glbs.astype(np.int64, copy=False), -glb_scores))
    candidate_bytes = float(byte_costs.sum())
    candidate_time = float(time_cost_subset.sum()) if time_cost_subset is not None else None

    count_rows: dict[str, dict[str, float]] = {}
    for budget in count_budgets:
        selected = order[: min(int(budget), order.size)]
        count_rows[str(budget)] = _budget_row(
            selected,
            byte_costs,
            time_cost_subset,
            glb_utility,
            required,
            candidate_bytes,
            candidate_time,
        )

    byte_rows: dict[str, dict[str, float]] = {}
    for budget in byte_budgets:
        selected = _select_strict_prefix(order, byte_costs, float(budget))
        byte_rows[str(budget)] = _budget_row(
            selected,
            byte_costs,
            time_cost_subset,
            glb_utility,
            required,
            candidate_bytes,
            candidate_time,
        )

    if time_cost_subset is None:
        time_block: dict[str, Any] = {
            "status": "not_available",
            "budgetUnit": "milliseconds",
            "budgets": [],
            "rows": {},
            "reason": "No explicit per-GLB decode/upload time index was supplied; bytes are not converted to time.",
        }
    else:
        time_rows: dict[str, dict[str, float]] = {}
        for budget in time_budgets_ms:
            selected = _select_strict_prefix(order, time_cost_subset, float(budget))
            time_rows[str(budget)] = _budget_row(
                selected,
                byte_costs,
                time_cost_subset,
                glb_utility,
                required,
                candidate_bytes,
                candidate_time,
            )
        time_block = {
            "status": "available",
            "budgetUnit": "milliseconds",
            "budgets": [float(b) for b in time_budgets_ms],
            "rows": time_rows,
        }

    if not np.any(glb_utility > 0.0):
        prefix_row = _budget_row(
            np.zeros((0,), dtype=np.int64),
            byte_costs,
            time_cost_subset,
            glb_utility,
            required,
            candidate_bytes,
            candidate_time,
        )
        prefix_row.update(
            {
                "candidateGlbCount": float(unique_glbs.size),
                "candidateBytes": candidate_bytes,
                "utilityStatus": "not_applicable",
                "requiredStatus": "not_applicable",
            }
        )
        if candidate_time is not None:
            prefix_row["candidateTimeMs"] = candidate_time
        return {
            "aggregation": aggregation,
            "topK": int(top_k),
            "countBudget": count_rows,
            "utilityAtBytes": {
                "status": "available",
                "budgetUnit": "bytes",
                "budgets": [float(b) for b in byte_budgets],
                "rows": byte_rows,
            },
            "utilityAtTime": time_block,
            "bytePrefixAtUtilityTarget": prefix_row,
        }

    cumulative = np.cumsum(glb_utility[order])
    target = float(target_utility_recall) * max(1e-9, float(glb_utility.sum()))
    hit = np.flatnonzero(cumulative >= target)
    prefix_count = int(hit[0] + 1) if hit.size else int(order.size)
    prefix = order[:prefix_count]
    prefix_row = _budget_row(
        prefix,
        byte_costs,
        time_cost_subset,
        glb_utility,
        required,
        candidate_bytes,
        candidate_time,
    )
    prefix_row.update(
        {
            "candidateGlbCount": float(unique_glbs.size),
            "candidateBytes": candidate_bytes,
            "utilityRecall": float(cumulative[prefix_count - 1] / max(1e-9, float(glb_utility.sum())))
            if prefix_count > 0
            else 1.0,
        }
    )
    if candidate_time is not None:
        prefix_row["candidateTimeMs"] = candidate_time

    return {
        "aggregation": aggregation,
        "topK": int(top_k),
        "countBudget": count_rows,
        "utilityAtBytes": {
            "status": "available",
            "budgetUnit": "bytes",
            "budgets": [float(b) for b in byte_budgets],
            "rows": byte_rows,
        },
        "utilityAtTime": time_block,
        "bytePrefixAtUtilityTarget": prefix_row,
    }


def _score_mode_result(result: Any, mode: str) -> tuple[np.ndarray | None, dict[str, str]]:
    visibility = np.asarray(getattr(result, "scores", None), dtype=np.float64).reshape(-1)
    if mode == "visibility-only":
        return np.clip(visibility, 0.0, 1.0), {
            "status": "implemented",
            "definition": "instance visibility probability",
            "source": "result.scores",
        }
    if mode == "independent-utility":
        values = getattr(result, "independent_utility_scores", None)
        if values is None:
            return None, {
                "status": "not_implemented",
                "definition": "independently trained GLB utility/ranking model",
                "reason": "This runner does not expose an independently trained utility score.",
            }
        return _probability_scores(values), {
            "status": "implemented",
            "definition": "independently trained GLB utility/ranking model",
            "source": "result.independent_utility_scores",
        }
    if mode == "current-cascade":
        values = getattr(result, "download_scores", None)
        if values is None:
            return None, {
                "status": "not_available",
                "definition": "current visibility-to-utility-to-download cascade",
                "reason": "This runner does not expose a download head.",
            }
        return _probability_scores(values), {
            "status": "implemented",
            "definition": "sigmoid-normalized current cascade download logit",
            "source": "result.download_scores",
        }
    if mode == "visibility-gated":
        utility = getattr(result, "utility_scores", None)
        if utility is None:
            return None, {
                "status": "not_available",
                "definition": "visibility probability multiplied by utility/salience score",
                "reason": "This runner does not expose a utility head.",
            }
        utility_values = np.clip(np.asarray(utility, dtype=np.float64).reshape(-1), 0.0, 1.0)
        return np.clip(visibility, 0.0, 1.0) * utility_values, {
            "status": "diagnostic_proxy",
            "definition": "visibility probability multiplied by the current utility-head output",
            "source": "result.scores * result.utility_scores",
            "warning": "The current utility head itself reads visibility; this is not an independently trained salience head.",
        }
    raise ValueError(f"Unsupported score mode: {mode}")


def _average_numeric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    # A budget row may be structurally present but not applicable for a pose
    # (for example, zero visible utility is represented by ``None``).  Keep
    # that semantic distinction instead of coercing None to float and failing
    # an otherwise valid multi-pose aggregate.
    numeric_values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (bool, int, float, np.integer, np.floating)) and not isinstance(value, bool):
                numeric_values.setdefault(key, []).append(float(value))
    return {key: float(np.mean(values)) for key, values in sorted(numeric_values.items()) if values}


def _average_rows_with_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Average numeric budget fields without discarding applicability states."""

    output: dict[str, Any] = _average_numeric_rows(rows)
    status_keys = ("utilityStatus", "requiredStatus")
    for key in status_keys:
        values = {str(row[key]) for row in rows if key in row}
        if values:
            output[key] = next(iter(values)) if len(values) == 1 else "mixed"
    return output


def _average_aggregation_stats(stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not stats:
        return {}
    first = stats[0]

    def average_budget_block(block_name: str) -> dict[str, Any]:
        keys = sorted({key for item in stats for key in (item.get(block_name) or {}).keys()})
        return {
            key: _average_rows_with_status(
                [(item.get(block_name) or {})[key] for item in stats if key in (item.get(block_name) or {})]
            )
            for key in keys
        }

    def average_resource_block(block_name: str) -> dict[str, Any]:
        blocks = [item.get(block_name) or {} for item in stats]
        statuses = {str(block.get("status", "unknown")) for block in blocks}
        status = next(iter(statuses)) if len(statuses) == 1 else "mixed"
        output: dict[str, Any] = {
            "status": status,
            "budgetUnit": blocks[0].get("budgetUnit"),
            "budgets": blocks[0].get("budgets", []),
            "rows": {},
        }
        if status != "available":
            reasons = sorted({str(block.get("reason", "")) for block in blocks if block.get("reason")})
            if reasons:
                output["reason"] = reasons[0]
            return output
        keys = sorted({key for block in blocks for key in (block.get("rows") or {}).keys()})
        output["rows"] = {
            key: _average_rows_with_status(
                [(block.get("rows") or {})[key] for block in blocks if key in (block.get("rows") or {})]
            )
            for key in keys
        }
        return output

    return {
        "aggregation": first.get("aggregation"),
        "topK": int(first.get("topK", 0)),
        "budgetPoseCount": int(len(stats)),
        "countBudget": average_budget_block("countBudget"),
        "utilityAtBytes": average_resource_block("utilityAtBytes"),
        "utilityAtTime": average_resource_block("utilityAtTime"),
        "bytePrefixAtUtilityTarget": _average_rows_with_status(
            [item.get("bytePrefixAtUtilityTarget") or {} for item in stats]
        ),
    }


def _empty_pose_metrics(n_th: int) -> dict[str, np.ndarray]:
    return {key: np.ones((n_th,), dtype=np.float64) for key in ("tp", "tn", "precision", "recall", "specificity", "accuracy", "balanced_accuracy", "f1", "jaccard", "weighted_recall", "utility_recall")}


def evaluate_runner(
    runner: Any,
    split: Any,
    thresholds: np.ndarray,
    count_budgets: tuple[int, ...],
    byte_budgets: tuple[float, ...],
    time_costs: np.ndarray | None,
    time_budgets_ms: tuple[float, ...],
    glb_byte_cost: np.ndarray,
    poses_per_batch: int,
    max_eval_poses: int,
    max_candidates_per_pose: int,
    seed: int,
    target_utility_recall: float,
    target_weighted_recall: float,
    score_modes: tuple[str, ...],
    aggregations: tuple[str, ...],
    aggregation_top_k: int,
    sample_with_replacement: bool,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate one runner on every pose in the selected split.

    ``max_eval_poses`` and replacement sampling are retained for train-only
    smoke runs.  Formal validation, calibration, and test calls are checked by
    ``main`` and always traverse the complete unique split.
    """

    rng = np.random.default_rng(seed)
    if sample_with_replacement:
        steps = max(1, int(np.ceil(max(1, max_eval_poses) / max(1, poses_per_batch))))
    else:
        steps = None
    total_steps = (
        max(1, int(np.ceil(max(1, max_eval_poses) / max(1, poses_per_batch))))
        if sample_with_replacement
        else int(np.ceil(max(1, split.pose_indices.size) / max(1, poses_per_batch)))
    )

    thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
    n_th = int(thresholds.size)
    if n_th == 0:
        raise ValueError("At least one visibility threshold is required.")
    agg_tp = np.zeros((n_th,), dtype=np.float64)
    agg_fp = np.zeros((n_th,), dtype=np.float64)
    agg_fn = np.zeros((n_th,), dtype=np.float64)
    agg_tn = np.zeros((n_th,), dtype=np.float64)
    pose_precision = np.zeros((n_th,), dtype=np.float64)
    pose_recall = np.zeros((n_th,), dtype=np.float64)
    pose_specificity = np.zeros((n_th,), dtype=np.float64)
    pose_accuracy = np.zeros((n_th,), dtype=np.float64)
    pose_balanced_accuracy = np.zeros((n_th,), dtype=np.float64)
    pose_f1 = np.zeros((n_th,), dtype=np.float64)
    pose_jaccard = np.zeros((n_th,), dtype=np.float64)
    pose_weighted_recall = np.zeros((n_th,), dtype=np.float64)
    pose_utility_recall = np.zeros((n_th,), dtype=np.float64)
    pose_pred_count = np.zeros((n_th,), dtype=np.float64)
    pose_pred_glb_count = np.zeros((n_th,), dtype=np.float64)
    pose_pred_bytes = np.zeros((n_th,), dtype=np.float64)
    pose_useful_cull = np.zeros((n_th,), dtype=np.float64)
    pose_bad_cull = np.zeros((n_th,), dtype=np.float64)
    pose_negative_fpr = np.zeros((n_th,), dtype=np.float64)
    gt_count_sum = 0.0
    candidate_count_sum = 0.0
    candidate_glb_count_sum = 0.0
    candidate_bytes_sum = 0.0
    total_utility_sum = 0.0
    utility_valid_pose_count = 0
    utility_not_applicable_pose_count = 0
    pose_count = 0
    empty_gt_pose_count = 0
    empty_candidate_pose_count = 0
    forward_ms: list[float] = []
    total_ms: list[float] = []
    runner_diagnostics: dict[str, list[float]] = {}
    mode_status: dict[str, dict[str, str]] = {}
    mode_stats: dict[str, dict[str, list[dict[str, Any]]]] = {
        mode: {aggregation: [] for aggregation in aggregations} for mode in score_modes
    }

    batches = split.pose_set_batches(
        poses_per_batch,
        rng,
        steps,
        include_empty=True,
    )
    for pose_indices in tqdm(
        batches,
        total=total_steps,
        desc=runner.name,
        ascii=True,
        disable=not progress,
    ):
        batch = split.build_pose_set_batch(
            pose_indices,
            runner.world_aabbs,
            rng,
            max_candidates_per_pose=max_candidates_per_pose,
            include_empty=True,
        )
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
        batch_visible_counts = np.asarray(batch.get("visible_counts", []), dtype=np.int64)
        batch_candidate_counts = np.asarray(batch.get("candidate_counts", []), dtype=np.int64)
        empty_rows = np.flatnonzero(np.diff(offsets) == 0)
        for row_index in empty_rows.tolist():
            visible_count = int(batch_visible_counts[row_index]) if row_index < batch_visible_counts.size else 0
            candidate_count = int(batch_candidate_counts[row_index]) if row_index < batch_candidate_counts.size else 0
            if visible_count != 0 or candidate_count != 0:
                raise ValueError(
                    f"Invalid empty pose row {row_index}: visible_count={visible_count}, candidate_count={candidate_count}."
                )
            pose_accuracy += 1.0
            pose_balanced_accuracy += 1.0
            pose_precision += 1.0
            pose_recall += 1.0
            pose_specificity += 1.0
            pose_f1 += 1.0
            pose_jaccard += 1.0
            pose_weighted_recall += 1.0
            pose_count += 1
            empty_gt_pose_count += 1
            empty_candidate_pose_count += 1

        if batch["instance"].size == 0:
            continue
        result = runner.score_batch(batch)
        forward_ms.append(float(result.forward_ms))
        total_ms.append(float(result.total_ms))
        if getattr(result, "diagnostics", None):
            for key, value in result.diagnostics.items():
                try:
                    runner_diagnostics.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    continue

        ids_all = np.asarray(batch["instance"], dtype=np.int64).reshape(-1)
        if ids_all.size and (int(ids_all.min()) < 0 or int(ids_all.max()) >= runner.instance_to_glb.size):
            raise ValueError("Runner received an instance id outside instance_to_glb.")
        glb_all_raw = np.asarray(runner.instance_to_glb[ids_all], dtype=np.int64)
        if glb_all_raw.size and (int(glb_all_raw.min()) < 0 or int(glb_all_raw.max()) >= glb_byte_cost.size):
            raise ValueError("instance_to_glb contains an id without a corresponding byte cost.")
        target_all = np.asarray(batch["target"], dtype=bool).reshape(-1)
        weights_all = np.asarray(
            batch.get("visible_weights", np.zeros_like(target_all, dtype=np.float32)), dtype=np.float64
        ).reshape(-1)
        if not (ids_all.size == target_all.size == weights_all.size == np.diff(offsets).sum()):
            raise ValueError("CSR batch arrays have inconsistent lengths.")
        if not np.isfinite(weights_all).all():
            raise ValueError("visible_weights contain non-finite values.")
        if np.any(weights_all < 0.0):
            raise ValueError("visible_weights must be non-negative; refusing to clamp invalid utility evidence.")
        utility_all = np.where(target_all, np.log1p(weights_all), 0.0)
        mode_vectors: dict[str, np.ndarray] = {}
        for mode in score_modes:
            vector, status = _score_mode_result(result, mode)
            if mode not in mode_status:
                mode_status[mode] = status
            elif mode_status[mode].get("status") in {"not_available", "not_implemented"} and status.get("status") == "implemented":
                mode_status[mode] = status
            if vector is not None:
                if vector.size != ids_all.size:
                    raise ValueError(f"Score mode {mode} returned {vector.size} scores for {ids_all.size} rows.")
                mode_vectors[mode] = vector

        for pose_i in range(offsets.size - 1):
            start = int(offsets[pose_i])
            end = int(offsets[pose_i + 1])
            if end <= start:
                continue
            scores = np.asarray(result.scores[start:end], dtype=np.float64).reshape(-1)
            truth = target_all[start:end]
            weights = weights_all[start:end]
            utility = utility_all[start:end]
            glbs = glb_all_raw[start:end]
            if not np.all(np.isfinite(scores)):
                raise ValueError("Visibility scores contain non-finite values.")
            candidate_count = float(end - start)
            gt_count = float(truth.sum())
            negative_count = max(0.0, candidate_count - gt_count)
            total_utility = float(utility.sum())
            candidate_glbs = np.unique(glbs)
            candidate_bytes = float(glb_byte_cost[candidate_glbs].sum()) if candidate_glbs.size else 0.0
            pred = scores[:, None] >= thresholds[None, :]
            truth_col = truth[:, None]
            tp = np.logical_and(pred, truth_col).sum(axis=0).astype(np.float64)
            fp = np.logical_and(pred, ~truth_col).sum(axis=0).astype(np.float64)
            fn = np.logical_and(~pred, truth_col).sum(axis=0).astype(np.float64)
            tn = np.logical_and(~pred, ~truth_col).sum(axis=0).astype(np.float64)
            weighted_hit = (np.logical_and(pred, truth_col) * weights[:, None]).sum(axis=0)
            weighted_gt = float((truth * weights).sum())
            utility_hit = (np.logical_and(pred, truth_col) * utility[:, None]).sum(axis=0)
            precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0.0)
            recall = np.divide(tp, tp + fn, out=np.ones_like(tp), where=(tp + fn) > 0.0)
            specificity = np.divide(tn, tn + fp, out=np.ones_like(tn), where=(tn + fp) > 0.0)
            accuracy = (tp + tn) / max(1.0, candidate_count)
            balanced_accuracy = 0.5 * (recall + specificity)
            f1 = 2.0 * precision * recall / np.maximum(1e-8, precision + recall)
            jaccard = np.divide(tp, tp + fp + fn, out=np.ones_like(tp), where=(tp + fp + fn) > 0.0)
            weighted_recall = (
                np.divide(weighted_hit, weighted_gt, out=np.ones_like(weighted_hit), where=weighted_gt > 0.0)
                if weighted_gt > 0.0
                else np.ones_like(weighted_hit)
            )
            utility_valid = total_utility > 0.0
            utility_recall = utility_hit / total_utility if utility_valid else np.ones_like(utility_hit)
            useful_cull = tn / max(1.0, candidate_count)
            bad_cull = fn / max(1.0, candidate_count)
            negative_fpr = fp / max(1.0, negative_count)

            for threshold_index in range(n_th):
                predicted_glbs = np.unique(glbs[pred[:, threshold_index]])
                pose_pred_glb_count[threshold_index] += float(predicted_glbs.size)
                pose_pred_bytes[threshold_index] += float(glb_byte_cost[predicted_glbs].sum()) if predicted_glbs.size else 0.0

            agg_tp += tp
            agg_fp += fp
            agg_fn += fn
            agg_tn += tn
            pose_precision += precision
            pose_recall += recall
            pose_specificity += specificity
            pose_accuracy += accuracy
            pose_balanced_accuracy += balanced_accuracy
            pose_f1 += f1
            pose_jaccard += jaccard
            pose_weighted_recall += weighted_recall
            if utility_valid:
                pose_utility_recall += utility_recall
                utility_valid_pose_count += 1
            else:
                utility_not_applicable_pose_count += 1
            pose_pred_count += pred.sum(axis=0).astype(np.float64)
            pose_useful_cull += useful_cull
            pose_bad_cull += bad_cull
            pose_negative_fpr += negative_fpr
            gt_count_sum += gt_count
            candidate_count_sum += candidate_count
            candidate_glb_count_sum += float(candidate_glbs.size)
            candidate_bytes_sum += candidate_bytes
            total_utility_sum += float(utility.sum())

            for mode, mode_scores_all in mode_vectors.items():
                mode_scores = mode_scores_all[start:end]
                for aggregation in aggregations:
                    mode_stats[mode][aggregation].append(
                        glb_budget_stats(
                            glbs,
                            mode_scores,
                            utility,
                            glb_byte_cost,
                            count_budgets,
                            byte_budgets,
                            time_costs,
                            time_budgets_ms,
                            aggregation,
                            aggregation_top_k,
                            target_utility_recall,
                        )
                    )
            pose_count += 1

    rows: list[dict[str, Any]] = []
    average_pose_count = max(1, pose_count)
    avg_candidate = candidate_count_sum / average_pose_count
    avg_gt = gt_count_sum / average_pose_count
    avg_candidate_glb = candidate_glb_count_sum / average_pose_count
    avg_candidate_bytes = candidate_bytes_sum / average_pose_count
    for index, threshold in enumerate(thresholds):
        agg_precision = safe_div(agg_tp[index], agg_tp[index] + agg_fp[index])
        agg_recall = safe_div(agg_tp[index], agg_tp[index] + agg_fn[index])
        agg_specificity = safe_div(agg_tn[index], agg_tn[index] + agg_fp[index], default=1.0)
        pose_rec = pose_recall[index] / average_pose_count
        pose_weighted = pose_weighted_recall[index] / average_pose_count
        pose_utility = (
            float(pose_utility_recall[index] / utility_valid_pose_count)
            if utility_valid_pose_count > 0
            else None
        )
        safety_multiplier = min(1.0, pose_rec / max(1e-8, 0.95)) * min(1.0, pose_weighted / max(1e-8, 0.99))
        avg_pred = pose_pred_count[index] / average_pose_count
        rows.append(
            {
                "threshold": float(threshold),
                "pose_accuracy": float(pose_accuracy[index] / average_pose_count),
                "pose_balanced_accuracy": float(pose_balanced_accuracy[index] / average_pose_count),
                "pose_precision": float(pose_precision[index] / average_pose_count),
                "pose_recall": float(pose_rec),
                "pose_specificity": float(pose_specificity[index] / average_pose_count),
                "pose_f1": float(pose_f1[index] / average_pose_count),
                "pose_jaccard": float(pose_jaccard[index] / average_pose_count),
                "pose_weighted_recall": float(pose_weighted),
                "pose_visual_utility_recall": pose_utility,
                "pose_miss_visual_utility_rate": (
                    float(1.0 - pose_utility) if pose_utility is not None else None
                ),
                "pose_negative_fpr": float(pose_negative_fpr[index] / average_pose_count),
                "pose_raw_candidate_reduction_ratio": float(1.0 - avg_pred / max(1.0, avg_candidate)),
                "pose_useful_cull_candidate_ratio": float(pose_useful_cull[index] / average_pose_count),
                "pose_bad_cull_candidate_ratio": float(pose_bad_cull[index] / average_pose_count),
                "safety_adjusted_cull_score": float((pose_useful_cull[index] / average_pose_count) * safety_multiplier),
                "agg_accuracy": float(safe_div(agg_tp[index] + agg_tn[index], agg_tp[index] + agg_fp[index] + agg_fn[index] + agg_tn[index])),
                "agg_balanced_accuracy": float(0.5 * (agg_recall + agg_specificity)),
                "agg_precision": float(agg_precision),
                "agg_recall": float(agg_recall),
                "agg_specificity": float(agg_specificity),
                "agg_f1": float(2.0 * agg_precision * agg_recall / max(1e-8, agg_precision + agg_recall)),
                "avg_candidate_count": float(avg_candidate),
                "avg_gt_count": float(avg_gt),
                "avg_pred_count": float(avg_pred),
                "avg_candidate_glb_count": float(avg_candidate_glb),
                "avg_candidate_glb_bytes": float(avg_candidate_bytes),
                "avg_pred_glb_count": float(pose_pred_glb_count[index] / average_pose_count),
                "avg_pred_glb_bytes": float(pose_pred_bytes[index] / average_pose_count),
                "tp": int(agg_tp[index]),
                "fp": int(agg_fp[index]),
                "fn": int(agg_fn[index]),
                "tn": int(agg_tn[index]),
                "eval_pose_count": int(pose_count),
            }
        )

    weighted_safe = weighted_recall_safe_rows(rows, target_weighted_recall)
    utility_safe = [
        row
        for row in weighted_safe
        if row.get("pose_visual_utility_recall") is not None
        and row["pose_visual_utility_recall"] >= target_utility_recall
    ]
    high_recall = [row for row in rows if row["pose_recall"] >= 0.95]
    workpoints = {
        "primaryWeightedPrecision": select_weighted_precision_workpoint(rows, target_weighted_recall),
        "primaryUtilitySafeUsefulCull": max(utility_safe, key=lambda row: row["pose_useful_cull_candidate_ratio"]) if utility_safe else None,
        "bestPrecisionAtRecall95": max(high_recall, key=lambda row: row["pose_precision"]) if high_recall else None,
        "bestF1": max(rows, key=lambda row: row["pose_f1"]) if rows else None,
    }
    score_mode_summaries: dict[str, Any] = {}
    for mode in score_modes:
        status = mode_status.get(mode)
        if status is None:
            if mode == "independent-utility":
                status = {
                    "status": "not_implemented",
                    "definition": "independently trained GLB utility/ranking model",
                    "reason": "No independently trained ranker was registered.",
                }
            else:
                status = {"status": "not_available", "reason": "No non-empty candidate pose exposed the required score output."}
        score_mode_summaries[mode] = {
            **status,
            "aggregations": {
                aggregation: _average_aggregation_stats(mode_stats[mode][aggregation])
                for aggregation in aggregations
                if mode_stats[mode][aggregation]
            },
        }

    return {
        "name": runner.name,
        "kind": runner.kind,
        "runnerInfo": {
            "informationLevel": str(getattr(runner, "information_level", "unspecified")),
            "decisionMode": str(getattr(runner, "decision_mode", "threshold")),
        },
        "utilityTeacher": UTILITY_TEACHER,
        "best": workpoints["primaryWeightedPrecision"],
        "workpoints": workpoints,
        "thresholdRows": rows,
        "scoreModes": score_mode_summaries,
        "totalWeakUtility": float(total_utility_sum),
        "utilityValidPoseCount": int(utility_valid_pose_count),
        "utilityNotApplicablePoseCount": int(utility_not_applicable_pose_count),
        "utilityValidity": "utility recall excludes poses whose visible_weights sum to zero; those poses are reported as not_applicable",
        "avgForwardMsPerBatch": float(np.mean(forward_ms)) if forward_ms else 0.0,
        "avgTotalMsPerBatch": float(np.mean(total_ms)) if total_ms else 0.0,
        "emptyGtPoseCount": int(empty_gt_pose_count),
        "emptyCandidatePoseCount": int(empty_candidate_pose_count),
        "runnerDiagnostics": {key: float(np.mean(values)) for key, values in runner_diagnostics.items() if values},
    }


def _fmt(value: Any, digits: int = 3, default: str = "-") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return default


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    meta = payload["meta"]
    lines = [
        "# M7 Unified Download Scheduling Metrics",
        "",
        f"- Created: `{meta['created']}`",
        f"- Dataset: `{meta['datasetDir']}`",
        f"- Requested/resolved split: `{meta['requestedSplit']}` / `{meta['split']}`",
        f"- Threshold mode: `{meta['thresholdMode']}`",
        f"- Evaluated poses: `{meta['evaluatedPoses']}`",
        f"- Utility teacher: `{meta['utilityTeacher']}`",
        "",
        "`visible_weights` is weak importance evidence. It is not pixel coverage, and the report must not be read as a pixel-level image result.",
        "",
        "## Visibility Workpoints",
        "",
        "| Model | Workpoint | Threshold | Pose precision | Pose recall | Weighted recall | Utility recall | Useful cull | Bad cull | Avg predicted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in payload["summaries"]:
        for key, label in (
            ("primaryWeightedPrecision", "weighted-safe precision"),
            ("primaryUtilitySafeUsefulCull", "utility-safe useful cull"),
            ("bestPrecisionAtRecall95", "precision at recall 95"),
            ("bestF1", "best F1 diagnostic"),
        ):
            row = (summary.get("workpoints") or {}).get(key)
            if not row:
                continue
            lines.append(
                f"| {summary['name']} | {label} | {_fmt(row.get('threshold'))} | {_fmt(row.get('pose_precision'))} | "
                f"{_fmt(row.get('pose_recall'))} | {_fmt(row.get('pose_weighted_recall'))} | "
                f"{_fmt(row.get('pose_visual_utility_recall'))} | {_fmt(row.get('pose_useful_cull_candidate_ratio'))} | "
                f"{_fmt(row.get('pose_bad_cull_candidate_ratio'))} | {_fmt(row.get('avg_pred_count'), 2)} |"
            )
    lines.extend(
        [
            "",
            "## Score Modes And GLB Aggregations",
            "",
            "| Model | Score mode | Status | Aggregation | Budget poses | Byte budget status | Time budget status |",
            "|---|---|---|---|---:|---|---|",
        ]
    )
    for summary in payload["summaries"]:
        for mode, mode_summary in (summary.get("scoreModes") or {}).items():
            aggregations = mode_summary.get("aggregations") or {}
            if not aggregations:
                lines.append(
                    f"| {summary['name']} | {mode} | {mode_summary.get('status', '-')} | - | 0 | - | - |"
                )
                continue
            for aggregation, aggregation_summary in aggregations.items():
                lines.append(
                    f"| {summary['name']} | {mode} | {mode_summary.get('status', '-')} | {aggregation} | "
                    f"{aggregation_summary.get('budgetPoseCount', 0)} | "
                    f"{(aggregation_summary.get('utilityAtBytes') or {}).get('status', '-')} | "
                    f"{(aggregation_summary.get('utilityAtTime') or {}).get('status', '-')} |"
                )
    lines.extend(["", "## Utility At Strict Byte Budgets", "", "| Model | Score mode | Aggregation | Budget bytes | Utility recall | Selected bytes | Required recall |", "|---|---|---|---:|---:|---:|---:|"])
    for summary in payload["summaries"]:
        for mode, mode_summary in (summary.get("scoreModes") or {}).items():
            for aggregation, aggregation_summary in (mode_summary.get("aggregations") or {}).items():
                byte_block = aggregation_summary.get("utilityAtBytes") or {}
                if byte_block.get("status") != "available":
                    continue
                for budget, row in (byte_block.get("rows") or {}).items():
                    lines.append(
                        f"| {summary['name']} | {mode} | {aggregation} | {budget} | {_fmt(row.get('utilityRecall'))} | "
                        f"{_fmt(row.get('selectedBytes'), 0)} | {_fmt(row.get('requiredRecall'))} |"
                    )
    lines.extend(["", "## Utility At Time Budgets", "", "Time rows are emitted only when an explicit per-GLB decode/upload time index is supplied.", "", "| Model | Score mode | Aggregation | Status | Budget ms | Utility recall | Selected time ms |", "|---|---|---|---|---:|---:|---:|"])
    for summary in payload["summaries"]:
        for mode, mode_summary in (summary.get("scoreModes") or {}).items():
            for aggregation, aggregation_summary in (mode_summary.get("aggregations") or {}).items():
                time_block = aggregation_summary.get("utilityAtTime") or {}
                if time_block.get("status") != "available":
                    lines.append(f"| {summary['name']} | {mode} | {aggregation} | {time_block.get('status', '-')} | - | - | - |")
                    continue
                for budget, row in (time_block.get("rows") or {}).items():
                    lines.append(
                        f"| {summary['name']} | {mode} | {aggregation} | available | {budget} | {_fmt(row.get('utilityRecall'))} | "
                        f"{_fmt(row.get('selectedTimeMs'))} |"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _SelfTestSplit:
    pose_indices = np.asarray([0], dtype=np.int64)

    def pose_set_batches(self, poses_per_batch, rng, max_steps, include_empty=True):
        del poses_per_batch, rng, max_steps, include_empty
        yield np.asarray([0], dtype=np.int64)

    def build_pose_set_batch(self, pose_indices, world_aabbs, rng, max_candidates_per_pose=0, include_empty=True):
        del pose_indices, world_aabbs, rng, max_candidates_per_pose, include_empty
        return {
            "camera": np.zeros((2, 3), dtype=np.float32),
            "camera_world": np.zeros((2, 3), dtype=np.float32),
            "camera_view": np.tile(np.asarray([0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32), (2, 1)),
            "instance": np.asarray([0, 1], dtype=np.int64),
            "target": np.asarray([1.0, 0.0], dtype=np.float32),
            "visible_weights": np.asarray([3.0, 0.0], dtype=np.float32),
            "pose_offsets": np.asarray([0, 2], dtype=np.int64),
            "visible_counts": np.asarray([1], dtype=np.int64),
            "candidate_counts": np.asarray([2], dtype=np.int64),
        }


def run_self_test() -> dict[str, Any]:
    """Run a deterministic, model-free fixture covering M7 invariants."""

    glbs = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    scores = np.asarray([0.9, 0.1, 0.8, 0.7, 0.2], dtype=np.float64)
    utility = np.asarray([1.0, 0.0, 0.6, 0.2, 0.0], dtype=np.float64)
    byte_costs = np.asarray([10.0, 20.0, 30.0], dtype=np.float64)
    time_costs = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    stats = glb_budget_stats(
        glbs,
        scores,
        utility,
        byte_costs,
        (1, 2),
        (10.0, 25.0, 60.0),
        time_costs,
        (1.0, 3.0, 6.0),
        "max",
        2,
        0.98,
    )
    assert stats["utilityAtBytes"]["status"] == "available"
    assert stats["utilityAtTime"]["status"] == "available"
    assert stats["utilityAtBytes"]["rows"]["10.0"]["selectedBytes"] <= 10.0 + 1e-6
    assert stats["utilityAtBytes"]["rows"]["25.0"]["selectedBytes"] <= 25.0 + 1e-6
    assert stats["utilityAtBytes"]["rows"]["60.0"]["selectedBytes"] <= 60.0 + 1e-6
    no_time = glb_budget_stats(
        glbs,
        scores,
        utility,
        byte_costs,
        (1,),
        (10.0,),
        None,
        (),
        "noisy-or",
        2,
        0.98,
    )
    assert no_time["utilityAtTime"]["status"] == "not_available"
    result = SimpleNamespace(
        scores=np.asarray([0.8, 0.2], dtype=np.float32),
        utility_scores=np.asarray([0.7, 0.1], dtype=np.float32),
        download_scores=np.asarray([2.0, -1.0], dtype=np.float32),
    )
    assert _score_mode_result(result, "visibility-only")[1]["status"] == "implemented"
    assert _score_mode_result(result, "current-cascade")[0][0] > 0.5
    assert np.isclose(_score_mode_result(result, "visibility-gated")[0][0], np.float64(0.8 * 0.7))
    assert _score_mode_result(result, "independent-utility")[1]["status"] == "not_implemented"
    independent_result = SimpleNamespace(independent_utility_scores=np.asarray([0.6, 0.2], dtype=np.float32))
    assert _score_mode_result(independent_result, "independent-utility")[1]["status"] == "implemented"
    with tempfile.TemporaryDirectory() as temp_dir:
        time_index = Path(temp_dir) / "glb_costs.json"
        time_index.write_text(
            json.dumps(
                {
                    "schema": "neuralstreamweb3d-glb-cost-index-v1",
                    "entries": [
                        {"globalId": 0, "decodeMs": 2.0, "uploadMs": 3.0, "totalDecodeUploadMs": 5.0},
                        {"globalId": 1, "totalDecodeUploadMs": 7.0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        np.testing.assert_allclose(
            load_glb_time_costs(time_index, 2, np.asarray([0, 1], dtype=np.int64)),
            np.asarray([5.0, 7.0]),
        )

    class SelfTestRunner:
        name = "self-test-runner"
        kind = "fixture"
        information_level = "fixture"
        decision_mode = "fixture"
        world_aabbs = np.zeros((2, 6), dtype=np.float32)
        instance_to_glb = np.asarray([0, 1], dtype=np.int64)

        def score_batch(self, batch):
            del batch
            return SimpleNamespace(
                scores=np.asarray([0.8, 0.2], dtype=np.float32),
                utility_scores=np.asarray([0.7, 0.1], dtype=np.float32),
                download_scores=np.asarray([2.0, -1.0], dtype=np.float32),
                forward_ms=0.1,
                total_ms=0.2,
                diagnostics={},
            )

    summary = evaluate_runner(
        SelfTestRunner(),
        _SelfTestSplit(),
        np.asarray([0.5], dtype=np.float32),
        (1,),
        (10.0, 20.0),
        np.asarray([1.0, 2.0], dtype=np.float64),
        (1.0, 2.0),
        np.asarray([10.0, 20.0], dtype=np.float64),
        poses_per_batch=1,
        max_eval_poses=0,
        max_candidates_per_pose=0,
        seed=7,
        target_utility_recall=0.98,
        target_weighted_recall=0.99,
        score_modes=SCORE_MODES,
        aggregations=GLB_AGGREGATIONS,
        aggregation_top_k=2,
        sample_with_replacement=False,
        progress=False,
    )
    assert summary["thresholdRows"][0]["eval_pose_count"] == 1
    assert summary["scoreModes"]["independent-utility"]["status"] == "not_implemented"
    assert summary["scoreModes"]["current-cascade"]["aggregations"]["top-k"]["budgetPoseCount"] == 1
    with tempfile.TemporaryDirectory() as temp_dir:
        manifest = Path(temp_dir) / "thresholds.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": "neuralstreamweb3d-frozen-threshold-manifest-v1",
                    "selection": "thresholds selected on calibration only; test is evaluated once at the frozen value",
                    "thresholds": {"fixture": 0.5},
                    "provenance": {
                        "fixture": {
                            "protocol": "frozen_calibration_one_shot_test",
                            "testEvaluationCount": 1,
                            "frozenTestDigest": "fixture-digest",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        assert load_frozen_thresholds(manifest, ["fixture"], require_manifest=True) == {"fixture": 0.5}
        direct = Path(temp_dir) / "direct_mapping.json"
        direct.write_text(json.dumps({"thresholds": {"fixture": 0.5}}), encoding="utf-8")
        try:
            load_frozen_thresholds(direct, ["fixture"], require_manifest=True)
        except ValueError as exc:
            assert "frozen threshold manifest" in str(exc)
        else:
            raise AssertionError("Formal test accepted a direct threshold mapping without manifest provenance.")
    return {
        "status": "passed",
        "checked": [
            "strict byte/time budgets",
            "all four score modes with explicit independent-ranker gate",
            "all four GLB aggregations",
            "frozen threshold manifest loading",
            "complete pose-set evaluation fixture",
        ],
    }


def resolve_split(dataset: PoseCSRDataset, requested: str) -> tuple[str, str]:
    """Resolve validation aliases without silently inventing calibration data."""

    available = set(dataset.split_ids)
    if requested == "val":
        if "validation" in available:
            return "validation", "native_validation"
        if "val" in available:
            return "val", "legacy_val_alias"
    if requested == "validation" and "validation" not in available and "val" in available:
        return "val", "legacy_val_alias"
    if requested in available:
        return requested, "native"
    raise ValueError(
        f"Requested split {requested!r} is not present in dataset_meta.json. "
        f"Available split ids: {sorted(available)}. A calibration split must be materialized, not inferred from test."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified visibility and GLB download utility metrics.")
    parser.add_argument("--self-test", action="store_true", help="Run the model-free M7 fixture and exit.")
    parser.add_argument("--models", default="pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_hkust_fov66_best")
    parser.add_argument(
        "--learned-model-spec",
        action="append",
        default=[],
        help="Explicit dynamic learned runner: name|checkpoint|runtime_features|calibration summary; appended to --models.",
    )
    parser.add_argument(
        "--learned-aabb-ray-spec",
        action="append",
        default=[],
        help="Explicit M6 learned AABB-ray runner: name|checkpoint[|training summary].",
    )
    parser.add_argument(
        "--independent-ranker-spec",
        action="append",
        default=[],
        help="Explicit M7 independent RankNet utility runner: name|checkpoint.",
    )
    parser.add_argument("--dataset-dir", default=str(ROOT / "dataset/out/pose_csr_hkust_v3_viewcell_colorid_fov66"))
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument("--glb-time-index", default="", help="Optional explicit per-GLB decode/upload time JSON.")
    parser.add_argument("--output-dir", default=str(ROOT / "benchmark/out/m7_unified_download_scheduling"))
    parser.add_argument("--split", choices=["train", "val", "validation", "calibration", "test"], default="calibration")
    parser.add_argument("--budgets", default="50,100,200,384", help="GLB-count budgets.")
    parser.add_argument("--byte-budgets", default="1048576,5242880,10485760,20971520", help="Strict byte budgets.")
    parser.add_argument("--time-budgets-ms", default="", help="Strict time budgets; requires --glb-time-index.")
    parser.add_argument("--target-utility-recall", type=float, default=0.98)
    parser.add_argument("--target-weighted-recall", type=float, default=0.99)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--max-candidates-per-pose", type=int, default=0)
    parser.add_argument("--max-eval-poses", type=int, default=0, help="Train-only exploratory fixture limit.")
    parser.add_argument("--sample-with-replacement", action="store_true", help="Train-only exploratory sampling.")
    parser.add_argument("--score-modes", default=",".join(SCORE_MODES))
    parser.add_argument("--glb-aggregations", default=",".join(GLB_AGGREGATIONS))
    parser.add_argument("--aggregation-top-k", type=int, default=2)
    parser.add_argument("--frozen-threshold-file", default="", help="Required for formal test evaluation.")
    parser.add_argument("--allow-missing-glb-cost", action="store_true", help="Exploratory-only cost imputation.")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.self_test:
        print(json.dumps(run_self_test(), ensure_ascii=False, indent=2))
        return
    if args.poses_per_batch <= 0 or args.aggregation_top_k <= 0:
        parser.error("--poses-per-batch and --aggregation-top-k must be positive.")
    if args.max_candidates_per_pose < 0 or args.max_eval_poses < 0:
        parser.error("Exploratory limits must be non-negative.")
    if args.split in {"val", "validation", "calibration", "test"} and (args.sample_with_replacement or args.max_eval_poses > 0):
        parser.error("Validation/calibration/test evaluation must traverse the complete unique split without replacement.")
    if args.max_candidates_per_pose > 0:
        parser.error("Formal M7 evaluation does not permit candidate truncation; use the stored candidate set unchanged.")
    if args.split == "test":
        if not args.frozen_threshold_file:
            parser.error("Formal test evaluation requires --frozen-threshold-file; test threshold scanning is disabled.")
        if args.allow_missing_glb_cost:
            parser.error("Formal test evaluation cannot use imputed GLB byte costs.")
    if args.time_budgets_ms.strip() and not args.glb_time_index:
        parser.error("--time-budgets-ms requires --glb-time-index; bytes are never converted to time.")

    try:
        count_budgets = parse_int_list(args.budgets)
        byte_budgets = parse_float_list(args.byte_budgets)
        time_budgets_ms = parse_float_list(args.time_budgets_ms)
        score_modes = parse_choice_list(args.score_modes, SCORE_MODES, "score modes")
        aggregations = parse_choice_list(args.glb_aggregations, GLB_AGGREGATIONS, "GLB aggregations")
    except ValueError as exc:
        parser.error(str(exc))
    if not byte_budgets:
        parser.error("At least one byte budget is required for utility@bytes.")

    device = select_device(args.device)
    specs = selected_default_specs(args.models)
    try:
        dynamic_specs = parse_learned_model_specs(args.learned_model_spec)
        learned_aabb_specs = parse_learned_aabb_ray_specs(args.learned_aabb_ray_spec)
        independent_specs = parse_independent_ranker_specs(args.independent_ranker_spec)
    except ValueError as exc:
        parser.error(str(exc))
    dynamic_overlap = set(dynamic_specs).intersection(learned_aabb_specs)
    if dynamic_overlap:
        parser.error(f"learned model and learned AABB-ray specs duplicate names: {sorted(dynamic_overlap)}")
    dynamic_specs.update(learned_aabb_specs)
    dynamic_overlap = set(dynamic_specs).intersection(independent_specs)
    if dynamic_overlap:
        parser.error(f"learned and independent ranker specs duplicate names: {sorted(dynamic_overlap)}")
    dynamic_specs.update(independent_specs)
    overlap = set(specs).intersection(dynamic_specs)
    if overlap:
        parser.error(f"--learned-model-spec duplicates registered model names: {sorted(overlap)}")
    specs.update(dynamic_specs)
    if not specs:
        parser.error("--models must contain at least one model.")
    first = next(iter(specs.values()))
    if first.get("checkpoint"):
        checkpoint = torch.load(first["checkpoint"], map_location="cpu")
        num_instances = int(checkpoint["config"]["numInstances"])
    else:
        world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(args.runtime_meta)
        num_instances = int(world_aabbs.shape[0])
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=num_instances)
    resolved_split, split_resolution = resolve_split(dataset, args.split)
    split = dataset.split(resolved_split)
    model_names = list(specs.keys())
    frozen_thresholds = (
        load_frozen_thresholds(
            args.frozen_threshold_file,
            model_names,
            require_manifest=args.split == "test",
            dataset_dir=args.dataset_dir,
        )
        if args.frozen_threshold_file
        else {}
    )
    if args.split == "test" and not frozen_thresholds:
        raise RuntimeError("Internal error: formal test reached evaluation without frozen thresholds.")
    runners = {
        name: load_runner(name, specs[name], args.runtime_meta, device, dataset_dir=args.dataset_dir)
        for name in model_names
    }
    thresholds_for_model = {}
    for name in model_names:
        runner = runners[name]
        if name in frozen_thresholds:
            thresholds_for_model[name] = np.asarray([frozen_thresholds[name]], dtype=np.float32)
        elif str(getattr(runner, "decision_mode", "threshold")).startswith("fixed_"):
            # Fixed rules have no calibration degree of freedom.  Evaluate
            # their declared decision once instead of manufacturing a scan
            # that gives the bitset/keep-all rule a misleading workpoint.
            thresholds_for_model[name] = np.asarray([float(runner.threshold)], dtype=np.float32)
        else:
            thresholds_for_model[name] = threshold_grid()

    summaries: list[dict[str, Any]] = []
    time_cost_mode = "not_requested"
    for name, spec in specs.items():
        runner = runners[name]
        if runner.instance_to_glb.size == 0:
            raise ValueError(f"Runner {name} has no instance-to-GLB mapping.")
        num_glbs = int(np.max(runner.instance_to_glb)) + 1
        glb_byte_cost = load_glb_byte_costs(
            args.glb_index,
            args.glb_root,
            num_glbs,
            runner.instance_to_glb,
            allow_missing=args.allow_missing_glb_cost,
        )
        glb_time_cost = load_glb_time_costs(
            args.glb_time_index,
            num_glbs,
            runner.instance_to_glb,
            allow_missing=args.allow_missing_glb_cost,
        )
        if glb_time_cost is not None:
            time_cost_mode = "explicit_index" if not args.allow_missing_glb_cost else "imputed_exploratory"
        summary = evaluate_runner(
            runner,
            split,
            thresholds_for_model[name],
            count_budgets,
            byte_budgets,
            glb_time_cost,
            time_budgets_ms,
            glb_byte_cost,
            poses_per_batch=args.poses_per_batch,
            max_eval_poses=args.max_eval_poses,
            max_candidates_per_pose=args.max_candidates_per_pose,
            seed=args.seed,
            target_utility_recall=args.target_utility_recall,
            target_weighted_recall=args.target_weighted_recall,
            score_modes=score_modes,
            aggregations=aggregations,
            aggregation_top_k=args.aggregation_top_k,
            sample_with_replacement=args.sample_with_replacement,
            progress=True,
        )
        summaries.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_summary = summaries[0] if summaries else {}
    first_row = (first_summary.get("thresholdRows") or [None])[0] or {}
    evaluated = int(first_row.get("eval_pose_count", 0))
    threshold_mode = "frozen_one_shot" if args.split == "test" else ("frozen_manifest" if frozen_thresholds else "split_scan")
    payload = {
        "meta": {
            "created": datetime.now().isoformat(timespec="seconds"),
            "datasetDir": args.dataset_dir,
            "runtimeMeta": args.runtime_meta,
            "requestedSplit": args.split,
            "split": resolved_split,
            "splitResolution": split_resolution,
            "evalMode": "sampled_with_replacement" if args.sample_with_replacement else "all_unique_poses",
            "thresholdMode": threshold_mode,
            "frozenThresholdFile": args.frozen_threshold_file or None,
            "testEvaluationCount": 1 if args.split == "test" else None,
            "candidateSemanticsMode": "stored_candidate_set_strict",
            "candidateLimit": 0,
            "glbByteCostMode": "imputed_exploratory" if args.allow_missing_glb_cost else "strict_filesystem_bytes",
            "glbTimeCostMode": time_cost_mode,
            "evaluatedPoses": evaluated,
            "countBudgets": list(count_budgets),
            "byteBudgets": list(byte_budgets),
            "timeBudgetsMs": list(time_budgets_ms),
            "targetUtilityRecall": float(args.target_utility_recall),
            "targetWeightedRecall": float(args.target_weighted_recall),
            "thresholdSelectionRule": weighted_precision_selection_rule(args.target_weighted_recall),
            "utilityTeacher": UTILITY_TEACHER,
            "scoreModes": list(score_modes),
            "glbAggregations": list(aggregations),
            "aggregationTopK": int(args.aggregation_top_k),
            "primaryMetric": "visibility safety is reported separately; download ranking is compared at strict resource budgets",
        },
        "summaries": summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(output_dir / "summary.md", payload)
    print(json.dumps({"outputDir": str(output_dir), "evaluatedPoses": evaluated, "models": model_names}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
