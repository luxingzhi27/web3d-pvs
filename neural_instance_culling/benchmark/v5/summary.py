"""Scene-equal result summaries for the V5 shared and LOSO matrices."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import (
    ALL_VARIANTS,
    EXPECTED_SCENE_COUNT,
    EXPECTED_SEED_COUNT,
    LOSO_VARIANTS,
    RESULT_MATRIX_SCHEMA,
)
from .metrics import METRIC_FIELDS


def _finite_metric(value: Any, *, label: str) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{label} is not finite")
    return numeric


def _compact_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("V5 result row has no metrics object")
    required = {
        field: _finite_metric(metrics.get(field), label=f"metrics.{field}")
        for field in METRIC_FIELDS
    }
    result = {
        key: row.get(key)
        for key in (
            "protocol",
            "variant",
            "seed",
            "scene",
            "held_out_scene",
            "split",
            "threshold_mode",
            "threshold",
            "qualification",
            "mean_target_met",
            "confidence_target_met",
            "selection_split",
            "selection_test_read",
            "test_read",
        )
    }
    result["metrics"] = required
    return result


def _validate_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    protocol: str,
    expected_scenes: Sequence[str] | None,
    expected_seed_count: int,
) -> list[dict[str, Any]]:
    materialized = [_compact_row(row) for row in rows]
    if not materialized:
        raise ValueError("V5 result matrix is empty")
    if any(str(row.get("protocol")) != protocol for row in materialized):
        raise ValueError("V5 result rows mix protocols")
    scene_set = set(expected_scenes) if expected_scenes is not None else {
        str(row.get("scene")) for row in materialized
    }
    if len(scene_set) != EXPECTED_SCENE_COUNT:
        raise ValueError("V5 result matrix must contain exactly five scenes")
    if expected_scenes is not None and len(set(expected_scenes)) != EXPECTED_SCENE_COUNT:
        raise ValueError("expected_scenes must contain five unique names")
    variants = set(ALL_VARIANTS if protocol == "shared" else LOSO_VARIANTS)
    if {str(row.get("variant")) for row in materialized} != variants:
        raise ValueError(f"V5 {protocol} result matrix has an incomplete variant set")
    splits = {str(row.get("split")) for row in materialized}
    if len(splits) != 1 or splits - {"validation", "test"}:
        raise ValueError("V5 result rows must all be one frozen validation or test split")
    if any(row.get("selection_split") != "calibration" for row in materialized):
        raise ValueError("V5 result threshold selection must be calibration-only")
    if any(row.get("selection_test_read") is True for row in materialized):
        raise ValueError("V5 result threshold selection is test-tainted")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in materialized:
        try:
            seed = int(row["seed"])
        except (TypeError, ValueError) as exc:
            raise ValueError("V5 result seed is invalid") from exc
        row["seed"] = seed
        scene = str(row.get("scene"))
        if scene not in scene_set:
            raise ValueError(f"result row refers to an unexpected scene: {scene!r}")
        if protocol == "shared":
            if row.get("threshold_mode") != "target_calibrated":
                raise ValueError("shared V5 rows must use target_calibrated scene-local calibration")
            key = (str(row["variant"]), seed)
            groups.setdefault(key, []).append(row)
        else:
            if row.get("threshold_mode") not in {"source_global", "target_calibrated"}:
                raise ValueError("LOSO V5 rows must declare source_global or target_calibrated")
            held_out = str(row.get("held_out_scene"))
            if held_out != scene or held_out not in scene_set:
                raise ValueError("LOSO result held_out_scene must equal its evaluated scene")
            key = (str(row["variant"]), seed)
            groups.setdefault(key, []).append(row)
    seeds_by_variant: dict[str, set[int]] = {variant: set() for variant in variants}
    for (variant, seed), members in groups.items():
        seeds_by_variant[variant].add(seed)
        if len(seeds_by_variant[variant]) > int(expected_seed_count):
            raise ValueError(f"too many V5 seeds for {variant}")
        if protocol == "shared":
            if len(members) != EXPECTED_SCENE_COUNT or {str(row["scene"]) for row in members} != scene_set:
                raise ValueError(f"shared result {variant}/{seed} does not cover five scenes exactly")
        else:
            if len(members) != 2 * EXPECTED_SCENE_COUNT:
                raise ValueError(f"LOSO result {variant}/{seed} does not cover both modes and five folds")
            by_mode = {
                mode: {str(row["scene"]) for row in members if row["threshold_mode"] == mode}
                for mode in ("source_global", "target_calibrated")
            }
            if any(scene_values != scene_set for scene_values in by_mode.values()):
                raise ValueError(f"LOSO result {variant}/{seed} has incomplete fold coverage")
    if any(len(seed_values) != int(expected_seed_count) for seed_values in seeds_by_variant.values()):
        raise ValueError(
            f"V5 {protocol} result matrix requires {int(expected_seed_count)} seeds per variant"
        )
    if len({frozenset(values) for values in seeds_by_variant.values()}) != 1:
        raise ValueError("V5 result variants do not share the same seed set")
    return materialized


def _stats(values: Sequence[float | None]) -> dict[str, Any]:
    numeric = np.asarray([float(value) for value in values if value is not None], dtype=np.float64)
    if numeric.size == 0:
        return {"mean": None, "std": None, "count": 0}
    return {
        "mean": float(np.mean(numeric)),
        "std": float(np.std(numeric, ddof=1)) if numeric.size > 1 else 0.0,
        "count": int(numeric.size),
    }


def _metric_values(rows: Sequence[Mapping[str, Any]], metric: str) -> list[float | None]:
    return [row["metrics"].get(metric) for row in rows]


def _scene_equal_details(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenes: Sequence[str],
    seeds: Sequence[int],
    mode: str | None = None,
) -> dict[str, Any]:
    scene_rows: dict[str, list[Mapping[str, Any]]] = {scene: [] for scene in scenes}
    for row in rows:
        if mode is not None and row.get("threshold_mode") != mode:
            continue
        scene_rows[str(row["scene"])].append(row)
    per_scene: dict[str, Any] = {}
    for scene in scenes:
        members = scene_rows[scene]
        if {int(row["seed"]) for row in members} != set(seeds):
            raise ValueError(f"scene-equal summary has incomplete seed coverage for {scene!r}")
        per_scene[scene] = {
            metric: _stats(_metric_values(members, metric))
            for metric in METRIC_FIELDS
        }
    scene_means: dict[str, list[float | None]] = {metric: [] for metric in METRIC_FIELDS}
    for scene in scenes:
        members = scene_rows[scene]
        for metric in METRIC_FIELDS:
            values = _metric_values(members, metric)
            scene_means[metric].append(
                float(np.mean([float(value) for value in values if value is not None]))
                if any(value is not None for value in values) else None
            )
    seed_means: dict[str, list[float | None]] = {metric: [] for metric in METRIC_FIELDS}
    for seed in seeds:
        seed_rows = [row for row in rows if int(row["seed"]) == int(seed) and (mode is None or row.get("threshold_mode") == mode)]
        if len(seed_rows) != len(scenes):
            raise ValueError(f"scene-equal summary has incomplete scene coverage for seed {seed}")
        for metric in METRIC_FIELDS:
            values = _metric_values(seed_rows, metric)
            seed_means[metric].append(
                float(np.mean([float(value) for value in values if value is not None]))
                if any(value is not None for value in values) else None
            )
    return {
        "aggregation": "scene_equal_mean; seed_equal mean reported separately",
        "per_scene_seed_stats": per_scene,
        "scene_equal": {metric: _stats(values) for metric, values in scene_means.items()},
        "seed_equal": {metric: _stats(values) for metric, values in seed_means.items()},
    }


def summarize_shared(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_scenes: Sequence[str] | None = None,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> dict[str, Any]:
    """Summarize the four-variant shared five-scene matrix scene-equally."""
    validated = _validate_rows(
        rows,
        protocol="shared",
        expected_scenes=expected_scenes,
        expected_seed_count=expected_seed_count,
    )
    scenes = tuple(sorted({str(row["scene"]) for row in validated}))
    seeds = tuple(sorted({int(row["seed"]) for row in validated}))
    variants = {
        variant: _scene_equal_details(
            [row for row in validated if row["variant"] == variant],
            scenes=scenes,
            seeds=seeds,
        )
        for variant in ALL_VARIANTS
    }
    return {
        "schema": RESULT_MATRIX_SCHEMA,
        "protocol": "shared",
        "split": validated[0]["split"],
        "selection_split": "calibration",
        "test_read": validated[0]["split"] == "test",
        "scene_count": len(scenes),
        "seed_count": len(seeds),
        "scenes": list(scenes),
        "seeds": list(seeds),
        "variants": list(ALL_VARIANTS),
        "metrics": list(METRIC_FIELDS),
        "rows": validated,
        "summary": variants,
    }


def summarize_loso(
    rows: Iterable[Mapping[str, Any]],
    *,
    expected_scenes: Sequence[str] | None = None,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> dict[str, Any]:
    """Summarize both LOSO threshold modes with equal held-out-scene weight."""
    validated = _validate_rows(
        rows,
        protocol="loso",
        expected_scenes=expected_scenes,
        expected_seed_count=expected_seed_count,
    )
    scenes = tuple(sorted({str(row["scene"]) for row in validated}))
    seeds = tuple(sorted({int(row["seed"]) for row in validated}))
    variants = {
        variant: {
            mode: _scene_equal_details(
                [row for row in validated if row["variant"] == variant],
                scenes=scenes,
                seeds=seeds,
                mode=mode,
            )
            for mode in ("source_global", "target_calibrated")
        }
        for variant in LOSO_VARIANTS
    }
    return {
        "schema": RESULT_MATRIX_SCHEMA,
        "protocol": "loso",
        "split": validated[0]["split"],
        "selection_split": "calibration",
        "test_read": validated[0]["split"] == "test",
        "scene_count": len(scenes),
        "seed_count": len(seeds),
        "scenes": list(scenes),
        "seeds": list(seeds),
        "variants": list(LOSO_VARIANTS),
        "threshold_modes": ["source_global", "target_calibrated"],
        "metrics": list(METRIC_FIELDS),
        "rows": validated,
        "summary": variants,
    }


def write_summary(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON result matrix without allowing non-finite values."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["summarize_loso", "summarize_shared", "write_summary"]
