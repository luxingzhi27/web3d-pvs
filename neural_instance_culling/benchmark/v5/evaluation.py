"""Run the frozen V5 calibration protocols over shared and LOSO bundles."""
from __future__ import annotations

from typing import Any, Iterable

from .calibration import CalibrationSelection, calibrate_source_global, calibrate_target
from .contracts import V5Run, validate_matrix
from .metrics import evaluate_scene, metric_projection


def _selection_summary(selection: CalibrationSelection) -> dict[str, Any]:
    payload = selection.to_dict()
    payload.pop("threshold_rows", None)
    return payload


def _result_row(
    *,
    run: V5Run,
    scene: str,
    split: str,
    threshold_mode: str,
    selection: CalibrationSelection,
    metrics: dict[str, Any],
    held_out_scene: str | None = None,
) -> dict[str, Any]:
    return {
        "protocol": run.protocol,
        "variant": run.variant,
        "seed": int(run.seed),
        "scene": scene,
        "held_out_scene": held_out_scene,
        "split": split,
        "threshold_mode": threshold_mode,
        "threshold": float(selection.threshold),
        "qualification": selection.status,
        "mean_target_met": bool(selection.mean_target_met),
        "confidence_target_met": bool(selection.confidence_target_met),
        "selection_split": selection.selection_split,
        "selection_test_read": False,
        "calibration": _selection_summary(selection),
        "metrics": dict(metrics),
        "reported_metrics": metric_projection(metrics),
        "test_read": split == "test",
    }


def _check_evaluation_split(split: str) -> str:
    value = str(split)
    if value not in {"validation", "test"}:
        raise ValueError("frozen V5 evaluation split must be validation or test")
    return value


def evaluate_shared_matrix(
    runs: Iterable[V5Run],
    *,
    evaluation_split: str = "validation",
    target_weighted_recall: float = 0.99,
    calibration_bootstrap_replicates: int = 10_000,
    evaluation_bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    expected_scenes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate one shared checkpoint/seed on all five scenes.

    Shared in-domain calibration is scene-local: each scene uses its own
    calibration split, while the checkpoint and variant remain identical over
    the five scene bundles.  The returned rows are later reduced with a
    scene-equal summary rather than pooling candidate observations.
    """
    split = _check_evaluation_split(evaluation_split)
    materialized = validate_matrix(
        runs,
        protocol="shared",
        expected_scenes=expected_scenes,
    )
    rows: list[dict[str, Any]] = []
    for run in sorted(materialized, key=lambda item: (item.variant, item.seed)):
        for scene in sorted(run.scenes):
            bundle = run.scenes[scene]
            if split not in bundle.splits:
                raise ValueError(f"shared run {run.variant}/{run.seed} has no {split} rows for {scene}")
            selection = calibrate_target(
                bundle.splits["calibration"],
                scene=scene,
                target_weighted_recall=target_weighted_recall,
                bootstrap_replicates=calibration_bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + run.seed,
            )
            metrics = evaluate_scene(
                bundle.splits[split],
                selection.threshold,
                bootstrap_replicates=evaluation_bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + run.seed + 10_000,
            )
            rows.append(
                _result_row(
                    run=run,
                    scene=scene,
                    split=split,
                    threshold_mode="target_calibrated",
                    selection=selection,
                    metrics=metrics,
                )
            )
    return rows


def evaluate_loso_matrix(
    runs: Iterable[V5Run],
    *,
    evaluation_split: str = "validation",
    target_weighted_recall: float = 0.99,
    calibration_bootstrap_replicates: int = 10_000,
    evaluation_bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    expected_scenes: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every LOSO fold with both threshold provenance modes.

    ``source_global`` reads source-scene calibration rows only.  The held-out
    calibration rows are read separately for ``target_calibrated`` and never
    alter the source-global threshold or its result.
    """
    split = _check_evaluation_split(evaluation_split)
    materialized = validate_matrix(
        runs,
        protocol="loso",
        expected_scenes=expected_scenes,
    )
    rows: list[dict[str, Any]] = []
    for run in sorted(
        materialized,
        key=lambda item: (item.variant, item.seed, str(item.held_out_scene)),
    ):
        assert run.held_out_scene is not None
        source_calibration = {
            scene: run.scenes[scene].splits["calibration"]
            for scene in run.source_scenes
        }
        source_selection = calibrate_source_global(
            source_calibration,
            target_weighted_recall=target_weighted_recall,
            bootstrap_replicates=calibration_bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + run.seed,
        )
        target_bundle = run.scenes[run.held_out_scene]
        if split not in target_bundle.splits:
            raise ValueError(
                f"LOSO run {run.variant}/{run.seed}/{run.held_out_scene} has no {split} rows"
            )
        source_metrics = evaluate_scene(
            target_bundle.splits[split],
            source_selection.threshold,
            bootstrap_replicates=evaluation_bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + run.seed + 10_000,
        )
        rows.append(
            _result_row(
                run=run,
                scene=run.held_out_scene,
                held_out_scene=run.held_out_scene,
                split=split,
                threshold_mode="source_global",
                selection=source_selection,
                metrics=source_metrics,
            )
        )
        target_selection = calibrate_target(
            target_bundle.splits["calibration"],
            scene=run.held_out_scene,
            target_weighted_recall=target_weighted_recall,
            bootstrap_replicates=calibration_bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + run.seed + 20_000,
        )
        target_metrics = evaluate_scene(
            target_bundle.splits[split],
            target_selection.threshold,
            bootstrap_replicates=evaluation_bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + run.seed + 30_000,
        )
        rows.append(
            _result_row(
                run=run,
                scene=run.held_out_scene,
                held_out_scene=run.held_out_scene,
                split=split,
                threshold_mode="target_calibrated",
                selection=target_selection,
                metrics=target_metrics,
            )
        )
    return rows


__all__ = ["evaluate_loso_matrix", "evaluate_shared_matrix"]
