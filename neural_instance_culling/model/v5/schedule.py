"""Deterministic scene-balanced V5 training and parameter-scan schedules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class StepAssignment:
    scene_id: str
    source_kind: str
    scene_update: int
    yaw_quarter_turns: int


@dataclass(frozen=True)
class ScanConfiguration:
    name: str
    model_learning_rate: float
    dual_learning_rate: float
    updates: int = 12_000


def parameter_scan_matrix() -> tuple[ScanConfiguration, ...]:
    return tuple(
        ScanConfiguration(
            name=f"model_lr{model_lr:.0e}_dual_lr{dual_lr:.0e}".replace("+", ""),
            model_learning_rate=model_lr,
            dual_learning_rate=dual_lr,
        )
        for model_lr in (1e-4, 2e-4, 4e-4)
        for dual_lr in (1e-3, 3e-3)
    )


def balanced_step_schedule(
    real_scene_ids: Sequence[str],
    synthetic_scene_ids: Sequence[str],
    *,
    real_updates_per_scene: int | None = None,
    total_updates: int | None = None,
    seed: int,
) -> Iterator[StepAssignment]:
    """Yield one-scene steps with a fixed two-real/one-synthetic cadence."""

    real = tuple(str(value) for value in real_scene_ids)
    synthetic = tuple(str(value) for value in synthetic_scene_ids)
    if not real or len(set(real)) != len(real):
        raise ValueError("real scene IDs must be non-empty and unique")
    if not synthetic or len(set(synthetic)) != len(synthetic):
        raise ValueError("synthetic scene IDs must be non-empty and unique")
    if (real_updates_per_scene is None) == (total_updates is None):
        raise ValueError("choose exactly one of real_updates_per_scene or total_updates")
    if real_updates_per_scene is not None and real_updates_per_scene <= 0:
        raise ValueError("real_updates_per_scene must be positive")
    if total_updates is not None and total_updates <= 0:
        raise ValueError("total_updates must be positive")

    if real_updates_per_scene is not None:
        real_step_limit = len(real) * int(real_updates_per_scene)
        overall_limit = real_step_limit + real_step_limit // 2
    else:
        overall_limit = int(total_updates)
        real_step_limit = overall_limit - overall_limit // 3

    rng = np.random.default_rng(int(seed))
    synthetic_order = np.arange(len(synthetic), dtype=np.int64)
    rng.shuffle(synthetic_order)
    synthetic_cursor = 0
    update_counts = {scene_id: 0 for scene_id in (*real, *synthetic)}
    real_steps = 0
    step = 0
    while step < overall_limit:
        for _ in range(2):
            if step >= overall_limit or real_steps >= real_step_limit:
                break
            scene_id = real[real_steps % len(real)]
            count = update_counts[scene_id]
            yield StepAssignment(scene_id, "real", count, count % 4)
            update_counts[scene_id] = count + 1
            real_steps += 1
            step += 1
        if step >= overall_limit:
            break
        if synthetic_cursor and synthetic_cursor % len(synthetic_order) == 0:
            rng.shuffle(synthetic_order)
        scene_id = synthetic[int(synthetic_order[synthetic_cursor % len(synthetic_order)])]
        synthetic_cursor += 1
        count = update_counts[scene_id]
        yield StepAssignment(scene_id, "synthetic", count, count % 4)
        update_counts[scene_id] = count + 1
        step += 1


__all__ = [
    "ScanConfiguration",
    "StepAssignment",
    "balanced_step_schedule",
    "parameter_scan_matrix",
]
