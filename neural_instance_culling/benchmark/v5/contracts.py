"""Strict input and experiment-matrix contracts for GCOF-PVS V5.

V5 consumes one explicit score bundle.  A pose row contains the candidate
instances, one score per candidate, a binary visibility label, and the raw
``visible_weights`` used for weighted recall.  The contract intentionally
does not accept legacy V4 summaries: selecting a threshold from an aggregate
row without its pose-aligned scores would make the LOSO provenance unclear.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCORE_BUNDLE_SCHEMA = "gcof-pvs-v5-score-bundle-v1"
RESULT_MATRIX_SCHEMA = "gcof-pvs-v5-result-matrix-v1"
ALL_VARIANTS = (
    "FULL",
    "GEOMETRY_FIELD",
    "GENERIC_RELATION_28",
    "PBCE_OBJECTIVE",
)
LOSO_VARIANTS = (
    "FULL",
    "GEOMETRY_FIELD",
    "GENERIC_RELATION_28",
)
PROTOCOLS = ("shared", "loso")
SPLITS = ("calibration", "validation", "test")
EXPECTED_SCENE_COUNT = 5
EXPECTED_SEED_COUNT = 3


def _as_json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _as_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_as_json_value(item) for item in value]
    return value


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class PoseRecord:
    """Scores and labels for one view-cell pose.

    ``scores`` are continuous model scores and are compared with ``>=`` at a
    frozen threshold.  They are deliberately not forced into ``[0, 1]`` so
    that the evaluator remains faithful to a V5 score field whose numerical
    range is declared by its model export metadata.
    """

    pose_id: str
    split: str
    candidate_ids: np.ndarray
    scores: np.ndarray
    targets: np.ndarray
    visible_weights: np.ndarray
    scene: str | None = None

    def __post_init__(self) -> None:
        split = str(self.split)
        if split not in SPLITS:
            raise ValueError(f"unsupported V5 split: {split!r}")
        if not str(self.pose_id):
            raise ValueError("pose_id must be non-empty")
        candidates = np.asarray(self.candidate_ids, dtype=np.int64).reshape(-1)
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        targets = np.asarray(self.targets, dtype=np.uint8).reshape(-1)
        weights = np.asarray(self.visible_weights, dtype=np.float64).reshape(-1)
        if not (candidates.size == scores.size == targets.size == weights.size):
            raise ValueError("candidate_ids, scores, targets, and visible_weights must have equal lengths")
        if candidates.size and int(candidates.min()) < 0:
            raise ValueError("candidate_ids must be non-negative")
        if candidates.size != np.unique(candidates).size:
            raise ValueError(f"candidate_ids contain duplicates at pose {self.pose_id!r}")
        if not bool(np.isfinite(scores).all()):
            raise ValueError(f"scores contain non-finite values at pose {self.pose_id!r}")
        if bool(np.any(targets > 1)):
            raise ValueError(f"targets must be binary at pose {self.pose_id!r}")
        if not bool(np.isfinite(weights).all()) or bool(np.any(weights < 0.0)):
            raise ValueError(f"visible_weights must be finite and non-negative at pose {self.pose_id!r}")
        if self.scene is not None and not str(self.scene):
            raise ValueError("scene must be non-empty when provided")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "visible_weights", weights)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        split: str,
        scene: str | None = None,
    ) -> "PoseRecord":
        """Parse the sole V5 JSON pose-row spelling.

        The parent split supplies ``split``; this prevents a row from being
        silently reassigned while a bundle is loaded.
        """
        row = _require_mapping(value, "pose row")
        required = ("poseId", "candidateIds", "scores", "targets", "visibleWeights")
        missing = [field for field in required if field not in row]
        if missing:
            raise ValueError(f"pose row is missing V5 fields: {', '.join(missing)}")
        return cls(
            pose_id=str(row["poseId"]),
            split=str(split),
            candidate_ids=np.asarray(row["candidateIds"], dtype=np.int64),
            scores=np.asarray(row["scores"], dtype=np.float64),
            targets=np.asarray(row["targets"], dtype=np.uint8),
            visible_weights=np.asarray(row["visibleWeights"], dtype=np.float64),
            scene=scene,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "poseId": self.pose_id,
            "candidateIds": self.candidate_ids.tolist(),
            "scores": self.scores.tolist(),
            "targets": self.targets.tolist(),
            "visibleWeights": self.visible_weights.tolist(),
        }


@dataclass(frozen=True)
class SceneScores:
    """All available score rows for one scene in one model run."""

    scene: str
    splits: Mapping[str, tuple[PoseRecord, ...]]

    def __post_init__(self) -> None:
        scene = str(self.scene)
        if not scene:
            raise ValueError("scene must be non-empty")
        normalized: dict[str, tuple[PoseRecord, ...]] = {}
        for split, rows in self.splits.items():
            if split not in SPLITS:
                raise ValueError(f"unsupported V5 split in {scene}: {split!r}")
            materialized = tuple(rows)
            if not materialized:
                raise ValueError(f"V5 {split} split is empty for scene {scene!r}")
            if any(row.split != split for row in materialized):
                raise ValueError(f"pose split disagrees with SceneScores.{split} for {scene!r}")
            if any(row.scene not in (None, scene) for row in materialized):
                raise ValueError(f"pose scene disagrees with SceneScores for {scene!r}")
            pose_ids = [row.pose_id for row in materialized]
            if len(pose_ids) != len(set(pose_ids)):
                raise ValueError(f"duplicate pose IDs in {scene!r} {split} split")
            normalized[split] = materialized
        if "calibration" not in normalized:
            raise ValueError(f"scene {scene!r} has no calibration split")
        object.__setattr__(self, "scene", scene)
        object.__setattr__(self, "splits", normalized)

    @classmethod
    def from_mapping(cls, scene: str, value: Mapping[str, Any]) -> "SceneScores":
        payload = _require_mapping(value, f"scene {scene!r}")
        parsed: dict[str, tuple[PoseRecord, ...]] = {}
        for split in SPLITS:
            if split not in payload:
                continue
            rows = payload[split]
            if not isinstance(rows, list):
                raise ValueError(f"scene {scene!r} split {split!r} must be a JSON array")
            parsed[split] = tuple(
                PoseRecord.from_mapping(row, split=split, scene=scene) for row in rows
            )
        return cls(scene=scene, splits=parsed)

    def to_mapping(self) -> dict[str, Any]:
        return {
            split: [row.to_mapping() for row in rows]
            for split, rows in self.splits.items()
        }


@dataclass(frozen=True)
class V5Run:
    """One checkpoint/seed score bundle.

    A shared run contains the same model evaluated on all five scenes.  A LOSO
    run also contains all five geometric score tables, but identifies one
    held-out scene and the four source scenes explicitly.
    """

    protocol: str
    variant: str
    seed: int
    scenes: Mapping[str, SceneScores]
    held_out_scene: str | None = None
    source_scenes: tuple[str, ...] = ()
    test_read: bool = False

    def __post_init__(self) -> None:
        protocol = str(self.protocol)
        variant = str(self.variant)
        if protocol not in PROTOCOLS:
            raise ValueError(f"unsupported V5 protocol: {protocol!r}")
        allowed = ALL_VARIANTS if protocol == "shared" else LOSO_VARIANTS
        if variant not in allowed:
            raise ValueError(f"variant {variant!r} is not registered for {protocol}")
        if bool(self.test_read):
            raise ValueError("a V5 score bundle marked test_read=true cannot select a model or threshold")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        normalized = {str(scene): value for scene, value in self.scenes.items()}
        if any(scene != bundle.scene for scene, bundle in normalized.items()):
            raise ValueError("V5 scene mapping key disagrees with SceneScores.scene")
        if protocol == "shared":
            if self.held_out_scene is not None or self.source_scenes:
                raise ValueError("shared runs cannot declare LOSO source/held-out scenes")
        else:
            held_out = str(self.held_out_scene or "")
            sources = tuple(str(scene) for scene in self.source_scenes)
            if not held_out or len(sources) != 4 or len(set(sources)) != 4:
                raise ValueError("a LOSO run requires one held-out scene and four unique source scenes")
            if held_out in sources:
                raise ValueError("LOSO held-out scene cannot also be a source scene")
            if set(normalized) != {held_out, *sources}:
                raise ValueError("LOSO scene tables must be exactly held-out plus four source scenes")
            object.__setattr__(self, "held_out_scene", held_out)
            object.__setattr__(self, "source_scenes", sources)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "variant", variant)
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "scenes", normalized)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "V5Run":
        payload = _require_mapping(value, "V5 score bundle")
        if payload.get("schema") != SCORE_BUNDLE_SCHEMA:
            raise ValueError(f"V5 score bundle schema must be {SCORE_BUNDLE_SCHEMA!r}")
        assert_test_free_selection(payload)
        scenes_value = payload.get("scenes")
        if not isinstance(scenes_value, Mapping):
            raise ValueError("V5 score bundle must contain a scenes object")
        scenes = {
            str(scene): SceneScores.from_mapping(str(scene), _require_mapping(scene_value, f"scene {scene!r}"))
            for scene, scene_value in scenes_value.items()
        }
        source_value = payload.get("sourceScenes", [])
        if not isinstance(source_value, list):
            raise ValueError("sourceScenes must be a JSON array")
        return cls(
            protocol=str(payload.get("protocol", "")),
            variant=str(payload.get("variant", "")),
            seed=int(payload.get("seed", -1)),
            scenes=scenes,
            held_out_scene=(None if payload.get("heldOutScene") is None else str(payload["heldOutScene"])),
            source_scenes=tuple(str(scene) for scene in source_value),
            test_read=bool(payload.get("testRead", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "V5Run":
        source = Path(path).resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_mapping(_require_mapping(payload, str(source)))

    def to_mapping(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SCORE_BUNDLE_SCHEMA,
            "protocol": self.protocol,
            "variant": self.variant,
            "seed": self.seed,
            "scenes": {scene: bundle.to_mapping() for scene, bundle in self.scenes.items()},
            "testRead": False,
        }
        if self.protocol == "loso":
            payload["heldOutScene"] = self.held_out_scene
            payload["sourceScenes"] = list(self.source_scenes)
        return payload


def assert_test_free_selection(payload: Mapping[str, Any]) -> None:
    """Reject explicit test-tainted model or threshold provenance."""
    if payload.get("testRead") is True:
        raise ValueError("V5 model/threshold selection input is test-tainted")
    for field in ("modelSelectionSplit", "thresholdSelectionSplit", "selectionSplit"):
        if str(payload.get(field, "")).lower() == "test":
            raise ValueError(f"V5 {field} cannot be test")
    for field in ("selectedFromTest", "modelSelectedFromTest", "thresholdSelectedFromTest"):
        if payload.get(field) is True:
            raise ValueError(f"V5 {field} must be false")


def _normalize_runs(runs: Iterable[V5Run]) -> tuple[V5Run, ...]:
    materialized = tuple(runs)
    if not materialized:
        raise ValueError("V5 result matrix is empty")
    return materialized


def validate_matrix(
    runs: Iterable[V5Run],
    *,
    protocol: str,
    expected_scenes: Sequence[str] | None = None,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
) -> tuple[V5Run, ...]:
    """Validate the frozen four-variant shared / three-variant LOSO matrix.

    Shared requires four variants x three seeds x five scenes.  LOSO requires
    three variants x three seeds x five held-out folds.  A caller may provide
    different scene names for a fixture, but the five-scene cardinality and
    completeness rules remain fixed.
    """
    normalized_protocol = str(protocol)
    if normalized_protocol not in PROTOCOLS:
        raise ValueError(f"unsupported V5 protocol: {normalized_protocol!r}")
    materialized = _normalize_runs(runs)
    if any(run.protocol != normalized_protocol for run in materialized):
        raise ValueError("V5 matrix contains a run from another protocol")
    scene_set = set(expected_scenes) if expected_scenes is not None else set().union(*(run.scenes for run in materialized))
    if len(scene_set) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"V5 {normalized_protocol} matrix must contain exactly five scenes")
    if expected_scenes is not None and len(set(expected_scenes)) != EXPECTED_SCENE_COUNT:
        raise ValueError("expected_scenes must contain five unique names")
    required_variants = set(ALL_VARIANTS if normalized_protocol == "shared" else LOSO_VARIANTS)
    observed_variants = {run.variant for run in materialized}
    if observed_variants != required_variants:
        raise ValueError(
            f"V5 {normalized_protocol} matrix variants are {sorted(observed_variants)!r}; "
            f"expected {sorted(required_variants)!r}"
        )
    by_run_key: dict[tuple[str, int, str | None], list[V5Run]] = {}
    for run in materialized:
        if set(run.scenes) != scene_set:
            raise ValueError(f"run {run.variant}/{run.seed} does not cover the exact five scenes")
        for bundle in run.scenes.values():
            if "calibration" not in bundle.splits:
                raise ValueError(f"run {run.variant}/{run.seed} has a scene without calibration")
        held_out_key = None if normalized_protocol == "shared" else str(run.held_out_scene)
        key = (run.variant, run.seed, held_out_key)
        by_run_key.setdefault(key, []).append(run)
    seeds_by_variant: dict[str, set[int]] = {variant: set() for variant in required_variants}
    for (variant, seed, _held_out), members in by_run_key.items():
        if len(members) != 1:
            raise ValueError(f"duplicate V5 run for {variant}/{seed}")
        seeds_by_variant[variant].add(seed)
        run = members[0]
        if normalized_protocol == "shared":
            if any("validation" not in bundle.splits for bundle in run.scenes.values()):
                raise ValueError(f"shared run {variant}/{seed} must contain validation rows")
        else:
            if run.held_out_scene not in scene_set or set(run.source_scenes) != scene_set - {run.held_out_scene}:
                raise ValueError(f"LOSO run {variant}/{seed} has an invalid source/held-out partition")
            if "calibration" not in run.scenes[run.held_out_scene].splits:
                raise ValueError(f"LOSO target scene {run.held_out_scene!r} has no calibration rows")
            if "validation" not in run.scenes[run.held_out_scene].splits:
                raise ValueError(f"LOSO target scene {run.held_out_scene!r} has no validation rows")
    seed_sets = list(seeds_by_variant.values())
    if any(seed_set != seed_sets[0] for seed_set in seed_sets[1:]):
        raise ValueError("V5 variants do not share the same seed set")
    if len(seed_sets[0]) != int(expected_seed_count):
        raise ValueError(
            f"V5 {normalized_protocol} matrix requires {int(expected_seed_count)} seeds per variant"
        )
    if normalized_protocol == "shared":
        return materialized
    held_out_by_variant_seed: dict[tuple[str, int], set[str]] = {}
    for run in materialized:
        held_out_by_variant_seed.setdefault((run.variant, run.seed), set()).add(str(run.held_out_scene))
    for key, held_out in held_out_by_variant_seed.items():
        if held_out != scene_set:
            raise ValueError(f"LOSO fold coverage is incomplete for {key[0]}/{key[1]}")
    return materialized


__all__ = [
    "ALL_VARIANTS",
    "EXPECTED_SCENE_COUNT",
    "EXPECTED_SEED_COUNT",
    "LOSO_VARIANTS",
    "PROTOCOLS",
    "RESULT_MATRIX_SCHEMA",
    "SCORE_BUNDLE_SCHEMA",
    "SPLITS",
    "PoseRecord",
    "SceneScores",
    "V5Run",
    "assert_test_free_selection",
    "validate_matrix",
]
