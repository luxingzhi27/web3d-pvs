"""Strict manifest contracts for the columnar GCOF-PVS V5 benchmark."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .score_bundle import BUNDLE_SCHEMA, SIDECAR_SPLITS, SceneScores


SCORE_BUNDLE_SCHEMA = BUNDLE_SCHEMA
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
SPLITS = SIDECAR_SPLITS
EXPECTED_SCENE_COUNT = 5
EXPECTED_SEED_COUNT = 3


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _resolve(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _read_manifest(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"V5 score bundle manifest is not valid JSON: {path}") from exc
    return _require_mapping(payload, "V5 score bundle manifest")


def _validate_checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint = _require_mapping(payload.get("checkpoint"), "checkpoint")
    required = ("path", "schema", "protocol", "variant", "seed", "testRead")
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint metadata is missing: {', '.join(missing)}")
    if not isinstance(checkpoint["path"], str) or not checkpoint["path"]:
        raise ValueError("checkpoint.path must be non-empty")
    if not isinstance(checkpoint["schema"], str) or not checkpoint["schema"]:
        raise ValueError("checkpoint.schema must be non-empty")
    if not isinstance(checkpoint["seed"], int) or checkpoint["seed"] < 0:
        raise ValueError("checkpoint.seed must be a non-negative integer")
    if checkpoint["testRead"] is not False:
        raise ValueError("a V5 training checkpoint must declare testRead=false")
    return dict(checkpoint)


@dataclass
class V5Run:
    """One checkpoint manifest whose arrays remain memory-mapped sidecars."""

    protocol: str
    variant: str
    seed: int
    scenes: Mapping[str, SceneScores]
    held_out_scene: str | None = None
    source_scenes: tuple[str, ...] = ()
    test_read: bool = False
    manifest_path: Path | None = None
    checkpoint: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.protocol = str(self.protocol)
        self.variant = str(self.variant)
        self.seed = int(self.seed)
        if self.protocol not in PROTOCOLS:
            raise ValueError(f"unsupported V5 protocol: {self.protocol!r}")
        allowed = ALL_VARIANTS if self.protocol == "shared" else LOSO_VARIANTS
        if self.variant not in allowed:
            raise ValueError(f"variant {self.variant!r} is not registered for {self.protocol}")
        if self.seed < 0:
            raise ValueError("V5 run seed must be non-negative")
        if self.test_read and self.manifest_path is None:
            raise ValueError("testRead=true requires a final-test manifest")
        normalized = {str(scene): value for scene, value in self.scenes.items()}
        if any(scene != bundle.scene for scene, bundle in normalized.items()):
            raise ValueError("scene mapping key disagrees with SceneScores.scene")
        if self.protocol == "shared":
            if self.held_out_scene is not None:
                raise ValueError("shared V5 runs cannot declare heldOutScene")
        else:
            held_out = str(self.held_out_scene or "")
            sources = tuple(str(scene) for scene in self.source_scenes)
            if not held_out or len(sources) != 4 or len(set(sources)) != 4:
                raise ValueError("a LOSO V5 run requires one held-out scene and four source scenes")
            if held_out in sources or set(normalized) != {held_out, *sources}:
                raise ValueError("LOSO score manifest scenes must be held-out plus four source scenes")
            self.held_out_scene = held_out
            self.source_scenes = sources
        if not normalized:
            raise ValueError("V5 score manifest has no scenes")
        for scene, bundle in normalized.items():
            if not bundle.sidecars:
                raise ValueError(f"V5 score manifest has no sidecars for {scene!r}")
            if "train" in bundle.sidecars:
                raise ValueError("V5 score bundles cannot contain train scores")
            if any(split not in SIDECAR_SPLITS for split in bundle.sidecars):
                raise ValueError(f"V5 score manifest contains an invalid split for {scene!r}")
            if not self.test_read and "test" in bundle.sidecars:
                raise ValueError("test sidecars require an explicit final-test bundle")
        self.scenes = normalized
        self.source_scenes = tuple(str(scene) for scene in self.source_scenes)
        if self.manifest_path is not None:
            self.manifest_path = Path(self.manifest_path).resolve()
        if self.checkpoint is not None:
            self.checkpoint = dict(self.checkpoint)

    @classmethod
    def from_manifest(cls, path: str | Path, *, allow_test_read: bool = False) -> "V5Run":
        manifest_path = Path(path).resolve()
        payload = _read_manifest(manifest_path)
        if payload.get("schema") != SCORE_BUNDLE_SCHEMA:
            raise ValueError(
                f"V5 score bundle must use columnar schema {SCORE_BUNDLE_SCHEMA}; embedded pose rows are not accepted"
            )
        if int(payload.get("version", -1)) != 1:
            raise ValueError("unsupported columnar V5 score bundle version")
        protocol = str(payload.get("protocol", ""))
        variant = str(payload.get("variant", ""))
        seed = payload.get("seed")
        if not isinstance(seed, int):
            raise ValueError("V5 score bundle seed must be an integer")
        test_read = payload.get("testRead")
        if not isinstance(test_read, bool):
            raise ValueError("V5 score bundle testRead must be boolean")
        if test_read and not allow_test_read:
            raise PermissionError("test score bundle requires explicit final-test permission")
        score_splits = payload.get("scoreSplits")
        if not isinstance(score_splits, list) or not score_splits:
            raise ValueError("scoreSplits must be a non-empty list")
        normalized_score_splits = tuple(str(split) for split in score_splits)
        if len(set(normalized_score_splits)) != len(normalized_score_splits) or any(
            split not in SIDECAR_SPLITS for split in normalized_score_splits
        ):
            raise ValueError("scoreSplits contains an invalid or duplicate split")
        expected_score_splits = ("calibration", "validation", "test") if test_read else ("calibration", "validation")
        if set(normalized_score_splits) != set(expected_score_splits):
            raise ValueError("scoreSplits disagrees with testRead")
        checkpoint = _validate_checkpoint_metadata(payload)
        if checkpoint["protocol"] != protocol or checkpoint["variant"] != variant or int(checkpoint["seed"]) != seed:
            raise ValueError("checkpoint metadata disagrees with score bundle identity")
        held_out_raw = payload.get("heldOutScene")
        held_out = None if held_out_raw is None else str(held_out_raw)
        source_raw = payload.get("sourceScenes", [])
        if not isinstance(source_raw, list) or any(not isinstance(value, str) or not value for value in source_raw):
            raise ValueError("sourceScenes must be a list of non-empty strings")
        if protocol == "loso":
            if checkpoint.get("heldOutScene") != held_out:
                raise ValueError("checkpoint heldOutScene disagrees with score bundle")
            if tuple(checkpoint.get("sourceScenes", source_raw)) != tuple(source_raw):
                raise ValueError("checkpoint sourceScenes disagrees with score bundle")
        elif held_out is not None:
            raise ValueError("shared score bundle cannot declare heldOutScene")
        scenes_payload = _require_mapping(payload.get("scenes"), "scenes")
        scenes: dict[str, SceneScores] = {}
        for scene, raw in scenes_payload.items():
            scene_id = str(scene)
            scene_payload = _require_mapping(raw, f"scene {scene_id!r}")
            if any(isinstance(value, list) for value in scene_payload.values()):
                raise ValueError("embedded pose rows are not accepted; use manifest sidecar paths")
            splits_payload = _require_mapping(scene_payload.get("splits"), f"scene {scene_id}.splits")
            sidecars: dict[str, Path] = {}
            for split, sidecar in splits_payload.items():
                split_name = str(split)
                if split_name not in SIDECAR_SPLITS:
                    raise ValueError(f"invalid V5 sidecar split: {split_name!r}")
                sidecars[split_name] = _resolve(manifest_path.parent, sidecar, f"{scene_id}.{split_name}")
            if set(sidecars) != set(expected_score_splits):
                raise ValueError(f"{scene_id}.splits disagrees with root scoreSplits")
            pose_dataset = _resolve(manifest_path.parent, scene_payload.get("poseDataset"), f"{scene_id}.poseDataset")
            num_instances = int(scene_payload.get("numInstances", -1))
            if num_instances <= 0:
                raise ValueError(f"{scene_id}.numInstances must be positive")
            scenes[scene_id] = SceneScores(
                scene=scene_id,
                pose_dataset=pose_dataset,
                num_instances=num_instances,
                sidecars=sidecars,
            )
        return cls(
            protocol=protocol,
            variant=variant,
            seed=seed,
            scenes=scenes,
            held_out_scene=held_out,
            source_scenes=tuple(source_raw),
            test_read=test_read,
            manifest_path=manifest_path,
            checkpoint=checkpoint,
        )

    @classmethod
    def from_json(cls, path: str | Path, *, allow_test_read: bool = False) -> "V5Run":
        """Read the current manifest JSON; no embedded-row JSON is supported."""
        return cls.from_manifest(path, allow_test_read=allow_test_read)

    def records(self, scene: str, split: str, *, allow_test: bool = False):
        if scene not in self.scenes:
            raise ValueError(f"unknown V5 scene: {scene}")
        return self.scenes[scene].records(split, allow_test=allow_test)


def assert_test_free_selection(payload: Mapping[str, Any]) -> None:
    """Reject test-tainted provenance in calibration/model-selection metadata."""
    if payload.get("testRead") is True:
        raise ValueError("V5 model/threshold selection input is test-tainted")
    for field in ("modelSelectionSplit", "thresholdSelectionSplit", "selectionSplit"):
        if str(payload.get(field, "")).lower() == "test":
            raise ValueError(f"V5 {field} cannot be test")
    for field in ("selectedFromTest", "modelSelectedFromTest", "thresholdSelectedFromTest"):
        if payload.get(field) is True:
            raise ValueError(f"V5 {field} must be false")


def validate_matrix(
    runs: Iterable[V5Run],
    *,
    protocol: str,
    expected_scenes: Sequence[str] | None = None,
    expected_seed_count: int = EXPECTED_SEED_COUNT,
    allow_test_read: bool = False,
) -> tuple[V5Run, ...]:
    """Validate the frozen shared/LOSO matrix without loading score arrays."""
    normalized_protocol = str(protocol)
    if normalized_protocol not in PROTOCOLS:
        raise ValueError(f"unsupported V5 protocol: {normalized_protocol!r}")
    materialized = tuple(runs)
    if not materialized:
        raise ValueError("V5 result matrix is empty")
    if any(run.protocol != normalized_protocol for run in materialized):
        raise ValueError("V5 matrix contains a run from another protocol")
    if any(run.test_read and not allow_test_read for run in materialized):
        raise PermissionError("test matrix requires explicit final-test permission")
    scene_set = set(expected_scenes) if expected_scenes is not None else set().union(*(run.scenes for run in materialized))
    if len(scene_set) != EXPECTED_SCENE_COUNT:
        raise ValueError(f"V5 {normalized_protocol} matrix must contain exactly five scenes")
    if expected_scenes is not None and len(set(expected_scenes)) != EXPECTED_SCENE_COUNT:
        raise ValueError("expected_scenes must contain five unique names")
    required_variants = set(ALL_VARIANTS if normalized_protocol == "shared" else LOSO_VARIANTS)
    if {run.variant for run in materialized} != required_variants:
        raise ValueError("V5 matrix has an incomplete variant set")
    by_run_key: dict[tuple[str, int, str | None], V5Run] = {}
    seeds_by_variant: dict[str, set[int]] = {variant: set() for variant in required_variants}
    for run in materialized:
        if set(run.scenes) != scene_set:
            raise ValueError(f"run {run.variant}/{run.seed} does not cover the exact five scenes")
        if normalized_protocol == "shared":
            key = (run.variant, run.seed, None)
        else:
            if run.held_out_scene not in scene_set or set(run.source_scenes) != scene_set - {run.held_out_scene}:
                raise ValueError(f"LOSO run {run.variant}/{run.seed} has invalid held-out/source scenes")
            key = (run.variant, run.seed, str(run.held_out_scene))
        if key in by_run_key:
            raise ValueError(f"duplicate V5 run for {run.variant}/{run.seed}")
        by_run_key[key] = run
        seeds_by_variant[run.variant].add(run.seed)
        for scene in run.scenes.values():
            if "calibration" not in scene.splits or "validation" not in scene.splits:
                raise ValueError(f"run {run.variant}/{run.seed} is missing calibration or validation sidecars")
    seed_sets = list(seeds_by_variant.values())
    if any(values != seed_sets[0] for values in seed_sets[1:]):
        raise ValueError("V5 variants do not share the same seed set")
    if len(seed_sets[0]) != int(expected_seed_count):
        raise ValueError(f"V5 {normalized_protocol} matrix requires {int(expected_seed_count)} seeds per variant")
    if normalized_protocol == "loso":
        for variant in required_variants:
            for seed in seeds_by_variant[variant]:
                held_out = {
                    str(run.held_out_scene)
                    for (run_variant, run_seed, _), run in by_run_key.items()
                    if run_variant == variant and run_seed == seed
                }
                if held_out != scene_set:
                    raise ValueError(f"LOSO fold coverage is incomplete for {variant}/{seed}")
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
    "SceneScores",
    "V5Run",
    "assert_test_free_selection",
    "validate_matrix",
]
