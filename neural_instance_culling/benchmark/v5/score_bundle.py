"""Columnar V5 score bundles backed by the frozen PoseCSR dataset.

The score bundle deliberately stores only model scores.  Candidate IDs,
visible IDs and visible weights remain in the immutable PoseCSR dataset and
are checked when a sidecar is opened.  This keeps a large evaluation run
bounded by typed arrays instead of expanding every pose into JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset


BUNDLE_SCHEMA = "gcof-pvs-v5-columnar-score-bundle-v1"
SIDECAR_SCHEMA = "gcof-pvs-v5-columnar-score-sidecar-v1"
POSE_SCHEMA = "pose-csr-explicit-four-way-split-v1"
SIDECAR_SPLITS = ("calibration", "validation", "test")

_POSE_DTYPE = np.dtype("<i8")
_OFFSET_DTYPE = np.dtype("<u8")
_SCORE_DTYPE = np.dtype("<f4")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve_manifest_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _check_bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


@dataclass(frozen=True)
class PoseScores:
    """One lazy pose view assembled from PoseCSR and a score sidecar."""

    scene: str
    pose_index: int
    split: str
    candidate_ids: np.ndarray
    scores: np.ndarray
    targets: np.ndarray
    visible_weights: np.ndarray

    def __post_init__(self) -> None:
        if not self.scene:
            raise ValueError("pose score scene must be non-empty")
        if self.split not in SIDECAR_SPLITS:
            raise ValueError(f"unsupported pose score split: {self.split!r}")
        candidates = np.asarray(self.candidate_ids, dtype=np.uint32).reshape(-1)
        scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        targets = np.asarray(self.targets, dtype=np.uint8).reshape(-1)
        weights = np.asarray(self.visible_weights, dtype=np.float32).reshape(-1)
        if not (candidates.size == scores.size == targets.size == weights.size):
            raise ValueError("PoseCSR candidate, score, target and weight lengths disagree")
        if candidates.size and int(candidates.max()) < 0:
            raise ValueError("candidate IDs must be non-negative")
        if candidates.size != np.unique(candidates).size:
            raise ValueError(f"candidate IDs are duplicated at pose {self.pose_index}")
        if not bool(np.isfinite(scores).all()):
            raise ValueError(f"scores contain non-finite values at pose {self.pose_index}")
        if bool(np.any(targets > 1)):
            raise ValueError("PoseCSR targets must be binary")
        if not bool(np.isfinite(weights).all()) or bool(np.any(weights < 0.0)):
            raise ValueError("PoseCSR visible weights must be finite and non-negative")
        object.__setattr__(self, "candidate_ids", candidates)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "visible_weights", weights)
        object.__setattr__(self, "pose_index", int(self.pose_index))


@dataclass(frozen=True)
class PoseCSRValidation:
    scene: str
    split: str
    pose_count: int
    candidate_count: int
    visible_count: int


class FrozenPoseCSR:
    """Read-only, strict view of one registered PoseCSR split."""

    def __init__(self, dataset_dir: str | Path, *, scene: str, num_instances: int) -> None:
        self.dataset_dir = Path(dataset_dir).resolve()
        self.scene = str(scene)
        self.num_instances = int(num_instances)
        if not self.scene:
            raise ValueError("PoseCSR scene must be non-empty")
        if self.num_instances <= 0:
            raise ValueError("PoseCSR num_instances must be positive")
        meta = _read_json(self.dataset_dir / "dataset_meta.json", "PoseCSR dataset_meta")
        if meta.get("schema") != POSE_SCHEMA:
            raise ValueError(f"{self.scene} PoseCSR does not use {POSE_SCHEMA}")
        if int(meta.get("numInstances", -1)) != self.num_instances:
            raise ValueError(f"{self.scene} PoseCSR/runtime instance count disagrees")
        self.dataset = PoseCSRDataset(self.dataset_dir, self.num_instances)
        if self.dataset.meta.get("schema") != POSE_SCHEMA:
            raise ValueError("PoseCSR schema changed while opening the dataset")

    def split_indices(self, split: str) -> np.ndarray:
        if split not in SIDECAR_SPLITS:
            raise ValueError(f"V5 score sidecars cannot use split {split!r}")
        indices = np.asarray(self.dataset.split(split).pose_indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError(f"{self.scene} {split} split is empty")
        return indices

    @staticmethod
    def _strict_unique(values: np.ndarray, label: str, pose_index: int) -> None:
        if values.size and values.min() < 0:
            raise ValueError(f"{label} contains a negative ID at pose {pose_index}")
        if values.size != np.unique(values).size:
            raise ValueError(f"{label} contains duplicates at pose {pose_index}")

    def validate_split(
        self,
        split: str,
        *,
        excluded_unit_ids: Sequence[int] = (),
    ) -> PoseCSRValidation:
        indices = self.split_indices(split)
        excluded = np.asarray(excluded_unit_ids, dtype=np.uint32).reshape(-1)
        candidate_total = 0
        visible_total = 0
        for pose_index in indices.tolist():
            candidates = np.asarray(self.dataset.candidate_slice(int(pose_index)), dtype=np.uint32)
            visible, weights = self.dataset.visible_slice(int(pose_index))
            visible = np.asarray(visible, dtype=np.uint32)
            weights = np.asarray(weights)
            split_id = int(self.dataset.split_ids[split])
            if int(self.dataset.poses["split"][pose_index]) != split_id:
                raise ValueError(f"PoseCSR pose {pose_index} is not in declared {split} split")
            self._strict_unique(candidates, "candidate IDs", int(pose_index))
            self._strict_unique(visible, "visible IDs", int(pose_index))
            if candidates.size and int(candidates.max()) >= self.num_instances:
                raise ValueError(f"candidate ID exceeds instance count at pose {pose_index}")
            if visible.size and int(visible.max()) >= self.num_instances:
                raise ValueError(f"visible ID exceeds instance count at pose {pose_index}")
            if visible.size and not bool(np.all(np.isin(visible, candidates, assume_unique=False))):
                raise ValueError(f"visible IDs are not a subset of candidates at pose {pose_index}")
            if excluded.size and visible.size and bool(np.isin(visible, excluded).any()):
                raise ValueError(f"excluded non-renderable unit is visible at pose {pose_index}")
            if weights.size != visible.size:
                raise ValueError(f"visible IDs and weights disagree at pose {pose_index}")
            if not bool(np.isfinite(weights).all()) or bool(np.any(weights < 0.0)):
                raise ValueError(f"visible weights are invalid at pose {pose_index}")
            candidate_total += int(candidates.size - np.isin(candidates, excluded).sum())
            visible_total += int(visible.size)
        return PoseCSRValidation(
            scene=self.scene,
            split=split,
            pose_count=int(indices.size),
            candidate_count=candidate_total,
            visible_count=visible_total,
        )

    def pose_scores(
        self,
        pose_index: int,
        split: str,
        scores: np.ndarray,
        *,
        excluded_unit_ids: Sequence[int] = (),
    ) -> PoseScores:
        if int(pose_index) < 0 or int(pose_index) >= self.dataset.poses.size:
            raise ValueError(f"pose index is outside PoseCSR: {pose_index}")
        candidates = np.asarray(self.dataset.candidate_slice(int(pose_index)), dtype=np.uint32)
        excluded = np.asarray(excluded_unit_ids, dtype=np.uint32).reshape(-1)
        if excluded.size:
            candidates = candidates[~np.isin(candidates, excluded)]
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size != candidates.size:
            raise ValueError(f"score count disagrees with candidates at pose {pose_index}")
        visible, visible_weights = self.dataset.visible_slice(int(pose_index))
        visible = np.asarray(visible, dtype=np.uint32)
        visible_weights = np.asarray(visible_weights, dtype=np.float32)
        targets = np.zeros((candidates.size,), dtype=np.uint8)
        weights = np.zeros((candidates.size,), dtype=np.float32)
        if visible.size:
            locations = np.flatnonzero(np.isin(candidates, visible, assume_unique=False))
            if locations.size:
                visible_locations = np.searchsorted(visible, candidates[locations])
                # PoseCSR visible IDs are normally sorted.  This explicit map
                # also keeps the loader correct for a frozen unsorted fixture.
                if bool(np.all(visible_locations < visible.size)) and bool(
                    np.all(visible[visible_locations] == candidates[locations])
                ):
                    targets[locations] = 1
                    weights[locations] = visible_weights[visible_locations]
                else:
                    mapping = {int(value): float(weight) for value, weight in zip(visible, visible_weights)}
                    targets[locations] = 1
                    weights[locations] = np.asarray(
                        [mapping[int(value)] for value in candidates[locations]], dtype=np.float32
                    )
        return PoseScores(
            scene=self.scene,
            pose_index=int(pose_index),
            split=split,
            candidate_ids=candidates,
            scores=values,
            targets=targets,
            visible_weights=weights,
        )


class ColumnarScoreSidecarWriter:
    """Write one float32 score vector aligned with one complete PoseCSR split."""

    def __init__(
        self,
        root: str | Path,
        *,
        scene: str,
        split: str,
        pose_dataset: str | Path,
        num_instances: int,
        allow_test: bool = False,
        excluded_unit_ids: Sequence[int] = (),
    ) -> None:
        self.root = Path(root).resolve()
        self.scene = str(scene)
        self.split = str(split)
        if self.split not in SIDECAR_SPLITS:
            raise ValueError(f"unsupported V5 score split: {self.split!r}")
        if self.split == "test" and not allow_test:
            raise PermissionError("test score sidecars require explicit final-test permission")
        self.pose_dataset = Path(pose_dataset).resolve()
        self.num_instances = int(num_instances)
        excluded = np.asarray(excluded_unit_ids, dtype=np.int64).reshape(-1)
        if excluded.size:
            if bool((excluded < 0).any()) or bool((excluded >= self.num_instances).any()):
                raise ValueError("excludedUnitIds are outside the scene")
            if excluded.size != np.unique(excluded).size:
                raise ValueError("excludedUnitIds must be unique")
        self.excluded_unit_ids = tuple(int(value) for value in np.sort(excluded).tolist())
        self.root.mkdir(parents=True, exist_ok=True)
        if any(self.root.iterdir()):
            raise FileExistsError(f"score sidecar output is not empty: {self.root}")
        self._closed = False
        self._last_pose: int | None = None
        self._pose_count = 0
        self._candidate_count = 0
        self._pose_file = (self.root / "pose_indices_int64.bin.part").open("wb")
        self._offset_file = (self.root / "score_offsets_uint64.bin.part").open("wb")
        self._score_file = (self.root / "scores_float32.bin.part").open("wb")
        np.asarray([0], dtype=_OFFSET_DTYPE).tofile(self._offset_file)

    def append_pose(self, pose_index: int, scores: np.ndarray, *, candidate_count: int) -> None:
        if self._closed:
            raise RuntimeError("score sidecar writer is closed")
        pose = int(pose_index)
        if self._last_pose is not None and pose <= self._last_pose:
            raise ValueError("score sidecar pose indices must be strictly increasing")
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size != int(candidate_count):
            raise ValueError(f"score count does not equal candidate count at pose {pose}")
        if not bool(np.isfinite(values).all()):
            raise ValueError(f"scores contain non-finite values at pose {pose}")
        np.asarray([pose], dtype=_POSE_DTYPE).tofile(self._pose_file)
        values.tofile(self._score_file)
        self._candidate_count += int(values.size)
        np.asarray([self._candidate_count], dtype=_OFFSET_DTYPE).tofile(self._offset_file)
        self._pose_count += 1
        self._last_pose = pose

    def close(self) -> Path:
        if self._closed:
            return self.root / "manifest.json"
        for handle in (self._pose_file, self._offset_file, self._score_file):
            handle.close()
        files = {
            "poseIndices": "pose_indices_int64.bin",
            "scoreOffsets": "score_offsets_uint64.bin",
            "scores": "scores_float32.bin",
        }
        for name in files.values():
            part = self.root / f"{name}.part"
            part.replace(self.root / name)
        manifest = {
            "schema": SIDECAR_SCHEMA,
            "version": 1,
            "scene": self.scene,
            "split": self.split,
            "poseDataset": str(self.pose_dataset),
            "numInstances": self.num_instances,
            "poseCount": self._pose_count,
            "candidateCount": self._candidate_count,
            "scoreSpace": "logit",
            "scoreDtype": _SCORE_DTYPE.str,
            "testRead": self.split == "test",
            "excludedUnitIds": list(self.excluded_unit_ids),
            "files": files,
            "shapes": {
                "poseIndices": [self._pose_count],
                "scoreOffsets": [self._pose_count + 1],
                "scores": [self._candidate_count],
            },
            "dtypes": {
                "poseIndices": _POSE_DTYPE.str,
                "scoreOffsets": _OFFSET_DTYPE.str,
                "scores": _SCORE_DTYPE.str,
            },
        }
        _write_json(self.root / "manifest.json", manifest)
        self._closed = True
        return self.root / "manifest.json"

    def __enter__(self) -> "ColumnarScoreSidecarWriter":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()


class ColumnarScoreSidecar:
    """Memory-mapped reader for a validated score sidecar."""

    def __init__(self, manifest_path: str | Path, *, allow_test: bool = False) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = _read_json(self.manifest_path, "V5 score sidecar manifest")
        if self.manifest.get("schema") != SIDECAR_SCHEMA:
            raise ValueError(f"score sidecar schema must be {SIDECAR_SCHEMA}")
        self.scene = str(self.manifest.get("scene", ""))
        self.split = str(self.manifest.get("split", ""))
        if not self.scene or self.split not in SIDECAR_SPLITS:
            raise ValueError("score sidecar scene or split is invalid")
        test_read = _check_bool(self.manifest.get("testRead"), "score sidecar testRead")
        if test_read != (self.split == "test"):
            raise ValueError("score sidecar testRead disagrees with split")
        if test_read and not allow_test:
            raise PermissionError("test score sidecar requires explicit final-test permission")
        if self.manifest.get("scoreSpace") != "logit":
            raise ValueError("V5 score sidecars must declare scoreSpace=logit")
        if self.manifest.get("scoreDtype") != _SCORE_DTYPE.str:
            raise ValueError("V5 score sidecars must use little-endian float32 scores")
        self.pose_dataset = _resolve_manifest_path(self.root, self.manifest.get("poseDataset"), "poseDataset")
        self.num_instances = int(self.manifest.get("numInstances", -1))
        self.pose_count = int(self.manifest.get("poseCount", -1))
        self.candidate_count = int(self.manifest.get("candidateCount", -1))
        excluded = np.asarray(self.manifest.get("excludedUnitIds", []), dtype=np.int64).reshape(-1)
        if excluded.size:
            if bool((excluded < 0).any()) or bool((excluded >= self.num_instances).any()):
                raise ValueError("score sidecar excludedUnitIds are outside the scene")
            if excluded.size != np.unique(excluded).size or bool(np.any(np.diff(excluded) <= 0)):
                raise ValueError("score sidecar excludedUnitIds must be sorted and unique")
        self.excluded_unit_ids = tuple(int(value) for value in excluded.tolist())
        if self.num_instances <= 0 or self.pose_count < 0 or self.candidate_count < 0:
            raise ValueError("score sidecar counts are invalid")
        files = self.manifest.get("files")
        dtypes = self.manifest.get("dtypes")
        shapes = self.manifest.get("shapes")
        if not isinstance(files, Mapping) or not isinstance(dtypes, Mapping) or not isinstance(shapes, Mapping):
            raise ValueError("score sidecar files, dtypes and shapes are required")
        expected = {
            "poseIndices": (_POSE_DTYPE, self.pose_count),
            "scoreOffsets": (_OFFSET_DTYPE, self.pose_count + 1),
            "scores": (_SCORE_DTYPE, self.candidate_count),
        }
        arrays: dict[str, np.ndarray] = {}
        for field, (dtype, count) in expected.items():
            if str(dtypes.get(field)) != dtype.str or list(shapes.get(field, [])) != [count]:
                raise ValueError(f"score sidecar {field} shape or dtype is invalid")
            path = _resolve_manifest_path(self.root, files.get(field), f"files.{field}")
            if not path.is_file() or path.stat().st_size != count * dtype.itemsize:
                raise ValueError(f"score sidecar {field} byte length is invalid: {path}")
            arrays[field] = np.memmap(path, dtype=dtype, mode="r", shape=(count,)) if count else np.zeros((0,), dtype=dtype)
        self.pose_indices = arrays["poseIndices"]
        self.score_offsets = arrays["scoreOffsets"]
        self.scores = arrays["scores"]
        if self.score_offsets.size and int(self.score_offsets[0]) != 0:
            raise ValueError("score sidecar offsets must start at zero")
        if self.score_offsets.size and int(self.score_offsets[-1]) != self.candidate_count:
            raise ValueError("score sidecar offsets do not cover score array")
        if self.pose_indices.size and bool(np.any(np.diff(self.pose_indices) <= 0)):
            raise ValueError("score sidecar pose indices must be strictly increasing")
        if self.score_offsets.size > 1 and bool(np.any(np.diff(self.score_offsets) < 0)):
            raise ValueError("score sidecar offsets must be non-decreasing")
        if self.scores.size and not bool(np.isfinite(self.scores).all()):
            raise ValueError("score sidecar contains non-finite scores")

    def validate_against_pose_csr(self, csr: FrozenPoseCSR) -> PoseCSRValidation:
        if self.scene != csr.scene:
            raise ValueError("score sidecar scene disagrees with PoseCSR scene")
        if self.pose_dataset != csr.dataset_dir:
            raise ValueError("score sidecar is attached to a different PoseCSR dataset")
        if self.num_instances != csr.num_instances:
            raise ValueError("score sidecar instance count disagrees with PoseCSR")
        expected_indices = csr.split_indices(self.split)
        if not np.array_equal(self.pose_indices, expected_indices):
            raise ValueError(f"score sidecar does not cover the complete frozen {self.split} split")
        validation = csr.validate_split(
            self.split,
            excluded_unit_ids=self.excluded_unit_ids,
        )
        if validation.candidate_count != self.candidate_count:
            raise ValueError("score sidecar candidate count disagrees with PoseCSR")
        counts = np.asarray(self.score_offsets[1:] - self.score_offsets[:-1], dtype=np.uint64)
        expected_counts = np.asarray(
            [self._candidate_count_for_pose(csr, int(pose)) for pose in expected_indices], dtype=np.uint64
        )
        if not np.array_equal(counts, expected_counts):
            raise ValueError("score sidecar per-pose offsets disagree with PoseCSR candidates")
        return validation

    def _candidate_count_for_pose(self, csr: FrozenPoseCSR, pose_index: int) -> int:
        candidates = np.asarray(csr.dataset.candidate_slice(pose_index), dtype=np.uint32)
        if self.excluded_unit_ids:
            candidates = candidates[~np.isin(candidates, self.excluded_unit_ids)]
        return int(candidates.size)

    def iter_scores(self) -> Iterator[tuple[int, np.ndarray]]:
        for index, pose_index in enumerate(self.pose_indices.tolist()):
            start = int(self.score_offsets[index])
            end = int(self.score_offsets[index + 1])
            yield int(pose_index), self.scores[start:end]


@dataclass
class SceneScores:
    scene: str
    pose_dataset: Path
    num_instances: int
    sidecars: Mapping[str, Path]
    _csr: FrozenPoseCSR | None = None
    _sidecar_cache: dict[str, ColumnarScoreSidecar] = field(default_factory=dict, init=False, repr=False)
    _validated_splits: set[str] = field(default_factory=set, init=False, repr=False)

    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(sorted(self.sidecars))

    def _open_csr(self) -> FrozenPoseCSR:
        if self._csr is None:
            self._csr = FrozenPoseCSR(
                self.pose_dataset,
                scene=self.scene,
                num_instances=self.num_instances,
            )
        return self._csr

    def _open_sidecar(self, split: str, *, allow_test: bool) -> ColumnarScoreSidecar:
        sidecar = self._sidecar_cache.get(split)
        if sidecar is None:
            sidecar = ColumnarScoreSidecar(self.sidecars[split], allow_test=allow_test)
            self._sidecar_cache[split] = sidecar
        elif split == "test" and not allow_test:
            raise PermissionError("test score sidecar requires explicit final-test permission")
        if split not in self._validated_splits:
            sidecar.validate_against_pose_csr(self._open_csr())
            self._validated_splits.add(split)
        return sidecar

    def records(self, split: str, *, allow_test: bool = False) -> Iterator[PoseScores]:
        """Return a re-openable stream; calibration may scan it more than once."""
        if split not in self.sidecars:
            raise ValueError(f"scene {self.scene!r} has no {split} score sidecar")
        return _PoseScoreStream(self, split, allow_test=allow_test)


class _PoseScoreStream:
    def __init__(self, scene_scores: SceneScores, split: str, *, allow_test: bool) -> None:
        self.scene_scores = scene_scores
        self.split = split
        self.allow_test = bool(allow_test)

    def __iter__(self) -> Iterator[PoseScores]:
        csr = self.scene_scores._open_csr()
        sidecar = self.scene_scores._open_sidecar(self.split, allow_test=self.allow_test)
        for pose_index, scores in sidecar.iter_scores():
            yield csr.pose_scores(
                pose_index,
                self.split,
                scores,
                excluded_unit_ids=sidecar.excluded_unit_ids,
            )


def write_bundle_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, payload)
    return output


__all__ = [
    "BUNDLE_SCHEMA",
    "ColumnarScoreSidecar",
    "ColumnarScoreSidecarWriter",
    "FrozenPoseCSR",
    "POSE_SCHEMA",
    "PoseCSRValidation",
    "PoseScores",
    "SceneScores",
    "SIDECAR_SCHEMA",
    "SIDECAR_SPLITS",
    "write_bundle_manifest",
]
