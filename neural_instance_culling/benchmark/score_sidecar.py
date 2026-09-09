"""Typed, pose-aligned storage for formal visibility scores and IDs.

The JSON evaluation summary stays small.  Candidate IDs, probabilities, labels,
weights, and predicted IDs are stored in typed binary arrays, with uint64 pose
offsets describing the rows belonging to each pose.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SIDECAR_SCHEMA = "pvs-typed-score-sidecar-v1"
_DTYPES = {
    "poseIndices": np.dtype("<i8"),
    "poseOffsets": np.dtype("<u8"),
    "predictedOffsets": np.dtype("<u8"),
    "candidateIds": np.dtype("<u4"),
    "predictedIds": np.dtype("<u4"),
    "scores": np.dtype("<f4"),
    "targets": np.dtype("u1"),
    "visibleWeights": np.dtype("<f4"),
}


def _finite_float_array(value: Any, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{label} contains non-finite values")
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ScoreSidecarWriter:
    """Append one candidate-aligned record for each pose, then finalize it."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        threshold: float | None,
        checkpoint: str | Path | None = None,
        calibration: str | Path | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        existing = [path for path in self.root.iterdir() if path.name != "manifest.json"]
        if existing:
            raise FileExistsError(f"score sidecar directory is not empty: {self.root}")
        self.split = str(split)
        self.threshold = None if threshold is None else float(threshold)
        if self.threshold is not None and not np.isfinite(self.threshold):
            raise ValueError("sidecar threshold must be finite")
        self.checkpoint = None if checkpoint is None else str(Path(checkpoint).resolve())
        self.calibration = None if calibration is None else str(Path(calibration).resolve())
        self._closed = False
        self._last_pose_index: int | None = None
        self._pose_indices: list[int] = []
        self._pose_offsets = [0]
        self._predicted_offsets = [0]
        self._counts = {"candidate": 0, "predicted": 0}
        self._handles: dict[str, Any] = {}
        for field in ("candidateIds", "predictedIds", "scores", "targets", "visibleWeights"):
            path = self.root / f"{field[0].lower()}{field[1:]}.bin.tmp"
            self._handles[field] = path.open("wb")

    def append_pose(
        self,
        pose_index: int,
        candidate_ids: np.ndarray,
        scores: np.ndarray,
        targets: np.ndarray,
        visible_weights: np.ndarray,
        predicted_ids: np.ndarray | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("score sidecar is already closed")
        pose = int(pose_index)
        if self._last_pose_index is not None and pose <= self._last_pose_index:
            raise ValueError("sidecar pose indices must be strictly increasing")
        candidates = np.asarray(candidate_ids, dtype=np.uint32).reshape(-1)
        score_values = _finite_float_array(scores, "scores")
        target_values = np.asarray(targets, dtype=np.uint8).reshape(-1)
        weight_values = _finite_float_array(visible_weights, "visibleWeights")
        predicted = np.zeros((0,), dtype=np.uint32) if predicted_ids is None else np.asarray(predicted_ids, dtype=np.uint32).reshape(-1)
        if not (candidates.size == score_values.size == target_values.size == weight_values.size):
            raise ValueError("candidate IDs, scores, targets, and visible weights must have equal lengths")
        if candidates.size != np.unique(candidates).size:
            raise ValueError(f"candidate IDs contain duplicates at pose {pose}")
        if predicted.size != np.unique(predicted).size:
            raise ValueError(f"predicted IDs contain duplicates at pose {pose}")
        if predicted.size and not bool(np.all(np.isin(predicted, candidates, assume_unique=False))):
            raise ValueError(f"predicted IDs are not a subset of candidates at pose {pose}")
        if bool(np.any(target_values > 1)):
            raise ValueError("sidecar targets must be binary uint8 values")
        if bool(np.any(weight_values < 0.0)):
            raise ValueError("sidecar visible weights must be non-negative")
        if bool(np.any(score_values < 0.0)) or bool(np.any(score_values > 1.0)):
            raise ValueError("sidecar scores must lie in [0, 1]")

        candidates.tofile(self._handles["candidateIds"])
        predicted.tofile(self._handles["predictedIds"])
        score_values.tofile(self._handles["scores"])
        target_values.tofile(self._handles["targets"])
        weight_values.tofile(self._handles["visibleWeights"])
        self._pose_indices.append(pose)
        self._counts["candidate"] += int(candidates.size)
        self._counts["predicted"] += int(predicted.size)
        self._pose_offsets.append(self._counts["candidate"])
        self._predicted_offsets.append(self._counts["predicted"])
        self._last_pose_index = pose

    def close(self) -> Path:
        if self._closed:
            return self.root / "manifest.json"
        for handle in self._handles.values():
            handle.close()

        arrays = {
            "poseIndices": np.asarray(self._pose_indices, dtype=_DTYPES["poseIndices"]),
            "poseOffsets": np.asarray(self._pose_offsets, dtype=_DTYPES["poseOffsets"]),
            "predictedOffsets": np.asarray(self._predicted_offsets, dtype=_DTYPES["predictedOffsets"]),
        }
        for field, values in arrays.items():
            path = self.root / f"{field[0].lower()}{field[1:]}.bin"
            temporary = path.with_suffix(path.suffix + ".tmp")
            values.tofile(temporary)
            temporary.replace(path)
        for field in ("candidateIds", "predictedIds", "scores", "targets", "visibleWeights"):
            temporary = self.root / f"{field[0].lower()}{field[1:]}.bin.tmp"
            path = temporary.with_suffix("")
            temporary.replace(path)

        manifest = {
            "schema": SIDECAR_SCHEMA,
            "version": 1,
            "split": self.split,
            "threshold": self.threshold,
            "checkpoint": self.checkpoint,
            "calibration": self.calibration,
            "testRead": self.split == "test",
            "poseCount": len(self._pose_indices),
            "candidateCount": self._counts["candidate"],
            "predictedCount": self._counts["predicted"],
            "files": {
                field: f"{field[0].lower()}{field[1:]}.bin"
                for field in _DTYPES
            },
            "dtypes": {field: dtype.str for field, dtype in _DTYPES.items()},
        }
        manifest_path = self.root / "manifest.json"
        _write_json(manifest_path, manifest)
        self._closed = True
        return manifest_path

    def __enter__(self) -> "ScoreSidecarWriter":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _read_array(root: Path, manifest: Mapping[str, Any], field: str) -> np.ndarray:
    files = manifest.get("files")
    dtypes = manifest.get("dtypes")
    if not isinstance(files, Mapping) or not isinstance(dtypes, Mapping):
        raise ValueError("score sidecar manifest is missing files or dtypes")
    if field not in files or field not in dtypes:
        raise ValueError(f"score sidecar manifest is missing {field}")
    expected_dtype = _DTYPES[field]
    if str(dtypes[field]) != expected_dtype.str:
        raise ValueError(f"score sidecar dtype for {field} is invalid")
    path = root / str(files[field])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        return np.zeros((0,), dtype=expected_dtype)
    return np.memmap(path, dtype=expected_dtype, mode="r")


def read_score_sidecar(manifest_path: str | Path) -> dict[str, np.ndarray]:
    """Read and validate a sidecar without materializing its large arrays."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping) or manifest.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"invalid score sidecar schema: {path}")
    root = path.parent
    result = {field: _read_array(root, manifest, field) for field in _DTYPES}
    pose_count = int(manifest.get("poseCount", -1))
    candidate_count = int(manifest.get("candidateCount", -1))
    predicted_count = int(manifest.get("predictedCount", -1))
    if pose_count < 0 or candidate_count < 0 or predicted_count < 0:
        raise ValueError("score sidecar counts are invalid")
    if result["poseIndices"].size != pose_count:
        raise ValueError("score sidecar pose index count disagrees with manifest")
    if pose_count and bool(np.any(np.diff(result["poseIndices"]) <= 0)):
        raise ValueError("score sidecar pose indices are not strictly increasing")
    if result["poseOffsets"].size != pose_count + 1 or result["predictedOffsets"].size != pose_count + 1:
        raise ValueError("score sidecar offset count disagrees with manifest")
    if int(result["poseOffsets"][0]) != 0 or int(result["poseOffsets"][-1]) != candidate_count:
        raise ValueError("score sidecar candidate offsets do not cover candidates")
    if int(result["predictedOffsets"][0]) != 0 or int(result["predictedOffsets"][-1]) != predicted_count:
        raise ValueError("score sidecar predicted offsets do not cover predictions")
    if bool(np.any(np.diff(result["poseOffsets"]) < 0)) or bool(np.any(np.diff(result["predictedOffsets"]) < 0)):
        raise ValueError("score sidecar offsets are not non-decreasing")
    for field in ("candidateIds", "scores", "targets", "visibleWeights"):
        if result[field].size != candidate_count:
            raise ValueError(f"score sidecar {field} count disagrees with candidate offsets")
    if result["predictedIds"].size != predicted_count:
        raise ValueError("score sidecar predicted ID count disagrees with predicted offsets")
    return result


def average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    """Compute non-interpolated AP after merging equal score values."""
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(targets, dtype=np.uint8).reshape(-1)
    if values.size != labels.size:
        raise ValueError("scores and targets must have equal lengths")
    if not bool(np.isfinite(values).all()) or bool(np.any(labels > 1)):
        raise ValueError("AP inputs are invalid")
    positive_count = int(labels.sum())
    if positive_count == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    ordered_labels = labels[order].astype(np.float64, copy=False)
    ordered_scores = values[order]
    starts = np.concatenate(
        [
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]).astype(np.int64) + 1,
        ]
    )
    group_positive = np.add.reduceat(ordered_labels, starts)
    group_total = np.diff(np.concatenate([starts, np.asarray([ordered_labels.size], dtype=np.int64)])).astype(np.float64)
    cumulative_positive = np.cumsum(group_positive)
    cumulative_total = np.cumsum(group_total)
    return float(np.sum((group_positive / positive_count) * (cumulative_positive / np.maximum(cumulative_total, 1.0))))


__all__ = ["SIDECAR_SCHEMA", "ScoreSidecarWriter", "average_precision", "read_score_sidecar"]
