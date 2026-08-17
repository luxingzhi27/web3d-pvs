"""Train-only stratified sampler for occlusion survival observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


SAMPLER_SCHEMA = "stratified-survival-observation-sampler-v3"


def _group_indices(labels: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Group row indices with one stable sort instead of one full scan per ID."""

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    unique, starts, counts = np.unique(
        sorted_values,
        return_index=True,
        return_counts=True,
    )
    groups = {
        int(value): order[int(start):int(start + count)]
        for value, start, count in zip(unique.tolist(), starts.tolist(), counts.tolist())
    }
    return unique.astype(np.int64, copy=False), groups


def _numpy(value: Any, dtype: np.dtype[Any], name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=dtype)
    if result.ndim == 0:
        result = result.reshape(1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _direction_ids(observations: Mapping[str, Any], count: int) -> np.ndarray:
    if "direction_id" in observations:
        result = _numpy(observations["direction_id"], np.int64, "direction_id").reshape(-1)
    elif "direction_ids" in observations:
        result = _numpy(observations["direction_ids"], np.int64, "direction_ids").reshape(-1)
    elif "direction" in observations:
        direction = np.asarray(observations["direction"])
        if direction.ndim == 1:
            result = _numpy(direction, np.int64, "direction").reshape(-1)
        elif direction.shape == (count, 3):
            vectors = _numpy(direction, np.float32, "direction")
            azimuth = np.arctan2(vectors[:, 2], vectors[:, 0])
            horizontal = np.floor((azimuth + np.pi) / (2.0 * np.pi) * 4.0).astype(np.int64) % 4
            vertical = np.where(vectors[:, 1] > 0.35, 1, np.where(vectors[:, 1] < -0.35, 2, 0))
            result = vertical * 4 + horizontal
        else:
            raise ValueError("direction must contain N direction IDs or have shape [N, 3]")
    else:
        raise ValueError("observations need direction_id(s) or direction vectors")
    if result.size != count or np.any((result < 0) | (result >= 12)):
        raise ValueError("direction IDs must contain N values in [0, 11]")
    return result


@dataclass(frozen=True)
class SurvivalSampleBatch:
    indices: np.ndarray
    sampling_probability: np.ndarray
    importance_correction: np.ndarray


class StratifiedSurvivalObservationSampler:
    """Cycle train observations without fixing one repeated subset.

    Instance representatives are emitted first in a shuffled order each epoch,
    which guarantees direct supervision coverage when the caller consumes at
    least ``ceil(unique_instances / batch_size)`` batches.  Remaining slots are
    balanced over event/censor, twelve direction anchors, and depth buckets.
    """

    def __init__(
        self,
        observations: Mapping[str, Any],
        *,
        seed: int,
        depth_buckets: int = 4,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if int(depth_buckets) <= 0:
            raise ValueError("depth_buckets must be positive")
        metadata = dict(metadata or {})
        if metadata.get("trainOnly") is not True or set(metadata.get("splitNames", [])) != {"train"}:
            raise ValueError("survival sampler accepts train-only observations")
        instance = _numpy(observations["instance"], np.int64, "instance").reshape(-1)
        event = _numpy(observations["event"], np.float32, "event").reshape(-1)
        depth_source = observations.get("depth", observations.get("rho"))
        if depth_source is None:
            raise ValueError("observations need depth or rho")
        depth = _numpy(depth_source, np.float32, "depth").reshape(-1)
        count = int(instance.size)
        if count == 0 or event.size != count or depth.size != count:
            raise ValueError("observation arrays must be non-empty and aligned")
        if np.any(instance < 0) or np.any((event < 0.0) | (event > 1.0)):
            raise ValueError("invalid instance or event values")
        direction = _direction_ids(observations, count)

        # Quantiles are derived only from the already train-only observation
        # artifact.  Stable unique edges avoid empty zero-width buckets.
        quantile_edges = np.quantile(
            depth, np.linspace(0.0, 1.0, int(depth_buckets) + 1)
        )
        # Repeated depths can collapse several quantiles to the same value.
        # Remove those empty boundaries before assigning buckets so the
        # manifest describes the actual ordered depth partition.
        inner_edges = np.unique(quantile_edges[1:-1])
        edges = np.concatenate(
            [np.asarray([-np.inf]), inner_edges, np.asarray([np.inf])]
        )
        depth_id = np.searchsorted(edges[1:-1], depth, side="right").astype(np.int64)
        event_id = (event >= 0.5).astype(np.int64)
        code = (event_id * 12 + direction) * int(depth_buckets) + depth_id

        self.observations = observations
        self.instance = instance
        self.event = event
        self.direction_id = direction
        self.depth = depth
        self.depth_edges = edges
        self.stratum_code = code
        self.seed = int(seed)
        self.depth_buckets = int(depth_buckets)
        self.unique_instances, self._instance_members = _group_indices(instance)
        _stratum_ids, self._strata = _group_indices(code)
        self._epoch: int | None = None
        self._representatives = np.zeros((0,), dtype=np.int64)
        self._representative_cursor = 0
        self._bucket_values: dict[int, np.ndarray] = {}
        self._bucket_cursors: dict[int, int] = {}
        self._bucket_order = np.zeros((0,), dtype=np.int64)
        self._bucket_cursor = 0
        self._seen_instances: set[int] = set()
        self._seen_indices: set[int] = set()
        self._seen_directions: set[int] = set()
        self._seen_event_values: set[int] = set()

    def start_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed + int(epoch) * 1_000_003)
        representatives = [
            int(members[int(rng.integers(0, members.size))])
            for members in self._instance_members.values()
        ]
        self._representatives = rng.permutation(np.asarray(representatives, dtype=np.int64))
        self._representative_cursor = 0
        self._bucket_values = {
            code: rng.permutation(values) for code, values in self._strata.items()
        }
        self._bucket_cursors = {code: 0 for code in self._bucket_values}
        self._bucket_order = rng.permutation(np.asarray(sorted(self._bucket_values), dtype=np.int64))
        self._bucket_cursor = 0
        self._seen_instances = set()
        self._seen_indices = set()
        self._seen_directions = set()
        self._seen_event_values = set()
        self._epoch = int(epoch)

    def _take_from_bucket(self, code: int, rng: np.random.Generator) -> int:
        values = self._bucket_values[code]
        cursor = self._bucket_cursors[code]
        if cursor >= values.size:
            values = rng.permutation(values)
            self._bucket_values[code] = values
            cursor = 0
        result = int(values[cursor])
        self._bucket_cursors[code] = cursor + 1
        return result

    def sample_batch(self, batch_size: int, *, step: int) -> SurvivalSampleBatch:
        if self._epoch is None:
            raise RuntimeError("start_epoch must be called before sample_batch")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        rng = np.random.default_rng(self.seed + self._epoch * 1_000_003 + int(step) * 97 + 17)
        selected: list[int] = []
        probability: list[float] = []
        remaining_representatives = self._representatives.size - self._representative_cursor
        required = min(int(batch_size), max(0, remaining_representatives))
        if required:
            values = self._representatives[
                self._representative_cursor:self._representative_cursor + required
            ]
            self._representative_cursor += required
            for index in values.tolist():
                selected.append(int(index))
                probability.append(1.0 / float(self._instance_members[int(self.instance[index])].size))

        while len(selected) < int(batch_size):
            code = int(self._bucket_order[self._bucket_cursor % self._bucket_order.size])
            self._bucket_cursor += 1
            index = self._take_from_bucket(code, rng)
            selected.append(index)
            probability.append(1.0 / float(self._strata[code].size))

        indices = np.asarray(selected, dtype=np.int64)
        sampling_probability = np.asarray(probability, dtype=np.float64)
        inverse = 1.0 / np.maximum(sampling_probability, 1e-12)
        correction = (inverse / max(float(inverse.mean()), 1e-12)).astype(np.float32)
        self._seen_instances.update(int(value) for value in self.instance[indices].tolist())
        self._seen_indices.update(int(value) for value in indices.tolist())
        self._seen_directions.update(int(value) for value in self.direction_id[indices].tolist())
        self._seen_event_values.update(int(value >= 0.5) for value in self.event[indices].tolist())
        return SurvivalSampleBatch(indices, sampling_probability.astype(np.float32), correction)

    @property
    def epoch_instance_coverage(self) -> float:
        return float(len(self._seen_instances) / max(1, self.unique_instances.size))

    @property
    def epoch_observation_coverage(self) -> float:
        return float(len(self._seen_indices) / max(1, self.instance.size))

    @property
    def epoch_unique_observation_count(self) -> int:
        return len(self._seen_indices)

    @property
    def epoch_direction_coverage(self) -> float:
        available = set(int(value) for value in np.unique(self.direction_id).tolist())
        return float(len(self._seen_directions) / max(1, len(available)))

    @property
    def epoch_event_censor_coverage(self) -> float:
        available = set(int(value >= 0.5) for value in np.unique(self.event).tolist())
        return float(len(self._seen_event_values) / max(1, len(available)))

    def gather(self, batch: SurvivalSampleBatch, *, device: torch.device) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        indices = torch.as_tensor(batch.indices, dtype=torch.long)
        for key, value in self.observations.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            # Formal artifacts use compact uint8/uint32 files.  PyTorch CPU
            # does not implement advanced indexing for uint32, while every
            # discrete observation field is semantically an integer ID or
            # label.  Canonicalize those fields before indexing; floating
            # weights and depths retain their source precision here.
            if not tensor.is_floating_point() and tensor.dtype != torch.bool:
                tensor = tensor.long()
            result[key] = tensor[indices].to(device=device)
        correction = torch.from_numpy(batch.importance_correction).to(device=device)
        result["sampling_probability"] = torch.from_numpy(batch.sampling_probability).to(device=device)
        result["sampling_correction"] = correction
        if "weight" in result:
            result["weight"] = result["weight"].float() * correction
        return result

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": SAMPLER_SCHEMA,
            "trainOnly": True,
            "seed": self.seed,
            "observationCount": int(self.instance.size),
            "uniqueInstanceCount": int(self.unique_instances.size),
            "stratumCount": int(len(self._strata)),
            "depthBuckets": self.depth_buckets,
            # JSON has no representation for +/-inf. ``None`` denotes the
            # open lower/upper boundary while finite inner edges remain
            # numeric and preserve the ordered partition.
            "depthEdges": [
                float(value) if np.isfinite(value) else None
                for value in self.depth_edges.tolist()
            ],
            "depthEdgeSemantics": "null denotes an unbounded bucket boundary",
        }
