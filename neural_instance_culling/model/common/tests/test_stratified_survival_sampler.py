from __future__ import annotations

import unittest

import numpy as np
import torch

from neural_instance_culling.model.common.stratified_survival_sampler import (
    StratifiedSurvivalObservationSampler,
    _group_indices,
)


def observations() -> dict[str, np.ndarray]:
    count = 240
    return {
        "instance": np.repeat(np.arange(24), 10).astype(np.int64),
        "event": (np.arange(count) % 3 != 0).astype(np.float32),
        "direction_id": (np.arange(count) % 12).astype(np.int64),
        "depth": np.linspace(0.01, 0.99, count, dtype=np.float32),
        "weight": np.ones((count,), dtype=np.float32),
    }


class StratifiedSurvivalObservationSamplerTest(unittest.TestCase):
    def test_group_indices_matches_stable_membership(self) -> None:
        labels = np.asarray([4, 1, 4, 2, 1, 4], dtype=np.int64)
        unique, groups = _group_indices(labels)
        self.assertTrue(np.array_equal(unique, np.asarray([1, 2, 4])))
        self.assertTrue(np.array_equal(groups[1], np.asarray([1, 4])))
        self.assertTrue(np.array_equal(groups[2], np.asarray([3])))
        self.assertTrue(np.array_equal(groups[4], np.asarray([0, 2, 5])))

    def test_rejects_non_train_metadata(self) -> None:
        with self.assertRaises(ValueError):
            StratifiedSurvivalObservationSampler(
                observations(), seed=1, metadata={"trainOnly": False, "splitNames": ["train"]}
            )

    def test_epoch_covers_every_instance_and_changes_indices(self) -> None:
        sampler = StratifiedSurvivalObservationSampler(
            observations(), seed=7, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        sampler.start_epoch(0)
        first = sampler.sample_batch(16, step=0)
        second = sampler.sample_batch(16, step=1)
        self.assertFalse(np.array_equal(first.indices, second.indices))
        for step in range(2, 8):
            sampler.sample_batch(16, step=step)
        self.assertEqual(sampler.epoch_instance_coverage, 1.0)

    def test_seed_and_epoch_are_deterministic(self) -> None:
        left = StratifiedSurvivalObservationSampler(
            observations(), seed=11, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        right = StratifiedSurvivalObservationSampler(
            observations(), seed=11, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        left.start_epoch(3)
        right.start_epoch(3)
        self.assertTrue(np.array_equal(left.sample_batch(32, step=0).indices, right.sample_batch(32, step=0).indices))

    def test_gather_applies_finite_probability_correction(self) -> None:
        sampler = StratifiedSurvivalObservationSampler(
            observations(), seed=3, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        sampler.start_epoch(0)
        batch = sampler.sample_batch(32, step=0)
        result = sampler.gather(batch, device=torch.device("cpu"))
        self.assertTrue(bool(torch.isfinite(result["weight"]).all()))
        self.assertAlmostEqual(float(result["sampling_correction"].mean()), 1.0, places=5)

    def test_accepts_formal_artifact_direction_ids_under_direction_key(self) -> None:
        data = observations()
        data["instance"] = data["instance"].astype(np.uint32)
        data["direction"] = data.pop("direction_id").astype(np.uint8)
        data["subpose"] = np.arange(data["instance"].size, dtype=np.uint32)
        sampler = StratifiedSurvivalObservationSampler(
            data,
            seed=31,
            metadata={"trainOnly": True, "splitNames": ["train"]},
        )
        self.assertTrue(np.array_equal(sampler.direction_id, data["direction"]))
        sampler.start_epoch(0)
        gathered = sampler.gather(sampler.sample_batch(32, step=0), device=torch.device("cpu"))
        self.assertEqual(gathered["instance"].dtype, torch.int64)
        self.assertEqual(gathered["direction"].dtype, torch.int64)
        self.assertEqual(gathered["subpose"].dtype, torch.int64)

    def test_repeated_depth_quantiles_are_unique_and_keep_ordered_values(self) -> None:
        data = observations()
        data["depth"] = np.repeat(np.asarray([0.2, 0.8], dtype=np.float32), 120)
        sampler = StratifiedSurvivalObservationSampler(
            data,
            seed=19,
            metadata={"trainOnly": True, "splitNames": ["train"]},
        )
        finite_edges = sampler.depth_edges[np.isfinite(sampler.depth_edges)]
        self.assertEqual(finite_edges.size, np.unique(finite_edges).size)
        sampler.start_epoch(0)
        for step in range(15):
            sampler.sample_batch(16, step=step)
        self.assertEqual(sampler.epoch_instance_coverage, 1.0)

    def test_manifest_uses_null_for_open_depth_boundaries(self) -> None:
        sampler = StratifiedSurvivalObservationSampler(
            observations(),
            seed=29,
            metadata={"trainOnly": True, "splitNames": ["train"]},
        )
        manifest = sampler.manifest()
        self.assertIsNone(manifest["depthEdges"][0])
        self.assertIsNone(manifest["depthEdges"][-1])
        self.assertEqual(
            manifest["depthEdgeSemantics"],
            "null denotes an unbounded bucket boundary",
        )

    def test_new_epoch_rotates_observation_stream_reproducibly(self) -> None:
        left = StratifiedSurvivalObservationSampler(
            observations(), seed=23, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        right = StratifiedSurvivalObservationSampler(
            observations(), seed=23, metadata={"trainOnly": True, "splitNames": ["train"]}
        )
        left.start_epoch(0)
        right.start_epoch(0)
        left_stream = np.concatenate([left.sample_batch(16, step=step).indices for step in range(4)])
        right_stream = np.concatenate([right.sample_batch(16, step=step).indices for step in range(4)])
        self.assertTrue(np.array_equal(left_stream, right_stream))
        left.start_epoch(1)
        rotated = np.concatenate([left.sample_batch(16, step=step).indices for step in range(4)])
        self.assertFalse(np.array_equal(left_stream, rotated))


if __name__ == "__main__":
    unittest.main()
