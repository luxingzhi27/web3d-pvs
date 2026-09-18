from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_instance_culling.dataset.v5.build_external_hit_probes import (
    make_probe_manifest,
    load_probe_table,
    sample_current_status_observations,
)
from neural_instance_culling.dataset.v5.permissions import (
    ASSET_VISIBILITY_LABELS,
    authorize_asset_read,
    make_loso_training_policy,
)


class ExternalHitProbePermissionTest(unittest.TestCase):
    def test_source_probe_can_be_read_but_held_out_probe_cannot(self) -> None:
        policy = make_loso_training_policy(["source_scene"], "held_out_scene")
        manifest = make_probe_manifest("source_scene", 1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "external_hit_probe_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            rows = manifest["rowCount"]
            np.zeros(rows, dtype="<u4").tofile(root / manifest["files"]["unitIds"])
            directions = np.zeros((rows, 3), dtype="<f4")
            directions[:, 0] = 1.0
            directions.tofile(root / manifest["files"]["directions"])
            np.full(rows, np.nan, dtype="<f4").tofile(root / manifest["files"]["hitDistances"])
            np.full(rows, 1024.0, dtype="<f4").tofile(root / manifest["files"]["maxDistances"])
            np.tile(np.arange(16, dtype="u1"), 36).tofile(root / manifest["files"]["startIds"])
            np.repeat(np.arange(36, dtype="u1"), 16).tofile(root / manifest["files"]["directionIds"])
            table = load_probe_table(root, policy=policy)
            sampled = sample_current_status_observations(table, np.random.default_rng(7), count=32)
            self.assertEqual(sampled["unit_ids"].shape, (32,))
            held_out_manifest = dict(manifest)
            held_out_manifest["sceneId"] = "held_out_scene"
            (root / "external_hit_probe_manifest.json").write_text(
                json.dumps(held_out_manifest), encoding="utf-8"
            )
            with self.assertRaises(PermissionError):
                load_probe_table(root, policy=policy)
        with self.assertRaises(PermissionError):
            authorize_asset_read(
                policy,
                scene_id="held_out_scene",
                asset_kind=ASSET_VISIBILITY_LABELS,
                split="calibration",
            )


if __name__ == "__main__":
    unittest.main()
