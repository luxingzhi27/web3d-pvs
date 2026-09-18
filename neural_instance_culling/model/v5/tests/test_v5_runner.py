from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neural_instance_culling.model.v5.runner import (
    CONFIRMATION_UPDATES,
    PILOT_UPDATES,
    RUN_SCHEMA,
    _parser,
    _prepare_metrics_log,
    _validate_checkpoint,
    build_scan_matrix,
    build_training_context,
    capture_rng_states,
    expected_total_updates,
    restore_rng_states,
    seed_everything,
)
from neural_instance_culling.model.v5.core import GCOFPVSV5
from neural_instance_culling.model.v5.losses import DualGroupState
from neural_instance_culling.model.v5.train import checkpoint_payload


class V5RunnerPlanTests(unittest.TestCase):
    def test_preflight_cli_is_train_only_and_explicit(self) -> None:
        args = _parser().parse_args([
            "preflight",
            "--protocol", "shared",
            "--output", "preflight.json",
        ])
        self.assertEqual(args.command, "preflight")
        self.assertEqual(args.protocol, "shared")
        self.assertEqual(args.output, Path("preflight.json"))

    def test_resume_rewinds_metric_log_to_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train_metrics.jsonl"
            path.write_text(
                "".join(
                    json.dumps({"globalStep": step, "loss": float(step)}) + "\n"
                    for step in range(1, 6)
                ),
                encoding="utf-8",
            )
            _prepare_metrics_log(path, resume_step=3)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["globalStep"] for row in rows], [1, 2, 3])

            with self.assertRaises(FileExistsError):
                _prepare_metrics_log(path, resume_step=None)

    def test_loso_context_excludes_held_out_real_scene_and_all_test_reads(self) -> None:
        specs, policy = build_training_context(
            protocol="loso",
            held_out_scene="sponza_64k",
            require_files=False,
        )
        ids = [spec.scene_id for spec in specs]
        self.assertNotIn("sponza_64k", ids)
        self.assertEqual(sum(spec.source_kind == "real" for spec in specs), 4)
        self.assertEqual(sum(spec.source_kind == "synthetic" for spec in specs), 96)
        self.assertNotIn("sponza_64k", policy.to_manifest()["sourceTrainSceneIds"])
        self.assertFalse(policy.to_manifest()["allowHeldOutVisibilityLabels"])
        self.assertFalse(policy.to_manifest()["allowHeldOutExternalHitProbe"])
        real = [spec for spec in specs if spec.source_kind == "real"]
        synthetic = [spec for spec in specs if spec.source_kind == "synthetic"]
        self.assertTrue(all(spec.dual_group_id == spec.scene_id for spec in real))
        self.assertEqual(len({spec.dual_group_id for spec in synthetic}), 5)
        self.assertTrue(all(spec.dual_group_id.startswith("synthetic_family:") for spec in synthetic))

    def test_scan_matrix_has_six_pilots_and_two_explicit_confirmations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = build_scan_matrix(
                phase="pilot",
                protocol="shared",
                seed=0,
                held_out_scene=None,
                source_scene_ids=["real", "synthetic"],
                output_dir=root / "pilot",
            )
            self.assertEqual(len(pilot["runs"]), 6)
            self.assertTrue(all(row["updates"] == PILOT_UPDATES for row in pilot["runs"]))
            self.assertTrue(all(row["testRead"] is False for row in pilot["runs"]))

            names = [row["name"] for row in pilot["runs"][:2]]
            confirmation = build_scan_matrix(
                phase="confirmation",
                protocol="shared",
                seed=0,
                held_out_scene=None,
                source_scene_ids=["real", "synthetic"],
                output_dir=root / "confirmation",
                selected_configurations=names,
            )
            self.assertEqual(len(confirmation["runs"]), 2)
            self.assertTrue(all(row["updates"] == CONFIRMATION_UPDATES for row in confirmation["runs"]))
            payload = json.loads((root / "confirmation/scan_matrix.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["testRead"])

    def test_update_budget_matches_frozen_scene_schedule(self) -> None:
        self.assertEqual(expected_total_updates(5, total_updates=12_000), 12_000)
        self.assertEqual(expected_total_updates(4, real_updates_per_scene=36_000), 216_000)
        self.assertEqual(expected_total_updates(5, real_updates_per_scene=36_000), 270_000)

    def test_rng_checkpoint_round_trip_restores_data_stream(self) -> None:
        seed_everything(31)
        data_rng = np.random.default_rng(97)
        states = capture_rng_states(data_rng)
        expected = data_rng.integers(0, 1_000_000, size=8)
        restore_rng_states(states, data_rng)
        actual = data_rng.integers(0, 1_000_000, size=8)
        self.assertTrue(np.array_equal(expected, actual))

    def test_checkpoint_contract_contains_all_recovery_state(self) -> None:
        model = GCOFPVSV5("FULL")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        dual = DualGroupState(["scene"], dual_lr=3e-3)
        config = {
            "schema": RUN_SCHEMA,
            "protocol": "shared",
            "variant": "FULL",
            "objective": "FULL",
            "seed": 0,
            "heldOutScene": None,
            "modelLearningRate": 2e-4,
            "dualLearningRate": 3e-3,
            "weightDecay": 1e-5,
            "totalUpdates": 2,
            "realUpdatesPerScene": None,
            "scheduleSeed": 0,
            "poseCount": 4,
            "probeCount": 8192,
            "probeSampling": "uniform_unit_grouped_ray_distance_v1",
            "probeObservationsPerUnit": 16,
            "probeUnitsPerStep": 512,
            "geometryChunkSize": 512,
            "sourceSceneIds": ["scene"],
            "dualGrouping": "real_scene_and_synthetic_family_v1",
            "dualGroupByScene": {"scene": "scene"},
        }
        rng = np.random.default_rng(17)
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            dual_state=dual,
            global_step=0,
            scene_updates={"scene": 0},
            rng_states=capture_rng_states(rng),
            config=config,
        )
        _validate_checkpoint(payload, config)
        self.assertEqual(
            set(payload),
            {"schema", "modelConfig", "model", "optimizer", "scheduler", "dualState", "globalStep", "sceneUpdates", "rngStates", "config", "testRead"},
        )
        self.assertFalse(payload["testRead"])


if __name__ == "__main__":
    unittest.main()
