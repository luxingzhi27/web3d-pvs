from __future__ import annotations

from pathlib import Path
import unittest

from neural_instance_culling.benchmark.run_standard_graphics_64k_targeted_refinement import (
    CONTROL_NAME,
    CONFIGS,
    PILOT_SEEDS,
    config_source_checkpoint,
    configs_for_scene,
    qualification_tier,
    refinement_command,
)


class StandardGraphics64kTargetedRefinementTests(unittest.TestCase):
    def test_registered_matrix_is_small_and_scene_specific(self) -> None:
        self.assertEqual(CONTROL_NAME, "control_no_refinement")
        self.assertEqual(len(CONFIGS), 16)
        self.assertEqual(len(configs_for_scene("sponza_64k")), 4)
        self.assertEqual(len(configs_for_scene("bigcity_64k")), 8)
        self.assertEqual(len(configs_for_scene("viking_village_64k")), 4)
        self.assertEqual(PILOT_SEEDS["sponza_64k"], 20260803)
        self.assertEqual(PILOT_SEEDS["bigcity_64k"], 20260801)
        self.assertEqual(PILOT_SEEDS["viking_village_64k"], 20260803)

    def test_sponza_does_not_add_negative_only_pose_sampling(self) -> None:
        for config in configs_for_scene("sponza_64k"):
            self.assertEqual(config.negative_only_pose_fraction, 0.0)

    def test_bigcity_scans_two_legal_negative_pose_quotas(self) -> None:
        self.assertEqual(
            {config.negative_only_pose_fraction for config in configs_for_scene("bigcity_64k")},
            {0.125, 0.25},
        )
        self.assertEqual(
            {config.relation_k for config in configs_for_scene("bigcity_64k")},
            {8, 16},
        )

    def test_viking_uses_tail_separation_without_negative_only_poses(self) -> None:
        configs = configs_for_scene("viking_village_64k")
        self.assertEqual(
            {config.negative_only_pose_fraction for config in configs},
            {0.0},
        )
        self.assertEqual(
            {config.separation_weight for config in configs},
            {0.30, 0.45, 0.60, 0.90},
        )

    def test_visual_utility_configs_reduce_low_importance_positive_floor(self) -> None:
        configs = [config for config in CONFIGS if "visual_utility" in config.name]
        self.assertEqual(len(configs), 6)
        self.assertEqual(
            {config.positive_importance_floor for config in configs},
            {0.01, 0.05},
        )
        self.assertEqual(
            {config.positive_importance_power for config in configs},
            {1.0},
        )
        bigcity = next(config for config in configs if config.name == "b6_visual_utility")
        self.assertIn(
            "b1_guarded_neg25_seed20260801",
            str(config_source_checkpoint(bigcity, 20260801)),
        )

    def test_command_is_checkpoint_refinement_without_head_reset(self) -> None:
        config = configs_for_scene("bigcity_64k")[0]
        command = refinement_command(
            config,
            PILOT_SEEDS["bigcity_64k"],
            Path("/tmp/output"),
            Path("/tmp/source.pt"),
            final=False,
        )
        self.assertEqual(command[command.index("--epochs") + 1], "2")
        self.assertEqual(command[command.index("--steps-per-epoch") + 1], "450")
        self.assertEqual(command[command.index("--poses-per-batch") + 1], "16")
        self.assertEqual(command[command.index("--init-checkpoint") + 1], "/tmp/source.pt")
        self.assertEqual(command[command.index("--negative-only-pose-fraction") + 1], "0.125")
        self.assertEqual(command[command.index("--frontier-positive-importance-floor") + 1], "0.25")
        self.assertNotIn("--reset-runtime-heads", command)
        self.assertNotIn("test", command)

    def test_qualification_tiers_keep_lcb_as_target_not_rejection(self) -> None:
        self.assertEqual(
            qualification_tier({"agg_weighted_recall": 0.995, "weighted_recall_lower_confidence_bound": 0.991}),
            "confidence_target_met",
        )
        self.assertEqual(
            qualification_tier({"agg_weighted_recall": 0.995, "weighted_recall_lower_confidence_bound": 0.989}),
            "mean_target_met",
        )
        self.assertEqual(
            qualification_tier({"agg_weighted_recall": 0.989, "weighted_recall_lower_confidence_bound": 0.988}),
            "mean_target_not_met",
        )


if __name__ == "__main__":
    unittest.main()
