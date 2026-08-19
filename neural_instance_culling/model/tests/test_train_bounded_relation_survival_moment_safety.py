from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from neural_instance_culling.model.train_bounded_relation_survival_moment_safety import (
    CALIBRATION_FLOOR,
    _boundary_tail_refinement_objective_groups,
    _calibration_workpoints,
    _candidate_frustum_boundary_proximity,
    _boundary_opportunity_scale,
    _boundary_opportunity_pose_pool,
    _boundary_tail_selection_logits,
    _evaluation_batch_limits,
    _extreme_tail_selection_logits,
    _instance_calibration_blend,
    _instance_calibration_reliability,
    _instance_exposure_balance_weights,
    _dual_probe_rescue_batch_loss,
    _dual_probe_rescue_objective_groups,
    _load_dual_probe_rescue_init,
    _initialize_from_checkpoint,
    _operating_threshold_metrics,
    _refinement_pose_sampling,
    _recurrent_negative_priority_from_counts,
    _resolve_split,
    _save_safe_checkpoint_alias,
    _select_tail_hard_pose_pool,
    _select_weighted_safety_frontier_members,
    _set_refinement_scope,
    _update_safety_boundary_ema,
    _validate_extreme_tail_selection_contract,
    _v4_objective_groups,
    _weighted_recall_safety_gate,
    _weighted_quantile_numpy,
    parse_args,
)
from neural_instance_culling.model.bounded_relation_survival_moment_model import (
    BoundedRelationSurvivalMomentModel,
    DUAL_PROBE_RAW_QUERY_DIM,
    MODEL_SCHEMA,
)
from neural_instance_culling.model.current_pvs_utils import score_distribution_summary


def _dual_probe_init_spec() -> dict[str, object]:
    zero = [0.0] * DUAL_PROBE_RAW_QUERY_DIM

    def certificate(risk: float) -> dict[str, object]:
        return {
            "sourceSplit": "train",
            "fitRowsOnly": True,
            "risk": risk,
            "observedRisk": 0.0,
            "registeredRiskUpperBound": risk,
            "riskWithinRegisteredBound": True,
            "threshold": 0.25,
        }

    return {
        "schema": "pvs-dual-probe-rescue-init-v1",
        "version": 1,
        "enabled": True,
        "testRead": False,
        "selectedFromValidation": False,
        "selectionSplit": "calibration",
        "rawFeatureLayout": [
            {"name": "center_view", "offset": 0, "dimension": 9},
            {"name": "spectral_features", "offset": 9, "dimension": 64},
            {"name": "boundary_spectral_summary", "offset": 73, "dimension": 8},
            {
                "name": "region_extrema_lower_upper_span",
                "offset": 81,
                "dimension": 27,
            },
        ],
        "rawFeatureDimension": DUAL_PROBE_RAW_QUERY_DIM,
        "primary": {
            "fitSplit": "train",
            "mean": zero,
            "scale": [1.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "coefficients": [0.0] * (DUAL_PROBE_RAW_QUERY_DIM + 1),
            "threshold": 0.25,
            "riskCertificate": certificate(0.01),
        },
        "coverage": {
            "fitSplit": "train",
            "mean": zero,
            "scale": [1.0] * DUAL_PROBE_RAW_QUERY_DIM,
            "coefficients": [0.0] * (DUAL_PROBE_RAW_QUERY_DIM + 1),
            "threshold": 0.05,
            "riskCertificate": certificate(0.05),
        },
        "alpha": 0.5,
        "temperature": 0.05,
        "coverageWeight": 2.0,
    }


class BoundedRelationSurvivalMomentSafetyTrainingTest(unittest.TestCase):
    def test_cross_pose_operating_and_exposure_cli_contract_is_explicit(self) -> None:
        argv = [
            "train",
            "--dataset-dir", "dataset",
            "--relation-dir", "relation",
            "--runtime-meta", "runtime.json",
            "--initial-geo-features", "geometry.bin",
            "--glb-index", "glb-index.json",
            "--glb-root", "glb-root",
            "--output-dir", "output",
            "--loss-variant", "cross_pose_operating",
            "--cross-pose-positive-class-fraction", "0.2",
            "--cross-pose-weighted-recall-target", "0.993",
            "--cross-pose-operating-weight", "0.75",
            "--exposure-supervision-hidden-dim", "16",
            "--exposure-supervision-loss-weight", "0.05",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.loss_variant, "cross_pose_operating")
        self.assertEqual(args.cross_pose_positive_class_fraction, 0.2)
        self.assertEqual(args.cross_pose_weighted_recall_target, 0.993)
        self.assertEqual(args.cross_pose_operating_weight, 0.75)
        self.assertEqual(args.exposure_supervision_hidden_dim, 16)
        self.assertEqual(args.exposure_supervision_loss_weight, 0.05)
        self.assertEqual(args.refinement_scope, "all")
        self.assertIsNone(args.initial_checkpoint)

    def test_pose_balanced_frontier_cli_contract_is_explicit(self) -> None:
        argv = [
            "train",
            "--dataset-dir", "dataset",
            "--relation-dir", "relation",
            "--runtime-meta", "runtime.json",
            "--initial-geo-features", "geometry.bin",
            "--glb-index", "glb-index.json",
            "--glb-root", "glb-root",
            "--output-dir", "output",
            "--loss-variant", "pose_balanced_frontier",
            "--frontier-loss-weight", "0.4",
            "--frontier-positive-mass-fraction", "0.01",
            "--frontier-negative-fraction", "0.02",
            "--frontier-margin", "0.75",
            "--frontier-temperature", "0.2",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.loss_variant, "pose_balanced_frontier")
        self.assertEqual(args.frontier_loss_weight, 0.4)
        self.assertEqual(args.frontier_positive_mass_fraction, 0.01)
        self.assertEqual(args.frontier_negative_fraction, 0.02)
        self.assertEqual(args.frontier_margin, 0.75)
        self.assertEqual(args.frontier_temperature, 0.2)
        self.assertEqual(args.refinement_scope, "all")
        self.assertIsNone(args.initial_checkpoint)

    def test_query_tail_cli_defaults_to_joint_from_scratch_contract(self) -> None:
        argv = [
            "train",
            "--dataset-dir", "dataset",
            "--relation-dir", "relation",
            "--runtime-meta", "runtime.json",
            "--initial-geo-features", "geometry.bin",
            "--glb-index", "glb-index.json",
            "--glb-root", "glb-root",
            "--subpose-sidecar", "subpose",
            "--output-dir", "output",
            "--query-tail-separator-family", "mlp",
            "--query-tail-separation-loss-weight", "0.3",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.query_tail_separator_family, "mlp")
        self.assertEqual(args.query_tail_separation_loss_weight, 0.3)
        self.assertEqual(args.refinement_scope, "all")
        self.assertIsNone(args.initial_checkpoint)

    def test_dual_probe_init_protocol_rejects_test_or_validation_selected_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "init.json"
            payload = _dual_probe_init_spec()
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                _load_dual_probe_rescue_init(path)["schema"],
                "pvs-dual-probe-rescue-init-v1",
            )
            for field, value in (("testRead", True), ("selectedFromValidation", True)):
                rejected = dict(payload)
                rejected[field] = value
                path.write_text(json.dumps(rejected), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, field):
                    _load_dual_probe_rescue_init(path)

    def test_dual_probe_cli_parses_scope_and_loss_weights(self) -> None:
        argv = [
            "train",
            "--dataset-dir",
            "dataset",
            "--relation-dir",
            "relation",
            "--runtime-meta",
            "runtime.json",
            "--initial-geo-features",
            "geometry.bin",
            "--glb-index",
            "glb-index.json",
            "--glb-root",
            "glb-root",
            "--output-dir",
            "output",
            "--dual-probe-rescue-init",
            "probe.json",
            "--refinement-scope",
            "dual_probe_rescue",
            "--dual-probe-rescue-loss-weight",
            "1.5",
            "--dual-probe-rescue-primary-keep-weight",
            "2",
            "--dual-probe-rescue-negative-decay-weight",
            "0.25",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertEqual(args.refinement_scope, "dual_probe_rescue")
        self.assertEqual(args.dual_probe_rescue_init, Path("probe.json"))
        self.assertEqual(args.dual_probe_rescue_loss_weight, 1.5)
        self.assertEqual(args.dual_probe_rescue_primary_keep_weight, 2.0)
        self.assertEqual(args.dual_probe_rescue_negative_decay_weight, 0.25)

    def test_dual_probe_scope_only_leaves_attenuation_head_trainable(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            dual_probe_rescue=_dual_probe_init_spec(),
        )
        meta = _set_refinement_scope(model, "dual_probe_rescue", initialized=True)
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("dual_probe_rescue_attenuation_head.") for name in trainable)
        )
        self.assertFalse(meta["baseParametersTrainable"])
        self.assertTrue(meta["fixedPosteriorFrozen"])

    def test_dual_probe_loss_enters_only_safety_total_with_pose_offsets(self) -> None:
        count = 4
        aux = {
            "dual_probe_raw_features": torch.zeros((count, DUAL_PROBE_RAW_QUERY_DIM)),
            "dual_probe_rescue_base_logit": torch.zeros((count, 1)),
            "dual_probe_primary_gate": torch.ones((count, 1)),
            "dual_probe_coverage_gate": torch.ones((count, 1)),
            "dual_probe_primary_attenuation": torch.full((count, 1), 0.8),
            "dual_probe_coverage_attenuation": torch.full((count, 1), 0.7),
            "dual_probe_fixed_residual": torch.full((count, 1), 0.5),
            "dual_probe_residual": torch.full((count, 1), 0.4),
        }
        loss, diagnostics = _dual_probe_rescue_batch_loss(
            aux,
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            torch.ones(count),
            torch.tensor([0, 2, 4]),
            coverage_weight=2.0,
            primary_keep_weight=1.0,
            coverage_keep_weight=1.0,
            negative_decay_weight=1.0,
            high_negative_decay_weight=1.0,
            visible_weight_epsilon=1e-6,
        )
        groups = _dual_probe_rescue_objective_groups(loss, loss_weight=1.5)
        self.assertEqual(diagnostics["dualProbeRescuePoseCount"], 2)
        self.assertTrue(diagnostics["dualProbeRescuePoseOffsetsUsed"])
        torch.testing.assert_close(groups["total"], 1.5 * loss)
        torch.testing.assert_close(groups["safety"], groups["total"])
        self.assertEqual(float(groups["relation"]), 0.0)
        self.assertEqual(float(groups["schedule"]), 0.0)
        self.assertEqual(float(groups["efficiency"]), 0.0)

    def test_old_checkpoint_migration_adds_exactly_the_dual_probe_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            with torch.no_grad():
                source.visibility_head.bias.fill_(0.75)
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 24,
                    "experimentName": "source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            spec = _dual_probe_init_spec()
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                dual_probe_rescue=spec,
            )
            expected_primary_mean = target.dual_probe_primary_mean.detach().clone()
            initialization = _initialize_from_checkpoint(target, path)
            self.assertTrue(initialization["freshDualProbeRescue"])
            torch.testing.assert_close(target.dual_probe_primary_mean, expected_primary_mean)
            self.assertTrue(
                torch.equal(
                    target.dual_probe_rescue_attenuation_head[-1].weight,
                    torch.zeros_like(target.dual_probe_rescue_attenuation_head[-1].weight),
                )
            )
            self.assertTrue(
                torch.equal(
                    target.dual_probe_rescue_attenuation_head[-1].bias,
                    torch.zeros_like(target.dual_probe_rescue_attenuation_head[-1].bias),
                )
            )
            self.assertTrue(torch.equal(target.visibility_head.bias, source.visibility_head.bias))

    def test_dual_probe_checkpoint_can_resume_without_replacing_fixed_posterior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dual.pt"
            spec = _dual_probe_init_spec()
            source = BoundedRelationSurvivalMomentModel(
                3,
                2,
                dual_probe_rescue=spec,
            )
            with torch.no_grad():
                source.dual_probe_rescue_attenuation_head[-1].bias.copy_(
                    torch.tensor([0.25, -0.50])
                )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 2,
                    "experimentName": "dual-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                dual_probe_rescue=spec,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertFalse(initialization["freshDualProbeRescue"])
            self.assertEqual(initialization["mode"], "model-weights-only-fresh-optimizer")
            torch.testing.assert_close(
                target.dual_probe_primary_coefficients,
                source.dual_probe_primary_coefficients,
            )
            torch.testing.assert_close(
                target.dual_probe_rescue_attenuation_head[-1].bias,
                source.dual_probe_rescue_attenuation_head[-1].bias,
            )
            self.assertIn("dualProbeRescue", target.config)

    def test_initial_visibility_tail_selector_uses_frozen_head(self) -> None:
        current = torch.tensor([[5.0], [6.0]], requires_grad=True)
        query = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        weight = torch.tensor([[2.0, -1.0]], requires_grad=True)
        bias = torch.tensor([0.5], requires_grad=True)

        selected = _extreme_tail_selection_logits(
            "initial_visibility", current, query, weight, bias
        )

        torch.testing.assert_close(selected, torch.tensor([[0.5], [2.5]]))
        self.assertFalse(selected.requires_grad)
        self.assertIs(
            _extreme_tail_selection_logits(
                "current", current, query, None, None
            ),
            current,
        )

    def test_initial_visibility_tail_selector_requires_refinement_contract(self) -> None:
        valid = SimpleNamespace(
            tail_selection_source="initial_visibility",
            initial_checkpoint=Path("last.pt"),
            refinement_scope="viewcell_region_conditioned_visibility",
            viewcell_region_conditioned_visibility=True,
        )
        _validate_extreme_tail_selection_contract(valid)
        region_tail = SimpleNamespace(**vars(valid))
        region_tail.refinement_scope = "viewcell_region_tail"
        _validate_extreme_tail_selection_contract(region_tail)
        epoch_snapshot = SimpleNamespace(**vars(valid))
        epoch_snapshot.initial_checkpoint = Path("checkpoint_epoch_024.pt")
        _validate_extreme_tail_selection_contract(epoch_snapshot)
        for field, value in (
            ("initial_checkpoint", None),
            ("refinement_scope", "visibility_head"),
            ("viewcell_region_conditioned_visibility", False),
        ):
            invalid = SimpleNamespace(**vars(valid))
            setattr(invalid, field, value)
            with self.assertRaisesRegex(ValueError, "initial checkpoint"):
                _validate_extreme_tail_selection_contract(invalid)
        current = SimpleNamespace(
            tail_selection_source="current",
            initial_checkpoint=None,
            refinement_scope="all",
            viewcell_region_conditioned_visibility=False,
        )
        _validate_extreme_tail_selection_contract(current)
        calibration_selected = SimpleNamespace(**vars(valid))
        calibration_selected.initial_checkpoint = Path("best_safe.pt")
        with self.assertRaisesRegex(ValueError, "fixed-epoch checkpoint"):
            _validate_extreme_tail_selection_contract(calibration_selected)

    def test_boundary_tail_selection_source_uses_current_final_scores(self) -> None:
        logits = torch.tensor([-2.0, 0.5, 1.0], requires_grad=True)
        self.assertIs(
            _boundary_tail_selection_logits("current_final", logits), logits
        )
        self.assertIsNone(
            _boundary_tail_selection_logits("frozen_base", logits)
        )
        with self.assertRaisesRegex(ValueError, "selection source"):
            _boundary_tail_selection_logits("unknown", logits)

    def test_weighted_quantile_tracks_visual_utility_mass(self) -> None:
        values = np.asarray([-2.0, -1.0, 3.0], dtype=np.float32)
        weights = np.asarray([0.01, 0.89, 0.10], dtype=np.float32)

        self.assertEqual(_weighted_quantile_numpy(values, weights, 0.005), -2.0)
        self.assertEqual(_weighted_quantile_numpy(values, weights, 0.05), -1.0)

    def test_tail_hard_pose_pool_selects_largest_train_risks(self) -> None:
        poses = np.asarray([10, 11, 12, 13, 14], dtype=np.int64)
        risks = np.asarray([0.1, 3.0, -0.5, 2.0, 1.0], dtype=np.float64)

        selected, meta = _select_tail_hard_pose_pool(poses, risks, 0.4)

        self.assertEqual(selected.tolist(), [11, 13])
        self.assertEqual(meta["poolPoseCount"], 2)
        self.assertEqual(meta["riskMaximum"], 3.0)

    def test_fixed_frontier_keeps_important_low_positive_and_high_negatives(self) -> None:
        positive_ids, negative_ids, meta = (
            _select_weighted_safety_frontier_members(
                np.asarray([10, 11, 12, 20, 21, 22], dtype=np.int64),
                np.asarray([-5.0, 2.0, 3.0, 1.5, 1.0, -1.0], dtype=np.float32),
                np.asarray([1, 1, 1, 0, 0, 0], dtype=np.float32),
                np.asarray([100.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                positive_mass_fraction=0.01,
                positive_count_cap=16,
                negative_top_fraction=0.5,
                minimum_negative_count=2,
                maximum_negative_count=2,
            )
        )

        self.assertIn(10, positive_ids.tolist())
        self.assertEqual(negative_ids.tolist(), [20, 21])
        self.assertGreaterEqual(meta["overlapPositiveCount"], 1)
        self.assertEqual(np.intersect1d(positive_ids, negative_ids).size, 0)

    def test_fixed_frontier_does_not_apply_legacy_top64_positive_cap(self) -> None:
        positive_count = 70
        negative_count = 100
        ids = np.arange(positive_count + negative_count, dtype=np.int64)
        scores = np.concatenate(
            [
                np.linspace(-2.0, -1.0, positive_count),
                np.linspace(2.0, 1.0, negative_count),
            ]
        ).astype(np.float32)
        labels = np.concatenate(
            [np.ones(positive_count), np.zeros(negative_count)]
        ).astype(np.float32)
        weights = labels.copy()

        positive_ids, negative_ids, _meta = (
            _select_weighted_safety_frontier_members(
                ids,
                scores,
                labels,
                weights,
                positive_mass_fraction=0.005,
                positive_count_cap=1024,
                negative_top_fraction=0.04,
                minimum_negative_count=8,
                maximum_negative_count=16,
            )
        )

        self.assertEqual(positive_ids.size, positive_count)
        self.assertEqual(negative_ids.size, 8)

    def test_fixed_frontier_has_an_independent_positive_count_cap(self) -> None:
        argv = [
            "train_bounded_relation_survival_moment_safety.py",
            "--dataset-dir",
            "dataset",
            "--relation-dir",
            "relations",
            "--runtime-meta",
            "runtime.json",
            "--initial-geo-features",
            "geometry.bin",
            "--glb-index",
            "glb.json",
            "--glb-root",
            "assets",
            "--output-dir",
            "output",
            "--experiment-name",
            "frontier-cap-test",
            "--variant",
            "full",
            "--boundary-tail-residual-positive-tail-count-cap",
            "64",
            "--boundary-tail-fixed-frontier-positive-count-cap",
            "1024",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()

        self.assertEqual(args.boundary_tail_residual_positive_tail_count_cap, 64)
        self.assertEqual(args.boundary_tail_fixed_frontier_positive_count_cap, 1024)

    def test_weighted_np_cli_is_disabled_by_default_and_explicitly_configurable(self) -> None:
        base = [
            "train",
            "--dataset-dir",
            "dataset",
            "--relation-dir",
            "relations",
            "--runtime-meta",
            "runtime.json",
            "--initial-geo-features",
            "geometry.bin",
            "--glb-index",
            "glb.json",
            "--glb-root",
            "assets",
            "--output-dir",
            "output",
        ]
        with mock.patch.object(sys, "argv", base):
            disabled = parse_args()
        self.assertEqual(disabled.boundary_tail_np_weight, 0.0)
        self.assertEqual(
            disabled.boundary_tail_np_weighted_recall_target, 0.995
        )

        with mock.patch.object(
            sys,
            "argv",
            base
            + [
                "--boundary-tail-np-weight",
                "0.5",
                "--boundary-tail-np-weighted-recall-target",
                "0.9975",
                "--boundary-tail-np-temperature",
                "0.2",
                "--boundary-tail-np-dual-learning-rate",
                "0.1",
                "--boundary-tail-np-dual-maximum",
                "12",
            ],
        ):
            configured = parse_args()
        self.assertEqual(configured.boundary_tail_np_weight, 0.5)
        self.assertEqual(
            configured.boundary_tail_np_weighted_recall_target, 0.9975
        )
        self.assertEqual(configured.boundary_tail_np_temperature, 0.2)
        self.assertEqual(configured.boundary_tail_np_dual_learning_rate, 0.1)
        self.assertEqual(configured.boundary_tail_np_dual_maximum, 12.0)

    def test_recurrent_negative_priority_keeps_repeated_train_only_failures(self) -> None:
        priority, meta = _recurrent_negative_priority_from_counts(
            np.asarray([100, 100, 8, 100], dtype=np.int64),
            np.asarray([30, 5, 5, 40], dtype=np.int64),
            np.asarray([4, 4, 4, 0], dtype=np.int64),
            baseline_tail_fraction=0.04,
            selected_instance_fraction=0.5,
            minimum_negative_exposures=16,
            minimum_hard_count=4,
            smoothing_strength=16.0,
        )

        self.assertGreater(float(priority[0]), 0.0)
        self.assertEqual(float(priority[1]), 0.0)
        self.assertEqual(float(priority[2]), 0.0)
        self.assertEqual(float(priority[3]), 0.0)
        self.assertEqual(meta["selectedInstanceCount"], 1)
        self.assertFalse(meta["testRead"])

    def test_candidate_frustum_boundary_proximity_is_monotone_and_finite(self) -> None:
        cameras = torch.zeros((2, 3))
        views = torch.tensor(
            [[0.0, 0.0, -1.0, 1.0, 1.0], [0.0, 0.0, -1.0, 1.0, 1.0]]
        )
        aabbs = torch.tensor(
            [
                [-0.5, -0.5, -10.5, 0.5, 0.5, -9.5],
                [9.5, -0.5, -10.5, 10.5, 0.5, -9.5],
            ]
        )
        proximity = _candidate_frustum_boundary_proximity(
            cameras,
            views,
            aabbs,
        )
        self.assertTrue(bool(torch.isfinite(proximity).all()))
        self.assertTrue(bool(((proximity >= 0.0) & (proximity <= 1.0)).all()))
        self.assertGreater(float(proximity[1]), float(proximity[0]))

    def test_candidate_frustum_boundary_proximity_rejects_non_candidates(self) -> None:
        camera = torch.zeros((1, 3), dtype=torch.float32)
        view = torch.tensor([[0.0, 0.0, -1.0, 1.0, 1.0]], dtype=torch.float32)
        behind_camera = torch.tensor(
            [[-0.1, -0.1, 0.5, 0.1, 0.1, 0.7]], dtype=torch.float32
        )

        with self.assertRaisesRegex(ValueError, "near-plane candidate contract"):
            _candidate_frustum_boundary_proximity(camera, view, behind_camera)

    def test_candidate_frustum_boundary_proximity_respects_axis_fov(self) -> None:
        camera = torch.zeros((2, 3), dtype=torch.float32)
        view = torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.5, 1.0],
                [0.0, 0.0, -1.0, 1.0, 0.5],
            ],
            dtype=torch.float32,
        )
        same_box = torch.tensor(
            [
                [0.35, -0.05, -1.05, 0.45, 0.05, -0.95],
                [0.35, -0.05, -1.05, 0.45, 0.05, -0.95],
            ],
            dtype=torch.float32,
        )

        proximity = _candidate_frustum_boundary_proximity(camera, view, same_box)

        self.assertGreater(float(proximity[0]), float(proximity[1]))

    def test_instance_exposure_weights_counter_instance_prior_frequency(self) -> None:
        candidates = {
            0: np.asarray([0, 1], dtype=np.uint32),
            1: np.asarray([0, 1], dtype=np.uint32),
            2: np.asarray([0, 1], dtype=np.uint32),
            3: np.asarray([0, 1], dtype=np.uint32),
        }
        visible = {
            0: np.asarray([0, 1], dtype=np.uint32),
            1: np.asarray([0], dtype=np.uint32),
            2: np.asarray([0], dtype=np.uint32),
            3: np.asarray([], dtype=np.uint32),
        }
        dataset = SimpleNamespace(
            frustum_slice=lambda pose: candidates[int(pose)],
            visible_slice=lambda pose: (
                visible[int(pose)], np.ones(visible[int(pose)].size, dtype=np.float32)
            ),
        )
        split = SimpleNamespace(pose_indices=np.arange(4, dtype=np.int64))
        positive, negative, meta = _instance_exposure_balance_weights(
            dataset, split, 2, power=1.0, maximum_weight=8.0
        )
        self.assertGreater(float(positive[1]), float(positive[0]))
        self.assertGreater(float(negative[0]), float(negative[1]))
        self.assertEqual(meta["sourceSplit"], "train")
        self.assertEqual(meta["positiveReferenceCount"], 4)

    def test_runtime_visibility_refinement_freezes_offline_parameters(self) -> None:
        model = BoundedRelationSurvivalMomentModel(3, 2)
        meta = _set_refinement_scope(
            model, "runtime_visibility", initialized=True
        )
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(any(name.startswith("visibility_head.") for name in trainable))
        self.assertTrue(any(name.startswith("moment_query.") for name in trainable))
        self.assertFalse(
            any(name.startswith("offline_survival_encoder.") for name in trainable)
        )
        self.assertFalse(
            any(name.startswith("instance_calibration_residual_raw") for name in trainable)
        )
        self.assertTrue(meta["offlineSurvivalTableFrozen"])
        with self.assertRaisesRegex(ValueError, "initial checkpoint"):
            _set_refinement_scope(model, "runtime_visibility", initialized=False)

    def test_view_residual_refinement_trains_only_fresh_view_head(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            view_residual_max_abs=2.0,
            view_residual_hidden_dim=8,
        )
        meta = _set_refinement_scope(model, "view_residual", initialized=True)
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("view_residual_") for name in trainable))
        self.assertEqual(meta["scope"], "view_residual")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])
        disabled = BoundedRelationSurvivalMomentModel(3, 2)
        with self.assertRaisesRegex(ValueError, "enabled view residual"):
            _set_refinement_scope(disabled, "view_residual", initialized=True)

    def test_boundary_opportunity_refinement_trains_only_fresh_head(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            spectral_mode="moment_extrema",
            boundary_opportunity_hidden_dim=12,
        )
        meta = _set_refinement_scope(
            model, "boundary_opportunity", initialized=True
        )
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("boundary_opportunity_") for name in trainable)
        )
        self.assertEqual(meta["scope"], "boundary_opportunity")

    def test_boundary_tail_residual_refinement_trains_only_fresh_head(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
        )
        meta = _set_refinement_scope(
            model, "boundary_tail_residual", initialized=True
        )
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(
            all(name.startswith("boundary_tail_residual_") for name in trainable)
        )
        self.assertEqual(meta["scope"], "boundary_tail_residual")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])
        disabled = BoundedRelationSurvivalMomentModel(3, 2)
        with self.assertRaisesRegex(ValueError, "enabled residual head"):
            _set_refinement_scope(
                disabled, "boundary_tail_residual", initialized=True
            )

    def test_viewcell_extreme_visibility_refinement_trains_main_score_path(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        meta = _set_refinement_scope(
            model, "viewcell_extreme_visibility", initialized=True
        )
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(
            any(
                name.startswith("viewcell_extreme_visibility_projection.")
                for name in trainable
            )
        )
        self.assertTrue(
            any(
                name.startswith("viewcell_extreme_visibility_head.")
                for name in trainable
            )
        )
        self.assertIn("visibility_head.weight", trainable)
        self.assertIn("visibility_head.bias", trainable)
        self.assertTrue(
            all(
                name.startswith("viewcell_extreme_visibility_")
                or name.startswith("visibility_head.")
                for name in trainable
            )
        )
        self.assertEqual(meta["scope"], "viewcell_extreme_visibility")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])
        disabled = BoundedRelationSurvivalMomentModel(3, 2)
        with self.assertRaisesRegex(ValueError, "enabled branch"):
            _set_refinement_scope(
                disabled, "viewcell_extreme_visibility", initialized=True
            )

    def test_viewcell_region_conditioned_visibility_refinement_trains_main_score_path(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        meta = _set_refinement_scope(
            model, "viewcell_region_conditioned_visibility", initialized=True
        )
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertIn("visibility_head.weight", trainable)
        self.assertIn("visibility_head.bias", trainable)
        self.assertTrue(
            all(
                name.startswith("viewcell_region_conditioned_visibility_")
                or name.startswith("visibility_head.")
                for name in trainable
            )
        )
        self.assertEqual(meta["scope"], "viewcell_region_conditioned_visibility")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])
        disabled = BoundedRelationSurvivalMomentModel(3, 2)
        with self.assertRaisesRegex(ValueError, "enabled branch"):
            _set_refinement_scope(
                disabled, "viewcell_region_conditioned_visibility", initialized=True
            )

    def test_viewcell_region_tail_refinement_trains_only_region_branch(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            3,
            2,
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )

        meta = _set_refinement_scope(
            model, "viewcell_region_tail", initialized=True
        )

        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }
        self.assertTrue(trainable)
        self.assertFalse(any(name.startswith("visibility_head.") for name in trainable))
        self.assertTrue(
            all(
                name.startswith("viewcell_region_conditioned_visibility_")
                for name in trainable
            )
        )
        self.assertEqual(meta["scope"], "viewcell_region_tail")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])

    def test_region_conditioned_visibility_cli_and_refinement_choice_are_explicit(self) -> None:
        argv = [
            "train",
            "--dataset-dir",
            "dataset",
            "--relation-dir",
            "relation",
            "--runtime-meta",
            "runtime.json",
            "--initial-geo-features",
            "geometry.bin",
            "--glb-index",
            "glb-index.json",
            "--glb-root",
            "glb-root",
            "--output-dir",
            "output",
            "--viewcell-region-conditioned-visibility",
            "--refinement-scope",
            "viewcell_region_conditioned_visibility",
            "--tail-selection-source",
            "initial_visibility",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertTrue(args.viewcell_region_conditioned_visibility)
        self.assertEqual(
            args.refinement_scope, "viewcell_region_conditioned_visibility"
        )
        self.assertEqual(args.tail_selection_source, "initial_visibility")

    def test_boundary_opportunity_scale_ramps_without_warmup(self) -> None:
        self.assertEqual(
            _boundary_opportunity_scale(0, 100, ramp_fraction=0.10), 0.0
        )
        self.assertEqual(
            _boundary_opportunity_scale(5, 100, ramp_fraction=0.10), 0.5
        )
        self.assertEqual(
            _boundary_opportunity_scale(10, 100, ramp_fraction=0.10), 1.0
        )
        self.assertEqual(
            _boundary_opportunity_scale(0, 100, ramp_fraction=0.0), 1.0
        )

    def test_boundary_opportunity_pose_pool_keeps_all_mixed_train_poses(self) -> None:
        dataset = SimpleNamespace(
            has_subpose_robust_labels=True,
            visible_slice=lambda pose: (
                np.asarray({1: [2, 3], 2: [4]}[int(pose)], dtype=np.uint32),
                np.ones({1: 2, 2: 1}[int(pose)], dtype=np.float32),
            ),
            frustum_slice=lambda pose: np.asarray(
                {1: [2, 3, 5], 2: [4, 6]}[int(pose)], dtype=np.uint32
            ),
            visible_hit_count_slice=lambda pose: {
                1: np.asarray([1, 8], dtype=np.uint16),
                2: np.asarray([4], dtype=np.uint16),
            }[int(pose)],
            subpose_count=lambda _pose: 20,
        )
        split = SimpleNamespace(
            split_name="train", pose_indices=np.asarray([1, 2], dtype=np.int64)
        )
        selected, meta = _boundary_opportunity_pose_pool(
            dataset, split, rare_threshold=0.05
        )
        self.assertEqual(selected.tolist(), [1, 2])
        self.assertEqual(meta["rarePositivePoseCount"], 1)
        self.assertEqual(meta["rarePositiveReferenceCount"], 1)
        with self.assertRaisesRegex(ValueError, "cannot read test"):
            _boundary_opportunity_pose_pool(
                dataset,
                SimpleNamespace(split_name="test", pose_indices=np.asarray([1])),
                rare_threshold=0.05,
            )

    def test_tail_refinement_can_mix_premined_hard_poses(self) -> None:
        mixed = np.asarray([1, 2, 3, 4], dtype=np.int64)
        hard = np.asarray([2, 4], dtype=np.int64)
        selected, fraction = _refinement_pose_sampling(
            "boundary_tail_residual", mixed, hard, 0.5
        )
        self.assertIs(selected, hard)
        self.assertEqual(fraction, 0.5)
        fallback, fallback_fraction = _refinement_pose_sampling(
            "boundary_tail_residual", mixed, None, 0.0
        )
        self.assertIs(fallback, mixed)
        self.assertEqual(fallback_fraction, 1.0)
        opportunity, opportunity_fraction = _refinement_pose_sampling(
            "boundary_opportunity", mixed, hard, 0.5
        )
        self.assertIs(opportunity, mixed)
        self.assertEqual(opportunity_fraction, 1.0)

    def test_visibility_head_refinement_freezes_hidden_representation(self) -> None:
        model = BoundedRelationSurvivalMomentModel(3, 2)
        meta = _set_refinement_scope(model, "visibility_head", initialized=True)
        trainable = {
            name for name, value in model.named_parameters() if value.requires_grad
        }

        self.assertEqual(
            trainable,
            {"visibility_head.weight", "visibility_head.bias"},
        )
        self.assertEqual(meta["scope"], "visibility_head")
        self.assertTrue(meta["offlineSurvivalTableFrozen"])

    def test_safety_boundary_ema_is_detached_and_finite(self) -> None:
        first = _update_safety_boundary_ema(
            torch.tensor(-2.0, requires_grad=True),
            None,
            decay=0.9,
        )
        updated = _update_safety_boundary_ema(
            torch.tensor(-1.0, requires_grad=True),
            first,
            decay=0.9,
        )
        self.assertFalse(first.requires_grad)
        self.assertFalse(updated.requires_grad)
        self.assertAlmostEqual(float(first), -2.0)
        self.assertAlmostEqual(float(updated), -1.9, places=6)
        with self.assertRaisesRegex(ValueError, "decay"):
            _update_safety_boundary_ema(torch.tensor(0.0), None, decay=1.0)

    def test_score_distribution_reports_extreme_safety_tails(self) -> None:
        summary = score_distribution_summary(
            np.asarray([0.01, 0.60, 0.90, 0.20, 0.80, 0.99]),
            np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            np.asarray([10.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        )
        for key in (
            "positiveWeightedQ005",
            "positiveWeightedQ01",
            "negativeQ99",
            "negativeQ995",
            "positiveNegativeGapQ01Q99",
            "positiveNegativeGapQ005Q995",
            "rocAuc",
            "averagePrecision",
            "weightedRocAuc",
        ):
            self.assertIn(key, summary)
            self.assertIsNotNone(summary[key])
        self.assertLess(summary["positiveNegativeGapQ005Q995"], 0.0)
        self.assertAlmostEqual(summary["rocAuc"], 1.0 / 3.0)
        self.assertAlmostEqual(summary["averagePrecision"], 0.5)
        self.assertAlmostEqual(summary["weightedRocAuc"], 1.0 / 12.0)

    def test_test_split_is_rejected_without_reading_it(self) -> None:
        dataset = SimpleNamespace(split_ids={"train", "calibration", "validation", "test"})
        with self.assertRaisesRegex(ValueError, "test is forbidden"):
            _resolve_split(dataset, "test", "validation")

    def test_validation_values_do_not_enter_calibration_workpoint_selection(self) -> None:
        calibration_rows = [
            {
                "threshold": 0.02,
                "aggregateWeightedRecall": 0.995,
                "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                "agg_useful_cull": 0.20,
                "agg_balanced_accuracy": 0.80,
                "agg_precision": 0.80,
                "avg_pred_count": 10.0,
            },
            {
                "threshold": 0.05,
                "aggregateWeightedRecall": 0.996,
                "aggregateWeightedRecallLowerConfidenceBound": 0.993,
                "agg_useful_cull": 0.40,
                "agg_balanced_accuracy": 0.85,
                "agg_precision": 0.85,
                "avg_pred_count": 8.0,
            },
        ]
        selected, diagnostic, frozen = _calibration_workpoints(calibration_rows)
        self.assertIsNotNone(selected)
        self.assertIsNotNone(diagnostic)
        self.assertIs(frozen, selected)
        self.assertEqual(frozen["threshold"], 0.05)
        self.assertEqual(CALIBRATION_FLOOR, 0.99)

    def test_validation_safety_gate_requires_point_and_lower_bound(self) -> None:
        self.assertTrue(
            _weighted_recall_safety_gate(
                {
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                }
            )
        )
        self.assertFalse(
            _weighted_recall_safety_gate(
                {
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.989,
                }
            )
        )
        self.assertFalse(
            _weighted_recall_safety_gate(
                {"aggregateWeightedRecall": 0.995}
            )
        )

    def test_four_objective_groups_sum_to_logged_total(self) -> None:
        values = [torch.tensor(value, requires_grad=True) for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)]
        groups = _v4_objective_groups(
            *values,
            survival_weight=0.25,
            relation_consistency_weight=0.10,
            utility_weight=0.10,
            download_weight=0.10,
            regularization_weight=1e-5,
            instance_calibration_regularization_weight=0.02,
        )
        summed = groups["safety"] + groups["relation"] + groups["schedule"] + groups["efficiency"]
        self.assertTrue(torch.allclose(groups["total"], summed))
        groups["total"].backward()
        self.assertTrue(all(value.grad is not None for value in values))

    def test_exposure_supervision_is_routed_through_protected_relation_group(self) -> None:
        values = [
            torch.tensor(value, requires_grad=True)
            for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
        ]
        exposure = torch.tensor(9.0, requires_grad=True)
        groups = _v4_objective_groups(
            *values,
            survival_weight=0.25,
            relation_consistency_weight=0.10,
            utility_weight=0.10,
            download_weight=0.10,
            regularization_weight=1e-5,
            instance_calibration_regularization_weight=0.02,
            exposure_supervision=exposure,
            exposure_supervision_weight=0.05,
        )
        expected_without_exposure = (
            0.25 * values[1]
            + 0.10 * values[2]
            + 1e-5 * values[5]
            + 0.02 * values[6]
        )
        self.assertTrue(
            torch.allclose(
                groups["relation"], expected_without_exposure + 0.05 * exposure
            )
        )
        groups["total"].backward()
        self.assertIsNotNone(exposure.grad)

    def test_boundary_tail_refinement_keeps_negative_efficiency_gradient(self) -> None:
        safety_parameter = torch.tensor(2.0, requires_grad=True)
        paired_safety_parameter = torch.tensor(1.5, requires_grad=True)
        efficiency_parameter = torch.tensor(3.0, requires_grad=True)
        groups = _boundary_tail_refinement_objective_groups(
            safety_parameter.square(),
            paired_safety_parameter.square(),
            efficiency_parameter.square(),
        )

        self.assertTrue(
            torch.equal(
                groups["safety"],
                safety_parameter.square() + paired_safety_parameter.square(),
            )
        )
        self.assertTrue(
            torch.equal(groups["efficiency"], efficiency_parameter.square())
        )
        self.assertTrue(
            torch.equal(
                groups["total"],
                groups["safety"] + groups["efficiency"],
            )
        )
        groups["total"].backward()
        self.assertIsNotNone(safety_parameter.grad)
        self.assertIsNotNone(paired_safety_parameter.grad)
        self.assertIsNotNone(efficiency_parameter.grad)

    def test_instance_calibration_schedule_warms_up_then_reaches_one(self) -> None:
        values = [
            _instance_calibration_blend(
                step,
                100,
                warmup_fraction=0.10,
                ramp_fraction=0.20,
            )
            for step in (0, 9, 10, 19, 29, 99)
        ]
        self.assertEqual(values[:2], [0.0, 0.0])
        self.assertGreater(values[2], 0.0)
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        self.assertEqual(values[-1], 1.0)

    def test_instance_calibration_reliability_reads_train_references_only(self) -> None:
        class Dataset:
            @staticmethod
            def frustum_slice(pose_index: int) -> np.ndarray:
                return {
                    0: np.asarray([0, 1], dtype=np.uint32),
                    1: np.asarray([0, 2], dtype=np.uint32),
                }[pose_index]

        reliability, meta = _instance_calibration_reliability(
            Dataset(),
            SimpleNamespace(pose_indices=np.asarray([0, 1], dtype=np.int64)),
            {"instance": np.asarray([0, 0, 2], dtype=np.uint32)},
            4,
        )
        self.assertEqual(reliability.shape, (4,))
        self.assertEqual(reliability[3], 0.0)
        self.assertGreater(reliability[0], reliability[1])
        self.assertEqual(meta["sourceSplit"], "train")
        self.assertEqual(len(meta["sha256Float32"]), 64)

    def test_logged_operating_thresholds_are_seed_step_deterministic(self) -> None:
        first, first_metrics = _operating_threshold_metrics(20260801, 3)
        repeated, repeated_metrics = _operating_threshold_metrics(20260801, 3)
        changed, _changed_metrics = _operating_threshold_metrics(20260801, 4)
        self.assertEqual(first, repeated)
        self.assertEqual(first_metrics, repeated_metrics)
        self.assertNotEqual(first, changed)
        self.assertEqual(first_metrics["operatingThresholdCount"], float(len(first)))
        self.assertTrue(all(np.isfinite(value) for value in first))

    def test_diagnostic_pose_limit_is_exact(self) -> None:
        self.assertEqual(_evaluation_batch_limits(0, 4), (4, None))
        self.assertEqual(_evaluation_batch_limits(2, 4), (2, 1))
        self.assertEqual(_evaluation_batch_limits(5, 4), (1, 5))
        with self.assertRaises(ValueError):
            _evaluation_batch_limits(-1, 4)

    def test_safe_checkpoint_alias_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _save_safe_checkpoint_alias(
                {"schema": "unit-test", "tensor": torch.tensor([1.0, 2.0])}, root
            )
            self.assertEqual(
                (root / "best_safe.pt").read_bytes(),
                (root / "best.pt").read_bytes(),
            )

    def test_initial_checkpoint_loads_weights_and_preserves_calibration_blend(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            source.set_instance_calibration_blend(0.75)
            with torch.no_grad():
                source.visibility_head.bias.fill_(1.25)
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 24,
                    "experimentName": "source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(3, 2)
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            self.assertEqual(initialization["mode"], "model-weights-only-fresh-optimizer")
            self.assertEqual(initialization["sourceEpoch"], 24)
            self.assertEqual(float(target.instance_calibration_blend), 0.75)
            self.assertTrue(
                torch.equal(target.visibility_head.bias, source.visibility_head.bias)
            )

    def test_initial_checkpoint_can_add_only_a_fresh_cull_certificate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            with torch.no_grad():
                source.visibility_head.bias.fill_(0.75)
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                cull_certificate_max_suppression=2.0,
                cull_certificate_initial_suppression=0.05,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertIn("fresh-cull-certificate", initialization["mode"])
            self.assertTrue(
                torch.equal(target.visibility_head.bias, source.visibility_head.bias)
            )
            self.assertIsNotNone(target.cull_certificate_head)
            assert target.cull_certificate_head is not None
            self.assertTrue(
                torch.equal(
                    target.cull_certificate_head.weight,
                    torch.zeros_like(target.cull_certificate_head.weight),
                )
            )

    def test_initial_checkpoint_can_add_only_a_fresh_view_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            with torch.no_grad():
                source.visibility_head.bias.fill_(0.75)
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                view_residual_max_abs=2.0,
                view_residual_hidden_dim=8,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertIn("fresh-view-residual", initialization["mode"])
            self.assertTrue(initialization["freshViewResidual"])
            self.assertTrue(
                torch.equal(target.visibility_head.bias, source.visibility_head.bias)
            )
            self.assertIsNotNone(target.view_residual_head)
            assert target.view_residual_head is not None
            output = target.view_residual_head[-1]
            self.assertTrue(torch.equal(output.weight, torch.zeros_like(output.weight)))
            self.assertTrue(torch.equal(output.bias, torch.zeros_like(output.bias)))

    def test_initial_checkpoint_can_zero_migrate_moment_extrema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 24,
                    "experimentName": "source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3, 2, spectral_mode="moment_extrema"
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["migratedMomentExtrema"])
            source_weight = source.boundary_summary_head[0].weight
            target_weight = target.boundary_summary_head[0].weight
            torch.testing.assert_close(
                target_weight[:, : source_weight.shape[1]], source_weight
            )
            torch.testing.assert_close(
                target_weight[:, source_weight.shape[1] :],
                torch.zeros_like(target_weight[:, source_weight.shape[1] :]),
            )

    def test_initial_checkpoint_can_add_extrema_and_boundary_opportunity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            inputs = {
                "ids": torch.arange(3),
                "center": torch.zeros((3, 9)),
                "axes": torch.zeros((3, 9, 2)),
                "depth": torch.full((3, 1), 0.5),
                "runtime": torch.randn((3, 124)) * 0.05,
            }
            source_logits, _ = source.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                boundary_opportunity_hidden_dim=12,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["migratedMomentExtrema"])
            self.assertTrue(initialization["freshBoundaryOpportunity"])
            target_logits, aux = target.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.testing.assert_close(target_logits, source_logits)
            self.assertEqual(float(target.boundary_opportunity_scale), 0.0)
            self.assertTrue(
                torch.equal(
                    aux["boundary_opportunity_logit_uplift"],
                    torch.zeros_like(aux["boundary_opportunity_logit_uplift"]),
                )
            )

    def test_initial_checkpoint_can_add_extrema_and_boundary_tail_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            inputs = {
                "ids": torch.arange(3),
                "center": torch.zeros((3, 9)),
                "axes": torch.zeros((3, 9, 2)),
                "depth": torch.full((3, 1), 0.5),
                "runtime": torch.randn((3, 124)) * 0.05,
                "pose_offsets": torch.tensor([0, 2, 3]),
            }
            source_logits, _ = source.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                boundary_tail_residual_hidden_dim=12,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["migratedMomentExtrema"])
            self.assertTrue(initialization["freshBoundaryTailResidual"])
            target_logits, aux = target.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
                pose_offsets=inputs["pose_offsets"],
            )
            torch.testing.assert_close(target_logits, source_logits)
            torch.testing.assert_close(
                aux["boundary_tail_residual_centered"],
                torch.zeros_like(aux["boundary_tail_residual_centered"]),
            )
            for start, end in ((0, 2), (2, 3)):
                self.assertAlmostEqual(
                    float(aux["boundary_tail_residual_centered"][start:end].mean()),
                    0.0,
                    places=7,
                )

    def test_initial_checkpoint_can_keep_tail_residual_and_add_opportunity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                boundary_tail_residual_hidden_dim=12,
            )
            assert source.boundary_tail_residual_head is not None
            with torch.no_grad():
                source.boundary_tail_residual_head[-1].weight.fill_(0.1)
                source.boundary_tail_residual_head[-1].bias.fill_(0.05)
            inputs = {
                "ids": torch.arange(3),
                "center": torch.zeros((3, 9)),
                "axes": torch.zeros((3, 9, 2)),
                "depth": torch.full((3, 1), 0.5),
                "runtime": torch.randn((3, 124)) * 0.05,
                "pose_offsets": torch.tensor([0, 2, 3]),
            }
            source_logits, _ = source.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
                pose_offsets=inputs["pose_offsets"],
            )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 4,
                    "experimentName": "tail-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                boundary_tail_residual_hidden_dim=12,
                boundary_opportunity_hidden_dim=12,
            )
            initialization = _initialize_from_checkpoint(target, path)
            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["freshBoundaryOpportunity"])
            self.assertFalse(initialization["freshBoundaryTailResidual"])
            target_logits, _ = target.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
                pose_offsets=inputs["pose_offsets"],
            )
            torch.testing.assert_close(target_logits, source_logits)
            torch.testing.assert_close(
                target.boundary_tail_residual_head[-1].weight,
                source.boundary_tail_residual_head[-1].weight,
            )

    def test_initial_checkpoint_can_add_extrema_main_visibility_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            inputs = {
                "ids": torch.arange(3),
                "center": torch.zeros((3, 9)),
                "axes": torch.zeros((3, 9, 2)),
                "depth": torch.full((3, 1), 0.5),
                "runtime": torch.randn((3, 124)) * 0.05,
            }
            source_logits, _ = source.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                viewcell_extreme_visibility_enabled=True,
            )

            initialization = _initialize_from_checkpoint(target, path)

            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["migratedMomentExtrema"])
            self.assertTrue(initialization["freshViewcellExtremeVisibility"])
            target_logits, aux = target.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.testing.assert_close(target_logits, source_logits)
            torch.testing.assert_close(
                aux["viewcell_extreme_visibility_logit"],
                torch.zeros_like(aux["viewcell_extreme_visibility_logit"]),
            )

    def test_initial_checkpoint_can_add_extrema_region_conditioned_visibility_branch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.pt"
            source = BoundedRelationSurvivalMomentModel(3, 2)
            inputs = {
                "ids": torch.arange(3),
                "center": torch.zeros((3, 9)),
                "axes": torch.zeros((3, 9, 2)),
                "depth": torch.full((3, 1), 0.5),
                "runtime": torch.randn((3, 124)) * 0.05,
            }
            source_logits, _ = source.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.save(
                {
                    "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
                    "runtimeSchema": MODEL_SCHEMA,
                    "testRead": False,
                    "epoch": 80,
                    "experimentName": "formal-source",
                    "modelConfig": source.config,
                    "modelState": source.state_dict(),
                },
                path,
            )
            target = BoundedRelationSurvivalMomentModel(
                3,
                2,
                spectral_mode="moment_extrema",
                viewcell_region_conditioned_visibility_enabled=True,
            )

            initialization = _initialize_from_checkpoint(target, path)

            self.assertIsNotNone(initialization)
            assert initialization is not None
            self.assertTrue(initialization["migratedMomentExtrema"])
            self.assertTrue(initialization["freshViewcellRegionConditionedVisibility"])
            self.assertIn(
                "fresh-viewcell-region-conditioned-visibility",
                initialization["mode"],
            )
            target_logits, aux = target.forward_batch(
                inputs["ids"],
                inputs["center"],
                inputs["axes"],
                inputs["depth"],
                {"runtime_features": inputs["runtime"]},
            )
            torch.testing.assert_close(target_logits, source_logits)
            torch.testing.assert_close(
                aux["viewcell_region_conditioned_visibility_logit"],
                torch.zeros_like(
                    aux["viewcell_region_conditioned_visibility_logit"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
