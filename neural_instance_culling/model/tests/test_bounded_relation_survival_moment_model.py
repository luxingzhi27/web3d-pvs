from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODEL = Path(__file__).resolve().parents[1]
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from bounded_relation_survival_moment_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    EXPOSURE_SUPERVISION_SOURCES,
    RELATION_CONTRAST_DIM,
    RUNTIME_FEATURE_DIM,
    RUNTIME_RELATION_FEATURE_MODES,
    VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM,
    VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM,
    VIEW_RESIDUAL_DYNAMIC_INPUT_DIM,
)


class BoundedRelationSurvivalMomentModelContractTest(unittest.TestCase):
    def _model(
        self,
        *,
        spectral_mode: str = "moment_envelope",
        instance_calibration_mode: str = "residual",
        view_residual_max_abs: float = 0.0,
        view_residual_hidden_dim: int = 16,
        boundary_opportunity_hidden_dim: int = 0,
        boundary_tail_residual_hidden_dim: int = 0,
        boundary_tail_residual_centering: str = "pose_mean",
        boundary_tail_residual_shortcut: str = "none",
        boundary_tail_residual_fusion: str = "product",
        boundary_tail_residual_output_init_std: float = 0.0,
        viewcell_extreme_visibility_enabled: bool = False,
        viewcell_region_conditioned_visibility_enabled: bool = False,
        viewcell_region_conditioned_visibility_centering: str = "none",
        query_tail_separator_family: str = "disabled",
        query_tail_separator_hidden_dim: int = 8,
        exposure_supervision_hidden_dim: int = 0,
        runtime_relation_feature_mode: str = "basis",
        exposure_supervision_source: str = "hidden",
    ) -> BoundedRelationSurvivalMomentModel:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            spectral_mode=spectral_mode,
            instance_calibration_mode=instance_calibration_mode,
            depth_q01=0.0,
            depth_q99=4.0,
            view_residual_max_abs=view_residual_max_abs,
            view_residual_hidden_dim=view_residual_hidden_dim,
            boundary_opportunity_hidden_dim=boundary_opportunity_hidden_dim,
            boundary_tail_residual_hidden_dim=boundary_tail_residual_hidden_dim,
            boundary_tail_residual_centering=boundary_tail_residual_centering,
            boundary_tail_residual_shortcut=boundary_tail_residual_shortcut,
            boundary_tail_residual_fusion=boundary_tail_residual_fusion,
            boundary_tail_residual_output_init_std=(
                boundary_tail_residual_output_init_std
            ),
            viewcell_extreme_visibility_enabled=viewcell_extreme_visibility_enabled,
            viewcell_region_conditioned_visibility_enabled=(
                viewcell_region_conditioned_visibility_enabled
            ),
            viewcell_region_conditioned_visibility_centering=(
                viewcell_region_conditioned_visibility_centering
            ),
            query_tail_separator_family=query_tail_separator_family,
            query_tail_separator_hidden_dim=query_tail_separator_hidden_dim,
            exposure_supervision_hidden_dim=exposure_supervision_hidden_dim,
            runtime_relation_feature_mode=runtime_relation_feature_mode,
            exposure_supervision_source=exposure_supervision_source,
        )
        model.set_instance_world_aabbs(
            torch.tensor(
                [
                    [-1.0, -1.0, 4.0, 1.0, 1.0, 6.0],
                    [2.0, -1.0, 7.0, 4.0, 1.0, 9.0],
                    [-4.0, -2.0, 9.0, -2.0, 2.0, 11.0],
                    [0.0, -1.0, 13.0, 2.0, 1.0, 15.0],
                ],
                dtype=torch.float32,
            )
        )
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        return model

    @staticmethod
    def _inputs() -> dict[str, torch.Tensor]:
        torch.manual_seed(20260814)
        return {
            "camera": torch.zeros((4, 3), dtype=torch.float32),
            "camera_view": torch.tensor(
                [[0.0, 0.0, 1.0, 1.0, 0.65]], dtype=torch.float32
            ).expand(4, -1),
            "candidate_camera": torch.tensor(
                [[0.0, 0.0, -3.0]], dtype=torch.float32
            ).expand(4, -1),
            "query_center": torch.zeros((4, 3), dtype=torch.float32),
            "radius": torch.full((4, 1), 2.0, dtype=torch.float32),
            "instance_ids": torch.arange(4, dtype=torch.long),
            "runtime": torch.randn((4, RUNTIME_FEATURE_DIM), dtype=torch.float32) * 0.05,
        }

    def test_runtime_table_contract_is_124_fp16_values_per_instance(self) -> None:
        model = self._model()
        schema = model.export_schema()
        self.assertEqual(RUNTIME_FEATURE_DIM, 124)
        self.assertEqual(schema["fixedTable"]["shape"], ["N", 124])
        self.assertEqual(schema["fixedTable"]["dtype"], "float16")
        self.assertEqual(
            sum(int(row["dim"]) for row in schema["fixedTable"]["layout"]),
            124,
        )

    def test_saved_model_config_declares_runtime_and_offline_widths(self) -> None:
        config = self._model().config
        self.assertEqual(config["hiddenDim"], 64)
        self.assertEqual(config["relationHiddenDim"], 64)
        self.assertEqual(config["runtimeRelationFeatureMode"], "basis")
        self.assertEqual(config["exposureSupervisionSource"], "hidden")
        relation_feature = config["runtimeRelationFeature"]
        self.assertEqual(relation_feature["basisDim"], RELATION_CONTRAST_DIM)
        self.assertEqual(relation_feature["contrastDim"], RELATION_CONTRAST_DIM)
        self.assertEqual(relation_feature["relationConditionSlice"], [0, 4])
        self.assertEqual(
            relation_feature["formula"],
            "relation_contrast = basis * (1 + tanh(relation_condition[:, :4]))",
        )
        self.assertEqual(relation_feature["mainTrunkInput"], "basis")
        self.assertEqual(relation_feature["survivalSemanticInput"], "basis")
        self.assertEqual(config["instanceCalibration"]["mode"], "residual")
        self.assertEqual(config["instanceCalibration"]["runtimeExport"], "fused coefficients only")
        exposure = config["viewcellExposureSupervision"]
        self.assertEqual(exposure["source"], "hidden")
        self.assertEqual(exposure["inputDim"], 64)
        self.assertEqual(exposure["architecture"], None)
        self.assertFalse(config["queryTailSeparator"]["enabled"])

    def test_relation_contrast_formula_is_query_conditioned_and_semantics_stay_on_basis(self) -> None:
        inputs = self._inputs()
        basis_model = self._model(runtime_relation_feature_mode="basis")
        contrast_model = self._model(runtime_relation_feature_mode="gated_contrast")
        contrast_model.load_state_dict(basis_model.state_dict(), strict=True)

        _, basis_aux = basis_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        _, contrast_aux = contrast_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )

        expected_contrast = contrast_aux["survival_direction_basis"] * (
            1.0 + torch.tanh(contrast_aux["relation_condition"][:, :4])
        )
        torch.testing.assert_close(
            basis_aux["relation_contrast"], expected_contrast, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            contrast_aux["relation_contrast"], expected_contrast, rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            basis_aux["survival_semantic"],
            contrast_aux["survival_semantic"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            basis_aux["runtime_relation_feature"],
            basis_aux["survival_direction_basis"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            contrast_aux["runtime_relation_feature"],
            contrast_aux["relation_contrast"],
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(contrast_aux["runtime_relation_feature"].shape, (4, 4))
        self.assertFalse(
            torch.equal(
                basis_aux["runtime_relation_feature"],
                contrast_aux["runtime_relation_feature"],
            )
        )
        self.assertEqual(contrast_model.config["runtimeHeadInputDim"], 130)
        self.assertEqual(contrast_model.export_schema()["fixedTable"]["shape"], ["N", 124])
        self.assertEqual(
            contrast_model.export_schema()["outputs"]["runtimeRelationFeature"],
            ["B", RELATION_CONTRAST_DIM],
        )

    def test_relation_feature_and_exposure_mode_validation(self) -> None:
        for mode in RUNTIME_RELATION_FEATURE_MODES:
            self.assertEqual(
                self._model(runtime_relation_feature_mode=mode).config[
                    "runtimeRelationFeatureMode"
                ],
                mode,
            )
        for source in EXPOSURE_SUPERVISION_SOURCES:
            self.assertEqual(
                self._model(exposure_supervision_source=source).config[
                    "exposureSupervisionSource"
                ],
                source,
            )
        with self.assertRaisesRegex(ValueError, "runtime_relation_feature_mode"):
            self._model(runtime_relation_feature_mode="invalid")
        with self.assertRaisesRegex(ValueError, "exposure_supervision_source"):
            self._model(exposure_supervision_source="invalid")

    def test_exposure_supervision_head_is_training_only_and_runtime_shape_is_unchanged(self) -> None:
        torch.manual_seed(20260819)
        baseline = self._model()
        torch.manual_seed(20260819)
        supervised = self._model(exposure_supervision_hidden_dim=16)
        baseline_state = baseline.state_dict()
        shared_state = {
            key: value
            for key, value in baseline_state.items()
            if not key.startswith("exposure_supervision_head.")
        }
        supervised.load_state_dict(shared_state, strict=False)
        self.assertEqual(supervised.exposure_supervision_head[0].in_features, 64)
        inputs = self._inputs()
        baseline.train()
        supervised.train()
        baseline_logits, _ = baseline.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        supervised_logits, supervised_aux = supervised.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        torch.testing.assert_close(supervised_logits, baseline_logits)
        self.assertEqual(
            tuple(supervised_aux["viewcell_exposure_supervision_logits"].shape),
            (4, 1),
        )
        supervised.eval()
        _, eval_aux = supervised.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        self.assertNotIn("viewcell_exposure_supervision_logits", eval_aux)
        schema = supervised.export_schema()
        self.assertEqual(schema["fixedTable"]["shape"], ["N", 124])
        self.assertTrue(
            schema["modelConfig"]["viewcellExposureSupervision"]["trainingOnly"]
        )
        self.assertFalse(
            schema["modelConfig"]["viewcellExposureSupervision"]["runtimeExport"]
        )

    def test_relation_contrast_exposure_head_reads_4d_only_and_is_train_only(self) -> None:
        model = self._model(
            exposure_supervision_hidden_dim=16,
            exposure_supervision_source="relation_contrast",
        )
        self.assertIsNotNone(model.exposure_supervision_head)
        assert model.exposure_supervision_head is not None
        self.assertEqual(model.exposure_supervision_head[0].in_features, 4)
        self.assertEqual(
            model.config["viewcellExposureSupervision"]["inputDim"],
            RELATION_CONTRAST_DIM,
        )
        self.assertEqual(
            model.config["viewcellExposureSupervision"]["architecture"],
            [RELATION_CONTRAST_DIM, 16, 1],
        )
        self.assertEqual(
            model.export_schema()["modelConfig"]["viewcellExposureSupervision"]["source"],
            "relation_contrast",
        )

        inputs = self._inputs()
        model.train()
        _, train_aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertEqual(
            train_aux["viewcell_exposure_supervision_logits"].shape,
            (4, 1),
        )
        model.eval()
        _, eval_aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertNotIn("viewcell_exposure_supervision_logits", eval_aux)

    def test_query_tail_separator_families_are_zero_initialized_and_share_108d_input(self) -> None:
        inputs = self._inputs()
        for family in ("linear", "hinge", "mlp"):
            with self.subTest(family=family):
                torch.manual_seed(20260819)
                baseline = self._model()
                torch.manual_seed(20260819)
                separated = self._model(query_tail_separator_family=family)
                baseline_state = baseline.state_dict()
                for key, value in baseline_state.items():
                    torch.testing.assert_close(
                        separated.state_dict()[key], value, rtol=0.0, atol=0.0
                    )
                separated.load_state_dict(baseline_state, strict=False)
                baseline_logits, _ = baseline.compute_logits_with_aux(
                    inputs["camera"],
                    inputs["camera_view"],
                    inputs["candidate_camera"],
                    inputs["instance_ids"],
                    runtime_features=inputs["runtime"],
                    query_center_world=inputs["query_center"],
                    viewcell_radius_m=inputs["radius"],
                    pose_offsets=torch.tensor([0, 2, 4]),
                )
                logits, aux = separated.compute_logits_with_aux(
                    inputs["camera"],
                    inputs["camera_view"],
                    inputs["candidate_camera"],
                    inputs["instance_ids"],
                    runtime_features=inputs["runtime"],
                    query_center_world=inputs["query_center"],
                    viewcell_radius_m=inputs["radius"],
                    pose_offsets=torch.tensor([0, 2, 4]),
                )
                torch.testing.assert_close(logits, baseline_logits, rtol=0.0, atol=0.0)
                self.assertEqual(aux["query_tail_separator_raw_features"].shape, (4, 108))
                self.assertTrue(
                    torch.equal(
                        aux["query_tail_separator_residual"],
                        torch.zeros_like(aux["query_tail_separator_residual"]),
                    )
                )
                self.assertEqual(
                    separated.config["queryTailSeparator"]["family"], family
                )
                self.assertEqual(separated.config["runtimeFeatureDim"], 124)

    def test_query_tail_separator_is_bounded_pose_centered_and_differentiable(self) -> None:
        model = self._model(query_tail_separator_family="linear")
        assert isinstance(model.query_tail_separator, torch.nn.Linear)
        with torch.no_grad():
            model.query_tail_separator.weight.fill_(0.2)
            model.query_tail_separator.bias.fill_(0.1)
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        residual = aux["query_tail_separator_residual"]
        for start, end in ((0, 2), (2, 4)):
            self.assertAlmostEqual(float(residual[start:end].mean()), 0.0, places=6)
        self.assertLessEqual(float(residual.abs().max()), 0.5 + 1e-6)
        logits.sum().backward()
        self.assertIsNotNone(model.query_tail_separator.weight.grad)

    def test_moment_extrema_adds_analytic_query_without_runtime_assets(self) -> None:
        model = self._model(spectral_mode="moment_extrema")
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertEqual(logits.shape, (4, 1))
        self.assertEqual(aux["viewcell_extreme_features"].shape, (4, 17))
        self.assertFalse(
            model.config["viewcellExtremeEnvelope"]["additionalPerInstanceAssets"]
        )

    def test_viewcell_extreme_visibility_disabled_is_the_existing_model_contract(self) -> None:
        torch.manual_seed(20260818)
        reference = self._model(spectral_mode="moment_extrema")
        disabled = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=False,
        )
        disabled.load_state_dict(reference.state_dict(), strict=True)
        inputs = self._inputs()
        reference_logits, reference_aux = reference.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        disabled_logits, disabled_aux = disabled.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertTrue(torch.equal(reference_logits, disabled_logits))
        self.assertEqual(set(reference_aux), set(disabled_aux))
        self.assertIsNone(disabled.viewcell_extreme_visibility_projection)
        self.assertIsNone(disabled.viewcell_extreme_visibility_head)
        self.assertNotIn("viewcellExtremeVisibility", disabled.config)
        self.assertNotIn("viewcellExtremeVisibility", disabled.export_schema())
        self.assertFalse(
            any("viewcell_extreme_visibility" in key for key in disabled.state_dict())
        )

    def test_viewcell_extreme_visibility_config_and_state_reconstruct(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        config = model.config["viewcellExtremeVisibility"]
        self.assertTrue(config["enabled"])
        self.assertEqual(
            config["inputDim"], VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM
        )
        self.assertEqual(config["inputDim"], 27)
        self.assertEqual(config["queryAuxKey"], "viewcell_extreme_features")
        self.assertEqual(config["queryAuxFeatureDim"], 17)
        self.assertEqual(
            config["projectionDim"], VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM
        )
        self.assertEqual(config["activation"], "SiLU")
        self.assertEqual(config["poseReduction"], "none")
        self.assertFalse(config["boundedCorrection"])
        schema = model.export_schema()
        self.assertEqual(
            schema["viewcellExtremeVisibility"], config
        )
        self.assertEqual(
            schema["outputs"]["viewcellExtremeVisibilityLogit"], ["B", 1]
        )

        reconstructed = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=config["enabled"],
        )
        load_result = reconstructed.load_state_dict(model.state_dict(), strict=True)
        self.assertFalse(load_result.missing_keys)
        self.assertFalse(load_result.unexpected_keys)
        self.assertEqual(
            set(model.state_dict()), set(reconstructed.state_dict())
        )

    def test_old_checkpoint_migrates_to_zero_initialized_viewcell_extreme_visibility(self) -> None:
        old_model = self._model(spectral_mode="moment_extrema")
        new_model = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        load_result = new_model.load_state_dict(old_model.state_dict(), strict=False)
        self.assertTrue(load_result.missing_keys)
        self.assertTrue(
            all("viewcell_extreme_visibility" in key for key in load_result.missing_keys)
        )
        self.assertFalse(load_result.unexpected_keys)
        inputs = self._inputs()
        old_logits, _ = old_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        new_logits, new_aux = new_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertEqual(
            new_aux["viewcell_extreme_visibility_input"].shape,
            (4, VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM),
        )
        self.assertTrue(
            torch.equal(
                new_aux["viewcell_extreme_visibility_logit"],
                torch.zeros_like(new_aux["viewcell_extreme_visibility_logit"]),
            )
        )

    def test_viewcell_extreme_visibility_uses_27d_input_and_reaches_main_logit(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        assert model.viewcell_extreme_visibility_projection is not None
        assert model.viewcell_extreme_visibility_head is not None
        projection = model.viewcell_extreme_visibility_projection[0]
        head = model.viewcell_extreme_visibility_head
        self.assertEqual(projection.in_features, 27)
        self.assertEqual(
            projection.out_features, VIEWCELL_EXTREME_VISIBILITY_PROJECTION_DIM
        )
        self.assertIsInstance(model.viewcell_extreme_visibility_projection[1], torch.nn.SiLU)
        with torch.no_grad():
            projection.weight.fill_(0.05)
            projection.bias.fill_(0.01)
            head.weight.fill_(0.1)
            head.bias.zero_()

        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertEqual(
            aux["viewcell_extreme_visibility_input"].shape,
            (4, VIEWCELL_EXTREME_VISIBILITY_INPUT_DIM),
        )
        self.assertTrue(
            bool((aux["viewcell_extreme_visibility_logit"].abs() > 0.0).any())
        )
        logits.sum().backward()
        self.assertIsNotNone(projection.weight.grad)
        self.assertIsNotNone(projection.bias.grad)
        self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(projection.bias.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.weight.grad)
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)

    def test_viewcell_extreme_visibility_rejects_wrong_extreme_width(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        with self.assertRaisesRegex(RuntimeError, "viewcell_extreme_features"):
            model._viewcell_extreme_visibility_region_input(
                {"viewcell_extreme_features": torch.zeros((4, 16))},
                torch.zeros((4, 9)),
                torch.zeros((4, 1)),
            )

    def test_viewcell_extreme_visibility_keeps_124d_runtime_table(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_extreme_visibility_enabled=True,
        )
        self.assertEqual(model.config["runtimeFeatureDim"], 124)
        self.assertEqual(model.export_schema()["fixedTable"]["shape"], ["N", 124])
        self.assertEqual(
            sum(int(row["dim"]) for row in model.export_schema()["fixedTable"]["layout"]),
            124,
        )

    def test_viewcell_region_conditioned_visibility_disabled_has_no_new_state_or_schema(self) -> None:
        model = self._model(spectral_mode="moment_extrema")
        self.assertIsNone(model.viewcell_region_conditioned_visibility_region_projection)
        self.assertIsNone(model.viewcell_region_conditioned_visibility_hidden_projection)
        self.assertIsNone(model.viewcell_region_conditioned_visibility_head)
        self.assertFalse(
            any(
                "viewcell_region_conditioned_visibility" in key
                for key in model.state_dict()
            )
        )
        self.assertNotIn("viewcellRegionConditionedVisibility", model.config)
        self.assertNotIn(
            "viewcellRegionConditionedVisibility", model.export_schema()
        )

    def test_old_checkpoint_is_initially_equivalent_for_region_conditioned_visibility(self) -> None:
        old_model = self._model(spectral_mode="moment_extrema")
        new_model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        load_result = new_model.load_state_dict(old_model.state_dict(), strict=False)
        self.assertTrue(load_result.missing_keys)
        self.assertTrue(
            all(
                "viewcell_region_conditioned_visibility" in key
                for key in load_result.missing_keys
            )
        )
        self.assertFalse(load_result.unexpected_keys)
        assert new_model.viewcell_region_conditioned_visibility_head is not None
        self.assertTrue(
            torch.equal(
                new_model.viewcell_region_conditioned_visibility_head[-1].weight,
                torch.zeros_like(
                    new_model.viewcell_region_conditioned_visibility_head[-1].weight
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                new_model.viewcell_region_conditioned_visibility_head[-1].bias,
                torch.zeros_like(
                    new_model.viewcell_region_conditioned_visibility_head[-1].bias
                ),
            )
        )
        inputs = self._inputs()
        old_logits, _ = old_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        new_logits, new_aux = new_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        self.assertTrue(torch.equal(old_logits, new_logits))
        self.assertTrue(
            torch.equal(
                new_aux["viewcell_region_conditioned_visibility_logit"],
                torch.zeros_like(
                    new_aux["viewcell_region_conditioned_visibility_logit"]
                ),
            )
        )

    def test_viewcell_region_conditioned_visibility_interaction_changes_main_logit(self) -> None:
        base_model = self._model(spectral_mode="moment_extrema")
        conditioned_model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        conditioned_model.load_state_dict(base_model.state_dict(), strict=False)
        assert conditioned_model.viewcell_region_conditioned_visibility_region_projection is not None
        assert conditioned_model.viewcell_region_conditioned_visibility_hidden_projection is not None
        assert conditioned_model.viewcell_region_conditioned_visibility_head is not None
        with torch.no_grad():
            conditioned_model.viewcell_region_conditioned_visibility_region_projection[0].weight.fill_(0.05)
            conditioned_model.viewcell_region_conditioned_visibility_region_projection[0].bias.fill_(0.01)
            conditioned_model.viewcell_region_conditioned_visibility_hidden_projection[0].weight.fill_(0.05)
            conditioned_model.viewcell_region_conditioned_visibility_hidden_projection[0].bias.fill_(0.01)
            conditioned_model.viewcell_region_conditioned_visibility_head[0].weight.fill_(0.05)
            conditioned_model.viewcell_region_conditioned_visibility_head[0].bias.fill_(0.01)
            conditioned_model.viewcell_region_conditioned_visibility_head[-1].weight.fill_(0.1)
            conditioned_model.viewcell_region_conditioned_visibility_head[-1].bias.zero_()
        inputs = self._inputs()
        base_logits, _ = base_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        conditioned_logits, aux = conditioned_model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        branch_logit = aux["viewcell_region_conditioned_visibility_logit"]
        self.assertTrue(bool((branch_logit.abs() > 0.0).any()))
        self.assertFalse(torch.equal(base_logits, conditioned_logits))
        torch.testing.assert_close(
            conditioned_logits - base_logits,
            branch_logit,
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(
            aux["viewcell_region_conditioned_visibility_fusion"].shape,
            (4, VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM),
        )

    def test_viewcell_region_conditioned_visibility_gradients_reach_both_projections(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        assert model.viewcell_region_conditioned_visibility_region_projection is not None
        assert model.viewcell_region_conditioned_visibility_hidden_projection is not None
        assert model.viewcell_region_conditioned_visibility_head is not None
        with torch.no_grad():
            model.viewcell_region_conditioned_visibility_head[-1].weight.fill_(0.1)
            model.viewcell_region_conditioned_visibility_head[-1].bias.zero_()
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        logits.sum().backward()
        for module in (
            model.viewcell_region_conditioned_visibility_region_projection,
            model.viewcell_region_conditioned_visibility_hidden_projection,
        ):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                assert parameter.grad is not None
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)
        self.assertEqual(
            aux["viewcell_region_conditioned_visibility_region_projection"].shape,
            (4, VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM),
        )
        self.assertEqual(
            aux["viewcell_region_conditioned_visibility_hidden_projection"].shape,
            (4, VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM),
        )

    def test_viewcell_region_conditioned_visibility_rejects_wrong_dimensions(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        with self.assertRaisesRegex(RuntimeError, "hidden input"):
            model._viewcell_region_conditioned_visibility_logit(
                torch.zeros((4, model.hidden_dim - 1)),
                {},
                torch.zeros((4, 9)),
                torch.zeros((4, 1)),
            )
        with self.assertRaisesRegex(RuntimeError, "viewcell_extreme_features"):
            model._viewcell_region_conditioned_visibility_logit(
                torch.zeros((4, model.hidden_dim)),
                {"viewcell_extreme_features": torch.zeros((4, 16))},
                torch.zeros((4, 9)),
                torch.zeros((4, 1)),
            )

    def test_viewcell_region_conditioned_visibility_schema_and_runtime_table(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
        )
        config = model.config["viewcellRegionConditionedVisibility"]
        self.assertTrue(config["enabled"])
        self.assertEqual(
            config["regionInputDim"],
            VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM,
        )
        self.assertEqual(config["regionInputDim"], 27)
        self.assertEqual(config["queryAuxKey"], "viewcell_extreme_features")
        self.assertEqual(config["queryAuxFeatureDim"], 17)
        self.assertEqual(config["hiddenInputDim"], 64)
        self.assertEqual(
            config["projectionDim"],
            VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
        )
        self.assertEqual(
            config["fusionDim"], VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM
        )
        self.assertEqual(
            config["headHiddenDim"], VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM
        )
        self.assertEqual(config["activation"], "SiLU")
        self.assertEqual(config["centering"], "none")
        self.assertEqual(config["poseReduction"], "none")
        self.assertEqual(config["runtimeReduction"], "none")
        self.assertFalse(config["boundedCorrection"])
        schema = model.export_schema()
        self.assertEqual(
            schema["viewcellRegionConditionedVisibility"], config
        )
        self.assertEqual(
            schema["outputs"]["viewcellRegionConditionedVisibilityLogit"],
            ["B", 1],
        )
        self.assertEqual(model.config["runtimeFeatureDim"], 124)
        self.assertEqual(schema["fixedTable"]["shape"], ["N", 124])
        self.assertEqual(
            sum(int(row["dim"]) for row in schema["fixedTable"]["layout"]),
            124,
        )

    def _make_region_branch_vary_with_region_input(
        self, model: BoundedRelationSurvivalMomentModel
    ) -> None:
        """Make the branch deterministic while preserving candidate variation."""
        assert model.viewcell_region_conditioned_visibility_region_projection is not None
        assert model.viewcell_region_conditioned_visibility_head is not None
        with torch.no_grad():
            region_linear = (
                model.viewcell_region_conditioned_visibility_region_projection[0]
            )
            region_linear.weight.zero_()
            region_linear.bias.zero_()
            region_linear.weight[0, 0] = 1.0

            head_linear = model.viewcell_region_conditioned_visibility_head[0]
            head_linear.weight.zero_()
            head_linear.bias.zero_()
            head_linear.weight[0, 0] = 1.0
            output = model.viewcell_region_conditioned_visibility_head[-1]
            output.weight.zero_()
            output.bias.zero_()
            output.weight[0, 0] = 1.0

    def test_region_conditioned_visibility_pose_mean_has_zero_mean_per_pose_and_separates_raw_applied(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
            viewcell_region_conditioned_visibility_centering="pose_mean",
        )
        self._make_region_branch_vary_with_region_input(model)
        inputs = self._inputs()
        _, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        raw = aux["viewcell_region_conditioned_visibility_raw_logit"]
        applied = aux["viewcell_region_conditioned_visibility_logit"]
        self.assertEqual(raw.shape, (4, 1))
        self.assertEqual(applied.shape, (4, 1))
        self.assertFalse(torch.allclose(raw, applied))
        for start, end in ((0, 2), (2, 4)):
            self.assertAlmostEqual(float(applied[start:end].mean()), 0.0, places=6)
            torch.testing.assert_close(
                applied[start:end],
                raw[start:end] - raw[start:end].mean(dim=0, keepdim=True),
                rtol=0.0,
                atol=1e-6,
            )
        config = model.config["viewcellRegionConditionedVisibility"]
        self.assertEqual(config["centering"], "pose_mean")
        self.assertEqual(config["poseReduction"], "candidate_mean_per_pose")
        self.assertEqual(config["runtimeReduction"], "candidate_mean_per_pose")

    def test_region_conditioned_visibility_pose_offsets_support_multiple_pose_sizes(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
            viewcell_region_conditioned_visibility_centering="pose_mean",
        )
        self._make_region_branch_vary_with_region_input(model)
        inputs = self._inputs()
        _, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 1, 4]),
        )
        raw = aux["viewcell_region_conditioned_visibility_raw_logit"]
        applied = aux["viewcell_region_conditioned_visibility_logit"]
        for start, end in ((0, 1), (1, 4)):
            self.assertAlmostEqual(float(applied[start:end].mean()), 0.0, places=6)
            torch.testing.assert_close(
                applied[start:end],
                raw[start:end] - raw[start:end].mean(dim=0, keepdim=True),
                rtol=0.0,
                atol=1e-6,
            )

    def test_region_conditioned_visibility_pose_mean_without_offsets_centers_whole_batch(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            viewcell_region_conditioned_visibility_enabled=True,
            viewcell_region_conditioned_visibility_centering="pose_mean",
        )
        self._make_region_branch_vary_with_region_input(model)
        inputs = self._inputs()
        _, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        raw = aux["viewcell_region_conditioned_visibility_raw_logit"]
        applied = aux["viewcell_region_conditioned_visibility_logit"]
        self.assertAlmostEqual(float(applied.mean()), 0.0, places=6)
        torch.testing.assert_close(
            applied,
            raw - raw.mean(dim=0, keepdim=True),
            rtol=0.0,
            atol=1e-6,
        )

    def test_region_conditioned_visibility_rejects_unknown_centering(self) -> None:
        with self.assertRaises(ValueError):
            self._model(
                spectral_mode="moment_extrema",
                viewcell_region_conditioned_visibility_enabled=True,
                viewcell_region_conditioned_visibility_centering="batch_mean",
            )

    def test_moment_support_modes_preserve_runtime_table_shape(self) -> None:
        inputs = self._inputs()
        for mode, expected_boundary_width in (
            ("moment_support", 32),
            ("moment_extrema_support", 49),
        ):
            with self.subTest(mode=mode):
                model = self._model(spectral_mode=mode)
                logits, aux = model.compute_logits_with_aux(
                    inputs["camera"],
                    inputs["camera_view"],
                    inputs["candidate_camera"],
                    inputs["instance_ids"],
                    runtime_features=inputs["runtime"],
                    query_center_world=inputs["query_center"],
                    viewcell_radius_m=inputs["radius"],
                )
                self.assertEqual(logits.shape, (4, 1))
                self.assertEqual(aux["viewcell_support_features"].shape, (4, 32))
                source_width = 64 + 9 + 4 + 8
                self.assertEqual(
                    model.boundary_summary_head[0].in_features - source_width,
                    expected_boundary_width,
                )
                self.assertEqual(model.config["runtimeFeatureDim"], 124)

    def test_boundary_opportunity_is_one_way_and_keeps_fixed_asset_shape(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_opportunity_hidden_dim=12,
        )
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        torch.testing.assert_close(
            logits, aux["pre_opportunity_visibility_logits"], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(
            aux["boundary_opportunity_logit_uplift"],
            torch.zeros_like(logits),
        )
        self.assertEqual(model.config["runtimeFeatureDim"], 124)
        self.assertEqual(model.export_schema()["fixedTable"]["shape"], ["N", 124])

    def test_boundary_opportunity_uses_bounded_logit_uplift(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_opportunity_hidden_dim=12,
        )
        assert model.boundary_opportunity_head is not None
        model.set_boundary_opportunity_scale(1.0)
        with torch.no_grad():
            model.boundary_opportunity_head[-1].weight.zero_()
            model.boundary_opportunity_head[-1].bias.zero_()
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
        )
        expected_uplift = torch.full_like(logits, 3.0)
        torch.testing.assert_close(
            aux["boundary_opportunity_raw_logit_uplift"], expected_uplift
        )
        torch.testing.assert_close(
            aux["boundary_opportunity_logit_uplift"], expected_uplift
        )
        torch.testing.assert_close(
            logits,
            aux["pre_opportunity_visibility_logits"] + expected_uplift,
        )
        self.assertTrue(
            torch.all(logits >= aux["pre_opportunity_visibility_logits"])
        )

    def test_boundary_opportunity_is_applied_after_signed_tail_residual(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_opportunity_hidden_dim=12,
            boundary_tail_residual_hidden_dim=12,
        )
        assert model.boundary_opportunity_head is not None
        assert model.boundary_tail_residual_head is not None
        model.set_boundary_opportunity_scale(1.0)
        with torch.no_grad():
            model.boundary_opportunity_head[-1].weight.zero_()
            model.boundary_opportunity_head[-1].bias.zero_()
            model.boundary_tail_residual_head[-1].weight.fill_(0.25)
            model.boundary_tail_residual_head[-1].bias.fill_(0.1)
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        torch.testing.assert_close(
            aux["pre_opportunity_visibility_logits"],
            aux["pre_tail_residual_visibility_logits"]
            + aux["boundary_tail_residual_centered"],
        )
        torch.testing.assert_close(
            logits,
            aux["pre_opportunity_visibility_logits"]
            + aux["boundary_opportunity_logit_uplift"],
        )

    def test_boundary_tail_residual_is_zero_initialized_and_asset_free(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
        )
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        torch.testing.assert_close(
            logits,
            aux["pre_tail_residual_visibility_logits"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            aux["boundary_tail_residual_centered"],
            torch.zeros_like(logits),
        )
        self.assertEqual(model.config["runtimeFeatureDim"], 124)
        self.assertEqual(model.export_schema()["fixedTable"]["shape"], ["N", 124])

    def test_boundary_tail_residual_is_pose_centered_signed_and_bounded(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
        )
        assert model.boundary_tail_residual_head is not None
        with torch.no_grad():
            output = model.boundary_tail_residual_head[-1]
            output.weight.fill_(0.25)
            output.bias.fill_(0.1)
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        residual = aux["boundary_tail_residual_centered"]
        self.assertAlmostEqual(float(residual[:2].sum()), 0.0, places=6)
        self.assertAlmostEqual(float(residual[2:].sum()), 0.0, places=6)
        self.assertLessEqual(float(residual.abs().max()), 1.0 + 1e-6)
        self.assertTrue(bool((residual > 0.0).any()))
        self.assertTrue(bool((residual < 0.0).any()))
        torch.testing.assert_close(
            logits,
            aux["pre_tail_residual_visibility_logits"] + residual,
        )

    def test_boundary_tail_residual_rejects_invalid_pose_offsets(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
        )
        inputs = self._inputs()
        with self.assertRaisesRegex(ValueError, "pose_offsets"):
            model.compute_logits_with_aux(
                inputs["camera"],
                inputs["camera_view"],
                inputs["candidate_camera"],
                inputs["instance_ids"],
                runtime_features=inputs["runtime"],
                query_center_world=inputs["query_center"],
                viewcell_radius_m=inputs["radius"],
                pose_offsets=torch.tensor([0, 3]),
            )

    def test_boundary_tail_residual_can_apply_without_pose_centering(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
            boundary_tail_residual_centering="none",
        )
        assert model.boundary_tail_residual_head is not None
        with torch.no_grad():
            output = model.boundary_tail_residual_head[-1]
            output.weight.zero_()
            output.bias.fill_(2.0)
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 3]),
        )
        applied = aux["boundary_tail_residual_centered"]
        self.assertGreater(float(applied.mean()), 0.9)
        self.assertLessEqual(float(applied.abs().max()), 1.0 + 1e-6)
        torch.testing.assert_close(
            logits,
            aux["pre_tail_residual_visibility_logits"] + applied,
        )
        self.assertEqual(model.config["boundaryTailResidual"]["centering"], "none")
        self.assertEqual(
            model.config["boundaryTailResidual"]["runtimeReduction"], "none"
        )

    def test_boundary_tail_region_shortcut_is_zero_initialized_and_receives_gradient(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
            boundary_tail_residual_shortcut="region_linear",
        )
        shortcut = model.boundary_tail_residual_region_shortcut
        self.assertIsNotNone(shortcut)
        assert shortcut is not None
        inputs = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        torch.testing.assert_close(
            logits,
            aux["pre_tail_residual_visibility_logits"],
            rtol=0.0,
            atol=0.0,
        )
        weights = torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])
        (logits * weights).sum().backward()
        self.assertIsNotNone(shortcut.weight.grad)
        self.assertGreater(float(shortcut.weight.grad.abs().sum()), 0.0)
        self.assertEqual(
            model.config["boundaryTailResidual"]["shortcut"], "region_linear"
        )

    def test_boundary_tail_affine_region_fusion_preserves_raw_region_and_initial_gradient(self) -> None:
        model = self._model(
            spectral_mode="moment_extrema",
            boundary_tail_residual_hidden_dim=12,
            boundary_tail_residual_fusion="affine_region",
            boundary_tail_residual_output_init_std=0.002,
        )
        assert model.boundary_tail_residual_head is not None
        first = model.boundary_tail_residual_head[0]
        self.assertEqual(first.in_features, 84)
        inputs = self._inputs()
        logits, _aux = model.compute_logits_with_aux(
            inputs["camera"],
            inputs["camera_view"],
            inputs["candidate_camera"],
            inputs["instance_ids"],
            runtime_features=inputs["runtime"],
            query_center_world=inputs["query_center"],
            viewcell_radius_m=inputs["radius"],
            pose_offsets=torch.tensor([0, 2, 4]),
        )
        (logits * torch.tensor([[1.0], [-1.0], [0.5], [-0.5]])).sum().backward()
        assert model.boundary_tail_residual_region_projection is not None
        projection = model.boundary_tail_residual_region_projection[0]
        self.assertIsNotNone(projection.weight.grad)
        self.assertGreater(float(projection.weight.grad.abs().sum()), 0.0)
        self.assertEqual(
            model.config["boundaryTailResidual"]["fusionMode"], "affine_region"
        )

    def test_boundary_opportunity_requires_analytic_extrema(self) -> None:
        with self.assertRaisesRegex(ValueError, "analytic moment-extrema"):
            self._model(boundary_opportunity_hidden_dim=12)

    def test_zero_initialized_instance_calibration_preserves_shared_prior(self) -> None:
        model = self._model()
        geometry = torch.randn((4, 96), dtype=torch.float32)
        result = model.offline_encode_survival(
            geometry,
            None,
            torch.arange(4),
            torch.arange(4),
            return_diagnostics=True,
        )
        self.assertTrue(
            torch.equal(
                result["survival_coefficients"],
                result["survival_prior_coefficients"],
            )
        )
        self.assertTrue(
            torch.equal(
                result["instance_calibration_applied_residual"],
                torch.zeros_like(result["instance_calibration_applied_residual"]),
            )
        )

    def test_instance_calibration_changes_only_the_selected_instance(self) -> None:
        model = self._model()
        model.set_instance_calibration_reliability(torch.ones(4))
        model.set_instance_calibration_blend(1.0)
        geometry = torch.randn((4, 96), dtype=torch.float32)
        with torch.no_grad():
            model.instance_calibration_residual_raw[2, 1, 3] = 0.75
        result = model.offline_encode_survival(
            geometry,
            None,
            torch.arange(4),
            torch.arange(4),
            return_diagnostics=True,
        )
        difference = (
            result["survival_coefficients"]
            - result["survival_prior_coefficients"]
        ).abs().flatten(1).sum(dim=1)
        self.assertEqual(torch.nonzero(difference > 0.0).reshape(-1).tolist(), [2])
        self.assertTrue(
            torch.allclose(
                result["survival_coefficients"],
                result["survival_prior_coefficients"]
                + result["instance_calibration_applied_residual"],
            )
        )

    def test_disabled_instance_calibration_is_exactly_the_shared_prior(self) -> None:
        model = self._model(instance_calibration_mode="disabled")
        model.set_instance_calibration_blend(1.0)
        geometry = torch.randn((4, 96), dtype=torch.float32)
        result = model.offline_encode_survival(
            geometry,
            None,
            torch.arange(4),
            torch.arange(4),
            return_diagnostics=True,
        )
        self.assertIsNone(model.instance_calibration_residual_raw)
        self.assertEqual(float(model.instance_calibration_blend), 0.0)
        self.assertTrue(
            torch.equal(
                result["survival_coefficients"],
                result["survival_prior_coefficients"],
            )
        )
        self.assertEqual(float(model.instance_calibration_regularization()), 0.0)
        self.assertEqual(model.config["instanceCalibration"]["mode"], "disabled")

    def test_query_gradient_reaches_only_selected_instance_residual_rows(self) -> None:
        model = self._model()
        model.set_instance_calibration_blend(1.0)
        geometry = torch.randn((4, 96), dtype=torch.float32)
        coefficients = model.offline_encode_survival(
            geometry,
            None,
            torch.arange(4),
            torch.arange(4),
        )
        selected = torch.tensor([1, 3], dtype=torch.long)
        logits, aux = model.forward_batch(
            selected,
            torch.zeros((2, 9), dtype=torch.float32),
            torch.zeros((2, 9, 2), dtype=torch.float32),
            torch.full((2, 1), 0.5, dtype=torch.float32),
            {
                "geometry": geometry,
                "survival_coefficients": coefficients,
            },
        )
        (
            logits.square().sum()
            + aux["utility_logits"].square().sum()
            + aux["download_logits"].square().sum()
        ).backward()

        gradient = model.instance_calibration_residual_raw.grad
        self.assertIsNotNone(gradient)
        row_norm = gradient.flatten(1).abs().sum(dim=1)
        self.assertGreater(float(row_norm[1]), 0.0)
        self.assertGreater(float(row_norm[3]), 0.0)
        self.assertEqual(float(row_norm[0]), 0.0)
        self.assertEqual(float(row_norm[2]), 0.0)

    def test_sparse_instances_receive_stronger_residual_regularization(self) -> None:
        model = self._model()
        model.set_instance_calibration_reliability(torch.tensor([0.0, 1.0, 1.0, 1.0]))
        with torch.no_grad():
            model.instance_calibration_residual_raw.zero_()
            model.instance_calibration_residual_raw[0, 0, 0] = 0.5
        sparse_loss = model.instance_calibration_regularization()
        model.set_instance_calibration_reliability(torch.tensor([1.0, 0.0, 1.0, 1.0]))
        dense_loss = model.instance_calibration_regularization()
        self.assertGreater(float(sparse_loss), float(dense_loss))

    def test_missing_query_center_is_rejected_without_candidate_camera_fallback(self) -> None:
        model = self._model()
        value = self._inputs()
        with self.assertRaisesRegex(ValueError, "query_center_world"):
            model.compute_logits_with_aux(
                value["camera"],
                value["camera_view"],
                value["candidate_camera"],
                value["instance_ids"],
                runtime_features=value["runtime"],
                viewcell_radius_m=value["radius"],
            )

    def test_point_control_zeroes_the_viewcell_disk_axes(self) -> None:
        model = self._model(spectral_mode="point")
        value = self._inputs()
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertTrue(torch.equal(aux["effective_disk_axes"], torch.zeros_like(aux["effective_disk_axes"])))

    def test_runtime_rejects_offline_relation_tables(self) -> None:
        model = self._model()
        value = self._inputs()
        with self.assertRaisesRegex(ValueError, "offline graph"):
            model.forward_batch(
                value["instance_ids"],
                torch.zeros((4, 9)),
                torch.zeros((4, 9, 2)),
                torch.full((4, 1), 0.5),
                {"runtime_features": value["runtime"], "relation_csr": torch.ones(1)},
            )

    def test_visibility_utility_and_download_heads_have_finite_gradients(self) -> None:
        model = self._model()
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        loss = logits.square().mean() + aux["utility_logits"].square().mean() + aux["download_logits"].square().mean()
        loss.backward()
        for name in ("visibility_head.weight", "utility_head.2.weight", "download_head.2.weight"):
            parameter = dict(model.named_parameters())[name]
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_view_residual_default_is_disabled_without_new_state_or_schema(self) -> None:
        model = self._model()
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertIsNone(model.view_residual_head)
        self.assertIsNone(model.view_residual_geometry_projection)
        self.assertIsNone(model.view_residual_dynamic_projection)
        self.assertFalse(any("view_residual" in key for key in model.state_dict()))
        self.assertNotIn("viewResidual", model.config)
        self.assertNotIn("viewResidual", model.export_schema())
        self.assertTrue(
            torch.equal(
                aux["view_conditioned_residual"],
                torch.zeros_like(aux["view_conditioned_residual"]),
            )
        )
        self.assertTrue(torch.equal(logits, aux["base_visibility_logits"]))

    def test_aux_exposes_computed_center_view_and_disk_axes(self) -> None:
        model = self._model()
        value = self._inputs()
        center_view = torch.arange(36, dtype=torch.float32).reshape(4, 9) / 10.0
        disk_axes = torch.arange(72, dtype=torch.float32).reshape(4, 9, 2) / 20.0
        normalized_depth = torch.full((4, 1), 0.4, dtype=torch.float32)
        _logits, aux = model.forward_batch(
            value["instance_ids"],
            center_view,
            disk_axes,
            normalized_depth,
            {"runtime_features": value["runtime"]},
        )
        self.assertEqual(aux["center_view"].shape, (4, 9))
        self.assertEqual(aux["disk_axes"].shape, (4, 9, 2))
        self.assertTrue(torch.equal(aux["center_view"], center_view))
        self.assertTrue(torch.equal(aux["disk_axes"], disk_axes))
        self.assertTrue(bool(torch.isfinite(aux["center_view"]).all()))
        self.assertTrue(bool(torch.isfinite(aux["disk_axes"]).all()))

    def test_loading_an_old_checkpoint_keeps_zero_initialized_residual_exactly_equivalent(self) -> None:
        old_model = self._model()
        new_model = self._model(view_residual_max_abs=0.75)
        load_result = new_model.load_state_dict(old_model.state_dict(), strict=False)
        self.assertTrue(load_result.missing_keys)
        self.assertFalse(load_result.unexpected_keys)
        value = self._inputs()
        old_logits, _old_aux = old_model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        new_logits, new_aux = new_model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertTrue(torch.equal(new_logits, old_logits))
        self.assertTrue(
            torch.equal(
                new_aux["view_conditioned_residual"],
                torch.zeros_like(new_aux["view_conditioned_residual"]),
            )
        )

    def test_view_residual_head_can_emit_both_signs(self) -> None:
        model = self._model(view_residual_max_abs=0.75)
        value = self._inputs()
        self.assertIsNotNone(model.view_residual_head)
        assert model.view_residual_head is not None
        output_layer = model.view_residual_head[-1]
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.fill_(12.0)
        _positive_logits, positive_aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        with torch.no_grad():
            output_layer.bias.fill_(-12.0)
        _negative_logits, negative_aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertTrue(bool((positive_aux["view_conditioned_residual"] > 0.0).all()))
        self.assertTrue(bool((negative_aux["view_conditioned_residual"] < 0.0).all()))

    def test_view_residual_is_strictly_bounded(self) -> None:
        maximum = 0.35
        model = self._model(view_residual_max_abs=maximum)
        value = self._inputs()
        assert model.view_residual_head is not None
        with torch.no_grad():
            model.view_residual_head[-1].weight.fill_(20.0)
            model.view_residual_head[-1].bias.fill_(20.0)
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        residual = aux["view_conditioned_residual"]
        self.assertTrue(bool(torch.isfinite(residual).all()))
        self.assertLessEqual(float(residual.abs().max()), maximum)

    def test_view_residual_has_no_one_sided_runtime_gate(self) -> None:
        model = self._model(view_residual_max_abs=1.0)
        value = self._inputs()
        assert model.view_residual_head is not None
        with torch.no_grad():
            model.view_residual_head[-1].weight.zero_()
            model.view_residual_head[-1].bias.fill_(20.0)
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )

        residual = aux["view_conditioned_residual"].abs().reshape(-1)
        self.assertTrue(torch.allclose(residual, torch.ones_like(residual), atol=1e-6, rtol=1e-6))
        self.assertNotIn("view_residual_working_set_gate", aux)

    def test_view_residual_gradients_reach_factorized_inputs_and_zero_output_layer(self) -> None:
        model = self._model(view_residual_max_abs=0.75)
        value = self._inputs()
        assert model.view_residual_head is not None
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        aux["view_conditioned_residual"].sum().backward()
        output_layer = model.view_residual_head[-1]
        self.assertGreater(float(output_layer.weight.grad.abs().sum()), 0.0)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            output_layer.weight.fill_(0.1)
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        aux["view_conditioned_residual"][0].sum().backward()
        for module in (
            model.view_residual_geometry_projection,
            model.view_residual_dynamic_projection,
            model.view_residual_head[0],
            model.view_residual_head[-1],
        ):
            assert module is not None
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                self.assertGreater(float(parameter.grad.abs().sum()), 0.0)

    def test_view_residual_head_has_no_geometry_or_instance_id_bypass(self) -> None:
        model = self._model(view_residual_max_abs=0.75, view_residual_hidden_dim=16)
        self.assertIsNotNone(model.view_residual_geometry_projection)
        self.assertIsNotNone(model.view_residual_dynamic_projection)
        self.assertIsNotNone(model.view_residual_head)
        assert model.view_residual_geometry_projection is not None
        assert model.view_residual_dynamic_projection is not None
        assert model.view_residual_head is not None
        self.assertEqual(model.view_residual_geometry_projection.in_features, 96)
        self.assertEqual(
            model.view_residual_dynamic_projection.in_features,
            VIEW_RESIDUAL_DYNAMIC_INPUT_DIM,
        )
        self.assertEqual(VIEW_RESIDUAL_DYNAMIC_INPUT_DIM, 35)
        self.assertEqual(model.view_residual_geometry_projection.out_features, model.hidden_dim)
        self.assertEqual(model.view_residual_dynamic_projection.out_features, model.hidden_dim)
        self.assertEqual(model.view_residual_head[0].in_features, 2 * model.hidden_dim)
        self.assertEqual(model.view_residual_head[-1].out_features, 1)
        view_config = model.config["viewResidual"]
        self.assertEqual(
            view_config["headInputs"],
            ["geometry_projection * dynamic_projection", "dynamic_projection"],
        )
        self.assertFalse(view_config["usesInstanceIds"])
        self.assertTrue(model.export_schema()["viewResidual"]["noAdditionalPerInstanceAssets"])
        self.assertIn("base_visibility_logit", view_config["runtimeInputs"])

    def test_certificate_and_view_residual_use_explicit_fusion_order(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=2.0,
            cull_certificate_initial_suppression=0.05,
            view_residual_max_abs=0.75,
        )
        model.set_instance_world_aabbs(self._model().instance_world_aabbs)
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        assert model.view_residual_head is not None
        with torch.no_grad():
            model.view_residual_head[-1].weight.zero_()
            model.view_residual_head[-1].bias.fill_(0.5)
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        expected = (
            aux["base_visibility_logits"]
            - aux["cull_certificate_suppression"]
            + aux["view_conditioned_residual"]
        )
        self.assertTrue(torch.equal(logits, expected))

    def test_cull_certificate_starts_as_a_constant_bounded_subtraction(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=2.0,
            cull_certificate_initial_suppression=0.05,
        )
        model.set_instance_world_aabbs(self._model().instance_world_aabbs)
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        expected = torch.full_like(logits, 0.05)
        self.assertTrue(
            torch.allclose(aux["cull_certificate_suppression"], expected, atol=1e-6)
        )
        self.assertTrue(torch.allclose(logits, aux["base_logits"] - expected))
        self.assertGreaterEqual(float(aux["cull_certificate_suppression"].min()), 0.0)
        self.assertLessEqual(float(aux["cull_certificate_suppression"].max()), 2.0)

    def test_certificate_only_gradient_does_not_reach_shared_trunk(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=2.0,
            cull_certificate_initial_suppression=0.05,
        )
        model.set_instance_world_aabbs(self._model().instance_world_aabbs)
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        value = self._inputs()
        _logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        aux["cull_certificate_suppression"].sum().backward()
        self.assertIsNotNone(model.cull_certificate_head)
        assert model.cull_certificate_head is not None
        self.assertIsNotNone(model.cull_certificate_head.weight.grad)
        self.assertGreater(float(model.cull_certificate_head.weight.grad.abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.shared_trunk.parameters())
        )
        self.assertIsNone(model.visibility_head.weight.grad)

    def test_nonlinear_certificate_preserves_the_registered_initial_suppression(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=1.0,
            cull_certificate_initial_suppression=0.1,
            cull_certificate_hidden_dim=16,
        )
        model.set_instance_world_aabbs(self._model().instance_world_aabbs)
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertTrue(
            torch.allclose(
                aux["cull_certificate_suppression"],
                torch.full_like(logits, 0.1),
                atol=1e-6,
            )
        )
        self.assertEqual(model.config["cullCertificate"]["architecture"], [64, 16, 1])

    def test_certificate_can_query_existing_explicit_survival_evidence(self) -> None:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=1.0,
            cull_certificate_initial_suppression=0.1,
            cull_certificate_hidden_dim=16,
            cull_certificate_input_mode="evidence",
        )
        model.set_instance_world_aabbs(self._model().instance_world_aabbs)
        model.set_instance_to_glb(torch.tensor([0, 0, 1, 1]))
        value = self._inputs()
        logits, aux = model.compute_logits_with_aux(
            value["camera"],
            value["camera_view"],
            value["candidate_camera"],
            value["instance_ids"],
            runtime_features=value["runtime"],
            query_center_world=value["query_center"],
            viewcell_radius_m=value["radius"],
        )
        self.assertTrue(
            torch.allclose(
                aux["cull_certificate_suppression"],
                torch.full_like(logits, 0.1),
                atol=1e-6,
            )
        )
        certificate = model.config["cullCertificate"]
        self.assertEqual(certificate["inputMode"], "evidence")
        self.assertEqual(certificate["inputDim"], 10)
        self.assertEqual(certificate["architecture"], [10, 16, 1])

        aux["cull_certificate_suppression"].sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.shared_trunk.parameters())
        )
        self.assertIsNone(model.visibility_head.weight.grad)

    def test_certificate_can_bypass_a_tail_information_bottleneck(self) -> None:
        runtime_model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=1.0,
            cull_certificate_initial_suppression=0.1,
            cull_certificate_hidden_dim=32,
            cull_certificate_input_mode="runtime",
        )
        combined_model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            depth_q01=0.0,
            depth_q99=4.0,
            cull_certificate_max_suppression=1.0,
            cull_certificate_initial_suppression=0.1,
            cull_certificate_hidden_dim=32,
            cull_certificate_input_mode="hidden_runtime",
        )
        self.assertEqual(
            runtime_model.config["cullCertificate"]["architecture"],
            [130, 32, 1],
        )
        self.assertEqual(
            combined_model.config["cullCertificate"]["architecture"],
            [194, 32, 1],
        )


if __name__ == "__main__":
    unittest.main()
