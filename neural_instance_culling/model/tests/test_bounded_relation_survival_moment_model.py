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
    RUNTIME_FEATURE_DIM,
)


class BoundedRelationSurvivalMomentModelContractTest(unittest.TestCase):
    def _model(
        self,
        *,
        spectral_mode: str = "moment_envelope",
        instance_calibration_mode: str = "residual",
    ) -> BoundedRelationSurvivalMomentModel:
        model = BoundedRelationSurvivalMomentModel(
            4,
            2,
            relation_source="geometry_only",
            spectral_mode=spectral_mode,
            instance_calibration_mode=instance_calibration_mode,
            depth_q01=0.0,
            depth_q99=4.0,
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
        self.assertEqual(config["instanceCalibration"]["mode"], "residual")
        self.assertEqual(config["instanceCalibration"]["runtimeExport"], "fused coefficients only")

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


if __name__ == "__main__":
    unittest.main()
