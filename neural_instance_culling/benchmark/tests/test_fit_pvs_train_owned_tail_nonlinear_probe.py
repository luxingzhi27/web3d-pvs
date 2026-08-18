from __future__ import annotations

import unittest

import numpy as np

from neural_instance_culling.benchmark.fit_pvs_train_owned_tail_nonlinear_probe import (
    fit_probe,
)
from neural_instance_culling.benchmark.scan_pvs_train_owned_tail_residual import (
    ProbeSpec,
    probe_linear_scores,
)


def _synthetic(seed: int, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(count, 4))
    labels = ((values[:, 0] * values[:, 1] > 0.0) | (values[:, 2] > 1.0))
    visible_weights = np.where(labels, 0.0, 1.0 + np.abs(values[:, 3]))
    return values, labels, visible_weights


class TrainOwnedTailNonlinearProbeTest(unittest.TestCase):
    def test_linear_fit_matches_runtime_probe_schema(self) -> None:
        train = _synthetic(21, 240)
        calibration = _synthetic(22, 100)
        validation = _synthetic(23, 100)
        record = fit_probe(
            train[0], train[1], train[2],
            calibration[0], calibration[1],
            validation[0], validation[1],
            probe_type="linear",
            sample_weight_power=0.5,
            max_samples=-1,
            ridge=1e-3,
            seed=29,
            device="cpu",
        )
        self.assertEqual(record["probeType"], "standardized_ridge_linear")
        self.assertEqual(record["parameterCount"], 5)
        self.assertEqual(len(record["coefficients"]), 5)
        self.assertEqual(record["riskCertificates"]["sourceSplit"], "train")

    def test_hinge_fit_is_train_owned_and_loader_compatible(self) -> None:
        train = _synthetic(1, 240)
        calibration = _synthetic(2, 100)
        validation = _synthetic(3, 100)
        record = fit_probe(
            train[0], train[1], train[2],
            calibration[0], calibration[1],
            validation[0], validation[1],
            probe_type="hinge",
            sample_weight_power=0.5,
            max_samples=200,
            ridge=1e-3,
            hinge_knots=(-1.0, 0.0, 1.0),
            seed=7,
            device="cpu",
        )
        self.assertEqual(record["probeType"], "standardized_ridge_hinge")
        self.assertEqual(record["fitSplit"], "train")
        self.assertFalse(record["testRead"])
        self.assertEqual(record["parameterCount"], 1 + 4 * 4)
        self.assertEqual(record["riskCertificates"]["sourceSplit"], "train")
        self.assertGreater(float(record["validation"]["rocAuc"]), 0.55)

    def test_mlp_fit_exports_exact_numpy_runtime_parameters(self) -> None:
        train = _synthetic(11, 320)
        calibration = _synthetic(12, 120)
        validation = _synthetic(13, 120)
        record = fit_probe(
            train[0], train[1], train[2],
            calibration[0], calibration[1],
            validation[0], validation[1],
            probe_type="mlp",
            sample_weight_power=0.0,
            max_samples=300,
            ridge=1e-4,
            hidden_dim=8,
            epochs=20,
            batch_size=64,
            learning_rate=5e-3,
            seed=19,
            device="cpu",
        )
        parameters = record["parameters"]
        probe = ProbeSpec(
            family="combined",
            feature_count=4,
            coefficients=np.zeros(0),
            mean=np.asarray(record["standardization"]["mean"]),
            scale=np.asarray(record["standardization"]["scale"]),
            fit_split="train",
            ridge=float(record["ridge"]),
            probe_type=record["probeType"],
            hidden_weight=np.asarray(parameters["hiddenWeight"]),
            hidden_bias=np.asarray(parameters["hiddenBias"]),
            output_weight=np.asarray(parameters["outputWeight"]),
            output_bias=float(parameters["outputBias"]),
            activation=record["activation"],
        )
        scores = probe_linear_scores(validation[0], probe)
        self.assertEqual(scores.shape, (120,))
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(float(record["validation"]["rocAuc"]), 0.65)
        self.assertEqual(record["riskCertificates"]["sourceSplit"], "train")


if __name__ == "__main__":
    unittest.main()
