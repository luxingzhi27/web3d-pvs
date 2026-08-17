from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "model"
if str(MODEL) not in sys.path:
    sys.path.insert(0, str(MODEL))

from export_hierarchical_relation_survival_integrated import (  # noqa: E402
    GEO_DIM,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
    _build_model_meta,
    _resolve_threshold,
    _to_fp32,
)


class HierarchicalRelationSurvivalExportTest(unittest.TestCase):
    def test_aggregate_only_workpoint_is_accepted(self) -> None:
        threshold, info = _resolve_threshold(
            {
                "calibration": {
                    "selected": {
                        "threshold": 0.35,
                        "aggregateWeightedRecall": 0.994,
                        "aggregateWeightedRecallLowerConfidenceBound": 0.991,
                    },
                },
            },
            requested=None,
            allow_unsafe=False,
        )

        self.assertEqual(threshold, 0.35)
        self.assertTrue(info["safe"])
        self.assertEqual(info["selected"]["aggregateWeightedRecall"], 0.994)
        self.assertEqual(
            info["selected"]["aggregateWeightedRecallLowerConfidenceBound"],
            0.991,
        )

    def test_pose_only_workpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "aggregateWeightedRecall"):
            _resolve_threshold(
                {
                    "calibration": {
                        "selected": {
                            "threshold": 0.35,
                            "pose_weighted_recall": 1.0,
                            "weighted_recall_lower_confidence_bound": 1.0,
                        },
                    },
                },
                requested=None,
                allow_unsafe=False,
            )

    def test_aabb_meta_and_payload_use_fp32_size(self) -> None:
        num_instances = 2
        aabb = _to_fp32(
            np.arange(num_instances * 6, dtype=np.float64).reshape(num_instances, 6),
            "instance AABB",
            (num_instances, 6),
        )
        raw_aabb = aabb.tobytes(order="C")
        self.assertEqual(aabb.dtype, np.dtype("<f4"))
        self.assertEqual(len(raw_aabb), num_instances * 6 * np.dtype("<f4").itemsize)

        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint_path = Path(checkpoint.name)
            meta = _build_model_meta(
                checkpoint_path=checkpoint_path,
                checkpoint={},
                runtime_config={
                    "numInstances": num_instances,
                    "numGlbs": 1,
                    "spectralMode": "fourier117",
                },
                geometry=np.zeros((num_instances, GEO_DIM), dtype=np.float16),
                coefficient=np.zeros(
                    (num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM),
                    dtype=np.float16,
                ),
                aabb=aabb,
                instance_to_glb=np.zeros((num_instances,), dtype=np.uint32),
                query_weights=b"",
                query_layout=[],
                scene_bounds={"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]},
                runtime_meta_info={"path": "runtime.json", "sha256": "runtime-sha"},
                geometry_info={"source": "test"},
                coefficient_info={"source": "test"},
                threshold=0.35,
                threshold_info={},
                output_files={
                    "instance_geo_features_fp16.bin": b"",
                    "instance_survival_coefficients_fp16.bin": b"",
                    "instance_aabb_fp32.bin": raw_aabb,
                    "instance_to_glb_uint32.bin": b"",
                    "query_weights_fp16.bin": b"",
                },
                training_provenance={},
            )

        descriptor = meta["files"]["instanceAabb"]
        self.assertEqual(descriptor["file"], "instance_aabb_fp32.bin")
        self.assertEqual(descriptor["dtype"], "float32")
        self.assertEqual(descriptor["byteLength"], num_instances * 6 * 4)


if __name__ == "__main__":
    unittest.main()
