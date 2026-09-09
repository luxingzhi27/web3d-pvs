from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

MODEL_DIR = Path(__file__).resolve().parents[1]
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from export_pvs import (  # noqa: E402
    BOUNDARY_SUMMARY_DIM,
    CHECKPOINT_SCHEMA,
    CHI_TABLE_SIZE,
    EXPORT_SCHEMA,
    GEO_DIM,
    LOW_RANK_SUMMARY_DIM,
    MAX_NEURAL_ASSET_BYTES,
    MAX_NEURAL_ASSET_BYTES_PER_INSTANCE,
    MAX_SHARED_NEURAL_ASSET_BYTES,
    MODEL_SCHEMA,
    RELATION_CONDITION_DIM,
    RUNTIME_FEATURE_DIM,
    RUNTIME_HEAD_INPUT_DIM,
    RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION,
    SPECTRAL_FREQUENCY_COUNT,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
    _check_neural_asset_budget,
    _neural_asset_budget_limit,
    _runtime_weight_specs,
    export,
    parse_args,
)


class BoundedRelationSurvivalMomentExportTest(unittest.TestCase):
    hidden_dim = 4
    num_instances = 3
    num_glbs = 2

    def _checkpoint(self, root: Path) -> dict:
        state = {
            name: torch.full(shape, 0.01, dtype=torch.float32)
            for name, shape in _runtime_weight_specs(self.hidden_dim)
        }
        state["moment_query.frequency_cycles"] = torch.zeros((SPECTRAL_FREQUENCY_COUNT, 9))
        state["moment_query.frequency_cycles"][:, 0] = torch.linspace(0.05, 0.8, SPECTRAL_FREQUENCY_COUNT)
        state["moment_query.chi_table"] = torch.ones(CHI_TABLE_SIZE, dtype=torch.float32)
        state["instance_calibration_residual_raw"] = torch.zeros(
            self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM
        )
        # This represents real offline checkpoint content.  The exporter must
        # not copy it or serialize the entire checkpoint into the bundle.
        state["offline_survival_encoder.relation_csr_payload"] = torch.ones(2)

        config = {
            "runtimeSchema": MODEL_SCHEMA,
            "numInstances": self.num_instances,
            "numGlbs": self.num_glbs,
            "relationSource": "bounded_hierarchical",
            "spectralMode": "moment_envelope",
            "geometryDim": GEO_DIM,
            "survivalCoefficientShape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
            "runtimeFeatureDim": RUNTIME_FEATURE_DIM,
            "runtimeHeadInputDim": RUNTIME_HEAD_INPUT_DIM,
            "boundarySummaryDim": BOUNDARY_SUMMARY_DIM,
            "lowRankSummaryDim": LOW_RANK_SUMMARY_DIM,
            "hiddenDim": self.hidden_dim,
            "instanceCalibration": {
                "mode": "residual",
                "shape": [SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM],
                "initialization": "zero",
                "maximumAbsoluteResidual": 4.0,
                "sparseInstancePenalty": 3.0,
                "fusion": "survival_prior + blend * bounded_instance_residual",
                "runtimeExport": "fused coefficients only",
            },
            "depthNormalization": {
                "definition": "clip((log1p(distance/(radius+epsilon))-q01)/(q99-q01),0,1)",
                "q01": 0.11,
                "q99": 1.91,
                "epsilon": 1e-4,
                "sourceSplit": "train",
            },
            "frequency": {
                "count": SPECTRAL_FREQUENCY_COUNT,
                "units": "cycles",
                "maxNormCycles": 8.0,
            },
        }
        geometry = np.arange(self.num_instances * GEO_DIM, dtype=np.float16).reshape(
            self.num_instances, GEO_DIM
        )
        geometry_path = root / "instance_geo_features_fp16.bin"
        geometry.tofile(geometry_path)
        dataset_path = root / "dataset"
        dataset_path.mkdir(exist_ok=True)
        (dataset_path / "dataset_meta.json").write_text(
            json.dumps(
                {
                    "modelInputFovYDeg": 66.0,
                    "frontendRenderFovYDeg": 60.0,
                    "stats": {
                        "pvsBackOffsetRange": [3.464101552963257, 3.464101552963257]
                    },
                }
            ),
            encoding="utf-8",
        )
        relation_path = root / "relation"
        relation_path.mkdir(exist_ok=True)
        (relation_path / "relation_csr_meta.json").write_text(
            json.dumps(
                {
                    "schema": "pvs-viewcell-train-observed-relation-csr-v3",
                    "numInstances": self.num_instances,
                    "stats": {
                        "edgeCount": 4,
                        "rowCount": 5,
                        "survivalObservationCount": 6,
                    },
                }
            ),
            encoding="utf-8",
        )
        coefficients = torch.arange(
            self.num_instances * SURVIVAL_RANK * SURVIVAL_PARAMETER_DIM,
            dtype=torch.float32,
        ).reshape(self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM)
        residual = torch.full_like(coefficients, 0.25)
        prior = coefficients - residual
        return {
            "schema": CHECKPOINT_SCHEMA,
            "runtimeSchema": MODEL_SCHEMA,
            "modelSchema": MODEL_SCHEMA,
            "model": state,
            "config": config,
            "geometry": {
                "path": str(geometry_path),
                "sha256": "legacy-geometry-fingerprint",
                "shape": [self.num_instances, GEO_DIM],
                "dtype": "float16",
            },
            "instanceSurvivalCoefficients": coefficients,
            "instanceSurvivalPriorCoefficients": prior,
            "instanceSurvivalCalibrationResidual": residual,
            "instanceCalibration": {
                "mode": "residual",
                "blend": 1.0,
                "fusion": "prior_plus_applied_residual",
                "reliability": {"sourceSplit": "train"},
                "runtimeExport": "fused_coefficients_only",
            },
            "viewcell": {"shape": "horizontal_disk", "radiusM": 2.0},
            "datasetDigest": "1" * 64,
            "candidateDigest": "2" * 64,
            "candidateDigests": {
                "train": "2" * 64,
                "calibration": "3" * 64,
                "validation": "4" * 64,
            },
            "relationArtifactDigest": "5" * 64,
            "protocol": {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4",
                "variant": "full_integrated_visibility_mainline",
                "lossVariant": "pose_balanced_rvl_contrastive",
                "testRead": False,
                "candidateUnion": False,
                "instanceCalibration": {"mode": "residual"},
                "trainCandidateDigest": "2" * 64,
                "calibrationCandidateDigest": "3" * 64,
                "validationCandidateDigest": "4" * 64,
                "splitPoseCounts": {"train": 2772, "calibration": 168, "validation": 213},
                "dataset": {"path": str(dataset_path)},
                "runtimeMeta": {"path": str(root / "runtimeVisibilityMeta.json")},
            },
            "relation": {
                "schema": "pvs-viewcell-train-observed-relation-csr-v3",
                "path": str(relation_path),
                "edgeCount": 4,
                "rowCount": 5,
                "observationCount": 6,
                "candidateDigest": "2" * 64,
                "artifactDigest": "5" * 64,
            },
            "relationCsr": {"path": "/never/pack/relation.csr"},
            "groupIds": {"path": "/never/pack/groups.bin"},
            "observations": {"path": "/never/pack/observations.bin"},
            "calibration": {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-v4",
                "testRead": False,
                "selected": {
                    "threshold": 0.02,
                    "aggregateWeightedRecall": 0.995,
                    "aggregateWeightedRecallLowerConfidenceBound": 0.992,
                    "poseMacroWeightedRecall": 0.991,
                    "safe": True,
                },
                "selectedFromTest": False,
                "testEvaluationCount": 0,
            },
            "testRead": False,
        }

    def _runtime_meta(self, root: Path) -> Path:
        path = root / "runtimeVisibilityMeta.json"
        path.write_text(
            json.dumps(
                {
                    "schemaVersion": 2,
                    "sceneBounds": {"min": [-1, -2, -3], "max": [4, 5, 6]},
                    "componentRecords": [
                        {
                            "componentGlobalId": 0,
                            "globalGlbId": 0,
                            "bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
                        },
                        {
                            "componentGlobalId": 1,
                            "globalGlbId": 1,
                            "bounds": {"min": [1, 2, 3], "max": [2, 3, 4]},
                        },
                        {
                            "componentGlobalId": 2,
                            "globalGlbId": 0,
                            "bounds": {"min": [-1, -1, -1], "max": [0, 0, 0]},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def _control_checkpoint(self, root: Path, mode: str) -> dict:
        checkpoint = copy.deepcopy(self._checkpoint(root))
        if mode not in {"generic28", "none"}:
            raise ValueError(mode)
        runtime_dim = GEO_DIM + (28 if mode == "generic28" else 0)
        head_dim = (
            RUNTIME_HEAD_INPUT_DIM
            if mode == "generic28"
            else RUNTIME_HEAD_INPUT_DIM_WITHOUT_OCCLUSION
        )
        state = {
            name: torch.full(shape, 0.01, dtype=torch.float32)
            for name, shape in _runtime_weight_specs(self.hidden_dim, mode)
        }
        state["moment_query.frequency_cycles"] = torch.zeros(
            (SPECTRAL_FREQUENCY_COUNT, 9)
        )
        state["moment_query.frequency_cycles"][:, 0] = torch.linspace(
            0.05, 0.8, SPECTRAL_FREQUENCY_COUNT
        )
        state["moment_query.chi_table"] = torch.ones(
            CHI_TABLE_SIZE, dtype=torch.float32
        )
        features = (
            torch.arange(self.num_instances * 28, dtype=torch.float32).reshape(
                self.num_instances, 4, 7
            )
            if mode == "generic28"
            else torch.empty((self.num_instances, 0), dtype=torch.float32)
        )
        if mode == "generic28":
            state["generic_occlusion_features"] = features.clone()
        checkpoint["model"] = state
        checkpoint["occlusionRepresentation"] = mode
        checkpoint["instanceOcclusionFeatures"] = features
        checkpoint["config"].update(
            {
                "relationSource": "none",
                "occlusionRepresentation": {
                    "mode": mode,
                    "featureDim": 28 if mode == "generic28" else 0,
                },
                "survivalCoefficientShape": [],
                "runtimeFeatureDim": runtime_dim,
                "runtimeHeadInputDim": head_dim,
                "instanceCalibration": {
                    "mode": "disabled",
                    "shape": [],
                    "initialization": "zero",
                    "maximumAbsoluteResidual": 4.0,
                    "sparseInstancePenalty": 3.0,
                    "fusion": "not applicable",
                    "runtimeExport": "not applicable",
                },
            }
        )
        checkpoint["protocol"]["variant"] = (
            "core_generic28" if mode == "generic28" else "core_no_survival"
        )
        checkpoint["protocol"]["lossVariant"] = "pose_balanced_rvl_contrastive"
        checkpoint["protocol"]["instanceCalibration"] = {"mode": "disabled"}
        checkpoint["relation"] = {
            "enabled": False,
            "source": "depth_normalization_metadata_only",
            "relationGraphRead": False,
            "survivalObservationsRead": False,
            "testRead": False,
        }
        checkpoint["instanceCalibration"] = {
            "mode": "disabled",
            "blend": 0.0,
            "fusion": "not_applicable",
            "reliability": {"enabled": False, "sourceSplit": "train"},
            "runtimeExport": "not_applicable",
        }
        for key in (
            "instanceSurvivalCoefficients",
            "instanceSurvivalPriorCoefficients",
            "instanceSurvivalCalibrationResidual",
        ):
            checkpoint.pop(key, None)
        return checkpoint

    def _export(self, root: Path, checkpoint: dict, name: str = "bundle") -> Path:
        checkpoint_path = root / f"{name}.pt"
        torch.save(checkpoint, checkpoint_path)
        output_dir = root / f"{name}-out"
        result = export(
            parse_args(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--runtime-meta",
                    str(self._runtime_meta(root)),
                    "--output-dir",
                    str(output_dir),
                ]
            )
        )
        self.assertEqual(result["status"], "exported")
        self.assertEqual(result["schema"], EXPORT_SCHEMA)
        return output_dir

    def test_exports_124d_table_and_strict_runtime_meta(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._export(root, self._checkpoint(root))
            table = np.fromfile(output / "instance_runtime_features_fp16.bin", dtype="<f2")
            self.assertEqual(table.size, self.num_instances * RUNTIME_FEATURE_DIM)
            self.assertEqual(table.reshape(self.num_instances, RUNTIME_FEATURE_DIM).shape, (3, 124))

            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["schema"], EXPORT_SCHEMA)
            self.assertIs(meta["testRead"], False)
            self.assertEqual(meta["modelSchema"], MODEL_SCHEMA)
            self.assertEqual(meta["experimentName"], "pvs_mainline_v4")
            self.assertEqual(meta["fixedTable"]["shape"], [3, 124])
            self.assertEqual(meta["fixedTable"]["dtype"], "float16")
            self.assertEqual(
                meta["runtimeFeatureSource"]["source"],
                "checkpoint.geometryFeatures_plus_survivalCoefficients",
            )
            self.assertNotIn("sha256", meta["runtimeFeatureSource"]["geometry"])
            self.assertNotIn("sha256", meta["fixedTable"])
            self.assertEqual(
                meta["candidateCameraSemantics"],
                "66-degree back-camera candidate identity only",
            )
            self.assertEqual(
                meta["queryCenterSemantics"],
                "center of the same-direction view-cell visibility union",
            )
            self.assertEqual(meta["viewcell"]["shape"], "horizontal_disk")
            self.assertEqual(meta["viewcell"]["radiusM"], 2.0)
            self.assertAlmostEqual(
                meta["viewcell"]["candidateCameraBackOffsetM"],
                3.464101552963257,
            )
            self.assertEqual(meta["query"]["modelInputFovYDeg"], 66.0)
            self.assertEqual(meta["query"]["frontendRenderFovYDeg"], 60.0)
            ray_space = meta["query"]["raySpace"]
            self.assertEqual(meta["raySpace"], ray_space)
            self.assertEqual(meta["diskAxisBound"], ray_space["diskAxisBound"])
            self.assertEqual(meta["featureDomain"], ray_space["featureDomain"])
            self.assertEqual(meta["rangeGuarantee"], ray_space["rangeGuarantee"])
            self.assertEqual(
                ray_space["diskAxisBound"],
                "per-feature row norm <= 1 - abs(center feature)",
            )
            self.assertEqual(ray_space["featureDomain"], [-1.0, 1.0])
            self.assertEqual(
                ray_space["rangeGuarantee"]["rowFormula"],
                "||B[i,:]||_2 <= 1 - abs(centerFeature[i])",
            )
            self.assertAlmostEqual(
                ray_space["rangeGuarantee"]["maxTwoS"],
                4.0 * np.pi * np.sqrt(9.0) * 8.0,
            )
            self.assertLess(ray_space["rangeGuarantee"]["maxTwoS"], 320.0)
            self.assertTrue(ray_space["rangeGuarantee"]["twoSStrictlyInsideChiRange"])
            self.assertEqual(meta["frequency"]["units"], "cycles")
            self.assertEqual(meta["frequency"]["phaseFactor"], "2*pi")
            self.assertAlmostEqual(meta["frequency"]["twoPi"], 2.0 * np.pi)
            self.assertEqual(meta["chiLookup"]["interpolation"], "piecewise_linear")
            self.assertEqual(meta["chiLookup"]["range"], [0.0, 320.0])
            self.assertEqual(meta["chiLookup"]["table"]["bytes"], CHI_TABLE_SIZE * 4)
            self.assertEqual(meta["depth"]["q01"], 0.11)
            self.assertEqual(meta["depth"]["q99"], 1.91)
            self.assertEqual(meta["depth"]["epsilon"], 1e-4)
            self.assertEqual(meta["provenance"]["dataset"]["splitPoseCounts"], {"train": 2772, "calibration": 168, "validation": 213})
            self.assertEqual(meta["provenance"]["relation"]["edgeCount"], 4)
            self.assertEqual(meta["provenance"]["relation"]["observationCount"], 6)
            self.assertNotIn("datasetDigest", meta["provenance"])
            self.assertNotIn("relationArtifactDigest", meta["provenance"])
            self.assertTrue(meta["safety"]["safe"])
            self.assertEqual(meta["threshold"], 0.02)
            self.assertLessEqual(
                meta["neuralAssetBudget"]["usedBytes"], MAX_NEURAL_ASSET_BYTES
            )

    def test_generic28_exports_equal_capacity_unstructured_table(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._export(
                root, self._control_checkpoint(root, "generic28"), "generic28"
            )
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            table = np.fromfile(
                output / "instance_runtime_features_fp16.bin", dtype="<f2"
            )
            self.assertEqual(table.size, self.num_instances * 124)
            self.assertEqual(meta["fixedTable"]["shape"], [self.num_instances, 124])
            self.assertEqual(
                meta["fixedTable"]["layout"][1]["name"],
                "genericOcclusionFeatures",
            )
            self.assertEqual(
                meta["runtimeFeatureSource"]["source"],
                "checkpoint.geometryFeatures_plus_genericOcclusionFeatures",
            )
            self.assertFalse(meta["provenance"]["relation"]["enabled"])

    def test_no_survival_exports_real_96d_table_and_118d_query(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._export(
                root, self._control_checkpoint(root, "none"), "none"
            )
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            table = np.fromfile(
                output / "instance_runtime_features_fp16.bin", dtype="<f2"
            )
            self.assertEqual(table.size, self.num_instances * 96)
            self.assertEqual(meta["fixedTable"]["shape"], [self.num_instances, 96])
            self.assertEqual(
                meta["fixedTable"]["layout"],
                [{"name": "geometry", "offset": 0, "dim": 96}],
            )
            self.assertEqual(meta["modelConfig"]["runtimeHeadInputDim"], 118)
            self.assertEqual(
                meta["runtimeFeatureSource"]["source"],
                "checkpoint.geometryFeatures_only",
            )
            weight_names = {
                row["name"] for row in meta["networkWeights"]["layout"]
            }
            self.assertFalse(
                any(name.startswith("relation_condition_head") for name in weight_names)
            )
            self.assertFalse(
                any(name.startswith("direction_basis_head") for name in weight_names)
            )

    def test_dry_run_performs_full_validation_without_creating_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_path = root / "dry-run.pt"
            torch.save(self._checkpoint(root), checkpoint_path)
            output = root / "dry-run-out"
            result = export(
                parse_args(
                    [
                        "--checkpoint",
                        str(checkpoint_path),
                        "--runtime-meta",
                        str(self._runtime_meta(root)),
                        "--output-dir",
                        str(output),
                        "--dry-run",
                    ]
                )
            )

            self.assertEqual(result["status"], "dry-run")
            self.assertEqual(result["schema"], EXPORT_SCHEMA)
            self.assertFalse(output.exists())
            self.assertEqual(
                result["files"]["instance_runtime_features_fp16.bin"],
                self.num_instances * RUNTIME_FEATURE_DIM * 2,
            )
            self.assertEqual(result["meta"]["fixedTable"]["shape"], [3, 124])
            self.assertEqual(result["meta"]["calibrationFrozenThreshold"], 0.02)
            self.assertTrue(result["meta"]["safety"]["safe"])
            self.assertTrue(result["meta"]["neuralAssetBudget"]["withinLimit"])

    def test_point_query_ablation_uses_the_same_runtime_bundle_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint["config"]["spectralMode"] = "point"
            checkpoint["protocol"]["variant"] = "core_no_moment"
            output = self._export(root, checkpoint, name="point-query")
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["modelConfig"]["spectralMode"], "point")
            self.assertEqual(meta["fixedTable"]["shape"], [self.num_instances, RUNTIME_FEATURE_DIM])

    def test_unbounded_frequency_contract_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint["config"]["frequency"]["maxNormCycles"] = 8.01
            path = root / "unbounded-frequency.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "range contract"):
                export(
                    parse_args(
                        [
                            "--checkpoint",
                            str(path),
                            "--runtime-meta",
                            str(self._runtime_meta(root)),
                            "--output-dir",
                            str(root / "unbounded-frequency-out"),
                        ]
                    )
                )

    def test_missing_viewcell_radius_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint.pop("viewcell")
            path = root / "missing-viewcell.pt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "explicit checkpoint view-cell radius"):
                export(
                    parse_args(
                        [
                            "--checkpoint",
                            str(path),
                            "--runtime-meta",
                            str(self._runtime_meta(root)),
                            "--output-dir",
                            str(root / "missing-viewcell-out"),
                        ]
                    )
                )

    def test_offline_checkpoint_resources_are_not_in_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = self._export(root, self._checkpoint(root))
            names = {path.name for path in output.iterdir()}
            self.assertNotIn("relation.csr", names)
            self.assertNotIn("groups.bin", names)
            self.assertNotIn("observations.bin", names)
            self.assertNotIn("offline_survival_encoder.pt", names)
            self.assertEqual(
                names,
                {
                    "instance_runtime_features_fp16.bin",
                    "instance_aabb_fp32.bin",
                    "instance_to_glb_uint32.bin",
                    "query_weights_fp16.bin",
                    "frequency_cycles_fp32.bin",
                    "chi_table_fp32.bin",
                    "model_meta.json",
                },
            )
            text = (output / "model_meta.json").read_text(encoding="utf-8")
            self.assertNotIn("/never/pack", text)

    def test_accepts_the_training_entry_checkpoint_field_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            geometry_path = root / "instance_geo_features_fp16.bin"
            np.zeros((self.num_instances, GEO_DIM), dtype="<f2").tofile(geometry_path)
            checkpoint["runtimeSchema"] = checkpoint.pop("modelSchema")
            checkpoint["modelState"] = checkpoint.pop("model")
            checkpoint["modelConfig"] = checkpoint.pop("config")
            checkpoint["geometry"] = {
                "path": str(geometry_path),
                "sha256": "legacy-ignored",
                "shape": [self.num_instances, GEO_DIM],
                "dtype": "float16",
            }
            checkpoint["instanceSurvivalCoefficients"] = torch.zeros(
                (self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM), dtype=torch.float16
            )
            checkpoint["instanceSurvivalPriorCoefficients"] = torch.zeros(
                (self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM), dtype=torch.float16
            )
            checkpoint["instanceSurvivalCalibrationResidual"] = torch.zeros(
                (self.num_instances, SURVIVAL_RANK, SURVIVAL_PARAMETER_DIM), dtype=torch.float16
            )
            checkpoint["protocol"]["dataset"] = {"path": str(root / "dataset")}
            checkpoint.pop("datasetDigest")
            checkpoint.pop("candidateDigest")
            checkpoint.pop("candidateDigests")
            checkpoint["relation"] = {
                "schema": "pvs-viewcell-train-observed-relation-csr-v3",
                "path": str(root / "relation"),
                "edgeCount": 4,
                "rowCount": 5,
                "observationCount": 6,
                "candidateDigest": "2" * 64,
                "artifactDigest": "5" * 64,
            }
            checkpoint.pop("relationArtifactDigest")
            checkpoint["calibration"]["selectedSafe"] = checkpoint["calibration"].pop("selected")
            output = self._export(root, checkpoint, name="training-entry")
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["provenance"]["dataset"]["path"], str((root / "dataset").resolve()))
            self.assertEqual(meta["provenance"]["relation"]["edgeCount"], 4)
            self.assertEqual(meta["calibration"]["source"], "checkpoint.calibration.selectedSafe")
            self.assertEqual(
                meta["runtimeFeatureSource"]["source"],
                "checkpoint.geometryFeatures_plus_survivalCoefficients",
            )
            self.assertNotIn("sha256", meta["runtimeFeatureSource"]["geometry"])
            self.assertNotIn("sha256", meta["fixedTable"])

    def test_legacy_geometry_fingerprint_is_ignored_after_structural_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint["geometry"]["sha256"] = "0" * 64
            output = self._export(root, checkpoint, name="geometry-mismatch")
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertNotIn("sha256", json.dumps(meta))

    def test_only_v4_schema_and_calibration_threshold_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong = self._checkpoint(root)
            wrong["modelSchema"] = "pvs-hierarchical-relation-survival-integrated-v1"
            checkpoint_path = root / "wrong.pt"
            torch.save(wrong, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "modelSchema"):
                export(
                    parse_args(
                        [
                            "--checkpoint",
                            str(checkpoint_path),
                            "--runtime-meta",
                            str(self._runtime_meta(root)),
                            "--output-dir",
                            str(root / "wrong-out"),
                        ]
                    )
                )

            wrong_top_schema = self._checkpoint(root)
            wrong_top_schema["schema"] = MODEL_SCHEMA
            wrong_top_path = root / "wrong-top-schema.pt"
            torch.save(wrong_top_schema, wrong_top_path)
            with self.assertRaisesRegex(ValueError, "checkpoint schema"):
                export(
                    parse_args(
                        [
                            "--checkpoint",
                            str(wrong_top_path),
                            "--runtime-meta",
                            str(self._runtime_meta(root)),
                            "--output-dir",
                            str(root / "wrong-top-schema-out"),
                        ]
                    )
                )

            with self.assertRaisesRegex(ValueError, "must equal the checkpoint calibration"):
                self._export_with_threshold(root, self._checkpoint(root), 0.5)

            mismatched_relation = self._checkpoint(root)
            mismatched_relation["relation"]["candidateDigest"] = "6" * 64
            output = self._export(root, mismatched_relation, name="mismatched-relation")
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["provenance"]["relation"]["schema"], "pvs-viewcell-train-observed-relation-csr-v3")

    def test_export_uses_member_local_geometry_after_source_moves(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint["geometry"]["path"] = "/deleted/old-experiment/geometry.bin"
            output = self._export(root, checkpoint)
            self.assertTrue((output / "instance_runtime_features_fp16.bin").is_file())

    def test_unsafe_calibration_requires_explicit_diagnostic_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._checkpoint(root)
            checkpoint["calibration"]["selected"]["aggregateWeightedRecall"] = 0.98
            checkpoint["calibration"]["selected"]["aggregateWeightedRecallLowerConfidenceBound"] = 0.97
            checkpoint["calibration"]["selected"]["safe"] = False
            path = root / "unsafe.pt"
            torch.save(checkpoint, path)
            runtime_meta = self._runtime_meta(root)
            with self.assertRaisesRegex(ValueError, "unsafe"):
                export(
                    parse_args(
                        [
                            "--checkpoint",
                            str(path),
                            "--runtime-meta",
                            str(runtime_meta),
                            "--output-dir",
                            str(root / "unsafe-out"),
                        ]
                    )
                )
            output = root / "unsafe-diagnostic-out"
            result = export(
                parse_args(
                    [
                        "--checkpoint",
                        str(path),
                        "--runtime-meta",
                        str(runtime_meta),
                        "--output-dir",
                        str(output),
                        "--allow-unsafe-threshold",
                    ]
                )
            )
            self.assertEqual(result["threshold"], 0.02)
            meta = json.loads((output / "model_meta.json").read_text(encoding="utf-8"))
            self.assertFalse(meta["safety"]["safe"])
            self.assertEqual(meta["calibration"]["source"], "checkpoint.calibration.selected")

    def test_neural_asset_budget_is_a_hard_gate(self) -> None:
        with self.assertRaisesRegex(ValueError, "budget exceeded"):
            _check_neural_asset_budget(
                {
                    "instance_runtime_features_fp16.bin": b"0",
                    "query_weights_fp16.bin": b"0" * (MAX_NEURAL_ASSET_BYTES + 1),
                    "frequency_cycles_fp32.bin": b"0",
                    "chi_table_fp32.bin": b"0",
                }
            )

    def test_neural_asset_budget_scales_with_fixed_instance_table(self) -> None:
        instance_count = 41_298
        expected = instance_count * MAX_NEURAL_ASSET_BYTES_PER_INSTANCE + MAX_SHARED_NEURAL_ASSET_BYTES
        self.assertEqual(_neural_asset_budget_limit(instance_count), expected)
        used = instance_count * 248 + 64 * 1024
        self.assertEqual(
            _check_neural_asset_budget(
                {
                    "instance_runtime_features_fp16.bin": b"0" * (instance_count * 248),
                    "query_weights_fp16.bin": b"0" * (32 * 1024),
                    "frequency_cycles_fp32.bin": b"0" * (16 * 1024),
                    "chi_table_fp32.bin": b"0" * (16 * 1024),
                },
                instance_count,
            ),
            used,
        )

    def _export_with_threshold(self, root: Path, checkpoint: dict, threshold: float) -> Path:
        checkpoint_path = root / "threshold.pt"
        torch.save(checkpoint, checkpoint_path)
        output = root / "threshold-out"
        export(
            parse_args(
                [
                    "--checkpoint",
                    str(checkpoint_path),
                    "--runtime-meta",
                    str(self._runtime_meta(root)),
                    "--output-dir",
                    str(output),
                    "--threshold",
                    str(threshold),
                ]
            )
        )
        return output


if __name__ == "__main__":
    unittest.main()
