#!/usr/bin/env python3
"""Run the calibrated relation-prior survival experiment.

The runner owns only experiment orchestration.  Training is performed by
``train_bounded_relation_survival_moment_safety.py`` and formal evaluation is
delegated to the registered validator/evaluator/summarizer CLIs.  The local
preflight and checkpoint checks are v4-specific, so a legacy checkpoint or a
legacy relation artifact cannot silently enter this line.

No test split is opened by this module.  Formal validation is a separate,
calibration-frozen post-processing pipeline.  A failed post-processing stage
leaves all completed training members untouched and records
``training_complete_evaluation_incomplete``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import audit_native_aabb_candidates  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    SCHEMA_V3 as RELATION_SCHEMA_V3,
    load_survival_observations_v3,
    validate_relation_metadata_v3,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from v4_semantic_contract import (  # noqa: E402
    validate_native_candidate_contract,
    validate_replay_payload,
)


PREFIX = "pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4"
DATA_PREFIX = "pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3"
TRAIN = ROOT / "neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py"
DEFAULT_DATASET = ROOT / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3"
DEFAULT_RELATION = ROOT / "neural_instance_culling/dataset/out" / DATA_PREFIX / "bounded_relation_csr_v3"
DEFAULT_RUNTIME_META = ROOT / "hkust-v3/assets/runtimeVisibilityMeta.json"
DEFAULT_GEO = ROOT / (
    "neural_instance_culling/model/out/"
    "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/"
    "instance_geo_features_fp16.bin"
)
DEFAULT_GLB_INDEX = ROOT / "hkust-v3/assets/glbIndex.json"
DEFAULT_GLB_ROOT = ROOT / "hkust-v3/assets"
DEFAULT_FORMAL_VALIDATION = ROOT / "neural_instance_culling/benchmark/out" / f"{PREFIX}_formal_validation"

VALIDATOR_CLI = ROOT / "neural_instance_culling/benchmark/validate_pvs_bounded_relation_survival_moment_v4.py"
EVALUATOR_CLI = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py"
SUMMARIZER_CLI = ROOT / "neural_instance_culling/benchmark/summarize_pvs_bounded_relation_survival_moment_v4.py"
EXPORTER_CLI = ROOT / "neural_instance_culling/model/export_bounded_relation_survival_moment.py"
IMAGE_CLI = ROOT / "neural_instance_culling/benchmark/evaluate_viewcell_image_per.py"
RESOURCE_CLI = ROOT / "neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py"
ROUTE_CLI = ROOT / "neural_instance_culling/benchmark/decide_pvs_bounded_relation_survival_moment_v4.py"

CALIBRATION_SUMMARY_SCHEMA_V4 = "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4"
FROZEN_CONFIG_SCHEMA_V4 = "pvs-bounded-relation-prior-instance-calibrated-v4-scan-frozen-config-v1"
DATASET_SCHEMA_V4 = "pose-csr-viewcell-back-camera-color-id-fov66-v2"
DATASET_EXPERIMENT_V4 = "pvs_bounded_relation_survival_moment_v3"
EXPECTED_NUM_INSTANCES = 18831
EXPECTED_SPLIT_POSE_COUNTS = {"train": 2772, "calibration": 168, "validation": 213}
TARGET_WEIGHTED_RECALL = 0.99
MINIMUM_WEIGHTED_RECALL_LCB = 0.99
# The checkpoint stores prior, applied residual, and fused coefficients as
# separate FP16 tensors.  Comparing their FP32 views needs a small bound for
# three independent FP16 roundings; this is not a training-quality tolerance.
MAX_FP16_FUSION_ABS_ERROR = 0.02

FORMAL_SEEDS = (20260801, 20260802, 20260803)
SCAN_SEED = 20260801
SMOKE_SEED = 20260801

FORMAL_VARIANTS: tuple[str, ...] = (
    "full",
    "without_bounded_relation",
    "without_viewcell_moment_envelope",
    "without_safety_reserve_utility",
    "without_instance_calibration_residual",
)

DEFAULT_HYPERPARAMETERS: dict[str, float] = {
    "learningRate": 2e-4,
    "weightDecay": 1e-5,
    "survivalLossWeight": 0.25,
    "relationConsistencyWeight": 0.10,
    "utilityLossWeight": 0.10,
    "downloadLossWeight": 0.10,
    "boundaryTailWeight": 0.30,
    "negativeBandWeight": 0.03,
    "glbResourceWeight": 0.015,
    "relationGradientCap": 0.25,
    "scheduleGradientCap": 0.25,
    "efficiencyGradientCap": 0.25,
    "instanceCalibrationRegularizationWeight": 0.02,
    "instanceCalibrationMaxAbs": 4.0,
    "sparseInstancePenalty": 3.0,
    "instanceCalibrationWarmupFraction": 0.10,
    "instanceCalibrationRampFraction": 0.20,
}

# These eight points are fixed before scan8 starts.  They cover the
# registered ranges without allowing a result-dependent point to be appended.
SCAN_CONFIGS: tuple[dict[str, Any], ...] = tuple(
    {
        "configId": f"s{index:02d}",
        "hyperparameters": {
            **DEFAULT_HYPERPARAMETERS,
            "learningRate": learning_rate,
            "survivalLossWeight": survival,
            "relationConsistencyWeight": relation,
            "boundaryTailWeight": tail,
            "negativeBandWeight": negative,
            "glbResourceWeight": resource,
            "relationGradientCap": gradient,
            "efficiencyGradientCap": efficiency,
            "instanceCalibrationRegularizationWeight": calibration_regularization,
        },
    }
    for index, (
        learning_rate,
        survival,
        relation,
        tail,
        negative,
        resource,
        gradient,
        efficiency,
        calibration_regularization,
    ) in enumerate(
        (
            (1e-4, 0.10, 0.05, 0.15, 0.01, 0.005, 0.10, 0.10, 0.005),
            (2e-4, 0.25, 0.10, 0.30, 0.03, 0.015, 0.25, 0.25, 0.020),
            (3e-4, 0.40, 0.10, 0.50, 0.06, 0.030, 0.25, 0.25, 0.080),
            (1e-4, 0.25, 0.05, 0.50, 0.06, 0.005, 0.10, 0.25, 0.005),
            (3e-4, 0.10, 0.10, 0.15, 0.01, 0.030, 0.25, 0.10, 0.020),
            (2e-4, 0.40, 0.05, 0.30, 0.03, 0.030, 0.10, 0.25, 0.080),
            (1e-4, 0.40, 0.10, 0.50, 0.01, 0.015, 0.25, 0.10, 0.020),
            (3e-4, 0.25, 0.05, 0.15, 0.06, 0.015, 0.10, 0.25, 0.005),
        )
    )
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _required_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _required_dir(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _ensure_new_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _next_attempt_path(root: Path, stem: str, suffix: str = ".json") -> Path:
    first = root / f"{stem}{suffix}"
    if not first.exists():
        return first
    index = 2
    while True:
        candidate = root / f"{stem}_attempt_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def _latest_record(root: Path, stem: str) -> tuple[Path, dict[str, Any]] | None:
    paths = sorted(root.glob(f"{stem}.json")) + sorted(root.glob(f"{stem}_attempt_*.json"))
    if not paths:
        return None
    path = paths[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"record is not an object: {path}")
    return path, payload


def _variant_spec(name: str) -> dict[str, Any]:
    if name == "full":
        return {"relationSource": "bounded_hierarchical", "spectralMode": "moment_envelope", "lossVariant": "safety_reserve", "relationArtifact": "native", "instanceCalibrationMode": "residual"}
    if name == "without_bounded_relation":
        return {"relationSource": "geometry_only", "spectralMode": "moment_envelope", "lossVariant": "safety_reserve", "relationArtifact": "native", "instanceCalibrationMode": "residual"}
    if name == "without_viewcell_moment_envelope":
        return {"relationSource": "bounded_hierarchical", "spectralMode": "point", "lossVariant": "safety_reserve", "relationArtifact": "native", "instanceCalibrationMode": "residual"}
    if name == "without_safety_reserve_utility":
        return {"relationSource": "bounded_hierarchical", "spectralMode": "moment_envelope", "lossVariant": "normalized_rvl", "relationArtifact": "native", "instanceCalibrationMode": "residual"}
    if name == "without_instance_calibration_residual":
        return {"relationSource": "bounded_hierarchical", "spectralMode": "moment_envelope", "lossVariant": "safety_reserve", "relationArtifact": "native", "instanceCalibrationMode": "disabled"}
    raise ValueError(f"unknown v4 variant: {name}")


def _validate_requested(values: Sequence[str] | None, expected: Sequence[str], label: str) -> list[str]:
    actual = list(expected) if values is None else [str(value) for value in values]
    if actual != list(expected) or len(actual) != len(set(actual)):
        raise ValueError(f"{label} must be exactly {list(expected)} in registered order")
    return actual


def _validate_seed_set(values: Sequence[int] | None, expected: Sequence[int], label: str) -> list[int]:
    actual = list(expected) if values is None else [int(value) for value in values]
    if actual != list(expected) or len(actual) != len(set(actual)):
        raise ValueError(f"{label} must be exactly {list(expected)} in registered order")
    return actual


def _mode_config(
    mode: str,
    requested_variants: Sequence[str] | None = None,
    requested_seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Return the immutable member contract for one runner mode."""
    mode = str(mode)
    if mode == "smoke":
        names = ["full"] if requested_variants is None else [str(value) for value in requested_variants]
        seeds = [SMOKE_SEED] if requested_seeds is None else [int(value) for value in requested_seeds]
        if names != ["full"] or seeds != [SMOKE_SEED]:
            raise ValueError("smoke accepts exactly the registered full model and seed")
        return {"mode": mode, "variants": names, "seeds": seeds, "epochs": 1, "stepsPerEpoch": 1, "observationBatchSize": 32768, "evalEvery": 1, "snapshotEvery": 1, "maxEvalPoses": 2, "calibrationBootstrapReplicates": 2, "formalPostprocess": False, "testRead": False}
    if mode == "scan8":
        if requested_variants not in (None, [], ["full"]):
            raise ValueError("scan8 only trains the registered full model")
        seeds = _validate_seed_set(requested_seeds, (SCAN_SEED,), "scan8 seeds")
        return {"mode": mode, "variants": ["full"], "seeds": seeds, "epochs": 8, "stepsPerEpoch": 50, "evalEvery": 8, "snapshotEvery": 4, "maxEvalPoses": 0, "calibrationBootstrapReplicates": 1000, "scanConfigs": [dict(value) for value in SCAN_CONFIGS], "formalPostprocess": False, "testRead": False}
    if mode == "formal80":
        names = _validate_requested(requested_variants, FORMAL_VARIANTS, "formal80 variants")
        seeds = _validate_seed_set(requested_seeds, FORMAL_SEEDS, "formal80 seeds")
        return {"mode": mode, "variants": names, "seeds": seeds, "epochs": 80, "stepsPerEpoch": 100, "evalEvery": 4, "snapshotEvery": 4, "maxEvalPoses": 0, "calibrationBootstrapReplicates": 10000, "formalPostprocess": True, "testRead": False}
    raise ValueError(f"unsupported v4 runner mode: {mode!r}")


def _mode_settings(
    mode: str,
    requested_variants: Sequence[str] | None = None,
    requested_seeds: Sequence[int] | None = None,
) -> tuple[int, int, list[str], list[int]]:
    """Compatibility-shaped view used by small mode tests and callers."""
    config = _mode_config(mode, requested_variants, requested_seeds)
    return int(config["epochs"]), int(config["stepsPerEpoch"]), list(config["variants"]), list(config["seeds"])


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_frozen_config_payload(payload: Mapping[str, Any], source: Path) -> dict[str, Any]:
    expected_schema = FROZEN_CONFIG_SCHEMA_V4
    if payload.get("schema") != expected_schema or payload.get("testRead") is not False:
        raise ValueError(f"scan frozen configuration has the wrong schema or opened test: {source}")
    config_id = str(payload.get("configId", ""))
    registered = {str(config["configId"]): config for config in SCAN_CONFIGS}
    if config_id not in registered:
        raise ValueError(f"scan frozen configuration has an unregistered configId: {config_id!r}")
    if not isinstance(payload.get("hyperparameters"), Mapping):
        raise ValueError(f"scan frozen configuration has no hyperparameters: {source}")
    groups = payload.get("groups")
    if not isinstance(groups, list) or len(groups) != len(SCAN_CONFIGS):
        raise ValueError("scan frozen configuration does not contain all eight scan groups")
    expected_ids = set(registered)
    actual_ids = {str(group.get("configId", "")) for group in groups if isinstance(group, Mapping)}
    if actual_ids != expected_ids:
        raise ValueError("scan frozen configuration group IDs do not match the registered eight configurations")
    if not all(
        isinstance(group, Mapping)
        and bool(group.get("completeScanMember"))
        and bool(group.get("metricContractComplete"))
        for group in groups
    ):
        raise ValueError("scan frozen configuration was not produced from eight complete scan members")
    selected = payload.get("selected")
    if not isinstance(selected, Mapping) or str(selected.get("configId", "")) != config_id:
        raise ValueError("scan frozen configuration selected record disagrees with configId")
    if dict(payload["hyperparameters"]) != dict(registered[config_id]["hyperparameters"]):
        raise ValueError("scan frozen configuration hyperparameters disagree with the registered scan point")
    return {
        "source": str(source.resolve()),
        "configId": config_id,
        "hyperparameters": dict(payload["hyperparameters"]),
        "selection": payload,
    }


def _frozen_config_from_scan(scan_root: Path) -> dict[str, Any]:
    frozen = scan_root / "frozen_config.json"
    if not frozen.is_file():
        raise FileNotFoundError(f"scan root has no immutable frozen configuration: {scan_root}")
    return _validate_frozen_config_payload(_load_json(frozen), frozen)


def _metric_from_calibration(member_dir: Path, key: str) -> float | None:
    path = member_dir / "calibration_ready_summary.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    validation = payload.get("validationAtFrozenThreshold")
    if not isinstance(validation, Mapping):
        return None
    aliases = {
        "weightedRecall": ("aggregateWeightedRecall", "agg_weighted_recall"),
        "weightedRecallLcb": (
            "aggregateWeightedRecallLowerConfidenceBound",
            "aggregate_weighted_recall_lower_confidence_bound",
        ),
        "balancedAccuracy": ("agg_balanced_accuracy", "balancedAccuracy"),
        "usefulCull": ("agg_useful_cull", "usefulCull"),
        "precision": ("agg_precision", "precision"),
        "predictedCount": ("avg_pred_count", "avgPredCount"),
        "predictedGlbBytes": ("avg_pred_glb_bytes", "predictedGlbBytes"),
    }
    for field in aliases.get(key, (key,)):
        try:
            value = float(validation[field])
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _select_scan_configuration(
    output_root: Path,
    training_results: Sequence[Mapping[str, Any]],
    *,
    run_manifest: Mapping[str, Any] | None = None,
    jobs: Sequence[Mapping[str, Any]] | None = None,
    expected_epochs: int | None = None,
) -> dict[str, Any]:
    """Freeze one deterministic scan configuration for formal80."""
    jobs_by_key = {
        (str(job.get("configId", "")), str(job.get("variant", "")), int(job.get("seed", -1))): job
        for job in (jobs or ())
    }
    groups: list[dict[str, Any]] = []
    expected = {int(SCAN_SEED)}
    for config in SCAN_CONFIGS:
        config_id = str(config["configId"])
        rows = [
            row for row in training_results
            if str(row.get("configId")) == config_id and int(row.get("returnCode", 1)) == 0
        ]
        seed_set = {int(row.get("seed", -1)) for row in rows}
        safe = []
        pose_contracts: list[bool] = []
        metrics: list[dict[str, float | None]] = []
        missing_by_seed: dict[str, list[str]] = {}
        for row in rows:
            member_dir = Path(str(row.get("outputDir", output_root / str(row.get("member", "")))))
            summary = _load_json(member_dir / "calibration_ready_summary.json") if (member_dir / "calibration_ready_summary.json").is_file() else {}
            member_metrics = {
                "weightedRecall": _metric_from_calibration(member_dir, "weightedRecall"),
                "weightedRecallLcb": _metric_from_calibration(member_dir, "weightedRecallLcb"),
                "balancedAccuracy": _metric_from_calibration(member_dir, "balancedAccuracy"),
                "usefulCull": _metric_from_calibration(member_dir, "usefulCull"),
                "precision": _metric_from_calibration(member_dir, "precision"),
                "predictedGlbBytes": _metric_from_calibration(member_dir, "predictedGlbBytes"),
                "predictedCount": _metric_from_calibration(member_dir, "predictedCount"),
            }
            metrics.append(member_metrics)
            missing = sorted(key for key, value in member_metrics.items() if value is None)
            checkpoint_contract_error: str | None = None
            if run_manifest is not None:
                job = jobs_by_key.get(
                    (config_id, str(row.get("variant", "full")), int(row.get("seed", -1)))
                )
                try:
                    if job is None:
                        raise ValueError("scan result has no registered job identity")
                    if not _training_artifacts_complete(
                        member_dir,
                        expected_epochs=expected_epochs,
                    ):
                        raise ValueError("scan member did not reach the registered epoch budget")
                    checkpoint = _selected_checkpoint(member_dir)
                    _validate_v4_checkpoint(checkpoint, job, run_manifest)
                except Exception as exc:  # preserve the reason in the freeze summary
                    checkpoint_contract_error = str(exc)
            validation_row = summary.get("validationAtFrozenThreshold")
            calibration_best = summary.get("bestSafe") or summary.get("bestDiagnostic")
            calibration_selection = (
                calibration_best.get("selection", calibration_best)
                if isinstance(calibration_best, Mapping)
                else None
            )
            pose_contract = bool(
                isinstance(validation_row, Mapping)
                and int(validation_row.get("eval_pose_count", -1)) == EXPECTED_SPLIT_POSE_COUNTS["validation"]
                and isinstance(calibration_selection, Mapping)
                and int(calibration_selection.get("eval_pose_count", -1)) == EXPECTED_SPLIT_POSE_COUNTS["calibration"]
            )
            if checkpoint_contract_error is not None:
                pose_contract = False
            pose_contracts.append(pose_contract)
            if not pose_contract:
                missing.extend(["calibrationPoseCount", "validationPoseCount"])
            if checkpoint_contract_error is not None:
                missing.extend(["checkpointContract", f"checkpointContractError:{checkpoint_contract_error}"])
            if missing:
                missing_by_seed[str(int(row.get("seed", -1)))] = sorted(set(missing))
            validation_safe = (
                summary.get("status") == "safe"
                and member_metrics["weightedRecall"] is not None
                and member_metrics["weightedRecallLcb"] is not None
                and float(member_metrics["weightedRecall"]) > TARGET_WEIGHTED_RECALL
                and float(member_metrics["weightedRecallLcb"]) > MINIMUM_WEIGHTED_RECALL_LCB
            )
            safe.append(validation_safe)
        metric_names = (
            "weightedRecall", "weightedRecallLcb", "balancedAccuracy", "usefulCull",
            "precision", "predictedGlbBytes", "predictedCount",
        )
        mean: dict[str, float | None] = {}
        for key in metric_names:
            values = [value[key] for value in metrics]
            mean[key] = (
                float(np.mean(np.asarray(values, dtype=np.float64)))
                if values and all(value is not None and np.isfinite(value) for value in values)
                else None
            )
        metric_contract_complete = bool(
            len(rows) == len(expected)
            and metrics
            and pose_contracts
            and all(pose_contracts)
            and all(
                value is not None and np.isfinite(value)
                for row_metrics in metrics
                for value in row_metrics.values()
            )
        )
        groups.append({
            "configId": config_id,
            "hyperparameters": dict(config["hyperparameters"]),
            "seedCount": len(seed_set),
            "completeScanMember": seed_set == expected and len(rows) == len(expected),
            "metricContractComplete": metric_contract_complete,
            "poseContractComplete": bool(pose_contracts) and all(pose_contracts),
            "missingValidationMetricsBySeed": missing_by_seed,
            "safeAcrossScan": (
                seed_set == expected
                and len(rows) == len(expected)
                and metric_contract_complete
                and all(safe)
            ),
            "mean": mean,
            "testRead": False,
        })
    complete = [
        group
        for group in groups
        if group["completeScanMember"] and group["metricContractComplete"]
    ]
    if not complete:
        raise RuntimeError(
            "scan8 has no complete configuration with validation classification/resource metrics"
        )
    safe = [group for group in complete if group["safeAcrossScan"]]
    pool = safe or complete
    if safe:
        ordered = sorted(
            pool,
            key=lambda group: (
                -float(group["mean"]["balancedAccuracy"]),
                -float(group["mean"]["usefulCull"]),
                -float(group["mean"]["precision"]),
                float(group["mean"]["predictedGlbBytes"]),
                float(group["mean"]["predictedCount"]),
                -float(group["mean"]["weightedRecallLcb"]),
                -float(group["mean"]["weightedRecall"]),
                str(group["configId"]),
            ),
        )
        selection_rule = (
            "safe pool first; balanced accuracy, useful cull, precision, predicted GLB bytes, "
            "predicted count, weighted recall LCB/point estimate, config ID"
        )
    else:
        ordered = sorted(
            pool,
            key=lambda group: (
                -float(group["mean"]["weightedRecallLcb"]),
                -float(group["mean"]["weightedRecall"]),
                -float(group["mean"]["balancedAccuracy"]),
                -float(group["mean"]["usefulCull"]),
                -float(group["mean"]["precision"]),
                float(group["mean"]["predictedGlbBytes"]),
                float(group["mean"]["predictedCount"]),
                str(group["configId"]),
            ),
        )
        selection_rule = (
            "diagnostic pool; weighted recall LCB/point estimate, balanced accuracy, useful cull, "
            "precision, predicted GLB bytes/count, config ID"
        )
    winner = ordered[0]
    return {
        "schema": FROZEN_CONFIG_SCHEMA_V4,
        "configId": winner["configId"],
        "hyperparameters": dict(winner["hyperparameters"]),
        "selectionPool": "safe_candidate_pool" if safe else "diagnostic_candidate_pool",
        "selectionRule": selection_rule,
        "groups": groups,
        "selected": winner,
        "testRead": False,
    }


def _job_hyperparameters(job: Mapping[str, Any]) -> dict[str, Any]:
    value = job.get("hyperparameters", DEFAULT_HYPERPARAMETERS)
    if not isinstance(value, Mapping):
        raise ValueError("job hyperparameters must be an object")
    return {str(key): value[key] for key in value}


def _make_jobs(
    args: argparse.Namespace,
    mode_config: Mapping[str, Any],
    frozen_config: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    mode = str(mode_config["mode"])
    epochs = int(mode_config["epochs"])
    jobs: list[dict[str, Any]] = []
    if mode == "scan8":
        for config in mode_config["scanConfigs"]:
            for seed in mode_config["seeds"]:
                config_id = str(config["configId"])
                jobs.append({
                    "variant": "full",
                    "seed": int(seed),
                    "configId": config_id,
                    "hyperparameters": dict(config["hyperparameters"]),
                    "variantSpec": _variant_spec("full"),
                    "member": f"{config_id}_full_seed{int(seed)}_e{epochs}",
                    "testRead": False,
                })
        return jobs
    for variant in mode_config["variants"]:
        for seed in mode_config["seeds"]:
            hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
            config_id = "registered"
            if mode == "formal80":
                if frozen_config is None:
                    hyperparameters = dict(DEFAULT_HYPERPARAMETERS)
                    config_id = "pending_scan_freeze"
                else:
                    hyperparameters = dict(frozen_config["hyperparameters"])
                    config_id = str(frozen_config.get("configId", "frozen"))
            jobs.append({
                "variant": str(variant),
                "seed": int(seed),
                "configId": config_id,
                "hyperparameters": hyperparameters,
                "variantSpec": _variant_spec(str(variant)),
                "member": f"{variant}_seed{int(seed)}_e{epochs}",
                "testRead": False,
            })
    return jobs


def _split_name(dataset: PoseCSRDataset, requested: str) -> str:
    if requested in dataset.split_ids:
        return requested
    raise ValueError(f"dataset has no explicit {requested!r} split")


def _relation_preflight(
    relation_dir: Path,
    *,
    num_instances: int,
) -> dict[str, Any]:
    relation_dir = _required_dir(relation_dir, "v3 relation artifact")
    metadata_path = _required_file(relation_dir / "relation_csr_meta.json", "relation metadata")
    metadata = _load_json(metadata_path)
    if metadata.get("schema") != RELATION_SCHEMA_V3:
        raise ValueError(f"relation artifact is not v3: {relation_dir}")
    validate_relation_metadata_v3(metadata, expected_num_instances=num_instances)
    relation = ObservedRelationCSR.load(relation_dir)
    observations = load_survival_observations_v3(relation_dir, metadata)
    hierarchy = metadata.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        raise ValueError(f"relation hierarchy declaration is missing: {relation_dir}")
    hierarchy_files = hierarchy.get("files")
    if not isinstance(hierarchy_files, Mapping):
        raise ValueError(f"relation hierarchy files are missing: {relation_dir}")
    hierarchy_shapes: dict[str, int] = {}
    for key in ("localGroupIds", "structuralGroupIds"):
        filename = hierarchy_files.get(key)
        path = _required_file(relation_dir / str(filename), f"relation hierarchy file {key}")
        values = np.fromfile(path, dtype="<u4")
        if values.size != num_instances:
            raise ValueError(
                f"relation hierarchy file {key} has {values.size} entries, expected {num_instances}"
            )
        hierarchy_shapes[key] = int(values.size)
    stats = metadata.get("stats")
    if not isinstance(stats, Mapping):
        raise ValueError(f"relation stats declaration is missing: {relation_dir}")
    declared_edge_count = int(stats.get("edgeCount", relation.edge_count))
    declared_row_count = int(stats.get("rowCount", relation.row_count))
    declared_observation_count = int(stats.get("survivalObservationCount", observations["instance"].size))
    if declared_edge_count != relation.edge_count or declared_row_count != relation.row_count:
        raise ValueError(f"relation edge/row counts disagree with binary CSR: {relation_dir}")
    if declared_observation_count != int(observations["instance"].size):
        raise ValueError(f"relation observation count disagrees with binary observations: {relation_dir}")
    return {
        "path": str(relation_dir),
        "schema": RELATION_SCHEMA_V3,
        "numInstances": int(relation.num_instances),
        "directionBins": int(relation.direction_bins),
        "depthShells": int(relation.depth_shells),
        "edgeCount": int(relation.edge_count),
        "rowCount": int(relation.row_count),
        "observationCount": int(observations["instance"].size),
        "eventCount": int(np.sum(np.asarray(observations["event"]) == 1)),
        "rightCensoredCount": int(np.sum(np.asarray(observations["event"]) == 0)),
        "subposeCount": int(np.unique(np.asarray(observations["subpose"])).size),
        "hierarchy": {
            "localGroupCount": int(hierarchy.get("localGroupCount", 0)),
            "structuralGroupCount": int(hierarchy.get("structuralGroupCount", 0)),
            "localMaxSize": int(hierarchy.get("localMaxSize", 0)),
            "structuralMaxSize": int(hierarchy.get("structuralMaxSize", 0)),
            "maxStructuralInstanceFraction": float(hierarchy.get("maxStructuralInstanceFraction", 0.0)),
            "fileLengths": hierarchy_shapes,
        },
        "trainOnly": metadata.get("trainOnly") is True,
        "splitNames": list(metadata.get("splitNames", [])),
        "testRead": False,
    }


def _preflight(args: argparse.Namespace, mode_config: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate data identity without opening a test split."""
    dataset_dir = _required_dir(Path(args.dataset_dir), "PoseCSR dataset")
    runtime_path = _required_file(Path(args.runtime_meta), "runtime metadata")
    geometry_path = _required_file(Path(args.initial_geo_features), "fixed geometry features")
    glb_index = _required_file(Path(args.glb_index), "GLB index")
    _required_dir(Path(args.glb_root), "GLB root")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_path)
    num_instances = int(world_aabbs.shape[0])
    if num_instances != EXPECTED_NUM_INSTANCES:
        raise ValueError(
            f"v4 requires {EXPECTED_NUM_INSTANCES} runtime instances, got {num_instances}"
        )
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances, subpose_sidecar=(Path(args.subpose_sidecar) if args.subpose_sidecar else None))
    if dataset.meta.get("schema") != DATASET_SCHEMA_V4:
        raise ValueError(f"v4 requires dataset schema {DATASET_SCHEMA_V4!r}")
    if dataset.meta.get("experiment") != DATASET_EXPERIMENT_V4:
        raise ValueError(f"v4 requires dataset experiment {DATASET_EXPERIMENT_V4!r}")
    train_name = _split_name(dataset, "train")
    calibration_name = _split_name(dataset, "calibration")
    validation_name = "validation" if "validation" in dataset.split_ids else _split_name(dataset, "val")
    split_names = {"train": train_name, "calibration": calibration_name, "validation": validation_name}
    splits = {key: dataset.split(name) for key, name in split_names.items()}
    split_pose_indices = {key: np.asarray(value.pose_indices, dtype="<i8") for key, value in splits.items()}
    actual_split_counts = {key: int(value.size) for key, value in split_pose_indices.items()}
    if actual_split_counts != EXPECTED_SPLIT_POSE_COUNTS:
        raise ValueError(
            f"v4 split pose counts disagree with the registered dataset: {actual_split_counts}"
        )
    split_semantics = {
        key: validate_native_candidate_contract(
            dataset,
            value,
            num_instances=num_instances,
        )
        for key, value in split_pose_indices.items()
    }
    candidate_audits = {
        key: audit_native_aabb_candidates(dataset, world_aabbs, value)
        for key, value in split_pose_indices.items()
    }
    if geometry_path.stat().st_size != num_instances * 96 * 2:
        raise ValueError("initial geometry features must be N x 96 FP16")
    if not np.isfinite(np.asarray(world_aabbs)).all() or not np.isfinite(np.asarray(instance_to_glb)).all():
        raise ValueError("runtime metadata contains non-finite instance data")
    relation_dirs = {str(job["variantSpec"]["relationArtifact"]): Path(args.relation_dir) for job in jobs}
    relation_artifacts = {
        role: _relation_preflight(path, num_instances=num_instances)
        for role, path in sorted(relation_dirs.items())
    }
    return {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-preflight-v1",
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "schema": str(dataset.meta.get("schema")),
            "experiment": str(dataset.meta.get("experiment")),
        },
        "runtimeMeta": {
            "path": str(runtime_path),
            "numInstances": num_instances,
            "numGlbs": int(runtime.get("numGlbs", int(instance_to_glb.max()) + 1)),
        },
        "geometry": {
            "path": str(geometry_path),
            "contract": "formal80 uses pvs_directional_occlusion_proxy_encoder_rvl_w042_full40",
            "shape": [num_instances, 96],
            "dtype": "float16",
            "byteLength": int(geometry_path.stat().st_size),
        },
        "glb": {"index": str(glb_index), "root": str(Path(args.glb_root).resolve())},
        "splitPoseCounts": {key: int(value.size) for key, value in split_pose_indices.items()},
        "splitPoseIndices": {key: [int(value) for value in values.tolist()] for key, values in split_pose_indices.items()},
        "splitSemantics": split_semantics,
        "candidateAudits": candidate_audits,
        "relationArtifacts": relation_artifacts,
        "candidateSemantics": "stored native back-camera candidates; GT union disabled",
        "visibleWeightSemantics": str(dataset.visible_weight_semantics),
        "testRead": False,
    }


def _relation_dir_for_job(args: argparse.Namespace, job: Mapping[str, Any]) -> Path:
    return Path(args.relation_dir).resolve()


def _train_command(args: argparse.Namespace, job: Mapping[str, Any], member: Path, mode_config: Mapping[str, Any]) -> list[str]:
    hp = _job_hyperparameters(job)
    spec = job["variantSpec"]
    observation_batch_size = int(mode_config.get("observationBatchSize", args.observation_batch_size))
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(Path(args.dataset_dir).resolve()),
        "--relation-dir", str(_relation_dir_for_job(args, job)),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--initial-geo-features", str(Path(args.initial_geo_features).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--output-dir", str(member),
        "--experiment-name", f"{PREFIX}_{job['variant']}_seed{int(job['seed'])}_e{int(mode_config['epochs'])}",
        "--variant", str(job["variant"]),
        "--relation-source", str(spec["relationSource"]),
        "--spectral-mode", str(spec["spectralMode"]),
        "--loss-variant", str(spec["lossVariant"]),
        "--instance-calibration-mode", str(spec["instanceCalibrationMode"]),
        "--epochs", str(int(mode_config["epochs"])),
        "--steps-per-epoch", str(int(mode_config["stepsPerEpoch"])),
        "--poses-per-batch", str(int(args.poses_per_batch)),
        "--observation-batch-size", str(observation_batch_size),
        "--eval-every", str(int(mode_config["evalEvery"])),
        "--snapshot-every", str(int(mode_config.get("snapshotEvery", mode_config["evalEvery"]))),
        "--max-eval-poses", str(int(mode_config["maxEvalPoses"])),
        "--calibration-bootstrap-replicates", str(int(mode_config["calibrationBootstrapReplicates"])),
        "--seed", str(int(job["seed"])),
        "--device", "cuda",
        "--learning-rate", str(float(hp["learningRate"])),
        "--weight-decay", str(float(hp["weightDecay"])),
        "--survival-loss-weight", str(float(hp["survivalLossWeight"])),
        "--relation-consistency-weight", str(float(hp["relationConsistencyWeight"])),
        "--utility-loss-weight", str(float(hp["utilityLossWeight"])),
        "--download-loss-weight", str(float(hp["downloadLossWeight"])),
        "--boundary-tail-weight", str(float(hp["boundaryTailWeight"])),
        "--negative-band-weight", str(float(hp["negativeBandWeight"])),
        "--glb-resource-weight", str(float(hp["glbResourceWeight"])),
        "--relation-gradient-cap", str(float(hp["relationGradientCap"])),
        "--schedule-gradient-cap", str(float(hp["scheduleGradientCap"])),
        "--efficiency-gradient-cap", str(float(hp["efficiencyGradientCap"])),
        "--instance-calibration-regularization-weight", str(float(hp["instanceCalibrationRegularizationWeight"])),
        "--instance-calibration-max-abs", str(float(hp["instanceCalibrationMaxAbs"])),
        "--sparse-instance-penalty", str(float(hp["sparseInstancePenalty"])),
        "--instance-calibration-warmup-fraction", str(float(hp["instanceCalibrationWarmupFraction"])),
        "--instance-calibration-ramp-fraction", str(float(hp["instanceCalibrationRampFraction"])),
    ]
    if args.subpose_sidecar:
        command.extend(["--subpose-sidecar", str(Path(args.subpose_sidecar).resolve())])
    if args.allow_missing_glb_costs:
        command.append("--allow-missing-glb-costs")
    return command


def _training_artifacts_complete(member: Path, *, expected_epochs: int | None = None) -> bool:
    """Return true only for a member that reached the requested epoch budget.

    Presence of a few files is not enough for resume: an interrupted process can
    leave those files behind before the final epoch is written.  The training
    history is written after every completed epoch and is therefore the cheap,
    checkpoint-independent completion marker used by the queue.
    """
    required = ("last.pt", "model_meta.json", "calibration_ready_summary.json", "train_history.json")
    if not all((member / name).is_file() for name in required):
        return False
    if expected_epochs is None:
        return True
    try:
        history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            return False
        epochs = [int(row["epoch"]) for row in history if isinstance(row, Mapping) and "epoch" in row]
        return bool(epochs) and max(epochs) >= int(expected_epochs)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _completed_member_dir(
    output_root: Path,
    logical_member: str,
    *,
    expected_epochs: int | None = None,
) -> Path | None:
    candidates = [output_root / logical_member]
    candidates.extend(sorted(output_root.glob(f"{logical_member}_attempt_*")))
    completed = [
        path for path in candidates
        if _training_artifacts_complete(path, expected_epochs=expected_epochs)
    ]
    return completed[-1] if completed else None


def _completed_member_matches_job(member: Path, job: Mapping[str, Any]) -> bool:
    """Verify the cheap identity fields before reusing an orphaned member.

    A complete-looking directory is not sufficient after a parent-process
    interruption: the directory may belong to another variant or seed.  The
    full checkpoint validator runs later; this check only prevents accidental
    cross-member reuse during queue recovery.
    """
    try:
        payload = _load_json(member / "model_meta.json")
        protocol = payload.get("protocol")
        if not isinstance(protocol, Mapping):
            return False
        if str(protocol.get("variant", "")) != str(job.get("variant", "")):
            return False
        if int(protocol.get("seed", -1)) != int(job.get("seed", -2)):
            return False
        experiment = str(protocol.get("experiment", ""))
        return str(job.get("member", "")) in experiment
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _next_member_attempt(output_root: Path, logical_member: str) -> Path:
    base = output_root / logical_member
    if not base.exists() or not any(base.iterdir()):
        return base
    index = 2
    while True:
        candidate = output_root / f"{logical_member}_attempt_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _run_one(
    args: argparse.Namespace,
    job: Mapping[str, Any],
    output_root: Path,
    log_root: Path,
    mode_config: Mapping[str, Any],
    gpu: int,
) -> dict[str, Any]:
    logical_member = str(job["member"])
    member = _next_member_attempt(output_root, logical_member)
    member.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{member.name}.stdout.log"
    stderr_path = log_root / f"{member.name}.stderr.log"
    command = _train_command(args, job, member, mode_config)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    result = {
        "variant": str(job["variant"]),
        "seed": int(job["seed"]),
        "configId": str(job.get("configId", "registered")),
        "member": logical_member,
        "gpu": int(gpu),
        "outputDir": str(member),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "testRead": False,
    }
    if completed.returncode == 0 and not _training_artifacts_complete(
        member, expected_epochs=int(mode_config["epochs"])
    ):
        result["returnCode"] = 86
        result["artifactError"] = "training exited successfully but required artifacts are incomplete"
    return result


def _dynamic_queue(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
    output_root: Path,
    log_root: Path,
    mode_config: Mapping[str, Any],
    gpu_ids: Sequence[int],
    run_one: Callable[..., dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run a persistent one-process-per-GPU completion queue."""
    if not gpu_ids or len(set(int(value) for value in gpu_ids)) != len(gpu_ids):
        raise ValueError("gpu ids must be a non-empty unique sequence")
    worker = _run_one if run_one is None else run_one
    results: list[dict[str, Any]] = []
    pending = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        active: dict[Future[dict[str, Any]], int] = {}

        def submit_next(gpu: int) -> None:
            try:
                job = next(pending)
            except StopIteration:
                return
            active[executor.submit(worker, args, job, output_root, log_root, mode_config, int(gpu))] = int(gpu)

        for gpu in gpu_ids:
            submit_next(int(gpu))
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                gpu = active.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # preserve the queue record for postmortem
                    result = {"gpu": gpu, "returnCode": 1, "error": repr(exc), "testRead": False}
                results.append(result)
                submit_next(gpu)
    results.sort(key=lambda item: (str(item.get("member", "")), int(item.get("seed", -1))))
    return results


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def _selected_checkpoint(member: Path) -> Path:
    """Select only the checkpoint declared by the member calibration contract."""
    calibration_path = member / "calibration_ready_summary.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"member calibration summary is missing: {calibration_path}")
    calibration = _load_json(calibration_path)
    if calibration.get("schema") != CALIBRATION_SUMMARY_SCHEMA_V4 or calibration.get("testRead") is not False:
        raise ValueError(f"member calibration summary is not v4/test-free: {calibration_path}")

    status = str(calibration.get("status", ""))
    if status == "safe":
        declared = calibration.get("safeCheckpoint")
        if not declared or Path(str(declared)).name != "best_safe.pt":
            raise ValueError(f"safe calibration summary does not declare best_safe.pt: {calibration_path}")
        checkpoint = member / "best_safe.pt"
        alias = member / "best.pt"
        if not checkpoint.is_file() or not alias.is_file():
            raise FileNotFoundError(f"safe member must contain best_safe.pt and best.pt: {member}")
        return checkpoint

    if status == "no_qualified_safety_workpoint":
        declared = calibration.get("diagnosticCheckpoint")
        if not declared or Path(str(declared)).name != "best_diagnostic.pt":
            raise ValueError(
                f"diagnostic calibration summary does not declare best_diagnostic.pt: {calibration_path}"
            )
        checkpoint = member / "best_diagnostic.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"diagnostic member has no best_diagnostic.pt: {member}")
        return checkpoint

    raise ValueError(f"unsupported calibration status {status!r}: {calibration_path}")


def _require_formal_evaluation_clis(args: argparse.Namespace) -> None:
    missing = [
        option
        for option, value in (("--image-cli", args.image_cli), ("--resource-cli", args.resource_cli))
        if not str(value).strip()
    ]
    if missing:
        raise RuntimeError(
            "formal evaluation requires image metric backfill and frozen-threshold resource evaluation; "
            f"missing {', '.join(missing)}"
        )


def _validate_v4_checkpoint(
    checkpoint_path: Path,
    member: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("schema") != "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4":
        raise ValueError(f"checkpoint schema is not v4: {checkpoint_path}")
    if checkpoint.get("runtimeSchema") != "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4":
        raise ValueError(f"checkpoint runtime schema is not v4: {checkpoint_path}")
    if checkpoint.get("testRead") is not False:
        raise ValueError(f"checkpoint is not test-free: {checkpoint_path}")
    protocol = checkpoint.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError(f"checkpoint has no protocol: {checkpoint_path}")
    if protocol.get("testRead") is not False or protocol.get("candidateUnion") is not False:
        raise ValueError(f"checkpoint protocol is not test-free/native-candidate: {checkpoint_path}")
    if protocol.get("schema") != "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4":
        raise ValueError(f"checkpoint protocol schema is not v4: {checkpoint_path}")
    expected_seed = int(member["seed"])
    if int(protocol.get("seed", -1)) != expected_seed:
        raise ValueError(f"checkpoint seed mismatch: {checkpoint_path}")
    expected_variant = str(member["variant"])
    if str(protocol.get("variant", "")) != expected_variant:
        raise ValueError(f"checkpoint variant mismatch: {checkpoint_path}")
    spec = member["variantSpec"]
    if protocol.get("lossVariant") != spec["lossVariant"]:
        raise ValueError(f"checkpoint loss variant disagrees with job: {checkpoint_path}")
    protocol_calibration = protocol.get("instanceCalibration")
    if (
        not isinstance(protocol_calibration, Mapping)
        or protocol_calibration.get("mode") != spec["instanceCalibrationMode"]
    ):
        raise ValueError(f"checkpoint calibration protocol disagrees with job: {checkpoint_path}")
    viewcell = protocol.get("viewcell")
    if (
        not isinstance(viewcell, Mapping)
        or viewcell.get("shape") != "horizontal_disk"
        or not np.isfinite(float(viewcell.get("radiusM", float("nan"))))
        or float(viewcell.get("radiusM", 0.0)) <= 0.0
        or viewcell.get("candidateCameraSemantics")
        != "66-degree back-camera candidate identity only"
        or viewcell.get("queryCenterSemantics")
        != "center of the same-direction view-cell visibility union"
    ):
        raise ValueError(f"checkpoint view-cell protocol is invalid: {checkpoint_path}")
    preflight = manifest["preflight"]
    if protocol.get("splitPoseCounts") != preflight.get("splitPoseCounts"):
        raise ValueError(f"checkpoint split pose counts disagree with preflight: {checkpoint_path}")
    relation = checkpoint.get("relation")
    if not isinstance(relation, Mapping):
        raise ValueError(f"checkpoint relation provenance is missing: {checkpoint_path}")
    role = str(member["variantSpec"]["relationArtifact"])
    expected_relation = preflight["relationArtifacts"].get(role)
    if not isinstance(expected_relation, Mapping):
        raise ValueError(f"checkpoint relation role is absent from preflight: {role}")
    if relation.get("schema") != expected_relation.get("schema"):
        raise ValueError(f"checkpoint relation schema disagrees with preflight: {checkpoint_path}")
    for field in ("edgeCount", "rowCount"):
        if int(relation.get(field, -1)) != int(expected_relation.get(field, -2)):
            raise ValueError(f"checkpoint relation {field} disagrees with preflight: {checkpoint_path}")
    observations = relation.get("observations")
    expected_observations = expected_relation.get("observationCount")
    if not isinstance(observations, Mapping) or int(observations.get("observationCount", -1)) != int(expected_observations):
        raise ValueError(f"checkpoint relation observations disagree with preflight: {checkpoint_path}")
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint model config is missing: {checkpoint_path}")
    if int(config.get("numInstances", -1)) != int(preflight["runtimeMeta"]["numInstances"]):
        raise ValueError(f"checkpoint instance count disagrees with runtime metadata: {checkpoint_path}")
    instance_calibration = config.get("instanceCalibration")
    if (
        config.get("relationSource") != spec["relationSource"]
        or config.get("spectralMode") != spec["spectralMode"]
        or not isinstance(instance_calibration, Mapping)
        or instance_calibration.get("mode") != spec["instanceCalibrationMode"]
    ):
        raise ValueError(f"checkpoint model variant config disagrees with job: {checkpoint_path}")
    checkpoint_calibration = checkpoint.get("instanceCalibration")
    if (
        not isinstance(checkpoint_calibration, Mapping)
        or checkpoint_calibration.get("mode") != spec["instanceCalibrationMode"]
        or checkpoint_calibration.get("fusion") != "prior_plus_applied_residual"
        or checkpoint_calibration.get("runtimeExport") != "fused_coefficients_only"
        or not np.isfinite(float(checkpoint_calibration.get("blend", float("nan"))))
    ):
        raise ValueError(f"checkpoint instance calibration state is invalid: {checkpoint_path}")
    blend = float(checkpoint_calibration["blend"])
    if not 0.0 <= blend <= 1.0:
        raise ValueError(f"checkpoint instance calibration blend is invalid: {checkpoint_path}")
    model_state = checkpoint.get("modelState")
    if not isinstance(model_state, Mapping):
        raise ValueError(f"checkpoint model state is missing: {checkpoint_path}")
    residual_parameter_name = "instance_calibration_residual_raw"
    residual = checkpoint.get("instanceSurvivalCalibrationResidual")
    prior = checkpoint.get("instanceSurvivalPriorCoefficients")
    coefficients = checkpoint.get("instanceSurvivalCoefficients")
    if residual is None or prior is None or coefficients is None:
        raise ValueError(f"checkpoint survival coefficient fusion is incomplete: {checkpoint_path}")
    residual_np = np.asarray(
        residual.detach().cpu() if hasattr(residual, "detach") else residual,
        dtype=np.float32,
    )
    prior_np = np.asarray(
        prior.detach().cpu() if hasattr(prior, "detach") else prior,
        dtype=np.float32,
    )
    coefficients_np = np.asarray(
        coefficients.detach().cpu() if hasattr(coefficients, "detach") else coefficients,
        dtype=np.float32,
    )
    if (
        residual_np.shape != prior_np.shape
        or residual_np.shape != coefficients_np.shape
        or not np.isfinite(residual_np).all()
        or not np.isfinite(prior_np).all()
        or not np.isfinite(coefficients_np).all()
        or float(np.max(np.abs(coefficients_np - prior_np - residual_np), initial=0.0)) > MAX_FP16_FUSION_ABS_ERROR
    ):
        raise ValueError(f"checkpoint survival coefficient fusion is invalid: {checkpoint_path}")
    if spec["instanceCalibrationMode"] == "disabled":
        if (
            abs(blend) > 1e-7
            or float(np.max(np.abs(residual_np), initial=0.0)) > 1e-7
            or float(np.max(np.abs(coefficients_np - prior_np), initial=0.0)) > MAX_FP16_FUSION_ABS_ERROR
            or residual_parameter_name in model_state
        ):
            raise ValueError(f"disabled calibration checkpoint contains residual state: {checkpoint_path}")
    elif residual_parameter_name not in model_state:
        raise ValueError(f"residual calibration checkpoint has no residual parameter: {checkpoint_path}")
    calibration_path = checkpoint_path.parent / "calibration_ready_summary.json"
    if not calibration_path.is_file():
        raise FileNotFoundError(f"checkpoint calibration summary is missing: {calibration_path}")
    calibration = _load_json(calibration_path)
    if calibration.get("testRead") is not False or calibration.get("schema") != CALIBRATION_SUMMARY_SCHEMA_V4:
        raise ValueError(f"checkpoint calibration summary is not v4/test-free: {calibration_path}")
    calibration_best_key = "bestSafe" if calibration.get("status") == "safe" else "bestDiagnostic"
    calibration_best = calibration.get(calibration_best_key)
    calibration_selection = (
        calibration_best.get("selection", calibration_best)
        if isinstance(calibration_best, Mapping)
        else None
    )
    expected_counts = preflight.get("splitPoseCounts", {})
    if (
        not isinstance(calibration_selection, Mapping)
        or int(calibration_selection.get("eval_pose_count", -1))
        != int(expected_counts.get("calibration", -2))
    ):
        raise ValueError(f"checkpoint calibration selection does not cover the full calibration split: {checkpoint_path}")
    validation_at_frozen = calibration.get("validationAtFrozenThreshold")
    if (
        not isinstance(validation_at_frozen, Mapping)
        or int(validation_at_frozen.get("eval_pose_count", -1))
        != int(expected_counts.get("validation", -2))
    ):
        raise ValueError(f"checkpoint validation summary does not cover the full validation split: {checkpoint_path}")
    best = checkpoint.get("best")
    if not isinstance(best, Mapping):
        raise ValueError(f"checkpoint has no frozen calibration selection: {checkpoint_path}")
    selection = best.get("selection", best)
    if not isinstance(selection, Mapping) or not np.isfinite(float(selection.get("threshold", float("nan")))):
        raise ValueError(f"checkpoint has no frozen threshold: {checkpoint_path}")
    summary_key = "bestSafe" if calibration.get("status") == "safe" else "bestDiagnostic"
    summary_best = calibration.get(summary_key)
    if not isinstance(summary_best, Mapping):
        raise ValueError(f"checkpoint calibration summary has no {summary_key}: {calibration_path}")
    summary_selection = summary_best.get("selection", summary_best)
    if not isinstance(summary_selection, Mapping) or not np.isfinite(float(summary_selection.get("threshold", float("nan")))):
        raise ValueError(f"checkpoint calibration summary has no frozen threshold: {calibration_path}")
    if not np.isclose(float(selection["threshold"]), float(summary_selection["threshold"]), rtol=0.0, atol=1e-7):
        raise ValueError(f"checkpoint threshold disagrees with calibration_ready_summary: {checkpoint_path}")
    geometry = checkpoint.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"checkpoint geometry provenance is missing: {checkpoint_path}")
    if list(geometry.get("shape", [])) != [int(config.get("numInstances", -1)), 96] or geometry.get("dtype") != "float16":
        raise ValueError(f"checkpoint geometry schema is invalid: {checkpoint_path}")
    model_meta_path = checkpoint_path.parent / "model_meta.json"
    if not model_meta_path.is_file():
        raise FileNotFoundError(f"model_meta.json is missing: {model_meta_path}")
    model_meta = _load_json(model_meta_path)
    if model_meta.get("schema") != "pvs-bounded-relation-prior-instance-calibrated-moment-envelope-v4" or model_meta.get("testRead") is not False:
        raise ValueError(f"model meta is not v4/test-free: {model_meta_path}")
    if model_meta.get("modelConfig") != dict(config):
        raise ValueError(f"model meta config disagrees with checkpoint: {model_meta_path}")
    if model_meta.get("protocol") != protocol:
        raise ValueError(f"model meta protocol disagrees with checkpoint: {model_meta_path}")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "modelMeta": str(model_meta_path.resolve()),
        "geometry": {
            "shape": list(geometry.get("shape", [])),
            "dtype": str(geometry.get("dtype")),
        },
        "relation": {
            "role": role,
            "schema": str(relation.get("schema")),
            "edgeCount": int(relation.get("edgeCount", -1)),
            "rowCount": int(relation.get("rowCount", -1)),
            "observationCount": int(observations.get("observationCount", -1)),
        },
        "threshold": float(selection["threshold"]),
        "safeWorkpoint": bool(calibration.get("status") == "safe" and best.get("safe") is True),
        "epoch": int(checkpoint.get("epoch", -1)),
        "testRead": False,
    }


def _command_result(
    command: Sequence[str],
    *,
    stage: str,
    label: str,
    log_root: Path,
    gpu: int | None = None,
) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(character if character.isalnum() or character in "._-" else "_" for character in label)
    stdout_path = log_root / f"{safe_label}.stdout.log"
    stderr_path = log_root / f"{safe_label}.stderr.log"
    environment = dict(os.environ)
    if gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(list(command), cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
        return {"stage": stage, "label": label, "command": list(command), "gpu": gpu, "returnCode": int(completed.returncode), "stdout": str(stdout_path), "stderr": str(stderr_path), "elapsedSeconds": time.time() - started, "testRead": False}
    except Exception as exc:
        return {"stage": stage, "label": label, "command": list(command), "gpu": gpu, "returnCode": 1, "error": repr(exc), "stdout": str(stdout_path), "stderr": str(stderr_path), "elapsedSeconds": time.time() - started, "testRead": False}


def _validation_command(
    args: argparse.Namespace,
    checkpoint: Path,
    output: Path,
    relation_dir: Path | None = None,
    seed: int | None = None,
) -> list[str]:
    command = [
        sys.executable, str(Path(args.evaluator_cli).resolve()),
        "--checkpoint", str(checkpoint),
        "--dataset-dir", str(Path(args.dataset_dir).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--initial-geo-features", str(Path(args.initial_geo_features).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--output", str(output), "--split", "validation", "--poses-per-batch", str(int(args.poses_per_batch)), "--device", "cuda",
        "--allow-unsafe-diagnostic", "--persist-ids",
    ]
    if relation_dir is not None:
        command.extend(["--relation-dir", str(relation_dir.resolve())])
    if seed is not None:
        command.extend(["--seed", str(int(seed))])
    return command


def _validator_command(
    args: argparse.Namespace,
    checkpoint: Path,
    *,
    expected_variant: str | None = None,
    expected_seed: int | None = None,
) -> list[str] | None:
    if not args.validator_cli:
        return None
    command = [
        sys.executable,
        str(Path(args.validator_cli).resolve()),
        "--checkpoint", str(checkpoint),
        "--model-meta", str(checkpoint.parent / "model_meta.json"),
        "--calibration", str(checkpoint.parent / "calibration_ready_summary.json"),
    ]
    if expected_variant is not None:
        command.extend(["--expected-variant", str(expected_variant)])
    if expected_seed is not None:
        command.extend(["--expected-seed", str(int(expected_seed))])
    return command


def _summarizer_command(
    args: argparse.Namespace,
    validation_root: Path,
    output: Path,
    image_root: Path | None = None,
    resource_root: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(args.summarizer_cli).resolve()),
        "--root", str(validation_root),
        "--output", str(output),
        "--bootstrap-replicates", "10000",
        "--lcb-replicates", "10000",
    ]
    if image_root is not None:
        command.extend(["--image-root", str(image_root)])
    if resource_root is not None:
        command.extend(["--resource-root", str(resource_root)])
    return command


def _export_command(
    args: argparse.Namespace,
    member: Mapping[str, Any],
    output: Path,
) -> list[str] | None:
    if not args.exporter_cli:
        return None
    checkpoint = Path(str(member["checkpointInfo"]["checkpoint"]))
    command = [
        sys.executable,
        str(Path(args.exporter_cli).resolve()),
        "--checkpoint", str(checkpoint),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--output-dir", str(output),
    ]
    if not bool(member["checkpointInfo"].get("safeWorkpoint", False)):
        command.append("--allow-unsafe-threshold")
    return command


def _validate_runtime_bundle(member: Mapping[str, Any], output: Path) -> dict[str, Any]:
    meta_path = output / "model_meta.json"
    table_path = output / "instance_runtime_features_fp16.bin"
    if not meta_path.is_file() or not table_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint-specific runtime bundle: {output}")
    meta = _load_json(meta_path)
    checkpoint_info = member["checkpointInfo"]
    if meta.get("schema") != "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4":
        raise ValueError(f"unexpected runtime bundle schema: {meta_path}")
    if meta.get("testRead") is not False:
        raise ValueError(f"runtime bundle is not test-free: {meta_path}")
    fixed_table = meta.get("fixedTable")
    if not isinstance(fixed_table, Mapping):
        raise ValueError(f"runtime bundle has no fixed table descriptor: {meta_path}")
    if fixed_table.get("file") != table_path.name:
        raise ValueError(f"runtime bundle fixed-table file is invalid: {table_path}")
    if fixed_table.get("dtype") != "float16":
        raise ValueError(f"runtime bundle fixed-table dtype is invalid: {meta_path}")
    declared_shape = fixed_table.get("shape")
    expected_instances = int(member["checkpointInfo"].get("geometry", {}).get("shape", [0])[0])
    if (
        not isinstance(declared_shape, list)
        or len(declared_shape) != 2
        or int(declared_shape[0]) != expected_instances
        or int(declared_shape[1]) != 124
    ):
        raise ValueError(f"runtime bundle fixed-table shape is invalid: {meta_path}")
    declared_bytes = fixed_table.get("byteLength")
    if declared_bytes is not None and int(declared_bytes) != int(table_path.stat().st_size):
        raise ValueError(f"runtime bundle fixed-table byte length is invalid: {table_path}")
    source = meta.get("runtimeFeatureSource")
    geometry = source.get("geometry") if isinstance(source, Mapping) else None
    if not isinstance(source, Mapping) or source.get("source") != "checkpoint.geometryFeatures_plus_survivalCoefficients":
        raise ValueError(f"runtime bundle did not rebuild checkpoint-owned coefficients: {meta_path}")
    if not isinstance(geometry, Mapping):
        raise ValueError(f"runtime bundle geometry source is missing: {meta_path}")
    if geometry.get("shape") != [expected_instances, 96] or geometry.get("dtype") != "float16":
        raise ValueError(f"runtime bundle geometry schema is invalid: {meta_path}")
    threshold = float(meta.get("threshold", float("nan")))
    if not np.isclose(threshold, float(checkpoint_info["threshold"]), rtol=0.0, atol=1e-7):
        raise ValueError(f"runtime bundle threshold disagrees with checkpoint calibration: {meta_path}")
    return {
        "path": str(output.resolve()),
        "modelMeta": str(meta_path.resolve()),
        "runtimeFeatures": str(table_path.resolve()),
        "runtimeFeatureByteLength": int(table_path.stat().st_size),
        "threshold": threshold,
        "safeWorkpoint": bool(checkpoint_info.get("safeWorkpoint", False)),
        "testRead": False,
    }


def _image_command(args: argparse.Namespace, member: Mapping[str, Any], output: Path) -> list[str] | None:
    if not args.image_cli:
        return None
    checkpoint = Path(str(member["checkpointInfo"]["checkpoint"]))
    runtime = Path(str(member["runtimeBundle"]["runtimeFeatures"]))
    calibration = Path(str(member["checkpointInfo"]["calibration"]))
    name = f"{member['variant']}_seed{int(member['seed'])}"
    spec = f"{name}|bounded_relation_survival_moment_v4|{checkpoint}|{runtime}|{calibration}"
    return [
        sys.executable,
        str(Path(args.image_cli).resolve()),
        "--pose-csr", str(Path(args.dataset_dir).resolve()),
        "--viewcell-dataset", str(Path(args.viewcell_dataset).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--split", "validation",
        "--model-spec", spec,
        "--threshold", str(float(member["checkpointInfo"]["threshold"])),
        "--output-dir", str(output),
        "--formal-image-evaluation",
        "--device", "cuda",
        "--require-hardware-gpu",
    ]


def _resource_command(
    args: argparse.Namespace,
    members: Sequence[Mapping[str, Any]],
    output: Path,
    threshold_manifest: Path,
) -> list[str] | None:
    if not args.resource_cli:
        return None
    specs = []
    for member in members:
        checkpoint = Path(str(member["checkpointInfo"]["checkpoint"]))
        runtime = Path(str(member["runtimeBundle"]["runtimeFeatures"]))
        calibration = Path(str(member["checkpointInfo"]["calibration"]))
        name = f"{member['variant']}_seed{int(member['seed'])}"
        specs.append(
            f"{name}|bounded_relation_survival_moment_v4|{checkpoint}|{runtime}|{calibration}"
        )
    return [
        sys.executable,
        str(Path(args.resource_cli).resolve()),
        "--models", "",
        "--dataset-dir", str(Path(args.dataset_dir).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--split", "validation",
        "--frozen-threshold-file", str(threshold_manifest),
        "--learned-model-spec", specs[0],
        *sum((["--learned-model-spec", value] for value in specs[1:]), []),
        "--output-dir", str(output),
        "--device", "cuda",
    ]


def _route_command(args: argparse.Namespace, summary: Path, output: Path, report: Path) -> list[str] | None:
    if not args.route_cli:
        return None
    return [sys.executable, str(Path(args.route_cli).resolve()), "--summary", str(summary), "--output", str(output), "--markdown", str(report)]


def _write_formal_validation_report(
    summary_path: Path,
    route_path: Path,
    route_report_path: Path,
    evaluation_root: Path,
) -> Path:
    """Publish a compact, linked formal report without changing model assets."""
    summary = _load_json(summary_path)
    route = _load_json(route_path)
    report_dir = ROOT / "docs" / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    if evaluation_root.resolve() == DEFAULT_FORMAL_VALIDATION.resolve():
        attempt_label = "initial"
    else:
        attempt_label = evaluation_root.name
    base = report_dir / f"{PREFIX}_formal_validation_{attempt_label}_2026-08-15.md"
    report_path = base
    if report_path.exists() and str(summary_path.resolve()) not in report_path.read_text(encoding="utf-8"):
        index = 2
        while True:
            candidate = report_dir / f"{PREFIX}_formal_validation_{attempt_label}_attempt_{index:03d}_2026-08-15.md"
            if not candidate.exists():
                report_path = candidate
                break
            index += 1

    def fmt(value: Any, digits: int = 6) -> str:
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "-"

    lines = [
        "# v4 有界关系生存矩包络正式验证报告",
        "",
        "本报告对应 v4 完整组合、四个核心消融、三个随机种子和 validation 全量评价。阈值只来自各成员 calibration；test 未用于选择。",
        "",
        f"- validation 汇总：`{summary_path.resolve()}`",
        f"- 路线判定：`{route_path.resolve()}`",
        f"- 路线报告：`{route_report_path.resolve()}`",
        f"- `testRead`：`{summary.get('testRead')}`",
        f"- bootstrap：`{(summary.get('bootstrap') or {}).get('replicates', '-')}` 次，按 seed 聚类并在 pose 内配对重采样",
        "",
        f"## 路线结论：`{route.get('route', '-')}`",
        "",
        str(route.get("routeReason", "")),
        "",
        "## 成员验证汇总",
        "",
        "| 变体 | seed | weighted recall | weighted recall 下界 | precision | balanced accuracy | useful cull | bad cull | 平均预测数 | 安全工作点 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    members = summary.get("members")
    if isinstance(members, Mapping):
        for key in sorted(members):
            record = members[key]
            if not isinstance(record, Mapping):
                continue
            aggregate = record.get("aggregate") if isinstance(record.get("aggregate"), Mapping) else {}
            parts = str(key).split(":", 1)
            variant = parts[0]
            seed = parts[1] if len(parts) == 2 else "-"
            lcb = aggregate.get("aggregateWeightedRecallLowerConfidenceBound", aggregate.get("weightedRecallLowerConfidenceBound"))
            lines.append(
                f"| `{variant}` | {seed} | {fmt(aggregate.get('aggregateWeightedRecall', aggregate.get('weightedRecall')))} | "
                f"{fmt(lcb)} | {fmt(aggregate.get('precision'))} | {fmt(aggregate.get('balancedAccuracy'))} | "
                f"{fmt(aggregate.get('usefulCull'))} | {fmt(aggregate.get('badCull'))} | {fmt(aggregate.get('avgPredCount'))} | "
                f"{'通过' if record.get('safeWorkpoint') or record.get('calibrationSafeWorkpoint') else '未通过'} |"
            )
    lines.extend([
        "",
        "## 指标口径",
        "",
        "`weighted recall` 衡量重要可见实例权重的找回率；`useful cull` 是正确剔除不可见候选的比例；`bad cull` 是错误漏掉真实可见实例的比例。useful cull 不单独作为模型优劣结论，同时参考 recall、precision、accuracy、balanced accuracy、资源字节和图像指标。",
        "",
        "## 消融判定",
        "",
        "路线 JSON 保存四组 full-minus-ablation 的原始差值、95% bootstrap 置信区间和安全/分类/资源/图像分层判定。图像评价使用实例级 Color-ID 浏览器路径；资源评价使用冻结阈值和实际 GLB 字节。",
        "",
        "## 训练与运行资产",
        "",
        f"- formal80 训练目录：`{(ROOT / 'neural_instance_culling/model/out' / (PREFIX + '_formal80')).resolve()}`",
        f"- validation 评价目录：`{evaluation_root.resolve()}`",
        "- 默认模型、默认阈值和前端部署资产：本实验未修改。",
        "",
        "## 路线原文",
        "",
    ])
    if route_report_path.is_file():
        lines.extend(route_report_path.read_text(encoding="utf-8").splitlines())
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _queue_commands(
    commands: Sequence[tuple[str, Sequence[str]]],
    *,
    stage: str,
    log_root: Path,
    gpu_ids: Sequence[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending = iter(commands)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        active: dict[Future[dict[str, Any]], int] = {}

        def submit_next(gpu: int) -> None:
            try:
                label, command = next(pending)
            except StopIteration:
                return
            active[executor.submit(_command_result, command, stage=stage, label=label, log_root=log_root, gpu=gpu)] = gpu

        for gpu in gpu_ids:
            submit_next(int(gpu))
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                gpu = active.pop(future)
                result = future.result()
                result["gpu"] = gpu
                results.append(result)
                submit_next(gpu)
    results.sort(key=lambda value: str(value.get("label", "")))
    return results


def _validate_replay_identity(
    args: argparse.Namespace,
    run_manifest: Mapping[str, Any],
    checkpoint_infos: Sequence[Mapping[str, Any]],
    attempt_root: Path,
) -> dict[str, Any]:
    """Recheck validation identity from explicit IDs without opening ``test``."""
    runtime_path = _required_file(Path(args.runtime_meta), "runtime metadata")
    world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(runtime_path)
    dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), num_instances=int(world_aabbs.shape[0]))
    validation_name = "validation" if "validation" in dataset.split_ids else _split_name(dataset, "val")
    validation = dataset.split(validation_name)
    pose_indices = np.asarray(validation.pose_indices, dtype="<i8")
    expected = run_manifest["preflight"]
    expected_pose_indices = pose_indices.tolist()
    registered_pose_indices = expected.get("splitPoseIndices", {}).get("validation")
    if registered_pose_indices is not None and [int(value) for value in registered_pose_indices] != expected_pose_indices:
        raise ValueError("validation pose order changed after preflight")
    native_summary = validate_native_candidate_contract(
        dataset,
        pose_indices,
        num_instances=int(world_aabbs.shape[0]),
    )
    registered_summary = expected.get("splitSemantics", {}).get("validation")
    if isinstance(registered_summary, Mapping):
        for field in ("poseCount", "candidateReferenceCount", "visibleReferenceCount"):
            if int(registered_summary.get(field, -1)) != int(native_summary[field]):
                raise ValueError(f"validation {field} changed after preflight")
    checked: list[str] = []
    member_contracts: dict[str, Any] = {}
    for info in checkpoint_infos:
        path = attempt_root / f"{info['variant']}_seed{int(info['seed'])}_e80" / "validation_evaluation.json"
        payload = _load_json(path)
        contract = validate_replay_payload(
            payload,
            dataset,
            expected_pose_indices,
            expected_variant=str(info["variant"]),
            expected_seed=int(info["seed"]),
            expected_checkpoint=str(info["checkpointInfo"]["checkpoint"]),
            expected_threshold=float(info["checkpointInfo"]["threshold"]),
            num_instances=int(world_aabbs.shape[0]),
        )
        member_contracts[f"{info['variant']}:seed{int(info['seed'])}"] = contract
        checked.append(str(path))
    return {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-replay-semantic-contract-v1",
        "validationName": validation_name,
        "poseCount": int(pose_indices.size),
        "poseIndices": expected_pose_indices,
        "candidateReferenceCount": int(native_summary["candidateReferenceCount"]),
        "visibleReferenceCount": int(native_summary["visibleReferenceCount"]),
        "members": member_contracts,
        "checkedOutputs": checked,
        "testRead": False,
    }


def _formal_postprocess(
    args: argparse.Namespace,
    run_manifest: Mapping[str, Any],
    training_results: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    gpu_ids: Sequence[int],
) -> dict[str, Any]:
    """Run formal post-processing without touching training members."""
    benchmark_root = Path(args.benchmark_output_root).resolve()
    output_root = Path(str(run_manifest["outputRoot"])).resolve()
    previous_pipeline = _latest_record(output_root, "pipeline_status")
    previous_payload: Mapping[str, Any] | None = None
    attempt_root: Path
    if previous_pipeline is not None:
        candidate = Path(str(previous_pipeline[1].get("evaluationRoot", ""))).resolve()
        previous_status = str(previous_pipeline[1].get("status", ""))
        if previous_status == "evaluation_complete" and candidate.is_dir():
            summary_path = candidate / "validation_summary.json"
            route_path = candidate / "route_decision.json"
            route_report_path = candidate / "route_report.md"
            if summary_path.is_file() and route_path.is_file() and route_report_path.is_file():
                formal_report = _write_formal_validation_report(
                    summary_path,
                    route_path,
                    route_report_path,
                    candidate,
                )
                return {
                    **dict(previous_pipeline[1]),
                    "formalReport": str(formal_report.resolve()),
                    "reusedEvaluation": True,
                    "testRead": False,
                }
            previous_pipeline = None
        elif previous_status == "training_complete_evaluation_incomplete" and candidate.is_dir():
            attempt_root = candidate
            previous_payload = previous_pipeline[1]
        else:
            previous_pipeline = None
    if previous_pipeline is None:
        if benchmark_root.exists() and any(benchmark_root.iterdir()):
            attempt_index = 1
            while (benchmark_root / f"attempt_{attempt_index:03d}").exists():
                attempt_index += 1
            attempt_root = benchmark_root / f"attempt_{attempt_index:03d}"
        else:
            attempt_root = benchmark_root
        _ensure_new_root(attempt_root)

    prior_stage_records = (
        previous_payload.get("stages", [])
        if isinstance(previous_payload, Mapping)
        else []
    )

    def stage_was_passed(stage: str, label: str) -> bool:
        return any(
            isinstance(record, Mapping)
            and record.get("stage") == stage
            and record.get("label") == label
            and record.get("status") == "passed"
            for record in prior_stage_records
        )

    def previous_stage_record(stage: str, label: str) -> Mapping[str, Any] | None:
        for record in reversed(prior_stage_records):
            if (
                isinstance(record, Mapping)
                and record.get("stage") == stage
                and record.get("label") == label
                and record.get("status") == "passed"
            ):
                return record
        return None

    def reused_stage_record(
        stage: str,
        label: str,
        *,
        artifact: Path | None = None,
    ) -> dict[str, Any]:
        previous = previous_stage_record(stage, label)
        record = dict(previous) if previous is not None else {
            "stage": stage,
            "label": label,
            "status": "passed",
            "testRead": False,
        }
        record["status"] = "passed"
        record["reused"] = True
        if artifact is not None:
            record["artifact"] = str(artifact.resolve())
        record["testRead"] = False
        return record

    def reusable_output(stage: str, label: str, path: Path) -> bool:
        return stage_was_passed(stage, label) and path.exists()

    log_root = attempt_root / "logs"
    members_by_key = {(str(row["variant"]), int(row["seed"])): row for row in training_results}
    replay_dataset: PoseCSRDataset | None = None
    replay_pose_indices: list[int] | None = None

    def reusable_replay(info: Mapping[str, Any], label: str, path: Path) -> bool:
        """Reuse a replay only after validating its explicit semantic payload."""
        nonlocal replay_dataset, replay_pose_indices
        if not stage_was_passed("calibration_frozen_validation_replay", label) or not path.is_file():
            return False
        try:
            if replay_dataset is None:
                num_instances = int(run_manifest["preflight"]["runtimeMeta"]["numInstances"])
                replay_dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), num_instances=num_instances)
                validation_name = "validation" if "validation" in replay_dataset.split_ids else _split_name(replay_dataset, "val")
                replay_pose_indices = [
                    int(value) for value in replay_dataset.split(validation_name).pose_indices.tolist()
                ]
            payload = _load_json(path)
            validate_replay_payload(
                payload,
                replay_dataset,
                replay_pose_indices or [],
                expected_variant=str(info["variant"]),
                expected_seed=int(info["seed"]),
                expected_checkpoint=str(info["checkpointInfo"]["checkpoint"]),
                expected_threshold=float(info["checkpointInfo"]["threshold"]),
                num_instances=int(run_manifest["preflight"]["runtimeMeta"]["numInstances"]),
            )
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    checkpoint_infos: list[dict[str, Any]] = []
    stage_records: list[dict[str, Any]] = []
    try:
        _require_formal_evaluation_clis(args)
        for job in jobs:
            key = (str(job["variant"]), int(job["seed"]))
            training = members_by_key.get(key)
            if training is None or int(training.get("returnCode", 1)) != 0:
                raise RuntimeError(f"formal postprocess has no successful training result for {key}")
            member_dir = Path(str(training["outputDir"]))
            checkpoint = _selected_checkpoint(member_dir)
            info = _validate_v4_checkpoint(checkpoint, {**job, "variantSpec": job["variantSpec"]}, run_manifest)
            checkpoint_infos.append({
                "variant": key[0],
                "seed": key[1],
                "relationRole": str(job["variantSpec"]["relationArtifact"]),
                "relationDir": str(_relation_dir_for_job(args, job)),
                "checkpointInfo": info,
                "testRead": False,
            })
            stage_records.append({
                "stage": "checkpoint_schema_validation",
                "label": f"local_{key[0]}_seed{key[1]}",
                "command": ["<in-process>", "_validate_v4_checkpoint", str(checkpoint)],
                "status": "passed",
                "testRead": False,
            })
            validator = _validator_command(
                args,
                checkpoint,
                expected_variant=key[0],
                expected_seed=key[1],
            )
            if validator is not None:
                validator_label = f"external_{key[0]}_seed{key[1]}"
                if stage_was_passed("checkpoint_schema_validation", validator_label):
                    stage_records.append(
                        reused_stage_record("checkpoint_schema_validation", validator_label)
                    )
                else:
                    stage_records.append({
                        "stage": "checkpoint_schema_validation",
                        "label": validator_label,
                        "command": validator,
                        "status": "planned",
                        "testRead": False,
                    })
        validation_commands = []
        for info in checkpoint_infos:
            checkpoint = Path(str(info["checkpointInfo"]["checkpoint"]))
            validator = _validator_command(
                args,
                checkpoint,
                expected_variant=str(info["variant"]),
                expected_seed=int(info["seed"]),
            )
            if validator is not None:
                validator_label = f"external_{info['variant']}_seed{int(info['seed'])}"
                if not stage_was_passed("checkpoint_schema_validation", validator_label):
                    validation_commands.append((validator_label, validator))
        if validation_commands:
            validation_results = _queue_commands(validation_commands, stage="checkpoint_schema_validation", log_root=log_root, gpu_ids=gpu_ids)
            stage_records.extend({**result, "status": "passed" if int(result["returnCode"]) == 0 else "failed"} for result in validation_results)
            if any(int(result["returnCode"]) != 0 for result in validation_results):
                raise RuntimeError("checkpoint/schema validator CLI failed")
        export_commands: list[tuple[str, list[str]]] = []
        runtime_bundle_root = attempt_root / "runtime_bundles"
        for info in checkpoint_infos:
            label = f"{info['variant']}_seed{int(info['seed'])}"
            output = runtime_bundle_root / label
            export_label = f"export_{label}"
            if reusable_output("checkpoint_runtime_bundle_export", export_label, output):
                stage_records.append(
                    reused_stage_record(
                        "checkpoint_runtime_bundle_export",
                        export_label,
                        artifact=output,
                    )
                )
                continue
            command = _export_command(args, info, output)
            if command is None:
                raise RuntimeError("checkpoint-specific runtime exporter is not configured")
            export_commands.append((export_label, command))
            stage_records.append({
                "stage": "checkpoint_runtime_bundle_export",
                "label": export_label,
                "command": command,
                "status": "planned",
                "testRead": False,
            })
        if export_commands:
            export_results = _queue_commands(
                export_commands,
                stage="checkpoint_runtime_bundle_export",
                log_root=log_root,
                gpu_ids=gpu_ids,
            )
            stage_records.extend(
                {**result, "status": "passed" if int(result["returnCode"]) == 0 else "failed"}
                for result in export_results
            )
            if any(int(result["returnCode"]) != 0 for result in export_results):
                raise RuntimeError("checkpoint-specific runtime bundle export failed")
        for info in checkpoint_infos:
            label = f"{info['variant']}_seed{int(info['seed'])}"
            bundle = _validate_runtime_bundle(info, runtime_bundle_root / label)
            info["runtimeBundle"] = bundle
            stage_records.append({
                "stage": "checkpoint_runtime_bundle_validation",
                "label": label,
                "command": ["<in-process>", "_validate_runtime_bundle", bundle["path"]],
                "status": "passed",
                "details": bundle,
                "testRead": False,
            })
        replay_commands: list[tuple[str, list[str]]] = []
        for info in checkpoint_infos:
            checkpoint = Path(str(info["checkpointInfo"]["checkpoint"]))
            member_root = attempt_root / f"{info['variant']}_seed{int(info['seed'])}_e80"
            member_root.mkdir(parents=True, exist_ok=True)
            output = member_root / "validation_evaluation.json"
            replay_label = f"replay_{info['variant']}_seed{int(info['seed'])}"
            if reusable_replay(info, replay_label, output):
                stage_records.append(
                    reused_stage_record(
                        "calibration_frozen_validation_replay",
                        replay_label,
                        artifact=output,
                    )
                )
                continue
            replay_commands.append((
                replay_label,
                _validation_command(
                    args,
                    checkpoint,
                    output,
                    relation_dir=Path(str(info.get("relationDir", ""))) if info.get("relationDir") else None,
                    seed=int(info["seed"]),
                ),
            ))
            stage_records.append({
                "stage": "calibration_frozen_validation_replay",
                "label": replay_label,
                "command": replay_commands[-1][1],
                "status": "planned",
                "testRead": False,
            })
        if replay_commands:
            replay_results = _queue_commands(
                replay_commands,
                stage="calibration_frozen_validation_replay",
                log_root=log_root,
                gpu_ids=gpu_ids,
            )
            stage_records.extend(
                {**result, "status": "passed" if int(result["returnCode"]) == 0 else "failed"}
                for result in replay_results
            )
            if any(int(result["returnCode"]) != 0 for result in replay_results):
                raise RuntimeError("calibration-frozen validation replay failed")
        replay_identity = _validate_replay_identity(args, run_manifest, checkpoint_infos, attempt_root)
        stage_records.append({
            "stage": "pose_candidate_gt_identity_validation",
            "label": "validation_identity",
            "command": ["<in-process>", "_validate_replay_identity"],
            "status": "passed",
            "details": replay_identity,
            "testRead": False,
        })
        validation_manifest = {
            "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-formal-validation-manifest-v1",
            "mode": "formal80_validation_replay",
            "epochs": 80,
            "variants": {str(variant): _variant_spec(str(variant)) for variant in FORMAL_VARIANTS},
            "seeds": list(FORMAL_SEEDS),
            "splitPoseCounts": dict(run_manifest["preflight"]["splitPoseCounts"]),
            "validationPoseIndices": replay_identity["poseIndices"],
            "splitSemantics": dict(run_manifest["preflight"].get("splitSemantics", {})),
            "candidateAudits": dict(run_manifest["preflight"].get("candidateAudits", {})),
            "relationArtifacts": dict(run_manifest["preflight"].get("relationArtifacts", {})),
            "inputProvenance": {
                "dataset": dict(run_manifest["preflight"]["dataset"]),
                "runtimeMeta": dict(run_manifest["preflight"]["runtimeMeta"]),
                "geometry": dict(run_manifest["preflight"]["geometry"]),
                "glb": dict(run_manifest["preflight"]["glb"]),
            },
            "trainingManifest": str((Path(run_manifest["outputRoot"]) / "run_manifest.json").resolve()),
            "members": checkpoint_infos,
            "replayOutputs": [str(attempt_root / f"{info['variant']}_seed{int(info['seed'])}_e80" / "validation_evaluation.json") for info in checkpoint_infos],
            "testRead": False,
        }
        _write_json(attempt_root / "matrix_manifest.json", validation_manifest)
        threshold_manifest_path = attempt_root / "frozen_validation_thresholds.json"
        threshold_names = {
            f"{info['variant']}_seed{int(info['seed'])}": float(info["checkpointInfo"]["threshold"])
            for info in checkpoint_infos
        }
        _write_json(
            threshold_manifest_path,
            {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-validation-thresholds-v1",
                "selection": "checkpoint-owned calibration thresholds; validation is one-shot replay only",
                "thresholds": threshold_names,
                "members": {
                    f"{info['variant']}_seed{int(info['seed'])}": {
                        "checkpoint": info["checkpointInfo"]["checkpoint"],
                        "runtimeFeatures": info["runtimeBundle"]["runtimeFeatures"],
                        "runtimeFeatureByteLength": info["runtimeBundle"].get("runtimeFeatureByteLength"),
                        "safeWorkpoint": bool(info["checkpointInfo"]["safeWorkpoint"]),
                    }
                    for info in checkpoint_infos
                },
                "split": "validation",
                "selectedFromValidation": False,
                "testRead": False,
            },
        )
        def valid_json_artifact(path: Path) -> Mapping[str, Any] | None:
            if not path.is_file():
                return None
            try:
                payload = _load_json(path)
            except Exception:
                return None
            return payload if isinstance(payload, Mapping) else None

        def image_output_ready(path: Path) -> bool:
            payload = valid_json_artifact(path / "summary.json")
            if payload is None or payload.get("formalImageEvaluationReady") is not True:
                return False
            renderer = payload.get("renderer")
            gate = renderer.get("gpuGate") if isinstance(renderer, Mapping) else None
            return isinstance(gate, Mapping) and gate.get("hardware") is True

        def resource_output_ready(path: Path) -> bool:
            payload = valid_json_artifact(path / "summary.json")
            if payload is None or not isinstance(payload.get("summaries"), list):
                return False
            meta = payload.get("meta")
            return isinstance(meta, Mapping) and meta.get("requestedSplit") == "validation"

        def validation_summary_ready(path: Path) -> bool:
            payload = valid_json_artifact(path)
            if payload is None:
                return False
            bootstrap = payload.get("bootstrap")
            try:
                bootstrap_replicates = int(bootstrap.get("replicates", 0)) if isinstance(bootstrap, Mapping) else 0
            except (TypeError, ValueError):
                return False
            return bool(
                payload.get("schema") == "pvs-bounded-relation-prior-instance-calibrated-v4-validation-summary-v1"
                and payload.get("split") == "validation"
                and payload.get("testRead") is False
                and isinstance(payload.get("members"), Mapping)
                and len(payload["members"]) == len(checkpoint_infos)
                and isinstance(bootstrap, Mapping)
                and bootstrap_replicates >= 10000
                and bootstrap.get("paired") is True
            )

        def route_output_ready(path: Path, report: Path) -> bool:
            payload = valid_json_artifact(path)
            return bool(
                payload is not None
                and payload.get("schema") == "pvs-bounded-relation-prior-instance-calibrated-v4-route-decision-v1"
                and payload.get("testRead") is False
                and report.is_file()
                and report.stat().st_size > 0
            )

        image_commands: list[tuple[str, list[str], Path]] = []
        image_root = attempt_root / "image"
        for info in checkpoint_infos:
            label = f"{info['variant']}_seed{int(info['seed'])}"
            output = image_root / label
            image_label = label
            info["imageOutput"] = str(output.resolve())
            if (
                stage_was_passed("formal_hardware_image_evaluation", image_label)
                and image_output_ready(output)
            ):
                stage_records.append(
                    reused_stage_record(
                        "formal_hardware_image_evaluation",
                        image_label,
                        artifact=output,
                    )
                )
                continue
            command = _image_command(args, info, output)
            if command is None:
                raise RuntimeError(f"formal image evaluator is not configured for {label}")
            image_commands.append((label, command, output))
            stage_records.append({
                "stage": "formal_hardware_image_evaluation",
                "label": image_label,
                "command": command,
                "status": "planned",
                "testRead": False,
            })
        if len(image_commands) != len(checkpoint_infos):
            expected_reused = len(checkpoint_infos) - len(image_commands)
            if expected_reused < 0:
                raise RuntimeError("formal image evaluation has too many commands")
        resource = _resource_command(
            args,
            checkpoint_infos,
            attempt_root / "resource",
            threshold_manifest_path,
        )
        if resource is None:
            raise RuntimeError("formal frozen-threshold resource evaluator is not configured")
        image_manifest = {
            "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-image-resource-manifest-v1",
            "testRead": False,
            "image": {
                "commands": [
                    {"label": label, "command": command, "output": str(output)}
                    for label, command, output in image_commands
                ],
                "status": "planned",
                "execution": "sequential_single_model_hardware_browser",
            },
            "resource": {
                "command": resource,
                "status": "planned",
                "thresholdManifest": str(threshold_manifest_path),
                "thresholdSelectionSplit": "calibration",
            },
        }
        _write_json(attempt_root / "image_resource_manifest.json", image_manifest)
        stage_records.append({
            "stage": "image_resource_manifest",
            "label": "image_resource_manifest",
            "command": ["<in-process>", "write_image_resource_manifest", str(attempt_root / "image_resource_manifest.json")],
            "status": "passed",
            "manifest": str(attempt_root / "image_resource_manifest.json"),
            "testRead": False,
        })
        for label, command, _output in image_commands:
            result = _command_result(
                command,
                stage="formal_hardware_image_evaluation",
                label=label,
                log_root=log_root,
                gpu=gpu_ids[0],
            )
            result["status"] = "passed" if int(result["returnCode"]) == 0 else "failed"
            stage_records.append(result)
            if int(result["returnCode"]) != 0:
                raise RuntimeError(f"formal image evaluation failed for {label}")
        resource_output = attempt_root / "resource"
        if reusable_output(
            "frozen_threshold_resource_evaluation",
            "resource",
            resource_output / "summary.json",
        ) and resource_output_ready(resource_output):
            stage_records.append(
                reused_stage_record(
                    "frozen_threshold_resource_evaluation",
                    "resource",
                    artifact=resource_output,
                )
            )
        else:
            result = _command_result(
                resource,
                stage="frozen_threshold_resource_evaluation",
                label="resource",
                log_root=log_root,
                gpu=gpu_ids[0],
            )
            result["status"] = "passed" if int(result["returnCode"]) == 0 else "failed"
            stage_records.append(result)
            if int(result["returnCode"]) != 0:
                raise RuntimeError("frozen-threshold resource evaluation failed")
        summary_path = attempt_root / "validation_summary.json"
        summarize = _summarizer_command(
            args,
            attempt_root,
            summary_path,
            image_root=image_root,
            resource_root=attempt_root / "resource",
        )
        if reusable_output(
            "paired_seed_pose_bootstrap_10000_and_image_backfill",
            "summarizer",
            summary_path,
        ) and validation_summary_ready(summary_path):
            stage_records.append(
                reused_stage_record(
                    "paired_seed_pose_bootstrap_10000_and_image_backfill",
                    "summarizer",
                    artifact=summary_path,
                )
            )
        else:
            stage_records.append({
                "stage": "paired_seed_pose_bootstrap_10000_and_image_backfill",
                "label": "summarizer",
                "command": summarize,
                "status": "planned",
                "bootstrapReplicates": 10000,
                "testRead": False,
            })
            summary_result = _command_result(
                summarize,
                stage="paired_seed_pose_bootstrap_10000_and_image_backfill",
                label="summarizer",
                log_root=log_root,
            )
            summary_result["status"] = "passed" if int(summary_result["returnCode"]) == 0 else "failed"
            stage_records.append(summary_result)
            if int(summary_result["returnCode"]) != 0:
                raise RuntimeError("10k paired seed/pose bootstrap or image backfill failed")
        route = _route_command(args, summary_path, attempt_root / "route_decision.json", attempt_root / "route_report.md")
        if route is None:
            raise RuntimeError("route decision/report CLI is not configured")
        route_output = attempt_root / "route_decision.json"
        route_report = attempt_root / "route_report.md"
        if (
            reusable_output("route_decision_report", "route", route_output)
            and route_output_ready(route_output, route_report)
        ):
            stage_records.append(
                reused_stage_record(
                    "route_decision_report",
                    "route",
                    artifact=route_output,
                )
            )
        else:
            stage_records.append({"stage": "route_decision_report", "label": "route", "command": route, "status": "planned", "testRead": False})
            route_result = _command_result(route, stage="route_decision_report", label="route", log_root=log_root)
            route_result["status"] = "passed" if int(route_result["returnCode"]) == 0 else "failed"
            stage_records.append(route_result)
            if int(route_result["returnCode"]) != 0:
                raise RuntimeError("route decision/report failed")
        formal_report = _write_formal_validation_report(
            summary_path,
            route_output,
            route_report,
            attempt_root,
        )
        stage_records.append({
            "stage": "formal_evaluation_report",
            "label": "docs_evaluation_report",
            "artifact": str(formal_report.resolve()),
            "status": "passed",
            "testRead": False,
        })
        result = {
            "status": "evaluation_complete",
            "evaluationRoot": str(attempt_root),
            "formalReport": str(formal_report.resolve()),
            "stages": stage_records,
            "testRead": False,
        }
        _write_json(_next_attempt_path(Path(run_manifest["outputRoot"]), "pipeline_status"), {"schema": "pvs-bounded-relation-prior-instance-calibrated-v4-pipeline-status-v1", **result})
        return result
    except Exception as exc:
        result = {"status": "training_complete_evaluation_incomplete", "evaluationRoot": str(attempt_root), "stages": stage_records, "error": repr(exc), "trainingMembersUntouched": True, "testRead": False}
        _write_json(_next_attempt_path(Path(run_manifest["outputRoot"]), "pipeline_status"), {"schema": "pvs-bounded-relation-prior-instance-calibrated-v4-pipeline-status-v1", **result})
        return result


def _write_training_status(output_root: Path, payload: Mapping[str, Any]) -> Path:
    path = _next_attempt_path(output_root, "training_status")
    _write_json(path, payload)
    return path


def _build_manifest(
    args: argparse.Namespace,
    mode_config: Mapping[str, Any],
    preflight: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
    output_root: Path,
    log_root: Path,
    benchmark_root: Path,
) -> dict[str, Any]:
    planned = []
    for job in jobs:
        member = output_root / str(job["member"])
        planned.append({
            **dict(job),
            "outputDir": str(member),
            "relationDir": str(_relation_dir_for_job(args, job)),
            "command": _train_command(args, job, member, mode_config),
            "status": "planned",
            "testRead": False,
        })
    return {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-v4-run-manifest-v1",
        "experimentPrefix": PREFIX,
        "mode": str(mode_config["mode"]),
        "outputRoot": str(output_root),
        "logRoot": str(log_root),
        "benchmarkOutputRoot": str(benchmark_root),
        "createdAtUnix": time.time(),
        "trainEntry": str(TRAIN),
        "epochs": int(mode_config["epochs"]),
        "stepsPerEpoch": int(mode_config["stepsPerEpoch"]),
        "snapshotEvery": int(mode_config.get("snapshotEvery", mode_config["evalEvery"])),
        "observationBatchSize": int(mode_config.get("observationBatchSize", args.observation_batch_size)),
        "variants": list(mode_config["variants"]),
        "seeds": list(mode_config["seeds"]),
        "plannedMemberCount": len(planned),
        "gpuIds": [int(value) for value in args.gpu_ids],
        "gpuPolicy": "four persistent single-GPU workers with dynamic completion queue; no distributed training",
        "fixedGeometry": {
            "path": preflight.get("geometry", {}).get("path"),
            "shape": preflight.get("geometry", {}).get("shape"),
            "dtype": preflight.get("geometry", {}).get("dtype"),
            "byteLength": preflight.get("geometry", {}).get("byteLength"),
        },
        "preflight": dict(preflight),
        "plannedMembers": planned,
        "postprocess": {
            "requiredOrder": [
                "checkpoint_schema_validation",
                "checkpoint_runtime_bundle_export",
                "checkpoint_runtime_bundle_validation",
                "calibration_frozen_validation_replay",
                "pose_candidate_gt_identity_validation",
                "formal_hardware_image_evaluation",
                "frozen_threshold_resource_evaluation",
                "paired_seed_pose_bootstrap_10000_and_image_backfill",
                "route_decision_report",
            ],
            "validatorCli": str(args.validator_cli) if args.validator_cli else None,
            "evaluatorCli": str(args.evaluator_cli) if args.evaluator_cli else None,
            "summarizerCli": str(args.summarizer_cli) if args.summarizer_cli else None,
            "exporterCli": str(args.exporter_cli) if args.exporter_cli else None,
            "imageCli": str(args.image_cli) if args.image_cli else None,
            "resourceCli": str(args.resource_cli) if args.resource_cli else None,
            "routeCli": str(args.route_cli) if args.route_cli else None,
        },
        "testRead": False,
    }


def _default_output_root(args: argparse.Namespace, mode: str) -> Path:
    return ROOT / "neural_instance_culling/model/out" / f"{PREFIX}_{mode}"


def _job_identity(job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "variant": str(job.get("variant", "")),
        "seed": int(job.get("seed", -1)),
        "configId": str(job.get("configId", "")),
        "hyperparameters": dict(job.get("hyperparameters", {})),
        "variantSpec": dict(job.get("variantSpec", {})),
        "member": str(job.get("member", "")),
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    mode_config: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> None:
    expected_scalars = {
        "mode": str(mode_config["mode"]),
        "epochs": int(mode_config["epochs"]),
        "stepsPerEpoch": int(mode_config["stepsPerEpoch"]),
        "observationBatchSize": int(mode_config.get("observationBatchSize", 8192)),
        "snapshotEvery": int(mode_config.get("snapshotEvery", mode_config["evalEvery"])),
        "variants": list(mode_config["variants"]),
        "seeds": list(mode_config["seeds"]),
        "plannedMemberCount": len(jobs),
    }
    for key, expected in expected_scalars.items():
        if manifest.get(key) != expected:
            raise ValueError(f"resume manifest {key} disagrees with the registered v4 mode")
    planned = manifest.get("plannedMembers")
    if not isinstance(planned, list) or len(planned) != len(jobs):
        raise ValueError("resume manifest planned member list is incomplete")
    actual_identity = sorted(
        (_job_identity(value) for value in planned if isinstance(value, Mapping)),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    expected_identity = sorted(
        (_job_identity(value) for value in jobs),
        key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")),
    )
    if actual_identity != expected_identity:
        raise ValueError("resume manifest member/config identity disagrees with the registered v4 matrix")


def _resume_preflight_view(value: Any, path: tuple[str, ...] = ()) -> Any:
    """Return the semantic preflight view used by an existing run manifest.

    Older v4 manifests recorded file fingerprints. They have no effect on
    data semantics, so resume ignores all legacy fingerprint fields while
    retaining schema, shape, split, candidate/GT and finite-value checks.
    """

    def _is_legacy_fingerprint_key(key: Any) -> bool:
        normalized = str(key).lower()
        return (
            normalized in {"sha", "sha1", "sha256", "digest", "hash", "checksum"}
            or normalized.endswith("sha1")
            or normalized.endswith("sha256")
            or normalized.endswith("digest")
            or normalized.endswith("hash")
            or normalized.endswith("checksum")
        )

    if isinstance(value, Mapping):
        return {
            str(key): _resume_preflight_view(item, path + (str(key),))
            for key, item in value.items()
            if not _is_legacy_fingerprint_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_resume_preflight_view(item, path + (str(index),)) for index, item in enumerate(value)]
    return value


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    mode_config = _mode_config(args.mode, args.variants, args.seeds)
    frozen_config = None
    if args.mode == "formal80" and args.frozen_config:
        source = Path(args.frozen_config).resolve()
        frozen_config = _validate_frozen_config_payload(_load_json(source), source)
    elif args.mode == "formal80" and args.scan_root:
        scan_root = Path(args.scan_root).resolve()
        # A dry-run must exercise the same immutable configuration contract as
        # the real run; a placeholder config would make its 15 planned jobs
        # different from the jobs that can actually be executed.
        frozen_config = _frozen_config_from_scan(scan_root)
    elif args.mode == "formal80" and not args.dry_run:
        raise ValueError("formal80 requires --frozen-config or --scan-root")
    jobs = _make_jobs(args, mode_config, frozen_config)
    output_root = Path(args.output_root).resolve() if args.output_root else _default_output_root(args, args.mode).resolve()
    log_root = Path(args.log_root).resolve() if args.log_root else (output_root / "logs").resolve()
    benchmark_root = Path(args.benchmark_output_root).resolve() if args.benchmark_output_root else DEFAULT_FORMAL_VALIDATION.resolve()
    preflight = _preflight(args, mode_config, jobs)
    existing_manifest = output_root / "run_manifest.json"
    resume = False
    if output_root.exists() and any(output_root.iterdir()):
        if not existing_manifest.is_file():
            raise FileExistsError(f"refusing to reuse non-empty output root without a run manifest: {output_root}")
        existing = _load_json(existing_manifest)
        if existing.get("mode") != args.mode:
            raise ValueError("existing run manifest belongs to a different v4 runner mode")
        if _resume_preflight_view(existing.get("preflight")) != _resume_preflight_view(preflight):
            raise ValueError("resume preflight contract differs from the immutable run manifest")
        _validate_resume_manifest(existing, mode_config, jobs)
        manifest = existing
        jobs = [dict(value) for value in manifest.get("plannedMembers", [])]
        resume = True
    else:
        _ensure_new_root(output_root)
        _ensure_new_root(log_root)
        manifest = _build_manifest(args, mode_config, preflight, jobs, output_root, log_root, benchmark_root)
        _write_json(existing_manifest, manifest)
    if args.dry_run:
        return {"status": "dry_run", "manifest": str(existing_manifest), "plannedMemberCount": len(jobs), "testRead": False}
    previous_training = _latest_record(output_root, "training_status")
    prior_results = previous_training[1].get("members", []) if previous_training else []
    prior_by_key = {
        (str(row.get("variant")), int(row.get("seed", -1)), str(row.get("configId", "registered"))): row
        for row in prior_results
        if isinstance(row, Mapping)
    }
    pending_jobs = []
    reused_results: list[dict[str, Any]] = []
    for job in jobs:
        key = (str(job["variant"]), int(job["seed"]), str(job.get("configId", "registered")))
        member_dir = _completed_member_dir(
            output_root,
            str(job["member"]),
            expected_epochs=int(mode_config["epochs"]),
        )
        prior = prior_by_key.get(key)
        member_identity_matches = (
            member_dir is not None
            and _completed_member_matches_job(member_dir, job)
        )
        if member_identity_matches:
            recovered = dict(prior) if prior is not None else {
                "variant": str(job["variant"]),
                "seed": int(job["seed"]),
                "configId": str(job.get("configId", "registered")),
                "member": str(job["member"]),
                "returnCode": 0,
            }
            reused_results.append({
                **recovered,
                "outputDir": str(member_dir),
                "reusedExisting": True,
                "recoveredWithoutTrainingStatus": prior is None,
                "testRead": False,
            })
        else:
            pending_jobs.append(job)
    new_results = _dynamic_queue(args, pending_jobs, output_root, log_root, mode_config, args.gpu_ids) if pending_jobs else []
    training_results = sorted(reused_results + new_results, key=lambda item: (str(item.get("variant", "")), int(item.get("seed", -1)), str(item.get("member", ""))))
    training_ok = len(training_results) == len(jobs) and all(
        int(row.get("returnCode", 1)) == 0
        and _training_artifacts_complete(
            Path(str(row.get("outputDir", ""))),
            expected_epochs=int(mode_config["epochs"]),
        )
        for row in training_results
    )
    training_status = "training_complete_pending_evaluation" if training_ok else "training_incomplete"
    _write_training_status(output_root, {"schema": "pvs-bounded-relation-prior-instance-calibrated-v4-training-status-v1", "status": training_status, "members": training_results, "reusedCompletedMembers": [row.get("member") for row in reused_results], "testRead": False})
    if not training_ok:
        return {"status": training_status, "manifest": str(existing_manifest), "members": training_results, "testRead": False}
    if bool(mode_config["formalPostprocess"]):
        return _formal_postprocess(args, manifest, training_results, jobs, args.gpu_ids)
    if args.mode == "scan8":
        if not (output_root / "scan_member_summary.json").exists():
            _write_json(output_root / "scan_member_summary.json", {"schema": f"{PREFIX}-scan8-summary-v1", "members": training_results, "testRead": False})
        frozen = _select_scan_configuration(
            output_root,
            training_results,
            run_manifest=manifest,
            jobs=jobs,
            expected_epochs=int(mode_config["epochs"]),
        )
        if not (output_root / "scan_configuration_summary.json").exists():
            _write_json(output_root / "scan_configuration_summary.json", {"schema": f"{PREFIX}-scan8-configuration-summary-v1", "selected": frozen, "testRead": False})
        frozen_path = output_root / "frozen_config.json"
        if frozen_path.exists():
            existing = _validate_frozen_config_payload(_load_json(frozen_path), frozen_path)
            current = _validate_frozen_config_payload(frozen, frozen_path)
            if (
                existing["configId"] != current["configId"]
                or existing["hyperparameters"] != current["hyperparameters"]
            ):
                raise ValueError("existing frozen scan configuration disagrees with the current complete scan")
        else:
            _write_json(output_root / "frozen_config.json", frozen)
    else:
        summary_path = output_root / f"{args.mode}_member_summary.json"
        if not summary_path.exists():
            _write_json(summary_path, {"schema": f"{PREFIX}-{args.mode}-summary-v1", "members": training_results, "testRead": False})
    return {"status": "training_complete", "manifest": str(existing_manifest), "members": training_results, "testRead": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "scan8", "formal80"))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--relation-dir", default=str(DEFAULT_RELATION))
    parser.add_argument("--runtime-meta", default=str(DEFAULT_RUNTIME_META))
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_GEO))
    parser.add_argument("--subpose-sidecar", default="")
    parser.add_argument("--glb-index", default=str(DEFAULT_GLB_INDEX))
    parser.add_argument("--glb-root", default=str(DEFAULT_GLB_ROOT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--benchmark-output-root", default=str(DEFAULT_FORMAL_VALIDATION))
    parser.add_argument("--scan-root", default="")
    parser.add_argument("--frozen-config", default="")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--observation-batch-size", type=int, default=8192)
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validator-cli", default=str(VALIDATOR_CLI))
    parser.add_argument("--evaluator-cli", default=str(EVALUATOR_CLI))
    parser.add_argument("--summarizer-cli", default=str(SUMMARIZER_CLI))
    parser.add_argument("--exporter-cli", default=str(EXPORTER_CLI))
    parser.add_argument("--image-cli", default=str(IMAGE_CLI))
    parser.add_argument("--resource-cli", default=str(RESOURCE_CLI))
    parser.add_argument("--route-cli", default=str(ROUTE_CLI))
    parser.add_argument("--viewcell-dataset", default=str(ROOT / "neural_instance_culling/dataset/out/hkust_v3_viewcell_colorid_fov66_source"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids) or any(int(value) < 0 for value in args.gpu_ids):
        raise ValueError("gpu-ids must be unique non-negative IDs")
    if args.poses_per_batch <= 0 or args.observation_batch_size <= 0:
        raise ValueError("pose and observation batch sizes must be positive")
    result = run_experiment(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
