#!/usr/bin/env python3
"""Train the relation-prior PVS model with per-instance calibration.

This entry reads train, calibration, and validation only.  The train-owned
relation graph first generates a shared prior.  A zero-initialized train-only
residual then calibrates each instance before both terms are fused into the
fixed 124-value runtime table.  Calibration alone freezes each checkpoint's
threshold, while validation is replayed only at that frozen threshold.  Test
is not a valid argument or fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from bounded_relation_survival_moment_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    DUAL_PROBE_RAW_QUERY_DIM,
    GEO_DIM,
    MODEL_SCHEMA,
    QUERY_TAIL_SEPARATOR_FAMILIES,
    RUNTIME_FEATURE_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM,
    VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM,
)
from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.dual_probe_rescue_loss import (  # noqa: E402
    dual_probe_rescue_attenuation_loss,
)
from common.viewcell_extreme_envelope import (  # noqa: E402
    VIEWCELL_EXTREME_ENVELOPE_DIM,
    VIEWCELL_SUPPORT_ENVELOPE_DIM,
)
from common.viewcell_boundary_opportunity_loss import (  # noqa: E402
    viewcell_boundary_opportunity_loss,
)
from common.viewcell_tail_partial_auc_loss import (  # noqa: E402
    viewcell_tail_partial_auc_loss,
)
from common.weighted_safety_frontier_loss import (  # noqa: E402
    weighted_safety_frontier_pair_loss,
)
from common.weighted_neyman_pearson_operating_loss import (  # noqa: E402
    WeightedNeymanPearsonDualState,
    weighted_neyman_pearson_operating_loss,
)
from common.viewcell_tail_boundary_deficit_loss import (  # noqa: E402
    viewcell_tail_boundary_deficit_loss,
)
from common.occlusion_edges import glb_priority_loss  # noqa: E402
from common.provenance import relation_artifact_digest  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.safety_reserve_operating_utility_loss import (  # noqa: E402
    OPERATING_THRESHOLD_ANCHORS,
    OPERATING_THRESHOLD_RANGE,
    candidate_boundary_hard_negative_loss,
    cull_certificate_pose_tail_pair_loss,
    cull_certificate_tail_classification_loss,
    instance_exposure_balanced_bce_loss,
    project_operating_utility_gradient_groups,
    safety_reserve_operating_utility_loss,
    same_instance_cross_view_rank_loss,
    sample_train_operating_thresholds,
    view_residual_tail_regularizers,
    weighted_positive_safety_boundary_logit,
)
from common.stratified_survival_sampler import StratifiedSurvivalObservationSampler  # noqa: E402
from common.survival_loss import stratified_survival_censoring_loss  # noqa: E402
from common.threshold_selection import (  # noqa: E402
    aggregate_weighted_cull_selection_rule,
    select_aggregate_weighted_cull_workpoint,
)
from common.train_observed_relation_csr import (  # noqa: E402
    ObservedRelationCSR,
    SCHEMA_V3 as RELATION_SCHEMA_V3,
    load_survival_observations_v3,
    validate_relation_metadata_v3,
    validate_survival_observations_v3,
)
from current_pvs_utils import evaluate_thresholds, threshold_grid, visual_utility_loss  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


EXPERIMENT_PREFIX = "pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4"
CALIBRATION_FLOOR = 0.99
SURVIVAL_SHAPE = (4, 7)
DUAL_PROBE_RESCUE_INIT_SCHEMA = "pvs-dual-probe-rescue-init-v1"


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_value(value), ensure_ascii=False) + "\n")


def _load_dual_probe_rescue_init(path: Path | None) -> dict[str, Any] | None:
    """Load the train-owned dual-probe posterior without selecting anything here."""
    if path is None:
        return None
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"dual-probe rescue init does not exist: {resolved}")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"dual-probe rescue init is not valid JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("dual-probe rescue init must contain a JSON object")
    if payload.get("schema") != DUAL_PROBE_RESCUE_INIT_SCHEMA:
        raise ValueError(
            f"dual-probe rescue init must use schema {DUAL_PROBE_RESCUE_INIT_SCHEMA}"
        )
    if payload.get("testRead") is not False:
        raise ValueError("dual-probe rescue init must have testRead=false")
    if payload.get("selectedFromValidation") is not False:
        raise ValueError(
            "dual-probe rescue init must have selectedFromValidation=false"
        )
    if payload.get("enabled") is not True:
        raise ValueError("dual-probe rescue init must explicitly enable dual-probe rescue")
    try:
        raw_feature_dimension = int(payload.get("rawFeatureDimension", -1))
    except (TypeError, ValueError) as exc:
        raise ValueError("dual-probe rescue init rawFeatureDimension must be an integer") from exc
    if raw_feature_dimension != int(DUAL_PROBE_RAW_QUERY_DIM):
        raise ValueError(
            f"dual-probe rescue init must declare rawFeatureDimension={DUAL_PROBE_RAW_QUERY_DIM}"
        )
    expected_layout = [
        {"name": "center_view", "offset": 0, "dimension": 9},
        {"name": "spectral_features", "offset": 9, "dimension": 64},
        {"name": "boundary_spectral_summary", "offset": 73, "dimension": 8},
        {
            "name": "region_extrema_lower_upper_span",
            "offset": 81,
            "dimension": 27,
        },
    ]
    if payload.get("rawFeatureLayout") != expected_layout:
        raise ValueError("dual-probe rescue init rawFeatureLayout does not match the 108D contract")
    for role in ("primary", "coverage"):
        record = payload.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"dual-probe rescue init is missing the {role} probe")
        if record.get("fitSplit") != "train":
            raise ValueError(f"dual-probe {role} probe must declare fitSplit=train")
        certificate = record.get("riskCertificate")
        if not isinstance(certificate, Mapping):
            raise ValueError(f"dual-probe {role} probe is missing its riskCertificate")
        if (
            certificate.get("sourceSplit") != "train"
            or certificate.get("fitRowsOnly") is not True
            or certificate.get("riskWithinRegisteredBound") is not True
        ):
            raise ValueError(
                f"dual-probe {role} riskCertificate must be a train-only bounded certificate"
            )
        for key in ("mean", "scale", "coefficients", "threshold"):
            if key not in record:
                raise ValueError(f"dual-probe {role} probe is missing {key}")
    return dict(payload)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_safe_checkpoint_alias(payload: Mapping[str, Any], output_dir: Path) -> None:
    """Write one safe checkpoint and make best.pt its byte-identical alias."""
    safe_path = output_dir / "best_safe.pt"
    alias_path = output_dir / "best.pt"
    torch.save(payload, safe_path)
    shutil.copyfile(safe_path, alias_path)
    if _sha256(safe_path) != _sha256(alias_path):
        raise IOError("best.pt is not a byte-identical copy of best_safe.pt")


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty experiment output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _initialize_from_checkpoint(
    model: BoundedRelationSurvivalMomentModel,
    checkpoint_path: Path | None,
) -> dict[str, Any] | None:
    """Load model weights for train-only tail refinement with a fresh optimizer."""
    if checkpoint_path is None:
        return None
    path = checkpoint_path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"initial checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("initial checkpoint payload must be a mapping")
    if checkpoint.get("schema") != "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4":
        raise ValueError("initial checkpoint is not a v4 bounded-relation checkpoint")
    if checkpoint.get("runtimeSchema") != MODEL_SCHEMA or checkpoint.get("testRead") is not False:
        raise ValueError("initial checkpoint runtime schema or test provenance is invalid")
    source_config = checkpoint.get("modelConfig")
    if not isinstance(source_config, Mapping):
        raise ValueError("initial checkpoint model config is missing")
    target_config = model.config
    fresh_certificate = False
    fresh_view_residual = False
    fresh_boundary_opportunity = False
    fresh_boundary_tail_residual = False
    fresh_viewcell_extreme_visibility = False
    fresh_viewcell_region_conditioned_visibility = False
    fresh_dual_probe_rescue = False
    migrated_moment_extrema = False
    if source_config != target_config:
        target_without_fresh_heads = dict(target_config)

        def newly_enabled(key: str) -> bool:
            target_value = target_config.get(key)
            source_value = source_config.get(key)
            target_enabled = isinstance(target_value, Mapping) and bool(
                target_value.get("enabled")
            )
            source_enabled = isinstance(source_value, Mapping) and bool(
                source_value.get("enabled")
            )
            if target_enabled and not source_enabled:
                target_without_fresh_heads.pop(key, None)
                return True
            return False

        fresh_certificate = newly_enabled("cullCertificate")
        fresh_view_residual = newly_enabled("viewResidual")
        fresh_boundary_opportunity = newly_enabled("boundaryOpportunity")
        fresh_boundary_tail_residual = newly_enabled("boundaryTailResidual")
        fresh_viewcell_extreme_visibility = newly_enabled(
            "viewcellExtremeVisibility"
        )
        fresh_viewcell_region_conditioned_visibility = newly_enabled(
            "viewcellRegionConditionedVisibility"
        )
        fresh_dual_probe_rescue = newly_enabled("dualProbeRescue")
        source_without_spectral = dict(source_config)
        target_without_spectral = dict(target_without_fresh_heads)
        source_spectral = source_without_spectral.pop("spectralMode", None)
        target_spectral = target_without_spectral.pop("spectralMode", None)
        target_without_spectral.pop("viewcellExtremeEnvelope", None)
        target_without_spectral.pop("viewcellSupportEnvelope", None)
        migration_extra_dim = {
            "moment_extrema": VIEWCELL_EXTREME_ENVELOPE_DIM,
            "moment_support": VIEWCELL_SUPPORT_ENVELOPE_DIM,
            "moment_extrema_support": (
                VIEWCELL_EXTREME_ENVELOPE_DIM + VIEWCELL_SUPPORT_ENVELOPE_DIM
            ),
        }.get(str(target_spectral), 0)
        migrated_moment_extrema = (
            source_spectral == "moment_envelope"
            and migration_extra_dim > 0
            and source_without_spectral == target_without_spectral
        )
        if (
            not (
                fresh_certificate
                or fresh_view_residual
                or fresh_boundary_opportunity
                or fresh_boundary_tail_residual
                or fresh_viewcell_extreme_visibility
                or fresh_viewcell_region_conditioned_visibility
                or fresh_dual_probe_rescue
            )
            and not migrated_moment_extrema
        ) or (
            not migrated_moment_extrema
            and source_config != target_without_fresh_heads
        ):
            raise ValueError(
                "initial checkpoint model config does not match the requested model"
            )
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("initial checkpoint has no model state")
    migrated_state = dict(state)
    if migrated_moment_extrema:
        key = "boundary_summary_head.0.weight"
        source_weight = torch.as_tensor(state[key])
        target_weight = model.state_dict()[key].clone()
        if (
            source_weight.ndim != 2
            or target_weight.ndim != 2
            or source_weight.shape[0] != target_weight.shape[0]
            or target_weight.shape[1] - source_weight.shape[1]
            != migration_extra_dim
        ):
            raise ValueError("moment-extrema checkpoint migration shape is invalid")
        target_weight[:, : source_weight.shape[1]].copy_(source_weight)
        target_weight[:, source_weight.shape[1] :].zero_()
        migrated_state[key] = target_weight
    if (
        fresh_certificate
        or fresh_view_residual
        or fresh_boundary_opportunity
        or fresh_boundary_tail_residual
        or fresh_viewcell_extreme_visibility
        or fresh_viewcell_region_conditioned_visibility
        or fresh_dual_probe_rescue
    ):
        incompatible = model.load_state_dict(migrated_state, strict=False)
        expected_missing: set[str] = set()
        if fresh_certificate:
            assert model.cull_certificate_head is not None
            expected_missing.update(
                f"cull_certificate_head.{name}"
                for name in model.cull_certificate_head.state_dict()
            )
        if fresh_view_residual:
            assert model.view_residual_geometry_projection is not None
            assert model.view_residual_dynamic_projection is not None
            assert model.view_residual_head is not None
            for prefix, module in (
                ("view_residual_geometry_projection", model.view_residual_geometry_projection),
                ("view_residual_dynamic_projection", model.view_residual_dynamic_projection),
                ("view_residual_head", model.view_residual_head),
            ):
                expected_missing.update(
                    f"{prefix}.{name}" for name in module.state_dict()
                )
        if fresh_boundary_opportunity:
            assert model.boundary_opportunity_geometry_projection is not None
            assert model.boundary_opportunity_region_projection is not None
            assert model.boundary_opportunity_head is not None
            expected_missing.add("boundary_opportunity_scale")
            for prefix, module in (
                (
                    "boundary_opportunity_geometry_projection",
                    model.boundary_opportunity_geometry_projection,
                ),
                (
                    "boundary_opportunity_region_projection",
                    model.boundary_opportunity_region_projection,
                ),
                ("boundary_opportunity_head", model.boundary_opportunity_head),
            ):
                expected_missing.update(
                    f"{prefix}.{name}" for name in module.state_dict()
                )
        if fresh_boundary_tail_residual:
            assert model.boundary_tail_residual_geometry_projection is not None
            assert model.boundary_tail_residual_region_projection is not None
            assert model.boundary_tail_residual_head is not None
            tail_modules = [
                (
                    "boundary_tail_residual_geometry_projection",
                    model.boundary_tail_residual_geometry_projection,
                ),
                (
                    "boundary_tail_residual_region_projection",
                    model.boundary_tail_residual_region_projection,
                ),
                ("boundary_tail_residual_head", model.boundary_tail_residual_head),
            ]
            if model.boundary_tail_residual_region_shortcut is not None:
                tail_modules.append(
                    (
                        "boundary_tail_residual_region_shortcut",
                        model.boundary_tail_residual_region_shortcut,
                    )
                )
            for prefix, module in tail_modules:
                expected_missing.update(
                    f"{prefix}.{name}" for name in module.state_dict()
                )
        if fresh_viewcell_extreme_visibility:
            assert model.viewcell_extreme_visibility_projection is not None
            assert model.viewcell_extreme_visibility_head is not None
            for prefix, module in (
                (
                    "viewcell_extreme_visibility_projection",
                    model.viewcell_extreme_visibility_projection,
                ),
                (
                    "viewcell_extreme_visibility_head",
                    model.viewcell_extreme_visibility_head,
                ),
            ):
                expected_missing.update(
                    f"{prefix}.{name}" for name in module.state_dict()
                )
        if fresh_viewcell_region_conditioned_visibility:
            assert model.viewcell_region_conditioned_visibility_region_projection is not None
            assert model.viewcell_region_conditioned_visibility_hidden_projection is not None
            assert model.viewcell_region_conditioned_visibility_head is not None
            for prefix, module in (
                (
                    "viewcell_region_conditioned_visibility_region_projection",
                    model.viewcell_region_conditioned_visibility_region_projection,
                ),
                (
                    "viewcell_region_conditioned_visibility_hidden_projection",
                    model.viewcell_region_conditioned_visibility_hidden_projection,
                ),
                (
                    "viewcell_region_conditioned_visibility_head",
                    model.viewcell_region_conditioned_visibility_head,
                ),
            ):
                expected_missing.update(
                    f"{prefix}.{name}" for name in module.state_dict()
                )
        if fresh_dual_probe_rescue:
            expected_missing.update(
                name
                for name in model.state_dict()
                if name.startswith("dual_probe_")
            )
        if (
            set(incompatible.missing_keys) != expected_missing
            or incompatible.unexpected_keys
        ):
            raise ValueError(
                "initial checkpoint differs by more than the registered fresh query heads"
            )
    elif migrated_moment_extrema:
        model.load_state_dict(migrated_state, strict=True)
    else:
        model.load_state_dict(state, strict=True)
    blend = float(model.instance_calibration_blend.detach().cpu())
    if not math.isfinite(blend) or not 0.0 <= blend <= 1.0:
        raise ValueError("initial checkpoint has an invalid instance calibration blend")
    return {
        "mode": (
            "model-weights-plus-zero-initialized-moment-extrema-and-fresh-viewcell-extreme-visibility-fresh-optimizer"
            if migrated_moment_extrema and fresh_viewcell_extreme_visibility
            else "model-weights-plus-fresh-dual-probe-rescue-fresh-optimizer"
            if fresh_dual_probe_rescue
            else "model-weights-plus-fresh-viewcell-extreme-visibility-fresh-optimizer"
            if fresh_viewcell_extreme_visibility
            else "model-weights-plus-fresh-viewcell-region-conditioned-visibility-fresh-optimizer"
            if fresh_viewcell_region_conditioned_visibility
            else
            "model-weights-plus-zero-initialized-moment-extrema-and-fresh-boundary-opportunity-fresh-optimizer"
            if migrated_moment_extrema and fresh_boundary_opportunity
            else "model-weights-plus-zero-initialized-moment-extrema-and-fresh-boundary-tail-residual-fresh-optimizer"
            if migrated_moment_extrema and fresh_boundary_tail_residual
            else "model-weights-plus-zero-initialized-moment-extrema-fresh-optimizer"
            if migrated_moment_extrema
            else
            "model-weights-plus-fresh-cull-certificate-and-view-residual-fresh-optimizer"
            if fresh_certificate and fresh_view_residual
            else "model-weights-plus-fresh-cull-certificate-fresh-optimizer"
            if fresh_certificate
            else "model-weights-plus-fresh-view-residual-fresh-optimizer"
            if fresh_view_residual
            else "model-weights-plus-fresh-boundary-tail-residual-fresh-optimizer"
            if fresh_boundary_tail_residual
            else "model-weights-only-fresh-optimizer"
        ),
        "path": str(path),
        "sourceEpoch": int(checkpoint.get("epoch", 0)),
        "sourceExperiment": str(checkpoint.get("experimentName", "")),
        "instanceCalibrationBlend": blend,
        "freshCullCertificate": bool(fresh_certificate),
        "freshViewResidual": bool(fresh_view_residual),
        "freshBoundaryOpportunity": bool(fresh_boundary_opportunity),
        "freshBoundaryTailResidual": bool(fresh_boundary_tail_residual),
        "freshViewcellExtremeVisibility": bool(
            fresh_viewcell_extreme_visibility
        ),
        "freshViewcellRegionConditionedVisibility": bool(
            fresh_viewcell_region_conditioned_visibility
        ),
        "freshDualProbeRescue": bool(fresh_dual_probe_rescue),
        "migratedMomentExtrema": bool(migrated_moment_extrema),
        "sourceTestRead": False,
    }


def _load_geometry(path: Path, num_instances: int) -> tuple[torch.Tensor, dict[str, Any]]:
    expected = int(num_instances) * GEO_DIM
    if not path.is_file() or path.stat().st_size != expected * 2:
        raise ValueError(f"fixed geometry must be [{num_instances}, {GEO_DIM}] FP16: {path}")
    values = np.fromfile(path, dtype="<f2").reshape(num_instances, GEO_DIM)
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry contains non-finite values")
    return torch.from_numpy(values.astype(np.float32)), {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "shape": [num_instances, GEO_DIM],
        "dtype": "float16",
    }


def _read_uint32(path: Path, expected: int, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"missing {name}: {path}")
    values = np.fromfile(path, dtype="<u4").astype(np.int64)
    if values.size != int(expected):
        raise ValueError(f"{name} has {values.size} rows, expected {expected}")
    if values.size and (int(values.min()) < 0 or not np.array_equal(np.unique(values), np.arange(int(values.max()) + 1))):
        raise ValueError(f"{name} must use contiguous non-negative IDs")
    return values


def _load_relation_bundle(
    relation_dir: Path,
    dataset: PoseCSRDataset,
    train_split: Any,
    num_instances: int,
) -> tuple[
    ObservedRelationCSR,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, np.ndarray],
    dict[str, Any],
]:
    relation = ObservedRelationCSR.load(relation_dir)
    if relation.metadata.get("schema") != RELATION_SCHEMA_V3:
        raise ValueError("v4 training refuses legacy relation artifacts")
    train_digest = candidate_digest_for_pose_sequence(dataset, train_split.pose_indices)
    validate_relation_metadata_v3(
        relation.metadata,
        expected_num_instances=num_instances,
        expected_candidate_digest=train_digest,
    )
    if relation.direction_bins != 12 or relation.depth_shells != 3:
        raise ValueError("v4 requires 12 directions and 3 ordered depth shells")
    hierarchy = relation.metadata["hierarchy"]
    hierarchy_files = hierarchy["files"]
    local = _read_uint32(
        relation_dir / hierarchy_files["localGroupIds"], num_instances, "local group IDs"
    )
    local_count = int(local.max()) + 1
    structural = _read_uint32(
        relation_dir / hierarchy_files["structuralGroupIds"],
        local_count,
        "structural group IDs",
    )
    local_counts = np.bincount(local, minlength=local_count)
    structural_counts = np.bincount(structural, minlength=int(structural.max()) + 1)
    structural_instance_counts = np.bincount(
        structural[local], minlength=int(structural.max()) + 1
    )
    if int(local_counts.max()) > 32 or int(structural_counts.max()) > 64:
        raise ValueError("loaded hierarchy violates v3 capacity bounds")
    if float(structural_instance_counts.max() / max(1, num_instances)) > 0.10 + 1e-9:
        raise ValueError("loaded hierarchy contains an oversized structural group")
    source_k = int(relation.metadata["evidenceTopK"]["k"])
    if np.any(np.diff(relation.row_offsets).astype(np.int64) > source_k):
        raise ValueError("relation CSR declares top-k but contains an oversized row")
    retained_quality = float(relation.metadata["evidenceTopK"].get("minRetainedQuality", 0.0))
    quality_floor = float(relation.metadata["evidenceTopK"].get("qualityFloor", 0.90))
    if retained_quality + 1e-9 < quality_floor:
        raise ValueError("relation evidence top-k retained quality is below its registered floor")

    envelope = relation.metadata.get("survivalObservations")
    if not isinstance(envelope, Mapping) or envelope.get("schema") != "pvs-viewcell-train-observed-survival-censoring-v3":
        raise ValueError("relation artifact lacks corrected v3 event/censor observations")
    files = envelope.get("files")
    if not isinstance(files, Mapping) or "normalizedDepth" not in files:
        raise ValueError("v3 observations must store normalizedDepth, not historical rho")
    loaded_observations = load_survival_observations_v3(relation_dir, relation.metadata)
    observations: dict[str, np.ndarray] = {
        "instance": loaded_observations["instance"],
        "direction": loaded_observations["direction"],
        "normalized_depth": loaded_observations["normalizedDepth"],
        "event": loaded_observations["event"],
        "weight": loaded_observations["weight"],
        "subpose": loaded_observations["subpose"],
        "rawPixelCount": loaded_observations["rawPixelCount"],
    }
    observation_files: dict[str, Any] = {}
    for name, file_key in (
        ("instance", "instance"),
        ("direction", "direction"),
        ("normalized_depth", "normalizedDepth"),
        ("event", "event"),
        ("weight", "weight"),
        ("subpose", "subpose"),
        ("rawPixelCount", "rawPixelCount"),
    ):
        relative = files[file_key]
        path = relation_dir / relative
        observation_files[name] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    validation_alias = {
        "instance": observations["instance"],
        "direction": observations["direction"],
        "normalizedDepth": observations["normalized_depth"],
        "event": observations["event"],
        "weight": observations["weight"],
        "subpose": observations["subpose"],
        "rawPixelCount": observations["rawPixelCount"],
    }
    observation_summary = validate_survival_observations_v3(
        validation_alias,
        num_instances=num_instances,
        direction_bins=relation.direction_bins,
    )
    if observation_summary["eventCount"] <= 0 or observation_summary["rightCensoredCount"] <= 0:
        raise ValueError("v3 survival supervision must contain both events and right-censored rows")
    observations["depth"] = observations["normalized_depth"]
    artifact_digest, artifact_files = relation_artifact_digest(relation_dir, relation.metadata)
    provenance = {
        "schema": relation.metadata["schema"],
        "path": str(relation_dir.resolve()),
        "artifactDigest": artifact_digest,
        "artifactFiles": artifact_files,
        "candidateDigest": train_digest,
        "edgeCount": relation.edge_count,
        "rowCount": relation.row_count,
        "hierarchy": {
            "localGroupCount": local_count,
            "structuralGroupCount": int(structural.max()) + 1,
            "localMaxSize": int(local_counts.max()),
            "structuralMaxSize": int(structural_counts.max()),
            "maxStructuralInstanceFraction": float(
                structural_instance_counts.max() / max(1, num_instances)
            ),
        },
        "observations": {**observation_summary, "files": observation_files},
    }
    return (
        relation,
        relation.to_torch(),
        torch.from_numpy(local),
        torch.from_numpy(structural),
        observations,
        provenance,
    )


def _load_glb_bytes(
    index_path: Path,
    root: Path,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    *,
    allow_missing: bool,
) -> np.ndarray:
    if not index_path.is_file():
        raise FileNotFoundError(f"missing GLB index: {index_path}")
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    result = np.zeros((num_glbs,), dtype=np.float32)
    for entry in payload.get("entries", []):
        glb_id = int(entry.get("globalId", -1))
        path = root / str(entry.get("path", ""))
        if 0 <= glb_id < num_glbs and path.is_file():
            result[glb_id] = float(path.stat().st_size)
    used = np.unique(instance_to_glb.astype(np.int64))
    missing = used[(used < 0) | (used >= num_glbs) | (result[np.clip(used, 0, num_glbs - 1)] <= 0)]
    if missing.size and not allow_missing:
        raise FileNotFoundError(f"formal training has missing GLB byte costs: {missing[:20].tolist()}")
    if missing.size:
        fallback = float(np.median(result[result > 0])) if np.any(result > 0) else 1.0
        result[result <= 0] = max(1.0, fallback)
    return result


def _resolve_split(dataset: PoseCSRDataset, requested: str, role: str) -> tuple[str, Any]:
    if requested.lower() == "test":
        raise ValueError("test is forbidden during v4 training and threshold selection")
    if requested == "auto":
        choices = {
            "train": ("train",),
            "calibration": ("calibration",),
            "validation": ("validation", "val"),
        }[role]
        requested = next((value for value in choices if value in dataset.split_ids), "")
    if not requested or requested not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit {role} split")
    split = dataset.split(requested)
    if split.pose_indices.size == 0:
        raise ValueError(f"{role} split is empty")
    return requested, split


def _normalized_log_support(counts: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(counts, dtype=np.float64).reshape(-1)
    positive = values[values > 0.0]
    if positive.size == 0:
        return np.zeros(values.shape, dtype=np.float32), 0.0
    reference = max(float(np.percentile(positive, 95.0)), 1.0)
    normalized = np.log1p(values) / math.log1p(reference)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32), reference


def _instance_calibration_reliability(
    dataset: PoseCSRDataset,
    train_split: Any,
    observations: Mapping[str, np.ndarray],
    num_instances: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a train-only reliability prior for regularizing instance residuals."""
    candidate_counts = np.zeros((int(num_instances),), dtype=np.int64)
    for pose_index in np.asarray(train_split.pose_indices, dtype=np.int64).tolist():
        ids = np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.int64)
        if ids.size:
            candidate_counts += np.bincount(ids, minlength=int(num_instances))
    observation_ids = np.asarray(observations["instance"], dtype=np.int64).reshape(-1)
    if observation_ids.size and (
        int(observation_ids.min()) < 0 or int(observation_ids.max()) >= int(num_instances)
    ):
        raise ValueError("survival observation instance is outside the runtime instance table")
    observation_counts = np.bincount(
        observation_ids, minlength=int(num_instances)
    ).astype(np.int64, copy=False)
    candidate_support, candidate_reference = _normalized_log_support(candidate_counts)
    observation_support, observation_reference = _normalized_log_support(observation_counts)
    reliability = np.maximum(candidate_support, observation_support).astype(np.float32, copy=False)
    if not np.isfinite(reliability).all() or np.any((reliability < 0.0) | (reliability > 1.0)):
        raise FloatingPointError("instance calibration reliability is invalid")
    raw = np.ascontiguousarray(reliability.astype("<f4", copy=False)).tobytes(order="C")
    return reliability, {
        "sourceSplit": "train",
        "definition": "max(log1p(candidate_count)/log1p(candidate_p95), log1p(observation_count)/log1p(observation_p95))",
        "candidateCountP95Reference": candidate_reference,
        "observationCountP95Reference": observation_reference,
        "candidateReferenceCount": int(candidate_counts.sum()),
        "observationReferenceCount": int(observation_counts.sum()),
        "zeroReliabilityCount": int(np.count_nonzero(reliability <= 0.0)),
        "meanReliability": float(reliability.mean()),
        "p50Reliability": float(np.percentile(reliability, 50.0)),
        "p95Reliability": float(np.percentile(reliability, 95.0)),
        "sha256Float32": hashlib.sha256(raw).hexdigest(),
        "testRead": False,
    }


def _instance_exposure_balance_weights(
    dataset: PoseCSRDataset,
    train_split: Any,
    num_instances: int,
    *,
    power: float,
    maximum_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build train-only weights that remove instance visibility-prior shortcuts."""
    if not 0.0 < float(power) <= 1.0 or float(maximum_weight) < 1.0:
        raise ValueError("instance exposure balance arguments are invalid")
    candidate_counts = np.zeros((int(num_instances),), dtype=np.int64)
    positive_counts = np.zeros((int(num_instances),), dtype=np.int64)
    for pose_index in np.asarray(train_split.pose_indices, dtype=np.int64).tolist():
        candidate_ids = np.asarray(dataset.frustum_slice(int(pose_index)), dtype=np.int64)
        positive_ids = np.asarray(dataset.visible_slice(int(pose_index))[0], dtype=np.int64)
        if candidate_ids.size:
            candidate_counts += np.bincount(candidate_ids, minlength=int(num_instances))
        if positive_ids.size:
            positive_counts += np.bincount(positive_ids, minlength=int(num_instances))
    if np.any(positive_counts > candidate_counts):
        raise ValueError("train positives exceed train candidate exposure")
    negative_counts = candidate_counts - positive_counts

    def class_weights(class_counts: np.ndarray) -> np.ndarray:
        result = np.zeros((int(num_instances),), dtype=np.float64)
        supported = class_counts > 0
        result[supported] = np.power(
            candidate_counts[supported] / class_counts[supported], float(power)
        )
        result = np.minimum(result, float(maximum_weight))
        denominator = float((result * class_counts).sum())
        if denominator > 0.0:
            result *= float(class_counts.sum()) / denominator
        return result.astype(np.float32)

    positive_weights = class_weights(positive_counts)
    negative_weights = class_weights(negative_counts)
    if not bool(np.isfinite(positive_weights).all() and np.isfinite(negative_weights).all()):
        raise FloatingPointError("instance exposure balance weights are non-finite")
    positive_rates = positive_counts[candidate_counts > 0] / candidate_counts[candidate_counts > 0]
    return positive_weights, negative_weights, {
        "sourceSplit": "train",
        "definition": "class observation weight = min(maxWeight, (candidate_count/class_count)^power), normalized to class-observation mean one",
        "power": float(power),
        "maximumWeight": float(maximum_weight),
        "candidateReferenceCount": int(candidate_counts.sum()),
        "positiveReferenceCount": int(positive_counts.sum()),
        "negativeReferenceCount": int(negative_counts.sum()),
        "supportedInstanceCount": int(np.count_nonzero(candidate_counts)),
        "positiveSupportedInstanceCount": int(np.count_nonzero(positive_counts)),
        "negativeSupportedInstanceCount": int(np.count_nonzero(negative_counts)),
        "positiveRateMean": float(positive_rates.mean()) if positive_rates.size else 0.0,
        "positiveRateMedian": float(np.median(positive_rates)) if positive_rates.size else 0.0,
        "positiveWeightMax": float(positive_weights.max(initial=0.0)),
        "negativeWeightMax": float(negative_weights.max(initial=0.0)),
        "testRead": False,
    }


def _candidate_frustum_boundary_proximity(
    candidate_camera_world: torch.Tensor,
    camera_view: torch.Tensor,
    instance_aabbs: torch.Tensor,
    *,
    boundary_threshold: float = 0.05,
    temperature: float = 0.05,
    near: float = 0.05,
) -> torch.Tensor:
    """Return a train-only proximity to the registered candidate-frustum edge."""
    camera = candidate_camera_world.float()
    view = camera_view.float()
    aabbs = instance_aabbs.float()
    if camera.ndim != 2 or camera.shape[1] != 3:
        raise ValueError("candidate cameras must have shape [B, 3]")
    if view.shape != (camera.shape[0], 5) or aabbs.shape != (camera.shape[0], 6):
        raise ValueError("candidate view and AABB rows must align")
    if (
        float(boundary_threshold) < 0.0
        or float(temperature) <= 0.0
        or float(near) <= 0.0
    ):
        raise ValueError("candidate boundary proximity arguments are invalid")
    if not bool(
        torch.isfinite(camera).all()
        and torch.isfinite(view).all()
        and torch.isfinite(aabbs).all()
    ):
        raise ValueError("candidate boundary proximity inputs must be finite")

    forward = torch.nn.functional.normalize(view[:, :3], dim=-1, eps=1e-6)
    up_reference = torch.zeros_like(forward)
    up_reference[:, 1] = 1.0
    alternative = torch.zeros_like(forward)
    alternative[:, 2] = 1.0
    up_reference = torch.where(
        (forward * up_reference).sum(dim=-1, keepdim=True).abs() > 0.98,
        alternative,
        up_reference,
    )
    right = torch.nn.functional.normalize(
        torch.cross(forward, up_reference, dim=-1), dim=-1, eps=1e-6
    )
    up = torch.nn.functional.normalize(
        torch.cross(right, forward, dim=-1), dim=-1, eps=1e-6
    )

    centers = (aabbs[:, :3] + aabbs[:, 3:]) * 0.5
    extents = (aabbs[:, 3:] - aabbs[:, :3]).clamp_min(0.0) * 0.5
    delta = centers - camera
    z = (delta * forward).sum(dim=-1)
    x = (delta * right).sum(dim=-1)
    y = (delta * up).sum(dim=-1)
    rz = (extents * forward.abs()).sum(dim=-1)
    rx = (extents * right.abs()).sum(dim=-1)
    ry = (extents * up.abs()).sum(dim=-1)
    front_extent = z + rz
    if bool((front_extent < float(near) - 1e-5).any()):
        raise ValueError(
            "candidate AABB lies outside the registered near-plane candidate contract"
        )
    front_extent = front_extent.clamp_min(float(near))
    qx = front_extent * view[:, 3].clamp_min(1e-4)
    qy = front_extent * view[:, 4].clamp_min(1e-4)
    slack_x = (qx - x.abs() + rx) / qx.clamp_min(1e-6)
    slack_y = (qy - y.abs() + ry) / qy.clamp_min(1e-6)
    slack_z = (front_extent - float(near)) / front_extent.clamp_min(1e-6)
    slack = torch.minimum(torch.minimum(slack_x, slack_y), slack_z).clamp(0.0, 1.0)
    proximity = torch.sigmoid(
        (float(boundary_threshold) - slack) / float(temperature)
    )
    if not bool(torch.isfinite(proximity).all()):
        raise FloatingPointError("candidate boundary proximity is non-finite")
    return proximity.detach()


def _set_refinement_scope(
    model: BoundedRelationSurvivalMomentModel,
    scope: str,
    *,
    initialized: bool,
) -> dict[str, Any]:
    """Freeze offline state for an initialized runtime-query refinement."""
    if str(scope) not in {
        "all",
        "runtime_visibility",
        "visibility_head",
        "view_residual",
        "boundary_opportunity",
        "boundary_tail_residual",
        "viewcell_extreme_visibility",
        "viewcell_region_conditioned_visibility",
        "viewcell_region_tail",
        "dual_probe_rescue",
    }:
        raise ValueError(
            "refinement scope must be all, runtime_visibility, visibility_head, "
            "view_residual, boundary_opportunity, boundary_tail_residual, "
            "viewcell_extreme_visibility, viewcell_region_conditioned_visibility, "
            "viewcell_region_tail, or dual_probe_rescue"
        )
    if str(scope) == "all":
        return {
            "scope": "all",
            "trainableParameterCount": int(
                sum(value.numel() for value in model.parameters() if value.requires_grad)
            ),
        }
    if not initialized:
        raise ValueError(f"{scope} refinement requires an initial checkpoint")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if str(scope) == "view_residual":
        if (
            model.view_residual_geometry_projection is None
            or model.view_residual_dynamic_projection is None
            or model.view_residual_head is None
        ):
            raise ValueError(
                "view_residual refinement requires an enabled view residual head"
            )
        runtime_modules = (
            model.view_residual_geometry_projection,
            model.view_residual_dynamic_projection,
            model.view_residual_head,
        )
    elif str(scope) == "boundary_opportunity":
        if (
            model.boundary_opportunity_geometry_projection is None
            or model.boundary_opportunity_region_projection is None
            or model.boundary_opportunity_head is None
        ):
            raise ValueError(
                "boundary_opportunity refinement requires an enabled opportunity head"
            )
        runtime_modules = (
            model.boundary_opportunity_geometry_projection,
            model.boundary_opportunity_region_projection,
            model.boundary_opportunity_head,
        )
    elif str(scope) == "boundary_tail_residual":
        if (
            model.boundary_tail_residual_geometry_projection is None
            or model.boundary_tail_residual_region_projection is None
            or model.boundary_tail_residual_head is None
        ):
            raise ValueError(
                "boundary_tail_residual refinement requires an enabled residual head"
            )
        runtime_modules = (
            model.boundary_tail_residual_geometry_projection,
            model.boundary_tail_residual_region_projection,
            model.boundary_tail_residual_head,
        )
        if model.boundary_tail_residual_region_shortcut is not None:
            runtime_modules = runtime_modules + (
                model.boundary_tail_residual_region_shortcut,
            )
    elif str(scope) == "viewcell_extreme_visibility":
        if (
            model.viewcell_extreme_visibility_projection is None
            or model.viewcell_extreme_visibility_head is None
        ):
            raise ValueError(
                "viewcell_extreme_visibility refinement requires an enabled branch"
            )
        runtime_modules = (
            model.viewcell_extreme_visibility_projection,
            model.viewcell_extreme_visibility_head,
            model.visibility_head,
        )
    elif str(scope) in {
        "viewcell_region_conditioned_visibility",
        "viewcell_region_tail",
    }:
        if (
            model.viewcell_region_conditioned_visibility_region_projection is None
            or model.viewcell_region_conditioned_visibility_hidden_projection is None
            or model.viewcell_region_conditioned_visibility_head is None
        ):
            raise ValueError(
                "viewcell_region_conditioned_visibility refinement requires an enabled branch"
            )
        runtime_modules = (
            model.viewcell_region_conditioned_visibility_region_projection,
            model.viewcell_region_conditioned_visibility_hidden_projection,
            model.viewcell_region_conditioned_visibility_head,
        )
        if str(scope) == "viewcell_region_conditioned_visibility":
            runtime_modules = runtime_modules + (model.visibility_head,)
    elif str(scope) == "dual_probe_rescue":
        if model.dual_probe_rescue_attenuation_head is None:
            raise ValueError(
                "dual_probe_rescue refinement requires an enabled dual-probe attenuation head"
            )
        runtime_modules = (model.dual_probe_rescue_attenuation_head,)
    elif str(scope) == "runtime_visibility":
        runtime_modules = (
            model.moment_query,
            model.relation_condition_head,
            model.boundary_summary_head,
            model.direction_basis_head,
            model.shared_trunk,
            model.visibility_head,
        )
    else:
        runtime_modules = (model.visibility_head,)
    for module in runtime_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable_names = [
        name for name, value in model.named_parameters() if value.requires_grad
    ]
    if str(scope) == "dual_probe_rescue":
        if not trainable_names or not all(
            name.startswith("dual_probe_rescue_attenuation_head.")
            for name in trainable_names
        ):
            raise RuntimeError(
                "dual_probe_rescue refinement must leave only the attenuation head trainable"
            )
    result = {
        "scope": str(scope),
        "trainableParameterCount": int(
            sum(value.numel() for value in model.parameters() if value.requires_grad)
        ),
        "trainablePrefixes": sorted({name.split(".", 1)[0] for name in trainable_names}),
        "offlineSurvivalTableFrozen": True,
        "runtimeAssetShapeChanged": False,
    }
    if str(scope) == "dual_probe_rescue":
        result.update(
            {
                "baseParametersTrainable": False,
                "fixedPosteriorFrozen": True,
            }
        )
    return result


def _dual_probe_rescue_batch_loss(
    aux: Mapping[str, Any],
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    pose_offsets: torch.Tensor,
    *,
    coverage_weight: float,
    primary_keep_weight: float,
    coverage_keep_weight: float,
    negative_decay_weight: float,
    high_negative_decay_weight: float,
    visible_weight_epsilon: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the train-only attenuation objective to one pose-set batch."""
    required = (
        "dual_probe_raw_features",
        "dual_probe_rescue_base_logit",
        "dual_probe_primary_gate",
        "dual_probe_coverage_gate",
        "dual_probe_primary_attenuation",
        "dual_probe_coverage_attenuation",
        "dual_probe_fixed_residual",
        "dual_probe_residual",
    )
    missing = [key for key in required if key not in aux]
    if missing:
        raise ValueError(f"dual-probe rescue batch is missing aux keys: {missing}")
    count = int(target.numel())
    offsets = torch.as_tensor(pose_offsets, device=target.device).reshape(-1)
    if offsets.numel() < 2 or offsets[0].item() != 0 or offsets[-1].item() != count:
        raise ValueError("dual-probe rescue pose_offsets must cover the candidate batch")
    if not bool(torch.equal(offsets, offsets.round())):
        raise ValueError("dual-probe rescue pose_offsets must contain integer boundaries")
    offsets = offsets.to(dtype=torch.long)
    if bool((offsets[1:] <= offsets[:-1]).any()):
        raise ValueError("dual-probe rescue pose_offsets must contain non-empty poses")
    raw = aux["dual_probe_raw_features"]
    if not isinstance(raw, torch.Tensor) or raw.shape != (count, DUAL_PROBE_RAW_QUERY_DIM):
        raise ValueError("dual-probe rescue raw features must be candidate-aligned 108D rows")
    if not bool(torch.isfinite(raw).all()):
        raise FloatingPointError("dual-probe rescue raw features are non-finite")
    loss, diagnostics = dual_probe_rescue_attenuation_loss(
        torch.cat(
            [
                aux["dual_probe_primary_attenuation"],
                aux["dual_probe_coverage_attenuation"],
            ],
            dim=-1,
        ),
        aux["dual_probe_rescue_base_logit"],
        aux["dual_probe_primary_gate"],
        aux["dual_probe_coverage_gate"],
        target,
        visible_weights,
        fixed_residual=aux["dual_probe_fixed_residual"],
        residual=aux["dual_probe_residual"],
        coverage_weight=coverage_weight,
        primary_keep_weight=primary_keep_weight,
        coverage_keep_weight=coverage_keep_weight,
        negative_decay_weight=negative_decay_weight,
        high_negative_decay_weight=high_negative_decay_weight,
        visible_weight_epsilon=visible_weight_epsilon,
    )
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "dualProbeRescuePoseCount": int(offsets.numel() - 1),
            "dualProbeRescueCandidateCount": count,
            "dualProbeRescuePoseOffsetsUsed": True,
        }
    )
    return loss, diagnostics


def _dual_probe_rescue_objective_groups(
    loss: torch.Tensor,
    *,
    loss_weight: float,
) -> dict[str, torch.Tensor]:
    """Route only attenuation loss gradients during dual-probe refinement."""
    if not math.isfinite(float(loss_weight)) or float(loss_weight) < 0.0:
        raise ValueError("dual-probe rescue loss weight must be finite and non-negative")
    total = float(loss_weight) * loss
    zero = total * 0.0
    return {
        "safety": total,
        "relation": zero,
        "schedule": zero,
        "efficiency": zero,
        "total": total,
    }


def _instance_calibration_blend(
    completed_steps: int,
    total_steps: int,
    *,
    warmup_fraction: float,
    ramp_fraction: float,
) -> float:
    if total_steps <= 0 or completed_steps < 0:
        raise ValueError("calibration schedule steps must be non-negative with a positive total")
    if not 0.0 <= warmup_fraction < 1.0 or not 0.0 < ramp_fraction <= 1.0:
        raise ValueError("calibration warmup/ramp fractions are invalid")
    warmup_steps = int(math.ceil(float(total_steps) * float(warmup_fraction)))
    ramp_steps = max(1, int(math.ceil(float(total_steps) * float(ramp_fraction))))
    if int(completed_steps) < warmup_steps:
        return 0.0
    return float(min(1.0, (int(completed_steps) - warmup_steps + 1) / ramp_steps))


def _boundary_opportunity_scale(
    completed_steps: int,
    total_steps: int,
    *,
    ramp_fraction: float,
) -> float:
    if total_steps <= 0 or completed_steps < 0:
        raise ValueError(
            "boundary opportunity schedule requires non-negative steps"
        )
    if not 0.0 <= float(ramp_fraction) <= 1.0:
        raise ValueError("boundary opportunity ramp fraction must lie in [0, 1]")
    if float(ramp_fraction) == 0.0:
        return 1.0
    ramp_steps = max(1, int(math.ceil(total_steps * float(ramp_fraction))))
    return float(min(1.0, int(completed_steps) / ramp_steps))


def _boundary_opportunity_pose_pool(
    dataset: PoseCSRDataset,
    split: Any,
    *,
    rare_threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if str(getattr(split, "split_name", "")).lower() == "test":
        raise ValueError("boundary opportunity pose mining cannot read test")
    if not dataset.has_subpose_robust_labels:
        raise ValueError("boundary opportunity pose mining requires subpose labels")
    threshold = float(rare_threshold)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("boundary opportunity rare threshold must lie in (0, 1]")
    selected: list[int] = []
    rare_reference_count = 0
    rare_pose_count = 0
    for pose_index in np.asarray(split.pose_indices, dtype=np.int64):
        visible_ids, _visible_weights = dataset.visible_slice(int(pose_index))
        candidate_ids = dataset.frustum_slice(int(pose_index))
        hit_counts = np.asarray(
            dataset.visible_hit_count_slice(int(pose_index)), dtype=np.float64
        )
        subpose_count = dataset.subpose_count(int(pose_index))
        if subpose_count <= 0:
            raise ValueError("train pose has no successful subpose supervision")
        rare = (hit_counts > 0.0) & (
            hit_counts / float(subpose_count) <= threshold
        )
        count = int(rare.sum())
        has_negative = bool(
            np.setdiff1d(candidate_ids, visible_ids, assume_unique=False).size > 0
        )
        if visible_ids.size > 0 and has_negative:
            selected.append(int(pose_index))
        if count > 0:
            rare_pose_count += 1
        rare_reference_count += count
    if not selected:
        raise ValueError("train split has no mixed positive/negative opportunity poses")
    return np.asarray(selected, dtype=np.int64), {
        "sourceSplit": str(getattr(split, "split_name", "train")),
        "rareThreshold": threshold,
        "poseCount": len(selected),
        "rarePositivePoseCount": rare_pose_count,
        "rarePositiveReferenceCount": rare_reference_count,
        "selection": "all train poses with at least one visible and one invisible candidate",
        "testRead": False,
    }


def _refinement_pose_sampling(
    scope: str,
    mixed_pose_indices: np.ndarray | None,
    hard_pose_indices: np.ndarray | None,
    hard_pose_fraction: float,
) -> tuple[np.ndarray | None, float]:
    """Resolve the train-pose mixture for the two tail-only refinements."""
    if scope == "boundary_opportunity":
        return mixed_pose_indices, 1.0
    if scope == "boundary_tail_residual":
        if hard_pose_indices is not None and float(hard_pose_fraction) > 0.0:
            return hard_pose_indices, float(hard_pose_fraction)
        return mixed_pose_indices, 1.0
    return hard_pose_indices, float(hard_pose_fraction)


def _boundary_tail_selection_logits(
    selection_source: str,
    current_final_logits: torch.Tensor,
) -> torch.Tensor | None:
    """Return the detached-selection source consumed by the tail loss.

    The loss itself owns the detach so this helper preserves one explicit
    contract: frozen-base selection is represented by ``None`` and dynamic
    mining receives the current final score tensor.
    """
    if selection_source == "frozen_base":
        return None
    if selection_source == "current_final":
        return current_final_logits
    raise ValueError(f"unknown boundary-tail selection source: {selection_source}")


def _validate_extreme_tail_selection_contract(args: argparse.Namespace) -> None:
    """Validate the train-only score source used to freeze tail membership."""
    if args.tail_selection_source == "current":
        return
    if args.tail_selection_source != "initial_visibility":
        raise ValueError(
            f"unknown extreme-tail selection source: {args.tail_selection_source}"
        )
    if (
        args.initial_checkpoint is None
        or args.refinement_scope not in {
            "viewcell_region_conditioned_visibility",
            "viewcell_region_tail",
        }
        or not args.viewcell_region_conditioned_visibility
    ):
        raise ValueError(
            "initial_visibility tail selection requires an initial checkpoint "
            "and a region-conditioned visibility refinement"
        )
    checkpoint_name = Path(args.initial_checkpoint).name
    fixed_epoch_checkpoint = (
        checkpoint_name == "last.pt"
        or (
            (
                checkpoint_name.startswith("epoch_")
                or checkpoint_name.startswith("checkpoint_epoch_")
            )
            and checkpoint_name.endswith(".pt")
        )
    )
    if not fixed_epoch_checkpoint:
        raise ValueError(
            "initial_visibility tail selection requires a fixed-epoch checkpoint "
            "(last.pt or an epoch snapshot), not a calibration-selected checkpoint"
        )


def _extreme_tail_selection_logits(
    selection_source: str,
    current_logits: torch.Tensor,
    query_features: torch.Tensor,
    initial_visibility_weight: torch.Tensor | None,
    initial_visibility_bias: torch.Tensor | None,
) -> torch.Tensor:
    """Return scores used only to choose fixed or dynamic tail members."""
    if selection_source == "current":
        return current_logits
    if selection_source != "initial_visibility":
        raise ValueError(
            f"unknown extreme-tail selection source: {selection_source}"
        )
    if initial_visibility_weight is None or initial_visibility_bias is None:
        raise ValueError("initial visibility selector parameters are missing")
    if query_features.ndim != 2 or initial_visibility_weight.ndim != 2:
        raise ValueError("initial visibility selector tensors have invalid rank")
    if query_features.shape[1] != initial_visibility_weight.shape[1]:
        raise ValueError("initial visibility selector feature dimension is invalid")
    return F.linear(
        query_features.detach(),
        initial_visibility_weight.detach(),
        initial_visibility_bias.detach(),
    )


def _runtime_features(
    model: BoundedRelationSurvivalMomentModel,
    geometry: torch.Tensor,
    relation: Mapping[str, torch.Tensor],
    local_ids: torch.Tensor,
    structural_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    diagnostics = model.offline_encode_survival(
        geometry,
        relation,
        local_ids,
        structural_ids,
        return_diagnostics=True,
    )
    if not isinstance(diagnostics, Mapping):
        raise TypeError("offline encoder did not return calibration diagnostics")
    coefficients = diagnostics.get("survival_coefficients")
    if not isinstance(coefficients, torch.Tensor) or coefficients.shape != (
        geometry.shape[0], *SURVIVAL_SHAPE
    ):
        raise ValueError("offline encoder did not produce [N, 4, 7]")
    runtime = torch.cat([geometry, coefficients.reshape(geometry.shape[0], -1)], dim=-1)
    if runtime.shape != (geometry.shape[0], RUNTIME_FEATURE_DIM) or not bool(torch.isfinite(runtime).all()):
        raise FloatingPointError("v4 fixed runtime table is invalid")
    return runtime, coefficients, dict(diagnostics)


def _weighted_quantile_numpy(
    values: np.ndarray,
    weights: np.ndarray,
    fraction: float,
) -> float:
    scores = np.asarray(values, dtype=np.float64).reshape(-1)
    mass = np.asarray(weights, dtype=np.float64).reshape(-1)
    if scores.size == 0 or mass.size != scores.size or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("weighted quantile inputs are invalid")
    if not bool(np.isfinite(scores).all() and np.isfinite(mass).all()) or bool(
        (mass < 0.0).any()
    ):
        raise ValueError("weighted quantile inputs must be finite and non-negative")
    if float(mass.sum()) <= 0.0:
        mass = np.ones_like(mass)
    order = np.argsort(scores, kind="stable")
    ordered_mass = mass[order]
    target = float(fraction) * float(ordered_mass.sum())
    index = int(
        min(
            order.size - 1,
            np.searchsorted(np.cumsum(ordered_mass), target, side="left"),
        )
    )
    return float(scores[order[index]])


def _select_tail_hard_pose_pool(
    pose_indices: np.ndarray,
    risks: np.ndarray,
    pool_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    poses = np.asarray(pose_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(risks, dtype=np.float64).reshape(-1)
    if (
        poses.size == 0
        or values.size != poses.size
        or not bool(np.isfinite(values).all())
        or not 0.0 < float(pool_fraction) <= 1.0
    ):
        raise ValueError("tail hard-pose pool inputs are invalid")
    count = min(
        poses.size,
        max(1, int(math.ceil(float(pool_fraction) * poses.size))),
    )
    order = np.argsort(-values, kind="stable")
    selected = np.sort(poses[order[:count]])
    return selected, {
        "poolPoseCount": int(selected.size),
        "riskMinimum": float(values.min()),
        "riskMedian": float(np.median(values)),
        "riskP95": float(np.quantile(values, 0.95)),
        "riskMaximum": float(values.max()),
    }


def _select_weighted_safety_frontier_members(
    instance_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    visible_weights: np.ndarray,
    *,
    positive_mass_fraction: float,
    positive_count_cap: int,
    negative_top_fraction: float,
    minimum_negative_count: int,
    maximum_negative_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Freeze the important-positive/high-negative frontier for one train pose."""
    ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    targets = np.asarray(labels, dtype=np.float32).reshape(-1) > 0.5
    weights = np.asarray(visible_weights, dtype=np.float64).reshape(-1)
    if (
        ids.size == 0
        or values.shape != ids.shape
        or targets.shape != ids.shape
        or weights.shape != ids.shape
        or np.unique(ids).size != ids.size
        or not bool(np.isfinite(values).all() and np.isfinite(weights).all())
        or bool((weights < 0.0).any())
        or not 0.0 < float(positive_mass_fraction) <= 1.0
        or int(positive_count_cap) <= 0
        or not 0.0 < float(negative_top_fraction) <= 1.0
        or int(minimum_negative_count) <= 0
        or int(maximum_negative_count) < int(minimum_negative_count)
    ):
        raise ValueError("weighted safety-frontier inputs are invalid")
    positive_rows = np.flatnonzero(targets)
    negative_rows = np.flatnonzero(~targets)
    if positive_rows.size == 0 or negative_rows.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            {
                "positiveCount": 0,
                "negativeCount": 0,
                "overlapPositiveCount": 0,
            },
        )

    negative_count = min(
        int(maximum_negative_count),
        negative_rows.size,
        max(
            int(minimum_negative_count),
            int(math.ceil(float(negative_top_fraction) * negative_rows.size)),
        ),
    )
    negative_order = np.argsort(-values[negative_rows], kind="stable")
    selected_negative_rows = negative_rows[negative_order[:negative_count]]
    negative_boundary = float(values[selected_negative_rows].min())

    positive_order = np.argsort(values[positive_rows], kind="stable")
    ordered_positive_rows = positive_rows[positive_order]
    ordered_mass = weights[ordered_positive_rows].copy()
    if float(ordered_mass.sum()) <= 0.0:
        ordered_mass.fill(1.0)
    target_mass = float(positive_mass_fraction) * float(ordered_mass.sum())
    weighted_count = min(
        ordered_positive_rows.size,
        int(np.searchsorted(np.cumsum(ordered_mass), target_mass, side="left")) + 1,
    )
    overlap_count = int(
        np.searchsorted(
            values[ordered_positive_rows],
            negative_boundary,
            side="right",
        )
    )
    positive_count = min(
        int(positive_count_cap),
        ordered_positive_rows.size,
        max(1, weighted_count, overlap_count),
    )
    selected_positive_rows = ordered_positive_rows[:positive_count]
    selected_positive_ids = ids[selected_positive_rows].astype(np.int64, copy=True)
    selected_negative_ids = ids[selected_negative_rows].astype(np.int64, copy=True)
    if np.intersect1d(selected_positive_ids, selected_negative_ids).size:
        raise ValueError("weighted safety frontier contains overlapping class members")
    return selected_positive_ids, selected_negative_ids, {
        "positiveCount": int(selected_positive_ids.size),
        "negativeCount": int(selected_negative_ids.size),
        "weightedTailPositiveCount": int(weighted_count),
        "overlapPositiveCount": int(overlap_count),
        "negativeBoundaryLogit": negative_boundary,
        "positiveMinimumLogit": float(values[selected_positive_rows].min()),
        "positiveMaximumLogit": float(values[selected_positive_rows].max()),
    }


@torch.no_grad()
def _mine_initial_tail_hard_poses(
    model: BoundedRelationSurvivalMomentModel,
    train_split: Any,
    world_aabbs: np.ndarray,
    runtime_features: torch.Tensor,
    device: torch.device,
    *,
    pool_fraction: float,
    seed: int,
    poses_per_batch: int = 8,
    build_fixed_frontier: bool = False,
    frontier_positive_mass_fraction: float = 0.005,
    frontier_positive_count_cap: int = 1024,
    frontier_negative_top_fraction: float = 0.04,
    frontier_minimum_negative_count: int = 64,
    frontier_maximum_negative_count: int = 256,
) -> tuple[
    np.ndarray,
    dict[str, Any],
    dict[int, tuple[torch.Tensor, torch.Tensor]],
]:
    """Mine difficult train-only view-cells from the initialized checkpoint."""
    if str(getattr(train_split, "split_name", "")) != "train":
        raise ValueError("tail hard-pose/frontier mining is restricted to train split")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(int(seed))
    pose_rows: list[int] = []
    risks: list[float] = []
    positive_boundaries: list[float] = []
    negative_boundaries: list[float] = []
    all_positive_scores: list[np.ndarray] = []
    all_positive_weights: list[np.ndarray] = []
    frontier_by_pose: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    frontier_positive_counts: list[int] = []
    frontier_negative_counts: list[int] = []
    frontier_overlap_counts: list[int] = []
    for pose_indices in train_split.pose_set_batches(
        max(1, int(poses_per_batch)),
        rng,
        max_steps=None,
        include_empty=False,
    ):
        batch = train_split.build_pose_set_batch(
            pose_indices,
            world_aabbs,
            rng,
            max_candidates_per_pose=0,
            allow_candidate_visible_union=False,
            include_empty=False,
        )
        if batch["instance"].size == 0:
            continue
        instance_ids = torch.from_numpy(batch["instance"]).to(device)
        logits, _aux = model.compute_logits_with_aux(
            torch.from_numpy(batch["camera"]).to(device),
            torch.from_numpy(batch["camera_view"]).to(device),
            torch.from_numpy(batch["candidate_camera_world"]).to(device),
            instance_ids,
            runtime_features=runtime_features,
            query_center_world=torch.from_numpy(batch["query_center_world"]).to(device),
            viewcell_radius_m=torch.from_numpy(batch["viewcell_radius_m"]).to(device),
            pose_offsets=torch.from_numpy(batch["pose_offsets"]).to(device),
        )
        scores = logits.detach().float().reshape(-1).cpu().numpy()
        labels = np.asarray(batch["target"], dtype=np.float32).reshape(-1)
        weights = np.asarray(batch["visible_weights"], dtype=np.float32).reshape(-1)
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64).reshape(-1)
        built_poses = np.asarray(batch["pose_indices"], dtype=np.int64).reshape(-1)
        for local_index, pose_index in enumerate(built_poses.tolist()):
            start = int(offsets[local_index])
            end = int(offsets[local_index + 1])
            local_scores = scores[start:end]
            local_labels = labels[start:end] > 0.5
            if not bool(local_labels.any()) or not bool((~local_labels).any()):
                continue
            positive_boundary = _weighted_quantile_numpy(
                local_scores[local_labels],
                weights[start:end][local_labels],
                0.01,
            )
            positive_reference = _weighted_quantile_numpy(
                local_scores[local_labels],
                weights[start:end][local_labels],
                0.05,
            )
            negative_boundary = float(np.quantile(local_scores[~local_labels], 0.995))
            if build_fixed_frontier:
                positive_ids, negative_ids, frontier_meta = (
                    _select_weighted_safety_frontier_members(
                        np.asarray(batch["instance"], dtype=np.int64)[start:end],
                        local_scores,
                        local_labels,
                        weights[start:end],
                        positive_mass_fraction=frontier_positive_mass_fraction,
                        positive_count_cap=frontier_positive_count_cap,
                        negative_top_fraction=frontier_negative_top_fraction,
                        minimum_negative_count=frontier_minimum_negative_count,
                        maximum_negative_count=frontier_maximum_negative_count,
                    )
                )
                if positive_ids.size > 0 and negative_ids.size > 0:
                    frontier_by_pose[int(pose_index)] = (
                        torch.from_numpy(positive_ids),
                        torch.from_numpy(negative_ids),
                    )
                    frontier_positive_counts.append(
                        int(frontier_meta["positiveCount"])
                    )
                    frontier_negative_counts.append(
                        int(frontier_meta["negativeCount"])
                    )
                    frontier_overlap_counts.append(
                        int(frontier_meta["overlapPositiveCount"])
                    )
            all_positive_scores.append(local_scores[local_labels].astype(np.float64, copy=True))
            all_positive_weights.append(
                weights[start:end][local_labels].astype(np.float64, copy=True)
            )
            robust_scale = max(
                float(np.quantile(local_scores, 0.90) - np.quantile(local_scores, 0.10)),
                1e-3,
            )
            overlap = (negative_boundary - positive_boundary) / robust_scale
            positive_spread = max(
                0.0,
                (positive_reference - positive_boundary) / robust_scale,
            )
            pose_rows.append(int(pose_index))
            risks.append(float(overlap + 0.25 * positive_spread))
            positive_boundaries.append(positive_boundary)
            negative_boundaries.append(negative_boundary)
    if was_training:
        model.train()
    selected, meta = _select_tail_hard_pose_pool(
        np.asarray(pose_rows, dtype=np.int64),
        np.asarray(risks, dtype=np.float64),
        pool_fraction,
    )
    positive_scores = np.concatenate(all_positive_scores)
    positive_weights = np.concatenate(all_positive_weights)
    meta.update(
        {
            "sourceSplit": "train",
            "scoredPoseCount": int(len(pose_rows)),
            "positiveBoundaryMeanLogit": float(np.mean(positive_boundaries)),
            "negativeBoundaryMeanLogit": float(np.mean(negative_boundaries)),
            "globalWeightedPositiveQ01Logit": _weighted_quantile_numpy(
                positive_scores,
                positive_weights,
                0.01,
            ),
            "riskDefinition": "(negative q99.5 - weighted-positive q1) / robust logit range + 0.25 * positive q1-to-q5 spread",
            "modelState": "initial checkpoint before refinement",
            "testRead": False,
            "fixedFrontier": {
                "enabled": bool(build_fixed_frontier),
                "poseCount": int(len(frontier_by_pose)),
                "positiveCount": int(sum(frontier_positive_counts)),
                "negativeCount": int(sum(frontier_negative_counts)),
                "overlapPositiveCount": int(sum(frontier_overlap_counts)),
                "meanPositiveCountPerPose": (
                    float(np.mean(frontier_positive_counts))
                    if frontier_positive_counts
                    else 0.0
                ),
                "meanNegativeCountPerPose": (
                    float(np.mean(frontier_negative_counts))
                    if frontier_negative_counts
                    else 0.0
                ),
                "positiveMassFraction": float(
                    frontier_positive_mass_fraction
                ),
                "positiveCountCap": int(frontier_positive_count_cap),
                "negativeTopFraction": float(frontier_negative_top_fraction),
                "minimumNegativeCount": int(frontier_minimum_negative_count),
                "maximumNegativeCount": int(frontier_maximum_negative_count),
                "selection": (
                    "union of initial-checkpoint weighted low-positive tail and "
                    "positives not above the selected high-negative boundary"
                ),
                "sourceSplit": "train",
                "testRead": False,
            },
        }
    )
    return selected, meta, frontier_by_pose


def _recurrent_negative_priority_from_counts(
    negative_exposures: np.ndarray,
    hard_negative_counts: np.ndarray,
    positive_exposures: np.ndarray,
    *,
    baseline_tail_fraction: float,
    selected_instance_fraction: float,
    minimum_negative_exposures: int,
    minimum_hard_count: int,
    smoothing_strength: float = 16.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert train-only repeated hard-negative counts into a sparse prior."""
    negative = np.asarray(negative_exposures, dtype=np.int64).reshape(-1)
    hard = np.asarray(hard_negative_counts, dtype=np.int64).reshape(-1)
    positive = np.asarray(positive_exposures, dtype=np.int64).reshape(-1)
    if negative.size == 0 or hard.shape != negative.shape or positive.shape != negative.shape:
        raise ValueError("recurrent negative count arrays must be non-empty and aligned")
    if np.any(negative < 0) or np.any(hard < 0) or np.any(positive < 0):
        raise ValueError("recurrent negative counts must be non-negative")
    if np.any(hard > negative):
        raise ValueError("hard-negative counts cannot exceed negative exposures")
    if not 0.0 < float(baseline_tail_fraction) < 1.0:
        raise ValueError("baseline tail fraction must lie in (0, 1)")
    if not 0.0 < float(selected_instance_fraction) <= 1.0:
        raise ValueError("selected instance fraction must lie in (0, 1]")
    if int(minimum_negative_exposures) <= 0 or int(minimum_hard_count) <= 0:
        raise ValueError("recurrent negative minimum counts must be positive")
    if not math.isfinite(float(smoothing_strength)) or float(smoothing_strength) < 0.0:
        raise ValueError("recurrent negative smoothing strength must be non-negative")

    baseline = float(baseline_tail_fraction)
    posterior = (
        hard.astype(np.float64) + float(smoothing_strength) * baseline
    ) / (negative.astype(np.float64) + float(smoothing_strength))
    excess = np.maximum(0.0, (posterior - baseline) / (1.0 - baseline))
    exposed = negative[negative > 0]
    exposure_reference = float(np.quantile(exposed, 0.95)) if exposed.size else 1.0
    exposure_scale = np.clip(
        np.log1p(negative.astype(np.float64))
        / max(math.log1p(max(exposure_reference, 1.0)), 1e-12),
        0.0,
        1.0,
    )
    raw = excess * exposure_scale
    eligible = (
        (negative >= int(minimum_negative_exposures))
        & (hard >= int(minimum_hard_count))
        & (positive > 0)
        & (raw > 0.0)
    )
    eligible_ids = np.flatnonzero(eligible)
    priority = np.zeros_like(raw, dtype=np.float32)
    selected_ids = np.empty((0,), dtype=np.int64)
    if eligible_ids.size:
        selected_count = min(
            int(eligible_ids.size),
            max(
                1,
                int(
                    math.ceil(
                        float(selected_instance_fraction) * int(eligible_ids.size)
                    )
                ),
            )
        )
        order = np.argsort(-raw[eligible_ids], kind="stable")
        selected_ids = eligible_ids[order[:selected_count]]
        selected_values = raw[selected_ids]
        priority[selected_ids] = (
            selected_values / max(float(selected_values.mean()), 1e-12)
        ).astype(np.float32)
    return priority, {
        "eligibleInstanceCount": int(eligible_ids.size),
        "selectedInstanceCount": int(selected_ids.size),
        "selectedInstanceFraction": float(selected_instance_fraction),
        "baselineNegativeTailFraction": baseline,
        "minimumNegativeExposures": int(minimum_negative_exposures),
        "minimumHardNegativeCount": int(minimum_hard_count),
        "smoothingStrength": float(smoothing_strength),
        "negativeExposureP95": exposure_reference,
        "selectedPriorityMean": (
            float(priority[selected_ids].mean()) if selected_ids.size else 0.0
        ),
        "selectedPriorityMaximum": (
            float(priority[selected_ids].max()) if selected_ids.size else 0.0
        ),
        "testRead": False,
    }


@torch.no_grad()
def _mine_recurrent_hard_negative_prior(
    model: BoundedRelationSurvivalMomentModel,
    train_split: Any,
    world_aabbs: np.ndarray,
    runtime_features: torch.Tensor,
    device: torch.device,
    *,
    num_instances: int,
    negative_tail_fraction: float,
    selected_instance_fraction: float,
    minimum_negative_exposures: int,
    minimum_hard_count: int,
    smoothing_strength: float,
    seed: int,
    poses_per_batch: int = 8,
    maximum_hard_negatives_per_pose: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Mine recurrent high-score negatives from the initialized train split."""
    if str(getattr(train_split, "split_name", "train")).lower() == "test":
        raise ValueError("recurrent hard-negative mining cannot read test")
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(int(seed))
    negative_exposures = np.zeros((int(num_instances),), dtype=np.int64)
    hard_negative_counts = np.zeros((int(num_instances),), dtype=np.int64)
    positive_exposures = np.zeros((int(num_instances),), dtype=np.int64)
    scored_pose_count = 0
    for pose_indices in train_split.pose_set_batches(
        max(1, int(poses_per_batch)),
        rng,
        max_steps=None,
        include_empty=False,
    ):
        batch = train_split.build_pose_set_batch(
            pose_indices,
            world_aabbs,
            rng,
            max_candidates_per_pose=0,
            allow_candidate_visible_union=False,
            include_empty=False,
        )
        if batch["instance"].size == 0:
            continue
        instance_ids = torch.from_numpy(batch["instance"]).to(device)
        logits, _aux = model.compute_logits_with_aux(
            torch.from_numpy(batch["camera"]).to(device),
            torch.from_numpy(batch["camera_view"]).to(device),
            torch.from_numpy(batch["candidate_camera_world"]).to(device),
            instance_ids,
            runtime_features=runtime_features,
            query_center_world=torch.from_numpy(batch["query_center_world"]).to(device),
            viewcell_radius_m=torch.from_numpy(batch["viewcell_radius_m"]).to(device),
            pose_offsets=torch.from_numpy(batch["pose_offsets"]).to(device),
        )
        scores = logits.detach().float().reshape(-1).cpu().numpy()
        labels = np.asarray(batch["target"], dtype=np.float32).reshape(-1) > 0.5
        ids = np.asarray(batch["instance"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(batch["pose_offsets"], dtype=np.int64).reshape(-1)
        for start, end in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
            local_ids = ids[int(start) : int(end)]
            local_labels = labels[int(start) : int(end)]
            local_scores = scores[int(start) : int(end)]
            if bool(local_labels.any()):
                np.add.at(positive_exposures, local_ids[local_labels], 1)
            negative_mask = ~local_labels
            negative_ids = local_ids[negative_mask]
            if negative_ids.size == 0:
                continue
            np.add.at(negative_exposures, negative_ids, 1)
            hard_count = min(
                int(negative_ids.size),
                int(maximum_hard_negatives_per_pose),
                max(
                    1,
                    int(
                        math.ceil(
                            float(negative_tail_fraction) * int(negative_ids.size)
                        )
                    ),
                ),
            )
            order = np.argsort(-local_scores[negative_mask], kind="stable")
            np.add.at(hard_negative_counts, negative_ids[order[:hard_count]], 1)
            scored_pose_count += 1
    if was_training:
        model.train()
    priority, meta = _recurrent_negative_priority_from_counts(
        negative_exposures,
        hard_negative_counts,
        positive_exposures,
        baseline_tail_fraction=negative_tail_fraction,
        selected_instance_fraction=selected_instance_fraction,
        minimum_negative_exposures=minimum_negative_exposures,
        minimum_hard_count=minimum_hard_count,
        smoothing_strength=smoothing_strength,
    )
    meta.update(
        {
            "sourceSplit": "train",
            "scoredPoseCount": int(scored_pose_count),
            "modelState": "initial checkpoint before refinement",
            "runtimeCost": "none; train-only detached priority",
            "testRead": False,
        }
    )
    return priority, meta


def _gradients(
    objective: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor | None, ...]:
    if not objective.requires_grad:
        return tuple(None for _ in parameters)
    return torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def _assign_projected_gradients(
    parameters: Sequence[torch.nn.Parameter],
    groups: Mapping[str, Sequence[torch.Tensor | None]],
) -> None:
    for index, parameter in enumerate(parameters):
        values = [group[index] for group in groups.values() if group[index] is not None]
        parameter.grad = None if not values else sum(values[1:], values[0].clone())


def _float_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                result[key] = float(value.detach().cpu())
        elif isinstance(value, (float, int, np.generic)):
            result[key] = float(value)
    return result


def _diagnostic_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
            float(row.get("aggregateWeightedRecall") or -1.0),
            float(row.get("agg_balanced_accuracy") or 0.0),
            float(row.get("agg_useful_cull") or 0.0),
            float(row.get("agg_precision") or 0.0),
            -float(row.get("avg_pred_count") or 0.0),
        ),
    )


def _calibration_workpoints(
    calibration_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Choose both workpoints from one checkpoint's calibration rows only."""
    selected = select_aggregate_weighted_cull_workpoint(
        calibration_rows,
        target_weighted_recall=CALIBRATION_FLOOR,
        minimum_lower_confidence_bound=CALIBRATION_FLOOR,
    )
    diagnostic = _diagnostic_row(calibration_rows)
    frozen = selected if selected is not None else diagnostic
    return selected, diagnostic, frozen


def _weighted_recall_safety_gate(row: Mapping[str, Any] | None) -> bool:
    """Require both weighted-recall point estimate and LCB on one split."""
    if row is None:
        return False
    recall = row.get("aggregateWeightedRecall", row.get("agg_weighted_recall"))
    lower = row.get(
        "aggregateWeightedRecallLowerConfidenceBound",
        row.get("weighted_recall_lower_confidence_bound"),
    )
    if recall is None or lower is None:
        return False
    return float(recall) > CALIBRATION_FLOOR and float(lower) > CALIBRATION_FLOOR


def _update_safety_boundary_ema(
    current_boundary: torch.Tensor,
    previous_ema: torch.Tensor | None,
    *,
    decay: float,
) -> torch.Tensor:
    """Update the detached train-only positive boundary state."""
    if not 0.0 < float(decay) < 1.0:
        raise ValueError("safety boundary EMA decay must lie in (0, 1)")
    current = current_boundary.float().reshape(()).detach()
    if not bool(torch.isfinite(current)):
        raise ValueError("current safety boundary must be finite")
    if previous_ema is None:
        return current
    previous = previous_ema.to(device=current.device, dtype=current.dtype).reshape(()).detach()
    if not bool(torch.isfinite(previous)):
        raise ValueError("previous safety boundary EMA must be finite")
    return (
        float(decay) * previous + (1.0 - float(decay)) * current
    ).detach()


def _v4_objective_groups(
    safety: torch.Tensor,
    survival: torch.Tensor,
    relation_consistency: torch.Tensor,
    utility: torch.Tensor,
    download: torch.Tensor,
    regularization: torch.Tensor,
    instance_calibration_regularization: torch.Tensor,
    efficiency: torch.Tensor,
    *,
    survival_weight: float,
    relation_consistency_weight: float,
    utility_weight: float,
    download_weight: float,
    regularization_weight: float,
    instance_calibration_regularization_weight: float,
) -> dict[str, torch.Tensor]:
    """Build the four logged groups and the exact scalar optimized by the step."""
    relation = (
        float(survival_weight) * survival
        + float(relation_consistency_weight) * relation_consistency
        + float(regularization_weight) * regularization
        + float(instance_calibration_regularization_weight)
        * instance_calibration_regularization
    )
    schedule = float(utility_weight) * utility + float(download_weight) * download
    return {
        "safety": safety,
        "relation": relation,
        "schedule": schedule,
        "efficiency": efficiency,
        "total": safety + relation + schedule + efficiency,
    }


def _boundary_tail_refinement_objective_groups(
    tail_safety: torch.Tensor,
    paired_safety: torch.Tensor,
    negative_efficiency: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Route cross-view pair gradients while refining the signed tail head."""
    zero = tail_safety * 0.0
    safety = tail_safety + paired_safety
    return {
        "safety": safety,
        "relation": zero,
        "schedule": zero,
        "efficiency": negative_efficiency,
        "total": safety + negative_efficiency,
    }


def _operating_threshold_metrics(
    train_seed: int,
    global_step: int,
) -> tuple[tuple[float, ...], dict[str, float]]:
    thresholds = sample_train_operating_thresholds(
        train_seed,
        global_step,
        split="train",
    )
    metrics = {
        "operatingThresholdCount": float(len(thresholds)),
        **{
            f"operatingThreshold{index}": float(value)
            for index, value in enumerate(thresholds)
        },
    }
    return thresholds, metrics


def _evaluation_batch_limits(max_poses: int, poses_per_batch: int) -> tuple[int, int | None]:
    """Choose an exact diagnostic pose cap without changing full evaluation."""
    batch_size = int(poses_per_batch)
    pose_limit = int(max_poses)
    if batch_size <= 0 or pose_limit < 0:
        raise ValueError("poses_per_batch must be positive and max_poses non-negative")
    if pose_limit == 0:
        # Zero is the registered full-split value, not one batch.  Passing
        # ``None`` lets PoseCSRSplit iterate every pose deterministically.
        return batch_size, None
    exact_batch_size = math.gcd(batch_size, pose_limit)
    return exact_batch_size, pose_limit // exact_batch_size


def _evaluate(
    model: BoundedRelationSurvivalMomentModel,
    split: Any,
    runtime: torch.Tensor,
    world_aabbs: np.ndarray,
    device: torch.device,
    *,
    thresholds: np.ndarray,
    seed: int,
    poses_per_batch: int,
    max_poses: int,
    bootstrap_replicates: int,
    instance_to_glb: np.ndarray | None = None,
    glb_bytes: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    evaluation_batch_size, max_steps = _evaluation_batch_limits(max_poses, poses_per_batch)
    return evaluate_thresholds(
        model,
        split,
        runtime,
        world_aabbs,
        device,
        poses_per_batch=evaluation_batch_size,
        max_steps=max_steps,
        max_candidates_per_pose=0,
        seed=seed,
        thresholds=thresholds,
        collect_pose_stats=True,
        allow_candidate_visible_union=False,
        bootstrap_replicates=bootstrap_replicates,
        instance_to_glb=instance_to_glb,
        glb_bytes=glb_bytes,
    )


def _checkpoint(
    model: BoundedRelationSurvivalMomentModel,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    protocol: Mapping[str, Any],
    relation_provenance: Mapping[str, Any],
    geometry_meta: Mapping[str, Any],
    coefficients: torch.Tensor,
    coefficient_diagnostics: Mapping[str, Any],
    calibration_reliability_meta: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    validation: Mapping[str, Any] | None,
    best: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prior = coefficient_diagnostics.get("survival_prior_coefficients")
    residual = coefficient_diagnostics.get("instance_calibration_applied_residual")
    if not isinstance(prior, torch.Tensor) or not isinstance(residual, torch.Tensor):
        raise ValueError("checkpoint requires prior and applied instance calibration tensors")
    if prior.shape != coefficients.shape or residual.shape != coefficients.shape:
        raise ValueError("checkpoint calibration tensors disagree with fused coefficients")
    if not torch.allclose(coefficients, prior + residual, rtol=1e-5, atol=1e-6):
        raise ValueError("checkpoint fused coefficients do not equal prior plus calibration residual")
    return {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4",
        "runtimeSchema": MODEL_SCHEMA,
        "experimentName": args.experiment_name,
        "epoch": int(epoch),
        "globalStep": int(global_step),
        "modelConfig": model.config,
        "modelState": model.state_dict(),
        "instanceSurvivalCoefficients": coefficients.detach().cpu().half(),
        "instanceSurvivalPriorCoefficients": prior.detach().cpu().half(),
        "instanceSurvivalCalibrationResidual": residual.detach().cpu().half(),
        "instanceCalibration": {
            "mode": model.instance_calibration_mode,
            "blend": float(model.instance_calibration_blend.detach().cpu()),
            "fusion": "prior_plus_applied_residual",
            "reliability": dict(calibration_reliability_meta),
            "runtimeExport": "fused_coefficients_only",
        },
        "protocol": dict(protocol),
        "relation": dict(relation_provenance),
        "geometry": dict(geometry_meta),
        "calibration": calibration,
        "validationAtCalibration": validation,
        "best": best,
        "trainingState": dict(training_state or {}),
        "testRead": False,
    }


def _save_fp16(path: Path, values: torch.Tensor) -> None:
    values.detach().cpu().numpy().astype("<f2").tofile(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, default=None)
    parser.add_argument("--dual-probe-rescue-init", type=Path, default=None)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--subpose-sidecar", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", default=EXPERIMENT_PREFIX)
    parser.add_argument("--variant", default="full")
    parser.add_argument(
        "--relation-source",
        choices=("bounded_hierarchical", "geometry_only"),
        default="bounded_hierarchical",
    )
    parser.add_argument(
        "--spectral-mode",
        choices=(
            "moment_envelope",
            "moment_extrema",
            "moment_support",
            "moment_extrema_support",
            "point",
        ),
        default="moment_envelope",
    )
    parser.add_argument(
        "--instance-calibration-mode",
        choices=("residual", "disabled"),
        default="residual",
    )
    parser.add_argument("--instance-calibration-max-abs", type=float, default=4.0)
    parser.add_argument("--sparse-instance-penalty", type=float, default=3.0)
    parser.add_argument("--instance-calibration-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--instance-calibration-ramp-fraction", type=float, default=0.20)
    parser.add_argument("--view-residual-max-abs", type=float, default=0.0)
    parser.add_argument("--view-residual-hidden-dim", type=int, default=16)
    parser.add_argument("--viewcell-extreme-visibility", action="store_true")
    parser.add_argument(
        "--viewcell-region-conditioned-visibility", action="store_true"
    )
    parser.add_argument(
        "--viewcell-region-conditioned-visibility-centering",
        choices=("none", "pose_mean"),
        default="none",
    )
    parser.add_argument("--view-residual-anchor-weight", type=float, default=0.01)
    parser.add_argument("--view-residual-center-weight", type=float, default=0.05)
    parser.add_argument("--view-residual-positive-guard-weight", type=float, default=0.25)
    parser.add_argument(
        "--view-residual-positive-importance-mix", type=float, default=0.50
    )
    parser.add_argument("--boundary-opportunity-hidden-dim", type=int, default=0)
    parser.add_argument("--boundary-opportunity-projection-dim", type=int, default=24)
    parser.add_argument("--boundary-opportunity-initial-logit", type=float, default=-6.0)
    parser.add_argument("--boundary-opportunity-max-logit-uplift", type=float, default=6.0)
    parser.add_argument("--boundary-opportunity-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-opportunity-rare-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-opportunity-positive-tail-mass-fraction", type=float, default=0.02)
    parser.add_argument("--boundary-opportunity-positive-tail-count-cap-fraction", type=float, default=0.05)
    parser.add_argument("--boundary-opportunity-min-positives", type=int, default=2)
    parser.add_argument("--boundary-opportunity-all-negative-uplift-weight", type=float, default=1.0)
    parser.add_argument("--boundary-opportunity-hard-negative-uplift-weight", type=float, default=1.0)
    parser.add_argument("--boundary-opportunity-negative-tail-fraction", type=float, default=0.01)
    parser.add_argument("--boundary-opportunity-min-negatives", type=int, default=32)
    parser.add_argument("--boundary-opportunity-margin", type=float, default=0.25)
    parser.add_argument("--boundary-opportunity-temperature", type=float, default=0.25)
    parser.add_argument("--boundary-opportunity-scale-ramp-fraction", type=float, default=0.10)
    parser.add_argument("--boundary-tail-residual-hidden-dim", type=int, default=0)
    parser.add_argument("--boundary-tail-residual-projection-dim", type=int, default=24)
    parser.add_argument("--boundary-tail-residual-max-abs", type=float, default=1.0)
    parser.add_argument(
        "--boundary-tail-residual-centering",
        choices=("pose_mean", "none"),
        default="pose_mean",
    )
    parser.add_argument(
        "--boundary-tail-residual-shortcut",
        choices=("none", "region_linear"),
        default="none",
    )
    parser.add_argument(
        "--boundary-tail-residual-fusion",
        choices=("product", "affine_region"),
        default="product",
    )
    parser.add_argument(
        "--boundary-tail-residual-output-init-std",
        type=float,
        default=0.0,
    )
    parser.add_argument("--boundary-tail-residual-loss-weight", type=float, default=1.0)
    parser.add_argument("--boundary-tail-residual-rare-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-tail-residual-positive-tail-mass-fraction", type=float, default=0.005)
    parser.add_argument("--boundary-tail-residual-positive-tail-count-cap", type=int, default=64)
    parser.add_argument("--boundary-tail-residual-min-positives", type=int, default=2)
    parser.add_argument("--boundary-tail-residual-negative-tail-fraction", type=float, default=0.04)
    parser.add_argument("--boundary-tail-residual-min-negatives", type=int, default=64)
    parser.add_argument("--boundary-tail-residual-max-negatives", type=int, default=256)
    parser.add_argument("--boundary-tail-residual-margin", type=float, default=0.20)
    parser.add_argument("--boundary-tail-residual-temperature", type=float, default=0.25)
    parser.add_argument("--boundary-tail-residual-pose-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--boundary-tail-residual-pose-cvar-weight", type=float, default=0.50)
    parser.add_argument("--boundary-tail-residual-negative-positive-weight", type=float, default=0.50)
    parser.add_argument("--boundary-tail-residual-positive-negative-weight", type=float, default=0.50)
    parser.add_argument("--boundary-tail-residual-all-positive-negative-weight", type=float, default=0.0)
    parser.add_argument("--boundary-tail-residual-classification-weight", type=float, default=0.0)
    parser.add_argument("--boundary-tail-residual-classification-margin", type=float, default=0.20)
    parser.add_argument("--boundary-tail-residual-regularization-weight", type=float, default=0.02)
    parser.add_argument(
        "--query-tail-separator-family",
        choices=QUERY_TAIL_SEPARATOR_FAMILIES,
        default="disabled",
    )
    parser.add_argument("--query-tail-separator-hidden-dim", type=int, default=8)
    parser.add_argument("--query-tail-separator-max-abs", type=float, default=0.5)
    parser.add_argument(
        "--query-tail-separator-centering",
        choices=("none", "pose_mean"),
        default="pose_mean",
    )
    parser.add_argument("--query-tail-separation-loss-weight", type=float, default=0.0)
    parser.add_argument("--query-tail-positive-mass-fraction", type=float, default=0.005)
    parser.add_argument("--query-tail-positive-count-cap", type=int, default=64)
    parser.add_argument("--query-tail-min-positives", type=int, default=2)
    parser.add_argument("--query-tail-negative-fraction", type=float, default=0.04)
    parser.add_argument("--query-tail-min-negatives", type=int, default=64)
    parser.add_argument("--query-tail-max-negatives", type=int, default=256)
    parser.add_argument("--query-tail-margin", type=float, default=0.0)
    parser.add_argument("--query-tail-temperature", type=float, default=0.25)
    parser.add_argument("--query-tail-pose-cvar-fraction", type=float, default=0.25)
    parser.add_argument("--query-tail-pose-cvar-weight", type=float, default=0.50)
    parser.add_argument("--query-tail-negative-positive-weight", type=float, default=0.25)
    parser.add_argument("--query-tail-positive-negative-weight", type=float, default=0.25)
    parser.add_argument("--query-tail-regularization-weight", type=float, default=0.02)
    parser.add_argument("--query-tail-positive-importance-power", type=float, default=0.5)
    parser.add_argument("--query-tail-cross-pose-pair-weight", type=float, default=0.40)
    parser.add_argument("--query-tail-global-pair-weight", type=float, default=0.25)
    parser.add_argument(
        "--boundary-tail-residual-positive-importance-transform",
        choices=("log1p", "power"),
        default="log1p",
    )
    parser.add_argument(
        "--boundary-tail-residual-positive-importance-power",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--boundary-tail-residual-cross-pose-pair-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--boundary-tail-residual-global-tail-pair-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--boundary-tail-residual-objective",
        choices=("partial_auc", "boundary_deficit"),
        default="partial_auc",
    )
    parser.add_argument(
        "--boundary-tail-residual-deficit-huber-beta", type=float, default=0.25
    )
    parser.add_argument(
        "--boundary-tail-residual-deficit-gap-weight", type=float, default=0.25
    )
    parser.add_argument(
        "--boundary-tail-residual-selection-source",
        choices=("frozen_base", "current_final"),
        default="frozen_base",
    )
    parser.add_argument(
        "--boundary-tail-fixed-frontier-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--boundary-tail-fixed-frontier-positive-count-cap",
        type=int,
        default=1024,
    )
    parser.add_argument("--boundary-tail-np-weight", type=float, default=0.0)
    parser.add_argument(
        "--boundary-tail-np-weighted-recall-target",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--boundary-tail-np-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--boundary-tail-np-dual-learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--boundary-tail-np-dual-maximum",
        type=float,
        default=20.0,
    )
    parser.add_argument("--cull-certificate-max-suppression", type=float, default=0.0)
    parser.add_argument("--cull-certificate-initial-suppression", type=float, default=0.05)
    parser.add_argument("--cull-certificate-hidden-dim", type=int, default=0)
    parser.add_argument(
        "--cull-certificate-input-mode",
        choices=(
            "hidden",
            "evidence",
            "hidden_evidence",
            "runtime",
            "hidden_runtime",
        ),
        default="hidden",
    )
    parser.add_argument("--cull-certificate-regularization-weight", type=float, default=0.005)
    parser.add_argument("--cull-certificate-tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--cull-certificate-positive-guard-weight", type=float, default=1.0)
    parser.add_argument("--cull-certificate-negative-tail-fraction", type=float, default=0.02)
    parser.add_argument("--cull-certificate-positive-uniform-mix", type=float, default=0.20)
    parser.add_argument("--cull-certificate-rare-positive-weight", type=float, default=0.0)
    parser.add_argument("--cull-certificate-center-anchor-weight", type=float, default=0.0)
    parser.add_argument("--cull-certificate-pair-weight", type=float, default=0.0)
    parser.add_argument("--cull-certificate-pair-positive-fraction", type=float, default=0.01)
    parser.add_argument("--cull-certificate-pair-negative-fraction", type=float, default=0.01)
    parser.add_argument("--cull-certificate-pair-margin", type=float, default=0.25)
    parser.add_argument("--cull-certificate-pair-temperature", type=float, default=0.25)
    parser.add_argument("--cull-certificate-pair-pose-cvar-fraction", type=float, default=0.50)
    parser.add_argument("--cull-certificate-pair-pose-cvar-weight", type=float, default=0.25)
    parser.add_argument("--cull-certificate-head-only", action="store_true")
    parser.add_argument(
        "--refinement-scope",
        choices=(
            "all",
            "runtime_visibility",
            "visibility_head",
            "view_residual",
            "boundary_opportunity",
            "boundary_tail_residual",
            "viewcell_extreme_visibility",
            "viewcell_region_conditioned_visibility",
            "viewcell_region_tail",
            "dual_probe_rescue",
        ),
        default="all",
    )
    parser.add_argument(
        "--loss-variant",
        choices=("safety_reserve", "normalized_rvl"),
        default="safety_reserve",
    )
    parser.add_argument("--train-split", default="auto")
    parser.add_argument("--calibration-split", default="auto")
    parser.add_argument("--validation-split", default="auto")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--observation-batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--survival-loss-weight", type=float, default=0.25)
    parser.add_argument("--relation-consistency-weight", type=float, default=0.10)
    parser.add_argument("--utility-loss-weight", type=float, default=0.10)
    parser.add_argument("--download-loss-weight", type=float, default=0.10)
    parser.add_argument("--regularization-weight", type=float, default=1e-5)
    parser.add_argument("--instance-calibration-regularization-weight", type=float, default=0.02)
    parser.add_argument("--dual-probe-rescue-loss-weight", type=float, default=1.0)
    parser.add_argument("--dual-probe-rescue-primary-keep-weight", type=float, default=1.0)
    parser.add_argument("--dual-probe-rescue-coverage-keep-weight", type=float, default=1.0)
    parser.add_argument("--dual-probe-rescue-negative-decay-weight", type=float, default=1.0)
    parser.add_argument(
        "--dual-probe-rescue-high-negative-decay-weight", type=float, default=1.0
    )
    parser.add_argument("--dual-probe-rescue-visible-weight-epsilon", type=float, default=1e-6)
    parser.add_argument("--boundary-tail-weight", type=float, default=0.30)
    parser.add_argument("--negative-band-weight", type=float, default=0.03)
    parser.add_argument("--negative-band-temperature", type=float, default=0.10)
    parser.add_argument(
        "--negative-band-shape", choices=("sigmoid", "softplus"), default="sigmoid"
    )
    parser.add_argument("--glb-resource-weight", type=float, default=0.015)
    parser.add_argument("--rvl-bce-positive-weight", type=float, default=14.0)
    parser.add_argument("--rvl-tversky-fn-weight", type=float, default=7.0)
    parser.add_argument("--rvl-count-weight", type=float, default=0.10)
    parser.add_argument(
        "--rvl-fp-normalization",
        choices=("positive", "negative", "candidate"),
        default="positive",
    )
    parser.add_argument("--efficiency-warmup-fraction", type=float, default=0.10)
    parser.add_argument("--efficiency-primary-fraction", type=float, default=0.0)
    parser.add_argument("--rvl-rank-weight", type=float, default=0.45)
    parser.add_argument("--rvl-rank-negative-top-k", type=int, default=256)
    parser.add_argument("--instance-exposure-balance-weight", type=float, default=0.0)
    parser.add_argument("--instance-exposure-balance-power", type=float, default=0.5)
    parser.add_argument("--instance-exposure-balance-max-weight", type=float, default=8.0)
    parser.add_argument("--same-instance-cross-view-rank-weight", type=float, default=0.0)
    parser.add_argument("--same-instance-cross-view-margin", type=float, default=0.5)
    parser.add_argument("--same-instance-cross-view-temperature", type=float, default=0.25)
    parser.add_argument("--same-instance-cross-view-importance-mix", type=float, default=0.5)
    parser.add_argument(
        "--same-instance-cross-view-recurrence-selected-fraction",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--same-instance-cross-view-recurrence-negative-tail-fraction",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--same-instance-cross-view-recurrence-min-negative-exposures",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--same-instance-cross-view-recurrence-min-hard-count",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--same-instance-cross-view-recurrence-smoothing-strength",
        type=float,
        default=16.0,
    )
    parser.add_argument("--candidate-boundary-hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--candidate-boundary-threshold", type=float, default=0.05)
    parser.add_argument("--candidate-boundary-proximity-temperature", type=float, default=0.05)
    parser.add_argument("--candidate-boundary-negative-tail-fraction", type=float, default=0.01)
    parser.add_argument("--candidate-boundary-selection-bias", type=float, default=1.0)
    parser.add_argument("--candidate-boundary-proximity-gain", type=float, default=1.0)
    parser.add_argument("--candidate-boundary-margin", type=float, default=0.0)
    parser.add_argument("--candidate-boundary-loss-temperature", type=float, default=0.25)
    parser.add_argument("--positive-tail-compactness-weight", type=float, default=0.0)
    parser.add_argument("--positive-tail-mass-fraction", type=float, default=0.01)
    parser.add_argument("--positive-tail-reference-quantile", type=float, default=0.05)
    parser.add_argument("--positive-tail-allowed-relative-gap", type=float, default=0.20)
    parser.add_argument("--positive-tail-temperature", type=float, default=0.25)
    parser.add_argument("--positive-tail-ramp-fraction", type=float, default=0.0)
    parser.add_argument("--tail-separation-weight", type=float, default=0.0)
    parser.add_argument(
        "--tail-selection-source",
        choices=("current", "initial_visibility"),
        default="current",
    )
    parser.add_argument("--tail-positive-fraction", type=float, default=0.01)
    parser.add_argument("--tail-negative-fraction", type=float, default=0.005)
    parser.add_argument("--tail-margin", type=float, default=0.50)
    parser.add_argument("--tail-temperature", type=float, default=0.20)
    parser.add_argument("--tail-pose-cvar-fraction", type=float, default=0.50)
    parser.add_argument("--tail-pose-cvar-weight", type=float, default=0.50)
    parser.add_argument("--tail-positive-importance-mix", type=float, default=0.80)
    parser.add_argument("--tail-positive-gradient-scale", type=float, default=1.0)
    parser.add_argument("--coverage-tail-separation-weight", type=float, default=0.0)
    parser.add_argument("--coverage-tail-positive-fraction", type=float, default=0.05)
    parser.add_argument(
        "--tail-objective-group", choices=("safety", "efficiency"), default="safety"
    )
    parser.add_argument("--tail-ramp-fraction", type=float, default=0.05)
    parser.add_argument("--tail-hard-pose-batch-fraction", type=float, default=0.0)
    parser.add_argument("--tail-hard-pose-pool-fraction", type=float, default=0.20)
    parser.add_argument("--safety-boundary-excess-weight", type=float, default=0.0)
    parser.add_argument(
        "--safety-boundary-scope", choices=("batch", "pose"), default="batch"
    )
    parser.add_argument("--safety-boundary-positive-mass-fraction", type=float, default=0.01)
    parser.add_argument("--safety-boundary-negative-tail-fraction", type=float, default=1.0)
    parser.add_argument("--safety-boundary-margin", type=float, default=0.0)
    parser.add_argument("--safety-boundary-temperature", type=float, default=0.20)
    parser.add_argument("--safety-boundary-pose-cvar-fraction", type=float, default=0.50)
    parser.add_argument("--safety-boundary-pose-cvar-weight", type=float, default=0.50)
    parser.add_argument("--safety-boundary-ramp-fraction", type=float, default=0.05)
    parser.add_argument("--safety-boundary-ema-decay", type=float, default=0.0)
    parser.add_argument("--rvl-budget-initial-scale", type=float, default=1.0)
    parser.add_argument("--rvl-budget-start-fraction", type=float, default=0.0)
    parser.add_argument("--rvl-budget-ramp-fraction", type=float, default=0.0)
    parser.add_argument("--operating-threshold-seed", type=int, default=None)
    parser.add_argument("--relation-gradient-cap", type=float, default=0.25)
    parser.add_argument("--schedule-gradient-cap", type=float, default=0.25)
    parser.add_argument("--efficiency-gradient-cap", type=float, default=0.25)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=0)
    parser.add_argument("--max-eval-poses", type=int, default=0)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dual_probe_rescue_spec = _load_dual_probe_rescue_init(
        args.dual_probe_rescue_init
    )
    if args.refinement_scope == "dual_probe_rescue":
        if dual_probe_rescue_spec is None:
            raise ValueError(
                "dual_probe_rescue refinement requires --dual-probe-rescue-init"
            )
        if args.initial_checkpoint is None:
            raise ValueError(
                "dual_probe_rescue refinement requires an initial checkpoint"
            )
        if args.dual_probe_rescue_loss_weight <= 0.0:
            raise ValueError(
                "dual_probe_rescue refinement requires a positive loss weight"
            )
    elif dual_probe_rescue_spec is not None:
        raise ValueError(
            "--dual-probe-rescue-init is only valid with --refinement-scope dual_probe_rescue"
        )
    if (
        not math.isfinite(args.dual_probe_rescue_loss_weight)
        or args.dual_probe_rescue_loss_weight < 0.0
        or not math.isfinite(args.dual_probe_rescue_primary_keep_weight)
        or args.dual_probe_rescue_primary_keep_weight < 0.0
        or not math.isfinite(args.dual_probe_rescue_coverage_keep_weight)
        or args.dual_probe_rescue_coverage_keep_weight < 0.0
        or not math.isfinite(args.dual_probe_rescue_negative_decay_weight)
        or args.dual_probe_rescue_negative_decay_weight < 0.0
        or not math.isfinite(args.dual_probe_rescue_high_negative_decay_weight)
        or args.dual_probe_rescue_high_negative_decay_weight < 0.0
        or not math.isfinite(args.dual_probe_rescue_visible_weight_epsilon)
        or args.dual_probe_rescue_visible_weight_epsilon <= 0.0
    ):
        raise ValueError("dual-probe rescue loss weights and epsilon are invalid")
    if min(args.epochs, args.steps_per_epoch, args.poses_per_batch, args.observation_batch_size) <= 0:
        raise ValueError("epoch, step, pose-batch, and observation-batch counts must be positive")
    if args.eval_every <= 0 or args.calibration_bootstrap_replicates <= 0:
        raise ValueError("evaluation cadence and calibration bootstrap count must be positive")
    if (
        args.instance_calibration_max_abs <= 0.0
        or args.sparse_instance_penalty < 0.0
        or args.instance_calibration_regularization_weight < 0.0
        or args.view_residual_max_abs < 0.0
        or args.view_residual_hidden_dim < 0
        or args.boundary_opportunity_hidden_dim < 0
        or args.boundary_opportunity_projection_dim <= 0
        or not math.isfinite(args.boundary_opportunity_initial_logit)
        or not math.isfinite(args.boundary_opportunity_max_logit_uplift)
        or args.boundary_opportunity_max_logit_uplift <= 0.0
        or args.boundary_opportunity_loss_weight < 0.0
        or not 0.0 < args.boundary_opportunity_rare_threshold <= 1.0
        or not 0.0 < args.boundary_opportunity_positive_tail_mass_fraction <= 1.0
        or not 0.0 < args.boundary_opportunity_positive_tail_count_cap_fraction <= 1.0
        or args.boundary_opportunity_min_positives <= 0
        or args.boundary_opportunity_all_negative_uplift_weight < 0.0
        or args.boundary_opportunity_hard_negative_uplift_weight < 0.0
        or not 0.0 < args.boundary_opportunity_negative_tail_fraction <= 1.0
        or args.boundary_opportunity_min_negatives <= 0
        or args.boundary_opportunity_margin < 0.0
        or args.boundary_opportunity_temperature <= 0.0
        or not 0.0 <= args.boundary_opportunity_scale_ramp_fraction <= 1.0
        or args.boundary_tail_residual_hidden_dim < 0
        or args.boundary_tail_residual_projection_dim <= 0
        or not math.isfinite(args.boundary_tail_residual_max_abs)
        or args.boundary_tail_residual_max_abs <= 0.0
        or args.boundary_tail_residual_loss_weight < 0.0
        or not 0.0 < args.boundary_tail_residual_rare_threshold <= 1.0
        or not 0.0 < args.boundary_tail_residual_positive_tail_mass_fraction <= 1.0
        or args.boundary_tail_residual_positive_tail_count_cap <= 0
        or args.boundary_tail_residual_positive_tail_count_cap < args.boundary_tail_residual_min_positives
        or args.boundary_tail_residual_min_positives <= 0
        or not 0.0 < args.boundary_tail_residual_negative_tail_fraction <= 1.0
        or args.boundary_tail_residual_min_negatives <= 0
        or args.boundary_tail_residual_max_negatives < args.boundary_tail_residual_min_negatives
        or args.boundary_tail_residual_margin < 0.0
        or args.boundary_tail_residual_temperature <= 0.0
        or not 0.0 < args.boundary_tail_residual_pose_cvar_fraction <= 1.0
        or args.boundary_tail_residual_pose_cvar_weight < 0.0
        or args.boundary_tail_residual_negative_positive_weight < 0.0
        or args.boundary_tail_residual_positive_negative_weight < 0.0
        or args.boundary_tail_residual_all_positive_negative_weight < 0.0
        or args.boundary_tail_residual_classification_weight < 0.0
        or args.boundary_tail_residual_classification_margin < 0.0
        or args.boundary_tail_residual_regularization_weight < 0.0
        or not math.isfinite(
            args.boundary_tail_residual_positive_importance_power
        )
        or args.boundary_tail_residual_positive_importance_power <= 0.0
        or args.boundary_tail_residual_cross_pose_pair_weight < 0.0
        or args.boundary_tail_residual_global_tail_pair_weight < 0.0
        or not math.isfinite(args.boundary_tail_fixed_frontier_weight)
        or args.boundary_tail_fixed_frontier_weight < 0.0
        or args.boundary_tail_fixed_frontier_positive_count_cap <= 0
        or not math.isfinite(args.boundary_tail_np_weight)
        or args.boundary_tail_np_weight < 0.0
        or not math.isfinite(args.boundary_tail_np_weighted_recall_target)
        or not 0.0 < args.boundary_tail_np_weighted_recall_target <= 1.0
        or not math.isfinite(args.boundary_tail_np_temperature)
        or args.boundary_tail_np_temperature <= 0.0
        or not math.isfinite(args.boundary_tail_np_dual_learning_rate)
        or args.boundary_tail_np_dual_learning_rate < 0.0
        or not math.isfinite(args.boundary_tail_np_dual_maximum)
        or args.boundary_tail_np_dual_maximum <= 0.0
        or not math.isfinite(args.boundary_tail_residual_deficit_huber_beta)
        or args.boundary_tail_residual_deficit_huber_beta <= 0.0
        or not math.isfinite(args.boundary_tail_residual_deficit_gap_weight)
        or args.boundary_tail_residual_deficit_gap_weight < 0.0
        or not math.isfinite(args.boundary_tail_residual_output_init_std)
        or args.boundary_tail_residual_output_init_std < 0.0
        or args.cull_certificate_max_suppression < 0.0
        or args.cull_certificate_hidden_dim < 0
        or args.cull_certificate_regularization_weight < 0.0
        or args.cull_certificate_tail_loss_weight < 0.0
        or args.cull_certificate_positive_guard_weight < 0.0
        or args.cull_certificate_center_anchor_weight < 0.0
        or args.cull_certificate_pair_weight < 0.0
        or args.cull_certificate_rare_positive_weight < 0.0
    ):
        raise ValueError("instance calibration magnitude and regularization arguments are invalid")
    if args.view_residual_max_abs > 0.0 and args.view_residual_hidden_dim <= 0:
        raise ValueError("enabled view residual requires a positive hidden dimension")
    if args.refinement_scope == "view_residual" and args.view_residual_max_abs <= 0.0:
        raise ValueError("view_residual refinement requires an enabled view residual head")
    if args.refinement_scope == "boundary_opportunity" and (
        args.boundary_opportunity_hidden_dim <= 0
        or args.spectral_mode not in {"moment_extrema", "moment_extrema_support"}
        or args.boundary_opportunity_loss_weight <= 0.0
        or args.initial_checkpoint is None
        or args.subpose_sidecar is None
    ):
        raise ValueError(
            "boundary_opportunity refinement requires an initial checkpoint, "
            "subpose sidecar, analytic extrema, enabled head, and positive loss weight"
        )
    if args.refinement_scope == "boundary_tail_residual" and (
        args.boundary_tail_residual_hidden_dim <= 0
        or args.spectral_mode not in {"moment_extrema", "moment_extrema_support"}
        or args.boundary_tail_residual_loss_weight <= 0.0
        or args.initial_checkpoint is None
        or args.subpose_sidecar is None
        or args.poses_per_batch < 2
    ):
        raise ValueError(
            "boundary_tail_residual refinement requires an initial checkpoint, "
            "subpose sidecar, analytic extrema, enabled head, positive loss weight, "
            "and at least two poses per batch"
        )
    query_tail_enabled = args.query_tail_separator_family != "disabled"
    query_tail_scalars = (
        args.query_tail_separator_max_abs,
        args.query_tail_separation_loss_weight,
        args.query_tail_positive_mass_fraction,
        args.query_tail_negative_fraction,
        args.query_tail_margin,
        args.query_tail_temperature,
        args.query_tail_pose_cvar_fraction,
        args.query_tail_pose_cvar_weight,
        args.query_tail_negative_positive_weight,
        args.query_tail_positive_negative_weight,
        args.query_tail_regularization_weight,
        args.query_tail_positive_importance_power,
        args.query_tail_cross_pose_pair_weight,
        args.query_tail_global_pair_weight,
    )
    if not all(math.isfinite(float(value)) for value in query_tail_scalars):
        raise ValueError("query-tail separator arguments must be finite")
    if (
        args.query_tail_separator_hidden_dim <= 0
        or args.query_tail_separator_max_abs <= 0.0
        or not 0.0 < args.query_tail_positive_mass_fraction <= 1.0
        or args.query_tail_positive_count_cap <= 0
        or args.query_tail_min_positives < 0
        or args.query_tail_min_positives > args.query_tail_positive_count_cap
        or not 0.0 < args.query_tail_negative_fraction <= 1.0
        or args.query_tail_min_negatives < 0
        or args.query_tail_max_negatives <= 0
        or args.query_tail_min_negatives > args.query_tail_max_negatives
        or args.query_tail_margin < 0.0
        or args.query_tail_temperature <= 0.0
        or not 0.0 < args.query_tail_pose_cvar_fraction <= 1.0
        or args.query_tail_pose_cvar_weight < 0.0
        or args.query_tail_negative_positive_weight < 0.0
        or args.query_tail_positive_negative_weight < 0.0
        or args.query_tail_regularization_weight < 0.0
        or args.query_tail_positive_importance_power <= 0.0
        or args.query_tail_cross_pose_pair_weight < 0.0
        or args.query_tail_global_pair_weight < 0.0
    ):
        raise ValueError("query-tail separator dimensions or loss arguments are invalid")
    if query_tail_enabled and (
        args.query_tail_separation_loss_weight <= 0.0
        or args.refinement_scope != "all"
        or args.initial_checkpoint is not None
        or args.subpose_sidecar is None
        or args.poses_per_batch < 2
    ):
        raise ValueError(
            "query-tail separator requires from-scratch joint training, train-only "
            "subpose supervision, a positive loss weight, and at least two poses per batch"
        )
    if not query_tail_enabled and args.query_tail_separation_loss_weight != 0.0:
        raise ValueError(
            "disabled query-tail separator requires zero separation loss weight"
        )
    if args.boundary_tail_fixed_frontier_weight > 0.0 and (
        args.refinement_scope != "boundary_tail_residual"
        or args.initial_checkpoint is None
    ):
        raise ValueError(
            "fixed weighted safety frontier requires initialized boundary-tail refinement"
        )
    if args.boundary_tail_np_weight > 0.0 and (
        args.refinement_scope not in {"boundary_tail_residual", "view_residual"}
        or args.initial_checkpoint is None
        or args.poses_per_batch < 2
    ):
        raise ValueError(
            "weighted Neyman-Pearson refinement requires an initialized signed "
            "view-query refinement and at least two poses per batch"
        )
    if args.refinement_scope == "viewcell_extreme_visibility" and (
        not args.viewcell_extreme_visibility
        or args.spectral_mode not in {"moment_extrema", "moment_extrema_support"}
        or args.initial_checkpoint is None
    ):
        raise ValueError(
            "viewcell_extreme_visibility refinement requires an initial checkpoint, "
            "analytic extrema, and an enabled raw-extrema main-score branch"
        )
    if args.refinement_scope == "viewcell_region_conditioned_visibility" and (
        not args.viewcell_region_conditioned_visibility
        or args.spectral_mode not in {"moment_extrema", "moment_extrema_support"}
        or args.initial_checkpoint is None
    ):
        raise ValueError(
            "viewcell_region_conditioned_visibility refinement requires an initial checkpoint, "
            "analytic extrema, and an enabled region-conditioned main-score branch"
        )
    if args.refinement_scope == "viewcell_region_tail" and (
        not args.viewcell_region_conditioned_visibility
        or args.spectral_mode not in {"moment_extrema", "moment_extrema_support"}
        or args.initial_checkpoint is None
        or args.subpose_sidecar is None
        or args.tail_selection_source != "initial_visibility"
        or args.boundary_tail_residual_objective != "partial_auc"
        or args.boundary_tail_residual_loss_weight <= 0.0
        or args.poses_per_batch < 2
    ):
        raise ValueError(
            "viewcell_region_tail refinement requires an initial fixed-epoch "
            "checkpoint, subpose supervision, analytic extrema, the enabled "
            "region branch, initial_visibility selection, a positive partial-AUC "
            "loss weight, and at least two poses per batch"
        )
    if args.cull_certificate_max_suppression > 0.0 and not (
        0.0
        < args.cull_certificate_initial_suppression
        < args.cull_certificate_max_suppression
    ):
        raise ValueError(
            "enabled cull certificate requires 0 < initial suppression < maximum suppression"
        )
    if args.cull_certificate_head_only and args.cull_certificate_max_suppression <= 0.0:
        raise ValueError("head-only refinement requires an enabled cull certificate")
    if not 0.0 < args.cull_certificate_negative_tail_fraction <= 1.0:
        raise ValueError("certificate negative tail fraction must lie in (0, 1]")
    if not 0.0 <= args.cull_certificate_positive_uniform_mix <= 1.0:
        raise ValueError("certificate positive uniform mix must lie in [0, 1]")
    if not 0.0 < args.cull_certificate_pair_positive_fraction <= 1.0:
        raise ValueError("certificate pair positive fraction must lie in (0, 1]")
    if not 0.0 < args.cull_certificate_pair_negative_fraction <= 1.0:
        raise ValueError("certificate pair negative fraction must lie in (0, 1]")
    if args.cull_certificate_pair_margin < 0.0 or args.cull_certificate_pair_temperature <= 0.0:
        raise ValueError("certificate pair margin and temperature are invalid")
    if not 0.0 < args.cull_certificate_pair_pose_cvar_fraction <= 1.0:
        raise ValueError("certificate pair pose CVaR fraction must lie in (0, 1]")
    if args.cull_certificate_pair_pose_cvar_weight < 0.0:
        raise ValueError("certificate pair pose CVaR weight must be non-negative")
    if (
        (
            args.cull_certificate_tail_loss_weight > 0.0
            or args.cull_certificate_pair_weight > 0.0
        )
        and args.cull_certificate_max_suppression <= 0.0
    ):
        raise ValueError("certificate tail objectives require an enabled cull certificate")
    if (
        args.rvl_bce_positive_weight <= 0.0
        or args.rvl_tversky_fn_weight <= 0.0
        or args.rvl_count_weight < 0.0
        or not 0.0 <= args.efficiency_warmup_fraction <= 1.0
        or not 0.0 <= args.efficiency_primary_fraction <= 1.0
        or args.rvl_rank_weight < 0.0
        or args.rvl_rank_negative_top_k <= 0
        or args.instance_exposure_balance_weight < 0.0
        or not 0.0 < args.instance_exposure_balance_power <= 1.0
        or args.instance_exposure_balance_max_weight < 1.0
        or args.same_instance_cross_view_rank_weight < 0.0
        or args.same_instance_cross_view_margin < 0.0
        or args.same_instance_cross_view_temperature <= 0.0
        or not 0.0 <= args.same_instance_cross_view_importance_mix <= 1.0
        or not 0.0
        <= args.same_instance_cross_view_recurrence_selected_fraction
        <= 1.0
        or not 0.0
        < args.same_instance_cross_view_recurrence_negative_tail_fraction
        < 1.0
        or args.same_instance_cross_view_recurrence_min_negative_exposures <= 0
        or args.same_instance_cross_view_recurrence_min_hard_count <= 0
        or args.same_instance_cross_view_recurrence_smoothing_strength < 0.0
        or args.candidate_boundary_hard_negative_weight < 0.0
        or args.candidate_boundary_threshold < 0.0
        or args.candidate_boundary_proximity_temperature <= 0.0
        or not 0.0 < args.candidate_boundary_negative_tail_fraction <= 1.0
        or args.candidate_boundary_selection_bias < 0.0
        or args.candidate_boundary_proximity_gain < 0.0
        or args.candidate_boundary_margin < 0.0
        or args.candidate_boundary_loss_temperature <= 0.0
        or args.view_residual_anchor_weight < 0.0
        or args.view_residual_center_weight < 0.0
        or args.view_residual_positive_guard_weight < 0.0
        or not 0.0 <= args.view_residual_positive_importance_mix <= 1.0
        or args.negative_band_temperature <= 0.0
        or args.positive_tail_compactness_weight < 0.0
        or not (
            0.0
            < args.positive_tail_mass_fraction
            < args.positive_tail_reference_quantile
            < 1.0
        )
        or args.positive_tail_allowed_relative_gap < 0.0
        or args.positive_tail_temperature <= 0.0
        or not 0.0 <= args.positive_tail_ramp_fraction <= 1.0
        or args.tail_separation_weight < 0.0
        or args.safety_boundary_excess_weight < 0.0
        or not 0.0 < args.tail_positive_fraction <= 1.0
        or not 0.0 < args.tail_negative_fraction <= 1.0
        or args.tail_margin < 0.0
        or args.tail_temperature <= 0.0
        or not 0.0 < args.tail_pose_cvar_fraction <= 1.0
        or args.tail_pose_cvar_weight < 0.0
        or not 0.0 <= args.tail_positive_importance_mix <= 1.0
        or not 0.0 <= args.tail_positive_gradient_scale <= 1.0
        or args.coverage_tail_separation_weight < 0.0
        or not 0.0 < args.coverage_tail_positive_fraction <= 1.0
        or not 0.0 <= args.tail_ramp_fraction <= 1.0
        or not 0.0 <= args.tail_hard_pose_batch_fraction <= 1.0
        or not 0.0 < args.tail_hard_pose_pool_fraction <= 1.0
        or not 0.0 < args.safety_boundary_positive_mass_fraction < 1.0
        or not 0.0 < args.safety_boundary_negative_tail_fraction <= 1.0
        or args.safety_boundary_margin < 0.0
        or args.safety_boundary_temperature <= 0.0
        or not 0.0 < args.safety_boundary_pose_cvar_fraction <= 1.0
        or args.safety_boundary_pose_cvar_weight < 0.0
        or not 0.0 <= args.safety_boundary_ramp_fraction <= 1.0
        or not 0.0 <= args.safety_boundary_ema_decay < 1.0
        or not 0.0 <= args.rvl_budget_initial_scale <= 1.0
        or not 0.0 <= args.rvl_budget_start_fraction <= 1.0
        or not 0.0 <= args.rvl_budget_ramp_fraction <= 1.0
        or (args.operating_threshold_seed is not None and args.operating_threshold_seed < 0)
    ):
        raise ValueError("RVL diagnostic weights or efficiency warmup are invalid")
    if args.safety_boundary_ema_decay > 0.0 and (
        args.safety_boundary_scope != "batch"
        or args.safety_boundary_excess_weight <= 0.0
    ):
        raise ValueError(
            "safety boundary EMA requires an enabled batch-scope boundary loss"
        )
    if args.cull_certificate_head_only and args.refinement_scope != "all":
        raise ValueError("cull-certificate head-only and runtime refinement scopes are exclusive")
    if args.cull_certificate_max_suppression > 0.0 and (
        args.instance_exposure_balance_weight > 0.0
        or args.same_instance_cross_view_rank_weight > 0.0
    ):
        raise ValueError("instance-balanced refinement cannot be combined with a cull certificate")
    if args.same_instance_cross_view_rank_weight > 0.0 and args.poses_per_batch < 2:
        raise ValueError("same-instance cross-view ranking requires at least two poses per batch")
    if args.same_instance_cross_view_recurrence_selected_fraction > 0.0 and (
        args.same_instance_cross_view_rank_weight <= 0.0
        or args.initial_checkpoint is None
    ):
        raise ValueError(
            "recurrent cross-view ranking requires a positive rank weight and initial checkpoint"
        )
    if args.tail_hard_pose_batch_fraction > 0.0 and args.initial_checkpoint is None:
        raise ValueError("tail hard-pose sampling requires an initial checkpoint")
    if args.refinement_scope == "view_residual" and args.tail_separation_weight <= 0.0:
        raise ValueError(
            "view-residual refinement requires an exact-tail separation objective"
        )
    if args.refinement_scope == "view_residual" and (
        args.loss_variant != "safety_reserve"
        or args.tail_positive_gradient_scale <= 0.0
    ):
        raise ValueError(
            "view-residual refinement requires safety-reserve exact-tail gradients on both classes"
        )
    if args.refinement_scope == "view_residual" and (
        args.initial_checkpoint is None
        or args.view_residual_max_abs <= 0.0
    ):
        raise ValueError(
            "view-residual refinement requires an initialized bounded residual head"
        )
    _validate_extreme_tail_selection_contract(args)
    _instance_calibration_blend(
        0,
        int(args.epochs) * int(args.steps_per_epoch),
        warmup_fraction=args.instance_calibration_warmup_fraction,
        ramp_fraction=args.instance_calibration_ramp_fraction,
    )
    _boundary_opportunity_scale(
        0,
        int(args.epochs) * int(args.steps_per_epoch),
        ramp_fraction=args.boundary_opportunity_scale_ramp_fraction,
    )
    if args.allow_missing_glb_costs and args.epochs > 12:
        raise ValueError("synthetic GLB costs are restricted to short diagnostic runs")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)
    operating_threshold_seed = (
        int(args.seed)
        if args.operating_threshold_seed is None
        else int(args.operating_threshold_seed)
    )
    device = _device(args.device)
    world_aabbs, instance_to_glb_np, runtime_meta = load_runtime_meta(args.runtime_meta)
    num_instances = int(world_aabbs.shape[0])
    num_glbs = int(runtime_meta.get("numGlbs", int(instance_to_glb_np.max()) + 1))
    dataset = PoseCSRDataset(
        args.dataset_dir,
        num_instances=num_instances,
        subpose_sidecar=args.subpose_sidecar,
    )
    if dataset.query_centers_world is None or dataset.candidate_cameras_world is None or dataset.viewcell_radii_m is None:
        raise ValueError("v3 PoseCSR must explicitly store query center, candidate camera, and view-cell radius")
    if np.allclose(dataset.query_centers_world, dataset.candidate_cameras_world, rtol=0.0, atol=1e-6):
        raise ValueError("query centers must not be silently replaced by candidate back-camera centers")
    model_fov = float(dataset.meta.get("modelInputFovYDeg", dataset.meta.get("fovYDeg", float("nan"))))
    if not math.isfinite(model_fov) or not math.isclose(model_fov, 66.0, abs_tol=1e-5):
        raise ValueError("v3 candidates and model input must use the registered 66 degree FOV")

    train_name, train_split = _resolve_split(dataset, args.train_split, "train")
    calibration_name, calibration_split = _resolve_split(dataset, args.calibration_split, "calibration")
    validation_name, validation_split = _resolve_split(dataset, args.validation_split, "validation")
    if args.refinement_scope in {
        "boundary_opportunity",
        "boundary_tail_residual",
        "viewcell_region_tail",
    } or query_tail_enabled:
        if not dataset.has_subpose_robust_labels or dataset.subpose_counts is None:
            raise ValueError(
                "boundary-tail refinement requires valid train-only subpose hit-rate supervision"
            )
        train_subpose_counts = np.asarray(
            dataset.subpose_counts[train_split.pose_indices], dtype=np.int64
        )
        if train_subpose_counts.size == 0 or np.any(train_subpose_counts <= 0):
            raise ValueError(
                "boundary-tail refinement requires at least one successful subpose for every train pose"
            )
        if args.refinement_scope in {
            "boundary_opportunity",
            "boundary_tail_residual",
        }:
            (
                boundary_opportunity_pose_indices,
                boundary_opportunity_pose_meta,
            ) = _boundary_opportunity_pose_pool(
                dataset,
                train_split,
                rare_threshold=(
                    args.boundary_opportunity_rare_threshold
                    if args.refinement_scope == "boundary_opportunity"
                    else args.boundary_tail_residual_rare_threshold
                ),
            )
        else:
            boundary_opportunity_pose_indices = None
            boundary_opportunity_pose_meta = {
                "sourceSplit": "train",
                "enabled": False,
                "testRead": False,
            }
    else:
        boundary_opportunity_pose_indices = None
        boundary_opportunity_pose_meta = {
            "sourceSplit": "train",
            "enabled": False,
            "testRead": False,
        }
    split_pose_sets = [
        set(train_split.pose_indices.tolist()),
        set(calibration_split.pose_indices.tolist()),
        set(validation_split.pose_indices.tolist()),
    ]
    if split_pose_sets[0] & split_pose_sets[1] or split_pose_sets[0] & split_pose_sets[2] or split_pose_sets[1] & split_pose_sets[2]:
        raise ValueError("train, calibration, and validation pose sets must be disjoint")
    allowed_pose_indices = np.concatenate(
        [train_split.pose_indices, calibration_split.pose_indices, validation_split.pose_indices]
    ).astype(np.int64, copy=False)
    viewcell_radii = np.asarray(
        dataset.viewcell_radii_m[allowed_pose_indices], dtype=np.float32
    ).reshape(-1)
    if (
        viewcell_radii.size == 0
        or not np.isfinite(viewcell_radii).all()
        or np.any(viewcell_radii <= 0.0)
    ):
        raise ValueError("permitted PoseCSR view-cell radii must be finite and positive")
    viewcell_radius_m = float(viewcell_radii[0])
    if not np.allclose(viewcell_radii, viewcell_radius_m, rtol=0.0, atol=1e-6):
        raise ValueError("the registered v4 runtime requires one fixed view-cell radius")

    geometry_cpu, geometry_meta = _load_geometry(args.initial_geo_features, num_instances)
    (
        relation,
        relation_cpu,
        local_cpu,
        structural_cpu,
        observations_np,
        relation_provenance,
    ) = _load_relation_bundle(args.relation_dir, dataset, train_split, num_instances)
    depth_meta = relation.metadata["depthNormalization"]
    depth_q01 = float(depth_meta["q01"])
    depth_q99 = float(depth_meta["q99"])
    depth_epsilon = float(depth_meta["epsilon"])
    if not depth_q99 > depth_q01 or depth_epsilon <= 0.0 or depth_meta.get("sourceSplit") != "train":
        raise ValueError("relation depth normalization is not a valid train-frozen contract")

    sampler = StratifiedSurvivalObservationSampler(
        observations_np,
        seed=args.seed,
        metadata={"trainOnly": True, "splitNames": ["train"]},
    )
    required_sampler_steps = math.ceil(
        sampler.unique_instances.size / int(args.observation_batch_size)
    )
    if int(args.steps_per_epoch) < required_sampler_steps:
        raise ValueError(
            f"steps_per_epoch={args.steps_per_epoch} cannot cover all observed instances; "
            f"need at least {required_sampler_steps}"
        )
    calibration_reliability_np, calibration_reliability_meta = (
        _instance_calibration_reliability(
            dataset,
            train_split,
            observations_np,
            num_instances,
        )
    )
    if args.instance_exposure_balance_weight > 0.0:
        (
            exposure_positive_np,
            exposure_negative_np,
            exposure_meta,
        ) = _instance_exposure_balance_weights(
            dataset,
            train_split,
            num_instances,
            power=args.instance_exposure_balance_power,
            maximum_weight=args.instance_exposure_balance_max_weight,
        )
    else:
        exposure_positive_np = np.ones((num_instances,), dtype=np.float32)
        exposure_negative_np = np.ones((num_instances,), dtype=np.float32)
        exposure_meta = {
            "sourceSplit": "train",
            "enabled": False,
            "testRead": False,
        }
    glb_bytes_np = _load_glb_bytes(
        args.glb_index,
        args.glb_root,
        num_glbs,
        instance_to_glb_np,
        allow_missing=args.allow_missing_glb_costs,
    )
    glb_cost_norm_np = np.log1p(glb_bytes_np)
    glb_cost_norm_np /= max(float(glb_cost_norm_np.max()), 1e-6)

    geometry = geometry_cpu.to(device)
    relation_tensors: dict[str, Any] = {
        key: value.to(device) for key, value in relation_cpu.items()
    }
    relation_tensors["metadata"] = relation.metadata
    local_ids = local_cpu.to(device)
    structural_ids = structural_cpu.to(device)
    glb_bytes = torch.from_numpy(glb_bytes_np).to(device)
    glb_cost_norm = torch.from_numpy(glb_cost_norm_np.astype(np.float32)).to(device)
    exposure_positive_weights = torch.from_numpy(exposure_positive_np).to(device)
    exposure_negative_weights = torch.from_numpy(exposure_negative_np).to(device)
    model = BoundedRelationSurvivalMomentModel(
        num_instances,
        num_glbs,
        relation_source=args.relation_source,
        spectral_mode=args.spectral_mode,
        depth_q01=depth_q01,
        depth_q99=depth_q99,
        depth_epsilon=depth_epsilon,
        instance_calibration_mode=args.instance_calibration_mode,
        instance_calibration_max_abs=args.instance_calibration_max_abs,
        sparse_instance_penalty=args.sparse_instance_penalty,
        view_residual_max_abs=args.view_residual_max_abs,
        view_residual_hidden_dim=args.view_residual_hidden_dim,
        boundary_opportunity_hidden_dim=args.boundary_opportunity_hidden_dim,
        boundary_opportunity_projection_dim=(
            args.boundary_opportunity_projection_dim
        ),
        boundary_opportunity_initial_logit=(
            args.boundary_opportunity_initial_logit
        ),
        boundary_opportunity_max_logit_uplift=(
            args.boundary_opportunity_max_logit_uplift
        ),
        boundary_tail_residual_hidden_dim=(
            args.boundary_tail_residual_hidden_dim
        ),
        boundary_tail_residual_projection_dim=(
            args.boundary_tail_residual_projection_dim
        ),
        boundary_tail_residual_max_abs=args.boundary_tail_residual_max_abs,
        boundary_tail_residual_centering=(
            args.boundary_tail_residual_centering
        ),
        boundary_tail_residual_shortcut=args.boundary_tail_residual_shortcut,
        boundary_tail_residual_fusion=args.boundary_tail_residual_fusion,
        boundary_tail_residual_output_init_std=(
            args.boundary_tail_residual_output_init_std
        ),
        viewcell_extreme_visibility_enabled=(
            args.viewcell_extreme_visibility
        ),
        viewcell_region_conditioned_visibility_enabled=(
            args.viewcell_region_conditioned_visibility
        ),
        viewcell_region_conditioned_visibility_centering=(
            args.viewcell_region_conditioned_visibility_centering
        ),
        query_tail_separator_family=args.query_tail_separator_family,
        query_tail_separator_hidden_dim=args.query_tail_separator_hidden_dim,
        query_tail_separator_max_abs=args.query_tail_separator_max_abs,
        query_tail_separator_centering=args.query_tail_separator_centering,
        cull_certificate_max_suppression=args.cull_certificate_max_suppression,
        cull_certificate_initial_suppression=(
            args.cull_certificate_initial_suppression
        ),
        cull_certificate_hidden_dim=args.cull_certificate_hidden_dim,
        cull_certificate_input_mode=args.cull_certificate_input_mode,
        dual_probe_rescue=dual_probe_rescue_spec,
    ).to(device)
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb_np).to(device))
    model.set_instance_calibration_reliability(
        torch.from_numpy(calibration_reliability_np).to(device)
    )
    initialization = _initialize_from_checkpoint(model, args.initial_checkpoint)
    if args.tail_selection_source == "initial_visibility":
        initial_visibility_weight = model.visibility_head.weight.detach().clone()
        initial_visibility_bias = model.visibility_head.bias.detach().clone()
    else:
        initial_visibility_weight = None
        initial_visibility_bias = None
    if args.cull_certificate_head_only:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert model.cull_certificate_head is not None
        for parameter in model.cull_certificate_head.parameters():
            parameter.requires_grad_(True)
        refinement_scope_meta = {
            "scope": "cull_certificate_head_only",
            "trainableParameterCount": int(
                sum(value.numel() for value in model.parameters() if value.requires_grad)
            ),
        }
    else:
        refinement_scope_meta = _set_refinement_scope(
            model,
            args.refinement_scope,
            initialized=initialization is not None,
        )
    refinement_calibration_blend = (
        float(initialization["instanceCalibrationBlend"])
        if initialization is not None
        else None
    )
    if dual_probe_rescue_spec is None:
        dual_probe_rescue_meta: dict[str, Any] = {
            "enabled": False,
            "sourceSplit": "train",
            "testRead": False,
        }
    else:
        dual_probe_rescue_meta = {
            "enabled": True,
            "initSpecPath": str(args.dual_probe_rescue_init.resolve()),
            "initSpecSchema": dual_probe_rescue_spec["schema"],
            "sourceSplit": "train",
            "selectedFromValidation": False,
            "testRead": False,
            "primaryRiskCertificate": dict(
                dual_probe_rescue_spec["primary"]["riskCertificate"]
            ),
            "coverageRiskCertificate": dict(
                dual_probe_rescue_spec["coverage"]["riskCertificate"]
            ),
            "lossWeight": float(args.dual_probe_rescue_loss_weight),
            "primaryKeepWeight": float(
                args.dual_probe_rescue_primary_keep_weight
            ),
            "coverageKeepWeight": float(
                args.dual_probe_rescue_coverage_keep_weight
            ),
            "negativeDecayWeight": float(
                args.dual_probe_rescue_negative_decay_weight
            ),
            "highNegativeDecayWeight": float(
                args.dual_probe_rescue_high_negative_decay_weight
            ),
            "visibleWeightEpsilon": float(
                args.dual_probe_rescue_visible_weight_epsilon
            ),
            "posteriorHyperparametersOwnedByInit": True,
            "thresholdsAndCoefficientsOverriddenByCommand": False,
            "runtimeAssets": "shared frozen probe buffers; no per-instance asset",
        }
    needs_initial_runtime_features = (
        args.tail_hard_pose_batch_fraction > 0.0
        or args.boundary_tail_fixed_frontier_weight > 0.0
        or args.boundary_tail_np_weight > 0.0
        or args.same_instance_cross_view_recurrence_selected_fraction > 0.0
    )
    if needs_initial_runtime_features:
        initial_runtime_features, _initial_coefficients, _initial_diagnostics = (
            _runtime_features(
                model,
                geometry,
                relation_tensors,
                local_ids,
                structural_ids,
            )
        )
    else:
        initial_runtime_features = None
    if (
        args.tail_hard_pose_batch_fraction > 0.0
        or args.boundary_tail_fixed_frontier_weight > 0.0
        or args.boundary_tail_np_weight > 0.0
    ):
        assert initial_runtime_features is not None
        hard_pose_indices, hard_pose_meta, fixed_safety_frontier = (
            _mine_initial_tail_hard_poses(
                model,
                train_split,
                world_aabbs,
                initial_runtime_features,
                device,
                pool_fraction=args.tail_hard_pose_pool_fraction,
                seed=args.seed + 70_000,
                build_fixed_frontier=(
                    args.boundary_tail_fixed_frontier_weight > 0.0
                ),
                frontier_positive_mass_fraction=(
                    args.boundary_tail_residual_positive_tail_mass_fraction
                ),
                frontier_positive_count_cap=(
                    args.boundary_tail_fixed_frontier_positive_count_cap
                ),
                frontier_negative_top_fraction=(
                    args.boundary_tail_residual_negative_tail_fraction
                ),
                frontier_minimum_negative_count=(
                    args.boundary_tail_residual_min_negatives
                ),
                frontier_maximum_negative_count=(
                    args.boundary_tail_residual_max_negatives
                ),
            )
        )
        fixed_safety_frontier = {
            int(pose_index): (
                positive_ids.to(device),
                negative_ids.to(device),
            )
            for pose_index, (positive_ids, negative_ids) in (
                fixed_safety_frontier.items()
            )
        }
    else:
        hard_pose_indices = None
        fixed_safety_frontier = {}
        hard_pose_meta = {
            "sourceSplit": "train",
            "enabled": False,
            "testRead": False,
        }
    weighted_np_boundary_logit: torch.nn.Parameter | None = None
    weighted_np_dual_state: WeightedNeymanPearsonDualState | None = None
    weighted_np_initial_boundary_logit: float | None = None
    if args.boundary_tail_np_weight > 0.0:
        weighted_np_initial_boundary_logit = float(
            hard_pose_meta["globalWeightedPositiveQ01Logit"]
        )
        if not math.isfinite(weighted_np_initial_boundary_logit):
            raise FloatingPointError(
                "train-only weighted-positive q1 boundary is non-finite"
            )
        weighted_np_boundary_logit = torch.nn.Parameter(
            torch.tensor(
                weighted_np_initial_boundary_logit,
                dtype=torch.float32,
                device=device,
            )
        )
        weighted_np_dual_state = WeightedNeymanPearsonDualState(
            multiplier=0.0,
            learning_rate=args.boundary_tail_np_dual_learning_rate,
            maximum=args.boundary_tail_np_dual_maximum,
        )
    if args.same_instance_cross_view_recurrence_selected_fraction > 0.0:
        assert initial_runtime_features is not None
        recurrence_priority_np, recurrence_priority_meta = (
            _mine_recurrent_hard_negative_prior(
                model,
                train_split,
                world_aabbs,
                initial_runtime_features,
                device,
                num_instances=num_instances,
                negative_tail_fraction=(
                    args.same_instance_cross_view_recurrence_negative_tail_fraction
                ),
                selected_instance_fraction=(
                    args.same_instance_cross_view_recurrence_selected_fraction
                ),
                minimum_negative_exposures=(
                    args.same_instance_cross_view_recurrence_min_negative_exposures
                ),
                minimum_hard_count=(
                    args.same_instance_cross_view_recurrence_min_hard_count
                ),
                smoothing_strength=(
                    args.same_instance_cross_view_recurrence_smoothing_strength
                ),
                seed=args.seed + 80_000,
            )
        )
        recurrence_priority = torch.from_numpy(recurrence_priority_np).to(device)
    else:
        recurrence_priority = None
        recurrence_priority_meta = {
            "enabled": False,
            "sourceSplit": "train",
            "runtimeCost": "none",
            "testRead": False,
        }
    if initial_runtime_features is not None:
        del initial_runtime_features, _initial_coefficients, _initial_diagnostics
    view_residual_refinement_meta = {
        "enabled": args.refinement_scope == "view_residual",
        "objective": "final-score exact weighted-positive/negative tail pairwise ranking",
        "globalShiftInvariant": True,
        "positiveGuardWeight": float(args.view_residual_positive_guard_weight),
        "positiveImportanceMix": float(
            args.view_residual_positive_importance_mix
        ),
        "residualAnchorWeight": float(args.view_residual_anchor_weight),
        "residualCenterWeight": float(args.view_residual_center_weight),
        "runtimeGate": "none",
        "runtimeAssets": "no additional per-instance assets",
        "testRead": False,
    }

    protocol = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4",
        "experiment": args.experiment_name,
        "variant": args.variant,
        "lossVariant": args.loss_variant,
        "initialization": initialization or {"mode": "from-scratch"},
        "rvlDiagnostics": {
            "bcePositiveWeight": float(args.rvl_bce_positive_weight),
            "tverskyFnWeight": float(args.rvl_tversky_fn_weight),
            "countWeight": float(args.rvl_count_weight),
            "fpNormalization": str(args.rvl_fp_normalization),
            "efficiencyWarmupFraction": float(args.efficiency_warmup_fraction),
            "efficiencyPrimaryFraction": float(args.efficiency_primary_fraction),
            "rankWeight": float(args.rvl_rank_weight),
            "rankNegativeTopK": int(args.rvl_rank_negative_top_k),
            "negativeBandTemperature": float(args.negative_band_temperature),
            "negativeBandShape": str(args.negative_band_shape),
            "budgetInitialScale": float(args.rvl_budget_initial_scale),
            "budgetStartFraction": float(args.rvl_budget_start_fraction),
            "budgetRampFraction": float(args.rvl_budget_ramp_fraction),
        },
        "refinementScope": refinement_scope_meta,
        "dualProbeRescue": dual_probe_rescue_meta,
        "viewConditionedResidual": {
            "enabled": args.view_residual_max_abs > 0.0,
            "maximumAbsoluteResidual": float(args.view_residual_max_abs),
            "hiddenDim": int(args.view_residual_hidden_dim),
            "initialization": "zero output layer",
            "runtimeAssetShapeChanged": False,
            "viewResidualTailRefinement": view_residual_refinement_meta,
        },
        "viewcellExtremeVisibility": {
            "enabled": bool(args.viewcell_extreme_visibility),
            "refinementOnly": (
                args.refinement_scope == "viewcell_extreme_visibility"
            ),
            "inputDim": 27,
            "input": (
                "17D analytic view-cell extrema + 9D center ray query + "
                "1D normalized depth"
            ),
            "fusion": "unbounded additive main visibility logit",
            "poseReduction": "none",
            "runtimeInstanceFeatureDim": 124,
            "runtimeAssetShapeChanged": False,
            "onlineNeighborQuery": False,
            "onlineSubposeExpansion": False,
            "testRead": False,
        },
        "viewcellRegionConditionedVisibility": {
            "enabled": bool(args.viewcell_region_conditioned_visibility),
            "refinementOnly": (
                args.refinement_scope
                in {
                    "viewcell_region_conditioned_visibility",
                    "viewcell_region_tail",
                }
            ),
            "regionInputDim": int(VIEWCELL_REGION_CONDITIONED_VISIBILITY_REGION_DIM),
            "input": (
                "17D analytic view-cell extrema + 9D center ray query + "
                "1D normalized depth"
            ),
            "queryAuxKey": "viewcell_extreme_features",
            "queryAuxFeatureDim": int(VIEWCELL_EXTREME_ENVELOPE_DIM),
            "hiddenInputDim": int(model.hidden_dim),
            "projectionDim": int(VIEWCELL_REGION_CONDITIONED_VISIBILITY_PROJECTION_DIM),
            "fusionDim": int(VIEWCELL_REGION_CONDITIONED_VISIBILITY_FUSION_DIM),
            "headHiddenDim": int(VIEWCELL_REGION_CONDITIONED_VISIBILITY_HEAD_DIM),
            "activation": "SiLU",
            "fusion": (
                "concat(region_projection, hidden_projection, "
                "region_projection * hidden_projection)"
            ),
            "output": "unbounded additive main visibility logit",
            "outputInitialization": "zero weight and bias",
            "centering": str(
                args.viewcell_region_conditioned_visibility_centering
            ),
            "poseReduction": (
                "candidate_mean_per_pose"
                if args.viewcell_region_conditioned_visibility_centering
                == "pose_mean"
                else "none"
            ),
            "runtimeReduction": (
                "candidate_mean_per_pose"
                if args.viewcell_region_conditioned_visibility_centering
                == "pose_mean"
                else "none"
            ),
            "boundedCorrection": False,
            "runtimeInstanceFeatureDim": int(RUNTIME_FEATURE_DIM),
            "runtimeAssets": "no additional per-instance asset",
            "runtimeAssetShapeChanged": False,
            "onlineNeighborQuery": False,
            "onlineSubposeExpansion": False,
            "tailRefinement": {
                "enabled": args.refinement_scope == "viewcell_region_tail",
                "trainablePath": "region-conditioned branch only",
                "baseScore": "frozen initial-checkpoint visibility head",
                "selectionScore": "same frozen initial visibility score per train pose",
                "residual": (
                    "unbounded region-conditioned visibility logit, centered by "
                    "the candidate-set mean within each pose before application"
                    if args.viewcell_region_conditioned_visibility_centering
                    == "pose_mean"
                    else "unbounded additive region-conditioned visibility logit"
                ),
                "objective": (
                    "pose-local partial-AUC plus symmetric tail correction and residual regularization"
                ),
                "objectiveIsolation": (
                    "no RVL, relation, schedule, efficiency, or visibility-head gradient"
                ),
                "positiveImportanceTransform": str(
                    args.boundary_tail_residual_positive_importance_transform
                ),
                "positiveImportancePower": float(
                    args.boundary_tail_residual_positive_importance_power
                ),
                "crossPosePairWeight": float(
                    args.boundary_tail_residual_cross_pose_pair_weight
                ),
                "globalTailPairWeight": float(
                    args.boundary_tail_residual_global_tail_pair_weight
                ),
                "runtimeAssetShapeChanged": False,
                "testRead": False,
            },
            "testRead": False,
        },
        "viewcellBoundaryOpportunity": {
            "enabled": args.boundary_opportunity_hidden_dim > 0,
            "refinementOnly": args.refinement_scope == "boundary_opportunity",
            "hiddenDim": int(args.boundary_opportunity_hidden_dim),
            "projectionDim": int(args.boundary_opportunity_projection_dim),
            "initialLogit": float(args.boundary_opportunity_initial_logit),
            "maximumLogitUplift": float(
                args.boundary_opportunity_max_logit_uplift
            ),
            "lossWeight": float(args.boundary_opportunity_loss_weight),
            "rarePositiveHitRateThreshold": float(
                args.boundary_opportunity_rare_threshold
            ),
            "positiveTailMassFraction": float(
                args.boundary_opportunity_positive_tail_mass_fraction
            ),
            "positiveTailCountCapFraction": float(
                args.boundary_opportunity_positive_tail_count_cap_fraction
            ),
            "minimumPositivesPerPose": int(
                args.boundary_opportunity_min_positives
            ),
            "allNegativeUpliftWeight": float(
                args.boundary_opportunity_all_negative_uplift_weight
            ),
            "hardNegativeUpliftWeight": float(
                args.boundary_opportunity_hard_negative_uplift_weight
            ),
            "negativeTopFraction": float(
                args.boundary_opportunity_negative_tail_fraction
            ),
            "minimumNegativesPerPose": int(
                args.boundary_opportunity_min_negatives
            ),
            "margin": float(args.boundary_opportunity_margin),
            "temperature": float(args.boundary_opportunity_temperature),
            "scaleRampFraction": float(
                args.boundary_opportunity_scale_ramp_fraction
            ),
            "baseModelFrozen": args.refinement_scope == "boundary_opportunity",
            "trainingPosePool": boundary_opportunity_pose_meta,
            "subposeSupervisionRequired": True,
            "positiveSelection": "frozen-base weighted lower tail",
            "fusion": "one-way bounded logit uplift",
            "runtimeAssetShapeChanged": False,
            "testRead": False,
        },
        "viewcellBoundaryTailPartialAuc": {
            "enabled": args.boundary_tail_residual_hidden_dim > 0,
            "refinementOnly": args.refinement_scope == "boundary_tail_residual",
            "hiddenDim": int(args.boundary_tail_residual_hidden_dim),
            "projectionDim": int(args.boundary_tail_residual_projection_dim),
            "maximumAbsoluteResidual": float(
                args.boundary_tail_residual_max_abs
            ),
            "centering": str(args.boundary_tail_residual_centering),
            "shortcut": str(args.boundary_tail_residual_shortcut),
            "fusionMode": str(args.boundary_tail_residual_fusion),
            "outputInitializationStd": float(
                args.boundary_tail_residual_output_init_std
            ),
            "lossWeight": float(args.boundary_tail_residual_loss_weight),
            "rarePositiveHitRateThreshold": float(
                args.boundary_tail_residual_rare_threshold
            ),
            "positiveTailMassFraction": float(
                args.boundary_tail_residual_positive_tail_mass_fraction
            ),
            "positiveTailCountCap": int(
                args.boundary_tail_residual_positive_tail_count_cap
            ),
            "minimumPositivesPerPose": int(
                args.boundary_tail_residual_min_positives
            ),
            "negativeTopFraction": float(
                args.boundary_tail_residual_negative_tail_fraction
            ),
            "minimumNegativesPerPose": int(
                args.boundary_tail_residual_min_negatives
            ),
            "maximumNegativesPerPose": int(
                args.boundary_tail_residual_max_negatives
            ),
            "margin": float(args.boundary_tail_residual_margin),
            "temperature": float(args.boundary_tail_residual_temperature),
            "poseCvarFraction": float(
                args.boundary_tail_residual_pose_cvar_fraction
            ),
            "poseCvarWeight": float(
                args.boundary_tail_residual_pose_cvar_weight
            ),
            "negativePositiveResidualWeight": float(
                args.boundary_tail_residual_negative_positive_weight
            ),
            "positiveNegativeResidualWeight": float(
                args.boundary_tail_residual_positive_negative_weight
            ),
            "allPositiveNegativeResidualWeight": float(
                args.boundary_tail_residual_all_positive_negative_weight
            ),
            "tailClassificationWeight": float(
                args.boundary_tail_residual_classification_weight
            ),
            "tailClassificationMargin": float(
                args.boundary_tail_residual_classification_margin
            ),
            "tailClassificationDefinition": (
                "class-balanced selected positive residual >= margin and "
                "selected negative residual <= -margin"
            ),
            "residualRegularizationWeight": float(
                args.boundary_tail_residual_regularization_weight
            ),
            "positiveImportanceTransform": str(
                args.boundary_tail_residual_positive_importance_transform
            ),
            "positiveImportancePower": float(
                args.boundary_tail_residual_positive_importance_power
            ),
            "crossPosePairWeight": float(
                args.boundary_tail_residual_cross_pose_pair_weight
            ),
            "globalTailPairWeight": float(
                args.boundary_tail_residual_global_tail_pair_weight
            ),
            "objective": str(args.boundary_tail_residual_objective),
            "boundaryDeficitHuberBeta": float(
                args.boundary_tail_residual_deficit_huber_beta
            ),
            "boundaryDeficitGapWeight": float(
                args.boundary_tail_residual_deficit_gap_weight
            ),
            "baseModelFrozen": args.refinement_scope == "boundary_tail_residual",
            "trainingPosePool": boundary_opportunity_pose_meta,
            "selectionScoreSource": str(
                args.boundary_tail_residual_selection_source
            ),
            "positiveSelection": (
                f"{args.boundary_tail_residual_selection_source} weighted lower tail"
            ),
            "negativeSelection": (
                f"{args.boundary_tail_residual_selection_source} same-pose upper tail"
            ),
            "fusion": (
                "bounded signed residual with exact per-pose mean removal"
                if args.boundary_tail_residual_centering == "pose_mean"
                else "bounded signed residual without candidate-set reduction"
            ),
            "runtimeAssetShapeChanged": False,
            "runtimeReduction": (
                "one scalar candidate-set mean"
                if args.boundary_tail_residual_centering == "pose_mean"
                else "none"
            ),
            "testRead": False,
        },
        "queryTailSeparation": {
            "enabled": query_tail_enabled,
            "family": str(args.query_tail_separator_family),
            "inputDim": int(DUAL_PROBE_RAW_QUERY_DIM),
            "hiddenDim": int(args.query_tail_separator_hidden_dim),
            "maximumAbsoluteResidual": float(
                args.query_tail_separator_max_abs
            ),
            "centering": str(args.query_tail_separator_centering),
            "lossWeight": float(args.query_tail_separation_loss_weight),
            "positiveTailMassFraction": float(
                args.query_tail_positive_mass_fraction
            ),
            "positiveTailCountCap": int(args.query_tail_positive_count_cap),
            "minimumPositivesPerPose": int(args.query_tail_min_positives),
            "negativeTailFraction": float(args.query_tail_negative_fraction),
            "minimumNegativesPerPose": int(args.query_tail_min_negatives),
            "maximumNegativesPerPose": int(args.query_tail_max_negatives),
            "margin": float(args.query_tail_margin),
            "temperature": float(args.query_tail_temperature),
            "poseCvarFraction": float(args.query_tail_pose_cvar_fraction),
            "poseCvarWeight": float(args.query_tail_pose_cvar_weight),
            "negativePositiveResidualWeight": float(
                args.query_tail_negative_positive_weight
            ),
            "selectedPositiveNegativeResidualWeight": float(
                args.query_tail_positive_negative_weight
            ),
            "residualRegularizationWeight": float(
                args.query_tail_regularization_weight
            ),
            "positiveImportanceTransform": "power",
            "positiveImportancePower": float(
                args.query_tail_positive_importance_power
            ),
            "crossPosePairWeight": float(args.query_tail_cross_pose_pair_weight),
            "globalTailPairWeight": float(args.query_tail_global_pair_weight),
            "selectionScore": "detached pre-separator current visibility logit",
            "training": "joint from epoch one with the complete model",
            "initialCheckpoint": None,
            "refinementScope": str(args.refinement_scope),
            "subposeSupervisionSourceSplit": "train",
            "testRead": False,
        },
        "instanceExposureBalance": {
            "enabled": args.instance_exposure_balance_weight > 0.0,
            "weight": float(args.instance_exposure_balance_weight),
            **exposure_meta,
        },
        "sameInstanceCrossViewRank": {
            "enabled": args.same_instance_cross_view_rank_weight > 0.0,
            "weight": float(args.same_instance_cross_view_rank_weight),
            "marginLogit": float(args.same_instance_cross_view_margin),
            "temperatureLogit": float(args.same_instance_cross_view_temperature),
            "positiveImportanceMix": float(
                args.same_instance_cross_view_importance_mix
            ),
            "positiveGradientGroup": "safety",
            "negativeGradientGroup": (
                "safety"
                if args.refinement_scope == "boundary_tail_residual"
                else "efficiency"
            ),
            "gradientRouting": (
                "joint equal-and-opposite pair gradient during boundary-tail refinement"
                if args.refinement_scope == "boundary_tail_residual"
                else "split safety-positive and efficiency-negative gradients"
            ),
            "recurrentHardNegativePrior": {
                "enabled": (
                    args.same_instance_cross_view_recurrence_selected_fraction
                    > 0.0
                ),
                **recurrence_priority_meta,
            },
            "runtimeCost": "none",
        },
        "candidateBoundaryHardNegative": {
            "enabled": args.candidate_boundary_hard_negative_weight > 0.0,
            "weight": float(args.candidate_boundary_hard_negative_weight),
            "boundaryThreshold": float(args.candidate_boundary_threshold),
            "proximityTemperature": float(
                args.candidate_boundary_proximity_temperature
            ),
            "negativeTailFraction": float(
                args.candidate_boundary_negative_tail_fraction
            ),
            "selectionBias": float(args.candidate_boundary_selection_bias),
            "proximityGain": float(args.candidate_boundary_proximity_gain),
            "marginLogit": float(args.candidate_boundary_margin),
            "lossTemperatureLogit": float(
                args.candidate_boundary_loss_temperature
            ),
            "gradientGroup": "efficiency",
            "positiveGradient": False,
            "runtimeCost": "none",
        },
        "positiveTailCompactness": {
            "enabled": (
                args.loss_variant == "safety_reserve"
                and args.positive_tail_compactness_weight > 0.0
            ),
            "weight": float(args.positive_tail_compactness_weight),
            "tailMassFraction": float(args.positive_tail_mass_fraction),
            "referenceQuantile": float(args.positive_tail_reference_quantile),
            "allowedRelativeGap": float(args.positive_tail_allowed_relative_gap),
            "temperature": float(args.positive_tail_temperature),
            "rampFraction": float(args.positive_tail_ramp_fraction),
            "objectiveGroup": "safety",
            "referenceGradient": "detached",
            "referenceWeighting": "raw train-only visible weight mass",
            "scoreDomain": "translation-invariant raw logit gap",
            "positiveWeighting": "raw train-only visible weight mass",
        },
        "extremeTailSeparation": {
            "enabled": args.loss_variant == "safety_reserve" and args.tail_separation_weight > 0.0,
            "weight": float(args.tail_separation_weight),
            "positiveFraction": float(args.tail_positive_fraction),
            "negativeFraction": float(args.tail_negative_fraction),
            "margin": float(args.tail_margin),
            "temperature": float(args.tail_temperature),
            "poseCvarFraction": float(args.tail_pose_cvar_fraction),
            "poseCvarWeight": float(args.tail_pose_cvar_weight),
            "positiveImportanceMix": float(args.tail_positive_importance_mix),
            "positiveGradientScale": float(args.tail_positive_gradient_scale),
            "uniformCoverage": {
                "weight": float(args.coverage_tail_separation_weight),
                "positiveFraction": float(
                    args.coverage_tail_positive_fraction
                ),
                "positiveImportanceMix": 0.0,
            },
            "objectiveGroup": str(args.tail_objective_group),
            "rampFraction": float(args.tail_ramp_fraction),
            "scoreDomain": "per-pose robust-range normalized logit",
            "marginDomain": "per-pose robust-range normalized logit",
            "positiveWeighting": (
                f"{1.0 - float(args.tail_positive_importance_mix):.6g} uniform positive mass + "
                f"{float(args.tail_positive_importance_mix):.6g} log1p-normalized visible-weight mass"
            ),
            "aggregation": "per-pose mean plus upper-pose-CVaR",
            "selectionScoreSource": str(args.tail_selection_source),
            "selectionGradient": "detached",
            "initialVisibilitySelector": (
                "frozen initial-checkpoint visibility head over frozen query features"
                if args.tail_selection_source == "initial_visibility"
                else "disabled"
            ),
            "selectionCheckpointRule": (
                "fixed terminal or registered epoch; calibration-selected best aliases rejected"
                if args.tail_selection_source == "initial_visibility"
                else "not applicable"
            ),
        },
        "tailHardPoseSampling": {
            "enabled": args.tail_hard_pose_batch_fraction > 0.0,
            "batchFraction": float(args.tail_hard_pose_batch_fraction),
            "poolFraction": float(args.tail_hard_pose_pool_fraction),
            "uniformBatchFraction": float(1.0 - args.tail_hard_pose_batch_fraction),
            **hard_pose_meta,
        },
        "fixedWeightedSafetyFrontier": {
            "enabled": args.boundary_tail_fixed_frontier_weight > 0.0,
            "weight": float(args.boundary_tail_fixed_frontier_weight),
            "scoreSource": "initialized checkpoint final visibility logit",
            "membershipGradient": "detached and fixed before refinement",
            "pairObjective": "translation-invariant positive-minus-negative logit gap",
            "runtimeCost": "none",
            **dict(hard_pose_meta.get("fixedFrontier", {})),
        },
        "weightedNeymanPearsonOperatingPoint": {
            "enabled": args.boundary_tail_np_weight > 0.0,
            "weight": float(args.boundary_tail_np_weight),
            "weightedRecallTarget": float(
                args.boundary_tail_np_weighted_recall_target
            ),
            "temperatureLogit": float(args.boundary_tail_np_temperature),
            "dualLearningRate": float(
                args.boundary_tail_np_dual_learning_rate
            ),
            "dualMaximum": float(args.boundary_tail_np_dual_maximum),
            "initialBoundaryLogit": weighted_np_initial_boundary_logit,
            "initialBoundarySource": (
                "train-only global weighted-positive q1 logit"
                if args.boundary_tail_np_weight > 0.0
                else "disabled"
            ),
            "objective": (
                "soft false-positive rate plus a dual weighted-recall constraint"
            ),
            "runtimeExport": "none; train-only nuisance boundary is discarded",
            "calibrationAuthority": "checkpoint-own calibration split only",
            "testRead": False,
        },
        "safetyBoundaryNegativeExcess": {
            "enabled": (
                args.loss_variant == "safety_reserve"
                and args.safety_boundary_excess_weight > 0.0
            ),
            "weight": float(args.safety_boundary_excess_weight),
            "scope": str(args.safety_boundary_scope),
            "positiveMassFraction": float(args.safety_boundary_positive_mass_fraction),
            "negativeTailFraction": float(args.safety_boundary_negative_tail_fraction),
            "margin": float(args.safety_boundary_margin),
            "temperature": float(args.safety_boundary_temperature),
            "poseCvarFraction": float(args.safety_boundary_pose_cvar_fraction),
            "poseCvarWeight": float(args.safety_boundary_pose_cvar_weight),
            "rampFraction": float(args.safety_boundary_ramp_fraction),
            "emaDecay": float(args.safety_boundary_ema_decay),
            "emaRule": (
                "min(current_batch_boundary, train_batch_ema_boundary)"
                if args.safety_boundary_ema_decay > 0.0
                else "disabled"
            ),
            "objectiveGroup": "efficiency",
            "positiveGradient": "detached weighted-recall boundary",
            "positiveWeighting": "raw train-only visible weight mass",
            "scopeSemantics": (
                "minibatch surrogate using aggregate weighted-recall mass definition"
                if args.safety_boundary_scope == "batch"
                else "pose-balanced diagnostic surrogate"
            ),
        },
        "seed": int(args.seed),
        "testRead": False,
        "thresholdSource": "checkpoint-own-calibration-only",
        "diagnosticPoseSubset": {
            "maxPoses": int(args.max_eval_poses),
            "calibrationSeed": int(args.seed) + 50_000,
            "validationSeed": int(args.seed) + 60_000,
            "fixedAcrossEpochs": True,
        },
        "operatingThresholdSampling": {
            "source": "explicit-or-model-operating-seed-and-global-step-only",
            "seed": int(operating_threshold_seed),
            "range": [float(value) for value in OPERATING_THRESHOLD_RANGE],
            "sampleCount": 2,
            "anchors": [float(value) for value in OPERATING_THRESHOLD_ANCHORS],
            "anchorPeriod": 4,
        },
        "candidateUnion": False,
        "fov": {"candidateAndModelDeg": 66.0, "realRenderDeg": 60.0},
        "viewcell": {
            "shape": "horizontal_disk",
            "radiusM": viewcell_radius_m,
            "candidateCameraSemantics": "66-degree back-camera candidate identity only",
            "queryCenterSemantics": "center of the same-direction view-cell visibility union",
        },
        "splitNames": {
            "train": train_name,
            "calibration": calibration_name,
            "validation": validation_name,
        },
        "splitPoseCounts": {
            "train": int(train_split.pose_indices.size),
            "calibration": int(calibration_split.pose_indices.size),
            "validation": int(validation_split.pose_indices.size),
        },
        "candidateDigests": {
            "train": candidate_digest_for_pose_sequence(dataset, train_split.pose_indices),
            "calibration": candidate_digest_for_pose_sequence(dataset, calibration_split.pose_indices),
            "validation": candidate_digest_for_pose_sequence(dataset, validation_split.pose_indices),
        },
        "dataset": {
            "path": str(args.dataset_dir.resolve()),
            "metaSha256": _sha256(args.dataset_dir / "dataset_meta.json"),
        },
        "runtimeMeta": {
            "path": str(args.runtime_meta.resolve()),
            "sha256": _sha256(args.runtime_meta),
        },
        "relation": relation_provenance,
        "sampler": sampler.manifest(),
        "instanceCalibration": {
            "mode": args.instance_calibration_mode,
            "maximumAbsoluteResidual": float(args.instance_calibration_max_abs),
            "sparseInstancePenalty": float(args.sparse_instance_penalty),
            "regularizationWeight": float(
                args.instance_calibration_regularization_weight
            ),
            "warmupFraction": float(args.instance_calibration_warmup_fraction),
            "rampFraction": float(args.instance_calibration_ramp_fraction),
            "reliability": calibration_reliability_meta,
            "runtimeExport": "fused_coefficients_only",
        },
        "cullCertificate": {
            "enabled": model.cull_certificate_head is not None,
            "headOnlyRefinement": bool(args.cull_certificate_head_only),
            "maximumSuppressionLogit": float(
                args.cull_certificate_max_suppression
            ),
            "hiddenDim": int(args.cull_certificate_hidden_dim),
            "inputMode": str(args.cull_certificate_input_mode),
            "initialSuppressionLogit": float(
                args.cull_certificate_initial_suppression
            ),
            "regularizationWeight": float(
                args.cull_certificate_regularization_weight
            ),
            "tailClassificationWeight": float(
                args.cull_certificate_tail_loss_weight
            ),
            "positiveGuardWeight": float(
                args.cull_certificate_positive_guard_weight
            ),
            "negativeTailFraction": float(
                args.cull_certificate_negative_tail_fraction
            ),
            "positiveUniformMix": float(
                args.cull_certificate_positive_uniform_mix
            ),
            "rareSubposePositiveWeight": float(
                args.cull_certificate_rare_positive_weight
            ),
            "centerAnchorWeight": float(
                args.cull_certificate_center_anchor_weight
            ),
            "poseTailPair": {
                "weight": float(args.cull_certificate_pair_weight),
                "positiveTailFraction": float(
                    args.cull_certificate_pair_positive_fraction
                ),
                "negativeTailFraction": float(
                    args.cull_certificate_pair_negative_fraction
                ),
                "marginLogit": float(args.cull_certificate_pair_margin),
                "temperatureLogit": float(
                    args.cull_certificate_pair_temperature
                ),
                "poseCvarFraction": float(
                    args.cull_certificate_pair_pose_cvar_fraction
                ),
                "poseCvarWeight": float(
                    args.cull_certificate_pair_pose_cvar_weight
                ),
                "positiveGradientGroup": "safety",
                "negativeGradientGroup": "efficiency",
            },
            "efficiencyBaseDetached": model.cull_certificate_head is not None,
            "sharedHiddenDetached": model.cull_certificate_head is not None,
        },
    }
    if protocol["candidateDigests"]["train"] != relation_provenance["candidateDigest"]:
        raise ValueError("relation and PoseCSR train candidate hashes differ")

    _prepare_output(args.output_dir)
    manifest = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-moment-run-manifest-v4",
        "createdAtUnix": time.time(),
        "arguments": vars(args),
        "modelConfig": model.config,
        "protocol": protocol,
        "geometry": geometry_meta,
        "glbBytes": {
            "count": num_glbs,
            "sum": float(glb_bytes_np.sum()),
            "syntheticMissingAllowed": bool(args.allow_missing_glb_costs),
        },
        "device": {
            "type": str(device),
            "cudaName": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "testRead": False,
    }
    _write_json(args.output_dir / "run_manifest.json", manifest)
    _write_json(args.output_dir / "model_schema.json", model.export_schema())

    parameters = [value for value in model.parameters() if value.requires_grad]
    if weighted_np_boundary_logit is not None:
        parameters.append(weighted_np_boundary_logit)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    total_steps = int(args.epochs) * int(args.steps_per_epoch)
    history: list[dict[str, Any]] = []
    metrics_path = args.output_dir / "train_metrics.jsonl"
    best_safe_key: tuple[float, ...] | None = None
    best_diagnostic_key: tuple[float, ...] | None = None
    best_safe: dict[str, Any] | None = None
    best_diagnostic: dict[str, Any] | None = None
    best_safe_calibration: dict[str, Any] | None = None
    best_diagnostic_calibration: dict[str, Any] | None = None
    best_safe_validation: dict[str, Any] | None = None
    best_diagnostic_validation: dict[str, Any] | None = None
    safety_boundary_ema: torch.Tensor | None = None
    global_step = 0
    started = time.time()

    for epoch in range(int(args.epochs)):
        model.train()
        sampler.start_epoch(epoch)
        rng = np.random.default_rng(int(args.seed) + epoch * 1009)
        epoch_metrics: dict[str, list[float]] = {}
        processed_steps = 0
        sampling_pose_indices, sampling_pose_fraction = _refinement_pose_sampling(
            args.refinement_scope,
            boundary_opportunity_pose_indices,
            hard_pose_indices,
            args.tail_hard_pose_batch_fraction,
        )
        pose_batches = train_split.pose_set_batches(
            args.poses_per_batch,
            rng,
            max_steps=args.steps_per_epoch,
            include_empty=False,
            hard_pose_indices=sampling_pose_indices,
            hard_pose_fraction=sampling_pose_fraction,
        )
        for step, pose_indices in enumerate(pose_batches):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            batch = train_split.build_pose_set_batch(
                pose_indices,
                world_aabbs,
                rng,
                max_candidates_per_pose=0,
                allow_candidate_visible_union=False,
                include_empty=False,
            )
            if batch["instance"].size == 0:
                continue
            processed_steps += 1
            global_step += 1
            optimizer.zero_grad(set_to_none=True)
            if refinement_calibration_blend is None:
                model.set_instance_calibration_blend(
                    _instance_calibration_blend(
                        global_step - 1,
                        total_steps,
                        warmup_fraction=args.instance_calibration_warmup_fraction,
                        ramp_fraction=args.instance_calibration_ramp_fraction,
                    )
                )
            else:
                model.set_instance_calibration_blend(refinement_calibration_blend)
            if model.boundary_opportunity_head is not None:
                model.set_boundary_opportunity_scale(
                    _boundary_opportunity_scale(
                        global_step,
                        total_steps,
                        ramp_fraction=(
                            args.boundary_opportunity_scale_ramp_fraction
                        ),
                    )
                )
            runtime_features, coefficients, coefficient_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
            camera = torch.from_numpy(batch["camera"]).to(device)
            camera_view = torch.from_numpy(batch["camera_view"]).to(device)
            candidate_camera = torch.from_numpy(batch["candidate_camera_world"]).to(device)
            query_center = torch.from_numpy(batch["query_center_world"]).to(device)
            viewcell_radius = torch.from_numpy(batch["viewcell_radius_m"]).to(device)
            instance_ids = torch.from_numpy(batch["instance"]).to(device)
            target = torch.from_numpy(batch["target"]).to(device)
            visible_weights = torch.from_numpy(batch["visible_weights"]).to(device)
            visible_hit_rates = torch.from_numpy(batch["visible_hit_rates"]).to(device)
            pose_offsets = torch.from_numpy(batch["pose_offsets"]).to(device)
            batch_pose_indices = torch.from_numpy(batch["pose_indices"]).to(device)
            logits, aux = model.compute_logits_with_aux(
                camera,
                camera_view,
                candidate_camera,
                instance_ids,
                runtime_features=runtime_features,
                query_center_world=query_center,
                viewcell_radius_m=viewcell_radius,
                pose_offsets=pose_offsets,
            )
            if args.refinement_scope == "dual_probe_rescue":
                assert dual_probe_rescue_spec is not None
                dual_probe_rescue_loss, dual_probe_rescue_parts = (
                    _dual_probe_rescue_batch_loss(
                        aux,
                        target,
                        visible_weights,
                        pose_offsets,
                        coverage_weight=float(
                            model.dual_probe_coverage_weight.detach().cpu()
                        ),
                        primary_keep_weight=(
                            args.dual_probe_rescue_primary_keep_weight
                        ),
                        coverage_keep_weight=(
                            args.dual_probe_rescue_coverage_keep_weight
                        ),
                        negative_decay_weight=(
                            args.dual_probe_rescue_negative_decay_weight
                        ),
                        high_negative_decay_weight=(
                            args.dual_probe_rescue_high_negative_decay_weight
                        ),
                        visible_weight_epsilon=(
                            args.dual_probe_rescue_visible_weight_epsilon
                        ),
                    )
                )
            else:
                dual_probe_rescue_loss = logits.sum() * 0.0
                dual_probe_rescue_parts = {
                    "lossDualProbeRescueAttenuation": dual_probe_rescue_loss,
                    "dualProbeRescueEnabled": False,
                    "dualProbeRescuePoseCount": 0.0,
                    "dualProbeRescueCandidateCount": 0.0,
                    "dualProbeRescuePoseOffsetsUsed": False,
                }
            if args.refinement_scope == "boundary_opportunity":
                (
                    boundary_opportunity_loss,
                    boundary_opportunity_parts,
                ) = viewcell_boundary_opportunity_loss(
                    aux["boundary_opportunity_raw_logit_uplift"],
                    aux["pre_opportunity_visibility_logits"],
                    target,
                    visible_hit_rates,
                    visible_weights,
                    pose_offsets,
                    rare_threshold=args.boundary_opportunity_rare_threshold,
                    positive_tail_mass_fraction=(
                        args.boundary_opportunity_positive_tail_mass_fraction
                    ),
                    positive_tail_count_cap_fraction=(
                        args.boundary_opportunity_positive_tail_count_cap_fraction
                    ),
                    negative_top_fraction=(
                        args.boundary_opportunity_negative_tail_fraction
                    ),
                    min_positive_count=args.boundary_opportunity_min_positives,
                    min_negative_count=args.boundary_opportunity_min_negatives,
                    positive_margin_weight=1.0,
                    all_negative_uplift_weight=(
                        args.boundary_opportunity_all_negative_uplift_weight
                    ),
                    hard_negative_uplift_weight=(
                        args.boundary_opportunity_hard_negative_uplift_weight
                    ),
                    margin=args.boundary_opportunity_margin,
                    temperature=args.boundary_opportunity_temperature,
                )
            else:
                boundary_opportunity_loss = logits.sum() * 0.0
                boundary_opportunity_parts = {
                    "lossViewcellBoundaryOpportunity": boundary_opportunity_loss,
                    "tailPositiveCount": 0.0,
                    "hardNegativeCount": 0.0,
                }
            if args.refinement_scope == "boundary_tail_residual":
                if args.boundary_tail_residual_objective == "boundary_deficit":
                    boundary_tail_loss, boundary_tail_parts = (
                        viewcell_tail_boundary_deficit_loss(
                            aux["boundary_tail_residual_centered"],
                            aux["pre_tail_residual_visibility_logits"],
                            target,
                            visible_hit_rates,
                            visible_weights,
                            pose_offsets,
                            positive_mass_fraction=(
                                args.boundary_tail_residual_positive_tail_mass_fraction
                            ),
                            positive_count_cap=(
                                args.boundary_tail_residual_positive_tail_count_cap
                            ),
                            min_positive_count=(
                                args.boundary_tail_residual_min_positives
                            ),
                            negative_fraction=(
                                args.boundary_tail_residual_negative_tail_fraction
                            ),
                            min_negative_count=(
                                args.boundary_tail_residual_min_negatives
                            ),
                            negative_count_cap=(
                                args.boundary_tail_residual_max_negatives
                            ),
                            margin=args.boundary_tail_residual_margin,
                            temperature=args.boundary_tail_residual_temperature,
                            huber_beta=(
                                args.boundary_tail_residual_deficit_huber_beta
                            ),
                            gap_weight=(
                                args.boundary_tail_residual_deficit_gap_weight
                            ),
                            residual_l2_weight=(
                                args.boundary_tail_residual_regularization_weight
                            ),
                        )
                    )
                else:
                    boundary_tail_loss, boundary_tail_parts = (
                        viewcell_tail_partial_auc_loss(
                            aux["boundary_tail_residual_centered"],
                            aux["pre_tail_residual_visibility_logits"],
                            target,
                            visible_hit_rates,
                            visible_weights,
                            pose_offsets,
                            tail_selection_logits=_boundary_tail_selection_logits(
                                args.boundary_tail_residual_selection_source,
                                logits,
                            ),
                            positive_tail_mass_fraction=(
                                args.boundary_tail_residual_positive_tail_mass_fraction
                            ),
                            positive_tail_count_cap=(
                                args.boundary_tail_residual_positive_tail_count_cap
                            ),
                            min_positive_count=(
                                args.boundary_tail_residual_min_positives
                            ),
                            negative_top_fraction=(
                                args.boundary_tail_residual_negative_tail_fraction
                            ),
                            min_negative_count=(
                                args.boundary_tail_residual_min_negatives
                            ),
                            max_negative_count=(
                                args.boundary_tail_residual_max_negatives
                            ),
                            margin=args.boundary_tail_residual_margin,
                            temperature=args.boundary_tail_residual_temperature,
                            pose_cvar_fraction=(
                                args.boundary_tail_residual_pose_cvar_fraction
                            ),
                            pose_cvar_weight=(
                                args.boundary_tail_residual_pose_cvar_weight
                            ),
                            negative_positive_residual_weight=(
                                args.boundary_tail_residual_negative_positive_weight
                            ),
                            selected_positive_negative_residual_weight=(
                                args.boundary_tail_residual_positive_negative_weight
                            ),
                            all_positive_negative_residual_weight=(
                                args.boundary_tail_residual_all_positive_negative_weight
                            ),
                            tail_classification_weight=(
                                args.boundary_tail_residual_classification_weight
                            ),
                            tail_classification_margin=(
                                args.boundary_tail_residual_classification_margin
                            ),
                            residual_l2_weight=(
                                args.boundary_tail_residual_regularization_weight
                            ),
                            positive_importance_transform=(
                                args.boundary_tail_residual_positive_importance_transform
                            ),
                            positive_importance_power=(
                                args.boundary_tail_residual_positive_importance_power
                            ),
                            cross_pose_pair_weight=(
                                args.boundary_tail_residual_cross_pose_pair_weight
                            ),
                            global_tail_pair_weight=(
                                args.boundary_tail_residual_global_tail_pair_weight
                            ),
                        )
                    )
            elif args.refinement_scope == "viewcell_region_tail":
                region_base_logits = _extreme_tail_selection_logits(
                    args.tail_selection_source,
                    logits,
                    aux["query_features"],
                    initial_visibility_weight,
                    initial_visibility_bias,
                )
                boundary_tail_loss, boundary_tail_parts = (
                    viewcell_tail_partial_auc_loss(
                        aux["viewcell_region_conditioned_visibility_logit"],
                        region_base_logits,
                        target,
                        visible_hit_rates,
                        visible_weights,
                        pose_offsets,
                        tail_selection_logits=region_base_logits,
                        positive_tail_mass_fraction=(
                            args.boundary_tail_residual_positive_tail_mass_fraction
                        ),
                        positive_tail_count_cap=(
                            args.boundary_tail_residual_positive_tail_count_cap
                        ),
                        min_positive_count=(
                            args.boundary_tail_residual_min_positives
                        ),
                        negative_top_fraction=(
                            args.boundary_tail_residual_negative_tail_fraction
                        ),
                        min_negative_count=(
                            args.boundary_tail_residual_min_negatives
                        ),
                        max_negative_count=(
                            args.boundary_tail_residual_max_negatives
                        ),
                        margin=args.boundary_tail_residual_margin,
                        temperature=args.boundary_tail_residual_temperature,
                        pose_cvar_fraction=(
                            args.boundary_tail_residual_pose_cvar_fraction
                        ),
                        pose_cvar_weight=(
                            args.boundary_tail_residual_pose_cvar_weight
                        ),
                        negative_positive_residual_weight=(
                            args.boundary_tail_residual_negative_positive_weight
                        ),
                        selected_positive_negative_residual_weight=(
                            args.boundary_tail_residual_positive_negative_weight
                        ),
                        all_positive_negative_residual_weight=(
                            args.boundary_tail_residual_all_positive_negative_weight
                        ),
                        tail_classification_weight=(
                            args.boundary_tail_residual_classification_weight
                        ),
                        tail_classification_margin=(
                            args.boundary_tail_residual_classification_margin
                        ),
                        residual_l2_weight=(
                            args.boundary_tail_residual_regularization_weight
                        ),
                        positive_importance_transform=(
                            args.boundary_tail_residual_positive_importance_transform
                        ),
                        positive_importance_power=(
                            args.boundary_tail_residual_positive_importance_power
                        ),
                        cross_pose_pair_weight=(
                            args.boundary_tail_residual_cross_pose_pair_weight
                        ),
                        global_tail_pair_weight=(
                            args.boundary_tail_residual_global_tail_pair_weight
                        ),
                    )
                )
            else:
                boundary_tail_loss = logits.sum() * 0.0
                boundary_tail_parts = {
                    "lossViewcellTailObjective": boundary_tail_loss,
                    "poseCount": 0.0,
                    "selectedPositiveCount": 0.0,
                    "selectedNegativeCount": 0.0,
                }
            if query_tail_enabled:
                query_tail_loss, query_tail_parts = viewcell_tail_partial_auc_loss(
                    aux["query_tail_separator_residual"],
                    aux["pre_query_tail_separator_visibility_logits"],
                    target,
                    visible_hit_rates,
                    visible_weights,
                    pose_offsets,
                    tail_selection_logits=(
                        aux["pre_query_tail_separator_visibility_logits"]
                    ),
                    positive_tail_mass_fraction=(
                        args.query_tail_positive_mass_fraction
                    ),
                    positive_tail_count_cap=args.query_tail_positive_count_cap,
                    min_positive_count=args.query_tail_min_positives,
                    negative_top_fraction=args.query_tail_negative_fraction,
                    min_negative_count=args.query_tail_min_negatives,
                    max_negative_count=args.query_tail_max_negatives,
                    margin=args.query_tail_margin,
                    temperature=args.query_tail_temperature,
                    pose_cvar_fraction=args.query_tail_pose_cvar_fraction,
                    pose_cvar_weight=args.query_tail_pose_cvar_weight,
                    negative_positive_residual_weight=(
                        args.query_tail_negative_positive_weight
                    ),
                    selected_positive_negative_residual_weight=(
                        args.query_tail_positive_negative_weight
                    ),
                    all_positive_negative_residual_weight=0.0,
                    tail_classification_weight=0.0,
                    residual_l2_weight=args.query_tail_regularization_weight,
                    positive_importance_transform="power",
                    positive_importance_power=(
                        args.query_tail_positive_importance_power
                    ),
                    cross_pose_pair_weight=(
                        args.query_tail_cross_pose_pair_weight
                    ),
                    global_tail_pair_weight=args.query_tail_global_pair_weight,
                )
            else:
                query_tail_loss = logits.sum() * 0.0
                query_tail_parts = {
                    "lossViewcellTailPartialAuc": query_tail_loss,
                    "poseCount": 0.0,
                    "selectedPositiveCount": 0.0,
                    "selectedNegativeCount": 0.0,
                }
            if (
                args.refinement_scope == "boundary_tail_residual"
                and args.boundary_tail_fixed_frontier_weight > 0.0
            ):
                (
                    fixed_frontier_loss,
                    fixed_frontier_parts,
                ) = weighted_safety_frontier_pair_loss(
                    logits,
                    instance_ids,
                    visible_weights,
                    batch_pose_indices,
                    pose_offsets,
                    fixed_safety_frontier,
                    margin=args.boundary_tail_residual_margin,
                    temperature=args.boundary_tail_residual_temperature,
                    positive_importance_power=(
                        args.boundary_tail_residual_positive_importance_power
                    ),
                    pose_cvar_fraction=(
                        args.boundary_tail_residual_pose_cvar_fraction
                    ),
                    pose_cvar_weight=(
                        args.boundary_tail_residual_pose_cvar_weight
                    ),
                    batch_global_pair_weight=0.0,
                )
            else:
                fixed_frontier_loss = logits.sum() * 0.0
                fixed_frontier_parts = {
                    "lossWeightedSafetyFrontier": fixed_frontier_loss,
                    "lossPoseMean": fixed_frontier_loss,
                    "lossPoseCvar": fixed_frontier_loss,
                    "lossBatchGlobalPair": fixed_frontier_loss,
                    "poseCount": 0.0,
                    "positiveCount": 0.0,
                    "negativeCount": 0.0,
                    "pairCount": 0.0,
                    "meanGap": fixed_frontier_loss.detach(),
                    "worstGap": fixed_frontier_loss.detach(),
                    "globalPairCount": 0.0,
                }
            if weighted_np_boundary_logit is not None:
                assert weighted_np_dual_state is not None
                weighted_np_loss, weighted_np_parts = (
                    weighted_neyman_pearson_operating_loss(
                        logits,
                        target,
                        visible_weights,
                        weighted_np_boundary_logit,
                        args.boundary_tail_np_temperature,
                        weighted_np_dual_state.multiplier,
                        args.boundary_tail_np_weighted_recall_target,
                    )
                )
            else:
                weighted_np_loss = logits.sum() * 0.0
                weighted_np_parts = {
                    "lossWeightedNeymanPearsonOperating": weighted_np_loss,
                    "lossSoftFpr": weighted_np_loss,
                    "softWeightedRecall": logits.new_zeros(()),
                    "weightedRecallViolation": logits.new_zeros(()),
                    "dualPenalty": weighted_np_loss,
                    "boundaryLogit": logits.new_zeros(()),
                    "boundaryProbability": logits.new_zeros(()),
                    "dualMultiplier": 0.0,
                }
            efficiency_logits = (
                aux["base_logits"].detach()
                - aux["cull_certificate_suppression"]
                if model.cull_certificate_head is not None
                else None
            )

            use_safety_reserve = args.loss_variant == "safety_reserve"
            safety_boundary_reference: torch.Tensor | None = None
            if (
                use_safety_reserve
                and args.safety_boundary_excess_weight > 0.0
                and args.safety_boundary_ema_decay > 0.0
            ):
                if bool((target > 0.5).any()):
                    current_boundary = weighted_positive_safety_boundary_logit(
                        logits,
                        target,
                        visible_weights,
                        positive_mass_fraction=(
                            args.safety_boundary_positive_mass_fraction
                        ),
                    )
                    safety_boundary_ema = _update_safety_boundary_ema(
                        current_boundary,
                        safety_boundary_ema,
                        decay=args.safety_boundary_ema_decay,
                    )
                safety_boundary_reference = safety_boundary_ema
            visibility_training_logits = (
                aux["base_logits"]
                if model.cull_certificate_head is not None
                else aux["pre_opportunity_visibility_logits"]
                if args.refinement_scope == "boundary_opportunity"
                else logits
            )
            _combined_visibility, visibility_parts = safety_reserve_operating_utility_loss(
                visibility_training_logits,
                target,
                pose_offsets,
                visible_weights,
                visible_hit_rates,
                instance_ids,
                model.instance_to_glb,
                glb_bytes,
                train_seed=operating_threshold_seed,
                global_step=global_step - 1,
                total_optimizer_steps=total_steps,
                split="train",
                efficiency_logits=efficiency_logits,
                boundary_tail_weight=args.boundary_tail_weight if use_safety_reserve else 0.0,
                negative_band_weight=args.negative_band_weight if use_safety_reserve else 0.0,
                threshold_temperature=args.negative_band_temperature,
                negative_band_shape=args.negative_band_shape,
                glb_resource_weight=args.glb_resource_weight if use_safety_reserve else 0.0,
                warmup_fraction=args.efficiency_warmup_fraction,
                rvl_bce_positive_weight=args.rvl_bce_positive_weight,
                rvl_tversky_fn_weight=args.rvl_tversky_fn_weight,
                rvl_count_weight=args.rvl_count_weight,
                rvl_fp_normalization=args.rvl_fp_normalization,
                rvl_rank_weight=args.rvl_rank_weight,
                rvl_rank_negative_top_k=args.rvl_rank_negative_top_k,
                positive_tail_compactness_weight=(
                    args.positive_tail_compactness_weight
                    if use_safety_reserve
                    else 0.0
                ),
                positive_tail_mass_fraction=args.positive_tail_mass_fraction,
                positive_tail_reference_quantile=(
                    args.positive_tail_reference_quantile
                ),
                positive_tail_allowed_relative_gap=(
                    args.positive_tail_allowed_relative_gap
                ),
                positive_tail_temperature=args.positive_tail_temperature,
                positive_tail_ramp_fraction=args.positive_tail_ramp_fraction,
                tail_separation_weight=(
                    args.tail_separation_weight if use_safety_reserve else 0.0
                ),
                tail_selection_logits=_extreme_tail_selection_logits(
                    args.tail_selection_source,
                    visibility_training_logits,
                    aux["query_features"],
                    initial_visibility_weight,
                    initial_visibility_bias,
                ),
                tail_positive_fraction=args.tail_positive_fraction,
                tail_negative_fraction=args.tail_negative_fraction,
                tail_margin=args.tail_margin,
                tail_temperature=args.tail_temperature,
                tail_pose_cvar_fraction=args.tail_pose_cvar_fraction,
                tail_pose_cvar_weight=args.tail_pose_cvar_weight,
                tail_positive_importance_mix=args.tail_positive_importance_mix,
                tail_positive_gradient_scale=args.tail_positive_gradient_scale,
                coverage_tail_separation_weight=(
                    args.coverage_tail_separation_weight
                    if use_safety_reserve
                    else 0.0
                ),
                coverage_tail_positive_fraction=(
                    args.coverage_tail_positive_fraction
                ),
                tail_objective_group=args.tail_objective_group,
                tail_ramp_fraction=args.tail_ramp_fraction,
                safety_boundary_excess_weight=(
                    args.safety_boundary_excess_weight if use_safety_reserve else 0.0
                ),
                safety_boundary_scope=args.safety_boundary_scope,
                safety_boundary_positive_mass_fraction=(
                    args.safety_boundary_positive_mass_fraction
                ),
                safety_boundary_negative_tail_fraction=(
                    args.safety_boundary_negative_tail_fraction
                ),
                safety_boundary_margin=args.safety_boundary_margin,
                safety_boundary_temperature=args.safety_boundary_temperature,
                safety_boundary_pose_cvar_fraction=(
                    args.safety_boundary_pose_cvar_fraction
                ),
                safety_boundary_pose_cvar_weight=(
                    args.safety_boundary_pose_cvar_weight
                ),
                safety_boundary_ramp_fraction=args.safety_boundary_ramp_fraction,
                safety_boundary_reference_logit=safety_boundary_reference,
                rvl_budget_initial_scale=(
                    args.rvl_budget_initial_scale if use_safety_reserve else 1.0
                ),
                rvl_budget_start_fraction=(
                    args.rvl_budget_start_fraction if use_safety_reserve else 0.0
                ),
                rvl_budget_ramp_fraction=(
                    args.rvl_budget_ramp_fraction if use_safety_reserve else 0.0
                ),
            )
            efficiency_primary_fraction = float(args.efficiency_primary_fraction)
            if model.cull_certificate_head is not None:
                (
                    certificate_positive_guard,
                    certificate_hard_negative,
                    certificate_parts,
                ) = cull_certificate_tail_classification_loss(
                    aux["cull_certificate_raw"],
                    efficiency_logits,
                    target,
                    pose_offsets,
                    visible_weights,
                    visible_hit_rates=visible_hit_rates,
                    negative_tail_fraction=(
                        args.cull_certificate_negative_tail_fraction
                    ),
                    positive_uniform_mix=(
                        args.cull_certificate_positive_uniform_mix
                    ),
                    rare_positive_weight=(
                        args.cull_certificate_rare_positive_weight
                    ),
                )
                initial_certificate_ratio = (
                    float(args.cull_certificate_initial_suppression)
                    / float(args.cull_certificate_max_suppression)
                )
                initial_certificate_raw = math.log(
                    initial_certificate_ratio
                    / (1.0 - initial_certificate_ratio)
                )
                certificate_center_anchor = (
                    aux["cull_certificate_raw"].mean()
                    - initial_certificate_raw
                ).square()
                (
                    certificate_pair_positive,
                    certificate_pair_negative,
                    certificate_pair_parts,
                ) = cull_certificate_pose_tail_pair_loss(
                    logits,
                    target,
                    pose_offsets,
                    visible_weights,
                    positive_tail_fraction=(
                        args.cull_certificate_pair_positive_fraction
                    ),
                    negative_tail_fraction=(
                        args.cull_certificate_pair_negative_fraction
                    ),
                    margin=args.cull_certificate_pair_margin,
                    temperature=args.cull_certificate_pair_temperature,
                    pose_cvar_fraction=(
                        args.cull_certificate_pair_pose_cvar_fraction
                    ),
                    pose_cvar_weight=(
                        args.cull_certificate_pair_pose_cvar_weight
                    ),
                    positive_uniform_mix=(
                        args.cull_certificate_positive_uniform_mix
                    ),
                )
            else:
                certificate_positive_guard = logits.sum() * 0.0
                certificate_hard_negative = logits.sum() * 0.0
                certificate_center_anchor = logits.sum() * 0.0
                certificate_pair_positive = logits.sum() * 0.0
                certificate_pair_negative = logits.sum() * 0.0
                certificate_parts = {
                    "lossCullCertificatePositiveGuard": certificate_positive_guard,
                    "lossCullCertificateHardNegative": certificate_hard_negative,
                    "cullCertificatePositiveRawMean": logits.new_zeros(()),
                    "cullCertificateHardNegativeRawMean": logits.new_zeros(()),
                    "cullCertificatePositiveCount": 0.0,
                    "cullCertificateHardNegativeCount": 0.0,
                }
                certificate_pair_parts = {
                    "lossCullCertificatePairPositive": certificate_pair_positive,
                    "lossCullCertificatePairNegative": certificate_pair_negative,
                    "cullCertificatePairGap": logits.new_zeros(()),
                    "cullCertificatePairWorstGap": logits.new_zeros(()),
                    "cullCertificatePairViolationFraction": logits.new_zeros(()),
                    "cullCertificatePairPoseCount": 0.0,
                    "cullCertificatePairPositiveCount": 0.0,
                    "cullCertificatePairNegativeCount": 0.0,
                }
            certificate_pair_half_weight = 0.5 * float(
                args.cull_certificate_pair_weight
            )
            if args.instance_exposure_balance_weight > 0.0:
                (
                    exposure_positive_loss,
                    exposure_negative_loss,
                    exposure_parts,
                ) = instance_exposure_balanced_bce_loss(
                    logits,
                    target,
                    instance_ids,
                    exposure_positive_weights,
                    exposure_negative_weights,
                )
            else:
                exposure_positive_loss = logits.sum() * 0.0
                exposure_negative_loss = logits.sum() * 0.0
                exposure_parts = {
                    "lossInstanceExposurePositive": exposure_positive_loss,
                    "lossInstanceExposureNegative": exposure_negative_loss,
                    "instanceExposurePositiveCount": 0.0,
                    "instanceExposureNegativeCount": 0.0,
                }
            if args.same_instance_cross_view_rank_weight > 0.0:
                (
                    cross_view_positive_loss,
                    cross_view_negative_loss,
                    cross_view_parts,
                ) = same_instance_cross_view_rank_loss(
                    logits,
                    target,
                    instance_ids,
                    visible_weights,
                    recurrence_priority=(
                        recurrence_priority[instance_ids]
                        if recurrence_priority is not None
                        else None
                    ),
                    margin=args.same_instance_cross_view_margin,
                    temperature=args.same_instance_cross_view_temperature,
                    positive_importance_mix=(
                        args.same_instance_cross_view_importance_mix
                    ),
                )
            else:
                cross_view_positive_loss = logits.sum() * 0.0
                cross_view_negative_loss = logits.sum() * 0.0
                cross_view_parts = {
                    "lossSameInstanceCrossViewPositive": cross_view_positive_loss,
                    "lossSameInstanceCrossViewNegative": cross_view_negative_loss,
                    "sameInstanceCrossViewGap": logits.new_zeros(()),
                    "sameInstanceCrossViewWorstGap": logits.new_zeros(()),
                    "sameInstanceCrossViewViolationFraction": logits.new_zeros(()),
                    "sameInstanceCrossViewPairCount": 0.0,
                }
            if args.candidate_boundary_hard_negative_weight > 0.0:
                boundary_proximity = _candidate_frustum_boundary_proximity(
                    candidate_camera,
                    camera_view,
                    model.instance_world_aabbs[instance_ids],
                    boundary_threshold=args.candidate_boundary_threshold,
                    temperature=args.candidate_boundary_proximity_temperature,
                )
                (
                    candidate_boundary_loss,
                    candidate_boundary_parts,
                ) = candidate_boundary_hard_negative_loss(
                    logits,
                    target,
                    pose_offsets,
                    visible_weights,
                    boundary_proximity,
                    positive_mass_fraction=(
                        args.safety_boundary_positive_mass_fraction
                    ),
                    negative_tail_fraction=(
                        args.candidate_boundary_negative_tail_fraction
                    ),
                    selection_bias=args.candidate_boundary_selection_bias,
                    proximity_gain=args.candidate_boundary_proximity_gain,
                    margin=args.candidate_boundary_margin,
                    temperature=args.candidate_boundary_loss_temperature,
                )
            else:
                candidate_boundary_loss = logits.sum() * 0.0
                candidate_boundary_parts = {
                    "lossCandidateBoundaryHardNegative": candidate_boundary_loss,
                    "candidateBoundarySelectedProximity": logits.new_zeros(()),
                    "candidateBoundarySelectedNegativeCount": 0.0,
                    "candidateBoundaryPoseCount": 0.0,
                }
            if args.refinement_scope == "view_residual":
                (
                    view_residual_positive_guard,
                    view_residual_anchor,
                    view_residual_center,
                    view_residual_parts,
                ) = view_residual_tail_regularizers(
                    logits,
                    aux["base_logits"],
                    aux["view_conditioned_residual"],
                    target,
                    visible_weights,
                    positive_importance_mix=(
                        args.view_residual_positive_importance_mix
                    ),
                )
            else:
                view_residual_positive_guard = logits.sum() * 0.0
                view_residual_anchor = logits.sum() * 0.0
                view_residual_center = logits.sum() * 0.0
                view_residual_parts = {
                    "lossViewResidualPositiveGuard": view_residual_positive_guard,
                    "lossViewResidualAnchor": view_residual_anchor,
                    "lossViewResidualCenter": view_residual_center,
                    "viewResidualPositiveDropMean": logits.new_zeros(()),
                    "viewResidualPositiveDropFraction": logits.new_zeros(()),
                }
            exposure_half_weight = 0.5 * float(
                args.instance_exposure_balance_weight
            )
            cross_view_half_weight = 0.5 * float(
                args.same_instance_cross_view_rank_weight
            )
            if args.refinement_scope == "view_residual":
                safety_objective = (
                    visibility_parts["lossExtremeTailSeparationScaled"]
                    + float(args.view_residual_positive_guard_weight)
                    * view_residual_positive_guard
                    + float(args.view_residual_anchor_weight)
                    * view_residual_anchor
                    + float(args.view_residual_center_weight)
                    * view_residual_center
                    + float(args.boundary_tail_np_weight) * weighted_np_loss
                )
                efficiency_objective = logits.sum() * 0.0
            else:
                safety_objective = (
                    visibility_parts["lossSafety"]
                    + efficiency_primary_fraction
                    * visibility_parts["lossEfficiencyUngated"]
                    + float(args.cull_certificate_tail_loss_weight)
                    * float(args.cull_certificate_positive_guard_weight)
                    * certificate_positive_guard
                    + float(args.cull_certificate_center_anchor_weight)
                    * certificate_center_anchor
                    + certificate_pair_half_weight * certificate_pair_positive
                    + exposure_half_weight * exposure_positive_loss
                    + cross_view_half_weight * cross_view_positive_loss
                )
                efficiency_objective = (
                    (1.0 - efficiency_primary_fraction)
                    * visibility_parts["lossEfficiency"]
                    + float(args.cull_certificate_tail_loss_weight)
                    * certificate_hard_negative
                    + certificate_pair_half_weight * certificate_pair_negative
                    + exposure_half_weight * exposure_negative_loss
                    + cross_view_half_weight * cross_view_negative_loss
                    + float(args.candidate_boundary_hard_negative_weight)
                    * candidate_boundary_loss
                )
            certificate_regularization = (
                (
                    aux["cull_certificate_suppression"]
                    / float(args.cull_certificate_max_suppression)
                )
                .square()
                .mean()
                if model.cull_certificate_head is not None
                else logits.sum() * 0.0
            )
            efficiency_objective = efficiency_objective + float(
                args.cull_certificate_regularization_weight
            ) * certificate_regularization
            assert isinstance(safety_objective, torch.Tensor)
            assert isinstance(efficiency_objective, torch.Tensor)

            sample = sampler.sample_batch(args.observation_batch_size, step=step)
            sampled_observations = sampler.gather(sample, device=device)
            sampled_observations["normalized_depth"] = sampled_observations["depth"]
            survival_loss, survival_parts = stratified_survival_censoring_loss(
                model,
                coefficients,
                sampled_observations,
            )
            relation_loss, relation_parts = model.relation_consistency_loss(
                geometry,
                relation_tensors,
                seed=args.seed + global_step,
                max_edges=args.observation_batch_size,
            )
            utility_loss, utility_parts = visual_utility_loss(
                aux,
                target,
                visible_weights,
                pose_offsets,
                invisible_weight=0.25,
                rank_weight=0.30,
                rank_margin=0.10,
                rank_positive_top_k=32,
                rank_negative_top_k=128,
            )
            download_loss, download_parts = glb_priority_loss(
                aux["download_logits"],
                instance_ids,
                target,
                visible_weights,
                model.instance_to_glb,
                glb_cost_norm,
                pose_offsets,
            )
            regularization = model.regularization()
            instance_calibration_regularization = (
                model.instance_calibration_regularization()
            )
            if args.refinement_scope == "dual_probe_rescue":
                objective_groups = _dual_probe_rescue_objective_groups(
                    dual_probe_rescue_loss,
                    loss_weight=args.dual_probe_rescue_loss_weight,
                )
            elif args.refinement_scope == "boundary_opportunity":
                opportunity_objective = float(
                    args.boundary_opportunity_loss_weight
                ) * boundary_opportunity_loss
                zero_objective = opportunity_objective * 0.0
                objective_groups = {
                    "safety": opportunity_objective,
                    "relation": zero_objective,
                    "schedule": zero_objective,
                    "efficiency": zero_objective,
                    "total": opportunity_objective,
                }
            elif args.refinement_scope == "boundary_tail_residual":
                tail_objective = float(
                    args.boundary_tail_residual_loss_weight
                ) * boundary_tail_loss + float(
                    args.boundary_tail_fixed_frontier_weight
                ) * fixed_frontier_loss + float(
                    args.boundary_tail_np_weight
                ) * weighted_np_loss
                objective_groups = _boundary_tail_refinement_objective_groups(
                    tail_objective,
                    cross_view_half_weight
                    * (cross_view_positive_loss + cross_view_negative_loss),
                    tail_objective * 0.0,
                )
            elif args.refinement_scope == "viewcell_region_tail":
                tail_objective = float(
                    args.boundary_tail_residual_loss_weight
                ) * boundary_tail_loss
                zero_objective = tail_objective * 0.0
                objective_groups = {
                    "safety": tail_objective,
                    "relation": zero_objective,
                    "schedule": zero_objective,
                    "efficiency": zero_objective,
                    "total": tail_objective,
                }
            else:
                safety_objective = safety_objective + float(
                    args.query_tail_separation_loss_weight
                ) * query_tail_loss
                objective_groups = _v4_objective_groups(
                    safety_objective,
                    survival_loss,
                    relation_loss,
                    utility_loss,
                    download_loss,
                    regularization,
                    instance_calibration_regularization,
                    efficiency_objective,
                    survival_weight=args.survival_loss_weight,
                    relation_consistency_weight=args.relation_consistency_weight,
                    utility_weight=args.utility_loss_weight,
                    download_weight=args.download_loss_weight,
                    regularization_weight=args.regularization_weight,
                    instance_calibration_regularization_weight=(
                        args.instance_calibration_regularization_weight
                    ),
                )
            safety_objective = objective_groups["safety"]
            relation_objective = objective_groups["relation"]
            schedule_objective = objective_groups["schedule"]
            efficiency_objective = objective_groups["efficiency"]
            objective = objective_groups["total"]
            components = (
                objective,
                safety_objective,
                relation_objective,
                schedule_objective,
                efficiency_objective,
            )
            if not bool(torch.isfinite(torch.stack([value.float() for value in components])).all()):
                raise FloatingPointError(f"non-finite v4 objective at epoch {epoch + 1}, step {step + 1}")

            safety_gradients = _gradients(safety_objective, parameters, retain_graph=True)
            relation_gradients = _gradients(relation_objective, parameters, retain_graph=True)
            schedule_gradients = _gradients(schedule_objective, parameters, retain_graph=True)
            efficiency_gradients = _gradients(efficiency_objective, parameters, retain_graph=False)
            projected, gradient_parts = project_operating_utility_gradient_groups(
                safety_gradients,
                relation_gradients,
                schedule_gradients,
                efficiency_gradients,
                relation_norm_cap=args.relation_gradient_cap,
                schedule_norm_cap=args.schedule_gradient_cap,
                efficiency_norm_cap=args.efficiency_gradient_cap,
            )
            _assign_projected_gradients(parameters, projected)
            bad_gradients = [
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradients:
                raise FloatingPointError(f"non-finite v4 gradients: {bad_gradients}")
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            if weighted_np_boundary_logit is not None:
                assert weighted_np_dual_state is not None
                if not bool(torch.isfinite(weighted_np_boundary_logit).all()):
                    raise FloatingPointError(
                        "weighted Neyman-Pearson train boundary became non-finite"
                    )
                with torch.no_grad():
                    weighted_np_boundary_logit.clamp_(-20.0, 20.0)
                weighted_np_dual_state.update(
                    weighted_np_parts["weightedRecallViolation"]
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds = time.perf_counter() - step_started
            training_elapsed = time.time() - started
            remaining_steps = max(0, total_steps - global_step)
            eta_seconds = (
                training_elapsed / max(1, global_step) * remaining_steps
            )
            runtime_metrics: dict[str, float] = {
                "stepSeconds": float(step_seconds),
                "stepsPerSecond": float(1.0 / max(step_seconds, 1e-12)),
                "trainingElapsedSeconds": float(training_elapsed),
                "estimatedRemainingSeconds": float(eta_seconds),
            }
            if device.type == "cuda":
                runtime_metrics.update(
                    {
                        "cudaMemoryAllocatedBytes": float(torch.cuda.memory_allocated(device)),
                        "cudaPeakMemoryAllocatedBytes": float(torch.cuda.max_memory_allocated(device)),
                        "cudaPeakMemoryReservedBytes": float(torch.cuda.max_memory_reserved(device)),
                    }
                )

            operating_thresholds, threshold_metrics = _operating_threshold_metrics(
                operating_threshold_seed,
                global_step - 1,
            )
            step_metrics = _float_metrics(
                {
                    "loss": objective,
                    "lossSafetyGroup": safety_objective,
                    "lossRelationGroup": relation_objective,
                    "lossScheduleGroup": schedule_objective,
                    "lossEfficiencyGroup": efficiency_objective,
                    "lossDualProbeRescueAttenuationWeighted": (
                        float(args.dual_probe_rescue_loss_weight)
                        * dual_probe_rescue_loss
                    ),
                    "lossCullCertificateRegularizationWeighted": (
                        float(args.cull_certificate_regularization_weight)
                        * certificate_regularization
                    ),
                    "lossCullCertificatePositiveGuardWeighted": (
                        float(args.cull_certificate_tail_loss_weight)
                        * float(args.cull_certificate_positive_guard_weight)
                        * certificate_positive_guard
                    ),
                    "lossCullCertificateHardNegativeWeighted": (
                        float(args.cull_certificate_tail_loss_weight)
                        * certificate_hard_negative
                    ),
                    "lossCullCertificateCenterAnchorWeighted": (
                        float(args.cull_certificate_center_anchor_weight)
                        * certificate_center_anchor
                    ),
                    "lossCullCertificatePairPositiveWeighted": (
                        certificate_pair_half_weight
                        * certificate_pair_positive
                    ),
                    "lossCullCertificatePairNegativeWeighted": (
                        certificate_pair_half_weight
                        * certificate_pair_negative
                    ),
                    "lossInstanceExposurePositiveWeighted": (
                        exposure_half_weight * exposure_positive_loss
                    ),
                    "lossInstanceExposureNegativeWeighted": (
                        exposure_half_weight * exposure_negative_loss
                    ),
                    "lossSameInstanceCrossViewPositiveWeighted": (
                        cross_view_half_weight * cross_view_positive_loss
                    ),
                    "lossSameInstanceCrossViewNegativeWeighted": (
                        cross_view_half_weight * cross_view_negative_loss
                    ),
                    "lossCandidateBoundaryHardNegativeWeighted": (
                        float(args.candidate_boundary_hard_negative_weight)
                        * candidate_boundary_loss
                    ),
                    "lossViewResidualPositiveGuardWeighted": (
                        float(args.view_residual_positive_guard_weight)
                        * view_residual_positive_guard
                    ),
                    "lossViewResidualAnchorWeighted": (
                        float(args.view_residual_anchor_weight)
                        * view_residual_anchor
                    ),
                    "lossViewResidualCenterWeighted": (
                        float(args.view_residual_center_weight)
                        * view_residual_center
                    ),
                    "lossBoundaryOpportunityWeighted": (
                        float(args.boundary_opportunity_loss_weight)
                        * boundary_opportunity_loss
                    ),
                    "lossBoundaryTailObjectiveWeighted": (
                        float(args.boundary_tail_residual_loss_weight)
                        * boundary_tail_loss
                    ),
                    "lossQueryTailSeparationWeighted": (
                        float(args.query_tail_separation_loss_weight)
                        * query_tail_loss
                    ),
                    "lossWeightedSafetyFrontierWeighted": (
                        float(args.boundary_tail_fixed_frontier_weight)
                        * fixed_frontier_loss
                    ),
                    "lossWeightedNeymanPearsonOperatingWeighted": (
                        float(args.boundary_tail_np_weight) * weighted_np_loss
                    ),
                    "boundaryOpportunityScale": model.boundary_opportunity_scale,
                    "boundaryOpportunityRawLogitUpliftMean": aux[
                        "boundary_opportunity_raw_logit_uplift"
                    ].mean(),
                    "boundaryOpportunityAppliedLogitUpliftMean": aux[
                        "boundary_opportunity_logit_uplift"
                    ].mean(),
                    "boundaryTailResidualRawMean": aux[
                        "boundary_tail_residual_raw"
                    ].mean(),
                    "boundaryTailResidualCenteredMean": aux[
                        "boundary_tail_residual_centered"
                    ].mean(),
                    "boundaryTailResidualCenteredAbsMean": aux[
                        "boundary_tail_residual_centered"
                    ].abs().mean(),
                    "boundaryTailResidualCenteredMaxAbs": aux[
                        "boundary_tail_residual_centered"
                    ].abs().max(),
                    "boundaryTailResidualAppliedMean": aux[
                        "boundary_tail_residual_centered"
                    ].mean(),
                    "boundaryTailResidualAppliedAbsMean": aux[
                        "boundary_tail_residual_centered"
                    ].abs().mean(),
                    "queryTailSeparatorResidualMean": aux[
                        "query_tail_separator_residual"
                    ].mean(),
                    "queryTailSeparatorResidualAbsMean": aux[
                        "query_tail_separator_residual"
                    ].abs().mean(),
                    "queryTailSeparatorResidualMaxAbs": aux[
                        "query_tail_separator_residual"
                    ].abs().max(),
                    "lossSurvivalWeighted": float(args.survival_loss_weight) * survival_loss,
                    "lossRelationConsistencyWeighted": float(args.relation_consistency_weight) * relation_loss,
                    "lossUtilityWeighted": float(args.utility_loss_weight) * utility_loss,
                    "lossDownloadWeighted": float(args.download_loss_weight) * download_loss,
                    "lossRegularizationWeighted": float(args.regularization_weight) * regularization,
                    "lossInstanceCalibrationRegularizationWeighted": (
                        float(args.instance_calibration_regularization_weight)
                        * instance_calibration_regularization
                    ),
                    "cullCertificateSuppressionMean": aux[
                        "cull_certificate_suppression"
                    ].mean(),
                    "cullCertificateSuppressionMax": aux[
                        "cull_certificate_suppression"
                    ].max(),
                    "cullCertificateVisibleSuppressionMean": (
                        aux["cull_certificate_suppression"][target > 0.5].mean()
                        if bool((target > 0.5).any())
                        else logits.new_zeros(())
                    ),
                    "cullCertificateInvisibleSuppressionMean": (
                        aux["cull_certificate_suppression"][target <= 0.5].mean()
                        if bool((target <= 0.5).any())
                        else logits.new_zeros(())
                    ),
                    "viewResidualMean": aux["view_conditioned_residual"].mean(),
                    "viewResidualAbsMean": aux[
                        "view_conditioned_residual"
                    ].abs().mean(),
                    "viewResidualMaxAbs": aux[
                        "view_conditioned_residual"
                    ].abs().max(),
                    "viewResidualVisibleMean": (
                        aux["view_conditioned_residual"][target > 0.5].mean()
                        if bool((target > 0.5).any())
                        else logits.new_zeros(())
                    ),
                    "viewResidualInvisibleMean": (
                        aux["view_conditioned_residual"][target <= 0.5].mean()
                        if bool((target <= 0.5).any())
                        else logits.new_zeros(())
                    ),
                    **visibility_parts,
                    **certificate_parts,
                    **certificate_pair_parts,
                    **exposure_parts,
                    **cross_view_parts,
                    **candidate_boundary_parts,
                    **view_residual_parts,
                    **dual_probe_rescue_parts,
                    **boundary_opportunity_parts,
                    **{
                        "tailObjective" + key[:1].upper() + key[1:]: value
                        for key, value in boundary_tail_parts.items()
                    },
                    **{
                        "queryTail" + key[:1].upper() + key[1:]: value
                        for key, value in query_tail_parts.items()
                    },
                    **{
                        "fixedFrontier" + key[:1].upper() + key[1:]: value
                        for key, value in fixed_frontier_parts.items()
                    },
                    **{
                        "weightedNp" + key[:1].upper() + key[1:]: value
                        for key, value in weighted_np_parts.items()
                    },
                    **survival_parts,
                    **relation_parts,
                    **utility_parts,
                    **download_parts,
                    **gradient_parts,
                    **model.instance_calibration_diagnostics(),
                    **threshold_metrics,
                    **runtime_metrics,
                }
            )
            group_sum = (
                step_metrics["lossSafetyGroup"]
                + step_metrics["lossRelationGroup"]
                + step_metrics["lossScheduleGroup"]
                + step_metrics["lossEfficiencyGroup"]
            )
            if not math.isclose(step_metrics["loss"], group_sum, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError("logged v4 loss groups do not sum to the actual objective")
            for key, value in step_metrics.items():
                epoch_metrics.setdefault(key, []).append(value)
            if step == 0 or (step + 1) % max(1, args.steps_per_epoch // 5) == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "globalStep": global_step,
                            "loss": step_metrics["loss"],
                            "safetyGate": step_metrics.get("safetyReserveGate", 0.0),
                            "operatingThresholds": list(operating_thresholds),
                            "observationInstanceCoverage": sampler.epoch_instance_coverage,
                            "stepSeconds": step_metrics["stepSeconds"],
                            "etaSeconds": step_metrics["estimatedRemainingSeconds"],
                            "cudaPeakMemoryAllocatedMiB": (
                                step_metrics.get("cudaPeakMemoryAllocatedBytes", 0.0)
                                / (1024.0 * 1024.0)
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if processed_steps != int(args.steps_per_epoch):
            raise RuntimeError(
                f"processed {processed_steps} non-empty training steps, expected {args.steps_per_epoch}"
            )
        if sampler.epoch_instance_coverage < 1.0:
            raise RuntimeError("stratified observation sampler did not cover every observed instance")
        scheduler.step()
        train_summary = {
            key: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "count": len(values),
            }
            for key, values in epoch_metrics.items()
            if values
        }
        row: dict[str, Any] = {
            "epoch": epoch + 1,
            "globalStep": global_step,
            "lr": float(scheduler.get_last_lr()[0]),
            "trainMetricSummary": train_summary,
            "samplerCoverage": {
                "instances": sampler.epoch_instance_coverage,
                "observations": sampler.epoch_observation_coverage,
                "uniqueObservations": sampler.epoch_unique_observation_count,
                "directions": sampler.epoch_direction_coverage,
                "eventCensor": sampler.epoch_event_censor_coverage,
            },
            "elapsedSeconds": float(time.time() - started),
            "testRead": False,
        }
        checkpoint_training_state = {
            "safetyBoundaryEmaLogit": (
                None
                if safety_boundary_ema is None
                else float(safety_boundary_ema.detach().cpu())
            ),
            "safetyBoundaryEmaDecay": float(args.safety_boundary_ema_decay),
            "sourceSplit": "train",
            "weightedNeymanPearson": (
                {
                    "enabled": True,
                    "boundaryLogit": float(
                        weighted_np_boundary_logit.detach().cpu()
                    ),
                    "boundaryProbability": float(
                        torch.sigmoid(weighted_np_boundary_logit.detach()).cpu()
                    ),
                    "initialBoundaryLogit": weighted_np_initial_boundary_logit,
                    "dual": weighted_np_dual_state.as_dict(),
                    "sourceSplit": "train",
                    "runtimeExport": False,
                }
                if weighted_np_boundary_logit is not None
                and weighted_np_dual_state is not None
                else {"enabled": False, "sourceSplit": "train"}
            ),
        }

        calibration_payload: dict[str, Any] | None = None
        validation_row: dict[str, Any] | None = None
        model.eval()
        with torch.no_grad():
            runtime_snapshot, coefficient_snapshot, coefficient_snapshot_diagnostics = _runtime_features(
                model, geometry, relation_tensors, local_ids, structural_ids
            )
        if (epoch + 1) % int(args.eval_every) == 0 or epoch + 1 == int(args.epochs):
            calibration_rows = _evaluate(
                model,
                calibration_split,
                runtime_snapshot,
                world_aabbs,
                device,
                thresholds=threshold_grid(),
                seed=args.seed + 50_000,
                poses_per_batch=args.poses_per_batch,
                max_poses=args.max_eval_poses,
                bootstrap_replicates=args.calibration_bootstrap_replicates,
            )
            selected, diagnostic, frozen = _calibration_workpoints(calibration_rows)
            if frozen is not None:
                validation_rows = _evaluate(
                    model,
                    validation_split,
                    runtime_snapshot,
                    world_aabbs,
                    device,
                    thresholds=np.asarray([float(frozen["threshold"])], dtype=np.float32),
                    seed=args.seed + 60_000,
                    poses_per_batch=args.poses_per_batch,
                    max_poses=args.max_eval_poses,
                    bootstrap_replicates=args.calibration_bootstrap_replicates,
                    instance_to_glb=instance_to_glb_np,
                    glb_bytes=glb_bytes_np,
                )
                validation_row = validation_rows[0] if validation_rows else None
            calibration_payload = {
                "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-v4",
                "thresholdSource": "this-checkpoint-calibration-only",
                "selectedSafe": selected,
                "diagnostic": diagnostic,
                "thresholdRows": calibration_rows,
                "weightedRecallFloor": CALIBRATION_FLOOR,
                "weightedRecallLowerConfidenceBoundFloor": CALIBRATION_FLOOR,
                "selectionRule": aggregate_weighted_cull_selection_rule(
                    CALIBRATION_FLOOR, CALIBRATION_FLOOR
                ),
                "bootstrapReplicates": int(args.calibration_bootstrap_replicates),
                "validationBootstrapReplicates": int(args.calibration_bootstrap_replicates),
                "validationSafetyRequiredForBestCheckpoint": True,
                "validationSafetyPassed": _weighted_recall_safety_gate(validation_row),
                "testRead": False,
            }
            row["calibration"] = calibration_payload
            row["validationAtFrozenCalibrationThreshold"] = validation_row

            if diagnostic is not None:
                validation_safe = _weighted_recall_safety_gate(validation_row)
                diagnostic_key = (
                    float(validation_safe),
                    float(
                        (validation_row or {}).get(
                            "aggregateWeightedRecallLowerConfidenceBound"
                        )
                        or -1.0
                    ),
                    float(
                        (validation_row or {}).get("aggregateWeightedRecall")
                        or -1.0
                    ),
                    float(diagnostic.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0),
                    float(diagnostic.get("aggregateWeightedRecall") or -1.0),
                    float(diagnostic.get("agg_balanced_accuracy") or 0.0),
                    float(diagnostic.get("agg_useful_cull") or 0.0),
                    float(diagnostic.get("agg_precision") or 0.0),
                    -float(diagnostic.get("avg_pred_count") or 0.0),
                )
                if best_diagnostic_key is None or diagnostic_key > best_diagnostic_key:
                    best_diagnostic_key = diagnostic_key
                    best_diagnostic = {
                        "epoch": epoch + 1,
                        "threshold": float(diagnostic["threshold"]),
                        "safe": selected is not None and validation_safe,
                        "selection": diagnostic,
                        "validationSafetyPassed": validation_safe,
                    }
                    best_diagnostic_calibration = calibration_payload
                    best_diagnostic_validation = validation_row
                    torch.save(
                        _checkpoint(
                            model,
                            args,
                            epoch + 1,
                            global_step,
                            protocol,
                            relation_provenance,
                            geometry_meta,
                            coefficient_snapshot,
                            coefficient_snapshot_diagnostics,
                            calibration_reliability_meta,
                            calibration_payload,
                            validation_row,
                            best_diagnostic,
                            training_state=checkpoint_training_state,
                        ),
                        args.output_dir / "best_diagnostic.pt",
                    )
                    _save_fp16(
                        args.output_dir / "best_diagnostic_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

            if selected is not None and _weighted_recall_safety_gate(validation_row):
                safe_key = (
                    float(selected.get("agg_useful_cull") or 0.0),
                    float(selected.get("agg_balanced_accuracy") or 0.0),
                    float(selected.get("agg_precision") or 0.0),
                    -float(selected.get("avg_pred_count") or 0.0),
                )
                if best_safe_key is None or safe_key > best_safe_key:
                    best_safe_key = safe_key
                    best_safe = {
                        "epoch": epoch + 1,
                        "threshold": float(selected["threshold"]),
                        "safe": True,
                        "selection": selected,
                        "validationSafetyPassed": True,
                    }
                    best_safe_calibration = calibration_payload
                    best_safe_validation = validation_row
                    payload = _checkpoint(
                        model,
                        args,
                        epoch + 1,
                        global_step,
                        protocol,
                        relation_provenance,
                        geometry_meta,
                        coefficient_snapshot,
                        coefficient_snapshot_diagnostics,
                        calibration_reliability_meta,
                        calibration_payload,
                        validation_row,
                        best_safe,
                        training_state=checkpoint_training_state,
                    )
                    _save_safe_checkpoint_alias(payload, args.output_dir)
                    _save_fp16(
                        args.output_dir / "best_safe_instance_survival_coefficients_fp16.bin",
                        coefficient_snapshot,
                    )

        if args.snapshot_every > 0 and (epoch + 1) % int(args.snapshot_every) == 0:
            torch.save(
                _checkpoint(
                    model,
                    args,
                    epoch + 1,
                    global_step,
                    protocol,
                    relation_provenance,
                    geometry_meta,
                    coefficient_snapshot,
                    coefficient_snapshot_diagnostics,
                    calibration_reliability_meta,
                    calibration_payload,
                    validation_row,
                    best_safe or best_diagnostic,
                    training_state=checkpoint_training_state,
                ),
                args.output_dir / f"checkpoint_epoch_{epoch + 1:03d}.pt",
            )
        torch.save(
            _checkpoint(
                model,
                args,
                epoch + 1,
                global_step,
                protocol,
                relation_provenance,
                geometry_meta,
                coefficient_snapshot,
                coefficient_snapshot_diagnostics,
                calibration_reliability_meta,
                calibration_payload,
                validation_row,
                training_state=checkpoint_training_state,
            ),
            args.output_dir / "last.pt",
        )
        _save_fp16(args.output_dir / "instance_survival_coefficients_fp16.bin", coefficient_snapshot)
        history.append(row)
        _append_jsonl(metrics_path, row)
        _write_json(args.output_dir / "train_history.json", history)
        print(
            json.dumps(
                {
                    "epoch": epoch + 1,
                    "bestSafe": best_safe,
                    "bestDiagnostic": best_diagnostic,
                    "outputDir": str(args.output_dir),
                    "testRead": False,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    model.eval()
    with torch.no_grad():
        final_runtime, final_coefficients, final_coefficient_diagnostics = _runtime_features(
            model, geometry, relation_tensors, local_ids, structural_ids
        )
    _save_fp16(args.output_dir / "instance_runtime_features_fp16.bin", final_runtime)
    _save_fp16(args.output_dir / "instance_geo_features_fp16.bin", geometry)
    _save_fp16(args.output_dir / "instance_survival_coefficients_fp16.bin", final_coefficients)
    _save_fp16(
        args.output_dir / "instance_survival_prior_coefficients_fp16.bin",
        final_coefficient_diagnostics["survival_prior_coefficients"],
    )
    _save_fp16(
        args.output_dir / "instance_survival_calibration_residual_fp16.bin",
        final_coefficient_diagnostics["instance_calibration_applied_residual"],
    )
    calibration_summary = {
        "schema": "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4",
        "status": "safe" if best_safe is not None else "no_qualified_safety_workpoint",
        "primarySafetyMetric": "aggregateWeightedRecall",
        "weightedRecallFloor": CALIBRATION_FLOOR,
        "weightedRecallLowerConfidenceBoundFloor": CALIBRATION_FLOOR,
        "validationSafetyRequiredForBestCheckpoint": True,
        "bestSafe": best_safe,
        "bestDiagnostic": best_diagnostic,
        "calibration": best_safe_calibration or best_diagnostic_calibration,
        "validationAtFrozenThreshold": best_safe_validation or best_diagnostic_validation,
        "safeCheckpoint": str(args.output_dir / "best_safe.pt") if best_safe else None,
        "diagnosticCheckpoint": str(args.output_dir / "best_diagnostic.pt") if best_diagnostic else None,
        "testRead": False,
    }
    _write_json(args.output_dir / "calibration_ready_summary.json", calibration_summary)
    _write_json(
        args.output_dir / "model_meta.json",
        {
            "schema": MODEL_SCHEMA,
            "status": calibration_summary["status"],
            "modelConfig": model.config,
            "protocol": protocol,
            "runtimeFeature": {
                "shape": [num_instances, RUNTIME_FEATURE_DIM],
                "dtype": "float16",
                "file": "instance_runtime_features_fp16.bin",
                "bytes": int(num_instances * RUNTIME_FEATURE_DIM * 2),
            },
            "relationRuntimeExported": False,
            "instanceCalibrationRuntimeExportedSeparately": False,
            "survivalCoefficientFusion": "shared_relation_prior_plus_applied_instance_residual",
            "instanceCalibrationReliability": calibration_reliability_meta,
            "defaultFrontendModified": False,
            "testRead": False,
        },
    )
    print(
        json.dumps(
            {
                "status": calibration_summary["status"],
                "bestSafe": str(args.output_dir / "best_safe.pt") if best_safe else None,
                "bestDiagnostic": str(args.output_dir / "best_diagnostic.pt") if best_diagnostic else None,
                "last": str(args.output_dir / "last.pt"),
                "testRead": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
