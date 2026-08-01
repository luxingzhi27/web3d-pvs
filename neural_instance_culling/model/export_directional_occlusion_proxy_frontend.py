#!/usr/bin/env python3
"""Export directional occlusion proxy PVS assets for the WebGPU frontend."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common.runtime_meta import load_runtime_meta
from current_pvs_utils import (
    MODEL_INPUT_FOV_Y_DEG,
    RENDER_FOV_Y_DEG,
    AssetBuilder,
    infer_prediction_camera_fields,
    load_threshold,
)
from common.threshold_selection import (
    select_weighted_precision_workpoint,
    target_weighted_recall_from_payload,
    weighted_precision_selection_rule,
    weighted_recall_safe_rows,
)


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    return value.detach().cpu().numpy()


def state(model: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name not in model:
        raise KeyError(f"checkpoint is missing model tensor: {name}")
    return model[name]


def load_runtime_features(path: Path, num_instances: int, feature_dim: int) -> np.ndarray:
    data = np.fromfile(path, dtype=np.float16)
    expected = int(num_instances) * int(feature_dim)
    if data.size != expected:
        raise ValueError(f"{path} has {data.size} fp16 values, expected {expected}")
    return data.reshape(int(num_instances), int(feature_dim))


def compact_feature_meta(feature_meta: dict[str, Any]) -> dict[str, Any]:
    """Keep only runtime-relevant provenance; training resource reports can be huge."""
    keep = [
        "schema",
        "experiment",
        "numInstances",
        "geoDim",
        "contextDim",
        "proxyDim",
        "directionBins",
        "depthShells",
        "runtimeFeatureDim",
        "dtype",
        "geoSource",
        "contextSource",
        "proxySource",
        "runtimeUsesPointNet",
        "runtimeUsesGraphPropagation",
    ]
    compact = {key: feature_meta[key] for key in keep if key in feature_meta}
    evidence = feature_meta.get("evidenceMeta")
    if isinstance(evidence, dict):
        compact["evidenceMeta"] = {
            key: evidence[key]
            for key in ("schema", "directionBins", "depthShells", "sourceK", "semantics", "thresholds", "stats")
            if key in evidence
        }
    points = feature_meta.get("glbPointsMeta")
    if isinstance(points, dict):
        compact["glbPointsMeta"] = {
            key: points[key] for key in ("numGlbs", "pointsPerGlb", "dims") if key in points
        }
    files = feature_meta.get("files")
    if isinstance(files, dict):
        compact["files"] = {
            key: files[key]
            for key in ("instanceGeo", "instanceContext", "instanceOcclusionProxy", "instanceRuntime")
            if key in files
        }
    compact["runtimeMetadataCompaction"] = {
        "removedFields": ["resources", "glbPointsMeta.entries"],
        "reason": "Training-only resource reports are not needed by the browser runtime.",
    }
    return compact


def normalize_threshold_payload(payload: dict[str, Any], source: Path) -> tuple[dict[str, Any], str]:
    """Normalize formal calibration records without doing a new threshold scan.

    Current formal training deliberately stops before test and writes
    ``calibration_ready_summary.json``.  The older exporter only understood
    ``eval_summary.json`` and would therefore either fail to export the
    formal checkpoint or fall back to an unsafe default.  A pre-test record is
    already a frozen calibration decision, so the exporter must validate that
    decision and reuse it verbatim.
    """
    protocol = str(payload.get("protocol", ""))
    if protocol == "calibration_ready_pre_test":
        if int(payload.get("testEvaluationCount", -1)) != 0:
            raise RuntimeError(
                f"{source} is marked calibration_ready_pre_test but testEvaluationCount is not zero."
            )
        calibration = payload.get("calibration")
        selected = calibration.get("selected") if isinstance(calibration, dict) else None
        frozen = payload.get("frozenThreshold")
        if not isinstance(selected, dict) or frozen is None:
            raise RuntimeError(f"{source} has no frozen calibration workpoint.")
        if abs(float(selected.get("threshold", float("nan"))) - float(frozen)) > 1e-6:
            raise RuntimeError(f"{source} has inconsistent selected and frozen thresholds.")
        rows = payload.get("calibrationThresholdRows")
        if not isinstance(rows, list) or not rows:
            raise RuntimeError(f"{source} has no calibration threshold rows.")
        normalized = dict(payload)
        normalized["thresholdRows"] = rows
        normalized["workpoints"] = {
            "primaryWeightedPrecision": dict(selected),
            "calibration": calibration,
        }
        return normalized, protocol
    if protocol == "frozen_calibration_one_shot_test":
        if int(payload.get("testEvaluationCount", -1)) != 1:
            raise RuntimeError(f"{source} is not a complete one-shot frozen-test summary.")
        return payload, protocol
    if protocol:
        raise RuntimeError(f"Unsupported export threshold protocol in {source}: {protocol}")
    # Legacy exploratory summaries are still accepted for historical exports,
    # but the caller will require a recorded safe threshold row below.
    return payload, "legacy_eval_summary"


def make_display_name(model_name: str) -> str:
    label = model_name
    if label.startswith("pvs_"):
        label = label[4:]
    if label.startswith("directional_occlusion_proxy_encoder"):
        label = label.replace("directional_occlusion_proxy_encoder", "Directional Occlusion Proxy", 1)
    replacements = {
        "rvl": "RVL",
        "w042": "0.42",
        "full40": "full40",
        "epoch24": "epoch24",
        "best": "best",
    }
    parts = []
    for token in label.split("_"):
        parts.append(replacements.get(token, token))
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export directional occlusion proxy scheduler assets for slm2viewer.")
    parser.add_argument("--checkpoint", default="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/best.pt")
    parser.add_argument("--runtime-features", default="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_runtime_features_fp16.bin")
    parser.add_argument("--feature-meta", default="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_features_meta.json")
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--eval-summary", default="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/eval_summary.json")
    parser.add_argument("--eval-model-name", default="pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best")
    parser.add_argument("--output-dir", default="slm2viewer/public/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40_best")
    parser.add_argument(
        "--workpoint",
        default="primaryWeightedPrecision",
        help="Named workpoint in eval_summary.workpoints. Defaults to highest pose precision with weighted recall > 0.99.",
    )
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument(
        "--override-threshold",
        type=float,
        default=None,
        help="Explicit threshold; it must match a weighted-recall-safe row unless unsafe override is explicitly enabled.",
    )
    parser.add_argument(
        "--allow-unsafe-override",
        action="store_true",
        help="Allow an explicitly requested diagnostic threshold that does not satisfy weighted recall > 0.99.",
    )
    parser.add_argument("--prefetch-threshold", type=float, default=0.04)
    parser.add_argument("--dataset-meta", default=None)
    parser.add_argument("--prediction-camera-mode", choices=["active-camera", "viewcell-back-camera"], default=None)
    parser.add_argument("--pvs-back-offset-m", type=float, default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["config"])
    if config.get("runtimeSchema") != "directional-occlusion-proxy-scheduler-v1":
        raise ValueError(f"Unexpected runtimeSchema: {config.get('runtimeSchema')}")
    runtime_config = {
        key: value
        for key, value in config.items()
        if "fov" not in str(key).lower()
    }
    model = checkpoint["model"]
    num_instances = int(config["numInstances"])
    runtime_dim = int(config["runtimeFeatureDim"])
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(args.runtime_meta, num_instances)
    runtime_features = load_runtime_features(Path(args.runtime_features), num_instances, runtime_dim)
    feature_meta = json.loads(Path(args.feature_meta).read_text(encoding="utf-8")) if Path(args.feature_meta).exists() else {}

    builder = AssetBuilder()
    builder.append_fp16("runtime_features", runtime_features)
    builder.append_fp16("instance_world_aabbs", world_aabbs)
    builder.append_fp16("scene_min", state(model, "scene_min"))
    builder.append_fp16("scene_size_m", state(model, "scene_size_m"))

    builder.append_fp16("ray_proxy_gate_w0", state(model, "ray_proxy_gate.0.weight"))
    builder.append_fp16("ray_proxy_gate_b0", state(model, "ray_proxy_gate.0.bias"))
    builder.append_fp16("ray_proxy_gate_w1", state(model, "ray_proxy_gate.2.weight"))
    builder.append_fp16("ray_proxy_gate_b1", state(model, "ray_proxy_gate.2.bias"))
    builder.append_fp16("ray_runtime_gate_w", state(model, "ray_runtime_gate.weight"))
    builder.append_fp16("ray_runtime_gate_b", state(model, "ray_runtime_gate.bias"))

    builder.append_fp16("mlp_vis_cam_inter_w", state(model, "visibility_mlp.camera_interaction.weight"))
    builder.append_fp16("mlp_vis_cam_inter_b", state(model, "visibility_mlp.camera_interaction.bias"))
    builder.append_fp16("mlp_vis_inst_inter_w", state(model, "visibility_mlp.instance_interaction.weight"))
    builder.append_fp16("mlp_vis_inst_inter_b", state(model, "visibility_mlp.instance_interaction.bias"))
    builder.append_fp16("mlp_vis_w0", state(model, "visibility_mlp.net.0.weight"))
    builder.append_fp16("mlp_vis_b0", state(model, "visibility_mlp.net.0.bias"))
    builder.append_fp16("mlp_vis_w1", state(model, "visibility_mlp.net.2.weight"))
    builder.append_fp16("mlp_vis_b1", state(model, "visibility_mlp.net.2.bias"))
    builder.append_fp16("mlp_vis_w2", state(model, "visibility_mlp.net.4.weight"))
    builder.append_fp16("mlp_vis_b2", state(model, "visibility_mlp.net.4.bias"))

    builder.append_fp16("inhibition_w0", state(model, "inhibition_head.0.weight"))
    builder.append_fp16("inhibition_b0", state(model, "inhibition_head.0.bias"))
    builder.append_fp16("inhibition_w1", state(model, "inhibition_head.2.weight"))
    builder.append_fp16("inhibition_b1", state(model, "inhibition_head.2.bias"))

    builder.append_fp16("utility_w0", state(model, "utility_head.0.weight"))
    builder.append_fp16("utility_b0", state(model, "utility_head.0.bias"))
    builder.append_fp16("utility_w1", state(model, "utility_head.2.weight"))
    builder.append_fp16("utility_b1", state(model, "utility_head.2.bias"))
    builder.append_fp16("utility_w2", state(model, "utility_head.4.weight"))
    builder.append_fp16("utility_b2", state(model, "utility_head.4.bias"))

    builder.append_fp16("download_w0", state(model, "download_head.0.weight"))
    builder.append_fp16("download_b0", state(model, "download_head.0.bias"))
    builder.append_fp16("download_w1", state(model, "download_head.2.weight"))
    builder.append_fp16("download_b1", state(model, "download_head.2.bias"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / "instance_pvs_assets.bin"
    builder.write(bin_path)

    eval_summary_path = Path(args.eval_summary)
    if not eval_summary_path.exists():
        raise FileNotFoundError(
            f"Threshold provenance file does not exist: {eval_summary_path}. "
            "Formal exports require calibration_ready_summary.json or eval_summary.json."
        )
    eval_summary, threshold_protocol = normalize_threshold_payload(
        json.loads(eval_summary_path.read_text(encoding="utf-8")),
        eval_summary_path,
    )
    target_weighted_recall = target_weighted_recall_from_payload(eval_summary)
    threshold_rows = eval_summary.get("thresholdRows") if isinstance(eval_summary.get("thresholdRows"), list) else []
    workpoints = eval_summary.get("workpoints") if isinstance(eval_summary.get("workpoints"), dict) else {}
    selected_workpoint = workpoints.get(args.workpoint) if isinstance(workpoints.get(args.workpoint), dict) else None
    primary_workpoint = select_weighted_precision_workpoint(threshold_rows, target_weighted_recall)
    if threshold_protocol == "calibration_ready_pre_test":
        # The training process selected this row using the registered point
        # and bootstrap lower-bound floors.  Do not silently choose another
        # row with a larger point estimate while exporting.
        recorded = workpoints.get("primaryWeightedPrecision")
        if not isinstance(recorded, dict):
            raise RuntimeError(f"{eval_summary_path} has no recorded primary calibration workpoint.")
        primary_workpoint = dict(recorded)
        point_floor = float((eval_summary.get("args") or {}).get("calibration_point_floor", 0.0))
        lcb_floor = float((eval_summary.get("args") or {}).get("calibration_lcb_floor", 0.0))
        point = float(primary_workpoint.get("pose_weighted_recall", -1.0))
        lcb = float(primary_workpoint.get("weighted_recall_lower_confidence_bound", -1.0))
        if point <= target_weighted_recall or point < point_floor or lcb <= lcb_floor:
            raise RuntimeError(
                f"{eval_summary_path} recorded an unsafe calibration workpoint: "
                f"weighted_recall={point}, point_floor={point_floor}, lcb={lcb}, lcb_floor={lcb_floor}."
            )
    if args.workpoint == "primaryWeightedPrecision":
        selected_workpoint = primary_workpoint
    elif selected_workpoint is None:
        raise RuntimeError(f"Unknown frontend workpoint: {args.workpoint}")
    elif not weighted_recall_safe_rows([selected_workpoint], target_weighted_recall):
        raise RuntimeError(
            f"Workpoint {args.workpoint} does not satisfy strict weighted recall > "
            f"{target_weighted_recall:.3f}; use primaryWeightedPrecision for frontend export."
        )
    elif primary_workpoint is not None and selected_workpoint.get("threshold") != primary_workpoint.get("threshold"):
        raise RuntimeError(
            f"Workpoint {args.workpoint} is safe but is not the highest-pose-precision workpoint; "
            "use primaryWeightedPrecision for the default frontend asset."
        )
    threshold_selection = "primaryWeightedPrecision" if args.workpoint == "primaryWeightedPrecision" else args.workpoint
    best = dict(selected_workpoint or {})
    if not best:
        raise RuntimeError(
            "No workpoint satisfies the strict weighted-recall threshold rule; refusing to export an unsafe frontend asset."
        )
    if args.override_threshold is not None:
        threshold = float(args.override_threshold)
        threshold_selection = "manual-override"
        matching = [row for row in threshold_rows if abs(float(row.get("threshold", -1.0)) - threshold) < 1e-5]
        if matching:
            best = dict(matching[0])
            if not weighted_recall_safe_rows(matching, target_weighted_recall) and not args.allow_unsafe_override:
                raise RuntimeError(
                    f"Threshold {threshold:g} does not satisfy strict weighted recall > "
                    f"{target_weighted_recall:.3f}; pass --allow-unsafe-override only for diagnostics."
                )
        elif not args.allow_unsafe_override:
                raise RuntimeError(
                    "An override threshold must match a recorded threshold row so its weighted-recall safety can be verified."
                )
        else:
            best = {"threshold": threshold}
        best["threshold"] = threshold
    else:
        threshold = float(best.get("threshold", load_threshold(args.eval_summary, args.threshold, args.eval_model_name)))
    prediction_camera_fields = infer_prediction_camera_fields(
        checkpoint=checkpoint,
        config=config,
        dataset_meta_arg=args.dataset_meta,
        prediction_camera_mode_arg=args.prediction_camera_mode,
        pvs_back_offset_arg=args.pvs_back_offset_m,
    )
    meta: dict[str, Any] = {
        **runtime_config,
        "schemaVersion": 1,
        "runtimeSchema": "directional-occlusion-proxy-scheduler-v1",
        "runtimeModelName": args.eval_model_name,
        "runtimeModelDisplayName": make_display_name(args.eval_model_name),
        "assetFile": "instance_pvs_assets.bin",
        "usesRuntimeTriplane": False,
        "usesRuntimePointNet": False,
        "usesRuntimeGraphPropagation": False,
        "usesRuntimeAabbProjection": False,
        "usesDynamicOcclusionPool": False,
        "usesFixedInstanceFeatures": True,
        "fixedRuntimeFeatureSource": str(Path(args.runtime_features).as_posix()),
        "fixedRuntimeFeatureMeta": compact_feature_meta(feature_meta),
        "outputsVisualUtility": True,
        "outputsDownloadPriority": True,
        "outputsGlbPriority": True,
        "outputValueWords": 2,
        "visibilityThreshold": float(threshold),
        "prefetchThreshold": float(args.prefetch_threshold),
        "frontendRenderFovYDeg": RENDER_FOV_Y_DEG,
        **prediction_camera_fields,
        "sceneBounds": runtime_meta["sceneBounds"],
        "cameraBounds": checkpoint.get("cameraBounds") or runtime_meta["sceneBounds"],
        "layout": builder.layout,
        "totalBytes": int(bin_path.stat().st_size),
        "instanceToGlobalGlb": instance_to_glb.astype(np.int32).tolist(),
        "exportSourceCheckpoint": str(Path(args.checkpoint).as_posix()),
        "exportSourceEvalSummary": str(Path(args.eval_summary).as_posix()),
        "exportThresholdProtocol": threshold_protocol,
        "exportedBy": "neural_instance_culling/model/export_directional_occlusion_proxy_frontend.py",
        "exportedAt": datetime.now(timezone.utc).isoformat(),
        "thresholdSelection": threshold_selection,
        "targetWeightedRecall": float(target_weighted_recall),
        "thresholdSelectionMetric": "pose_precision",
        "thresholdSelectionConstraint": f"pose_weighted_recall > {target_weighted_recall:.3f}",
        "thresholdSelectionRule": weighted_precision_selection_rule(
            target_weighted_recall
        ),
        "testWorkpoint": {
            "threshold": float(best.get("threshold", threshold)),
            "posePrecision": float(best.get("pose_precision", 0.0)),
            "poseRecall": float(best.get("pose_recall", 0.0)),
            "poseWeightedRecall": float(best.get("pose_weighted_recall", 0.0)),
            "poseF1": float(best.get("pose_f1", 0.0)),
            "avgPred": float(best.get("avg_pred_count", 0.0)),
            "avgGt": float(best.get("avg_gt_count", 0.0)),
            "avgCandidate": float(best.get("avg_candidate_count", 0.0)),
            "candidateReduction": float(best.get("candidate_reduction_ratio", 0.0)),
            "evalPoseCount": int(best.get("eval_pose_count", 0)),
            "weightedRecallConstraintSatisfied": float(best.get("pose_weighted_recall", -1.0)) > target_weighted_recall,
        },
    }
    meta_path = output_dir / "instance_model_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "outputDir": str(output_dir),
                "meta": str(meta_path),
                "asset": str(bin_path),
                "threshold": float(threshold),
                "bytes": int(bin_path.stat().st_size),
                "schema": meta["runtimeSchema"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
