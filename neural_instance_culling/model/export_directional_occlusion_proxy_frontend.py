#!/usr/bin/env python3
"""Export directional occlusion proxy PVS assets for the WebGPU frontend."""
from __future__ import annotations

import argparse
import json
import struct
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


SPATIAL_PAGE_MAGIC = b"NSPF"
SPATIAL_PAGE_VERSION = 2
SPATIAL_PAGE_HEADER = struct.Struct("<4sIIIIII")


def _page_bounds(world_aabbs: np.ndarray, ids: list[int]) -> dict[str, list[float]]:
    values = np.asarray(world_aabbs[np.asarray(ids, dtype=np.int64)], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] == 0:
        raise ValueError("A spatial feature page must contain at least one six-value AABB row")
    return {
        "min": values[:, :3].min(axis=0).astype(float).tolist(),
        "max": values[:, 3:].max(axis=0).astype(float).tolist(),
    }


def write_spatial_feature_pages(
    output_dir: Path,
    world_aabbs: np.ndarray,
    runtime_features: np.ndarray,
    scene_bounds: dict[str, Any],
    *,
    cell_size_m: float,
    max_instances: int,
) -> dict[str, Any]:
    """Write spatially paged AABBs and fixed features for the experimental M9 schema.

    A page owns complete local instance rows.  Page selection is conservative: the
    directory stores the union AABB of every row, and the browser performs an exact
    per-instance AABB test after loading a page.  The page format deliberately keeps
    local instance IDs, AABBs, and features together so no full-scene feature table is
    needed for a first camera query.
    """
    if cell_size_m <= 0 or max_instances <= 0:
        raise ValueError("spatial page cell size and max instances must be positive")
    if world_aabbs.ndim != 2 or world_aabbs.shape[1] != 6:
        raise ValueError(f"world_aabbs must have shape [N, 6], got {world_aabbs.shape}")
    if runtime_features.ndim != 2 or runtime_features.shape[0] != world_aabbs.shape[0]:
        raise ValueError(
            "runtime features and AABB rows must have the same instance count: "
            f"features={runtime_features.shape}, aabbs={world_aabbs.shape}"
        )
    page_root = output_dir / "spatial_pages"
    page_root.mkdir(parents=True, exist_ok=True)
    for old in page_root.glob("page_*.bin"):
        old.unlink()

    if "min" in scene_bounds:
        origin = np.asarray(scene_bounds["min"], dtype=np.float64)
    else:
        center = np.asarray(scene_bounds["center"], dtype=np.float64)
        size = np.asarray(scene_bounds["size"], dtype=np.float64)
        origin = center - size * 0.5
    centers = (np.asarray(world_aabbs[:, :3], dtype=np.float64) + np.asarray(world_aabbs[:, 3:], dtype=np.float64)) * 0.5
    cells = np.floor((centers - origin[None, :]) / float(cell_size_m)).astype(np.int64)
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for local_id, cell in enumerate(cells.tolist()):
        grouped.setdefault((int(cell[0]), int(cell[1]), int(cell[2])), []).append(local_id)

    pages: list[dict[str, Any]] = []
    page_id = 0
    for cell in sorted(grouped):
        ids = grouped[cell]
        for start in range(0, len(ids), max_instances):
            page_ids = ids[start:start + max_instances]
            page_ids = sorted(int(value) for value in page_ids)
            # Keep exact FP32 AABBs for candidate selection.  The browser only
            # converts these rows to FP16 when assembling the neural query buffer;
            # using FP16 for the spatial predicate could drop boundary instances.
            aabbs = np.ascontiguousarray(world_aabbs[page_ids].astype("<f4", copy=False))
            features = np.ascontiguousarray(runtime_features[page_ids].astype(np.float16, copy=False))
            header = SPATIAL_PAGE_HEADER.pack(
                SPATIAL_PAGE_MAGIC,
                SPATIAL_PAGE_VERSION,
                len(page_ids),
                int(features.shape[1]),
                4,
                int(np.uint32(0)),
                0,
            )
            ids_bytes = np.asarray(page_ids, dtype="<u4").tobytes()
            data = header + ids_bytes + aabbs.reshape(-1).tobytes() + features.reshape(-1).tobytes()
            filename = f"page_{page_id:06d}.bin"
            (page_root / filename).write_bytes(data)
            pages.append({
                "pageId": page_id,
                "file": f"spatial_pages/{filename}",
                "count": len(page_ids),
                "byteSize": len(data),
                "bounds": _page_bounds(world_aabbs, page_ids),
                "cell": list(cell),
            })
            page_id += 1

    directory = {
        "schema": "directional-occlusion-proxy-spatial-pages-v1",
        "formatVersion": SPATIAL_PAGE_VERSION,
        "pageHeader": "<4sIIIIII: magic, version, count, runtimeFeatureDim, aabbByteWidth, reserved, reserved",
        "aabbDtype": "fp32",
        "featureDtype": "fp16",
        "numInstances": int(world_aabbs.shape[0]),
        "runtimeFeatureDim": int(runtime_features.shape[1]),
        "pageCellSizeM": float(cell_size_m),
        "pageMaxInstances": int(max_instances),
        "sceneOrigin": origin.astype(float).tolist(),
        "pageCount": len(pages),
        "pages": pages,
        "selectionSemantics": "page union AABB is conservative; browser exact-tests every loaded instance AABB",
    }
    directory_path = output_dir / "spatial_pages" / "page_directory.json"
    directory["directoryFile"] = "spatial_pages/page_directory.json"
    directory["directoryBytes"] = 0
    directory["featureBytes"] = int(runtime_features.astype(np.float16, copy=False).nbytes)
    directory["aabbBytes"] = int(world_aabbs.astype("<f4", copy=False).nbytes)
    directory["totalPageBytes"] = int(sum(int(page["byteSize"]) for page in pages))
    # The byte count is part of the manifest, so write until its decimal width
    # and the resulting file size agree. This keeps the on-disk manifest and
    # the returned export metadata self-consistent.
    for _ in range(4):
        directory_path.write_text(json.dumps(directory, ensure_ascii=False, indent=2), encoding="utf-8")
        actual_bytes = int(directory_path.stat().st_size)
        if actual_bytes == directory["directoryBytes"]:
            break
        directory["directoryBytes"] = actual_bytes
    else:
        raise RuntimeError(f"Could not stabilize spatial page directory byte count: {directory_path}")
    return directory


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
    parser.add_argument(
        "--spatial-pages",
        action="store_true",
        help="Export an experimental spatial-page asset: static model weights plus lazily fetched instance pages.",
    )
    parser.add_argument("--spatial-page-cell-size-m", type=float, default=256.0)
    parser.add_argument("--spatial-page-max-instances", type=int, default=1024)
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
    if not args.spatial_pages:
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
    page_directory = None
    if args.spatial_pages:
        page_directory = write_spatial_feature_pages(
            output_dir,
            world_aabbs,
            runtime_features,
            runtime_meta["sceneBounds"],
            cell_size_m=float(args.spatial_page_cell_size_m),
            max_instances=int(args.spatial_page_max_instances),
        )
    bin_path = output_dir / ("instance_pvs_model_weights.bin" if args.spatial_pages else "instance_pvs_assets.bin")
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
        "assetFile": bin_path.name,
        "usesRuntimeTriplane": False,
        "usesRuntimePointNet": False,
        "usesRuntimeGraphPropagation": False,
        "usesRuntimeAabbProjection": False,
        "usesDynamicOcclusionPool": False,
        "usesFixedInstanceFeatures": True,
        "usesSpatialFeaturePages": bool(args.spatial_pages),
        "spatialPageDirectory": page_directory.get("directoryFile") if page_directory else None,
        "spatialPageFormat": page_directory.get("schema") if page_directory else None,
        "spatialPageCount": int(page_directory.get("pageCount", 0)) if page_directory else 0,
        "spatialPageCellSizeM": float(page_directory.get("pageCellSizeM", 0.0)) if page_directory else None,
        "spatialPageMaxInstances": int(page_directory.get("pageMaxInstances", 0)) if page_directory else None,
        "spatialPageDirectoryBytes": int(page_directory.get("directoryBytes", 0)) if page_directory else 0,
        "spatialPageBytes": int(page_directory.get("totalPageBytes", 0)) if page_directory else 0,
        "spatialPageFeatureBytes": int(page_directory.get("featureBytes", 0)) if page_directory else 0,
        "spatialPageAabbBytes": int(page_directory.get("aabbBytes", 0)) if page_directory else 0,
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
        "staticAssetBytes": int(bin_path.stat().st_size),
        "fullPagedAssetBytes": int(bin_path.stat().st_size)
        + (int(page_directory.get("directoryBytes", 0)) if page_directory else 0)
        + (int(page_directory.get("totalPageBytes", 0)) if page_directory else 0),
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
