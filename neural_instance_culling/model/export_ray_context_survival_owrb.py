#!/usr/bin/env python3
"""Export the browser-only bundle for the ray-context/survival OWRB model.

The checkpoint contains the offline PointNet++ and relation encoder, but the
browser bundle deliberately contains neither.  Runtime data is a FP16 table
of 96 geometry values, 32 context coefficients and 28 survival coefficients
per instance.  The browser evaluates only the fixed direction bases, the
semantic survival query and the small visibility/utility/download heads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from common.threshold_selection import select_weighted_cull_workpoint  # noqa: E402


DENSE_WEIGHT_NAMES = (
    "survival_direction_basis.0.weight",
    "survival_direction_basis.0.bias",
    "survival_direction_basis.2.weight",
    "survival_direction_basis.2.bias",
    "context_direction_basis.0.weight",
    "context_direction_basis.0.bias",
    "context_direction_basis.2.weight",
    "context_direction_basis.2.bias",
    "base_trunk.0.weight",
    "base_trunk.0.bias",
    "base_trunk.2.weight",
    "base_trunk.2.bias",
    "base_visibility_head.weight",
    "base_visibility_head.bias",
    "utility_head.0.weight",
    "utility_head.0.bias",
    "utility_head.2.weight",
    "utility_head.2.bias",
    "download_head.0.weight",
    "download_head.0.bias",
    "download_head.2.weight",
    "download_head.2.bias",
)

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_feature_file(features_dir: Path) -> Path:
    for name in ("best_instance_runtime_features_fp16.bin", "instance_runtime_features_fp16.bin"):
        path = features_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"runtime feature table not found in {features_dir}")


def _frozen_threshold(checkpoint: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    calibration = checkpoint.get("calibration") or {}
    rows = calibration.get("rows", [])
    selected = select_weighted_cull_workpoint(
        rows,
        target_weighted_recall=0.99,
        minimum_lower_confidence_bound=0.99,
    )
    if selected is None:
        raise ValueError("checkpoint has no weighted-recall-safe calibration workpoint")
    threshold = float(selected["threshold"])
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"invalid frozen threshold {threshold!r}")
    return threshold, {
        "source": "checkpoint.calibration",
        "protocol": "calibration_ready_pre_test",
        "targetWeightedRecall": 0.99,
        "minimumWeightedRecallLowerConfidenceBound": 0.99,
        "selected": selected,
        "testEvaluationCount": 0,
    }


def _serialize_weights(
    state: dict[str, torch.Tensor],
) -> dict[str, Any]:
    missing = [name for name in DENSE_WEIGHT_NAMES if name not in state]
    if missing:
        raise ValueError(f"checkpoint is missing online weights: {missing}")
    output: dict[str, Any] = {}
    for name in DENSE_WEIGHT_NAMES:
        tensor = state[name].detach().cpu().float().contiguous()
        output[name] = {"shape": list(tensor.shape), "values": tensor.reshape(-1).tolist()}
    return output


def _ray_feature_layout(ray_mode: str) -> dict[str, Any]:
    if ray_mode == "direct9":
        return {
            "mode": "direct9",
            "dim": 9,
            "segments": [
                {
                    "name": "rayDirection",
                    "offset": 0,
                    "dim": 3,
                    "sourceDim": 3,
                    "encoding": "direct",
                    "components": ["x", "y", "z"],
                },
                {
                    "name": "rayScalars",
                    "offset": 3,
                    "dim": 6,
                    "sourceDim": 6,
                    "encoding": "direct",
                    "components": [
                        "logDistance",
                        "dotForward",
                        "u",
                        "v",
                        "angularOverTanX",
                        "angularOverTanY",
                    ],
                },
            ],
        }
    if ray_mode == "fourier117":
        return {
            "mode": "fourier117",
            "dim": 117,
            "encoding": "values_then_each_band_sin_cos",
            "segments": [
                {
                    "name": "rayDirectionFourier",
                    "offset": 0,
                    "dim": 63,
                    "sourceDim": 3,
                    "bands": 10,
                    "components": ["x", "y", "z"],
                    "layout": "[values, sin(pi*2^b*values), cos(pi*2^b*values)] for b=0..9",
                },
                {
                    "name": "rayScalarsFourier",
                    "offset": 63,
                    "dim": 54,
                    "sourceDim": 6,
                    "bands": 4,
                    "components": [
                        "logDistance",
                        "dotForward",
                        "u",
                        "v",
                        "angularOverTanX",
                        "angularOverTanY",
                    ],
                    "layout": "[values, sin(pi*2^b*values), cos(pi*2^b*values)] for b=0..3",
                },
            ],
        }
    raise ValueError(f"unsupported ray mode {ray_mode!r}")


def _query_input_layout(config: dict[str, Any]) -> dict[str, Any]:
    context_dim = int(config.get("contextQueryDim", 8))
    runtime_dim = int(config["runtimeFeatureDim"])
    ray_mode = str(config.get("rayMode", "direct9"))
    ray_layout = _ray_feature_layout(ray_mode)
    ray_dim = int(ray_layout["dim"])
    base_input_dim = int(config["baseInputDim"])
    expected_runtime_dim = 96 + 4 * context_dim + 28
    expected_base_dim = expected_runtime_dim + ray_dim + context_dim + 4
    if runtime_dim != expected_runtime_dim:
        raise ValueError(
            f"runtimeFeatureDim={runtime_dim} does not match the exported layout; "
            f"expected {expected_runtime_dim}"
        )
    if base_input_dim != expected_base_dim:
        raise ValueError(
            f"baseInputDim={base_input_dim} does not match {ray_mode}; expected {expected_base_dim}"
        )
    return {
        "runtimeFeatures": {
            "offset": 0,
            "dim": runtime_dim,
            "layout": {
                "geometry": {"offset": 0, "dim": 96},
                "contextCoefficients": {"offset": 96, "shape": [4, context_dim], "dim": 4 * context_dim},
                "survivalCoefficients": {
                    "offset": 96 + 4 * context_dim,
                    "shape": [4, 7],
                    "dim": 28,
                },
            },
        },
        "rayFeatures": {"offset": runtime_dim, **ray_layout},
        "contextQuery": {"offset": runtime_dim + ray_dim, "dim": context_dim},
        "survivalVisibility": {"offset": runtime_dim + ray_dim + context_dim, "dim": 4},
        "baseInput": {
            "dim": base_input_dim,
            "order": ["runtimeFeatures", "rayFeatures", "contextQuery", "survivalVisibility"],
        },
    }


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.checkpoint).resolve()
    features_dir = Path(args.features_dir).resolve()
    runtime_meta_path = Path(args.runtime_meta).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = dict(checkpoint.get("config") or {})
    if config.get("runtimeSchema") != "ray-context-survival-scheduler-v1":
        raise ValueError("checkpoint is not a ray-context-survival-scheduler-v1 checkpoint")
    export_config = {
        "contextQueryDim": int(config.get("contextQueryDim", 8)),
        "rayMode": str(config.get("rayMode", "direct9")),
        "survivalParameterization": str(config.get("survivalParameterization", "monotone")),
    }
    supported_ray_modes = ("direct9", "fourier117")
    if (
        export_config["contextQueryDim"] != 8
        or export_config["rayMode"] not in supported_ray_modes
        or export_config["survivalParameterization"] != "monotone"
    ):
        raise ValueError(
            "browser export supports contextQueryDim=8, rayMode in "
            f"{supported_ray_modes}, and monotone survival; received {export_config}."
        )
    query_input_layout = _query_input_layout(config)

    runtime_path = Path(args.runtime_features).resolve() if args.runtime_features else _resolve_feature_file(features_dir)
    expected_values = int(config["numInstances"]) * int(config["runtimeFeatureDim"])
    values = np.fromfile(runtime_path, dtype=np.float16)
    if values.size != expected_values:
        raise ValueError(f"runtime feature table has {values.size} values; expected {expected_values}")

    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path, int(config["numInstances"]))
    scene_min, _scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    threshold, threshold_info = _frozen_threshold(checkpoint)
    state = checkpoint.get("model") or {}
    weights = _serialize_weights(state)

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_out = output_dir / "instance_runtime_features_fp16.bin"
    aabb_out = output_dir / "instance_aabb_fp16.bin"
    weights_out = output_dir / "query_weights_fp32.json"
    values.astype(np.float16, copy=False).tofile(runtime_out)
    np.asarray(world_aabbs, dtype=np.float16).tofile(aabb_out)
    weights_out.write_text(
        json.dumps(
            {
                "schema": "ray-context-survival-owrb-query-weights-v1",
                "rayMode": export_config["rayMode"],
                "weights": weights,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    meta = {
        "schema": "ray-context-survival-owrb-webgpu-staging-v1",
        "checkpointSchema": config["runtimeSchema"],
        "checkpointSha256": sha256_file(checkpoint_path),
        "checkpointPathForAudit": str(checkpoint_path),
        "numInstances": int(config["numInstances"]),
        "numGlbs": int(config["numGlbs"]),
        "runtimeFeatureDim": int(config["runtimeFeatureDim"]),
        "runtimeFeatureLayout": query_input_layout["runtimeFeatures"]["layout"],
        "queryInputLayout": query_input_layout,
        "onlineWeightNames": list(DENSE_WEIGHT_NAMES),
        "modelInputFovYDeg": 66.0,
        "frontendRenderFovYDeg": 60.0,
        "threshold": threshold,
        "thresholdSelection": threshold_info,
        "sceneMin": scene_min.astype(np.float32).tolist(),
        "sceneSize": scene_size.astype(np.float32).tolist(),
        "instanceToGlobalGlb": instance_to_glb.astype(np.int64).tolist(),
        "modelConfig": config,
        "files": {
            "runtimeFeatures": runtime_out.name,
            "instanceAabb": aabb_out.name,
            "queryWeights": weights_out.name,
        },
        "bytes": {
            "runtimeFeatures": int(runtime_out.stat().st_size),
            "instanceAabb": int(aabb_out.stat().st_size),
            "queryWeights": int(weights_out.stat().st_size),
            "total": int(runtime_out.stat().st_size + aabb_out.stat().st_size + weights_out.stat().st_size),
        },
        "onlineOperators": {
            "featureLookup": "FP16 storage-buffer lookup",
            "rayFeatures": (
                "nine direct ray-space values"
                if export_config["rayMode"] == "direct9"
                else "117 Fourier ray values: 63 direction values followed by 54 scalar values"
            ),
            "contextQuery": "3->16->4 direction basis plus 32 FP16 coefficients",
            "survivalQuery": "3->16->4 direction basis plus 28 FP16 coefficients; monotone two-logistic survival",
            "visibilityHead": f"{int(config['baseInputDim'])}->64->32->1",
            "utilityHead": "33->32->1",
            "downloadHead": "33->32->1",
            "onlinePointEncoder": False,
            "onlineRelationPropagation": False,
            "onlineDepthBuffer": False,
            "onlineAabbProjection": False,
        },
        "runtimeSemantics": "fixed offline instance table; one lightweight query per native back-camera candidate",
        "trainingCheckpointIncluded": False,
    }
    (output_dir / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "exported", "outputDir": str(output_dir), "metadata": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtime-features", default=None)
    args = parser.parse_args()
    print(json.dumps(export(args), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
