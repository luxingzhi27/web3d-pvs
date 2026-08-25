#!/usr/bin/env python3
"""Compare one browser V4 query with the exported PyTorch runtime contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = REPO_ROOT / "neural_instance_culling" / "model"
sys.path.insert(0, str(MODEL_ROOT))

from common.viewcell_ray_space import (  # noqa: E402
    build_horizontal_disk_ray_query,
    normalized_relative_log_depth,
)
from pvs_model import BoundedRelationSurvivalMomentModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--max-absolute-error", type=float, default=0.002)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return dict(value)


def build_model(checkpoint: Mapping[str, Any]) -> BoundedRelationSurvivalMomentModel:
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint modelConfig is missing")
    depth = config.get("depthNormalization")
    calibration = config.get("instanceCalibration")
    frequency = config.get("frequency")
    if not all(isinstance(item, Mapping) for item in (depth, calibration, frequency)):
        raise ValueError("checkpoint runtime configuration is incomplete")
    representation = config.get("occlusionRepresentation")
    representation_mode = (
        str(representation.get("mode"))
        if isinstance(representation, Mapping)
        else str(checkpoint.get("occlusionRepresentation", "survival"))
    )
    model = BoundedRelationSurvivalMomentModel(
        num_instances=int(config["numInstances"]),
        num_glbs=int(config["numGlbs"]),
        relation_hidden_dim=int(config.get("relationHiddenDim", 64)),
        hidden_dim=int(config["hiddenDim"]),
        relation_source=str(config["relationSource"]),
        occlusion_representation=representation_mode,
        spectral_mode=str(config["spectralMode"]),
        depth_q01=float(depth["q01"]),
        depth_q99=float(depth["q99"]),
        depth_epsilon=float(depth["epsilon"]),
        max_frequency_norm_cycles=float(frequency["maxNormCycles"]),
        instance_calibration_mode=str(calibration["mode"]),
        instance_calibration_max_abs=float(calibration["maximumAbsoluteResidual"]),
        sparse_instance_penalty=float(calibration["sparseInstancePenalty"]),
    )
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint modelState is missing")
    model.load_state_dict(state, strict=True)
    return model


def apply_exported_query_parameters(
    model: BoundedRelationSurvivalMomentModel,
    meta: Mapping[str, Any],
    asset_dir: Path,
) -> None:
    packed = np.fromfile(asset_dir / "query_weights_fp16.bin", dtype="<f2")
    state = model.state_dict()
    for descriptor in meta["networkWeights"]["layout"]:
        name = str(descriptor["name"])
        shape = tuple(int(value) for value in descriptor["shape"])
        offset = int(descriptor["offsetElements"])
        count = int(descriptor["elementCount"])
        value = packed[offset : offset + count].astype(np.float32).reshape(shape)
        state[name] = torch.from_numpy(value)
    frequency = np.fromfile(asset_dir / "frequency_cycles_fp32.bin", dtype="<f4").reshape(16, 9)
    chi = np.fromfile(asset_dir / "chi_table_fp32.bin", dtype="<f4")
    state["moment_query.frequency_cycles"] = torch.from_numpy(frequency.copy())
    state["moment_query.chi_table"] = torch.from_numpy(chi.copy())
    model.load_state_dict(state, strict=True)


@torch.no_grad()
def main() -> int:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    capture = load_json(Path(args.capture).resolve())
    meta = load_json(asset_dir / "model_meta.json")
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model(checkpoint).eval()
    apply_exported_query_parameters(model, meta, asset_dir)

    candidate_ids = np.asarray(capture["candidateInstanceIds"], dtype=np.int64)
    browser_scores = np.asarray(capture["candidateScores"], dtype=np.float32)
    if candidate_ids.ndim != 1 or browser_scores.shape != candidate_ids.shape or candidate_ids.size == 0:
        raise ValueError("browser capture candidate arrays are invalid")
    runtime_features = np.fromfile(
        asset_dir / "instance_runtime_features_fp16.bin", dtype="<f2"
    ).astype(np.float32).reshape(int(meta["numInstances"]), 124)
    aabbs = np.fromfile(asset_dir / "instance_aabb_fp32.bin", dtype="<f4").reshape(-1, 6)
    camera = capture["camera"]
    center = torch.tensor(camera["position"], dtype=torch.float32).reshape(1, 3)
    forward = np.asarray(camera["forward"], dtype=np.float32)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    tan_y = np.tan(np.deg2rad(float(meta["query"]["modelInputFovYDeg"]) * 0.5))
    camera_view = torch.tensor(
        [*forward.tolist(), tan_y * float(camera["aspect"]), tan_y], dtype=torch.float32
    ).reshape(1, 5)
    rows = torch.from_numpy(aabbs[candidate_ids].copy())
    query = build_horizontal_disk_ray_query(
        center.expand(candidate_ids.size, -1),
        camera_view.expand(candidate_ids.size, -1),
        rows,
        float(meta["query"]["viewcellRadiusM"]),
    )
    depth_meta = meta["depth"]
    depth = normalized_relative_log_depth(
        query.distance_m,
        query.instance_radius_m,
        float(depth_meta["q01"]),
        float(depth_meta["q99"]),
        epsilon=float(depth_meta["epsilon"]),
    )
    logits, auxiliary = model.forward_batch(
        torch.from_numpy(candidate_ids),
        query.center_view,
        query.disk_axes,
        depth,
        {"runtime_features": torch.from_numpy(runtime_features)},
    )
    reference_scores = torch.sigmoid(logits).reshape(-1).cpu().numpy()
    diagnostic_stride = int(capture.get("diagnosticOutputFloats", 0))
    browser_diagnostics = np.asarray(capture.get("candidateDiagnostics", []), dtype=np.float32)
    stage_reports: dict[str, dict[str, float]] = {}
    if diagnostic_stride:
        if diagnostic_stride != 92:
            raise ValueError(f"browser diagnostic stride must be 92, got {diagnostic_stride}")
        if browser_diagnostics.size != candidate_ids.size * diagnostic_stride:
            raise ValueError("browser diagnostic rows do not match the candidate count")
        diagnostic_rows = browser_diagnostics.reshape(candidate_ids.size, diagnostic_stride)
        reference_stages = {
            "centerView": query.center_view.cpu().numpy(),
            "diskAxes": query.disk_axes.reshape(candidate_ids.size, -1).cpu().numpy(),
            "spectralFeatures": auxiliary["spectral_features"].cpu().numpy(),
        }
        browser_stages = {
            "centerView": diagnostic_rows[:, 1:10],
            "diskAxes": diagnostic_rows[:, 10:28],
            "spectralFeatures": diagnostic_rows[:, 28:92],
        }
        for name, reference in reference_stages.items():
            difference = np.abs(browser_stages[name] - reference)
            stage_reports[name] = {
                "meanAbsoluteError": float(difference.mean()),
                "maximumAbsoluteError": float(difference.max(initial=0.0)),
                "mismatchedValueCountAt1e5": int(np.count_nonzero(difference > 1e-5)),
            }
    absolute = np.abs(reference_scores - browser_scores)
    threshold = float(meta["threshold"])
    browser_visible = set(candidate_ids[browser_scores >= threshold].tolist())
    reference_visible = set(candidate_ids[reference_scores >= threshold].tolist())
    reported_visible = set(int(value) for value in capture.get("visibleInstanceIds", []))
    report = {
        "candidateCount": int(candidate_ids.size),
        "threshold": threshold,
        "maximumAbsoluteProbabilityError": float(absolute.max(initial=0.0)),
        "meanAbsoluteProbabilityError": float(absolute.mean()),
        "thresholdDecisionMismatchCount": len(browser_visible ^ reference_visible),
        "workerVisibleSetMismatchCount": len(browser_visible ^ reported_visible),
        "browserVisibleCount": len(browser_visible),
        "referenceVisibleCount": len(reference_visible),
        "stageParity": stage_reports,
        "largestErrors": [
            {
                "instanceId": int(candidate_ids[index]),
                "browserProbability": float(browser_scores[index]),
                "referenceProbability": float(reference_scores[index]),
                "absoluteError": float(absolute[index]),
                "centerView": [float(value) for value in query.center_view[index].tolist()],
                "maximumDiskAxis": float(query.disk_axes[index].abs().max()),
                "distanceM": float(query.distance_m[index]),
                "instanceRadiusM": float(query.instance_radius_m[index]),
            }
            for index in np.argsort(absolute)[-10:][::-1]
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["maximumAbsoluteProbabilityError"] > float(args.max_absolute_error):
        return 1
    if report["thresholdDecisionMismatchCount"] != 0 or report["workerVisibleSetMismatchCount"] != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
