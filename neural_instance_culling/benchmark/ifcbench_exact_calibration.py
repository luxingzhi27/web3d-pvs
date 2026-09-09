#!/usr/bin/env python3
"""Create exact IFCBench V4 score sidecars and freeze calibration thresholds.

The score sidecar keeps one flat float32 score/label/weight stream and an
offset vector for every pose.  Calibration uses the actual float32 score
change-points and one persisted set of 10,000 pose bootstrap draws.  This
script never opens the test split and does not modify a checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from train_pvs import _model_config_compatible  # noqa: E402
from pvs_model import (  # noqa: E402
    BoundedRelationSurvivalMomentModel,
    GEO_DIM,
    MODEL_SCHEMA,
    SURVIVAL_PARAMETER_DIM,
    SURVIVAL_RANK,
)
from pvs_threshold_metrics import score_distribution_summary  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


CHECKPOINT_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-checkpoint-v4"
TRAINING_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-training-v4"
SCORE_SIDECAR_SCHEMA = "pvs-ifcbench-v4-calibration-score-sidecar-v1"
BOOTSTRAP_SCHEMA = "pvs-ifcbench-v4-fixed-pose-bootstrap-v1"
CALIBRATION_SCHEMA = "pvs-ifcbench-v4-exact-calibration-v1"
VALIDATION_SCHEMA = "pvs-ifcbench-v4-frozen-threshold-validation-v1"
WEIGHTED_RECALL_FLOOR = 0.99
FORMAL_BOOTSTRAP_REPLICATES = 10_000
FORMAL_BOOTSTRAP_SEED = 20260909


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_checkpoint(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing V4 checkpoint: {resolved}")
    try:
        value = torch.load(resolved, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(resolved, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {resolved}")
    return dict(value)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(
        "cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu"
    )


def _load_geometry(path: Path, num_instances: int) -> torch.Tensor:
    expected = int(num_instances) * GEO_DIM
    if not path.is_file() or path.stat().st_size != expected * 2:
        raise ValueError(f"fixed geometry must be [{num_instances}, {GEO_DIM}] FP16: {path}")
    values = np.fromfile(path, dtype="<f2").reshape(num_instances, GEO_DIM)
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry contains non-finite values")
    return torch.from_numpy(values.astype(np.float32, copy=False))


def _model_from_checkpoint(
    checkpoint_path: Path,
    runtime_meta_path: Path,
    geometry_path: Path,
    device: torch.device,
) -> tuple[
    BoundedRelationSurvivalMomentModel,
    torch.Tensor,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    checkpoint = _load_checkpoint(checkpoint_path)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("exact IFCBench calibration requires the current V4 checkpoint schema")
    if checkpoint.get("runtimeSchema") != MODEL_SCHEMA or checkpoint.get("testRead") is not False:
        raise ValueError("V4 checkpoint schema or test provenance is invalid")
    protocol = checkpoint.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != TRAINING_SCHEMA:
        raise ValueError("V4 checkpoint training protocol is invalid")
    if protocol.get("testRead") is not False:
        raise ValueError("V4 checkpoint training protocol is not test-free")
    config = checkpoint.get("modelConfig")
    if not isinstance(config, Mapping):
        raise ValueError("V4 checkpoint modelConfig is missing")

    world_aabbs, instance_to_glb, _runtime_meta = load_runtime_meta(Path(runtime_meta_path).resolve())
    num_instances = int(world_aabbs.shape[0])
    representation_config = config.get("occlusionRepresentation")
    if not isinstance(representation_config, Mapping):
        raise ValueError("V4 checkpoint occlusion representation is missing")
    representation_mode = str(representation_config.get("mode"))
    survival_shape = config.get("survivalCoefficientShape")
    if representation_mode == "survival":
        if not (
            isinstance(survival_shape, list)
            and len(survival_shape) == 2
            and int(survival_shape[1]) == SURVIVAL_PARAMETER_DIM
        ):
            raise ValueError("V4 checkpoint survival coefficient shape is invalid")
        survival_rank = int(survival_shape[0])
    elif representation_mode == "generic28":
        survival_rank = int(representation_config.get("directionRank", SURVIVAL_RANK))
    elif representation_mode == "none":
        survival_rank = SURVIVAL_RANK
    else:
        raise ValueError(f"unsupported V4 occlusion representation: {representation_mode}")
    expected_runtime_dim = (
        GEO_DIM if representation_mode == "none" else GEO_DIM + survival_rank * SURVIVAL_PARAMETER_DIM
    )
    if int(config.get("numInstances", -1)) != num_instances or int(
        config.get("runtimeFeatureDim", -1)
    ) != expected_runtime_dim:
        raise ValueError("V4 checkpoint modelConfig does not match runtime metadata")
    depth = config.get("depthNormalization")
    instance_calibration = config.get("instanceCalibration")
    frequency = config.get("frequency")
    if not isinstance(depth, Mapping) or not isinstance(instance_calibration, Mapping):
        raise ValueError("V4 checkpoint normalization or instance calibration config is missing")
    if not isinstance(frequency, Mapping):
        raise ValueError("V4 checkpoint frequency config is missing")

    model = BoundedRelationSurvivalMomentModel(
        num_instances=num_instances,
        num_glbs=int(config.get("numGlbs", int(instance_to_glb.max()) + 1 if instance_to_glb.size else 0)),
        relation_hidden_dim=int(config.get("relationHiddenDim", 64)),
        hidden_dim=int(config.get("hiddenDim", 64)),
        survival_rank=survival_rank,
        relation_source=str(config.get("relationSource")),
        occlusion_representation=representation_mode,
        spectral_mode=str(config.get("spectralMode")),
        depth_q01=float(depth["q01"]),
        depth_q99=float(depth["q99"]),
        depth_epsilon=float(depth["epsilon"]),
        max_frequency_norm_cycles=float(frequency["maxNormCycles"]),
        instance_calibration_mode=str(instance_calibration.get("mode")),
        instance_calibration_max_abs=float(instance_calibration.get("maximumAbsoluteResidual")),
        sparse_instance_penalty=float(instance_calibration.get("sparseInstancePenalty")),
    ).to(device)
    if not _model_config_compatible(config, model.config):
        raise ValueError("V4 checkpoint modelConfig cannot be reconstructed exactly")
    model.set_instance_world_aabbs(torch.from_numpy(world_aabbs).to(device))
    model.set_instance_to_glb(torch.from_numpy(instance_to_glb).to(device))
    state = checkpoint.get("modelState")
    if not isinstance(state, Mapping):
        raise ValueError("V4 checkpoint modelState is missing")
    model.load_state_dict(state, strict=True)

    geometry_meta = checkpoint.get("geometry")
    if not isinstance(geometry_meta, Mapping) or list(geometry_meta.get("shape", [])) != [num_instances, GEO_DIM]:
        raise ValueError("V4 checkpoint geometry provenance does not match runtime metadata")
    geometry = _load_geometry(Path(geometry_path).resolve(), num_instances).to(device)
    if representation_mode == "survival":
        coefficients = torch.as_tensor(
            checkpoint.get("instanceSurvivalCoefficients"), dtype=torch.float32, device=device
        )
        prior = torch.as_tensor(
            checkpoint.get("instanceSurvivalPriorCoefficients"), dtype=torch.float32, device=device
        )
        residual = torch.as_tensor(
            checkpoint.get("instanceSurvivalCalibrationResidual"), dtype=torch.float32, device=device
        )
        expected_shape = (num_instances, survival_rank, SURVIVAL_PARAMETER_DIM)
        if (
            tuple(coefficients.shape) != expected_shape
            or tuple(prior.shape) != expected_shape
            or tuple(residual.shape) != expected_shape
            or not bool(torch.isfinite(coefficients).all())
            or not bool(torch.isfinite(prior).all())
            or not bool(torch.isfinite(residual).all())
            or not torch.allclose(coefficients, prior + residual, rtol=5e-3, atol=1e-2)
        ):
            raise ValueError("V4 checkpoint fused coefficients disagree with prior plus residual")
        runtime_features = torch.cat([geometry, coefficients.reshape(num_instances, -1)], dim=-1)
    elif representation_mode == "generic28":
        coefficients = torch.as_tensor(
            checkpoint.get("instanceOcclusionFeatures"), dtype=torch.float32, device=device
        )
        expected_shape = (num_instances, survival_rank, SURVIVAL_PARAMETER_DIM)
        if tuple(coefficients.shape) != expected_shape or not bool(torch.isfinite(coefficients).all()):
            raise ValueError("V4 generic occlusion features have an invalid shape")
        runtime_features = torch.cat([geometry, coefficients.reshape(num_instances, -1)], dim=-1)
    else:
        runtime_features = geometry
    if tuple(runtime_features.shape) != (num_instances, expected_runtime_dim) or not bool(
        torch.isfinite(runtime_features).all()
    ):
        raise ValueError("V4 runtime feature table is invalid")
    model.eval()
    return model, runtime_features, world_aabbs, instance_to_glb, checkpoint


def _prepare_empty(path: Path) -> None:
    resolved = Path(path)
    if resolved.is_symlink():
        raise ValueError(f"refusing to write through a symlink: {resolved}")
    if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
        raise FileExistsError(f"refusing to reuse non-empty output: {resolved}")
    resolved.mkdir(parents=True, exist_ok=True)


def _score_sidecar(args: argparse.Namespace) -> dict[str, Any]:
    split_name = str(args.split).lower()
    if split_name == "test":
        raise ValueError("test is forbidden during IFCBench scoring")
    if split_name not in {"calibration", "validation"}:
        raise ValueError("score sidecars are limited to calibration and validation")
    device = _device(args.device)
    model, runtime_features, world_aabbs, _instance_to_glb, checkpoint = _model_from_checkpoint(
        args.checkpoint, args.runtime_meta, args.initial_geo_features, device
    )
    dataset = PoseCSRDataset(Path(args.dataset_dir).resolve(), int(world_aabbs.shape[0]))
    if split_name not in dataset.split_ids:
        raise ValueError(f"dataset has no explicit split {split_name!r}")
    split = dataset.split(split_name)
    pose_indices = np.asarray(split.pose_indices, dtype=np.int64)
    if pose_indices.size == 0:
        raise ValueError(f"{split_name} split is empty")
    output_dir = Path(args.output_dir).resolve()
    _prepare_empty(output_dir)
    score_tmp = output_dir / "scores_f32.bin.tmp"
    label_tmp = output_dir / "labels_f32.bin.tmp"
    weight_tmp = output_dir / "weights_f32.bin.tmp"
    started = time.perf_counter()
    offsets = [0]
    total_count = 0
    rng = np.random.default_rng(0)
    with score_tmp.open("wb") as score_stream, label_tmp.open("wb") as label_stream, weight_tmp.open("wb") as weight_stream:
        for start in range(0, pose_indices.size, max(1, int(args.poses_per_batch))):
            pose_batch = pose_indices[start : start + max(1, int(args.poses_per_batch))]
            batch = split.build_pose_set_batch(
                pose_batch,
                world_aabbs,
                rng,
                max_candidates_per_pose=0,
                allow_candidate_visible_union=False,
                include_empty=True,
            )
            returned_poses = np.asarray(batch.get("pose_indices", []), dtype=np.int64)
            if not np.array_equal(returned_poses, pose_batch):
                raise ValueError("score sidecar pose order disagrees with the dataset split")
            local_offsets = np.asarray(batch["pose_offsets"], dtype=np.int64)
            if local_offsets.size != pose_batch.size + 1 or int(local_offsets[0]) != 0:
                raise ValueError("score sidecar received malformed pose offsets")
            labels = np.asarray(batch["target"], dtype="<f4")
            weights = np.asarray(batch["visible_weights"], dtype="<f4")
            if labels.size != int(local_offsets[-1]) or weights.size != labels.size:
                raise ValueError("score sidecar labels and weights do not match pose offsets")
            if not bool(np.isfinite(labels).all()) or not bool(np.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("score sidecar labels or weights are invalid")
            if labels.size:
                with torch.no_grad():
                    logits = model.compute_visibility_logits(
                        torch.from_numpy(batch["camera"]).to(device),
                        torch.from_numpy(batch["camera_view"]).to(device),
                        torch.from_numpy(batch["candidate_camera_world"]).to(device),
                        torch.from_numpy(batch["instance"]).to(device),
                        runtime_features=runtime_features,
                        query_center_world=torch.from_numpy(batch["query_center_world"]).to(device),
                        viewcell_radius_m=torch.from_numpy(batch["viewcell_radius_m"]).to(device),
                        pose_offsets=torch.from_numpy(local_offsets).to(device),
                    )
                    scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1).astype("<f4", copy=False)
            else:
                scores = np.zeros((0,), dtype="<f4")
            if scores.size != labels.size or not bool(np.isfinite(scores).all()):
                raise ValueError("score sidecar scores are invalid or misaligned")
            scores.tofile(score_stream)
            labels.tofile(label_stream)
            weights.tofile(weight_stream)
            base = offsets[-1]
            offsets.extend(base + local_offsets[1:])
            total_count += int(labels.size)

    final_paths = {
        "scores": output_dir / "scores_f32.bin",
        "labels": output_dir / "labels_f32.bin",
        "weights": output_dir / "weights_f32.bin",
    }
    score_tmp.replace(final_paths["scores"])
    label_tmp.replace(final_paths["labels"])
    weight_tmp.replace(final_paths["weights"])
    np.asarray(offsets, dtype="<u8").tofile(output_dir / "pose_offsets_u64.bin")
    pose_indices.astype("<i8", copy=False).tofile(output_dir / "pose_indices_i64.bin")
    manifest = {
        "schema": SCORE_SIDECAR_SCHEMA,
        "version": 1,
        "scene": "IFCBench/Fantasy Metropolis",
        "split": split_name,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpointSchema": checkpoint["schema"],
        "checkpointSeed": int(checkpoint.get("protocol", {}).get("seed")),
        "checkpointEpoch": int(checkpoint.get("epoch", 0)),
        "dataset": str(Path(args.dataset_dir).resolve()),
        "runtimeMeta": str(Path(args.runtime_meta).resolve()),
        "poseCount": int(pose_indices.size),
        "scoreCount": int(total_count),
        "files": {
            "scores": "scores_f32.bin",
            "labels": "labels_f32.bin",
            "weights": "weights_f32.bin",
            "poseOffsets": "pose_offsets_u64.bin",
            "poseIndices": "pose_indices_i64.bin",
        },
        "arrays": {
            "scores": {"dtype": "float32", "byteOrder": "little", "shape": [int(total_count)]},
            "labels": {"dtype": "float32", "byteOrder": "little", "shape": [int(total_count)]},
            "weights": {"dtype": "float32", "byteOrder": "little", "shape": [int(total_count)]},
            "poseOffsets": {"dtype": "uint64", "byteOrder": "little", "shape": [int(pose_indices.size + 1)]},
            "poseIndices": {"dtype": "int64", "byteOrder": "little", "shape": [int(pose_indices.size)]},
        },
        "alignment": "candidate-major scores, labels, and visible weights share pose_offsets_u64.bin",
        "candidateSemantics": "stored native back-camera candidates; GT positive union is not added",
        "labelSemantics": "candidate instance is in the view-cell visible_ids union",
        "weightSemantics": "visible_weights from the PoseCSR, not literal pixel coverage",
        "runtimeFeatureDim": int(runtime_features.shape[1]),
        "elapsedSeconds": float(time.perf_counter() - started),
        "testRead": False,
    }
    for key, path in final_paths.items():
        manifest["files"][f"{key}Bytes"] = int(path.stat().st_size)
    _write_json(output_dir / "sidecar_manifest.json", manifest)
    return manifest


@dataclass
class ScoreSidecar:
    manifest_path: Path
    manifest: dict[str, Any]
    scores: np.memmap
    labels: np.memmap
    weights: np.memmap
    pose_offsets: np.memmap
    pose_indices: np.memmap


def _load_sidecar(path: Path, expected_split: str | None = None) -> ScoreSidecar:
    root = Path(path).resolve()
    manifest_path = root / "sidecar_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("schema") != SCORE_SIDECAR_SCHEMA or manifest.get("testRead") is not False:
        raise ValueError("invalid or non-test-free score sidecar schema")
    split_name = str(manifest.get("split", "")).lower()
    if split_name == "test" or (expected_split is not None and split_name != expected_split):
        raise ValueError("score sidecar split does not satisfy the requested non-test split")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("score sidecar file manifest is missing")
    pose_count = int(manifest.get("poseCount", -1))
    score_count = int(manifest.get("scoreCount", -1))
    if pose_count <= 0 or score_count < 0:
        raise ValueError("score sidecar counts are invalid")
    def mmap(name: str, dtype: str, count: int) -> np.memmap:
        file_path = root / str(files[name])
        if not file_path.is_file() or file_path.stat().st_size != np.dtype(dtype).itemsize * count:
            raise ValueError(f"score sidecar array has the wrong size: {file_path}")
        return np.memmap(file_path, dtype=dtype, mode="r", shape=(count,))
    scores = mmap("scores", "<f4", score_count)
    labels = mmap("labels", "<f4", score_count)
    weights = mmap("weights", "<f4", score_count)
    offsets = mmap("poseOffsets", "<u8", pose_count + 1)
    poses = mmap("poseIndices", "<i8", pose_count)
    if int(offsets[0]) != 0 or int(offsets[-1]) != score_count or bool((offsets[1:] < offsets[:-1]).any()):
        raise ValueError("score sidecar pose offsets are invalid")
    if not bool(np.isfinite(scores).all()) or not bool(np.isfinite(labels).all()) or not bool(np.isfinite(weights).all()):
        raise FloatingPointError("score sidecar contains non-finite values")
    if bool((weights < 0).any()) or bool((labels < 0).any()) or bool((labels > 1).any()):
        raise ValueError("score sidecar labels or weights are outside their domains")
    return ScoreSidecar(manifest_path, manifest, scores, labels, weights, offsets, poses)


def _pose_row_ids(sidecar: ScoreSidecar) -> np.ndarray:
    counts = np.diff(np.asarray(sidecar.pose_offsets, dtype=np.int64))
    return np.repeat(np.arange(sidecar.pose_indices.size, dtype=np.int32), counts)


def _bootstrap_meta_path(path: Path) -> Path:
    return Path(f"{Path(path).resolve()}.json")


def _fixed_bootstrap(
    sidecar: ScoreSidecar,
    path: Path,
    *,
    replicates: int = FORMAL_BOOTSTRAP_REPLICATES,
    seed: int = FORMAL_BOOTSTRAP_SEED,
) -> tuple[np.memmap, np.ndarray, np.ndarray, dict[str, Any]]:
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicate count must be positive")
    row_ids = _pose_row_ids(sidecar)
    positive = np.asarray(sidecar.labels) > 0.5
    gt_mass = np.bincount(
        row_ids,
        weights=np.where(positive, np.asarray(sidecar.weights, dtype=np.float64), 0.0),
        minlength=sidecar.pose_indices.size,
    )
    valid_rows = np.flatnonzero(gt_mass > 1e-12).astype("<i8")
    if valid_rows.size == 0:
        raise ValueError("calibration sidecar contains no positive visible weight mass")
    bootstrap_path = Path(path).resolve()
    meta_path = _bootstrap_meta_path(bootstrap_path)
    expected_size = int(replicates) * int(valid_rows.size) * np.dtype("<i4").itemsize
    if bootstrap_path.exists() or meta_path.exists():
        if not bootstrap_path.is_file() or not meta_path.is_file():
            raise ValueError("fixed bootstrap index and manifest must be created together")
        metadata = _read_json(meta_path)
        if (
            metadata.get("schema") != BOOTSTRAP_SCHEMA
            or int(metadata.get("replicates", -1)) != int(replicates)
            or int(metadata.get("seed", -1)) != int(seed)
            or metadata.get("validPoseRows") != valid_rows.tolist()
            or bootstrap_path.stat().st_size != expected_size
            or metadata.get("testRead") is not False
        ):
            raise ValueError("existing fixed bootstrap index does not match this calibration split")
    else:
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bootstrap_path.with_name(f".{bootstrap_path.name}.tmp")
        rng = np.random.default_rng(int(seed))
        with temporary.open("wb") as stream:
            chunk = max(1, min(int(replicates), 512))
            for start in range(0, int(replicates), chunk):
                end = min(int(replicates), start + chunk)
                values = rng.integers(
                    0,
                    int(valid_rows.size),
                    size=(end - start, int(valid_rows.size)),
                    dtype=np.int32,
                )
                np.asarray(values, dtype="<i4").tofile(stream)
        temporary.replace(bootstrap_path)
        metadata = {
            "schema": BOOTSTRAP_SCHEMA,
            "split": str(sidecar.manifest["split"]),
            "replicates": int(replicates),
            "seed": int(seed),
            "validPoseRows": valid_rows.tolist(),
            "shape": [int(replicates), int(valid_rows.size)],
            "dtype": "int32",
            "byteOrder": "little",
            "unit": "pose row among positive-weight poses",
            "testRead": False,
        }
        _write_json(meta_path, metadata)
    bootstrap = np.memmap(
        bootstrap_path,
        dtype="<i4",
        mode="r",
        shape=(int(replicates), int(valid_rows.size)),
    )
    if bootstrap.size and (int(bootstrap.min()) < 0 or int(bootstrap.max()) >= valid_rows.size):
        raise ValueError("fixed bootstrap indices are outside the valid pose row range")
    return bootstrap, valid_rows, gt_mass, metadata


def _bootstrap_lcb(
    weighted_tp: np.ndarray,
    gt_mass: np.ndarray,
    bootstrap: np.memmap,
    valid_rows: np.ndarray,
) -> float:
    tp = np.asarray(weighted_tp, dtype=np.float64)[valid_rows]
    gt = np.asarray(gt_mass, dtype=np.float64)[valid_rows]
    values = np.empty((bootstrap.shape[0],), dtype=np.float64)
    for start in range(0, bootstrap.shape[0], 256):
        end = min(bootstrap.shape[0], start + 256)
        indices = np.asarray(bootstrap[start:end], dtype=np.int64)
        numerator = tp[indices].sum(axis=1)
        denominator = gt[indices].sum(axis=1)
        values[start:end] = np.divide(
            numerator,
            denominator,
            out=np.ones_like(numerator),
            where=denominator > 1e-12,
        )
    return float(np.quantile(values, 0.05))


def _weighted_tp_at_threshold(
    positive_scores: np.ndarray,
    positive_pose_rows: np.ndarray,
    positive_weights: np.ndarray,
    threshold: np.float32,
    pose_count: int,
) -> np.ndarray:
    selected = positive_scores >= threshold
    return np.bincount(
        positive_pose_rows[selected],
        weights=positive_weights[selected],
        minlength=pose_count,
    ).astype(np.float64, copy=False)


def _safe_value(
    positive_scores: np.ndarray,
    positive_pose_rows: np.ndarray,
    positive_weights: np.ndarray,
    gt_mass: np.ndarray,
    bootstrap: np.memmap,
    valid_rows: np.ndarray,
    threshold: np.float32,
    pose_count: int,
) -> tuple[float, float]:
    weighted_tp = _weighted_tp_at_threshold(
        positive_scores, positive_pose_rows, positive_weights, threshold, pose_count
    )
    aggregate = float(weighted_tp.sum() / max(1e-12, float(gt_mass.sum())))
    lower = _bootstrap_lcb(weighted_tp, gt_mass, bootstrap, valid_rows)
    return aggregate, lower


def _select_exact_threshold(sidecar: ScoreSidecar, bootstrap: np.memmap, valid_rows: np.ndarray, gt_mass: np.ndarray) -> dict[str, Any]:
    scores = np.asarray(sidecar.scores)
    labels = np.asarray(sidecar.labels)
    weights = np.asarray(sidecar.weights, dtype=np.float64)
    row_ids = _pose_row_ids(sidecar)
    positive = labels > 0.5
    positive_scores = scores[positive]
    positive_pose_rows = row_ids[positive]
    positive_weights = weights[positive]
    if positive_scores.size == 0 or float(positive_weights.sum()) <= 1e-12:
        raise ValueError("exact calibration requires positive visible score mass")
    score_points = np.unique(scores.astype("<f4", copy=False))
    positive_points = np.unique(positive_scores.astype("<f4", copy=False))
    low = 0
    high = int(score_points.size) - 1
    best = -1
    while low <= high:
        middle = (low + high) // 2
        threshold = np.float32(score_points[middle])
        aggregate, lower = _safe_value(
            positive_scores,
            positive_pose_rows,
            positive_weights,
            gt_mass,
            bootstrap,
            valid_rows,
            threshold,
            sidecar.pose_indices.size,
        )
        if aggregate > WEIGHTED_RECALL_FLOOR and lower > WEIGHTED_RECALL_FLOOR:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    if best < 0:
        threshold = np.float32(score_points[0])
        status = "no_qualified_safety_workpoint"
    else:
        threshold = np.float32(score_points[best])
        status = "safe"
    next_higher = None if best < 0 or best + 1 >= score_points.size else np.float32(score_points[best + 1])
    next_higher_safety = None
    if next_higher is not None:
        aggregate, lower = _safe_value(
            positive_scores,
            positive_pose_rows,
            positive_weights,
            gt_mass,
            bootstrap,
            valid_rows,
            next_higher,
            sidecar.pose_indices.size,
        )
        next_higher_safety = {
            "aggregateWeightedRecall": aggregate,
            "aggregateWeightedRecallLowerConfidenceBound": lower,
            "safe": bool(aggregate > WEIGHTED_RECALL_FLOOR and lower > WEIGHTED_RECALL_FLOOR),
        }
    return {
        "status": status,
        "threshold": float(threshold),
        "thresholdDtype": "float32",
        "candidateCount": int(score_points.size),
        "selectedCandidateIndex": int(best if best >= 0 else 0),
        "nextHigherThreshold": None if next_higher is None else float(next_higher),
        "nextHigherSafety": next_higher_safety,
        "allScoreChangePointCount": int(np.unique(scores).size),
        "positiveScoreChangePointCount": int(positive_points.size),
        "minimumScore": float(np.min(scores)),
        "maximumScore": float(np.max(scores)),
        "rule": "highest float32 score change-point with aggregateWeightedRecall > 0.99 and bootstrap LCB > 0.99",
        "searchOptimization": "negative-only points are retained; weighted recall is evaluated from positive rows",
    }


def _resource_metrics(
    sidecar: ScoreSidecar,
    threshold: float,
    dataset: PoseCSRDataset,
    instance_to_glb: np.ndarray,
    glb_bytes: np.ndarray,
) -> dict[str, float]:
    scores = np.asarray(sidecar.scores)
    total_candidate_glbs = total_predicted_glbs = 0.0
    total_candidate_bytes = total_predicted_bytes = 0.0
    total_utility = total_gt_utility = 0.0
    for row, pose in enumerate(np.asarray(sidecar.pose_indices, dtype=np.int64).tolist()):
        start = int(sidecar.pose_offsets[row])
        end = int(sidecar.pose_offsets[row + 1])
        candidates = np.asarray(dataset.candidate_slice(int(pose)), dtype=np.int64)
        if candidates.size != end - start:
            raise ValueError("score sidecar candidate counts disagree with the dataset")
        visible_ids, visible_weights = dataset.visible_slice(int(pose))
        candidate_glbs = np.unique(instance_to_glb[candidates]) if candidates.size else np.zeros(0, dtype=np.int64)
        predicted = candidates[scores[start:end] >= np.float32(threshold)]
        predicted_glbs = np.unique(instance_to_glb[predicted]) if predicted.size else np.zeros(0, dtype=np.int64)
        total_candidate_glbs += float(candidate_glbs.size)
        total_predicted_glbs += float(predicted_glbs.size)
        total_candidate_bytes += float(glb_bytes[candidate_glbs].sum()) if candidate_glbs.size else 0.0
        total_predicted_bytes += float(glb_bytes[predicted_glbs].sum()) if predicted_glbs.size else 0.0
        found = np.isin(visible_ids, predicted, assume_unique=False)
        total_utility += float(np.asarray(visible_weights, dtype=np.float64)[found].sum())
        total_gt_utility += float(np.asarray(visible_weights, dtype=np.float64).sum())
    pose_count = max(1, int(sidecar.pose_indices.size))
    return {
        "avgCandidateGlbCount": total_candidate_glbs / pose_count,
        "avgPredictedGlbCount": total_predicted_glbs / pose_count,
        "avgCandidateGlbBytes": total_candidate_bytes / pose_count,
        "avgPredictedGlbBytes": total_predicted_bytes / pose_count,
        "glbCountReduction": 1.0 - total_predicted_glbs / max(1.0, total_candidate_glbs),
        "glbByteReduction": 1.0 - total_predicted_bytes / max(1.0, total_candidate_bytes),
        "downloadUtilityRecall": total_utility / max(1e-12, total_gt_utility),
    }


def _load_glb_bytes(index_path: Path, root: Path, num_glbs: int) -> np.ndarray:
    payload = _read_json(Path(index_path).resolve())
    values = np.zeros((int(num_glbs),), dtype=np.float64)
    for entry in payload.get("entries", []):
        gid = int(entry.get("globalId", -1))
        candidate = Path(root).resolve() / str(entry.get("path", ""))
        if 0 <= gid < values.size and candidate.is_file():
            values[gid] = float(candidate.stat().st_size)
    positive = values[values > 0]
    values[values <= 0] = float(np.median(positive)) if positive.size else 1.0
    return values


def _metrics_at_threshold(
    sidecar: ScoreSidecar,
    threshold: float,
    bootstrap: np.memmap,
    valid_rows: np.ndarray,
    gt_mass: np.ndarray,
    *,
    score_distribution: dict[str, Any] | None = None,
    resource: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    scores = np.asarray(sidecar.scores)
    labels = np.asarray(sidecar.labels)
    weights = np.asarray(sidecar.weights, dtype=np.float64)
    row_ids = _pose_row_ids(sidecar)
    positive = labels > 0.5
    predicted = scores >= np.float32(threshold)
    tp_mask = predicted & positive
    fp_mask = predicted & ~positive
    fn_mask = ~predicted & positive
    tn_mask = ~predicted & ~positive
    tp = float(np.count_nonzero(tp_mask))
    fp = float(np.count_nonzero(fp_mask))
    fn = float(np.count_nonzero(fn_mask))
    tn = float(np.count_nonzero(tn_mask))
    candidate_count = tp + fp + fn + tn
    gt_count = tp + fn
    pred_count = tp + fp
    weighted_tp = np.bincount(
        row_ids,
        weights=np.where(tp_mask, weights, 0.0),
        minlength=sidecar.pose_indices.size,
    ).astype(np.float64, copy=False)
    aggregate_weighted_recall = float(weighted_tp.sum() / max(1e-12, float(gt_mass.sum())))
    aggregate_lcb = _bootstrap_lcb(weighted_tp, gt_mass, bootstrap, valid_rows)
    pose_tp = np.bincount(row_ids, weights=tp_mask.astype(np.float64), minlength=sidecar.pose_indices.size)
    pose_fp = np.bincount(row_ids, weights=fp_mask.astype(np.float64), minlength=sidecar.pose_indices.size)
    pose_fn = np.bincount(row_ids, weights=fn_mask.astype(np.float64), minlength=sidecar.pose_indices.size)
    pose_tn = np.bincount(row_ids, weights=tn_mask.astype(np.float64), minlength=sidecar.pose_indices.size)
    pose_candidates = np.diff(np.asarray(sidecar.pose_offsets, dtype=np.int64)).astype(np.float64)
    pose_precision = np.divide(pose_tp, pose_tp + pose_fp, out=np.ones_like(pose_tp), where=pose_tp + pose_fp > 0)
    pose_recall = np.divide(pose_tp, pose_tp + pose_fn, out=np.ones_like(pose_tp), where=pose_tp + pose_fn > 0)
    pose_specificity = np.divide(pose_tn, pose_tn + pose_fp, out=np.ones_like(pose_tn), where=pose_tn + pose_fp > 0)
    pose_weighted_recall = np.divide(weighted_tp, gt_mass, out=np.ones_like(weighted_tp), where=gt_mass > 1e-12)
    pose_accuracy = np.divide(pose_tp + pose_tn, np.maximum(1.0, pose_candidates))
    pose_f1 = 2.0 * pose_precision * pose_recall / np.maximum(1e-8, pose_precision + pose_recall)
    pose_balanced = 0.5 * (pose_recall + pose_specificity)
    pose_useful = pose_tn / np.maximum(1.0, pose_candidates)
    pose_bad = pose_fn / np.maximum(1.0, pose_candidates)
    aggregate_recall = tp / max(1.0, gt_count)
    aggregate_precision = tp / max(1.0, tp + fp)
    aggregate_specificity = tn / max(1.0, tn + fp)
    row: dict[str, Any] = {
        "threshold": float(np.float32(threshold)),
        "thresholdIsScoreChangePoint": bool(np.any(scores == np.float32(threshold))),
        "pose_precision": float(pose_precision.mean()),
        "pose_recall": float(pose_recall.mean()),
        "pose_weighted_recall": float(pose_weighted_recall.mean()),
        "pose_f1": float(pose_f1.mean()),
        "pose_jaccard": float(np.mean(np.divide(pose_tp, pose_tp + pose_fp + pose_fn, out=np.ones_like(pose_tp), where=pose_tp + pose_fp + pose_fn > 0))),
        "pose_accuracy": float(pose_accuracy.mean()),
        "pose_balanced_accuracy": float(pose_balanced.mean()),
        "pose_specificity": float(pose_specificity.mean()),
        "pose_useful_cull": float(pose_useful.mean()),
        "pose_bad_cull": float(pose_bad.mean()),
        "agg_precision": float(aggregate_precision),
        "agg_recall": float(aggregate_recall),
        "agg_weighted_recall": float(aggregate_weighted_recall),
        "agg_specificity": float(aggregate_specificity),
        "agg_accuracy": float((tp + tn) / max(1.0, candidate_count)),
        "agg_balanced_accuracy": float(0.5 * (aggregate_recall + aggregate_specificity)),
        "agg_useful_cull": float(tn / max(1.0, candidate_count)),
        "agg_bad_cull": float(fn / max(1.0, candidate_count)),
        "agg_f1": float(2.0 * aggregate_precision * aggregate_recall / max(1e-8, aggregate_precision + aggregate_recall)),
        "avg_pred_count": float(pred_count / max(1, sidecar.pose_indices.size)),
        "avg_gt_count": float(gt_count / max(1, sidecar.pose_indices.size)),
        "avg_candidate_count": float(candidate_count / max(1, sidecar.pose_indices.size)),
        "candidate_reduction_ratio": float(1.0 - pred_count / max(1.0, candidate_count)),
        "aggregateWeightedRecall": float(aggregate_weighted_recall),
        "aggregateWeightedRecallLowerConfidenceBound": float(aggregate_lcb),
        "aggregate_weighted_recall": float(aggregate_weighted_recall),
        "weighted_recall_lower_confidence_bound": float(aggregate_lcb),
        "poseMacroWeightedRecall": float(pose_weighted_recall.mean()),
        "poseMacroWeightedRecallLowerConfidenceBound": None,
        "weightedRecallBootstrapReplicates": int(bootstrap.shape[0]),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "eval_pose_count": int(sidecar.pose_indices.size),
        "scoreCount": int(scores.size),
        "positiveCount": int(np.count_nonzero(positive)),
        "positiveFraction": float(np.mean(positive)) if positive.size else 0.0,
    }
    if score_distribution is not None:
        row["scoreDistribution"] = score_distribution
    if resource is not None:
        row.update({str(key): float(value) for key, value in resource.items()})
    return row


def _formal_bootstrap_args(args: argparse.Namespace) -> tuple[int, int]:
    replicates = int(args.bootstrap_replicates)
    if replicates != FORMAL_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"formal IFCBench calibration requires exactly {FORMAL_BOOTSTRAP_REPLICATES} pose bootstrap replicates"
        )
    return replicates, int(args.bootstrap_seed)


def _calibrate(args: argparse.Namespace) -> dict[str, Any]:
    replicates, seed = _formal_bootstrap_args(args)
    sidecar = _load_sidecar(args.sidecar, expected_split="calibration")
    bootstrap, valid_rows, gt_mass, bootstrap_meta = _fixed_bootstrap(
        sidecar,
        args.bootstrap_indexes,
        replicates=replicates,
        seed=seed,
    )
    distribution = score_distribution_summary(
        np.asarray(sidecar.scores), np.asarray(sidecar.labels), np.asarray(sidecar.weights)
    )
    selection = _select_exact_threshold(sidecar, bootstrap, valid_rows, gt_mass)
    selected_row = _metrics_at_threshold(
        sidecar,
        selection["threshold"],
        bootstrap,
        valid_rows,
        gt_mass,
        score_distribution=distribution,
    )
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "scene": "IFCBench/Fantasy Metropolis",
        "split": "calibration",
        "checkpoint": sidecar.manifest["checkpoint"],
        "checkpointSeed": sidecar.manifest["checkpointSeed"],
        "checkpointEpoch": sidecar.manifest["checkpointEpoch"],
        "scoreSidecar": str(sidecar.manifest_path),
        "status": selection["status"],
        "selection": selection,
        "selected": selected_row,
        "bootstrap": {
            "schema": bootstrap_meta["schema"],
            "indexFile": str(Path(args.bootstrap_indexes).resolve()),
            "manifestFile": str(_bootstrap_meta_path(args.bootstrap_indexes)),
            "replicates": replicates,
            "seed": seed,
            "validPoseCount": int(valid_rows.size),
            "unit": "pose",
        },
        "thresholdSource": "actual float32 score change-points from calibration sidecar",
        "predictionRule": "score >= threshold",
        "testRead": False,
    }
    _write_json(Path(args.output).resolve(), payload)
    return payload


def _evaluate_frozen(args: argparse.Namespace) -> dict[str, Any]:
    replicates, seed = _formal_bootstrap_args(args)
    calibration = _read_json(Path(args.calibration).resolve())
    if calibration.get("schema") != CALIBRATION_SCHEMA or calibration.get("testRead") is not False:
        raise ValueError("frozen validation requires an exact, test-free calibration summary")
    if calibration.get("status") != "safe":
        raise ValueError("validation replay requires a calibration checkpoint with a safe threshold")
    selection = calibration.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("calibration summary has no frozen selection")
    threshold = float(selection["threshold"])
    sidecar = _load_sidecar(args.sidecar, expected_split="validation")
    if str(sidecar.manifest["checkpoint"]) != str(calibration["checkpoint"]):
        raise ValueError("validation score sidecar belongs to a different checkpoint")
    bootstrap, valid_rows, gt_mass, bootstrap_meta = _fixed_bootstrap(
        sidecar,
        args.bootstrap_indexes,
        replicates=replicates,
        seed=seed,
    )
    distribution = score_distribution_summary(
        np.asarray(sidecar.scores), np.asarray(sidecar.labels), np.asarray(sidecar.weights)
    )
    row = _metrics_at_threshold(
        sidecar,
        threshold,
        bootstrap,
        valid_rows,
        gt_mass,
        score_distribution=distribution,
    )
    payload = {
        "schema": VALIDATION_SCHEMA,
        "scene": "IFCBench/Fantasy Metropolis",
        "split": "validation",
        "checkpoint": sidecar.manifest["checkpoint"],
        "checkpointSeed": sidecar.manifest["checkpointSeed"],
        "checkpointEpoch": sidecar.manifest["checkpointEpoch"],
        "scoreSidecar": str(sidecar.manifest_path),
        "calibrationSummary": str(Path(args.calibration).resolve()),
        "threshold": threshold,
        "thresholdSource": "checkpoint-specific calibration exact score change-point",
        "metrics": row,
        "bootstrap": {
            "schema": bootstrap_meta["schema"],
            "indexFile": str(Path(args.bootstrap_indexes).resolve()),
            "manifestFile": str(_bootstrap_meta_path(args.bootstrap_indexes)),
            "replicates": replicates,
            "seed": seed,
            "validPoseCount": int(valid_rows.size),
            "unit": "pose",
        },
        "testRead": False,
    }
    _write_json(Path(args.output).resolve(), payload)
    return payload


def _add_score_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--initial-geo-features", type=Path, required=True)


def _add_calibration_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--bootstrap-indexes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=FORMAL_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=FORMAL_BOOTSTRAP_SEED)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="write float32 calibration/validation score arrays")
    _add_score_args(score_parser)
    score_parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    score_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser.add_argument("--poses-per-batch", type=int, default=2)
    score_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")

    calibration_parser = subparsers.add_parser("calibrate", help="freeze an exact calibration threshold")
    _add_calibration_args(calibration_parser)

    validation_parser = subparsers.add_parser("evaluate", help="evaluate a frozen threshold on validation")
    _add_calibration_args(validation_parser)
    validation_parser.add_argument("--calibration", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "score":
        manifest = _score_sidecar(args)
        print(json.dumps({"command": "score", "manifest": str(Path(args.output_dir).resolve()), "poseCount": manifest["poseCount"], "scoreCount": manifest["scoreCount"], "testRead": False}))
    elif args.command == "calibrate":
        payload = _calibrate(args)
        print(json.dumps({"command": "calibrate", "output": str(Path(args.output).resolve()), "status": payload["status"], "threshold": payload["selection"]["threshold"], "testRead": False}))
    else:
        payload = _evaluate_frozen(args)
        print(json.dumps({"command": "evaluate", "output": str(Path(args.output).resolve()), "threshold": payload["threshold"], "testRead": False}))


if __name__ == "__main__":
    main()
