"""Checkpoint-to-columnar-score export for GCOF-PVS V5."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from neural_instance_culling.config.pvs_v5_scene_registry import (
    DEFAULT_REGISTRY,
    REPO_ROOT,
    load_registry,
)
from neural_instance_culling.model.common.runtime_meta import load_runtime_meta
from neural_instance_culling.dataset.v5.schemas import validate_relation_manifest
from neural_instance_culling.model.pose_csr_dataset import PoseCSRDataset
from neural_instance_culling.model.v5.core import (
    FIELD_SHAPE,
    GCOFPVSV5,
    GEOMETRY_RELATION_SCHEMA,
    VARIANTS,
)
from neural_instance_culling.model.v5.train import CHECKPOINT_SCHEMA
from neural_instance_culling.model.v5.training_data import (
    _camera_basis,
    _query_geometry_numpy,
    _support_points,
)

from .score_bundle import (
    BUNDLE_SCHEMA,
    ColumnarScoreSidecar,
    ColumnarScoreSidecarWriter,
    FrozenPoseCSR,
    write_bundle_manifest,
)


SURFACE_HEADER = struct.Struct("<4sHHII")
SURFACE_MAGIC = b"GPV5"
COMPILED_GEOMETRY_SCHEMA = "gcof-pvs-v5-compiled-geometry-v1"


@dataclass(frozen=True)
class InferenceSceneSpec:
    scene_id: str
    pose_dataset: Path
    runtime_meta: Path
    compiled_dir: Path
    viewcell_shape: str
    viewcell_half_extent_m: tuple[float, float, float]
    camera_clip_m: tuple[float, float]
    region_manifest: Path | None = None


@dataclass(frozen=True)
class LoadedV5Checkpoint:
    path: Path
    payload: Mapping[str, Any]
    model: GCOFPVSV5
    protocol: str
    variant: str
    seed: int
    held_out_scene: str | None
    source_scenes: tuple[str, ...]


@dataclass(frozen=True)
class CompiledGeometry:
    scene: str
    variant: str
    num_units: int
    geometry: np.ndarray
    field: np.ndarray | None
    generic_latent: np.ndarray | None
    valid_unit_mask: np.ndarray


def _json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON object required: {path}")
    return value


def _prepare_empty_directory(path: Path, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    children = list(path.iterdir())
    if children and not overwrite:
        raise FileExistsError(f"output directory is not empty: {path}")
    if overwrite:
        for child in children:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def registered_scene_specs(
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    scene_ids: Sequence[str] | None = None,
    validate_files: bool = True,
) -> tuple[InferenceSceneSpec, ...]:
    registry = load_registry(Path(registry_path), validate_files=validate_files)
    wanted = None if scene_ids is None else {str(scene_id) for scene_id in scene_ids}
    result: list[InferenceSceneSpec] = []
    compiled_root = _repo_path(registry["compiledOutputRoot"])
    for raw in registry["scenes"]:
        scene_id = str(raw["id"])
        if wanted is not None and scene_id not in wanted:
            continue
        viewcell = raw["viewCell"]
        result.append(
            InferenceSceneSpec(
                scene_id=scene_id,
                pose_dataset=_repo_path(raw["poseDataset"]),
                runtime_meta=_repo_path(raw["runtimeMeta"]),
                compiled_dir=compiled_root / scene_id,
                viewcell_shape=str(viewcell["shape"]),
                viewcell_half_extent_m=tuple(float(value) for value in viewcell["halfExtentM"]),
                camera_clip_m=tuple(float(value) for value in raw["cameraClipM"]),
            )
        )
    if wanted is not None and {spec.scene_id for spec in result} != wanted:
        missing = sorted(wanted - {spec.scene_id for spec in result})
        raise ValueError(f"unknown V5 scene(s): {', '.join(missing)}")
    if not result:
        raise ValueError("no V5 scenes selected")
    return tuple(result)


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedV5Checkpoint:
    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"checkpoint must use {CHECKPOINT_SCHEMA}")
    if payload.get("testRead") is not False:
        raise ValueError("V5 inference refuses a test-tainted checkpoint")
    model_config = payload.get("modelConfig")
    if not isinstance(model_config, Mapping) or model_config.get("variant") not in VARIANTS:
        raise ValueError("checkpoint modelConfig is not a current V5 model")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("checkpoint training config is missing")
    protocol = str(config.get("protocol", ""))
    variant = str(config.get("variant", ""))
    if protocol not in {"shared", "loso"}:
        raise ValueError("checkpoint protocol must be shared or loso")
    if variant not in {"FULL", "GEOMETRY_FIELD", "GENERIC_RELATION_28", "FULL_NO_FIELD_NLL", "PBCE_OBJECTIVE"}:
        raise ValueError("checkpoint training variant is not registered")
    model_variant = "FULL" if variant in {"FULL_NO_FIELD_NLL", "PBCE_OBJECTIVE"} else variant
    if model_config.get("variant") != model_variant:
        raise ValueError("checkpoint training and model variants disagree")
    seed = int(config.get("seed", -1))
    if seed < 0:
        raise ValueError("checkpoint seed is invalid")
    held_out_raw = config.get("heldOutScene")
    held_out = None if held_out_raw is None else str(held_out_raw)
    source_raw = config.get("realSceneIds", config.get("sourceScenes", []))
    if not isinstance(source_raw, list):
        raise ValueError("checkpoint source scene list is invalid")
    source_scenes = tuple(str(value) for value in source_raw)
    model = GCOFPVSV5(model_variant).to(torch.device(device))
    if model.config != dict(model_config):
        raise ValueError("checkpoint model configuration disagrees with current V5 core")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return LoadedV5Checkpoint(
        path=checkpoint_path,
        payload=payload,
        model=model,
        protocol=protocol,
        variant=variant,
        seed=seed,
        held_out_scene=held_out,
        source_scenes=source_scenes,
    )


def _dense_asset(path: Path, *, feature_dim: int, items_per_unit: int) -> np.memmap:
    with path.open("rb") as stream:
        header = stream.read(SURFACE_HEADER.size)
    if len(header) != SURFACE_HEADER.size:
        raise ValueError(f"missing V5 surface header: {path}")
    magic, version, stored_dim, units, stored_items = SURFACE_HEADER.unpack(header)
    if magic != SURFACE_MAGIC or version != 1 or stored_dim != feature_dim or stored_items != items_per_unit:
        raise ValueError(f"invalid V5 surface header: {path}")
    count = int(units) * int(items_per_unit) * int(feature_dim)
    if path.stat().st_size != SURFACE_HEADER.size + count * 4:
        raise ValueError(f"V5 surface byte length disagrees with header: {path}")
    return np.memmap(
        path,
        dtype="<f4",
        mode="r",
        offset=SURFACE_HEADER.size,
        shape=(int(units), int(items_per_unit), int(feature_dim)),
    )


def _scene_assets(spec: InferenceSceneSpec) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Mapping[str, Any]]:
    world_aabbs, _instance_to_glb, runtime = load_runtime_meta(spec.runtime_meta)
    surface_dir = spec.compiled_dir / "surface"
    surface_manifest = _json(surface_dir / "surface_manifest.json")
    num_units = int(surface_manifest.get("numUnits", -1))
    if num_units != int(world_aabbs.shape[0]):
        raise ValueError(f"{spec.scene_id} surface/runtime unit count mismatch")
    files = surface_manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError(f"{spec.scene_id} surface manifest has no files")
    points = _dense_asset(surface_dir / str(files["points"]), feature_dim=6, items_per_unit=256)
    ratios = _dense_asset(surface_dir / str(files["sizeRatios"]), feature_dim=3, items_per_unit=1)[:, 0, :]
    valid_mask = np.ones((num_units,), dtype=bool)
    invalid_ids = np.asarray(surface_manifest.get("degenerateUnitIds", []), dtype=np.int64)
    if invalid_ids.size:
        if bool((invalid_ids < 0).any()) or bool((invalid_ids >= num_units).any()):
            raise ValueError(f"{spec.scene_id} degenerateUnitIds are outside the scene")
        valid_mask[invalid_ids] = False
    relation_dir = spec.compiled_dir / "relation"
    relation_manifest = validate_relation_manifest(_json(relation_dir / "relation_manifest.json"))
    source_ids = np.load(relation_dir / "source_ids_int64.npy", mmap_mode="r", allow_pickle=False)
    valid_edges = np.load(relation_dir / "valid_mask_bool.npy", mmap_mode="r", allow_pickle=False)
    edge_features = np.load(relation_dir / "edge_features_fp32.npy", mmap_mode="r", allow_pickle=False)
    if source_ids.shape != valid_edges.shape or source_ids.ndim != 3 or source_ids.shape[1] != 12:
        raise ValueError(f"{spec.scene_id} relation index shape is invalid")
    if edge_features.shape[:3] != source_ids.shape or edge_features.shape[3] != 8:
        raise ValueError(f"{spec.scene_id} relation feature shape is invalid")
    if source_ids.shape[0] != num_units:
        raise ValueError(f"{spec.scene_id} relation/unit count mismatch")
    if bool((source_ids[valid_edges] < 0).any()) or bool((source_ids[valid_edges] >= num_units).any()):
        raise ValueError(f"{spec.scene_id} relation source IDs are outside the geometry table")
    return {
        "points": points,
        "ratios": ratios,
        "sourceIds": source_ids,
        "validMask": valid_edges,
        "edgeFeatures": edge_features,
        "relationManifest": relation_manifest,
        "surfaceManifest": surface_manifest,
        "runtimeMeta": runtime,
    }, world_aabbs, valid_mask, source_ids, valid_edges, edge_features, relation_manifest


def _compile_relation(
    source_ids: np.ndarray,
    valid_mask: np.ndarray,
    edge_features: np.ndarray,
    relation_manifest: Mapping[str, Any],
    target_ids: np.ndarray,
) -> dict[str, Any]:
    ids = np.asarray(target_ids, dtype=np.int64)
    return {
        "source_ids": np.asarray(source_ids[ids]),
        "valid_mask": np.asarray(valid_mask[ids]),
        "edge_features": np.asarray(edge_features[ids]),
        "metadata": relation_manifest,
    }


def compile_geometry(
    checkpoint: LoadedV5Checkpoint,
    spec: InferenceSceneSpec,
    output_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    geometry_chunk_size: int = 512,
    overwrite: bool = False,
) -> Path:
    """Compile z and field/latent for one scene in bounded GPU chunks."""
    started = time.perf_counter()
    if geometry_chunk_size <= 0:
        raise ValueError("geometry_chunk_size must be positive")
    output = Path(output_dir).resolve()
    _prepare_empty_directory(output, overwrite=overwrite)
    assets, world_aabbs, valid_mask, source_ids, valid_edges, edge_features, relation_manifest = _scene_assets(spec)
    num_units = int(world_aabbs.shape[0])
    variant = checkpoint.model.variant
    target_name = "field" if variant != "GENERIC_RELATION_28" else "genericLatent"
    geometry_path = output / "geometry_fp32.bin"
    target_path = output / ("field_fp32.bin" if target_name == "field" else "generic_latent_fp32.bin")
    geometry = np.memmap(geometry_path, dtype="<f4", mode="w+", shape=(num_units, 32))
    target_shape = (num_units, *FIELD_SHAPE) if target_name == "field" else (num_units, 28)
    target = np.memmap(target_path, dtype="<f4", mode="w+", shape=target_shape)
    geometry[:] = 0.0
    target[:] = 0.0
    requested_device = torch.device(device) if device is not None else next(checkpoint.model.parameters()).device
    model = checkpoint.model.to(requested_device)
    valid_ids = np.flatnonzero(valid_mask).astype(np.int64, copy=False)
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    geometry_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, valid_ids.size, int(geometry_chunk_size)):
            ids = valid_ids[start : start + int(geometry_chunk_size)]
            points = torch.from_numpy(np.asarray(assets["points"][ids], dtype=np.float32)).to(requested_device)
            ratios = torch.from_numpy(np.asarray(assets["ratios"][ids], dtype=np.float32)).to(requested_device)
            geometry[ids] = model.encode_geometry(points, ratios).cpu().numpy().astype(np.float32, copy=False)
    geometry.flush()
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    geometry_seconds = time.perf_counter() - geometry_started
    geometry_table = torch.from_numpy(np.asarray(geometry, dtype=np.float32)).to(requested_device)
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    relation_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, valid_ids.size, int(geometry_chunk_size)):
            ids = valid_ids[start : start + int(geometry_chunk_size)]
            target_tensor = torch.from_numpy(ids).long().to(requested_device)
            relation = None
            if variant != "GEOMETRY_FIELD":
                relation = _compile_relation(source_ids, valid_edges, edge_features, relation_manifest, ids)
            compiled = model.compile_outputs(geometry_table, relation, target_tensor)
            values = compiled[target_name].cpu().numpy().astype(np.float32, copy=False)
            target[ids] = values
    target.flush()
    if requested_device.type == "cuda":
        torch.cuda.synchronize(requested_device)
    relation_seconds = time.perf_counter() - relation_started
    del geometry_table
    output_bytes = int(geometry_path.stat().st_size + target_path.stat().st_size)
    manifest = {
        "schema": COMPILED_GEOMETRY_SCHEMA,
        "version": 1,
        "scene": spec.scene_id,
        "variant": variant,
        "checkpointSchema": CHECKPOINT_SCHEMA,
        "checkpoint": str(checkpoint.path),
        "numUnits": num_units,
        "validUnitCount": int(valid_ids.size),
        "invalidUnitCount": int(num_units - valid_ids.size),
        "invalidUnitIds": np.flatnonzero(~valid_mask).astype(np.int64).tolist(),
        "files": {
            "geometry": geometry_path.name,
            target_name: target_path.name,
        },
        "shapes": {
            "geometry": [num_units, 32],
            target_name: list(target_shape),
        },
        "dtypes": {"geometry": "<f4", target_name: "<f4"},
        "compileTiming": {
            "geometryEncodeSeconds": float(geometry_seconds),
            "relationFieldCompileSeconds": float(relation_seconds),
            "totalSeconds": float(time.perf_counter() - started),
            "device": str(requested_device),
        },
        "outputBytes": output_bytes,
        "pythonInferenceReady": True,
        "browserRuntimeReady": False,
    }
    write_bundle_manifest(output / "manifest.json", manifest)
    return output / "manifest.json"


def load_compiled_geometry(path: str | Path, *, expected_scene: str, expected_variant: str) -> CompiledGeometry:
    root = Path(path).resolve()
    manifest = _json(root / "manifest.json")
    if manifest.get("schema") != COMPILED_GEOMETRY_SCHEMA:
        raise ValueError("unsupported compiled V5 geometry schema")
    if str(manifest.get("scene")) != str(expected_scene) or str(manifest.get("variant")) != str(expected_variant):
        raise ValueError("compiled geometry identity disagrees with requested inference")
    num_units = int(manifest.get("numUnits", -1))
    if num_units <= 0:
        raise ValueError("compiled geometry numUnits is invalid")
    files = manifest.get("files")
    shapes = manifest.get("shapes")
    dtypes = manifest.get("dtypes")
    if not isinstance(files, Mapping) or not isinstance(shapes, Mapping) or not isinstance(dtypes, Mapping):
        raise ValueError("compiled geometry manifest is incomplete")
    def load_array(key: str) -> np.memmap:
        if dtypes.get(key) != "<f4" or list(shapes.get(key, []))[0] != num_units:
            raise ValueError(f"compiled geometry {key} shape/dtype is invalid")
        shape = tuple(int(value) for value in shapes[key])
        path_value = root / str(files[key])
        if path_value.stat().st_size != int(np.prod(shape)) * 4:
            raise ValueError(f"compiled geometry {key} byte length is invalid")
        return np.memmap(path_value, dtype="<f4", mode="r", shape=shape)
    geometry = load_array("geometry")
    if expected_variant == "GENERIC_RELATION_28":
        latent = load_array("genericLatent")
        field = None
    else:
        field = load_array("field")
        latent = None
        if tuple(field.shape[1:]) != FIELD_SHAPE:
            raise ValueError("compiled field shape is invalid")
    invalid_count = int(manifest.get("invalidUnitCount", 0))
    valid_mask = np.ones((num_units,), dtype=bool)
    invalid = np.asarray(manifest.get("invalidUnitIds", []), dtype=np.int64)
    if invalid_count != int(invalid.size):
        raise ValueError("compiled geometry invalid-unit count disagrees with IDs")
    if invalid.size:
        if bool((invalid < 0).any()) or bool((invalid >= num_units).any()):
            raise ValueError("compiled geometry invalid-unit IDs are outside the scene")
        valid_mask[invalid] = False
    return CompiledGeometry(
        scene=str(expected_scene),
        variant=str(expected_variant),
        num_units=num_units,
        geometry=geometry,
        field=field,
        generic_latent=latent,
        valid_unit_mask=valid_mask,
    )


def _query_inputs(
    dataset: PoseCSRDataset,
    spec: InferenceSceneSpec,
    world_aabbs: np.ndarray,
    pose_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidates = np.asarray(dataset.candidate_slice(pose_index), dtype=np.uint32)
    bounds = np.asarray(world_aabbs[candidates], dtype=np.float32)
    centers = (bounds[:, :3] + bounds[:, 3:]) * 0.5
    radii = np.linalg.norm(bounds[:, 3:] - bounds[:, :3], axis=1) * 0.5
    safe_radii = np.where(radii > 1e-8, radii, 1.0).astype(np.float32)
    region_center = dataset.query_center_world(pose_index, required=True)
    view = dataset.camera_view(pose_index)
    basis = _camera_basis(view[:3])
    half_axes = np.asarray(spec.viewcell_half_extent_m, dtype=np.float32)
    if spec.viewcell_shape == "horizontal_disk":
        half_axes = np.asarray([half_axes[0], 0.0, half_axes[1]], dtype=np.float32)
        region_type = 0.0
    elif spec.viewcell_shape == "camera_aligned_box":
        region_type = 1.0
    else:
        raise ValueError(f"unsupported V5 view-cell shape: {spec.viewcell_shape}")
    support = _support_points(region_center, basis, spec.viewcell_shape, np.asarray(spec.viewcell_half_extent_m, dtype=np.float32))
    query = _query_geometry_numpy(
        centers,
        safe_radii,
        region_center,
        basis,
        half_axes,
        view[3:5],
        region_type,
        spec.camera_clip_m[0],
        spec.camera_clip_m[1],
    )
    return candidates, query, np.broadcast_to(support[None, :, :], (candidates.size, 9, 3)).copy()


def infer_split(
    checkpoint: LoadedV5Checkpoint,
    spec: InferenceSceneSpec,
    compiled: CompiledGeometry,
    output_dir: str | Path,
    *,
    split: str,
    device: str | torch.device | None = None,
    pose_chunk_size: int = 4096,
    allow_test: bool = False,
    overwrite: bool = False,
) -> Path:
    """Run one explicit PoseCSR split and write only float32 score columns."""
    if split not in {"calibration", "validation", "test"}:
        raise ValueError("V5 inference split must be calibration, validation or test")
    if split == "test" and not allow_test:
        raise PermissionError("test inference requires explicit final-test permission")
    if pose_chunk_size <= 0:
        raise ValueError("pose_chunk_size must be positive")
    output = Path(output_dir).resolve()
    _prepare_empty_directory(output, overwrite=overwrite)
    world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(spec.runtime_meta, compiled.num_units)
    dataset = PoseCSRDataset(spec.pose_dataset, compiled.num_units)
    frozen = FrozenPoseCSR(spec.pose_dataset, scene=spec.scene_id, num_instances=compiled.num_units)
    excluded_unit_ids = np.flatnonzero(~compiled.valid_unit_mask).astype(np.int64)
    validation = frozen.validate_split(
        split,
        excluded_unit_ids=excluded_unit_ids,
    )
    writer = ColumnarScoreSidecarWriter(
        output,
        scene=spec.scene_id,
        split=split,
        pose_dataset=spec.pose_dataset,
        num_instances=compiled.num_units,
        allow_test=allow_test,
        excluded_unit_ids=excluded_unit_ids,
    )
    requested_device = torch.device(device) if device is not None else next(checkpoint.model.parameters()).device
    model = checkpoint.model.to(requested_device)
    model.eval()
    with torch.inference_mode():
        for pose_index in frozen.split_indices(split).tolist():
            candidates, query, supports = _query_inputs(dataset, spec, world_aabbs, int(pose_index))
            radii = np.linalg.norm(world_aabbs[candidates, 3:] - world_aabbs[candidates, :3], axis=1) * 0.5
            invalid = (radii <= 1e-8) | (~compiled.valid_unit_mask[candidates])
            if bool(np.any(invalid & compiled.valid_unit_mask[candidates])):
                raise ValueError(f"{spec.scene_id} has a positive-manifest unit with degenerate runtime bounds")
            candidates = candidates[~invalid]
            query = query[~invalid]
            supports = supports[~invalid]
            safe_radii = radii[~invalid].astype(np.float32)
            scores = np.empty((candidates.size,), dtype=np.float32)
            for start in range(0, candidates.size, int(pose_chunk_size)):
                end = min(candidates.size, start + int(pose_chunk_size))
                ids = candidates[start:end]
                z = torch.from_numpy(np.asarray(compiled.geometry[ids], dtype=np.float32)).to(requested_device)
                q = torch.from_numpy(query[start:end]).to(requested_device)
                if checkpoint.model.variant == "GENERIC_RELATION_28":
                    latent = torch.from_numpy(np.asarray(compiled.generic_latent[ids], dtype=np.float32)).to(requested_device)
                    head_input = torch.cat([z, latent, q], dim=-1)
                else:
                    field = torch.from_numpy(np.asarray(compiled.field[ids], dtype=np.float32)).to(requested_device)
                    centers = torch.from_numpy(((world_aabbs[ids, :3] + world_aabbs[ids, 3:]) * 0.5).astype(np.float32)).to(requested_device)
                    support = torch.from_numpy(supports[start:end]).to(requested_device)
                    radius = torch.from_numpy(safe_radii[start:end]).to(requested_device)
                    stats = model.region_statistics(field, centers, support, radius)
                    head_input = torch.cat([z, stats, q], dim=-1)
                values = model.visibility_head(head_input).reshape(-1).cpu().numpy().astype(np.float32, copy=False)
                scores[start:end] = values
            writer.append_pose(int(pose_index), scores, candidate_count=int(candidates.size))
    manifest_path = writer.close()
    sidecar = ColumnarScoreSidecar(manifest_path, allow_test=allow_test)
    sidecar.validate_against_pose_csr(frozen)
    if validation.candidate_count != sidecar.candidate_count:
        raise RuntimeError("score sidecar candidate count changed after validation")
    return manifest_path


def _checkpoint_sources(checkpoint: LoadedV5Checkpoint) -> tuple[str, ...]:
    if checkpoint.protocol == "loso":
        return checkpoint.source_scenes
    return checkpoint.source_scenes


def export_score_bundle(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    registry_path: str | Path = DEFAULT_REGISTRY,
    scene_ids: Sequence[str] | None = None,
    protocol: str | None = None,
    held_out_scene: str | None = None,
    device: str | torch.device = "cuda",
    geometry_chunk_size: int = 512,
    pose_chunk_size: int = 4096,
    final_test: bool = False,
    overwrite: bool = False,
) -> Path:
    """Compile all selected scenes and export calibration/validation scores.

    Test is intentionally absent unless ``final_test=True``.  The checkpoint
    itself is never trained or calibrated by this function.
    """
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    if protocol is not None and str(protocol) != checkpoint.protocol:
        raise ValueError("requested protocol disagrees with checkpoint protocol")
    if held_out_scene is not None and held_out_scene != checkpoint.held_out_scene:
        raise ValueError("requested held-out scene disagrees with checkpoint")
    if checkpoint.protocol == "loso" and not checkpoint.held_out_scene:
        raise ValueError("LOSO checkpoint must declare heldOutScene")
    specs = registered_scene_specs(registry_path, scene_ids=scene_ids, validate_files=True)
    if checkpoint.protocol == "loso" and {spec.scene_id for spec in specs} != set(checkpoint.source_scenes) | {checkpoint.held_out_scene}:
        raise ValueError("LOSO score export must cover exactly source plus held-out scenes")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"V5 score bundle output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scene_payload: dict[str, Any] = {}
    split_names = ("calibration", "validation", "test") if final_test else ("calibration", "validation")
    for spec in specs:
        geometry_dir = output / "geometry" / spec.scene_id
        compile_geometry(
            checkpoint,
            spec,
            geometry_dir,
            device=device,
            geometry_chunk_size=geometry_chunk_size,
            overwrite=overwrite,
        )
        runtime = _json(spec.runtime_meta)
        num_instances = int(runtime.get("instanceCount", runtime.get("componentCount", -1)))
        split_payload: dict[str, str] = {}
        for split in split_names:
            sidecar_path = infer_split(
                checkpoint,
                spec,
                load_compiled_geometry(geometry_dir, expected_scene=spec.scene_id, expected_variant=checkpoint.model.variant),
                output / "scores" / spec.scene_id / split,
                split=split,
                device=device,
                pose_chunk_size=pose_chunk_size,
                allow_test=final_test,
                overwrite=overwrite,
            )
            split_payload[split] = str(sidecar_path.relative_to(output))
        scene_payload[spec.scene_id] = {
            "poseDataset": str(spec.pose_dataset),
            "numInstances": num_instances,
            "geometry": str((geometry_dir / "manifest.json").relative_to(output)),
            "splits": split_payload,
        }
    source_scenes = list(_checkpoint_sources(checkpoint))
    checkpoint_meta = {
        "path": str(checkpoint.path),
        "schema": CHECKPOINT_SCHEMA,
        "protocol": checkpoint.protocol,
        "variant": checkpoint.variant,
        "seed": checkpoint.seed,
        "testRead": False,
        "heldOutScene": checkpoint.held_out_scene,
        "sourceScenes": source_scenes,
    }
    payload = {
        "schema": BUNDLE_SCHEMA,
        "version": 1,
        "protocol": checkpoint.protocol,
        "variant": checkpoint.variant,
        "seed": checkpoint.seed,
        "testRead": bool(final_test),
        "heldOutScene": checkpoint.held_out_scene,
        "sourceScenes": source_scenes,
        "checkpoint": checkpoint_meta,
        "scoreSplits": list(split_names),
        "selectionSplit": "calibration",
        "scenes": scene_payload,
    }
    manifest_path = write_bundle_manifest(output / "manifest.json", payload)
    return manifest_path


__all__ = [
    "COMPILED_GEOMETRY_SCHEMA",
    "CompiledGeometry",
    "InferenceSceneSpec",
    "LoadedV5Checkpoint",
    "compile_geometry",
    "export_score_bundle",
    "infer_split",
    "load_checkpoint",
    "load_compiled_geometry",
    "registered_scene_specs",
]
