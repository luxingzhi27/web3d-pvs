"""Strict scene registry and LOSO permissions for GCOF-PVS V5."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "pvs-v5-scene-registry-v1"
POSE_SCHEMA = "pose-csr-explicit-four-way-split-v1"
EXPECTED_SCENE_COUNT = 5
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).with_name("pvs_v5_scene_registry.json")


@dataclass(frozen=True)
class SceneAccess:
    scene_id: str
    pose_dataset: Path
    runtime_meta: Path
    assets_dir: Path
    glb_index: Path
    viewcell_shape: str
    viewcell_half_extent_m: tuple[float, float, float]
    camera_clip_m: tuple[float, float]
    split_counts: Mapping[str, int]
    allow_train_labels: bool
    allow_field_probes: bool


def _repo_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay inside the repository: {value}")
    return REPO_ROOT / path


def load_registry(path: Path = DEFAULT_REGISTRY, *, validate_files: bool = True) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"V5 registry must use {SCHEMA}")
    _validate_registry(payload, validate_files=validate_files)
    return payload


def _validate_registry(payload: Mapping[str, Any], *, validate_files: bool) -> None:
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != EXPECTED_SCENE_COUNT:
        raise ValueError("V5 registry requires exactly five real scenes")
    identifiers = [scene.get("id") for scene in scenes if isinstance(scene, Mapping)]
    if len(identifiers) != len(set(identifiers)) or any(not isinstance(x, str) for x in identifiers):
        raise ValueError("V5 scene ids must be unique strings")

    permissions = payload.get("trainingPermissions")
    if not isinstance(permissions, Mapping):
        raise ValueError("trainingPermissions are required")
    if permissions.get("heldOutSceneLabelsAllowedForTraining") is not False:
        raise ValueError("held-out labels must be forbidden during training")
    if permissions.get("heldOutSceneProbesAllowedForTraining") is not False:
        raise ValueError("held-out field probes must be forbidden during training")
    if permissions.get("testAllowedForSelection") is not False:
        raise ValueError("test must be forbidden during selection")

    for scene in scenes:
        if not isinstance(scene, Mapping):
            raise ValueError("each V5 scene entry must be an object")
        _validate_scene(scene, validate_files=validate_files)

    synthetic = payload.get("synthetic")
    if not isinstance(synthetic, Mapping):
        raise ValueError("synthetic protocol is required")
    split_total = sum(int(synthetic.get(key, -1)) for key in (
        "trainSceneCount", "validationSceneCount", "diagnosticSceneCount"
    ))
    if int(synthetic.get("sceneCount", -1)) != 120 or split_total != 120:
        raise ValueError("synthetic scene split must be 96/12/12 over 120 seeds")
    families = synthetic.get("sceneFamilies")
    if not isinstance(families, list) or len(families) != 5:
        raise ValueError("synthetic protocol requires five scene families")


def _validate_scene(scene: Mapping[str, Any], *, validate_files: bool) -> None:
    required_paths = {
        key: _repo_path(scene.get(key), f"{scene.get('id')}.{key}")
        for key in ("poseDataset", "runtimeMeta", "assetsDir", "glbIndex")
    }
    viewcell = scene.get("viewCell")
    if not isinstance(viewcell, Mapping) or viewcell.get("shape") not in {
        "horizontal_disk", "camera_aligned_box"
    }:
        raise ValueError(f"{scene.get('id')} has an invalid view-cell shape")
    half_extent = viewcell.get("halfExtentM")
    if (
        not isinstance(half_extent, list)
        or len(half_extent) != 3
        or any(float(value) < 0 for value in half_extent)
    ):
        raise ValueError(f"{scene.get('id')} has invalid view-cell half extents")
    camera_clip = scene.get("cameraClipM")
    if (
        not isinstance(camera_clip, list)
        or len(camera_clip) != 2
        or float(camera_clip[0]) <= 0
        or float(camera_clip[1]) <= float(camera_clip[0])
    ):
        raise ValueError(f"{scene.get('id')} has invalid camera clip planes")
    if not validate_files:
        return
    for key, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{scene.get('id')}.{key} does not exist: {path}")

    dataset_meta = json.loads(
        (required_paths["poseDataset"] / "dataset_meta.json").read_text(encoding="utf-8")
    )
    runtime_meta = json.loads(required_paths["runtimeMeta"].read_text(encoding="utf-8"))
    if dataset_meta.get("schema") != POSE_SCHEMA:
        raise ValueError(f"{scene.get('id')} does not use the frozen four-way split schema")
    declared_splits = {key: int(value) for key, value in scene.get("splitCounts", {}).items()}
    actual_splits = {key: int(value) for key, value in dataset_meta.get("splitCounts", {}).items()}
    for split in ("train", "calibration", "validation", "test"):
        if declared_splits.get(split) != actual_splits.get(split) or actual_splits.get(split, 0) <= 0:
            raise ValueError(f"{scene.get('id')} split mismatch for {split}")
    runtime_instances = int(runtime_meta.get("instanceCount", runtime_meta.get("componentCount", -1)))
    if runtime_instances != int(dataset_meta.get("numInstances", -2)):
        raise ValueError(f"{scene.get('id')} runtime/CSR instance count mismatch")


def build_loso_access(registry: Mapping[str, Any], held_out_scene: str) -> tuple[SceneAccess, ...]:
    scenes = registry.get("scenes")
    if not isinstance(scenes, list) or held_out_scene not in {scene.get("id") for scene in scenes}:
        raise ValueError(f"unknown held-out V5 scene: {held_out_scene}")
    result: list[SceneAccess] = []
    for scene in scenes:
        held_out = scene["id"] == held_out_scene
        result.append(
            SceneAccess(
                scene_id=str(scene["id"]),
                pose_dataset=_repo_path(scene["poseDataset"], "poseDataset"),
                runtime_meta=_repo_path(scene["runtimeMeta"], "runtimeMeta"),
                assets_dir=_repo_path(scene["assetsDir"], "assetsDir"),
                glb_index=_repo_path(scene["glbIndex"], "glbIndex"),
                viewcell_shape=str(scene["viewCell"]["shape"]),
                viewcell_half_extent_m=tuple(float(x) for x in scene["viewCell"]["halfExtentM"]),
                camera_clip_m=tuple(float(x) for x in scene["cameraClipM"]),
                split_counts={key: int(value) for key, value in scene["splitCounts"].items()},
                allow_train_labels=not held_out,
                allow_field_probes=not held_out,
            )
        )
    return tuple(result)
