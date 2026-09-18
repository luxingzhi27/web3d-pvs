"""Explicit source/target asset permissions for V5 training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schemas import FOLD_ACCESS_SCHEMA, SchemaError

ASSET_GEOMETRY = "geometry"
ASSET_VISIBILITY_LABELS = "visibility_labels"
ASSET_EXTERNAL_HIT_PROBE = "external_hit_probe"
TRAINING_STAGE = "source_only_training"
CALIBRATION_STAGE = "target_calibration"
INFERENCE_STAGE = "inference"


@dataclass(frozen=True)
class FoldAccessPolicy:
    """A manifest-backed read policy.

    The policy uses scene identity and asset metadata, never path-name
    heuristics.  Geometry for a held-out scene is permitted during LOSO
    compilation, while target labels and target probes are not.
    """

    protocol: str
    source_train_scene_ids: tuple[str, ...]
    held_out_scene_id: str | None
    stage: str = TRAINING_STAGE

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": FOLD_ACCESS_SCHEMA,
            "version": 1,
            "protocol": self.protocol,
            "stage": self.stage,
            "sourceTrainSceneIds": list(self.source_train_scene_ids),
            "heldOutSceneId": self.held_out_scene_id,
            "allowHeldOutGeometry": True,
            "allowHeldOutVisibilityLabels": False,
            "allowHeldOutExternalHitProbe": False,
        }


def make_loso_training_policy(
    source_train_scene_ids: list[str] | tuple[str, ...],
    held_out_scene_id: str,
) -> FoldAccessPolicy:
    source_ids = tuple(str(scene_id) for scene_id in source_train_scene_ids)
    if not source_ids or any(not scene_id for scene_id in source_ids):
        raise SchemaError("LOSO source_train_scene_ids must be non-empty")
    if len(set(source_ids)) != len(source_ids):
        raise SchemaError("LOSO source scene IDs must be unique")
    if held_out_scene_id in source_ids:
        raise SchemaError("held-out scene cannot also be a source training scene")
    if not held_out_scene_id:
        raise SchemaError("held-out scene ID must be non-empty")
    return FoldAccessPolicy("loso", source_ids, held_out_scene_id, TRAINING_STAGE)


def make_universal_training_policy(source_train_scene_ids: list[str] | tuple[str, ...]) -> FoldAccessPolicy:
    source_ids = tuple(str(scene_id) for scene_id in source_train_scene_ids)
    if not source_ids or len(set(source_ids)) != len(source_ids):
        raise SchemaError("universal source scene IDs must be non-empty and unique")
    return FoldAccessPolicy("universal_final", source_ids, None, TRAINING_STAGE)


def validate_fold_access_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema",
        "version",
        "protocol",
        "stage",
        "sourceTrainSceneIds",
        "heldOutSceneId",
        "allowHeldOutGeometry",
        "allowHeldOutVisibilityLabels",
        "allowHeldOutExternalHitProbe",
    }
    if set(manifest) != required:
        missing = sorted(required - set(manifest))
        unknown = sorted(set(manifest) - required)
        raise SchemaError(f"fold access manifest keys invalid; missing={missing}, unknown={unknown}")
    if manifest["schema"] != FOLD_ACCESS_SCHEMA or manifest["version"] != 1:
        raise SchemaError("unsupported fold access schema")
    if manifest["protocol"] not in {"loso", "universal_final", "shared_in_domain"}:
        raise SchemaError("unsupported fold access protocol")
    if manifest["stage"] != TRAINING_STAGE:
        raise SchemaError("training access manifest must be source_only_training")
    source_ids = manifest["sourceTrainSceneIds"]
    if not isinstance(source_ids, list) or not source_ids or any(
        not isinstance(scene_id, str) or not scene_id for scene_id in source_ids
    ):
        raise SchemaError("sourceTrainSceneIds must be a non-empty list of strings")
    if len(set(source_ids)) != len(source_ids):
        raise SchemaError("sourceTrainSceneIds must be unique")
    held_out = manifest["heldOutSceneId"]
    if held_out is not None and (not isinstance(held_out, str) or not held_out):
        raise SchemaError("heldOutSceneId must be a string or null")
    if held_out in source_ids:
        raise SchemaError("held-out scene must not be listed as a source scene")
    if manifest["allowHeldOutGeometry"] is not True:
        raise SchemaError("held-out geometry must remain available for zero-shot compilation")
    if manifest["allowHeldOutVisibilityLabels"] is not False:
        raise SchemaError("held-out visibility labels must be denied during training")
    if manifest["allowHeldOutExternalHitProbe"] is not False:
        raise SchemaError("held-out external-hit probes must be denied during training")
    return dict(manifest)


def _asset_flags(asset_manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    if not isinstance(asset_manifest, Mapping):
        raise SchemaError("asset manifest must be a mapping")
    scene_id = asset_manifest.get("sceneId")
    asset_kind = asset_manifest.get("assetKind")
    split = asset_manifest.get("split")
    if not isinstance(scene_id, str) or not scene_id:
        raise SchemaError("asset manifest must declare sceneId")
    if not isinstance(asset_kind, str) or not asset_kind:
        raise SchemaError("asset manifest must declare assetKind")
    if not isinstance(split, str) or not split:
        raise SchemaError("asset manifest must declare split")
    return scene_id, asset_kind, split


def authorize_asset_read(
    policy: FoldAccessPolicy | Mapping[str, Any],
    *,
    scene_id: str,
    asset_kind: str,
    split: str,
    stage: str = TRAINING_STAGE,
) -> None:
    """Raise ``PermissionError`` unless an asset is legal for this stage."""

    if isinstance(policy, FoldAccessPolicy):
        manifest = policy.to_manifest()
    else:
        manifest = validate_fold_access_manifest(policy)
    if stage != TRAINING_STAGE:
        raise PermissionError("this reader is restricted to source-only training access")
    source_ids = set(manifest["sourceTrainSceneIds"])
    held_out = manifest["heldOutSceneId"]
    if asset_kind == ASSET_GEOMETRY:
        if scene_id not in source_ids and scene_id != held_out:
            raise PermissionError(f"geometry scene is outside the declared fold: {scene_id}")
        return
    if asset_kind not in {ASSET_VISIBILITY_LABELS, ASSET_EXTERNAL_HIT_PROBE}:
        raise PermissionError(f"unknown protected V5 asset kind: {asset_kind}")
    if scene_id == held_out:
        raise PermissionError(f"held-out {asset_kind} is forbidden during training: {scene_id}")
    if scene_id not in source_ids or split != "train":
        raise PermissionError(f"{asset_kind} is allowed only for source train scenes")


def authorize_asset_manifest_read(
    policy: FoldAccessPolicy | Mapping[str, Any], asset_manifest: Mapping[str, Any]
) -> None:
    scene_id, asset_kind, split = _asset_flags(asset_manifest)
    authorize_asset_read(
        policy,
        scene_id=scene_id,
        asset_kind=asset_kind,
        split=split,
    )


def assert_training_asset_access(
    policy: FoldAccessPolicy | Mapping[str, Any], asset_manifest: Mapping[str, Any]
) -> None:
    """Readable-name alias used by builders before opening protected files."""

    authorize_asset_manifest_read(policy, asset_manifest)
