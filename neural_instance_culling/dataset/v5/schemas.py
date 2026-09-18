"""Strict schema constants and validators for the V5 preprocessing assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

LOCAL_SURFACE_SCHEMA = "gcof-pvs-v5-local-surface-v1"
RELATION_SCHEMA = "pvs-geometry-proxy-relation-csr-v1"
REGION_SUPPORT_SCHEMA = "gcof-pvs-v5-region-support-v1"
EXTERNAL_HIT_PROBE_SCHEMA = "parallel_external_hit_current_status-v1"
FOLD_ACCESS_SCHEMA = "gcof-pvs-v5-fold-access-v1"
SYNTHETIC_SCENE_SCHEMA = "gcof-pvs-v5-synthetic-scene-catalog-v1"

ANCHOR_SCHEMA = "icosahedron12"
RELATION_TOP_K = 8
RELATION_ANCHOR_COUNT = 12
EDGE_FEATURE_DIM = 8
SUPPORT_POINT_COUNT = 9
POINTS_PER_UNIT = 256


class SchemaError(ValueError):
    """Raised when a V5 manifest or record violates its fixed contract."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str] = (),
    name: str = "manifest",
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    keys = set(value)
    missing = sorted(required_set - keys)
    unknown = sorted(keys - required_set - optional_set)
    if missing:
        raise SchemaError(f"{name} is missing required keys: {', '.join(missing)}")
    if unknown:
        raise SchemaError(f"{name} has unknown keys: {', '.join(unknown)}")


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{name} must be a non-empty string")
    return value


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchemaError(f"{name} must be a non-negative integer")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaError(f"{name} must be a boolean")
    return value


def _require_string_list(value: Any, name: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(value, list) or (nonempty and not value):
        raise SchemaError(f"{name} must be a {'non-empty ' if nonempty else ''}list")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_require_string(item, f"{name}[{index}]"))
    return result


def validate_relation_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the geometry-only relation provenance and dimensions.

    This validator is deliberately strict.  In particular, a relation asset
    cannot be accepted merely because it has twelve anchors and eight slots:
    its provenance must explicitly state that no visibility labels were used.
    """

    value = _require_mapping(manifest, "relation manifest")
    _require_exact_keys(
        value,
        {
            "schema",
            "version",
            "source",
            "usesVisibilityLabels",
            "anchors",
            "anchorCount",
            "K",
            "projection",
            "depthPredicate",
            "edgeFeatureSchema",
            "edgeFeatureDim",
            "storage",
            "numUnits",
        },
        name="relation manifest",
    )
    if value["schema"] != RELATION_SCHEMA:
        raise SchemaError("relation manifest is not a V5 geometry-proxy relation")
    if value["version"] != 1:
        raise SchemaError("unsupported relation manifest version")
    if value["source"] != "geometry_only":
        raise SchemaError("V5 relations must have source=geometry_only")
    if value["usesVisibilityLabels"] is not False:
        raise SchemaError("V5 relations must set usesVisibilityLabels=false")
    if value["anchors"] != ANCHOR_SCHEMA or value["anchorCount"] != RELATION_ANCHOR_COUNT:
        raise SchemaError("relation anchors must be the fixed 12-direction icosahedron")
    if value["K"] != RELATION_TOP_K:
        raise SchemaError("V5 relation K is fixed at 8")
    if value["projection"] != "orthographic_aabb_overlap":
        raise SchemaError("relation projection must be orthographic_aabb_overlap")
    if value["depthPredicate"] != "source_center_forward_and_depth_interval_camera_side":
        raise SchemaError("relation depth predicate is not the V5 camera-side interval predicate")
    if value["edgeFeatureSchema"] != "relative_direction_log_distance_log_radius_overlap_log_gap":
        raise SchemaError("unsupported V5 edge feature schema")
    if value["edgeFeatureDim"] != EDGE_FEATURE_DIM:
        raise SchemaError("V5 edge features must have eight dimensions")
    if value["storage"] != "dense_topk":
        raise SchemaError("unsupported V5 relation storage")
    _require_positive_int(value["numUnits"], "relation manifest numUnits")
    return dict(value)


def validate_local_surface_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(manifest, "local-surface manifest")
    _require_exact_keys(
        value,
        {
            "schema",
            "version",
            "pointsPerUnit",
            "pointFeatureDim",
            "pointFeatureOrder",
            "sizeRatioDim",
            "storage",
            "numUnits",
            "unitIds",
            "samplingSeed",
            "transformPolicy",
            "degenerateUnitIds",
            "files",
        },
        name="local-surface manifest",
    )
    if value["schema"] != LOCAL_SURFACE_SCHEMA or value["version"] != 1:
        raise SchemaError("unsupported local-surface schema")
    if value["pointsPerUnit"] != POINTS_PER_UNIT:
        raise SchemaError("V5 local surface sampling must use 256 points per unit")
    if value["pointFeatureDim"] != 6 or value["pointFeatureOrder"] != "normalized_xyz_unit_normal":
        raise SchemaError("V5 surface point features must be normalized xyz plus unit normal")
    if value["sizeRatioDim"] != 3:
        raise SchemaError("V5 size ratios must have three dimensions")
    if value["storage"] != "dense_fp32_header16":
        raise SchemaError("unsupported local-surface storage")
    num_units = _require_positive_int(value["numUnits"], "local-surface numUnits")
    if not isinstance(value["unitIds"], list) or len(value["unitIds"]) != num_units:
        raise SchemaError("local-surface unitIds length does not match numUnits")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value["unitIds"]):
        raise SchemaError("local-surface unitIds must be non-negative integers")
    if value["unitIds"] != sorted(value["unitIds"]):
        raise SchemaError("local-surface unitIds must be sorted")
    if len(set(value["unitIds"])) != num_units:
        raise SchemaError("local-surface unitIds must be unique")
    _require_nonnegative_int(value["samplingSeed"], "local-surface samplingSeed")
    if value["transformPolicy"] != "world_triangle_area_inverse_transpose_normal":
        raise SchemaError("unsupported local-surface transform policy")
    if not isinstance(value["degenerateUnitIds"], list):
        raise SchemaError("degenerateUnitIds must be a list")
    if any(item not in value["unitIds"] for item in value["degenerateUnitIds"]):
        raise SchemaError("degenerateUnitIds must be a subset of unitIds")
    files = _require_mapping(value["files"], "local-surface files")
    _require_exact_keys(
        files,
        {"points", "sizeRatios", "aabbMin", "aabbMax", "unitIds"},
        name="local-surface files",
    )
    return dict(value)


def validate_region_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(manifest, "region manifest")
    _require_exact_keys(
        value,
        {"schema", "version", "regionType", "supportCount", "center", "supportPoints", "geometry"},
        name="region manifest",
    )
    if value["schema"] != REGION_SUPPORT_SCHEMA or value["version"] != 1:
        raise SchemaError("unsupported region-support schema")
    if value["regionType"] not in {"disk", "oriented_box"}:
        raise SchemaError("regionType must be disk or oriented_box")
    if value["supportCount"] != SUPPORT_POINT_COUNT:
        raise SchemaError("V5 region queries must contain exactly nine support points")
    for key in ("center", "supportPoints"):
        if not isinstance(value[key], list):
            raise SchemaError(f"region {key} must be a JSON list")
    if len(value["center"]) != 3:
        raise SchemaError("region center must have three coordinates")
    if len(value["supportPoints"]) != SUPPORT_POINT_COUNT or any(
        not isinstance(point, list) or len(point) != 3 for point in value["supportPoints"]
    ):
        raise SchemaError("region supportPoints must have shape [9,3]")
    geometry = _require_mapping(value["geometry"], "region geometry")
    if value["regionType"] == "disk":
        _require_exact_keys(geometry, {"radius", "right", "forward"}, name="disk geometry")
        if len(geometry["right"]) != 3 or len(geometry["forward"]) != 3:
            raise SchemaError("disk basis vectors must have three coordinates")
        if not isinstance(geometry["radius"], (int, float)) or geometry["radius"] <= 0:
            raise SchemaError("disk radius must be positive")
    else:
        _require_exact_keys(geometry, {"halfAxes"}, name="oriented-box geometry")
        if (
            not isinstance(geometry["halfAxes"], list)
            or len(geometry["halfAxes"]) != 3
            or any(not isinstance(axis, list) or len(axis) != 3 for axis in geometry["halfAxes"])
        ):
            raise SchemaError("oriented-box halfAxes must have shape [3,3]")
    return dict(value)


def validate_probe_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(manifest, "probe manifest")
    _require_exact_keys(
        value,
        {
            "schema",
            "version",
            "assetKind",
            "sceneId",
            "split",
            "sourceRole",
            "containsVisibilityLabels",
            "raySchema",
            "surfaceStartsPerUnit",
            "directionsPerUnit",
            "directionSet",
            "distanceRatios",
            "maxTraceDistanceRatio",
            "recordFields",
            "numUnits",
            "recordsFile",
        },
        optional={
            "permissions",
            "storage",
            "rowCount",
            "rowLayout",
            "eventEncoding",
            "hitDistanceOrigin",
            "rayOriginOffset",
            "unitRadius",
            "unitOrder",
            "sceneNumUnits",
            "unitIds",
            "columnDtypes",
            "columnShapes",
            "byteOrder",
            "files",
            "materialRule",
            "acceleration",
            "shard",
        },
        name="probe manifest",
    )
    if value["schema"] != EXTERNAL_HIT_PROBE_SCHEMA or value["version"] != 1:
        raise SchemaError("unsupported external-hit probe schema")
    if value["assetKind"] != "external_hit_probe":
        raise SchemaError("probe assetKind must be external_hit_probe")
    _require_string(value["sceneId"], "probe sceneId")
    if value["split"] != "train" or value["sourceRole"] != "source_train":
        raise SchemaError("external-hit probes are source-train-only assets")
    if value["containsVisibilityLabels"] is not False:
        raise SchemaError("external-hit probes cannot contain visibility labels")
    if value["raySchema"] != "surface_origin_first_external_hit_right_censor_v1":
        raise SchemaError("unsupported external-hit ray schema")
    if value["surfaceStartsPerUnit"] != 16 or value["directionsPerUnit"] != 36:
        raise SchemaError("V5 probes require 16 starts and 36 fixed directions per unit")
    if value["directionSet"] != "icosahedron12_plus_fibonacci24":
        raise SchemaError("unsupported V5 probe direction set")
    ratios = value["distanceRatios"]
    if ratios != [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
        raise SchemaError("probe distance ratios do not match the frozen V5 grid")
    if value["maxTraceDistanceRatio"] != 1024.0:
        raise SchemaError("probe maxTraceDistanceRatio must be 1024")
    if value["recordFields"] != [
        "sceneId",
        "unitId",
        "probeId",
        "startPointWorld",
        "directionWorld",
        "hitDistance",
        "maxTraceDistance",
        "event",
    ]:
        raise SchemaError("probe record field order is not the V5 contract")
    _require_positive_int(value["numUnits"], "probe numUnits")
    _require_string(value["recordsFile"], "probe recordsFile")
    if "files" in value:
        files = _require_mapping(value["files"], "probe files")
        expected_files = {
            "unitIds",
            "directions",
            "hitDistances",
            "maxDistances",
            "startIds",
            "directionIds",
        }
        _require_exact_keys(files, expected_files, name="probe files")
        for key in expected_files:
            _require_string(files[key], f"probe files.{key}")
        if value.get("storage") != "columnar_memmap_little_endian_v1":
            raise SchemaError("columnar probe storage must be little-endian memmap v1")
        row_count = value.get("rowCount")
        if row_count != value["numUnits"] * 16 * 36:
            raise SchemaError("columnar probe rowCount must equal numUnits*16*36")
        if value.get("rowLayout") != "[unit][directionId][startId]":
            raise SchemaError("unsupported columnar probe row layout")
        if value.get("eventEncoding") != "finite_hit_distance_is_event":
            raise SchemaError("unsupported columnar probe event encoding")
        if value.get("byteOrder") != "little-endian":
            raise SchemaError("columnar probe byte order must be little-endian")
        dtypes = _require_mapping(value.get("columnDtypes"), "probe columnDtypes")
        _require_exact_keys(
            dtypes,
            expected_files,
            name="probe columnDtypes",
        )
        expected_dtypes = {
            "unitIds": "uint32",
            "directions": "float32[3]",
            "hitDistances": "float32",
            "maxDistances": "float32",
            "startIds": "uint8",
            "directionIds": "uint8",
        }
        if dict(dtypes) != expected_dtypes:
            raise SchemaError("columnar probe dtypes do not match the fixed ray columns")
        shapes = _require_mapping(value.get("columnShapes"), "probe columnShapes")
        _require_exact_keys(shapes, expected_files, name="probe columnShapes")
        expected_shapes = {
            "unitIds": [row_count],
            "directions": [row_count, 3],
            "hitDistances": [row_count],
            "maxDistances": [row_count],
            "startIds": [row_count],
            "directionIds": [row_count],
        }
        if dict(shapes) != expected_shapes:
            raise SchemaError("columnar probe shapes do not match rowCount")
        permissions = _require_mapping(value.get("permissions"), "probe permissions")
        if permissions.get("assetKind") != "external_hit_probe":
            raise SchemaError("probe permissions must identify external_hit_probe")
        if permissions.get("access") != "source_train_only":
            raise SchemaError("probe permissions must be source_train_only")
        if permissions.get("heldOutReadable") is not False:
            raise SchemaError("held-out external-hit probes must remain unreadable")
        if permissions.get("visibilityLabelsIncluded") is not False:
            raise SchemaError("external-hit probes must not contain visibility labels")
        unit_ids = value.get("unitIds")
        if not isinstance(unit_ids, list) or len(unit_ids) != value["numUnits"]:
            raise SchemaError("columnar probe unitIds must list the stored unit order")
        if any(isinstance(unit_id, bool) or not isinstance(unit_id, int) or unit_id < 0 for unit_id in unit_ids):
            raise SchemaError("columnar probe unitIds must be non-negative integers")
        if unit_ids != sorted(unit_ids) or len(set(unit_ids)) != len(unit_ids):
            raise SchemaError("columnar probe unitIds must be sorted and unique")
        scene_num_units = value.get("sceneNumUnits")
        if scene_num_units is not None and (
            isinstance(scene_num_units, bool)
            or not isinstance(scene_num_units, int)
            or scene_num_units < value["numUnits"]
        ):
            raise SchemaError("columnar probe sceneNumUnits is invalid")
        if scene_num_units is not None and any(unit_id >= scene_num_units for unit_id in unit_ids):
            raise SchemaError("columnar probe unitId exceeds sceneNumUnits")
        if "shard" in value:
            shard = _require_mapping(value["shard"], "probe shard")
            _require_exact_keys(shard, {"index", "count", "selectedUnitIds", "complete"}, name="probe shard")
            if (
                isinstance(shard["index"], bool)
                or not isinstance(shard["index"], int)
                or shard["index"] < 0
                or isinstance(shard["count"], bool)
                or not isinstance(shard["count"], int)
                or shard["count"] <= shard["index"]
                or shard["selectedUnitIds"] != unit_ids
                or shard["complete"] is not True
            ):
                raise SchemaError("probe shard metadata is invalid")
    return dict(value)


def validate_synthetic_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _require_mapping(manifest, "synthetic scene manifest")
    _require_exact_keys(
        value,
        {
            "schema",
            "version",
            "method",
            "baseSeed",
            "generator",
            "sceneCount",
            "scenesPerFamily",
            "structureFamilies",
            "splitCounts",
            "splitProtocol",
            "scenes",
        },
        name="synthetic scene manifest",
    )
    if value["schema"] != SYNTHETIC_SCENE_SCHEMA or value["version"] != 1:
        raise SchemaError("unsupported synthetic scene manifest schema")
    if value["method"] != "GCOF-PVS-V5":
        raise SchemaError("synthetic manifest method must be GCOF-PVS-V5")
    _require_nonnegative_int(value["baseSeed"], "synthetic baseSeed")
    if value["generator"] != "procedural_primitive_mesh_bank_v5":
        raise SchemaError("unsupported synthetic generator")
    if value["sceneCount"] != 120 or value["scenesPerFamily"] != 24:
        raise SchemaError("V5 synthetic catalog must contain 120 scenes, 24 per family")
    families = _require_string_list(value["structureFamilies"], "structureFamilies")
    if len(families) != 5 or len(set(families)) != 5:
        raise SchemaError("V5 synthetic catalog must contain five structure families")
    if value["splitCounts"] != {"train": 96, "validation": 12, "diagnostic": 12}:
        raise SchemaError("synthetic split counts must be 96/12/12")
    protocol = _require_mapping(value["splitProtocol"], "synthetic splitProtocol")
    _require_exact_keys(
        protocol,
        {"strategy", "key", "poseIndependent", "seedDisjoint", "assignmentOrder"},
        name="synthetic splitProtocol",
    )
    if protocol["strategy"] != "sorted_generator_seed_mod_10" or protocol["key"] != "ascending generatorSeed":
        raise SchemaError("unsupported synthetic split strategy")
    if protocol["poseIndependent"] is not True or protocol["seedDisjoint"] is not True:
        raise SchemaError("synthetic split must be pose-independent and seed-disjoint")
    if protocol["assignmentOrder"] != ["train", "validation", "diagnostic"]:
        raise SchemaError("unexpected synthetic split assignment order")
    scenes = value["scenes"]
    if not isinstance(scenes, list) or len(scenes) != 120:
        raise SchemaError("synthetic scenes must contain exactly 120 entries")
    required_scene_keys = {
        "sceneId",
        "structureFamily",
        "localSeed",
        "generatorSeed",
        "split",
        "unitCount",
        "streamingGranularityKiB",
        "randomization",
        "geometryRecipe",
    }
    ids: set[str] = set()
    seeds: set[int] = set()
    family_counts = {family: 0 for family in families}
    split_counts = {"train": 0, "validation": 0, "diagnostic": 0}
    for index, scene in enumerate(scenes):
        scene_value = _require_mapping(scene, f"synthetic scenes[{index}]")
        _require_exact_keys(scene_value, required_scene_keys, name=f"synthetic scenes[{index}]")
        scene_id = _require_string(scene_value["sceneId"], f"synthetic scenes[{index}].sceneId")
        if scene_id in ids:
            raise SchemaError("synthetic scene IDs must be unique")
        ids.add(scene_id)
        family = _require_string(scene_value["structureFamily"], f"synthetic scenes[{index}].structureFamily")
        if family not in family_counts:
            raise SchemaError(f"unknown synthetic structure family: {family}")
        family_counts[family] += 1
        local_seed = _require_nonnegative_int(scene_value["localSeed"], f"synthetic scenes[{index}].localSeed")
        generator_seed = _require_nonnegative_int(
            scene_value["generatorSeed"], f"synthetic scenes[{index}].generatorSeed"
        )
        if generator_seed in seeds:
            raise SchemaError("synthetic generator seeds must be unique")
        seeds.add(generator_seed)
        split = _require_string(scene_value["split"], f"synthetic scenes[{index}].split")
        if split not in split_counts:
            raise SchemaError("synthetic scene split must be train/validation/diagnostic")
        split_counts[split] += 1
        unit_count = scene_value["unitCount"]
        if isinstance(unit_count, bool) or not isinstance(unit_count, int) or not 256 <= unit_count <= 4096:
            raise SchemaError("synthetic unitCount must be between 256 and 4096")
        if scene_value["streamingGranularityKiB"] not in {32, 64, 128}:
            raise SchemaError("synthetic streaming granularity must be 32/64/128 KiB")
        if not isinstance(scene_value["randomization"], Mapping) or not isinstance(
            scene_value["geometryRecipe"], Mapping
        ):
            raise SchemaError("synthetic scene randomization and geometryRecipe must be objects")
    if family_counts != {family: 24 for family in families}:
        raise SchemaError(f"unexpected synthetic family counts: {family_counts}")
    if split_counts != {"train": 96, "validation": 12, "diagnostic": 12}:
        raise SchemaError(f"unexpected synthetic split counts: {split_counts}")
    return dict(value)


def write_json_manifest(path: str | Path, manifest: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_json_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON manifest: {source}") from exc
    if not isinstance(value, dict):
        raise SchemaError(f"manifest must be a JSON object: {source}")
    return value
