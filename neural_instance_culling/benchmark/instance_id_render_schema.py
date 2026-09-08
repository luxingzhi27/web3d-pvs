#!/usr/bin/env python3
"""Shared validation for the instance Color-ID render manifest.

This module deliberately inspects only the JSON chunk of each GLB.  It does
not decode geometry or start a browser.  The purpose is to make the mapping
from ``componentGlobalId`` to an instanced-mesh slot explicit and auditable
before an expensive image render is attempted.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any


INSTANCE_RENDER_MANIFEST_SCHEMA = "local-true-component-id-render-manifest-v2"
FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA = "local-true-component-id-formal-render-manifest-v1"
INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA = "local-true-component-id-render-batch-manifest-v1"
INSTANCE_BINDING_SCHEMA = "component-instance-binding-preflight-v1"
INSTANCE_ID_ENCODING = "componentGlobalId + 1, RGB24, 0 background"
RENDER_FOV_Y_DEG = 60.0
MODEL_INPUT_FOV_Y_DEG = 66.0

_GLB_MAGIC = 0x46546C67
_GLB_VERSION = 2
_JSON_CHUNK_TYPE = 0x4E4F534A


class InstanceBindingError(ValueError):
    """Raised when runtime metadata and a GLB cannot be joined safely."""


def _read_glb_json(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 20:
        raise InstanceBindingError(f"GLB is too small: {path}")
    magic, version, total_length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC or version != _GLB_VERSION:
        raise InstanceBindingError(f"unsupported GLB header: {path}")
    if total_length != len(data):
        raise InstanceBindingError(f"GLB length mismatch: {path}")

    offset = 12
    document: dict[str, Any] | None = None
    while offset + 8 <= len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        start = offset + 8
        end = start + chunk_length
        if end > len(data):
            raise InstanceBindingError(f"truncated GLB chunk: {path}")
        if chunk_type == _JSON_CHUNK_TYPE:
            try:
                document = json.loads(data[start:end].decode("utf-8").rstrip(" "))
            except json.JSONDecodeError as error:
                raise InstanceBindingError(f"invalid GLB JSON: {path}: {error}") from error
        offset = end
    if document is None:
        raise InstanceBindingError(f"GLB JSON chunk is missing: {path}")
    return document


def inspect_glb_instance_layout(path: Path) -> dict[str, Any]:
    """Return the renderable mesh/instance layout without decoding geometry."""
    document = _read_glb_json(path)
    accessors = document.get("accessors") or []
    mesh_nodes: list[dict[str, Any]] = []
    for node_index, node in enumerate(document.get("nodes") or []):
        if node.get("mesh") is None:
            continue
        extension = (node.get("extensions") or {}).get("EXT_mesh_gpu_instancing")
        instance_count: int | None = None
        if extension is not None:
            attributes = extension.get("attributes") or {}
            translation_accessor = attributes.get("TRANSLATION")
            if translation_accessor is None:
                raise InstanceBindingError(
                    f"EXT_mesh_gpu_instancing has no TRANSLATION accessor: {path}"
                )
            try:
                instance_count = int(accessors[int(translation_accessor)]["count"])
            except (IndexError, KeyError, TypeError, ValueError) as error:
                raise InstanceBindingError(
                    f"invalid instancing accessor in {path}: {translation_accessor}"
                ) from error
            if instance_count < 1:
                raise InstanceBindingError(f"empty instanced mesh is not renderable: {path}")
        mesh_nodes.append(
            {
                "nodeIndex": int(node_index),
                "meshIndex": int(node["mesh"]),
                "instanced": extension is not None,
                "instanceCount": instance_count if instance_count is not None else 1,
            }
        )

    if not mesh_nodes:
        return {
            "renderable": False,
            "meshNodeCount": 0,
            "instanced": False,
            "instanceCount": 0,
            "meshNodes": [],
        }
    if len(mesh_nodes) != 1:
        raise InstanceBindingError(
            f"component binding is ambiguous: expected one mesh node, found {len(mesh_nodes)} in {path}"
        )
    node = mesh_nodes[0]
    return {
        "renderable": True,
        "meshNodeCount": 1,
        "instanced": bool(node["instanced"]),
        "instanceCount": int(node["instanceCount"]),
        "meshNodes": mesh_nodes,
    }


def _as_nonnegative_int(value: Any, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise InstanceBindingError(f"{label} is not an integer: {value!r}") from error
    if result < 0:
        raise InstanceBindingError(f"{label} must be non-negative: {result}")
    return result


def build_instance_binding_preflight(
    runtime_meta: dict[str, Any],
    glb_index_path: Path,
    glb_root: Path,
) -> dict[str, Any]:
    """Validate and materialize the component-to-instance binding table.

    ``globalGlbRecords[].componentGlobalIds`` is the current producer's
    ordering contract: list position is the GLB instance slot.  The contract
    is checked against the GLB's instancing accessor count here.  Empty GLBs
    are retained as explicit non-renderable anomalies rather than silently
    being treated as valid image geometry.
    """
    index = json.loads(glb_index_path.read_text(encoding="utf-8"))
    entries = index.get("entries") or []
    glb_records = {
        _as_nonnegative_int(row.get("globalGlbId"), "globalGlbId"): row
        for row in (runtime_meta.get("globalGlbRecords") or [])
    }
    component_records = runtime_meta.get("componentRecords") or []
    component_to_glb: dict[int, int] = {}
    for row in component_records:
        component_id = _as_nonnegative_int(row.get("componentGlobalId"), "componentGlobalId")
        global_glb_id = _as_nonnegative_int(row.get("globalGlbId"), "globalGlbId")
        if component_id in component_to_glb:
            raise InstanceBindingError(f"duplicate componentGlobalId: {component_id}")
        component_to_glb[component_id] = global_glb_id

    by_glb: dict[str, dict[str, Any]] = {}
    component_to_binding: dict[str, dict[str, Any]] = {}
    empty_glbs: list[dict[str, Any]] = []
    seen_components: set[int] = set()
    seen_glbs: set[int] = set()

    root = glb_root.resolve()
    for entry in entries:
        global_glb_id = _as_nonnegative_int(entry.get("globalId"), "glbIndex.globalId")
        if global_glb_id in seen_glbs:
            raise InstanceBindingError(f"duplicate GLB globalId: {global_glb_id}")
        seen_glbs.add(global_glb_id)
        record = glb_records.get(global_glb_id)
        if record is None:
            raise InstanceBindingError(
                f"runtimeVisibilityMeta has no globalGlbRecord for GLB {global_glb_id}"
            )
        component_ids = [
            _as_nonnegative_int(value, f"componentGlobalIds[{global_glb_id}]")
            for value in (record.get("componentGlobalIds") or [])
        ]
        if len(component_ids) != len(set(component_ids)):
            raise InstanceBindingError(f"duplicate component ID inside GLB {global_glb_id}")
        relative_path = str(entry.get("path") or "")
        candidate = (glb_root / relative_path).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise InstanceBindingError(f"GLB path escapes root: {relative_path}") from error
        if not candidate.exists():
            raise InstanceBindingError(f"local GLB is missing: {candidate}")
        layout = inspect_glb_instance_layout(candidate)
        renderable = bool(layout["renderable"])
        if renderable and int(layout["instanceCount"]) != len(component_ids):
            raise InstanceBindingError(
                "instance count mismatch for GLB "
                f"{global_glb_id}: metadata={len(component_ids)} glb={layout['instanceCount']}"
            )
        if not renderable and component_ids:
            empty_glbs.append(
                {
                    "globalGlbId": global_glb_id,
                    "path": relative_path,
                    "componentGlobalIds": component_ids,
                    "reason": "GLB has no mesh node",
                }
            )

        binding = {
            "globalGlbId": global_glb_id,
            "path": relative_path,
            "componentGlobalIds": component_ids,
            "instanceCount": int(layout["instanceCount"]),
            "instanced": bool(layout["instanced"]),
            "renderable": renderable,
            "meshNodeCount": int(layout["meshNodeCount"]),
            "meshNodes": layout["meshNodes"],
        }
        by_glb[str(global_glb_id)] = binding
        for instance_index, component_id in enumerate(component_ids):
            expected_glb = component_to_glb.get(component_id)
            if expected_glb != global_glb_id:
                raise InstanceBindingError(
                    f"component {component_id} maps to GLB {expected_glb}, "
                    f"but appears in GLB {global_glb_id}"
                )
            if component_id in seen_components:
                raise InstanceBindingError(f"component appears in multiple GLBs: {component_id}")
            seen_components.add(component_id)
            component_to_binding[str(component_id)] = {
                "componentGlobalId": component_id,
                "globalGlbId": global_glb_id,
                "instanceIndex": int(instance_index),
                "renderable": renderable,
            }

    missing_components = sorted(set(component_to_glb) - seen_components)
    if missing_components:
        raise InstanceBindingError(
            f"component records missing from globalGlbRecords: {missing_components[:16]}"
        )
    unindexed_glbs = sorted(set(glb_records) - seen_glbs)
    if unindexed_glbs:
        raise InstanceBindingError(
            f"globalGlbRecords missing from glbIndex: {unindexed_glbs[:16]}"
        )

    return {
        "schema": INSTANCE_BINDING_SCHEMA,
        "idEncoding": INSTANCE_ID_ENCODING,
        "instanceOrderSemantics": "globalGlbRecords[globalGlbId].componentGlobalIds[instanceIndex]",
        "globalGlbCount": len(by_glb),
        "componentCount": len(component_to_glb),
        "renderableComponentCount": sum(1 for row in component_to_binding.values() if row["renderable"]),
        "emptyGlbCount": len(empty_glbs),
        "emptyGlbs": empty_glbs,
        "byGlobalGlbId": by_glb,
        "componentToBinding": component_to_binding,
    }


def validate_instance_render_manifest(manifest: dict[str, Any]) -> None:
    """Validate the browser-facing v2 manifest without touching assets."""
    if manifest.get("schema") != INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise InstanceBindingError(
            "refusing non-instance render manifest; expected "
            f"{INSTANCE_RENDER_MANIFEST_SCHEMA}, got {manifest.get('schema')!r}"
        )
    if manifest.get("idEncoding") != INSTANCE_ID_ENCODING:
        raise InstanceBindingError("manifest must encode componentGlobalId, not globalGlbId")
    try:
        fov = float(manifest.get("renderFovYDeg"))
    except (TypeError, ValueError) as error:
        raise InstanceBindingError("manifest renderFovYDeg is missing") from error
    if abs(fov - RENDER_FOV_Y_DEG) > 1e-6:
        raise InstanceBindingError(f"render FOV must be exactly {RENDER_FOV_Y_DEG}, got {fov}")
    if manifest.get("formalImageEvaluationReady") is not False:
        raise InstanceBindingError("component renderer manifest must be marked not formal-ready")
    try:
        model_fov = float(manifest.get("modelInputFovYDeg"))
    except (TypeError, ValueError) as error:
        raise InstanceBindingError("manifest modelInputFovYDeg is missing") from error
    if abs(model_fov - MODEL_INPUT_FOV_Y_DEG) > 1e-6:
        raise InstanceBindingError(
            f"model-input FOV must be exactly {MODEL_INPUT_FOV_Y_DEG}, got {model_fov}"
        )
    selected_glbs = manifest.get("selectedGlbs")
    if not isinstance(selected_glbs, list) or any(
        not isinstance(value, int) or value < 0 for value in selected_glbs
    ):
        raise InstanceBindingError("manifest.selectedGlbs must be a list of non-negative integers")
    if not isinstance(manifest.get("instanceBindings"), dict):
        raise InstanceBindingError("manifest.instanceBindings is missing")
    bindings = manifest["instanceBindings"]
    if bindings.get("schema") != INSTANCE_BINDING_SCHEMA:
        raise InstanceBindingError("manifest.instanceBindings has an unknown schema")
    by_glb = bindings.get("byGlobalGlbId")
    component_to_binding = bindings.get("componentToBinding")
    if not isinstance(by_glb, dict) or not isinstance(component_to_binding, dict):
        raise InstanceBindingError("manifest.instanceBindings is missing binding tables")
    selected_set = set(selected_glbs)
    binding_glb_set = {int(value) for value in by_glb}
    if selected_set != binding_glb_set:
        raise InstanceBindingError(
            "selectedGlbs must be the complete GLB inventory; it cannot be a predicted GLB subset"
        )
    for component_key, binding in component_to_binding.items():
        try:
            component_id = int(component_key)
            global_glb_id = int(binding["globalGlbId"])
            instance_index = int(binding["instanceIndex"])
        except (KeyError, TypeError, ValueError) as error:
            raise InstanceBindingError(f"invalid component binding: {component_key!r}") from error
        if component_id < 0 or global_glb_id < 0 or instance_index < 0:
            raise InstanceBindingError(f"invalid component binding: {component_key!r}")
        glb_binding = by_glb.get(str(global_glb_id))
        if glb_binding is None:
            raise InstanceBindingError(f"component {component_id} references unknown GLB {global_glb_id}")
        component_ids = glb_binding.get("componentGlobalIds") or []
        if instance_index >= len(component_ids) or int(component_ids[instance_index]) != component_id:
            raise InstanceBindingError(
                f"component {component_id} does not match GLB {global_glb_id} instance slot {instance_index}"
            )
    reference = manifest.get("reference") or {}
    if reference.get("mode") != "full_scene_renderable_instances" or reference.get("idSource") != "componentGlobalId":
        raise InstanceBindingError("manifest reference must be a full component-ID scene render")
    prediction = manifest.get("prediction") or {}
    if prediction.get("field") != "predictionComponentIds":
        raise InstanceBindingError("manifest prediction field must be predictionComponentIds")
    synthetic_smoke = manifest.get("syntheticComponentIdSmoke")
    if synthetic_smoke is not None:
        for field in ("referenceComponentIds", "predictionComponentIds"):
            values = synthetic_smoke.get(field) if isinstance(synthetic_smoke, dict) else None
            if not isinstance(values, list):
                raise InstanceBindingError(f"synthetic smoke is missing {field}")
            for component_id in values:
                if not isinstance(component_id, int) or component_id < 0:
                    raise InstanceBindingError(
                        f"synthetic smoke has an invalid componentGlobalId: {component_id!r}"
                    )
                if str(component_id) not in component_to_binding:
                    raise InstanceBindingError(
                        f"synthetic smoke contains unknown componentGlobalId {component_id}"
                    )
    for sample_index, sample in enumerate(manifest.get("samples") or []):
        if any(key in sample for key in ("referenceGlbs", "testGlbs", "predictionGlbIds")):
            raise InstanceBindingError(
                f"sample {sample_index} still uses GLB-level image IDs; use component IDs only"
            )
        if not isinstance(sample.get("predictionComponentIds"), list):
            raise InstanceBindingError(
                f"sample {sample_index} is missing predictionComponentIds"
            )
        if abs(float(sample.get("renderFovYDeg", fov)) - RENDER_FOV_Y_DEG) > 1e-6:
            raise InstanceBindingError(f"sample {sample_index} does not use 60 degree rendering")
        if abs(float(sample.get("modelInputFovYDeg", model_fov)) - MODEL_INPUT_FOV_Y_DEG) > 1e-6:
            raise InstanceBindingError(f"sample {sample_index} does not use 66 degree model input")
        for component_id in sample["predictionComponentIds"]:
            if not isinstance(component_id, int) or component_id < 0:
                raise InstanceBindingError(
                    f"sample {sample_index} has an invalid prediction component ID: {component_id!r}"
                )
            if str(component_id) not in component_to_binding:
                raise InstanceBindingError(
                    f"sample {sample_index} contains unknown componentGlobalId {component_id}"
                )


def validate_instance_render_batch_manifest(manifest: dict[str, Any]) -> None:
    """Validate several v2 prediction batches for one browser page.

    A batch manifest does not introduce a new image or instance semantic.  It
    only groups ordinary v2 samples so the browser can load the complete local
    GLB inventory once and then render every batch in sequence.  The flattened
    copy below deliberately goes through the v2 validator so the FOV, ID
    encoding, full-scene reference, and component-level prediction checks stay
    identical.
    """
    if manifest.get("schema") != INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA:
        raise InstanceBindingError(
            "refusing non-batch instance render manifest; expected "
            f"{INSTANCE_RENDER_BATCH_MANIFEST_SCHEMA}, got {manifest.get('schema')!r}"
        )
    batches = manifest.get("batches")
    if not isinstance(batches, list) or not batches:
        raise InstanceBindingError("batch manifest must contain a non-empty batches list")

    flattened: list[dict[str, Any]] = []
    seen_batch_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for batch_index, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise InstanceBindingError(f"batch {batch_index} must be an object")
        batch_id = str(batch.get("batchId") or "")
        if not batch_id:
            raise InstanceBindingError(f"batch {batch_index} is missing batchId")
        if batch_id in seen_batch_ids:
            raise InstanceBindingError(f"duplicate batchId: {batch_id}")
        seen_batch_ids.add(batch_id)
        samples = batch.get("samples")
        if not isinstance(samples, list) or not samples:
            raise InstanceBindingError(f"batch {batch_id} must contain a non-empty samples list")
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise InstanceBindingError(f"batch {batch_id} sample {sample_index} must be an object")
            sample_id = str(sample.get("sampleId") or "")
            if not sample_id:
                raise InstanceBindingError(
                    f"batch {batch_id} sample {sample_index} is missing sampleId"
                )
            if sample_id in seen_sample_ids:
                raise InstanceBindingError(f"duplicate sampleId across batches: {sample_id}")
            seen_sample_ids.add(sample_id)
            flattened.append(sample)

    legacy_manifest = dict(manifest)
    legacy_manifest["schema"] = INSTANCE_RENDER_MANIFEST_SCHEMA
    legacy_manifest["samples"] = flattened
    validate_instance_render_manifest(legacy_manifest)


def validate_formal_instance_render_manifest(manifest: dict[str, Any]) -> None:
    """Validate the stricter real-mesh, hardware-GPU image protocol.

    The legacy validator remains intentionally non-formal for historical
    smoke manifests.  This validator reuses its component/GLB consistency
    checks on a private legacy-shaped copy, then enforces the additional
    evidence that a formal image result cannot be a synthetic render, a GLB
    coarse mask, a partial inventory, or a pending browser implementation.
    """
    if manifest.get("schema") != FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise InstanceBindingError(
            "refusing non-formal image manifest; expected "
            f"{FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA}"
        )
    if manifest.get("formalImageEvaluationReady") is not True:
        raise InstanceBindingError("formal image manifest must set formalImageEvaluationReady=true")
    if manifest.get("syntheticComponentIdSmoke") is not None:
        raise InstanceBindingError("synthetic component-ID smoke cannot be formal")

    legacy = dict(manifest)
    legacy["schema"] = INSTANCE_RENDER_MANIFEST_SCHEMA
    legacy["formalImageEvaluationReady"] = False
    validate_instance_render_manifest(legacy)

    requirements = manifest.get("formalRequirements") or {}
    if requirements.get("requiresHardwareWebGL") is not True:
        raise InstanceBindingError("formal image manifest must require hardware WebGL")
    if requirements.get("syntheticSmokeAllowed") is not False:
        raise InstanceBindingError("formal image manifest must disallow synthetic smoke")
    if requirements.get("completeGlbInventory") is not True:
        raise InstanceBindingError("formal image manifest must retain the complete GLB inventory")

    reference = manifest.get("reference") or {}
    if reference.get("geometrySource") != "original_local_glb_meshes":
        raise InstanceBindingError("formal reference must use original local GLB meshes")
    if reference.get("completeInventory") is not True:
        raise InstanceBindingError("formal reference must declare a complete inventory")

    prediction = manifest.get("prediction") or {}
    if prediction.get("postFilter") != "component_visibility_mask_after_conservative_render_submission":
        raise InstanceBindingError("formal prediction must use an instance-level visibility mask")
    spatial = manifest.get("spatialCulling") or {}
    if spatial.get("completeInventoryRetained") is not True or spatial.get("componentLevelMask") is not True:
        raise InstanceBindingError("formal spatial submission must retain complete inventory and component masks")

    samples = manifest.get("samples") or []
    if not samples:
        raise InstanceBindingError("formal image evaluation requires at least one real sample")
    for index, sample in enumerate(samples):
        if sample.get("referenceMode") != "full_scene_renderable_instances":
            raise InstanceBindingError(f"formal sample {index} is not a full-scene reference")
        if sample.get("predictionComponentIds") is None:
            raise InstanceBindingError(f"formal sample {index} has no component prediction list")
