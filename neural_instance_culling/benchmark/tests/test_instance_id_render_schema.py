from __future__ import annotations

import copy
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = TESTS_DIR.parent
RENDERER = BENCHMARK_DIR / "render_local_glb_color_id_browser.mjs"
sys.path.insert(0, str(BENCHMARK_DIR))

from instance_id_render_schema import (
    INSTANCE_ID_ENCODING,
    INSTANCE_RENDER_MANIFEST_SCHEMA,
    MODEL_INPUT_FOV_Y_DEG,
    RENDER_FOV_Y_DEG,
    build_instance_binding_preflight,
    validate_instance_render_manifest,
)

def write_json_only_glb(path: Path, instance_count: int) -> None:
    positions = struct.pack(
        "<9f",
        -0.8, -0.8, 0.0,
        0.8, -0.8, 0.0,
        0.0, 0.8, 0.0,
    )
    indices = struct.pack("<3H", 0, 1, 2)
    translation_values = []
    for index in range(instance_count):
        x = -1.5 if instance_count > 1 and index == 0 else 1.5 if instance_count > 1 else 0.0
        translation_values.extend([x, 0.0, 0.0])
    translations = struct.pack("<%df" % len(translation_values), *translation_values)
    translation_offset = len(positions) + len(indices)
    translation_offset += (4 - translation_offset % 4) % 4
    binary = positions + indices + b"\0" * (translation_offset - len(positions) - len(indices)) + translations
    position_accessor = 0
    index_accessor = 1
    translation_accessor = 2
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["EXT_mesh_gpu_instancing"],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(positions), "target": 34962},
            {"buffer": 0, "byteOffset": len(positions), "byteLength": len(indices), "target": 34963},
            {"buffer": 0, "byteOffset": translation_offset, "byteLength": len(translations)},
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [-0.8, -0.8, 0.0],
                "max": [0.8, 0.8, 0.0],
            },
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
            {"bufferView": 2, "componentType": 5126, "count": instance_count, "type": "VEC3"},
        ],
        "meshes": [{"primitives": [{"attributes": {"POSITION": position_accessor}, "indices": index_accessor}]}],
        "nodes": [
            {
                "mesh": 0,
                "extensions": {
                    "EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": translation_accessor}}
                },
            }
        ],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    encoded = json.dumps(document, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary += b"\0" * ((4 - len(binary) % 4) % 4)
    total_length = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total_length)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def make_fixture(root: Path) -> tuple[dict, dict]:
    write_json_only_glb(root / "prototype_a.glb", 2)
    write_json_only_glb(root / "prototype_b.glb", 1)
    glb_index = {"schema": "synthetic-glb-index-v1", "entries": [
        {"globalId": 0, "path": "prototype_a.glb"},
        {"globalId": 1, "path": "prototype_b.glb"},
    ]}
    glb_index_path = root / "glbIndex.json"
    glb_index_path.write_text(json.dumps(glb_index), encoding="utf-8")
    runtime_meta = {
        "componentRecords": [
            {"componentGlobalId": 0, "globalGlbId": 0},
            {"componentGlobalId": 1, "globalGlbId": 0},
            {"componentGlobalId": 2, "globalGlbId": 1},
        ],
        "globalGlbRecords": [
            {"globalGlbId": 0, "componentGlobalIds": [0, 1]},
            {"globalGlbId": 1, "componentGlobalIds": [2]},
        ],
    }
    bindings = build_instance_binding_preflight(runtime_meta, glb_index_path, root)
    manifest = {
        "schema": INSTANCE_RENDER_MANIFEST_SCHEMA,
        "idEncoding": INSTANCE_ID_ENCODING,
        "width": 160,
        "height": 90,
        "renderFovYDeg": RENDER_FOV_Y_DEG,
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "glbRoot": str(root),
        "glbIndex": str(glb_index_path),
        "selectedGlbs": [0, 1],
        "reference": {"mode": "full_scene_renderable_instances", "idSource": "componentGlobalId"},
        "prediction": {"field": "predictionComponentIds", "postFilter": "active_camera_60deg_aabb_pending"},
        "instanceBindings": bindings,
        "samples": [
            {
                "sampleId": "synthetic-0",
                "cameraPosition": [0.0, 0.0, 6.0],
                "cameraForward": [0.0, 0.0, -1.0],
                "aspect": 16.0 / 9.0,
                "renderFovYDeg": RENDER_FOV_Y_DEG,
                "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
                "predictionComponentIds": [1, 2],
            }
        ],
        "syntheticComponentIdSmoke": {
            "referenceComponentIds": [0, 1],
            "predictionComponentIds": [0],
        },
        "previewSamples": 1,
        "saveIdBuffers": True,
        "formalImageEvaluationReady": False,
    }
    return runtime_meta, manifest


class InstanceIdRenderSchemaTests(unittest.TestCase):
    def test_preflight_binds_component_ids_to_instance_slots(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _runtime_meta, manifest = make_fixture(Path(temp_dir))
            bindings = manifest["instanceBindings"]
            self.assertEqual(bindings["componentToBinding"]["0"]["instanceIndex"], 0)
            self.assertEqual(bindings["componentToBinding"]["1"]["instanceIndex"], 1)
            self.assertEqual(bindings["componentToBinding"]["2"]["globalGlbId"], 1)
            validate_instance_render_manifest(manifest)

    def test_glb_level_fields_and_wrong_fov_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            _runtime_meta, manifest = make_fixture(Path(temp_dir))
            old_manifest = copy.deepcopy(manifest)
            old_manifest["samples"][0]["referenceGlbs"] = [0]
            with self.assertRaisesRegex(ValueError, "GLB-level"):
                validate_instance_render_manifest(old_manifest)

            wrong_fov = copy.deepcopy(manifest)
            wrong_fov["samples"][0]["renderFovYDeg"] = 66
            with self.assertRaisesRegex(ValueError, "60"):
                validate_instance_render_manifest(wrong_fov)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_node_validate_only_writes_non_formal_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _runtime_meta, manifest = make_fixture(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "render-output"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    "node",
                    str(RENDERER),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--validate-only",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output_dir / "render_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "schema_validated_not_rendered")
            self.assertFalse(summary["formalImageEvaluationReady"])
            self.assertIsNone(summary["imageMetrics"])

    @unittest.skipUnless(
        shutil.which("node")
        and shutil.which("google-chrome")
        and (BENCHMARK_DIR.parent.parent / "slm2viewer/node_modules/three/build/three.module.js").exists(),
        "Node, Chrome, or Three.js is not installed",
    )
    def test_node_synthetic_component_id_render_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _runtime_meta, manifest = make_fixture(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "render-output"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    "node",
                    str(RENDERER),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--chrome-exe",
                    shutil.which("google-chrome") or "google-chrome",
                    "--synthetic-render-smoke",
                    "--timeout-ms",
                    "60000",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output_dir / "render_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "rendered")
            self.assertFalse(summary["formalImageEvaluationReady"])
            self.assertGreater(summary["imageMetrics"]["missPixels"], 0)
            self.assertTrue((output_dir / "samples/synthetic_reference_component_id_u32.bin").exists())
            self.assertTrue((output_dir / "samples/synthetic_diff_mask_u8.bin").exists())

    @unittest.skipUnless(
        shutil.which("node")
        and shutil.which("google-chrome")
        and (BENCHMARK_DIR.parent.parent / "slm2viewer/node_modules/three/build/three.module.js").exists(),
        "Node, Chrome, or Three.js is not installed",
    )
    def test_node_component_instanced_glb_render_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _runtime_meta, manifest = make_fixture(root)
            manifest_path = root / "manifest.json"
            output_dir = root / "render-output"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            result = subprocess.run(
                [
                    "node",
                    str(RENDERER),
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                    "--chrome-exe",
                    shutil.which("google-chrome") or "google-chrome",
                    "--timeout-ms",
                    "60000",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((output_dir / "render_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["renderStatus"], "rendered_component_id_buffers")
            self.assertTrue(summary["componentIdShaderImplemented"])
            self.assertFalse(summary["formalImageEvaluationReady"])
            self.assertGreater(summary["imageMetrics"]["missPixels"], 0)
            self.assertTrue((output_dir / "samples/synthetic-0_reference_u32.bin").exists())


if __name__ == "__main__":
    unittest.main()
