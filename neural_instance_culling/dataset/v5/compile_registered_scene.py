#!/usr/bin/env python3
"""Compile registered real scenes into the canonical GCOF-PVS V5 assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from neural_instance_culling.config.pvs_v5_scene_registry import REPO_ROOT, load_registry
from neural_instance_culling.dataset.v5.proxy_relation_graph import (
    build_proxy_relation_graph,
    write_proxy_relation_graph,
)
from neural_instance_culling.model.common.runtime_meta import load_runtime_meta


SURFACE_GENERATOR = REPO_ROOT / "neural_instance_culling/dataset/generate_v5_instance_surface_samples.mjs"
PROBE_GENERATOR = REPO_ROOT / "neural_instance_culling/dataset/v5/generate_external_hit_probes.mjs"


def _scene_entry(registry: dict[str, Any], scene_id: str) -> dict[str, Any]:
    for scene in registry["scenes"]:
        if scene["id"] == scene_id:
            return scene
    raise ValueError(f"unregistered V5 scene: {scene_id}")


def _path(value: str) -> Path:
    return REPO_ROOT / value


def _prepare_stage(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite completed V5 stage: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(command, cwd=REPO_ROOT, stdout=stdout, stderr=stderr, check=True)


def compile_surface(scene: dict[str, Any], scene_root: Path, overwrite: bool) -> dict[str, Any]:
    output = scene_root / "surface"
    _prepare_stage(output, overwrite)
    command = [
        "node",
        str(SURFACE_GENERATOR),
        "--assets-dir",
        str(_path(scene["assetsDir"])),
        "--runtime-meta",
        str(_path(scene["runtimeMeta"])),
        "--glb-index",
        str(_path(scene["glbIndex"])),
        "--output-dir",
        str(output),
        "--progress-every",
        "50",
    ]
    started = time.monotonic()
    _run_logged(command, scene_root / "logs/surface_stdout.log", scene_root / "logs/surface_stderr.log")
    manifest = json.loads((output / "surface_manifest.json").read_text(encoding="utf-8"))
    return {
        "stage": "surface",
        "seconds": time.monotonic() - started,
        "numUnits": int(manifest["numUnits"]),
        "degenerateUnits": len(manifest["degenerateUnitIds"]),
    }


def compile_relation(scene: dict[str, Any], scene_root: Path, overwrite: bool) -> dict[str, Any]:
    output = scene_root / "relation"
    _prepare_stage(output, overwrite)
    world_aabbs, _mapping, _runtime = load_runtime_meta(_path(scene["runtimeMeta"]))
    aabbs = world_aabbs.reshape(-1, 2, 3)
    started = time.monotonic()
    graph = build_proxy_relation_graph(aabbs)
    write_proxy_relation_graph(graph, output)
    return {
        "stage": "relation",
        "seconds": time.monotonic() - started,
        "numUnits": int(graph.unit_ids.size),
        "edgeCount": int(graph.edge_count),
        "usesVisibilityLabels": False,
    }


def compile_probes(scene: dict[str, Any], scene_root: Path, overwrite: bool) -> dict[str, Any]:
    if not PROBE_GENERATOR.is_file():
        raise FileNotFoundError(f"V5 probe generator is not implemented: {PROBE_GENERATOR}")
    output = scene_root / "probes"
    _prepare_stage(output, overwrite)
    command = [
        "node",
        str(PROBE_GENERATOR),
        "--scene-id",
        str(scene["id"]),
        "--assets-dir",
        str(_path(scene["assetsDir"])),
        "--runtime-meta",
        str(_path(scene["runtimeMeta"])),
        "--glb-index",
        str(_path(scene["glbIndex"])),
        "--surface-manifest",
        str(scene_root / "surface/surface_manifest.json"),
        "--output-dir",
        str(output),
    ]
    started = time.monotonic()
    _run_logged(command, scene_root / "logs/probes_stdout.log", scene_root / "logs/probes_stderr.log")
    manifest = json.loads(
        (output / "external_hit_probe_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "stage": "probes",
        "seconds": time.monotonic() - started,
        "numUnits": int(manifest["numUnits"]),
        "rayCount": int(manifest["rowCount"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--stage", action="append", choices=("surface", "relation", "probes"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    registry = load_registry()
    scene_ids = args.scenes or [scene["id"] for scene in registry["scenes"]]
    stages = args.stage or ["surface", "relation", "probes"]
    root = _path(registry["compiledOutputRoot"])
    for scene_id in scene_ids:
        scene = _scene_entry(registry, scene_id)
        scene_root = root / scene_id
        results: list[dict[str, Any]] = []
        for stage in stages:
            if stage == "surface":
                results.append(compile_surface(scene, scene_root, args.overwrite))
            elif stage == "relation":
                results.append(compile_relation(scene, scene_root, args.overwrite))
            else:
                results.append(compile_probes(scene, scene_root, args.overwrite))
        summary = {
            "schema": "gcof-pvs-v5-scene-compilation-summary-v1",
            "sceneId": scene_id,
            "stages": results,
        }
        (scene_root / "compile_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
