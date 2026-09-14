#!/usr/bin/env python3
"""Orchestrate the formal Connected-SAH preprocessing chain.

The stages are intentionally independent.  ``convert`` creates the new
renderable-unit assets, ``geometry`` creates the fixed 96D table, ``color-id``
samples the already registered V2 plan, and the remaining stages build the
Pose CSR, train-only triangle-depth relation evidence, and K=8 relation CSR.
No stage is run on import, and every subprocess gets explicit stdout/stderr
logs under the scene's Connected-SAH output root.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

try:
    from .instance_id_render_schema import build_instance_binding_preflight
    from .standard_graphics_connected_sah_config import (
        DEPTH_HEIGHT,
        DEPTH_MAX_LAYERS,
        DEPTH_WIDTH,
        EXPERIMENT,
        FRONTEND_RENDER_FOV_Y_DEG,
        MODEL_FOV_Y_DEG,
        POINTS_PER_GLB,
        RELATION_K,
        REPRESENTATIVE_SUBPOSES_PER_VIEWCELL,
        ROOT,
        SCENES,
        SEEDS,
        SUBPOSES_PER_VIEWCELL,
        TARGET_UNIT_KIB,
        VIEWCELL_RADIUS_M,
        VIEWCELL_SHAPE,
    )
except ImportError:  # direct ``python path/to/script.py`` invocation
    from instance_id_render_schema import build_instance_binding_preflight  # type: ignore
    from standard_graphics_connected_sah_config import (  # type: ignore
        DEPTH_HEIGHT,
        DEPTH_MAX_LAYERS,
        DEPTH_WIDTH,
        EXPERIMENT,
        FRONTEND_RENDER_FOV_Y_DEG,
        MODEL_FOV_Y_DEG,
        POINTS_PER_GLB,
        RELATION_K,
        REPRESENTATIVE_SUBPOSES_PER_VIEWCELL,
        ROOT,
        SCENES,
        SEEDS,
        SUBPOSES_PER_VIEWCELL,
        TARGET_UNIT_KIB,
        VIEWCELL_RADIUS_M,
        VIEWCELL_SHAPE,
    )


CONVERTER = ROOT / "neural_instance_culling/tools/graphics_scene_importer/write_slm_scene.mjs"
POINT_GENERATOR = ROOT / "neural_instance_culling/dataset/generate_glb_points_v3.mjs"
GEOMETRY_PREPARER = ROOT / "neural_instance_culling/model/prepare_fixed_geometry_features.py"
COLOR_ID_SAMPLER = ROOT / "neural_instance_culling/sampler/run_scene_viewcell_colorid_sampling.mjs"
POSE_CSR_BUILDER = ROOT / "neural_instance_culling/dataset/build_rvc_viewcell_pose_csr.py"
DEPTH_MANIFEST_BUILDER = ROOT / "neural_instance_culling/dataset/build_triangle_depth_layer_manifest.py"
DEPTH_MANIFEST_SHARDER = ROOT / "neural_instance_culling/dataset/shard_triangle_depth_layer_manifest.py"
DEPTH_SHARD_RUNNER = ROOT / "neural_instance_culling/benchmark/run_triangle_depth_layer_shards.py"
RELATION_BUILDER = ROOT / "neural_instance_culling/dataset/build_observed_relation_csr.py"
CONNECTED_CONVERSION_SCHEMA = "pvs-standard-graphics-scene-connected-sah-pack-v2"
CONNECTED_PARTITION_SCHEMA = "connected-sah-pack-v2"

FORMAL_MODES = (
    "convert",
    "geometry",
    "color-id",
    "pose-csr",
    "depth-manifest",
    "shards",
    "relation",
)
MODES = ("plan", *FORMAL_MODES)


def _path(scene: str, key: str) -> Path:
    try:
        return Path(SCENES[scene][key])
    except KeyError as error:
        raise KeyError(f"unknown Connected-SAH scene/path key: {scene}/{key}") from error


def _python(script: Path) -> list[str]:
    return [sys.executable, "-u", str(script)]


def convert_command(scene: str) -> list[str]:
    """Return the deterministic source-to-Connected-SAH conversion command."""
    return [
        "node",
        str(CONVERTER),
        "--input",
        str(_path(scene, "source")),
        "--output-assets",
        str(_path(scene, "assets")),
        "--target-unit-kib",
        str(TARGET_UNIT_KIB),
        "--overwrite",
    ]


def geometry_commands(scene: str) -> list[list[str]]:
    """Return point-cache and fixed-geometry export commands.

    The point count is present in both commands because the feature exporter
    has a historical default of 64 and must never infer this experiment's
    1024-point contract.
    """
    return [
        [
            "node",
            str(POINT_GENERATOR),
            "--assets-dir",
            str(_path(scene, "assets")),
            "--glb-index",
            str(_path(scene, "glb_index")),
            "--points-per-glb",
            str(POINTS_PER_GLB),
            "--output",
            str(_path(scene, "glb_points")),
            "--meta",
            str(_path(scene, "glb_points_meta")),
        ],
        [
            *_python(GEOMETRY_PREPARER),
            "--encoder-checkpoint",
            str(_path(scene, "geometry_encoder")),
            "--glb-points",
            str(_path(scene, "glb_points")),
            "--runtime-meta",
            str(_path(scene, "runtime_meta")),
            "--output",
            str(_path(scene, "geometry")),
            "--points-per-glb",
            str(POINTS_PER_GLB),
            "--device",
            "cuda",
        ],
    ]


def color_id_command(
    scene: str,
    *,
    parallel: int | None = None,
    shards: int | None = None,
) -> list[str]:
    """Return the V2 Color-ID command with the wrapper's mandatory GPU gate.

    ``run_scene_viewcell_colorid_sampling.mjs`` always passes
    ``--require-hardware-gpu`` to each ``run_sampler.mjs`` child.  The wrapper
    intentionally has no software switch for this formal path.
    """
    config = SCENES[scene]
    command = [
        "node",
        str(COLOR_ID_SAMPLER),
        "--scene",
        scene,
        "--assets-dir",
        str(_path(scene, "assets")),
        "--pose-plan",
        str(_path(scene, "pose_plan")),
        "--output-dir",
        str(_path(scene, "color_id")),
        "--subposes-per-viewcell",
        str(SUBPOSES_PER_VIEWCELL),
        "--width",
        "512",
        "--height",
        "288",
        "--fov-y",
        str(int(MODEL_FOV_Y_DEG)),
        "--parallel",
        str(int(parallel if parallel is not None else config["color_id_parallel"])),
        "--shards",
        str(int(shards if shards is not None else config["color_id_shard_count"])),
    ]
    return command


def pose_csr_command(scene: str) -> list[str]:
    """Return the view-cell union builder command for the new asset inventory."""
    return [
        *_python(POSE_CSR_BUILDER),
        "--raw-dir",
        str(_path(scene, "color_id")),
        "--output-dir",
        str(_path(scene, "dataset")),
        "--runtime-meta",
        str(_path(scene, "runtime_meta")),
        "--experiment",
        f"{scene}_connected_sah_128k_fov66_sampling_v2",
        "--source-sampler",
        "three_color_id",
        "--default-fov-y",
        str(int(MODEL_FOV_Y_DEG)),
    ]


def _depth_common_command(scene: str, output: Path) -> list[str]:
    return [
        *_python(DEPTH_MANIFEST_BUILDER),
        "--dataset-dir",
        str(_path(scene, "dataset")),
        "--runtime-meta",
        str(_path(scene, "runtime_meta")),
        "--glb-index",
        str(_path(scene, "glb_index")),
        "--source-render-manifest",
        str(_path(scene, "source_render_manifest")),
        "--output",
        str(output),
        "--split",
        "train",
        "--viewcell-dataset",
        str(_path(scene, "dataset")),
        "--representative-subposes-per-viewcell",
        str(REPRESENTATIVE_SUBPOSES_PER_VIEWCELL),
    ]


def depth_manifest_command(scene: str) -> list[str]:
    """Return the complete train manifest command used as the audit record."""
    return _depth_common_command(scene, _path(scene, "depth_manifest"))


def depth_shard_ranges(scene: str) -> list[tuple[int, int, int]]:
    """Return ``(shard ordinal, train start, train count)`` for fixed shards."""
    config = SCENES[scene]
    total = int(config["split_counts"]["train"])  # type: ignore[index]
    shard_count = int(config["depth_shard_count"])
    if total <= 0 or shard_count <= 0:
        raise ValueError(f"invalid train/shard count for {scene}: {total}/{shard_count}")
    shard_size = math.ceil(total / shard_count)
    ranges: list[tuple[int, int, int]] = []
    for shard in range(shard_count):
        start = shard * shard_size
        if start >= total:
            break
        ranges.append((shard, start, min(shard_size, total - start)))
    return ranges


def depth_manifest_shard_command(scene: str) -> list[str]:
    """Split the complete audited manifest without duplicating scene records."""
    return [
        *_python(DEPTH_MANIFEST_SHARDER),
        "--input",
        str(_path(scene, "depth_manifest")),
        "--output-dir",
        str(_path(scene, "depth_manifests")),
        "--shards",
        str(int(SCENES[scene]["depth_shard_count"])),
    ]


def depth_shards_command(scene: str, *, jobs: int = 4) -> list[str]:
    """Return the hardware triangle-depth shard runner command."""
    if jobs <= 0:
        raise ValueError("depth shard jobs must be positive")
    return [
        *_python(DEPTH_SHARD_RUNNER),
        "--manifest-dir",
        str(_path(scene, "depth_manifests")),
        "--output-root",
        str(_path(scene, "depth_cache")),
        "--jobs",
        str(jobs),
        "--width",
        str(DEPTH_WIDTH),
        "--height",
        str(DEPTH_HEIGHT),
        "--max-layers",
        str(DEPTH_MAX_LAYERS),
        "--require-hardware-gpu",
        "--resume-existing",
    ]


def relation_command(scene: str) -> list[str]:
    """Return the train-only observed relation builder command for K=8."""
    return [
        *_python(RELATION_BUILDER),
        "--dataset-dir",
        str(_path(scene, "dataset")),
        "--runtime-meta",
        str(_path(scene, "runtime_meta")),
        "--sparse-cache-root",
        str(_path(scene, "depth_cache")),
        "--output-dir",
        str(_path(scene, "relation")),
        "--splits",
        "train",
        "--source-k",
        str(RELATION_K),
    ]


def stage_commands(
    scene: str,
    mode: str,
    *,
    color_parallel: int | None = None,
    color_shards: int | None = None,
    depth_jobs: int = 4,
) -> list[tuple[str, list[str]]]:
    """Return ``(stage name, command)`` entries without executing them."""
    if scene not in SCENES:
        raise ValueError(f"unknown scene: {scene}")
    if mode not in FORMAL_MODES:
        raise ValueError(f"unknown preprocessing mode: {mode}")
    if mode == "convert":
        return [("convert", convert_command(scene))]
    if mode == "geometry":
        names = ("glb_points_1024", "fixed_geometry_features_96d")
        return list(zip(names, geometry_commands(scene), strict=True))
    if mode == "color-id":
        return [("color_id_sampling_hardware", color_id_command(
            scene, parallel=color_parallel, shards=color_shards
        ))]
    if mode == "pose-csr":
        return [("pose_csr_viewcell_union", pose_csr_command(scene))]
    if mode == "depth-manifest":
        return [
            ("depth_manifest", depth_manifest_command(scene)),
            ("depth_manifest_shards", depth_manifest_shard_command(scene)),
        ]
    if mode == "shards":
        return [("triangle_depth_shards_hardware", depth_shards_command(scene, jobs=depth_jobs))]
    return [("relation_csr_k8_train_only", relation_command(scene))]


def _required_paths(scene: str, mode: str) -> Iterable[Path]:
    if mode == "convert":
        return (_path(scene, "source"),)
    if mode == "geometry":
        return (_path(scene, "assets"), _path(scene, "glb_index"), _path(scene, "runtime_meta"))
    if mode == "color-id":
        return (_path(scene, "assets"), _path(scene, "pose_plan"))
    if mode == "pose-csr":
        return (_path(scene, "color_id"), _path(scene, "runtime_meta"))
    if mode == "depth-manifest":
        return (_path(scene, "dataset"), _path(scene, "assets"), _path(scene, "glb_index"))
    if mode == "shards":
        return (_path(scene, "depth_manifests"),)
    return (_path(scene, "dataset"), _path(scene, "depth_cache"), _path(scene, "runtime_meta"))


def _validate_connected_assets(scene: str) -> None:
    manifest_path = _path(scene, "conversion_manifest")
    runtime_path = _path(scene, "runtime_meta")
    index_path = _path(scene, "glb_index")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CONNECTED_CONVERSION_SCHEMA:
        raise ValueError(f"{scene} conversion manifest is not Connected-SAH: {manifest_path}")
    if manifest.get("partitionSchema") != CONNECTED_PARTITION_SCHEMA:
        raise ValueError(f"{scene} conversion manifest has the wrong partition schema")
    if int(manifest.get("targetUnitBytes", -1)) != TARGET_UNIT_KIB * 1024:
        raise ValueError(f"{scene} conversion manifest has the wrong unit target")
    unit_count = int(manifest.get("unitCount", -1))
    if unit_count <= 0:
        raise ValueError(f"{scene} conversion manifest has no renderable units")
    if int(runtime.get("instanceCount", -1)) != unit_count:
        raise ValueError(f"{scene} runtime instance count disagrees with conversion manifest")
    if int(index.get("total", -1)) != unit_count:
        raise ValueError(f"{scene} GLB index count disagrees with conversion manifest")


def preflight(scene: str, mode: str) -> dict[str, Any]:
    """Check only the inputs needed by a selected stage."""
    missing = [str(path) for path in _required_paths(scene, mode) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene} {mode} missing input(s): {missing}")
    if mode != "convert":
        _validate_connected_assets(scene)
    config = SCENES[scene]
    return {
        "schema": "standard-graphics-connected-sah-preprocessing-preflight-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "mode": mode,
        "source": str(_path(scene, "source")),
        "assets": str(_path(scene, "assets")),
        "posePlan": str(_path(scene, "pose_plan")),
        "viewcell": {
            "shape": VIEWCELL_SHAPE,
            "radiusM": VIEWCELL_RADIUS_M,
            "subposesPerViewcell": SUBPOSES_PER_VIEWCELL,
            "modelFovYDeg": MODEL_FOV_Y_DEG,
            "frontendRenderFovYDeg": FRONTEND_RENDER_FOV_Y_DEG,
        },
        "splitCounts": dict(config["split_counts"]),  # type: ignore[arg-type]
        "representativeSubposesPerViewcell": REPRESENTATIVE_SUBPOSES_PER_VIEWCELL,
        "relationK": RELATION_K,
        "pointsPerGlb": POINTS_PER_GLB,
        "seeds": list(SEEDS),
        "testRead": False,
    }


def write_source_render_manifest(scene: str) -> Path:
    """Materialize the binding-only manifest needed by depth preprocessing."""
    runtime_path = _path(scene, "runtime_meta")
    index_path = _path(scene, "glb_index")
    assets = _path(scene, "assets")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    bindings = build_instance_binding_preflight(runtime, index_path, assets)
    payload = {
        "schema": "standard-graphics-connected-sah-source-render-manifest-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "assetsDir": str(assets.resolve()),
        "glbIndex": str(index_path.resolve()),
        "runtimeMeta": str(runtime_path.resolve()),
        "instanceBindings": bindings,
    }
    output = _path(scene, "source_render_manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _write_command_record(
    path: Path,
    *,
    scene: str,
    stage: str,
    command: list[str],
    stdout: Path,
    stderr: Path,
    return_code: int | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema": "standard-graphics-connected-sah-command-log-v1",
        "experiment": EXPERIMENT,
        "scene": scene,
        "stage": stage,
        "command": command,
        "cwd": str(ROOT),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if return_code is not None:
        payload["returnCode"] = int(return_code)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_logged(scene: str, stage: str, command: list[str]) -> None:
    """Run one stage with durable stdout/stderr and a replayable command file."""
    log_dir = _path(scene, "scene_root") / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = log_dir / f"{stage}.stdout.log"
    stderr = log_dir / f"{stage}.stderr.log"
    command_record = log_dir / f"{stage}.command.json"
    _write_command_record(
        command_record,
        scene=scene,
        stage=stage,
        command=command,
        stdout=stdout,
        stderr=stderr,
    )
    with stdout.open("w", encoding="utf-8") as stdout_stream, stderr.open(
        "w", encoding="utf-8"
    ) as stderr_stream:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout_stream,
            stderr=stderr_stream,
            check=False,
        )
    _write_command_record(
        command_record,
        scene=scene,
        stage=stage,
        command=command,
        stdout=stdout,
        stderr=stderr,
        return_code=int(result.returncode),
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_mode(
    scene: str,
    mode: str,
    *,
    color_parallel: int | None = None,
    color_shards: int | None = None,
    depth_jobs: int = 4,
) -> None:
    """Run exactly one named preprocessing mode for one scene."""
    preflight(scene, mode)
    if mode == "depth-manifest":
        write_source_render_manifest(scene)
    for stage, command in stage_commands(
        scene,
        mode,
        color_parallel=color_parallel,
        color_shards=color_shards,
        depth_jobs=depth_jobs,
    ):
        run_logged(scene, stage, command)
    if mode == "convert":
        write_source_render_manifest(scene)


def _scene_names(value: str) -> list[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(names) - set(SCENES))
    if unknown:
        raise ValueError(f"unknown scene(s) {unknown}; choose from {sorted(SCENES)}")
    if not names:
        raise ValueError("at least one scene is required")
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument(
        "--scenes",
        default=",".join(SCENES),
        help="comma-separated registered scene keys",
    )
    parser.add_argument("--color-parallel", type=int, default=None)
    parser.add_argument("--color-shards", type=int, default=None)
    parser.add_argument("--depth-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.color_parallel is not None and args.color_parallel <= 0:
        parser.error("--color-parallel must be positive")
    if args.color_shards is not None and args.color_shards <= 0:
        parser.error("--color-shards must be positive")
    if args.depth_jobs <= 0:
        parser.error("--depth-jobs must be positive")
    scenes = _scene_names(args.scenes)
    if args.mode == "plan":
        for scene in scenes:
            payload = {
                "preflight": preflight(scene, "convert")
                if _path(scene, "source").exists()
                else {"scene": scene, "source": str(_path(scene, "source"))},
                "stages": [
                    {
                        "name": stage,
                        "command": command,
                        "stdout": str(_path(scene, "scene_root") / "logs" / f"{stage}.stdout.log"),
                        "stderr": str(_path(scene, "scene_root") / "logs" / f"{stage}.stderr.log"),
                    }
                    for mode in FORMAL_MODES
                    for stage, command in stage_commands(
                        scene,
                        mode,
                        color_parallel=args.color_parallel,
                        color_shards=args.color_shards,
                        depth_jobs=args.depth_jobs,
                    )
                ],
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for scene in scenes:
        run_mode(
            scene,
            args.mode,
            color_parallel=args.color_parallel,
            color_shards=args.color_shards,
            depth_jobs=args.depth_jobs,
        )


if __name__ == "__main__":
    main()
