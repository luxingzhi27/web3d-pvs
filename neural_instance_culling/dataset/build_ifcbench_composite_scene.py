#!/usr/bin/env python3
"""Build an SLM-style composite scene from IFC-BenchV2 IFC models.

The script converts renderable IFC products into component-level GLB files and
places multiple independent buildings on a synthetic block layout. It writes the
minimal scene package expected by the existing runtime metadata builder:

  assets/
    sceneWeb.json
    conversionManifest.json
    task-0/glb/LOD0/sub_*.glb
    task-0/proxy/proxy.glb

Then run neural_instance_culling/dataset/build_scene_runtime_meta.mjs to produce
glbIndex.json and runtimeVisibilityMeta.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import ifcopenshell
    import ifcopenshell.geom
    import trimesh
except ImportError as exc:  # pragma: no cover - user-facing dependency error.
    raise SystemExit(
        "Missing Python dependency. Install in the active environment, or use a "
        "temporary target and set PYTHONPATH, for example:\n"
        "  python3 -m pip install --target temp/ifc_bench_probe/pydeps "
        "-i https://pypi.tuna.tsinghua.edu.cn/simple ifcopenshell trimesh\n"
        "  PYTHONPATH=temp/ifc_bench_probe/pydeps python3 "
        "neural_instance_culling/dataset/build_ifcbench_composite_scene.py ...\n"
        f"Original import error: {exc}"
    )


DEFAULT_FILTER_TYPES = {
    "IfcAnnotation",
    "IfcBuilding",
    "IfcBuildingStorey",
    "IfcGrid",
    "IfcOpeningElement",
    "IfcSite",
    "IfcSpace",
}

TYPE_COLORS = {
    "IfcWall": [0.74, 0.76, 0.72, 1.0],
    "IfcSlab": [0.58, 0.61, 0.62, 1.0],
    "IfcWindow": [0.45, 0.68, 0.88, 0.72],
    "IfcDoor": [0.58, 0.34, 0.18, 1.0],
    "IfcColumn": [0.66, 0.67, 0.63, 1.0],
    "IfcBeam": [0.58, 0.58, 0.56, 1.0],
    "IfcMember": [0.53, 0.56, 0.58, 1.0],
    "IfcPlate": [0.62, 0.65, 0.66, 1.0],
    "IfcFurniture": [0.65, 0.42, 0.28, 1.0],
    "IfcRailing": [0.40, 0.43, 0.45, 1.0],
    "IfcStair": [0.56, 0.52, 0.48, 1.0],
    "IfcFlowTerminal": [0.72, 0.72, 0.68, 1.0],
    "IfcBuildingElementProxy": [0.72, 0.64, 0.48, 1.0],
}


@dataclass
class BuildingSpec:
    name: str
    ifc_path: Path
    offset_x: float
    offset_z: float
    yaw_deg: float


@dataclass
class ComponentRecord:
    component_id: int
    building_name: str
    source_path: str
    ifc_global_id: str
    ifc_type: str
    ifc_name: str
    base_id: int
    path: str
    vertex_count: int
    face_count: int
    bounds_min: list[float]
    bounds_max: list[float]


@dataclass
class ProductGeometry:
    ifc_global_id: str
    ifc_type: str
    ifc_name: str
    raw_vertices: np.ndarray
    faces: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("temp/ifc_bench_probe/projects"),
        help="Directory containing per-project arc.ifc files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("temp/ifc_bench_probe/generated_ifcbench_composite"),
        help="Output scene root. The script writes <output-root>/assets.",
    )
    parser.add_argument(
        "--scene-name",
        default="ifcbench_composite",
        help="Scene name stored in generated config and manifest.",
    )
    parser.add_argument(
        "--projects",
        nargs="*",
        default=[
            "fantasy_hotel_1",
            "fantasy_hotel_2",
            "fantasy_office_building_1",
            "fantasy_office_building_2",
            "fantasy_office_building_3",
            "fantasy_residential_building_1",
        ],
        help="IFC-BenchV2 project names to compose.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=80.0,
        help="Grid spacing between buildings in generated scene units.",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Repeat the selected project list N times on a larger grid. Copies reuse source IFC files but get unique scene names.",
    )
    parser.add_argument(
        "--max-products-per-building",
        type=int,
        default=0,
        help="Optional pilot limit. 0 means no per-building limit.",
    )
    parser.add_argument(
        "--max-total-products",
        type=int,
        default=0,
        help="Optional global pilot limit. 0 means no global limit.",
    )
    parser.add_argument(
        "--min-diagonal",
        type=float,
        default=1e-4,
        help="Skip products whose converted bounding-box diagonal is smaller.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output-root before writing.",
    )
    parser.add_argument(
        "--skip-proxy",
        action="store_true",
        help="Do not generate task-0/proxy/proxy.glb.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N exported components.",
    )
    return parser.parse_args()


def sanitize_name(value: Any) -> str:
    text = str(value or "")
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in text)[:160]


def rgba_for_type(ifc_type: str) -> list[float]:
    if ifc_type in TYPE_COLORS:
        return TYPE_COLORS[ifc_type]
    seed = sum(ord(ch) for ch in ifc_type)
    hue = (seed * 37) % 360
    chroma = 0.16
    base = 0.58
    phase = math.radians(hue)
    return [
        max(0.25, min(0.85, base + chroma * math.cos(phase))),
        max(0.25, min(0.85, base + chroma * math.cos(phase - 2.094))),
        max(0.25, min(0.85, base + chroma * math.cos(phase + 2.094))),
        1.0,
    ]


def make_building_specs(input_root: Path, projects: list[str], spacing: float, repeat_count: int = 1) -> list[BuildingSpec]:
    specs: list[BuildingSpec] = []
    expanded: list[tuple[str, str]] = []
    repeats = max(1, int(repeat_count))
    for repeat_index in range(repeats):
        for project in projects:
            copy_name = project if repeats == 1 else f"{project}_copy{repeat_index + 1:02d}"
            expanded.append((project, copy_name))
    cols = max(1, math.ceil(math.sqrt(len(expanded))))
    for index, (project, copy_name) in enumerate(expanded):
        ifc_path = input_root / project / "arc.ifc"
        row = index // cols
        col = index % cols
        offset_x = (col - (cols - 1) * 0.5) * spacing
        offset_z = row * spacing
        yaw_deg = [0.0, 8.0, -10.0, 15.0, -6.0, 12.0][index % 6]
        specs.append(BuildingSpec(copy_name, ifc_path, offset_x, offset_z, yaw_deg))
    return specs


def setup_geom_settings() -> Any:
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    try:
        settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, True)
    except Exception:
        pass
    return settings


def iter_renderable_products(model: Any, excluded_types: set[str]) -> list[Any]:
    products = []
    for product in model.by_type("IfcProduct"):
        if not getattr(product, "Representation", None):
            continue
        if product.is_a() in excluded_types:
            continue
        products.append(product)
    return products


def load_project_geometry_cache(ifc_path: Path, settings: Any) -> dict[str, Any]:
    model = ifcopenshell.open(str(ifc_path))
    source_with_representation = [p for p in model.by_type("IfcProduct") if getattr(p, "Representation", None)]
    renderable = iter_renderable_products(model, DEFAULT_FILTER_TYPES)
    products: list[ProductGeometry] = []
    stats = {
        "sourceProductsWithRepresentation": len(source_with_representation),
        "candidateProductsAfterTypeFilter": len(renderable),
        "geometryFailures": 0,
        "skippedEmptyGeometry": 0,
    }
    for product in renderable:
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            geometry = shape.geometry
            raw_vertices = np.asarray(geometry.verts, dtype=np.float64).reshape((-1, 3))
            faces = np.asarray(geometry.faces, dtype=np.int64).reshape((-1, 3))
        except Exception:
            stats["geometryFailures"] += 1
            continue
        if raw_vertices.size == 0 or faces.size == 0:
            stats["skippedEmptyGeometry"] += 1
            continue
        products.append(
            ProductGeometry(
                ifc_global_id=str(getattr(product, "GlobalId", "")),
                ifc_type=product.is_a(),
                ifc_name=str(getattr(product, "Name", "") or ""),
                raw_vertices=raw_vertices,
                faces=faces,
            )
        )
    return {
        "products": products,
        "stats": stats,
    }


def ifc_to_gltf_vertices(raw_vertices: np.ndarray, spec: BuildingSpec) -> np.ndarray:
    """Convert IFC world coordinates to glTF-style Y-up scene coordinates."""
    x = raw_vertices[:, 0]
    y = raw_vertices[:, 1]
    z = raw_vertices[:, 2]
    # IFC is usually Z-up. Three.js/glTF runtime uses Y-up in current scenes.
    converted = np.column_stack([x, z, -y]).astype(np.float64)
    yaw = math.radians(spec.yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    out_x = converted[:, 0] * cos_yaw - converted[:, 2] * sin_yaw + spec.offset_x
    out_z = converted[:, 0] * sin_yaw + converted[:, 2] * cos_yaw + spec.offset_z
    converted[:, 0] = out_x
    converted[:, 2] = out_z
    return converted.astype(np.float32)


def make_mesh(vertices: np.ndarray, faces: np.ndarray, rgba: list[float]) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    rgb = np.array([int(max(0, min(1, c)) * 255) for c in rgba], dtype=np.uint8)
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(rgb, (len(vertices), 1)))
    return mesh


def export_glb(mesh: trimesh.Trimesh, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene(mesh)
    data = scene.export(file_type="glb")
    output_path.write_bytes(data)


def export_proxy_boxes(records: list[ComponentRecord], output_path: Path) -> None:
    scene = trimesh.Scene()
    for record in records:
        bounds_min = np.array(record.bounds_min, dtype=np.float64)
        bounds_max = np.array(record.bounds_max, dtype=np.float64)
        extents = np.maximum(bounds_max - bounds_min, 1e-4)
        center = (bounds_min + bounds_max) * 0.5
        transform = np.eye(4)
        transform[:3, 3] = center
        mesh = trimesh.creation.box(extents=extents, transform=transform)
        rgba = rgba_for_type(record.ifc_type)
        rgb = np.array([int(max(0, min(1, c)) * 255) for c in rgba], dtype=np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(rgb, (len(mesh.vertices), 1)))
        scene.add_geometry(mesh, node_name=f"proxy_{record.component_id}_{sanitize_name(record.ifc_type)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(scene.export(file_type="glb"))


def scene_bounds_from_records(records: list[ComponentRecord]) -> dict[str, list[float]]:
    bounds_min = np.array([record.bounds_min for record in records], dtype=np.float64).min(axis=0)
    bounds_max = np.array([record.bounds_max for record in records], dtype=np.float64).max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    size = np.maximum(bounds_max - bounds_min, 1e-6)
    return {
        "center": [float(x) for x in center],
        "size": [float(x) for x in size],
    }


def write_scene_web(
    assets_dir: Path,
    component_count: int,
    scene_name: str,
    has_proxy: bool,
    bounds: dict[str, list[float]],
) -> None:
    scene_web = {
        "materials": {
            "proxy": ["task-0/proxy/proxy.glb"] if has_proxy else [],
            "useHdrJpg": False,
        },
        "groups": [
            {
                "idRange": [0, max(0, component_count - 1)],
                "instances": {},
            }
        ],
        "config": {
            "sceneName": scene_name,
            "source": "IFC-BenchV2 composite generated by build_ifcbench_composite_scene.py",
            "bounds": bounds,
        },
    }
    (assets_dir / "sceneWeb.json").write_text(json.dumps(scene_web, ensure_ascii=False, indent=2), "utf8")


def write_manifest(
    assets_dir: Path,
    scene_name: str,
    specs: list[BuildingSpec],
    records: list[ComponentRecord],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "schemaVersion": 1,
        "sceneName": scene_name,
        "sourceDataset": "sylvainHellin/ifc-bench",
        "sourceDatasetUrl": "https://huggingface.co/datasets/sylvainHellin/ifc-bench",
        "licenseNote": "Probe projects downloaded from IFC-BenchV2 report MIT License in per-project license.txt.",
        "buildingPlacement": [
            {
                "name": spec.name,
                "ifcPath": str(spec.ifc_path),
                "offset": [spec.offset_x, 0.0, spec.offset_z],
                "yawDeg": spec.yaw_deg,
            }
            for spec in specs
        ],
        "componentCount": len(records),
        "stats": stats,
        "filters": {
            "excludedIfcTypes": sorted(DEFAULT_FILTER_TYPES),
            "minDiagonal": args.min_diagonal,
            "maxProductsPerBuilding": args.max_products_per_building,
            "maxTotalProducts": args.max_total_products,
        },
        "components": [record.__dict__ for record in records],
    }
    (assets_dir / "conversionManifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        "utf8",
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    assets_dir = output_root / "assets"
    glb_dir = assets_dir / "task-0" / "glb" / "LOD0"
    proxy_path = assets_dir / "task-0" / "proxy" / "proxy.glb"

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    glb_dir.mkdir(parents=True, exist_ok=True)

    specs = make_building_specs(args.input_root, args.projects, args.spacing, args.repeat_count)
    missing = [str(spec.ifc_path) for spec in specs if not spec.ifc_path.exists()]
    if missing:
        raise SystemExit("Missing IFC files:\n" + "\n".join(missing))

    settings = setup_geom_settings()
    geometry_cache_by_path: dict[str, dict[str, Any]] = {}
    records: list[ComponentRecord] = []
    stats: dict[str, Any] = {
        "startedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "buildings": {},
        "exported": 0,
        "skippedByType": 0,
        "skippedEmptyGeometry": 0,
        "skippedTinyGeometry": 0,
        "geometryFailures": 0,
    }
    start_time = time.time()

    for spec in specs:
        building_start = time.time()
        print(f"[ifcbench] loading {spec.name}: {spec.ifc_path}", flush=True)
        cache_key = str(spec.ifc_path.resolve())
        if cache_key not in geometry_cache_by_path:
            geometry_cache_by_path[cache_key] = load_project_geometry_cache(spec.ifc_path, settings)
        geometry_cache = geometry_cache_by_path[cache_key]
        renderable: list[ProductGeometry] = geometry_cache["products"]
        cached_stats = geometry_cache["stats"]
        building_stats = {
            "sourceProductsWithRepresentation": cached_stats["sourceProductsWithRepresentation"],
            "candidateProductsAfterTypeFilter": cached_stats["candidateProductsAfterTypeFilter"],
            "exported": 0,
            "geometryFailures": cached_stats["geometryFailures"],
            "skippedEmptyGeometry": cached_stats["skippedEmptyGeometry"],
            "skippedTinyGeometry": 0,
            "usedCachedGeometry": True,
        }
        stats["geometryFailures"] += cached_stats["geometryFailures"]
        stats["skippedEmptyGeometry"] += cached_stats["skippedEmptyGeometry"]
        per_building_exported = 0
        for product in renderable:
            if args.max_total_products and len(records) >= args.max_total_products:
                break
            if args.max_products_per_building and per_building_exported >= args.max_products_per_building:
                break
            ifc_type = product.ifc_type
            vertices = ifc_to_gltf_vertices(product.raw_vertices, spec)
            bounds_min = vertices.min(axis=0)
            bounds_max = vertices.max(axis=0)
            diagonal = float(np.linalg.norm(bounds_max - bounds_min))
            if not math.isfinite(diagonal) or diagonal < args.min_diagonal:
                stats["skippedTinyGeometry"] += 1
                building_stats["skippedTinyGeometry"] += 1
                continue
            component_id = len(records)
            out_rel = f"task-0/glb/LOD0/sub_{component_id}.glb"
            out_path = assets_dir / out_rel
            mesh = make_mesh(vertices, product.faces, rgba_for_type(ifc_type))
            export_glb(mesh, out_path)
            record = ComponentRecord(
                component_id=component_id,
                building_name=spec.name,
                source_path=str(spec.ifc_path),
                ifc_global_id=product.ifc_global_id,
                ifc_type=ifc_type,
                ifc_name=product.ifc_name,
                base_id=component_id,
                path=out_rel,
                vertex_count=int(len(vertices)),
                face_count=int(len(product.faces)),
                bounds_min=[float(x) for x in bounds_min],
                bounds_max=[float(x) for x in bounds_max],
            )
            records.append(record)
            stats["exported"] += 1
            building_stats["exported"] += 1
            per_building_exported += 1
            if args.progress_every > 0 and len(records) % args.progress_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[ifcbench] exported={len(records)} elapsed={elapsed:.1f}s "
                    f"last={spec.name}/{ifc_type}",
                    flush=True,
                )
        building_stats["elapsedSec"] = round(time.time() - building_start, 3)
        stats["buildings"][spec.name] = building_stats
        print(f"[ifcbench] finished {spec.name}: {building_stats}", flush=True)
        if args.max_total_products and len(records) >= args.max_total_products:
            break

    if not records:
        raise SystemExit("No components were exported.")

    scene_bounds = scene_bounds_from_records(records)
    write_scene_web(
        assets_dir,
        len(records),
        args.scene_name,
        has_proxy=not args.skip_proxy,
        bounds=scene_bounds,
    )
    if not args.skip_proxy:
        print(f"[ifcbench] writing proxy boxes: {proxy_path}", flush=True)
        export_proxy_boxes(records, proxy_path)
    stats["elapsedSec"] = round(time.time() - start_time, 3)
    stats["finishedAt"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_manifest(assets_dir, args.scene_name, specs, records, stats, args)
    print(
        json.dumps(
            {
                "sceneName": args.scene_name,
                "outputRoot": str(output_root),
                "assetsDir": str(assets_dir),
                "components": len(records),
                "elapsedSec": stats["elapsedSec"],
                "next": (
                    "node neural_instance_culling/dataset/build_scene_runtime_meta.mjs "
                    f"--assets-dir {assets_dir} --scene-name {args.scene_name} --overwrite"
                ),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
