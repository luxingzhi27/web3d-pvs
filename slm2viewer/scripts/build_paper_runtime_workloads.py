#!/usr/bin/env python3
"""Build frozen-test workloads and copy the selected paper runtime assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


RUNTIME_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-moment-runtime-v4"
SCENES = (
    {
        "id": "hkust",
        "label": "HKUST (684 test view-cells)",
        "dataset": "pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
        "source": "neural_instance_culling/model/out/"
        "hkust_v4_full_seed20260802_epoch036_image_eval_bundle_20260902",
        "asset": "pvs_hkust_v4_full_seed20260802_epoch036",
        "expected": 684,
        "instances": 18831,
        "threshold": 0.6800000071525574,
        "radius": 2.0,
    },
    {
        "id": "ifcbench",
        "label": "IFCBench / Metropolis (2710 test view-cells)",
        "dataset": "pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "source": "neural_instance_culling/model/out/"
        "pvs_ifcbench_v4_calibration_finetune_v2_refine/"
        "runtime_final_finetune_boundary_half_rvl030_boundary010_margin050_"
        "temp025_lr2e-5_seed20260801_v1",
        "asset": "pvs_ifcbench_v4_refine_seed20260801_epoch004",
        "expected": 2710,
        "instances": 41298,
        "threshold": 0.6882505416870117,
        "radius": 2.5,
    },
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    viewer_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=viewer_root.parent)
    parser.add_argument("--viewer-root", type=Path, default=viewer_root)
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_runtime(source: Path, scene: dict[str, Any]) -> None:
    meta_path = source / "model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing runtime metadata: {meta_path}")
    meta = _read_json(meta_path)
    if meta.get("schema") != RUNTIME_SCHEMA or meta.get("testRead") is not False:
        raise ValueError(f"invalid runtime schema or test provenance: {meta_path}")
    if int(meta.get("numInstances", -1)) != scene["instances"]:
        raise ValueError(f"runtime instance count mismatch for {scene['id']}")
    if abs(float(meta.get("threshold")) - scene["threshold"]) > 1e-7:
        raise ValueError(f"runtime threshold mismatch for {scene['id']}")
    viewcell = meta.get("viewcell")
    query = meta.get("query")
    if (
        not isinstance(viewcell, dict)
        or not isinstance(query, dict)
        or abs(float(viewcell.get("radiusM")) - scene["radius"]) > 1e-7
        or abs(float(query.get("viewcellRadiusM")) - scene["radius"]) > 1e-7
    ):
        raise ValueError(f"runtime view-cell radius mismatch for {scene['id']}")
    calibration = meta.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("safe") is not True:
        raise ValueError(f"runtime calibration is not safe for {scene['id']}")
    files = meta.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"runtime file manifest is missing for {scene['id']}")
    for record in files.values():
        if isinstance(record, dict) and record.get("file"):
            candidate = source / str(record["file"])
            if not candidate.is_file():
                raise FileNotFoundError(f"missing runtime asset: {candidate}")


def _install_runtime(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)


def build(data_root: Path, viewer_root: Path) -> dict[str, Any]:
    builder = viewer_root / "scripts/build_pvs_runtime_workload.py"
    public_root = viewer_root / "public"
    output_root = public_root / "assets/benchmark/pvs_paper_runtime"
    asset_root = public_root / "assets/neural_instance_culling"
    scene_rows = []
    for scene in SCENES:
        dataset = data_root / "neural_instance_culling/dataset/out" / scene["dataset"]
        source = data_root / scene["source"]
        destination = asset_root / scene["asset"]
        if not dataset.is_dir():
            raise FileNotFoundError(f"missing dataset for {scene['id']}: {dataset}")
        _validate_runtime(source, scene)
        _install_runtime(source, destination)
        _validate_runtime(destination, scene)

        scene_output = output_root / scene["id"]
        model_url = f"./assets/neural_instance_culling/{scene['asset']}"
        subprocess.run(
            [
                sys.executable,
                str(builder),
                "--dataset-dir",
                str(dataset),
                "--output-dir",
                str(scene_output),
                "--scene",
                scene["id"],
                "--experiment-name",
                f"pvs_paper_runtime_{scene['id']}_v1",
                "--model-asset-url",
                model_url,
                "--expected-test-count",
                str(scene["expected"]),
            ],
            cwd=viewer_root,
            check=True,
        )
        scene_rows.append(
            {
                "id": scene["id"],
                "label": scene["label"],
                "workloadUrl": f"./assets/benchmark/pvs_paper_runtime/{scene['id']}/workload.json",
                "modelAssetPath": model_url,
                "backends": ["webgpu", "wasm"],
            }
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "pvs-paper-runtime-scenes-v1", "scenes": scene_rows}
    (output_root / "scenes.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build(args.data_root.resolve(), args.viewer_root.resolve())
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
