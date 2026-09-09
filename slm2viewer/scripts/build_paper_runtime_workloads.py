#!/usr/bin/env python3
"""Build the two frozen test workloads used by the paper runtime page."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


SCENES = (
    {
        "id": "hkust",
        "label": "HKUST (684 test view-cells)",
        "dataset": "pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
        "model": "pvs_mainline_v4",
        "expected": 684,
    },
    {
        "id": "ifcbench",
        "label": "IFCBench / Metropolis (2710 test view-cells)",
        "dataset": "pose_csr_ifcbench_fantasy_metropolis_main_stratified_calibration_fov66_v1",
        "model": "pvs_mainline_v4_ifcbench",
        "expected": 2710,
    },
)


def parse_args() -> argparse.Namespace:
    viewer_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=viewer_root.parent)
    parser.add_argument("--viewer-root", type=Path, default=viewer_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    viewer_root = args.viewer_root.resolve()
    builder = viewer_root / "scripts/build_pvs_runtime_workload.py"
    output_root = viewer_root / "assets/benchmark/pvs_paper_runtime"
    scene_rows = []
    for scene in SCENES:
        dataset = data_root / "neural_instance_culling/dataset/out" / scene["dataset"]
        model = viewer_root / "assets/neural_instance_culling" / scene["model"]
        if not dataset.is_dir() or not (model / "model_meta.json").is_file():
            raise FileNotFoundError(f"missing dataset or exported model for {scene['id']}")
        scene_output = output_root / scene["id"]
        model_url = f"./assets/neural_instance_culling/{scene['model']}"
        command = [
            sys.executable,
            str(builder),
            "--dataset-dir", str(dataset),
            "--output-dir", str(scene_output),
            "--scene", scene["id"],
            "--experiment-name", f"pvs_paper_runtime_{scene['id']}_v1",
            "--model-asset-url", model_url,
            "--expected-test-count", str(scene["expected"]),
        ]
        subprocess.run(command, cwd=viewer_root, check=True)
        scene_rows.append({
            "id": scene["id"],
            "label": scene["label"],
            "workloadUrl": f"./assets/benchmark/pvs_paper_runtime/{scene['id']}/workload.json",
            "modelAssetPath": model_url,
            "backends": ["webgpu", "wasm"],
        })
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "pvs-paper-runtime-scenes-v1", "scenes": scene_rows}
    (output_root / "scenes.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
