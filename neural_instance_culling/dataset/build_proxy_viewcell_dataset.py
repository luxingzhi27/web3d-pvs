#!/usr/bin/env python3
"""构建 viewcell PVS 中间数据集。

该数据集为当前 back-camera pose CSR 构建提供输入：
- 不生成世界体素。
- 不使用 clean mask。
- 不生成 voxel -> instance 映射。
- GT 是同一 viewcell 内真实 dense subpose 的 rvcServer 可见集合并集。

如果提供修正后的 pose-plan，则只从 pose-plan 覆盖 viewcell/PVS 包络元数据；
visible ids/weights 仍来自已经采样的 raw JSONL。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


CATEGORY_IDS = {
    "street_gap": 0,
    "near_building": 1,
    "plaza": 2,
    "perimeter": 3,
    "sky": 4,
    "far": 5,
    "unknown": 255,
}
SPLIT_IDS = {"train": 0, "val": 1, "test": 2, "unknown": 255}
MODEL_INPUT_FOV_Y_DEG = 66.0
DEFAULT_ASPECT = 16.0 / 9.0


def horizontal_fov_deg(fov_y_deg: float, aspect: float) -> float:
    """Return the horizontal FOV for the model's vertical FOV and aspect ratio."""
    safe_aspect = max(1e-6, float(aspect))
    half_y = math.radians(float(fov_y_deg)) * 0.5
    return math.degrees(2.0 * math.atan(math.tan(half_y) * safe_aspect))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build viewcell PVS dataset without world voxels.")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, default=Path("hkust-v3/assets/runtimeVisibilityMeta.json"))
    parser.add_argument("--pose-plan", type=Path, default=None)
    parser.add_argument("--source-sampler", choices=["rvc", "three_color_id"], default="rvc")
    parser.add_argument("--experiment", default=None, help="Dataset experiment name; defaults to the output directory name.")
    parser.add_argument("--max-raw-rows", type=int, default=0)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path_or_dir: Path, max_rows: int = 0) -> list[dict[str, Any]]:
    files = sorted(path_or_dir.glob("*.jsonl")) if path_or_dir.is_dir() else [path_or_dir]
    rows: list[dict[str, Any]] = []
    for file in files:
        with file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
                if max_rows > 0 and len(rows) >= max_rows:
                    return rows
    return rows


def load_pose_plan(path: Path | None) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, dict[str, Any]]]:
    if path is None:
        return {}, {}
    by_pair: dict[tuple[int, int], dict[str, Any]] = {}
    by_pose: dict[int, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if "pose_index" in row:
            by_pose[int(row["pose_index"])] = row
        if "viewcell_id" in row and "subpose_id" in row:
            by_pair[(int(row["viewcell_id"]), int(row["subpose_id"]))] = row
    return by_pair, by_pose


def apply_pose_plan(records: list[dict[str, Any]], pose_plan: Path | None) -> dict[str, Any]:
    by_pair, by_pose = load_pose_plan(pose_plan)
    if pose_plan is None:
        return {"enabled": False}
    fields = [
        "viewcell_category",
        "viewcell_center",
        "viewcell_radius",
        "viewcell_forward",
        "viewcell_yaw_deg",
        "viewcell_pitch_deg",
        "pvs_fov_y",
        "pvs_fov_x",
        "pvs_back_offset",
        "split",
        "sample_category",
        "sample_category_id",
        "camera_pos",
        "camera_forward",
        "yaw_deg",
        "pitch_deg",
        "fov_y",
        "aspect",
        "width",
        "height",
    ]
    matched = 0
    missing = 0
    for rec in records:
        plan = None
        if "viewcell_id" in rec and "subpose_id" in rec:
            plan = by_pair.get((int(rec["viewcell_id"]), int(rec["subpose_id"])))
        if plan is None and "pose_index" in rec:
            plan = by_pose.get(int(rec["pose_index"]))
        if plan is None:
            missing += 1
            continue
        for field in fields:
            if field in plan:
                rec[field] = plan[field]
        matched += 1
    return {
        "enabled": True,
        "posePlan": str(pose_plan),
        "planRows": int(max(len(by_pair), len(by_pose))),
        "matchedRows": int(matched),
        "missingRows": int(missing),
    }


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i, rec in enumerate(records):
        groups[int(rec.get("viewcell_id", rec.get("pose_index", i)))].append(rec)
    out: list[dict[str, Any]] = []
    for viewcell_id in sorted(groups):
        rows = groups[viewcell_id]
        first = rows[0]
        weights_by_id: dict[int, float] = {}
        hits_by_id: dict[int, int] = defaultdict(int)
        for row in rows:
            ids = row.get("visible_component_ids") or []
            weights = row.get("component_weights") or []
            for j, cid_raw in enumerate(ids):
                cid = int(cid_raw)
                weight = float(weights[j]) if j < len(weights) else 1.0
                weights_by_id[cid] = max(weights_by_id.get(cid, 0.0), weight)
                hits_by_id[cid] += 1
        visible_ids = sorted(weights_by_id)
        aspect = float(first.get("aspect", DEFAULT_ASPECT))
        pvs_fov_y = float(first.get("pvs_fov_y", MODEL_INPUT_FOV_Y_DEG))
        pvs_fov_x = float(first.get("pvs_fov_x", horizontal_fov_deg(pvs_fov_y, aspect)))
        out.append(
            {
                "viewcell_id": int(viewcell_id),
                "sample_category": first.get("viewcell_category") or first.get("sample_category") or "unknown",
                "sample_category_id": int(first.get("sample_category_id", CATEGORY_IDS.get(first.get("sample_category", "unknown"), 255))),
                "split": first.get("split", "unknown"),
                "viewcell_center": first.get("viewcell_center") or first.get("camera_pos"),
                "viewcell_radius": float(first.get("viewcell_radius", 0.0)),
                "viewcell_forward": first.get("viewcell_forward") or first.get("camera_forward"),
                "viewcell_yaw_deg": float(first.get("viewcell_yaw_deg", first.get("yaw_deg", 0.0))),
                "viewcell_pitch_deg": float(first.get("viewcell_pitch_deg", first.get("pitch_deg", 0.0))),
                "render_fov_y": float(first.get("render_fov_y", first.get("fov_y", MODEL_INPUT_FOV_Y_DEG))),
                "pvs_fov_y": pvs_fov_y,
                "pvs_fov_x": pvs_fov_x,
                "pvs_back_offset": float(first.get("pvs_back_offset", 0.0)),
                "subposes": rows,
                "visible_ids": visible_ids,
                "visible_weights": [float(weights_by_id[cid]) for cid in visible_ids],
                "visible_hit_counts": [int(hits_by_id[cid]) for cid in visible_ids],
            }
        )
    return out


def write_dataset(
    viewcells: list[dict[str, Any]],
    output_dir: Path,
    runtime_meta: dict[str, Any],
    raw_summary: dict[str, Any],
    pose_plan_summary: dict[str, Any],
    source_sampler: str,
    experiment: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(viewcells)
    viewcell_ids = np.asarray([vc["viewcell_id"] for vc in viewcells], dtype="<u4")
    centers = np.asarray([vc["viewcell_center"] for vc in viewcells], dtype="<f4")
    forwards = np.asarray([vc["viewcell_forward"] for vc in viewcells], dtype="<f4")
    params = np.asarray(
        [
            [
                vc["viewcell_radius"],
                vc["render_fov_y"],
                vc["pvs_fov_y"],
                vc["pvs_fov_x"],
                vc["pvs_back_offset"],
                vc["viewcell_yaw_deg"],
                vc["viewcell_pitch_deg"],
                len(vc["subposes"]),
            ]
            for vc in viewcells
        ],
        dtype="<f4",
    )
    category_ids = np.asarray([CATEGORY_IDS.get(vc["sample_category"], CATEGORY_IDS["unknown"]) for vc in viewcells], dtype=np.uint8)
    split_ids = np.asarray([SPLIT_IDS.get(vc["split"], SPLIT_IDS["unknown"]) for vc in viewcells], dtype=np.uint8)
    visible_offsets = np.zeros((n + 1,), dtype="<u8")
    visible_ids: list[int] = []
    visible_weights: list[float] = []
    visible_hits: list[int] = []
    subpose_offsets = np.zeros((n + 1,), dtype="<u8")
    subpose_pos: list[list[float]] = []
    subpose_forward: list[list[float]] = []
    subpose_params: list[list[float]] = []
    subpose_pose_index: list[int] = []
    for i, vc in enumerate(viewcells):
        visible_ids.extend(vc["visible_ids"])
        visible_weights.extend(vc["visible_weights"])
        visible_hits.extend(vc["visible_hit_counts"])
        visible_offsets[i + 1] = len(visible_ids)
        for row in vc["subposes"]:
            subpose_pose_index.append(int(row.get("pose_index", len(subpose_pose_index))))
            subpose_pos.append(row.get("camera_pos") or vc["viewcell_center"])
            subpose_forward.append(row.get("camera_forward") or vc["viewcell_forward"])
            subpose_params.append(
                [
                    float(row.get("fov_y", vc["render_fov_y"])),
                    float(row.get("aspect", 16 / 9)),
                    float(row.get("width", 0.0)),
                    float(row.get("height", 0.0)),
                ]
            )
        subpose_offsets[i + 1] = len(subpose_pose_index)
    viewcell_ids.tofile(output_dir / "viewcell_ids.bin")
    centers.tofile(output_dir / "viewcell_centers.bin")
    forwards.tofile(output_dir / "viewcell_forwards.bin")
    params.tofile(output_dir / "viewcell_params.bin")
    category_ids.tofile(output_dir / "viewcell_category_ids.bin")
    split_ids.tofile(output_dir / "viewcell_split_ids.bin")
    visible_offsets.tofile(output_dir / "visible_offsets.bin")
    np.asarray(visible_ids, dtype="<u4").tofile(output_dir / "visible_ids.bin")
    np.asarray(visible_weights, dtype="<f4").tofile(output_dir / "visible_weights.bin")
    np.asarray(visible_hits, dtype="<u2").tofile(output_dir / "visible_hit_counts.bin")
    subpose_offsets.tofile(output_dir / "subpose_offsets.bin")
    np.asarray(subpose_pose_index, dtype="<u4").tofile(output_dir / "subpose_pose_indices.bin")
    np.asarray(subpose_pos, dtype="<f4").tofile(output_dir / "subpose_camera_pos.bin")
    np.asarray(subpose_forward, dtype="<f4").tofile(output_dir / "subpose_camera_forward.bin")
    np.asarray(subpose_params, dtype="<f4").tofile(output_dir / "subpose_params.bin")
    with (output_dir / "viewcells.jsonl").open("w", encoding="utf-8") as f:
        for vc in viewcells:
            slim = {k: v for k, v in vc.items() if k not in {"subposes", "visible_ids", "visible_weights", "visible_hit_counts"}}
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")
    records = runtime_meta.get("componentRecords", [])
    num_instances = max(int(r["componentGlobalId"]) for r in records) + 1 if records else 0
    if source_sampler == "three_color_id":
        weight_semantics = "Three.js color-id screen coverage in parts per million; max-pooled over dense subposes"
    else:
        weight_semantics = "rvcServer component_weights max-pooled over dense subposes; not pixel coverage"
    meta = {
        "schema": "proxy-viewcell-pvs-dataset-v1",
        "experiment": experiment,
        "sourceSampler": source_sampler,
        "viewcellCount": int(n),
        "numInstances": int(num_instances),
        "visibleCount": int(len(visible_ids)),
        "subposeCount": int(len(subpose_pose_index)),
        "rawSampling": raw_summary,
        "posePlanOverride": pose_plan_summary,
        "visibleWeightSemantics": weight_semantics,
        "modelInputFovYDeg": MODEL_INPUT_FOV_Y_DEG,
        "frontendRenderFovYDeg": 60.0,
        "worldVoxelDependency": False,
        "usesCleanMask": False,
        "categoryIds": CATEGORY_IDS,
        "splitIds": SPLIT_IDS,
        "categoryCounts": {name: int(np.count_nonzero(category_ids == cid)) for name, cid in CATEGORY_IDS.items() if name != "unknown"},
        "splitCounts": {name: int(np.count_nonzero(split_ids == sid)) for name, sid in SPLIT_IDS.items() if name != "unknown"},
        "files": {
            "viewcellIds": "viewcell_ids.bin",
            "viewcellCenters": "viewcell_centers.bin",
            "viewcellForwards": "viewcell_forwards.bin",
            "viewcellParams": "viewcell_params.bin",
            "viewcellCategoryIds": "viewcell_category_ids.bin",
            "viewcellSplitIds": "viewcell_split_ids.bin",
            "visibleOffsets": "visible_offsets.bin",
            "visibleIds": "visible_ids.bin",
            "visibleWeights": "visible_weights.bin",
            "visibleHitCounts": "visible_hit_counts.bin",
            "subposeOffsets": "subpose_offsets.bin",
            "subposePoseIndices": "subpose_pose_indices.bin",
            "subposeCameraPos": "subpose_camera_pos.bin",
            "subposeCameraForward": "subpose_camera_forward.bin",
            "subposeParams": "subpose_params.bin",
        },
    }
    (output_dir / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    args = parse_args()
    runtime_meta = read_json(args.runtime_meta)
    records = iter_jsonl(args.raw_dir, args.max_raw_rows)
    pose_plan_summary = apply_pose_plan(records, args.pose_plan)
    raw_summary = {
        "rawRows": len(records),
        "sampleErrorRows": int(sum(1 for r in records if r.get("sample_error"))),
        "emptyVisibleRows": int(sum(1 for r in records if not (r.get("visible_component_ids") or []))),
    }
    viewcells = aggregate(records)
    experiment = str(args.experiment or args.output_dir.name)
    meta = write_dataset(viewcells, args.output_dir, runtime_meta, raw_summary, pose_plan_summary, args.source_sampler, experiment)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
