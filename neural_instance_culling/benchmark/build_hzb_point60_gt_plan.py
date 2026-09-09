#!/usr/bin/env python3
"""Build canonical real-camera Point60 Color-ID sampling plans.

The input representative plan is aligned row-for-row with a directional Pose
CSR dataset.  Only the Pose CSR ``test`` rows are selected.  The output keeps
the original Pose CSR row as both ``pose_index`` and ``viewcell_id`` and emits
one canonical ``subpose_id=0`` camera per row.  Viewport groups are written to
separate JSONL files so an IFCBench plan never silently replaces its original
aspect, width, or height with one global viewport.

This script only builds CPU-readable plans.  The resulting plan files are
later passed to ``run_sampler.mjs --point60-gt``; the sampler's JSONL output,
not these plan rows, is the raw directory consumed by the HZB evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


POINT60_FOV_Y_DEG = 60.0
POSE_STRIDE_BYTES = 64

DIRECTIONAL_POSE_DTYPE = np.dtype(
    {
        "names": [
            "camera_norm",
            "camera_world",
            "camera_forward",
            "camera_view",
            "split",
            "category",
            "reserved0",
            "reserved1",
        ],
        "formats": [
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (2,)),
            "u1",
            "u1",
            "<u2",
            "<u4",
        ],
        "offsets": [0, 12, 24, 36, 44, 45, 46, 48],
        "itemsize": POSE_STRIDE_BYTES,
    }
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def finite_float(value: Any, field: str, source_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"representative row {source_index} has a non-numeric {field}") from error
    if not math.isfinite(number):
        raise ValueError(f"representative row {source_index} has a non-finite {field}")
    return number


def positive_int(value: Any, field: str, source_index: int) -> int:
    number = finite_float(value, field, source_index)
    rounded = round(number)
    if number <= 0 or abs(number - rounded) > 1e-6:
        raise ValueError(f"representative row {source_index} has an invalid positive integer {field}")
    return int(rounded)


def finite_vector(value: Any, field: str, source_index: int, *, nonzero: bool = False) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"representative row {source_index} {field} must be a length-3 array")
    vector = np.asarray(value, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError(f"representative row {source_index} has a non-finite {field}")
    if nonzero and float(np.linalg.norm(vector)) <= 1e-8:
        raise ValueError(f"representative row {source_index} has a zero {field}")
    return [float(number) for number in vector.tolist()]


def normalized_forward(value: Any, source_index: int) -> list[float]:
    vector = np.asarray(finite_vector(value, "camera_forward", source_index, nonzero=True), dtype=np.float64)
    vector /= float(np.linalg.norm(vector))
    return [float(number) for number in vector.tolist()]


def horizontal_fov_from_vertical(vertical_fov_deg: float, aspect: float) -> float:
    tangent = math.tan(math.radians(vertical_fov_deg) * 0.5)
    return math.degrees(2.0 * math.atan(tangent * aspect))


def aspect_from_row(row: dict[str, Any], width: int, height: int, source_index: int) -> tuple[float, str]:
    if row.get("aspect") is None:
        # Deriving from explicit dimensions is still a real viewport.  It is
        # different from imposing one default aspect on every scene/row.
        return float(width / height), "width_div_height"
    return finite_float(row["aspect"], "aspect", source_index), "representative_plan"


def aspect_token(aspect: float) -> str:
    # Python's shortest round-trip representation keeps names readable while
    # remaining distinct for distinct float viewport keys.
    token = repr(float(aspect))
    return token.replace("-", "m").replace(".", "p")


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Point60 output path is not a directory: {output_dir}")
        if any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite a non-empty Point60 output directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _load_test_indices(dataset_dir: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, int]:
    meta = read_json(dataset_dir / "dataset_meta.json")
    declared_stride = meta.get("poseStrideBytes")
    if declared_stride is not None and int(declared_stride) != POSE_STRIDE_BYTES:
        raise ValueError(
            f"Point60 plan builder requires the directional {POSE_STRIDE_BYTES}-byte Pose CSR format; "
            f"dataset declares poseStrideBytes={declared_stride}"
        )
    poses = np.fromfile(dataset_dir / "poses.bin", dtype=DIRECTIONAL_POSE_DTYPE)
    declared_count = meta.get("poseCount")
    if declared_count is not None and int(declared_count) != int(poses.size):
        raise ValueError(
            f"Pose CSR poseCount={declared_count} does not match poses.bin rows={poses.size}"
        )
    split_ids = meta.get("splitIds") or {}
    if "test" not in split_ids:
        raise ValueError("Pose CSR metadata must declare splitIds.test")
    test_id = int(split_ids["test"])
    test_indices = np.flatnonzero(poses["split"] == test_id).astype(np.int64, copy=False)
    if test_indices.size == 0:
        raise ValueError("Pose CSR test split is empty; cannot build a Point60 GT plan")
    declared_split_counts = meta.get("splitCounts") or {}
    if "test" in declared_split_counts and int(declared_split_counts["test"]) != int(test_indices.size):
        raise ValueError(
            "Pose CSR splitCounts.test does not match poses.bin: "
            f"declared={declared_split_counts['test']} actual={test_indices.size}"
        )
    return meta, poses, test_indices, test_id


def _point60_row(representative: dict[str, Any], source_index: int) -> tuple[dict[str, Any], tuple[int, int, float], str]:
    if representative.get("pose_index") is not None:
        try:
            declared_pose_index = int(representative["pose_index"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"representative row {source_index} has a non-integer pose_index") from error
        if declared_pose_index != source_index:
            raise ValueError(
                "representative plan must be row-aligned with Pose CSR: "
                f"row={source_index} pose_index={declared_pose_index}"
            )

    camera_pos = finite_vector(representative.get("camera_pos"), "camera_pos", source_index)
    camera_forward = normalized_forward(representative.get("camera_forward"), source_index)
    width = positive_int(representative.get("width"), "width", source_index)
    height = positive_int(representative.get("height"), "height", source_index)
    aspect, aspect_source = aspect_from_row(representative, width, height, source_index)
    if aspect <= 0:
        raise ValueError(f"representative row {source_index} has a non-positive aspect")
    rendered_width = max(1, round(height * aspect))
    if rendered_width != width:
        raise ValueError(
            "representative viewport is not reproducible by run_sampler: "
            f"row={source_index} width={width} height={height} aspect={aspect} "
            f"round(height*aspect)={rendered_width}"
        )

    center_value = representative.get("viewcell_center")
    center = finite_vector(
        camera_pos if center_value is None else center_value,
        "viewcell_center",
        source_index,
    )
    yaw = math.degrees(math.atan2(-camera_forward[0], -camera_forward[2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, camera_forward[1]))))
    row = dict(representative)
    row.update(
        {
            "pose_index": source_index,
            "viewcell_id": source_index,
            "subpose_id": 0,
            "source_pose_index": source_index,
            "split": "test",
            "point60_gt": True,
            "ground_truth_mode": "Point60",
            "camera_semantics": "real_camera_canonical_subpose",
            "camera_pos": camera_pos,
            "camera_forward": camera_forward,
            "viewcell_center": center,
            "viewcell_forward": camera_forward,
            "viewcell_yaw_deg": yaw,
            "viewcell_pitch_deg": pitch,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "viewcell_shape": representative.get("viewcell_shape", "point"),
            "viewcell_half_extent": representative.get("viewcell_half_extent", [0.0, 0.0, 0.0]),
            "viewcell_radius": representative.get("viewcell_radius", 0.0),
            "pvs_fov_y": POINT60_FOV_Y_DEG,
            "pvs_fov_x": horizontal_fov_from_vertical(POINT60_FOV_Y_DEG, aspect),
            "pvs_back_offset": 0.0,
            "render_fov_y": POINT60_FOV_Y_DEG,
            "fov_y": POINT60_FOV_Y_DEG,
            "aspect": aspect,
            "width": width,
            "height": height,
            "point60_aspect_source": aspect_source,
        }
    )
    return row, (width, height, aspect), aspect_source


def build_point60_plan(
    dataset_dir: Path,
    representative_plan: Path,
    output_dir: Path,
    *,
    scene: str | None = None,
) -> dict[str, Any]:
    dataset_dir = dataset_dir.resolve()
    representative_plan = representative_plan.resolve()
    meta, _poses, test_indices, test_split_id = _load_test_indices(dataset_dir)
    representatives = read_jsonl(representative_plan)
    if len(representatives) != int(meta.get("poseCount", len(representatives))):
        raise ValueError(
            "representative plan row count does not match Pose CSR poseCount: "
            f"plan={len(representatives)} poseCount={meta.get('poseCount')}"
        )

    selected_rows: list[dict[str, Any]] = []
    groups: dict[tuple[int, int, float], list[dict[str, Any]]] = {}
    aspect_sources: set[str] = set()
    for source_index in test_indices.tolist():
        row, key, aspect_source = _point60_row(representatives[source_index], int(source_index))
        selected_rows.append(row)
        groups.setdefault(key, []).append(row)
        aspect_sources.add(aspect_source)

    output_dir = _prepare_output_dir(output_dir)
    group_summaries: list[dict[str, Any]] = []
    for group_id, (key, rows) in enumerate(sorted(groups.items()), start=1):
        width, height, aspect = key
        filename = f"point60_test_w{width}_h{height}_a{aspect_token(aspect)}.jsonl"
        output_path = output_dir / filename
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")
        group_summaries.append(
            {
                "groupId": group_id,
                "file": filename,
                "width": width,
                "height": height,
                "aspect": aspect,
                "poseCount": len(rows),
                "sourcePoseIndices": [int(row["pose_index"]) for row in rows],
            }
        )

    summary = {
        "schema": "geometry-shell-hzb-point60-gt-plan-v1",
        "scene": scene or meta.get("scene") or dataset_dir.name,
        "datasetDir": str(dataset_dir),
        "representativePlan": str(representative_plan),
        "sourcePoseCount": int(meta.get("poseCount", len(representatives))),
        "sourceSplit": "test",
        "sourceSplitId": test_split_id,
        "selectedPoseCount": len(selected_rows),
        "viewcellCount": len(selected_rows),
        "subposesPerViewcell": 1,
        "canonicalSubposeId": 0,
        "fovYDeg": POINT60_FOV_Y_DEG,
        "cameraSource": "representative_plan.camera_pos/camera_forward",
        "cameraSemantics": "real_camera_not_pose_csr_back_camera",
        "aspectSemantics": "preserve_representative_aspect_and_explicit_viewport_dimensions",
        "aspectSources": sorted(aspect_sources),
        "groupCount": len(group_summaries),
        "groups": group_summaries,
        "selectedPoseIndices": [int(row["pose_index"]) for row in selected_rows],
        "outputDir": str(output_dir),
        "rawGtHandoff": (
            "Run each plan JSONL with run_sampler.mjs --point60-gt into a separate raw directory; "
            "pass that directory to evaluate_geometry_shell_hzb.py --point-gt-raw-dir."
        ),
        "testRead": True,
    }
    (output_dir / "point60_gt_plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directional Pose CSR dataset directory.")
    parser.add_argument("--representative-plan", type=Path, required=True, help="One representative JSONL row per Pose CSR row.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory for viewport-grouped Point60 plans.")
    parser.add_argument("--scene", default=None, help="Optional scene label written to the summary.")
    parser.add_argument("--split", choices=("test",), default="test", help="Point60 plans are fixed to the Pose CSR test split.")
    args = parser.parse_args(argv)
    if args.split != "test":
        parser.error("Point60 GT plan generation is fixed to --split test")
    return args


def main() -> None:
    args = parse_args()
    summary = build_point60_plan(
        args.dataset_dir,
        args.representative_plan,
        args.output_dir,
        scene=args.scene,
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
