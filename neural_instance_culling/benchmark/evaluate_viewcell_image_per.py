#!/usr/bin/env python3
"""Viewcell PVS image-level evaluation using local component IDs.

This benchmark combines three views of the same PVS result:
- set metrics on the viewcell visible-instance union;
- NeuralPVS-like frustum-cell false-negative / false-positive rates;
- image-level Color-ID PER on real dense subpose cameras.

The image manifest is component-ID based.  The browser path binds component IDs
and visibility masks to each loaded InstancedMesh; ``--render-schema-only``
performs metadata/manifest validation without starting Chrome.  The older AABB
proxy renderer remains only as an explicit debugging fallback.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runtime_meta import bounds_min_max, load_runtime_meta  # noqa: E402
from common.threshold_selection import (
    select_weighted_precision_workpoint,
    weighted_precision_selection_rule,
)  # noqa: E402
from compute_color_id_per import compute_metrics  # noqa: E402
from instance_id_render_schema import (  # noqa: E402
    FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
    INSTANCE_ID_ENCODING,
    MODEL_INPUT_FOV_Y_DEG,
    INSTANCE_RENDER_MANIFEST_SCHEMA,
    PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
    PREDICTION_KEY_FIELD,
    RENDER_FOV_Y_DEG,
    build_instance_binding_preflight,
    validate_formal_instance_render_manifest,
    validate_instance_render_manifest,
)
from model_runners import load_runner, selected_default_specs, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


CATEGORY_NAMES = {
    0: "street_gap",
    1: "near_building",
    2: "plaza",
    3: "perimeter",
    4: "sky",
    5: "far",
    255: "unknown",
}


@dataclass
class SetMetrics:
    precision: float
    recall: float
    f1: float
    jaccard: float
    tp: int
    fp: int
    fn: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate viewcell PVS with component-ID image manifests.")
    parser.add_argument("--model-name", default="baseline_keep_all")
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help=(
            "Explicit model spec: "
            "name|kind|checkpoint|runtime_features|calibration_summary."
        ),
    )
    parser.add_argument("--runtime-meta", type=Path, default=Path("hkust-v3/assets/runtimeVisibilityMeta.json"))
    parser.add_argument(
        "--viewcell-dataset",
        type=Path,
        default=ROOT / "dataset/out/hkust_v3_viewcell_colorid_fov66_source",
        help="View-cell source directory containing centers, subposes and GT unions.",
    )
    parser.add_argument(
        "--pose-csr",
        type=Path,
        default=ROOT / "dataset/out/pose_csr_hkust_v3_spatial_fov66_v1",
        help="Aligned Pose CSR directory; its spatial split labels are the formal split source by default.",
    )
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "collector/out/hkust_rvc_viewcell_froxel_v1_1080p_rx9070_raw")
    parser.add_argument("--glb-root", type=Path, default=Path("hkust-v3/assets"))
    parser.add_argument("--glb-index", type=Path, default=Path("hkust-v3/assets/glbIndex.json"))
    parser.add_argument("--glb-points-meta", type=Path, default=ROOT / "dataset/out/glb_points_v3_meta.json")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["train", "val", "validation", "calibration", "test", "guard"],
        default="test",
    )
    parser.add_argument(
        "--split-source",
        choices=["pose_csr", "viewcell"],
        default="pose_csr",
        help="Use formal Pose CSR split labels, or the source dataset's exploratory labels.",
    )
    parser.add_argument(
        "--max-viewcells",
        type=int,
        default=0,
        help="Maximum view-cells to evaluate; 0 means the complete selected split.",
    )
    parser.add_argument(
        "--subposes-per-viewcell",
        type=int,
        default=0,
        help=(
            "subposes per view-cell; 0 selects every dense subpose, a positive "
            "value selects that many deterministic evenly spaced subposes, and "
            "negative values are invalid"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the subpose selection contract checks without loading model or scene assets",
    )
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-renderer", choices=["true_glb", "glb_aabb"], default="true_glb")
    parser.add_argument("--true-renderer-script", type=Path, default=ROOT / "benchmark/render_local_glb_color_id_browser.mjs")
    parser.add_argument("--chrome-exe", type=Path, default=None)
    parser.add_argument("--render-timeout-sec", type=int, default=1800)
    parser.add_argument(
        "--render-schema-only",
        action="store_true",
        help="Validate and write the component-ID render manifest without starting Chrome.",
    )
    parser.add_argument(
        "--formal-image-evaluation",
        action="store_true",
        help="Require the complete hardware-GPU component-ID image protocol and emit a formal-ready result.",
    )
    parser.add_argument(
        "--require-hardware-gpu",
        action="store_true",
        help="Require the renderer to use a hardware GPU and capture host GPU evidence.",
    )
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--threshold-policy",
        choices=["weighted_precision", "precision80", "runner", "best_f1", "high_recall"],
        default="weighted_precision",
    )
    parser.add_argument("--target-precision", type=float, default=0.80)
    parser.add_argument("--target-weighted-recall", type=float, default=0.99)
    parser.add_argument(
        "--minimum-pose-recall",
        type=float,
        default=None,
        help="Optional ordinary pose-recall floor for calibrated image workpoints.",
    )
    parser.add_argument("--froxel-grid", default="64,36,64")
    parser.add_argument("--froxel-near", type=float, default=0.05)
    parser.add_argument("--froxel-far", type=float, default=8000.0)
    parser.add_argument("--render-near", type=float, default=0.05)
    parser.add_argument("--save-id-buffers", action="store_true")
    parser.add_argument("--preview-samples", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--skip-raw-subpose-gt", action="store_true", help="Skip optional raw subpose labels; this does not create a formal image reference.")
    return parser.parse_args()


def subpose_selection_self_test() -> None:
    """Check the public zero/all and deterministic sampling contract."""
    all_indices = ViewcellDataset.select_subpose_indices(10, 43, 0)
    if all_indices.size != 33 or int(all_indices[0]) != 10 or int(all_indices[-1]) != 42:
        raise AssertionError("count=0 must select all dense subposes")
    sampled = ViewcellDataset.select_subpose_indices(10, 43, 4)
    if sampled.tolist() != [10, 20, 31, 42]:
        raise AssertionError(f"unexpected deterministic subpose sample: {sampled.tolist()}")
    clipped = ViewcellDataset.select_subpose_indices(10, 13, 99)
    if clipped.tolist() != [10, 11, 12]:
        raise AssertionError("count greater than the available subposes must select all")
    try:
        ViewcellDataset.select_subpose_indices(10, 43, -1)
    except ValueError:
        pass
    else:
        raise AssertionError("negative subpose count was accepted")
    print("view-cell dense subpose selection self-test: PASS")


def parse_model_spec(value: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) == 5:
        name, kind, checkpoint, runtime_features, eval_summary = parts
    else:
        raise ValueError(
            "--model-spec must use "
            "name|kind|checkpoint|runtime_features|calibration_summary"
        )
    if any(not part for part in parts):
        raise ValueError("--model-spec contains an empty field")
    return name, {
        "kind": kind,
        "checkpoint": checkpoint,
        "runtime_features": runtime_features,
        "eval_summary": eval_summary,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_flat_output_dir(path: Path, suffixes: tuple[str, ...]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() and (not suffixes or child.suffix.lower() in suffixes):
            child.unlink()


def safe_prf(tp: int, fp: int, fn: int) -> SetMetrics:
    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    return SetMetrics(precision, recall, f1, jaccard, int(tp), int(fp), int(fn))


def set_metrics(pred_ids: np.ndarray, gt_ids: np.ndarray) -> SetMetrics:
    pred = np.unique(pred_ids.astype(np.uint32, copy=False))
    gt = np.unique(gt_ids.astype(np.uint32, copy=False))
    tp = int(np.intersect1d(pred, gt, assume_unique=True).size)
    fp = int(max(0, pred.size - tp))
    fn = int(max(0, gt.size - tp))
    return safe_prf(tp, fp, fn)


def weighted_recall(pred_ids: np.ndarray, gt_ids: np.ndarray, gt_weights: np.ndarray) -> float:
    if gt_ids.size == 0:
        return 1.0
    pred_set = set(int(v) for v in pred_ids.tolist())
    total = float(np.asarray(gt_weights, dtype=np.float64).sum())
    if total <= 1e-9:
        return float(len(pred_set.intersection(int(v) for v in gt_ids.tolist())) / max(1, gt_ids.size))
    hit = 0.0
    for cid, weight in zip(gt_ids.tolist(), gt_weights.tolist()):
        if int(cid) in pred_set:
            hit += float(weight)
    return float(hit / total)


class ViewcellDataset:
    REQUIRED_FILES = (
        "viewcell_ids.bin",
        "viewcell_centers.bin",
        "viewcell_forwards.bin",
        "viewcell_params.bin",
        "viewcell_category_ids.bin",
        "viewcell_split_ids.bin",
        "visible_offsets.bin",
        "visible_ids.bin",
        "visible_weights.bin",
        "subpose_offsets.bin",
        "subpose_pose_indices.bin",
        "subpose_camera_pos.bin",
        "subpose_camera_forward.bin",
        "subpose_params.bin",
    )

    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.meta = read_json(dataset_dir / "dataset_meta.json")
        missing = [name for name in self.REQUIRED_FILES if not (dataset_dir / name).is_file()]
        if missing:
            raise ValueError(
                "--viewcell-dataset must point to a view-cell source dataset; "
                f"{dataset_dir} is missing {', '.join(missing)}. "
                "A Pose CSR-only directory is not a substitute; use the matching "
                "*_viewcell_*_source directory for view-cell/subpose metadata."
            )
        declared_count = int(self.meta.get("viewcellCount", self.meta.get("poseCount", 0)))
        if declared_count <= 0:
            raise ValueError(f"{dataset_dir}/dataset_meta.json has no positive viewcellCount/poseCount")
        self.viewcell_ids = np.memmap(dataset_dir / "viewcell_ids.bin", dtype="<u4", mode="r")
        self.centers = np.memmap(dataset_dir / "viewcell_centers.bin", dtype="<f4", mode="r").reshape(-1, 3)
        self.forwards = np.memmap(dataset_dir / "viewcell_forwards.bin", dtype="<f4", mode="r").reshape(-1, 3)
        self.params = np.memmap(dataset_dir / "viewcell_params.bin", dtype="<f4", mode="r").reshape(-1, 8)
        self.category_ids = np.memmap(dataset_dir / "viewcell_category_ids.bin", dtype=np.uint8, mode="r")
        self.split_ids = np.memmap(dataset_dir / "viewcell_split_ids.bin", dtype=np.uint8, mode="r")
        self.visible_offsets = np.memmap(dataset_dir / "visible_offsets.bin", dtype="<u8", mode="r")
        self.visible_ids = np.memmap(dataset_dir / "visible_ids.bin", dtype="<u4", mode="r")
        self.visible_weights = np.memmap(dataset_dir / "visible_weights.bin", dtype="<f4", mode="r")
        self.subpose_offsets = np.memmap(dataset_dir / "subpose_offsets.bin", dtype="<u8", mode="r")
        self.subpose_pose_indices = np.memmap(dataset_dir / "subpose_pose_indices.bin", dtype="<u4", mode="r")
        self.subpose_pos = np.memmap(dataset_dir / "subpose_camera_pos.bin", dtype="<f4", mode="r").reshape(-1, 3)
        self.subpose_forward = np.memmap(dataset_dir / "subpose_camera_forward.bin", dtype="<f4", mode="r").reshape(-1, 3)
        self.subpose_params = np.memmap(dataset_dir / "subpose_params.bin", dtype="<f4", mode="r").reshape(-1, 4)
        self.split_lookup = {name: int(idx) for name, idx in self.meta.get("splitIds", {}).items()}
        self.split_alignment: dict[str, Any] = {
            "source": "viewcell",
            "validated": False,
        }
        if self.viewcell_ids.shape[0] != declared_count:
            raise ValueError(
                f"{dataset_dir}/viewcell_ids.bin contains {self.viewcell_ids.shape[0]} rows, "
                f"but dataset_meta.json declares {declared_count} view-cells"
            )
        if self.split_ids.shape[0] != declared_count or self.visible_offsets.shape[0] != declared_count + 1:
            raise ValueError(
                f"{dataset_dir} has inconsistent view-cell split/visible offset lengths; "
                "refusing to align image samples by position"
            )

    def validate_pose_csr_alignment(self, pose_dataset: PoseCSRDataset) -> dict[str, Any]:
        """Validate that Pose CSR rows and view-cell rows describe the same poses.

        The formal spatial labels are stored in Pose CSR, while subpose metadata and
        view-cell GT unions remain in this source directory.  Position-only row
        matching is unsafe, so validate the forward vector and the geometric relation
        ``view_cell_center = back_camera_position + offset * forward`` for every row.
        """
        pose_count = int(pose_dataset.poses.shape[0])
        if pose_count != int(self.viewcell_ids.shape[0]):
            raise ValueError(
                "Pose CSR/view-cell row count mismatch: "
                f"poseCount={pose_count}, viewcellCount={self.viewcell_ids.shape[0]}"
            )
        if "camera_forward" not in pose_dataset.poses.dtype.names:
            raise ValueError("Formal Pose CSR split source must contain camera_forward per row")

        source_forward = np.asarray(self.forwards, dtype=np.float64)
        pose_forward = np.asarray(pose_dataset.poses["camera_forward"], dtype=np.float64)
        forward_max_abs_error = float(np.max(np.abs(source_forward - pose_forward))) if pose_count else 0.0
        if not np.allclose(source_forward, pose_forward, rtol=1e-5, atol=2e-4):
            raise ValueError(
                "Pose CSR/view-cell row alignment failed: camera_forward differs; "
                f"maxAbsError={forward_max_abs_error:.9g}"
            )

        source_center = np.asarray(self.centers, dtype=np.float64)
        camera_world = np.asarray(pose_dataset.poses["camera_world"], dtype=np.float64)
        forward_norm = pose_forward / np.maximum(np.linalg.norm(pose_forward, axis=1, keepdims=True), 1e-12)
        center_delta = source_center - camera_world
        offsets = np.sum(center_delta * forward_norm, axis=1)
        center_residual = np.linalg.norm(center_delta - offsets[:, None] * forward_norm, axis=1)
        max_center_residual = float(np.max(center_residual)) if pose_count else 0.0
        min_back_offset = float(np.min(offsets)) if pose_count else 0.0
        camera_semantics = str(pose_dataset.meta.get("cameraSemantics", "")).lower()
        canonical_center_semantics = (
            "canonical viewcell" in camera_semantics
            or "canonical plan center" in camera_semantics
        )
        if canonical_center_semantics:
            # The formal spatial CSR keeps the canonical view-cell center as
            # camera_world.  It is still the same row-wise pose as the source
            # view-cell, but it does not apply the legacy back-camera offset.
            if max_center_residual > 1e-3 or min_back_offset < -1e-3:
                raise ValueError(
                    "Pose CSR/view-cell row alignment failed: canonical view-cell centers "
                    "do not match camera_world; "
                    f"maxCenterResidual={max_center_residual:.9g}, minCenterOffset={min_back_offset:.9g}"
                )
            center_relation = "source view-cell center equals canonical camera_world"
        else:
            if min_back_offset <= 0.0 or max_center_residual > 1e-3:
                raise ValueError(
                    "Pose CSR/view-cell row alignment failed: view-cell centers are not the "
                    "forward-offset centers of the back cameras; "
                    f"minBackOffset={min_back_offset:.9g}, maxCenterResidual={max_center_residual:.9g}"
                )
            center_relation = "source view-cell center is a positive forward offset from camera_world"

        self.split_alignment = {
            "source": "pose_csr",
            "validated": True,
            "poseCount": pose_count,
            "forwardMaxAbsError": forward_max_abs_error,
            "minBackOffset": min_back_offset,
            "maxCenterResidual": max_center_residual,
            "centerRelation": center_relation,
            "cameraSemantics": pose_dataset.meta.get("cameraSemantics"),
            "poseCsrDataset": str(pose_dataset.dataset_dir),
        }
        return dict(self.split_alignment)

    @staticmethod
    def _normalize_split_name(split: str, available: dict[str, int]) -> str:
        aliases = {"val": "validation", "validation": "val"}
        if split in available:
            return split
        alias = aliases.get(split)
        if alias is not None and alias in available:
            return alias
        raise KeyError(f"Split '{split}' not found; available splits: {sorted(available)}")

    def split_indices(
        self,
        split: str,
        limit: int,
        *,
        split_source: str = "viewcell",
        pose_dataset: PoseCSRDataset | None = None,
    ) -> np.ndarray:
        if split_source == "pose_csr":
            if pose_dataset is None:
                raise ValueError("pose_dataset is required when split_source='pose_csr'")
            self.validate_pose_csr_alignment(pose_dataset)
            split_ids = {str(name): int(value) for name, value in pose_dataset.split_ids.items()}
            split_name = self._normalize_split_name(split, split_ids)
            split_id = split_ids[split_name]
            indices = np.flatnonzero(pose_dataset.poses["split"] == split_id).astype(np.int64)
        else:
            split_name = self._normalize_split_name(split, self.split_lookup)
            split_id = self.split_lookup[split_name]
            indices = np.flatnonzero(self.split_ids == split_id).astype(np.int64)
        if limit > 0:
            indices = indices[:limit]
        return indices

    def visible_slice(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.visible_offsets[row])
        end = int(self.visible_offsets[row + 1])
        return np.asarray(self.visible_ids[start:end], dtype=np.uint32), np.asarray(self.visible_weights[start:end], dtype=np.float32)

    @staticmethod
    def select_subpose_indices(start: int, end: int, count: int) -> np.ndarray:
        """Select dense subposes with an explicit zero-means-all contract."""
        if int(count) < 0:
            raise ValueError("subposes-per-viewcell must be non-negative; 0 means all subposes")
        start = int(start)
        end = int(end)
        if end <= start:
            return np.zeros((0,), dtype=np.int64)
        total = end - start
        if int(count) == 0 or int(count) >= total:
            return np.arange(start, end, dtype=np.int64)
        local = np.linspace(0, total - 1, num=int(count), dtype=np.int64)
        return (start + local).astype(np.int64)

    def selected_subposes(self, row: int, count: int) -> np.ndarray:
        start = int(self.subpose_offsets[row])
        end = int(self.subpose_offsets[row + 1])
        return self.select_subpose_indices(start, end, count)


def iter_jsonl(path_or_dir: Path):
    files = sorted(path_or_dir.glob("*.jsonl")) if path_or_dir.is_dir() else [path_or_dir]
    for path in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def load_raw_visibility(raw_dir: Path, needed_pose_indices: set[int]) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if not raw_dir.exists() or not needed_pose_indices:
        return {}, {"enabled": False, "reason": "raw dir missing or no requested subposes", "rawDir": str(raw_dir)}
    remaining = set(int(v) for v in needed_pose_indices)
    found: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    rows_scanned = 0
    for row in iter_jsonl(raw_dir):
        rows_scanned += 1
        pose_index = row.get("pose_index")
        if pose_index is None:
            continue
        pose_index = int(pose_index)
        if pose_index not in remaining:
            continue
        ids = np.asarray(row.get("visible_component_ids") or [], dtype=np.uint32)
        weights_raw = row.get("component_weights") or []
        weights = np.ones((ids.size,), dtype=np.float32)
        if weights_raw:
            weights[: min(ids.size, len(weights_raw))] = np.asarray(weights_raw[: ids.size], dtype=np.float32)
        found[pose_index] = (ids, weights)
        remaining.remove(pose_index)
        if not remaining:
            break
    return found, {
        "enabled": True,
        "rawDir": str(raw_dir),
        "rowsScanned": int(rows_scanned),
        "requestedSubposes": int(len(needed_pose_indices)),
        "matchedSubposes": int(len(found)),
        "missingSubposes": int(len(remaining)),
    }


def normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.astype(np.float32, copy=True)
    return (v / n).astype(np.float32)


def camera_basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f = normalize(np.asarray(forward, dtype=np.float32), np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_ref))) > 0.98:
        up_ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(f, up_ref), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
    return f, right, up


def aabb_corners(aabb: np.ndarray) -> np.ndarray:
    mn = aabb[:3]
    mx = aabb[3:]
    return np.asarray(
        [
            [mn[0], mn[1], mn[2]],
            [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]],
            [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]],
            [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]],
            [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )


def load_glb_index(glb_index_path: Path, glb_root: Path) -> tuple[dict[int, Path], dict[str, Any]]:
    data = read_json(glb_index_path)
    root = glb_root.resolve()
    paths: dict[int, Path] = {}
    missing = 0
    outside = 0
    for entry in data.get("entries", []):
        gid = int(entry["globalId"])
        candidate = (glb_root / entry["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            outside += 1
            continue
        if not candidate.exists():
            missing += 1
        paths[gid] = candidate
    return paths, {
        "schema": "local-glb-index-check-v1",
        "glbRoot": str(root),
        "glbIndex": str(glb_index_path),
        "entryCount": int(len(data.get("entries", []))),
        "localPathCount": int(len(paths)),
        "missingLocalFiles": int(missing),
        "outsideRootFiles": int(outside),
        "remoteAccessAllowed": False,
    }


def load_glb_aabbs(runtime_meta: dict[str, Any], glb_points_meta_path: Path | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    num_glbs = int(runtime_meta.get("globalGlbCount", 0))
    if num_glbs <= 0:
        num_glbs = max(int(r["globalGlbId"]) for r in runtime_meta.get("componentRecords", [])) + 1
    mins = np.full((num_glbs, 3), np.inf, dtype=np.float32)
    maxs = np.full((num_glbs, 3), -np.inf, dtype=np.float32)
    source = "runtime component bounds aggregated per GLB"
    component_updates = 0
    for record in runtime_meta.get("componentRecords", []):
        gid = int(record["globalGlbId"])
        if gid < 0 or gid >= num_glbs:
            continue
        mn, mx = bounds_min_max(record["bounds"])
        mins[gid] = np.minimum(mins[gid], mn)
        maxs[gid] = np.maximum(maxs[gid], mx)
        component_updates += 1
    glb_meta_updates = 0
    if glb_points_meta_path and glb_points_meta_path.exists():
        meta = read_json(glb_points_meta_path)
        for entry in meta.get("entries", []):
            gid = int(entry.get("globalId", -1))
            bounds = entry.get("bounds")
            if gid < 0 or gid >= num_glbs or not bounds:
                continue
            mins[gid] = np.asarray(bounds["min"], dtype=np.float32)
            maxs[gid] = np.asarray(bounds["max"], dtype=np.float32)
            glb_meta_updates += 1
        source = "glb_points_v3 decoded local GLB bounds with runtime aggregation fallback"
    valid = np.isfinite(mins).all(axis=1) & np.isfinite(maxs).all(axis=1)
    aabbs = np.zeros((num_glbs, 6), dtype=np.float32)
    aabbs[:, :3] = np.where(valid[:, None], mins, 0.0)
    aabbs[:, 3:] = np.where(valid[:, None], maxs, 0.0)
    return aabbs, {
        "source": source,
        "numGlbs": int(num_glbs),
        "validGlbBounds": int(valid.sum()),
        "componentBoundUpdates": int(component_updates),
        "decodedGlbBoundUpdates": int(glb_meta_updates),
        "glbPointsMeta": str(glb_points_meta_path) if glb_points_meta_path else None,
    }


class GlbAabbColorIdRenderer:
    def __init__(self, glb_aabbs: np.ndarray, width: int, height: int, near: float):
        self.glb_aabbs = glb_aabbs.astype(np.float32, copy=False)
        self.width = int(width)
        self.height = int(height)
        self.near = float(near)

    def render(self, glb_ids: np.ndarray, camera_pos: np.ndarray, forward: np.ndarray, fov_y_deg: float, aspect: float) -> np.ndarray:
        ids = np.unique(glb_ids.astype(np.int64, copy=False))
        color = np.zeros((self.height, self.width), dtype=np.uint32)
        depth = np.full((self.height, self.width), np.inf, dtype=np.float32)
        f, right, up = camera_basis(forward)
        tan_y = math.tan(math.radians(float(fov_y_deg)) * 0.5)
        tan_x = tan_y * max(float(aspect), 1e-4)
        origin = np.asarray(camera_pos, dtype=np.float32)
        for gid_raw in ids.tolist():
            gid = int(gid_raw)
            if gid < 0 or gid >= self.glb_aabbs.shape[0]:
                continue
            aabb = self.glb_aabbs[gid]
            if not np.any(aabb[3:] > aabb[:3]):
                continue
            corners = aabb_corners(aabb)
            d = corners - origin[None, :]
            z = d @ f
            front = z > self.near
            if not bool(front.any()):
                continue
            x = d @ right
            y = d @ up
            z_safe = np.maximum(z[front], self.near)
            ndc_x = x[front] / (z_safe * tan_x)
            ndc_y = y[front] / (z_safe * tan_y)
            if ndc_x.max() < -1.0 or ndc_x.min() > 1.0 or ndc_y.max() < -1.0 or ndc_y.min() > 1.0:
                continue
            px0 = int(np.floor(np.clip((ndc_x.min() * 0.5 + 0.5) * self.width, 0, self.width - 1)))
            px1 = int(np.ceil(np.clip((ndc_x.max() * 0.5 + 0.5) * self.width, 0, self.width - 1)))
            py0 = int(np.floor(np.clip((0.5 - ndc_y.max() * 0.5) * self.height, 0, self.height - 1)))
            py1 = int(np.ceil(np.clip((0.5 - ndc_y.min() * 0.5) * self.height, 0, self.height - 1)))
            if px1 < px0 or py1 < py0:
                continue
            z_value = float(np.min(z[front]))
            tile_depth = depth[py0 : py1 + 1, px0 : px1 + 1]
            mask = z_value < tile_depth
            if not bool(mask.any()):
                continue
            tile_color = color[py0 : py1 + 1, px0 : px1 + 1]
            tile_color[mask] = np.uint32(gid + 1)
            tile_depth[mask] = np.float32(z_value)
        return color


def hash_color(ids: np.ndarray) -> np.ndarray:
    v = ids.astype(np.uint32)
    r = ((v * np.uint32(37) + np.uint32(17)) & np.uint32(255)).astype(np.uint8)
    g = ((v * np.uint32(67) + np.uint32(29)) & np.uint32(255)).astype(np.uint8)
    b = ((v * np.uint32(97) + np.uint32(53)) & np.uint32(255)).astype(np.uint8)
    out = np.stack([r, g, b], axis=-1)
    out[ids == 0] = 0
    return out


def diff_color(diff: np.ndarray) -> np.ndarray:
    out = np.zeros((*diff.shape, 3), dtype=np.uint8)
    out[diff == 1] = np.asarray([220, 40, 40], dtype=np.uint8)
    out[diff == 2] = np.asarray([240, 180, 30], dtype=np.uint8)
    out[diff == 3] = np.asarray([40, 110, 230], dtype=np.uint8)
    return out


def save_preview(path: Path, reference: np.ndarray, test: np.ndarray, diff: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_rgb = hash_color(reference)
    test_rgb = hash_color(test)
    diff_rgb = diff_color(diff)
    strip = np.concatenate([ref_rgb, test_rgb, diff_rgb], axis=1)
    Image.fromarray(strip, mode="RGB").save(path)


def _capture_gpu_snapshot(output_dir: Path, phase: str) -> dict[str, Any]:
    """Capture host-side GPU observations for one browser render window."""

    output_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidiaSmi": (["nvidia-smi"], output_dir / f"nvidia_smi_{phase}.txt"),
        "nvidiaSmiPmon": (
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            output_dir / f"nvidia_smi_pmon_{phase}.txt",
        ),
    }
    result: dict[str, Any] = {"phase": phase, "complete": True, "observations": {}}
    for name, (command, path) in commands.items():
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
                check=False,
            )
            path.write_text(
                (completed.stdout or "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            available = completed.returncode == 0 and path.stat().st_size > 0
            result["observations"][name] = {
                "command": command,
                "path": str(path.resolve()),
                "returnCode": int(completed.returncode),
                "available": bool(available),
            }
            result["complete"] = bool(result["complete"] and available)
        except OSError as exc:
            path.write_text(str(exc), encoding="utf-8")
            result["observations"][name] = {
                "command": command,
                "path": str(path.resolve()),
                "returnCode": None,
                "available": False,
                "error": repr(exc),
            }
            result["complete"] = False
    return result


def reference_error_contributors(reference: np.ndarray, test: np.ndarray) -> dict[int, int]:
    """Count reference GLB pixels whose final test ID is missing or different."""
    mask = (reference != 0) & (test != reference)
    if not bool(mask.any()):
        return {}
    ids, counts = np.unique(reference[mask], return_counts=True)
    return {int(encoded_id) - 1: int(count) for encoded_id, count in zip(ids.tolist(), counts.tolist()) if int(encoded_id) > 0}


def glbs_for_components(component_ids: np.ndarray, instance_to_glb: np.ndarray) -> np.ndarray:
    if component_ids.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    valid = component_ids[component_ids < instance_to_glb.shape[0]]
    if valid.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    return np.unique(instance_to_glb[valid.astype(np.int64, copy=False)].astype(np.uint32, copy=False))


def check_local_glbs(glb_ids: np.ndarray, glb_paths: dict[int, Path]) -> list[int]:
    missing = []
    for gid in np.unique(glb_ids.astype(np.int64, copy=False)).tolist():
        path = glb_paths.get(int(gid))
        if path is None or not path.exists():
            missing.append(int(gid))
    return missing


def mark_frustum_cells(
    component_ids: np.ndarray,
    world_aabbs: np.ndarray,
    camera_world: np.ndarray,
    camera_view: np.ndarray,
    grid: tuple[int, int, int],
    near: float,
    far: float,
) -> np.ndarray:
    gw, gh, gd = grid
    cells = np.zeros((gd, gh, gw), dtype=bool)
    if component_ids.size == 0:
        return cells
    f, right, up = camera_basis(camera_view[:3])
    tan_x = max(float(camera_view[3]), 1e-4)
    tan_y = max(float(camera_view[4]), 1e-4)
    origin = np.asarray(camera_world, dtype=np.float32)
    log_den = max(1e-6, math.log(max(far, near * 2.0) / max(near, 1e-6)))
    for cid_raw in np.unique(component_ids.astype(np.int64, copy=False)).tolist():
        cid = int(cid_raw)
        if cid < 0 or cid >= world_aabbs.shape[0]:
            continue
        aabb = world_aabbs[cid]
        corners = aabb_corners(aabb)
        d = corners - origin[None, :]
        z = d @ f
        front = z > near
        if not bool(front.any()):
            continue
        z_safe = np.maximum(z[front], near)
        ndc_x = (d[front] @ right) / (z_safe * tan_x)
        ndc_y = (d[front] @ up) / (z_safe * tan_y)
        if ndc_x.max() < -1.0 or ndc_x.min() > 1.0 or ndc_y.max() < -1.0 or ndc_y.min() > 1.0:
            continue
        x0 = int(np.floor(np.clip((ndc_x.min() * 0.5 + 0.5) * gw, 0, gw - 1)))
        x1 = int(np.ceil(np.clip((ndc_x.max() * 0.5 + 0.5) * gw, 0, gw - 1)))
        y0 = int(np.floor(np.clip((0.5 - ndc_y.max() * 0.5) * gh, 0, gh - 1)))
        y1 = int(np.ceil(np.clip((0.5 - ndc_y.min() * 0.5) * gh, 0, gh - 1)))
        d0 = int(np.floor(np.clip(math.log(max(float(z_safe.min()), near) / near) / log_den * gd, 0, gd - 1)))
        d1 = int(np.ceil(np.clip(math.log(max(float(z_safe.max()), near) / near) / log_den * gd, 0, gd - 1)))
        cells[d0 : d1 + 1, y0 : y1 + 1, x0 : x1 + 1] = True
    return cells


def froxel_metrics(gt_ids: np.ndarray, pred_ids: np.ndarray, world_aabbs: np.ndarray, camera_world: np.ndarray, camera_view: np.ndarray, grid: tuple[int, int, int], near: float, far: float) -> dict[str, Any]:
    gt = mark_frustum_cells(gt_ids, world_aabbs, camera_world, camera_view, grid, near, far)
    pred = mark_frustum_cells(pred_ids, world_aabbs, camera_world, camera_view, grid, near, far)
    fn = np.logical_and(gt, ~pred)
    fp = np.logical_and(~gt, pred)
    tp = np.logical_and(gt, pred)
    gt_count = int(gt.sum())
    pred_count = int(pred.sum())
    grid_count = int(gt.size)
    return {
        "frustumCellFNR": float(fn.sum() / max(1, gt_count)),
        "frustumCellFPR": float(fp.sum() / max(1, grid_count - gt_count)),
        "frustumCellIoU": float(tp.sum() / max(1, np.logical_or(gt, pred).sum())),
        "gtCellCount": gt_count,
        "predCellCount": pred_count,
        "fnCellCount": int(fn.sum()),
        "fpCellCount": int(fp.sum()),
        "gridCellCount": grid_count,
    }


def resolve_threshold(args: argparse.Namespace, spec: dict[str, str], runner) -> tuple[float, dict[str, Any]]:
    if args.threshold is not None:
        return float(args.threshold), {"source": "command line", "threshold": float(args.threshold)}
    eval_summary = spec.get("eval_summary")
    if not eval_summary or not Path(eval_summary).exists():
        if args.threshold_policy == "weighted_precision":
            raise RuntimeError(
                "The default weighted-precision policy requires an eval_summary with thresholdRows; "
                "refusing to fall back to an uncalibrated runner threshold."
            )
        return float(runner.threshold), {"source": "runner fallback", "threshold": float(runner.threshold)}
    data = read_json(Path(eval_summary))
    # Keep the resolver compatible with older programmatic callers that build
    # a small SimpleNamespace instead of the current argparse namespace.
    minimum_pose_recall = getattr(args, "minimum_pose_recall", None)

    def validate_frozen_workpoint(workpoint: Any, threshold: float) -> tuple[float, dict[str, Any]]:
        if minimum_pose_recall is not None:
            if not isinstance(workpoint, dict):
                raise RuntimeError(
                    f"{eval_summary} does not record the calibration workpoint needed for the "
                    f"pose-recall floor {float(minimum_pose_recall):.3f}."
                )
            observed = float(workpoint.get("pose_recall", -1.0))
            if observed < float(minimum_pose_recall):
                raise RuntimeError(
                    f"{eval_summary} frozen threshold {threshold} has pose_recall={observed:.6f}, "
                    f"below required {float(minimum_pose_recall):.3f}."
                )
        test_evaluation_count = int(data.get("testEvaluationCount", 0))
        return threshold, {
            "source": (
                "pre-test calibration summary; no threshold scan"
                if test_evaluation_count == 0
                else "calibration summary was read after a test evaluation"
            ),
            "threshold": threshold,
            "protocol": data.get("protocol"),
            "testEvaluationCount": test_evaluation_count,
            "selectionSplit": "calibration",
            "testRead": bool(test_evaluation_count),
        }

    if data.get("schema") == "pvs-bounded-relation-prior-instance-calibrated-calibration-summary-v4":
        if data.get("testRead") is not False:
            raise RuntimeError(f"{eval_summary} must be a test-free calibration summary.")
        best_safe = data.get("bestSafe")
        selection = best_safe.get("selection") if isinstance(best_safe, dict) else None
        if not isinstance(best_safe, dict) or best_safe.get("safe") is not True or not isinstance(selection, dict):
            raise RuntimeError(f"{eval_summary} has no frozen safe calibration workpoint.")
        threshold = float(best_safe.get("threshold", selection.get("threshold", float("nan"))))
        weighted_recall = float(selection.get("agg_weighted_recall", selection.get("aggregateWeightedRecall", -1.0)))
        weighted_lcb = float(selection.get("aggregateWeightedRecallLowerConfidenceBound", -1.0))
        target = float(data.get("weightedRecallFloor", args.target_weighted_recall))
        lcb_target = float(data.get("weightedRecallLowerConfidenceBoundFloor", target))
        if (
            not np.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or weighted_recall <= target
            or weighted_lcb <= lcb_target
        ):
            raise RuntimeError(f"{eval_summary} does not contain a valid strict weighted-recall workpoint.")
        return threshold, {
            "source": "current V4 bestSafe calibration workpoint; no threshold scan",
            "threshold": threshold,
            "workpoint": selection,
            "selectionSplit": "calibration",
            "testRead": False,
            "testEvaluationCount": 0,
        }

    if data.get("protocol") == "calibration_ready_pre_test":
        if int(data.get("testEvaluationCount", 0)) != 0:
            raise RuntimeError(
                f"{eval_summary} is marked pre-test but records testEvaluationCount={data.get('testEvaluationCount')}."
            )
        frozen = data.get("frozenThreshold")
        if frozen is None or not np.isfinite(float(frozen)) or not 0.0 <= float(frozen) <= 1.0:
            raise RuntimeError(f"{eval_summary} has no valid calibration frozenThreshold.")
        return validate_frozen_workpoint(data.get("calibration", {}).get("selected"), float(frozen))
    if data.get("protocol") == "frozen_calibration_one_shot_test":
        if int(data.get("testEvaluationCount", 0)) != 1:
            raise RuntimeError(
                f"{eval_summary} is marked frozen but does not record exactly one test evaluation."
            )
        frozen = data.get("frozenThreshold")
        if frozen is None or not np.isfinite(float(frozen)) or not 0.0 <= float(frozen) <= 1.0:
            raise RuntimeError(f"{eval_summary} has no valid frozenThreshold.")
        workpoint = data.get("calibration", {}).get("selected")
        threshold, resolved = validate_frozen_workpoint(workpoint, float(frozen))
        resolved["source"] = "frozen calibration summary; no threshold scan"
        resolved["testEvaluationCount"] = int(data.get("testEvaluationCount"))
        return threshold, resolved
    rows = data.get("thresholdRows") or []
    if args.threshold_policy == "runner":
        return float(runner.threshold), {"source": "runner eval_summary best", "threshold": float(runner.threshold)}
    if args.threshold_policy in {"weighted_precision", "high_recall"}:
        workpoint = select_weighted_precision_workpoint(
            rows,
            args.target_weighted_recall,
            minimum_pose_recall=minimum_pose_recall,
        )
        if workpoint is None:
            raise RuntimeError(
                "No threshold satisfies the strict weighted-recall rule "
                f"pose_weighted_recall > {float(args.target_weighted_recall):.3f}."
            )
        threshold = float(workpoint.get("threshold", runner.threshold))
        return threshold, {
            "source": weighted_precision_selection_rule(
                args.target_weighted_recall,
                minimum_pose_recall=minimum_pose_recall,
            ),
            "threshold": threshold,
            "workpoint": workpoint,
            "selectionSplit": "calibration",
            "testRead": False,
        }
    if args.threshold_policy == "best_f1":
        workpoint = data.get("workpoints", {}).get("bestF1") or (max(rows, key=lambda r: r.get("pose_f1", 0.0)) if rows else {})
        threshold = float(workpoint.get("threshold", runner.threshold))
        return threshold, {"source": "eval_summary best F1 workpoint", "threshold": threshold, "workpoint": workpoint}
    valid = [r for r in rows if float(r.get("pose_precision", 0.0)) >= float(args.target_precision)]
    if valid:
        selected = max(valid, key=lambda r: (float(r.get("pose_recall", 0.0)), float(r.get("pose_precision", 0.0)), float(r.get("pose_f1", 0.0))))
        threshold = float(selected.get("threshold", runner.threshold))
        return threshold, {
            "source": f"highest recall with pose precision >= {args.target_precision:.2f}",
            "threshold": threshold,
            "workpoint": selected,
        }
    threshold = float(data.get("workpoints", {}).get("bestF1", {}).get("threshold", runner.threshold))
    return threshold, {
        "source": f"no eval threshold reached precision >= {args.target_precision:.2f}; fallback to best F1/runner",
        "threshold": threshold,
    }


def metric_dict(prefix: str, metrics: SetMetrics) -> dict[str, float | int]:
    return {
        f"{prefix}Precision": metrics.precision,
        f"{prefix}Recall": metrics.recall,
        f"{prefix}F1": metrics.f1,
        f"{prefix}Jaccard": metrics.jaccard,
        f"{prefix}TP": metrics.tp,
        f"{prefix}FP": metrics.fp,
        f"{prefix}FN": metrics.fn,
    }


def summarize_set_acc(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    count = max(1, len(rows))
    tp = int(sum(int(r[f"{prefix}TP"]) for r in rows))
    fp = int(sum(int(r[f"{prefix}FP"]) for r in rows))
    fn = int(sum(int(r[f"{prefix}FN"]) for r in rows))
    agg = safe_prf(tp, fp, fn)
    return {
        "posePrecision": float(np.mean([r[f"{prefix}Precision"] for r in rows])) if rows else 0.0,
        "poseRecall": float(np.mean([r[f"{prefix}Recall"] for r in rows])) if rows else 0.0,
        "poseF1": float(np.mean([r[f"{prefix}F1"] for r in rows])) if rows else 0.0,
        "poseJaccard": float(np.mean([r[f"{prefix}Jaccard"] for r in rows])) if rows else 0.0,
        "aggPrecision": agg.precision,
        "aggRecall": agg.recall,
        "aggF1": agg.f1,
        "aggJaccard": agg.jaccard,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "count": count,
    }


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    s = summary
    c = s["componentSetMetrics"]
    g = s["glbSetMetrics"]
    f = s["frustumCellMetrics"]
    i = s["imageMetrics"]
    lines = [
        "# Viewcell Image-Level PVS Evaluation",
        "",
        f"- Created: `{s['created']}`",
        f"- Model: `{s['modelName']}`",
        f"- Threshold: `{s['threshold']:.6f}` ({s['thresholdSelection']['source']})",
        f"- Dataset: `{s['viewcellDataset']}`",
        f"- Pose CSR: `{s['poseCsr']}`",
        f"- Split source: `{s.get('splitSource', 'unknown')}`",
        f"- Split alignment: `{s.get('splitAlignment', {}).get('validated', False)}`",
        f"- Local GLB root: `{s['localGlbCheck']['glbRoot']}`",
        f"- Renderer: `{s['renderer']['schema']}`",
        f"- Formal image gate: `{bool(s.get('formalImageEvaluationReady', False))}`",
        f"- WebGL hardware gate: `{bool((s.get('renderer', {}).get('gpuGate') or {}).get('hardware', False))}`",
        "",
        "## Metrics",
        "",
        "| Scope | Precision | Recall | F1 | Jaccard | Avg Pred | Avg GT |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Component set | {c['posePrecision']:.4f} | {c['poseRecall']:.4f} | {c['poseF1']:.4f} | {c['poseJaccard']:.4f} | {s['avgPredComponents']:.2f} | {s['avgGtComponents']:.2f} |",
        f"| GLB set | {g['posePrecision']:.4f} | {g['poseRecall']:.4f} | {g['poseF1']:.4f} | {g['poseJaccard']:.4f} | {s['avgPredGlbs']:.2f} | {s['avgGtGlbs']:.2f} |",
        "",
        f"- Weighted recall: `{s['weightedRecall']:.4f}`",
        f"- Frustum-cell FNR: `{f['frustumCellFNR']:.6f}`",
        f"- Frustum-cell FPR: `{f['frustumCellFPR']:.6f}`",
        f"- Frustum-cell IoU: `{f['frustumCellIoU']:.6f}`",
        f"- Image PER: `{float(i.get('PER', 0.0)):.6f}`",
        f"- Mean / median / p95 PER: `{float(i.get('meanPER') or 0.0):.6f}` / `{float(i.get('medianPER') or 0.0):.6f}` / `{float(i.get('p95PER') or 0.0):.6f}`",
        f"- Miss pixel rate: `{float(i.get('missPixelRate', 0.0)):.6f}`",
        f"- Mean / median / p95 miss pixel rate: `{float(i.get('meanMissPixelRate') or 0.0):.6f}` / `{float(i.get('medianMissPixelRate') or 0.0):.6f}` / `{float(i.get('p95MissPixelRate') or 0.0):.6f}`",
        f"- Wrong-ID pixel rate: `{float(i.get('wrongInstancePixelRate', 0.0)):.6f}`",
        f"- Mean / median / p95 wrong-ID pixel rate: `{float(i.get('meanWrongInstancePixelRate') or 0.0):.6f}` / `{float(i.get('medianWrongInstancePixelRate') or 0.0):.6f}` / `{float(i.get('p95WrongInstancePixelRate') or 0.0):.6f}`",
        f"- Extra pixel rate over image: `{float(i.get('extraPixelRateOverImage', 0.0)):.6f}`",
        f"- Self-consistency PER: `{float(s.get('selfConsistencyPER') or 0.0):.6f}`",
        "",
        "## Notes",
        "",
        "- PER 使用真实 dense subpose 相机；后退扩大视锥只用于模型输入和视锥格诊断。",
        f"- {s['renderer']['description']}",
        "- `visible_weights` 仍然只表示 rvcServer 重要性权重，不是像素覆盖率。",
        "- 本脚本只检查并使用本地 GLB 路径，禁止远程资源访问。",
    ]
    top_missed = s.get("topMissedComponents", s.get("topMissedGlbs", []))
    if top_missed:
        lines.extend(["", "## Top Missed Components", "", "| Component | GLB | Miss Pixels | Path |", "|---:|---:|---:|---|"])
        for row in top_missed[:10]:
            lines.append(
                f"| {row.get('componentGlobalId', row.get('globalGlbId', ''))} | "
                f"{row.get('globalGlbId', '')} | {row['missPixels']} | `{row.get('path', '')}` |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_true_glb_renderer(
    args: argparse.Namespace,
    output_dir: Path,
    manifest_samples: list[dict[str, Any]],
    prediction_component_ids_by_key: dict[str, list[int]],
    threshold_selection: dict[str, Any],
    subpose_selection: dict[str, Any],
    glb_paths: dict[int, Path],
    instance_bindings: dict[str, Any],
    glb_aabbs: np.ndarray,
    component_aabbs: np.ndarray,
) -> dict[str, Any]:
    # A formal reference is full-scene.  Keep every local GLB in the manifest
    # even though the browser instance-reordering implementation is pending;
    # never silently shrink the image problem to the predicted GLB set.
    selected_glbs = sorted(int(gid) for gid in glb_paths)
    missing = [gid for gid in selected_glbs if gid not in glb_paths or not glb_paths[gid].exists()]
    if missing:
        raise RuntimeError(f"True GLB renderer refuses to run because local GLBs are missing: {missing[:16]}")
    formal = bool(args.formal_image_evaluation)
    if formal and args.split == "test" and (
        threshold_selection.get("selectionSplit") != "calibration"
        or threshold_selection.get("testRead") is not False
    ):
        raise RuntimeError(
            "formal test image evaluation requires a threshold frozen from calibration; "
            "test-time command-line threshold provenance is not accepted"
        )
    if formal and args.split == "test" and args.split_source != "pose_csr":
        raise RuntimeError("formal test image evaluation requires split_source=pose_csr")
    if formal and args.split == "test" and (
        int(args.max_viewcells) != 0 or int(args.subposes_per_viewcell) != 0
    ):
        raise RuntimeError(
            "formal test image evaluation requires all unique view-cells and all dense subposes"
        )
    selected_viewcell_count = int(subpose_selection.get("viewcellCount", 0))
    selected_sample_count = int(len(manifest_samples))
    manifest = {
        "schema": FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA if formal else INSTANCE_RENDER_MANIFEST_SCHEMA,
        "created": datetime.now().isoformat(timespec="seconds"),
        "width": int(args.width),
        "height": int(args.height),
        "renderFovYDeg": float(RENDER_FOV_Y_DEG),
        "modelInputFovYDeg": float(MODEL_INPUT_FOV_Y_DEG),
        "glbRoot": str(args.glb_root.resolve()),
        "glbIndex": str(args.glb_index.resolve()),
        "runtimeMeta": str(args.runtime_meta.resolve()),
        "selectedGlbs": selected_glbs,
        "split": str(args.split),
        "testRead": bool(args.split == "test"),
        "testEvaluationCount": int(1 if args.split == "test" else 0),
        "splitSource": str(args.split_source),
        "viewcellCount": selected_viewcell_count,
        "threshold": float(threshold_selection.get("threshold", args.threshold or 0.0)),
        "thresholdSelection": dict(threshold_selection),
        "thresholdProvenance": {
            "selectionSplit": threshold_selection.get("selectionSplit"),
            "testRead": threshold_selection.get("testRead", False),
            "source": threshold_selection.get("source"),
            "checkpointCalibrationOnly": threshold_selection.get("selectionSplit") == "calibration",
        },
        "testCoverage": {
            "schema": "pvs-formal-test-image-coverage-v1",
            "split": str(args.split),
            "selection": (
                "all_unique_test_viewcells"
                if args.split == "test" and int(args.max_viewcells) == 0
                else "selected_split_viewcells"
            ),
            "viewcellCount": selected_viewcell_count,
            "sampleCount": selected_sample_count,
            "uniquePoseCount": selected_sample_count,
            "maxViewcells": int(args.max_viewcells),
            "sampledWithReplacement": False,
            "subposesPerViewcell": int(args.subposes_per_viewcell),
        },
        "subposeSelection": subpose_selection,
        "reference": {
            "mode": "full_scene_renderable_instances",
            "idSource": "componentGlobalId",
            "geometrySource": "original_local_glb_meshes",
            "completeInventory": True,
        },
        "prediction": {
            "field": PREDICTION_COMPONENT_IDS_BY_KEY_FIELD,
            "keyField": PREDICTION_KEY_FIELD,
            "keyScope": "viewcellRow",
            "postFilter": "component_visibility_mask_after_conservative_render_submission",
        },
        PREDICTION_COMPONENT_IDS_BY_KEY_FIELD: {
            str(prediction_key): [int(component_id) for component_id in component_ids]
            for prediction_key, component_ids in prediction_component_ids_by_key.items()
        },
        "instanceBindings": instance_bindings,
        "glbAabbs": {
            str(global_glb_id): {
                "min": [float(value) for value in glb_aabbs[global_glb_id, :3]],
                "max": [float(value) for value in glb_aabbs[global_glb_id, 3:]],
            }
            for global_glb_id in selected_glbs
            if np.any(glb_aabbs[global_glb_id, 3:] > glb_aabbs[global_glb_id, :3])
        },
        "componentAabbs": {
            str(component_id): {
                "min": [float(value) for value in component_aabbs[component_id, :3]],
                "max": [float(value) for value in component_aabbs[component_id, 3:]],
            }
            for component_id in range(component_aabbs.shape[0])
            if np.any(component_aabbs[component_id, 3:] > component_aabbs[component_id, :3])
        },
        "spatialCulling": {
            "schema": "aabb-frustum-conservative-v1",
            "source": "runtimeMeta.globalGlbRecords[].aabb",
            "purpose": "render_submission_only",
            "renderFovYDeg": float(RENDER_FOV_Y_DEG),
            "near": float(args.render_near),
            "far": 20000.0,
            "completeInventoryRetained": True,
            "componentLevelMask": True,
        },
        "samples": manifest_samples,
        "previewSamples": int(args.preview_samples),
        "saveIdBuffers": bool(args.save_id_buffers),
        "localOnly": True,
        "idEncoding": INSTANCE_ID_ENCODING,
        "formalImageEvaluationReady": formal,
        "formalRequirements": {
            "requiresHardwareWebGL": formal,
            "syntheticSmokeAllowed": False,
            "completeGlbInventory": True,
            "renderFovYDeg": float(RENDER_FOV_Y_DEG),
            "modelInputFovYDeg": float(MODEL_INPUT_FOV_Y_DEG),
        },
        "limitations": [] if formal else [
            "The browser uses the original loaded GLB scene and an instance-level visibility mask; it does not compact instance matrices.",
            "This manifest is for validation/calibration smoke and remains non-formal until the registered image gates are met.",
        ],
    }
    if formal:
        validate_formal_instance_render_manifest(manifest)
    else:
        validate_instance_render_manifest(manifest)
    manifest_path = output_dir / "true_glb_render_manifest.json"
    # Large scenes share one view-cell prediction across several real subposes.
    # Compact JSON keeps the browser manifest below Node's string limit without
    # changing any sample or prediction semantics.
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    cmd = [
        "node",
        str(args.true_renderer_script),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(output_dir / "true_glb_render"),
    ]
    if args.require_hardware_gpu or formal:
        cmd.append("--require-hardware-gpu")
    if args.render_schema_only:
        cmd.append("--validate-only")
    if args.chrome_exe is not None:
        cmd.extend(["--chrome-exe", str(args.chrome_exe)])
    print(f"[viewcell-image-per] component render manifest: samples={len(manifest_samples)} glbs={len(selected_glbs)}", flush=True)
    render_output_dir = output_dir / "true_glb_render"
    gpu_evidence = {
        "schema": "pvs-browser-hardware-gpu-evidence-v1",
        "required": bool(args.require_hardware_gpu or formal),
        "chromeCommand": [str(value) for value in cmd],
        "phases": {},
    }
    gpu_evidence["phases"]["before"] = _capture_gpu_snapshot(render_output_dir, "before")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    gpu_evidence["pid"] = int(proc.pid)
    # Give Chrome a short window to create its GPU process before taking the
    # in-flight observation. The browser itself remains the source of the
    # WebGL vendor/renderer gate; host snapshots are supplementary evidence.
    time.sleep(0.5)
    gpu_evidence["phases"]["during"] = _capture_gpu_snapshot(render_output_dir, "during")
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=max(30, int(args.render_timeout_sec)))
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        stdout, stderr = proc.communicate()
    finally:
        gpu_evidence["phases"]["after"] = _capture_gpu_snapshot(render_output_dir, "after")
    (output_dir / "true_glb_render_stdout.log").write_text(stdout or "", encoding="utf-8")
    (output_dir / "true_glb_render_stderr.log").write_text(stderr or "", encoding="utf-8")
    gpu_evidence["returnCode"] = int(proc.returncode) if proc.returncode is not None else None
    gpu_evidence["timedOut"] = bool(timed_out)
    phase_complete = {
        phase: bool((details or {}).get("complete", False))
        for phase, details in gpu_evidence["phases"].items()
    }
    gpu_evidence["phaseComplete"] = phase_complete
    gpu_evidence["complete"] = bool(all(phase_complete.values()))
    summary_path = output_dir / "true_glb_render" / "render_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"True GLB renderer did not write {summary_path}")
    render_summary = read_json(summary_path)
    if not isinstance(render_summary, dict):
        raise ValueError(f"True GLB renderer summary is not an object: {summary_path}")
    render_summary["hardwareEvidence"] = gpu_evidence
    browser_gate = render_summary.get("gpuGate")
    if isinstance(browser_gate, dict):
        render_summary["gpuGate"] = {
            **browser_gate,
            "hostEvidenceRequired": bool(gpu_evidence["required"]),
            "hostEvidenceComplete": bool(gpu_evidence["complete"]),
        }
    summary_path.write_text(json.dumps(render_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if timed_out or proc.returncode != 0:
        raise RuntimeError(
            "True GLB renderer failed; see true_glb_render_stdout.log and true_glb_render_stderr.log. "
            f"Return code: {proc.returncode}"
        )
    if gpu_evidence["required"] and not gpu_evidence["complete"]:
        raise RuntimeError(
            "formal image evaluation requires complete before/during/after host GPU evidence; "
            f"see {render_output_dir}"
        )
    return render_summary


def main() -> None:
    args = parse_args()
    if args.self_test:
        subpose_selection_self_test()
        return
    if args.formal_image_evaluation and args.image_renderer != "true_glb":
        raise ValueError("formal image evaluation requires --image-renderer true_glb")
    if args.formal_image_evaluation and args.split == "test":
        if args.split_source != "pose_csr":
            raise ValueError("formal test image evaluation requires --split-source pose_csr")
        if args.threshold is not None:
            raise ValueError(
                "formal test image evaluation requires the checkpoint calibration summary; "
                "a command-line threshold is not accepted"
            )
        if int(args.max_viewcells) != 0 or int(args.subposes_per_viewcell) != 0:
            raise ValueError(
                "formal test image evaluation requires --max-viewcells 0 and "
                "--subposes-per-viewcell 0"
            )
    if len(args.model_spec) > 1:
        raise ValueError("--model-spec may be supplied at most once")
    dynamic_spec = None
    if args.model_spec:
        dynamic_spec = parse_model_spec(args.model_spec[0])
        output_model_name = dynamic_spec[0]
    else:
        output_model_name = args.model_name
    output_dir = args.output_dir or (ROOT / "benchmark/out" / f"viewcell_image_per_{output_model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepare_flat_output_dir(output_dir / "samples", (".bin",))
    prepare_flat_output_dir(output_dir / "previews", (".png",))

    grid = tuple(int(v.strip()) for v in args.froxel_grid.split(",") if v.strip())
    if len(grid) != 3:
        raise ValueError("--froxel-grid must be formatted as W,H,D")

    device = select_device(args.device)
    world_aabbs, instance_to_glb, runtime_meta = load_runtime_meta(args.runtime_meta)
    viewcells = ViewcellDataset(args.viewcell_dataset)
    pose_dataset = PoseCSRDataset(args.pose_csr, num_instances=world_aabbs.shape[0])
    selected_rows = viewcells.split_indices(
        args.split,
        args.max_viewcells,
        split_source=args.split_source,
        pose_dataset=pose_dataset,
    )
    if selected_rows.size == 0:
        raise RuntimeError(f"No viewcells selected for split={args.split}")
    if pose_dataset.poses.shape[0] != viewcells.viewcell_ids.shape[0]:
        raise RuntimeError(f"Pose CSR count {pose_dataset.poses.shape[0]} does not match viewcell count {viewcells.viewcell_ids.shape[0]}")

    specs = {dynamic_spec[0]: dynamic_spec[1]} if dynamic_spec is not None else selected_default_specs(args.model_name)
    canonical_name, spec = next(iter(specs.items()))
    runner = load_runner(canonical_name, spec, args.runtime_meta, device)
    threshold, threshold_info = resolve_threshold(args, spec, runner)

    glb_paths, local_glb_check = load_glb_index(args.glb_index, args.glb_root)
    # This is a JSON/header-only audit.  It verifies that the component ID
    # lists in runtimeVisibilityMeta have a safe instance-slot interpretation
    # before any image renderer is allowed to run.
    instance_bindings = build_instance_binding_preflight(
        runtime_meta,
        args.glb_index,
        args.glb_root,
    )
    glb_aabbs, glb_aabb_meta = load_glb_aabbs(runtime_meta, args.glb_points_meta)
    renderer = GlbAabbColorIdRenderer(glb_aabbs, args.width, args.height, args.render_near)

    chosen_subposes_by_row = {
        int(row): viewcells.selected_subposes(int(row), args.subposes_per_viewcell)
        for row in selected_rows.tolist()
    }
    selected_subpose_counts = [
        int(indices.size) for indices in chosen_subposes_by_row.values()
    ]
    if any(count <= 0 for count in selected_subpose_counts):
        raise RuntimeError("selected view-cell has no dense subpose; refusing incomplete image evaluation")
    subpose_selection = {
        "schema": "viewcell-subpose-selection-v1",
        "requestedPerViewcell": int(args.subposes_per_viewcell),
        "mode": "all" if int(args.subposes_per_viewcell) == 0 else "deterministic_evenly_spaced",
        "selectedSubposeCount": int(sum(selected_subpose_counts)),
        "viewcellCount": int(len(selected_subpose_counts)),
        "minPerViewcell": int(min(selected_subpose_counts)),
        "maxPerViewcell": int(max(selected_subpose_counts)),
        "meanPerViewcell": float(np.mean(selected_subpose_counts)),
    }
    needed_pose_indices = {
        int(viewcells.subpose_pose_indices[subpose_index])
        for subpose_indices in chosen_subposes_by_row.values()
        for subpose_index in subpose_indices.tolist()
    }
    raw_visibility: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    raw_summary = {"enabled": False, "reason": "disabled"}
    if not args.skip_raw_subpose_gt:
        raw_visibility, raw_summary = load_raw_visibility(args.raw_dir, needed_pose_indices)

    sample_rows: list[dict[str, Any]] = []
    image_totals = {
        "totalPixels": 0,
        "validReferencePixels": 0,
        "errorPixels": 0,
        "missPixels": 0,
        "wrongInstancePixels": 0,
        "extraPixels": 0,
        "backgroundReferencePixels": 0,
    }
    froxel_totals = {
        "gtCellCount": 0,
        "predCellCount": 0,
        "fnCellCount": 0,
        "fpCellCount": 0,
        "gridCellCount": 0,
        "intersectionCellCount": 0,
        "unionCellCount": 0,
    }
    weighted_recalls: list[float] = []
    pred_component_counts: list[int] = []
    gt_component_counts: list[int] = []
    pred_glb_counts: list[int] = []
    gt_glb_counts: list[int] = []
    top_missed: dict[int, int] = {}
    self_consistency_per: float | None = None
    true_render_manifest_samples: list[dict[str, Any]] = []
    prediction_component_ids_by_key: dict[str, list[int]] = {}
    missing_glb_samples = 0
    render_failed_samples = 0
    prediction_ms: list[float] = []
    t_start = time.perf_counter()

    samples_jsonl = output_dir / "samples.jsonl"
    with samples_jsonl.open("w", encoding="utf-8") as sample_log:
        for ordinal, row in enumerate(selected_rows.tolist()):
            row = int(row)
            gt_ids, gt_weights = pose_dataset.visible_slice(row)
            gt_ids = np.asarray(gt_ids, dtype=np.uint32)
            gt_weights = np.asarray(gt_weights, dtype=np.float32)
            candidate_ids = pose_dataset.candidate_slice(row)
            camera_norm = np.asarray(pose_dataset.poses["camera_norm"][row], dtype=np.float32)
            camera_world = np.asarray(pose_dataset.poses["camera_world"][row], dtype=np.float32)
            camera_view = pose_dataset.camera_view(row)
            mvp = pose_dataset.mvp_slice(row) if pose_dataset.mvp is not None else None
            pred_ids, pred_result = runner.predict_viewcell_ids(
                camera_norm,
                camera_world,
                camera_view,
                candidate_ids,
                query_center_world=pose_dataset.query_center_world(row, required=True),
                viewcell_radius_m=pose_dataset.viewcell_radius_m(row, required=True),
                mvp=mvp,
                threshold=threshold,
            )
            pred_ids = np.asarray(pred_ids, dtype=np.uint32)
            prediction_ms.append(float(pred_result.total_ms))
            prediction_key = f"vc{row:05d}"
            if args.image_renderer == "true_glb":
                prediction_component_ids_by_key[prediction_key] = [
                    int(value) for value in pred_ids.tolist()
                ]

            comp_metrics = set_metrics(pred_ids, gt_ids)
            wrec = weighted_recall(pred_ids, gt_ids, gt_weights)
            weighted_recalls.append(wrec)
            pred_glbs = glbs_for_components(pred_ids, instance_to_glb)
            gt_glbs = glbs_for_components(gt_ids, instance_to_glb)
            glb_metrics = set_metrics(pred_glbs, gt_glbs)
            froxel = froxel_metrics(gt_ids, pred_ids, world_aabbs, camera_world, camera_view, grid, args.froxel_near, args.froxel_far)
            froxel_totals["gtCellCount"] += int(froxel["gtCellCount"])
            froxel_totals["predCellCount"] += int(froxel["predCellCount"])
            froxel_totals["fnCellCount"] += int(froxel["fnCellCount"])
            froxel_totals["fpCellCount"] += int(froxel["fpCellCount"])
            froxel_totals["gridCellCount"] += int(froxel["gridCellCount"])
            union_count = int(froxel["gtCellCount"] + froxel["fpCellCount"])
            intersection_count = int(froxel["gtCellCount"] - froxel["fnCellCount"])
            froxel_totals["unionCellCount"] += union_count
            froxel_totals["intersectionCellCount"] += intersection_count

            pred_component_counts.append(int(pred_ids.size))
            gt_component_counts.append(int(gt_ids.size))
            pred_glb_counts.append(int(pred_glbs.size))
            gt_glb_counts.append(int(gt_glbs.size))

            sample_base = {
                "viewcellRow": row,
                "viewcellId": int(viewcells.viewcell_ids[row]),
                "category": CATEGORY_NAMES.get(int(viewcells.category_ids[row]), "unknown"),
                "candidateCount": int(candidate_ids.size),
                "gtComponentCount": int(gt_ids.size),
                "predComponentCount": int(pred_ids.size),
                "gtGlbCount": int(gt_glbs.size),
                "predGlbCount": int(pred_glbs.size),
                "weightedRecall": wrec,
                **metric_dict("component", comp_metrics),
                **metric_dict("glb", glb_metrics),
                "froxel": froxel,
                "subposes": [],
            }
            subpose_indices = chosen_subposes_by_row[row]
            for local_idx, subpose_index in enumerate(subpose_indices.tolist()):
                pose_index = int(viewcells.subpose_pose_indices[subpose_index])
                sub_pos = np.asarray(viewcells.subpose_pos[subpose_index], dtype=np.float32)
                sub_forward = np.asarray(viewcells.subpose_forward[subpose_index], dtype=np.float32)
                sub_params = np.asarray(viewcells.subpose_params[subpose_index], dtype=np.float32)
                model_fov_y = float(sub_params[0]) if sub_params.size >= 1 and sub_params[0] > 0 else 66.0
                render_fov_y = float(RENDER_FOV_Y_DEG)
                aspect = float(sub_params[1]) if sub_params.size >= 2 and sub_params[1] > 0 else (args.width / max(1, args.height))
                if pose_index in raw_visibility:
                    sub_gt_ids, _sub_weights = raw_visibility[pose_index]
                    sub_gt_source = "raw subpose visible_component_ids"
                else:
                    sub_gt_ids = gt_ids
                    sub_gt_source = "viewcell union fallback"
                ref_glbs = glbs_for_components(np.asarray(sub_gt_ids, dtype=np.uint32), instance_to_glb)
                missing = check_local_glbs(np.concatenate([ref_glbs, pred_glbs]), glb_paths)
                sub_summary: dict[str, Any] = {
                    "poseIndex": pose_index,
                    "subposeDatasetIndex": int(subpose_index),
                    "referenceGtSource": sub_gt_source,
                    "referenceGlbCount": int(ref_glbs.size),
                    "predGlbCount": int(pred_glbs.size),
                    "missingLocalGlbs": missing[:32],
                    "missingLocalGlbCount": int(len(missing)),
                }
                if missing:
                    missing_glb_samples += 1
                    render_failed_samples += 1
                    sub_summary["renderSkipped"] = True
                    sub_summary["renderSkipReason"] = "missing local GLB files"
                    sample_base["subposes"].append(sub_summary)
                    continue
                sample_id = f"vc{row:05d}_sp{local_idx:02d}_pose{pose_index}"
                if args.image_renderer == "true_glb":
                    true_render_manifest_samples.append(
                        {
                            "sampleId": sample_id,
                            "viewcellRow": row,
                            "viewcellId": int(viewcells.viewcell_ids[row]),
                            "poseIndex": pose_index,
                            "cameraPosition": [float(v) for v in sub_pos.tolist()],
                            "cameraForward": [float(v) for v in sub_forward.tolist()],
                            # The sampled 66 degree camera is the model query
                            # camera.  Image evaluation has its own fixed 60
                            # degree contract and must not inherit this value.
                            "renderFovYDeg": float(RENDER_FOV_Y_DEG),
                            "modelInputFovYDeg": float(model_fov_y),
                            "aspect": float(aspect),
                            PREDICTION_KEY_FIELD: prediction_key,
                            "referenceMode": "full_scene_renderable_instances",
                        }
                    )
                    sub_summary["renderSkipped"] = False
                    sub_summary["renderBackend"] = "true_glb_browser_component_id"
                    sample_base["subposes"].append(sub_summary)
                    continue
                reference = renderer.render(ref_glbs, sub_pos, sub_forward, render_fov_y, aspect)
                test = renderer.render(pred_glbs, sub_pos, sub_forward, render_fov_y, aspect)
                metrics, diff, _per_glb = compute_metrics(reference, test)
                for key in image_totals:
                    image_totals[key] += int(metrics.get(key, 0))
                for gid, pixels in reference_error_contributors(reference, test).items():
                    top_missed[gid] = top_missed.get(gid, 0) + int(pixels)
                sub_summary.update(
                    {
                        "renderSkipped": False,
                        "PER": metrics["PER"],
                        "missPixelRate": metrics["missPixelRate"],
                        "wrongInstancePixelRate": metrics["wrongInstancePixelRate"],
                        "extraPixelRateOverImage": metrics["extraPixelRateOverImage"],
                        "validReferencePixels": int(metrics["validReferencePixels"]),
                        "errorPixels": int(metrics["errorPixels"]),
                    }
                )
                # Full formal ID-buffer dumps can be several GB, so the default
                # keeps raw buffers for preview samples and writes every sample
                # only when explicitly requested.
                if args.save_id_buffers or ordinal < args.preview_samples:
                    reference.tofile(output_dir / "samples" / f"{sample_id}_reference_u32.bin")
                    test.tofile(output_dir / "samples" / f"{sample_id}_test_u32.bin")
                    diff.tofile(output_dir / "samples" / f"{sample_id}_diff_u8.bin")
                if ordinal < args.preview_samples:
                    save_preview(output_dir / "previews" / f"{sample_id}_preview.png", reference, test, diff)
                if ordinal == 0 and local_idx == 0:
                    self_metrics, _self_diff, _self_glb = compute_metrics(reference, reference)
                    sub_summary["selfConsistencyPER"] = self_metrics["PER"]
                    self_consistency_per = float(self_metrics["PER"])
                sample_base["subposes"].append(sub_summary)

            sample_log.write(json.dumps(sample_base, ensure_ascii=False) + "\n")
            sample_log.flush()
            sample_rows.append(sample_base)
            if args.log_every > 0 and ((ordinal + 1) % args.log_every == 0 or (ordinal + 1) == selected_rows.size):
                elapsed = time.perf_counter() - t_start
                print(
                    f"[viewcell-image-per] {ordinal + 1}/{selected_rows.size} "
                    f"avgPred={np.mean(pred_component_counts):.1f} avgGT={np.mean(gt_component_counts):.1f} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    if args.image_renderer == "true_glb":
        render_summary = run_true_glb_renderer(
            args,
            output_dir,
            true_render_manifest_samples,
            prediction_component_ids_by_key,
            threshold_info,
            subpose_selection,
            glb_paths,
            instance_bindings,
            glb_aabbs,
            world_aabbs,
        )
        if args.formal_image_evaluation and not args.render_schema_only:
            gpu_gate = render_summary.get("gpuGate") or {}
            if render_summary.get("formalImageEvaluationReady") is not True or gpu_gate.get("hardware") is not True:
                raise RuntimeError(
                    "formal image evaluation did not pass the browser instance-ID and hardware-GPU gates"
                )
        image_metrics = render_summary.get("imageMetrics") or {
            "schema": "component-id-image-schema-validation-v1",
            "formalImageEvaluationReady": False,
            "evaluatedSubposeCount": 0,
            "reason": "browser instance renderer is not implemented; validation only",
        }
        top_missed_rows = render_summary.get(
            "topMissedComponents", render_summary.get("topMissedGlbs", [])
        )
        self_consistency_per = render_summary.get("selfConsistencyPER")
        renderer_meta = {
            "schema": "local-true-component-id-browser-v3",
            "description": "浏览器一次加载完整本地 GLB 清单，将 componentGlobalId 和逐实例可见性绑定到 InstancedMesh 的 ID shader，并在同一页面顺序处理所有样本。",
            "formalImageEvaluationReady": bool(render_summary.get("formalImageEvaluationReady", False)),
            "gpuGate": render_summary.get("gpuGate"),
            "hardwareEvidence": render_summary.get("hardwareEvidence"),
            "renderSummary": str(output_dir / "true_glb_render" / "render_summary.json"),
            "limitations": [
                "The browser path uses a per-instance visibility mask rather than matrix compaction; it preserves component semantics but still submits the complete loaded scene.",
                "No formal image result may be reported from schema-only output or synthetic smoke.",
                "Separate evaluator invocations do not share a Chrome page; run one formal manifest per invocation.",
            ],
        }
    else:
        image_valid = max(1, image_totals["validReferencePixels"])
        image_total = max(1, image_totals["totalPixels"])
        image_metrics = {
            "schema": "color-id-per-v1-aggregate",
            **image_totals,
            "PER": float(image_totals["errorPixels"] / image_valid),
            "missPixelRate": float(image_totals["missPixels"] / image_valid),
            "wrongInstancePixelRate": float(image_totals["wrongInstancePixels"] / image_valid),
            "extraPixelRateOverImage": float(image_totals["extraPixels"] / image_total),
            "evaluatedSubposeCount": int(sum(1 for row in sample_rows for sp in row["subposes"] if not sp.get("renderSkipped"))),
            "renderFailedSubposeCount": int(render_failed_samples),
            "missingGlbSubposeCount": int(missing_glb_samples),
        }
        top_missed_rows = []
        for gid, pixels in sorted(top_missed.items(), key=lambda item: item[1], reverse=True)[:50]:
            path = glb_paths.get(int(gid))
            top_missed_rows.append({"globalGlbId": int(gid), "missPixels": int(pixels), "path": str(path) if path else ""})
        renderer_meta = {
            "schema": "local-glb-aabb-color-id-proxy-v1",
            "description": "GLB-level z-buffer ID rendering from decoded local GLB bounds; no remote asset access.",
            "glbBounds": glb_aabb_meta,
            "limitations": [
                "Measures GLB-level download/rendering loss, not component-level loss inside one GLB.",
                "Uses local GLB AABB proxy geometry; use --image-renderer true_glb for real mesh rendering.",
            ],
        }
    frustum_cell_metrics = {
        "schema": "instance-set-to-frustum-cell-diagnostic-v1",
        "grid": list(grid),
        "near": float(args.froxel_near),
        "far": float(args.froxel_far),
        **froxel_totals,
        "frustumCellFNR": float(froxel_totals["fnCellCount"] / max(1, froxel_totals["gtCellCount"])),
        "frustumCellFPR": float(froxel_totals["fpCellCount"] / max(1, froxel_totals["gridCellCount"] - froxel_totals["gtCellCount"])),
        "frustumCellIoU": float(froxel_totals["intersectionCellCount"] / max(1, froxel_totals["unionCellCount"])),
    }
    summary = {
        "schema": "viewcell-image-per-benchmark-v2",
        "created": datetime.now().isoformat(timespec="seconds"),
        "neuralPvsMetricReference": "https://windingwind.github.io/neuralpvs/index.html",
        "modelName": canonical_name,
        "modelSpec": spec,
        "threshold": float(threshold),
        "thresholdSelection": threshold_info,
        "device": str(device),
        "viewcellDataset": str(args.viewcell_dataset),
        "poseCsr": str(args.pose_csr),
        "splitSource": args.split_source,
        "splitAlignment": viewcells.split_alignment,
        "runtimeMeta": str(args.runtime_meta),
        "split": args.split,
        "testRead": bool(args.split == "test"),
        "testEvaluationCount": int(1 if args.split == "test" else 0),
        "viewcellCount": int(selected_rows.size),
        "subposesPerViewcell": int(args.subposes_per_viewcell),
        "subposeSelection": subpose_selection,
        "imageResolution": [int(args.width), int(args.height)],
        "componentSetMetrics": summarize_set_acc(sample_rows, "component"),
        "glbSetMetrics": summarize_set_acc(sample_rows, "glb"),
        "weightedRecall": float(np.mean(weighted_recalls)) if weighted_recalls else 0.0,
        "avgPredComponents": float(np.mean(pred_component_counts)) if pred_component_counts else 0.0,
        "avgGtComponents": float(np.mean(gt_component_counts)) if gt_component_counts else 0.0,
        "avgPredGlbs": float(np.mean(pred_glb_counts)) if pred_glb_counts else 0.0,
        "avgGtGlbs": float(np.mean(gt_glb_counts)) if gt_glb_counts else 0.0,
        "avgPredictionMs": float(np.mean(prediction_ms)) if prediction_ms else 0.0,
        "frustumCellMetrics": frustum_cell_metrics,
        "imageMetrics": image_metrics,
        "selfConsistencyPER": self_consistency_per,
        "topMissedComponents": top_missed_rows,
        "rawSubposeGt": raw_summary,
        "localGlbCheck": local_glb_check,
        "renderer": renderer_meta,
        "hardwareEvidence": renderer_meta.get("hardwareEvidence"),
        "formalImageEvaluationReady": bool(
            args.formal_image_evaluation and renderer_meta.get("formalImageEvaluationReady", False)
        ),
        "instanceBindingPreflight": instance_bindings,
        "sampleLog": str(samples_jsonl),
        "previewDir": str(output_dir / "previews"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_md(output_dir / "summary.md", summary)
    print(json.dumps({"outputDir": str(output_dir), "imageMetrics": image_metrics, "componentSetMetrics": summary["componentSetMetrics"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
