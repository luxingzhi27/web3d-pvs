#!/usr/bin/env python3
"""Build the 66-degree browser manifest for triangle depth peeling.

The existing M5 component-ID manifest supplies the audited component-to-GLB
bindings.  Pose records and the candidate dataset remain the authoritative
source for the new experiment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path: sys.path.insert(0, str(MODEL_DIR))
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from common.candidate_identity import (  # noqa: E402
    audit_native_aabb_candidates,
    candidate_digest_for_pose_sequence,
)
from common.runtime_meta import load_runtime_meta, scene_min_max  # noqa: E402
from viewcell_representative_subposes import (  # noqa: E402
    describe_selection,
    select_representative_subposes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--glb-index", required=True)
    parser.add_argument("--source-render-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--pose-start", type=int, default=0, help="Start ordinal within the selected split; used for independent shards.")
    parser.add_argument("--pose-count", type=int, default=0, help="Number of poses within the selected split; zero means through the end.")
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument(
        "--viewcell-dataset",
        default=None,
        help="Optional NeuralPVS-style view-cell source; selects center plus spatial representatives per cell.",
    )
    parser.add_argument("--representative-subposes-per-viewcell", type=int, default=5)
    parser.add_argument("--max-glbs", type=int, default=0, help="Smoke only; formal manifests must use all GLBs.")
    args = parser.parse_args()
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=int(json.loads(Path(args.runtime_meta).read_text())["instanceCount"]))
    world_aabbs, _mapping, runtime_meta = load_runtime_meta(args.runtime_meta)
    source = json.loads(Path(args.source_render_manifest).read_text(encoding="utf-8"))
    if source.get("schema") != "local-true-component-id-render-manifest-v2":
        raise ValueError("source render manifest must be the audited component-ID manifest")
    if source.get("instanceBindings", {}).get("schema") != "component-instance-binding-preflight-v1":
        raise ValueError("source render manifest has no audited component bindings")
    if args.split not in dataset.split_ids:
        raise ValueError(f"dataset has no split {args.split}")
    canonical_split_indices = np.flatnonzero(dataset.poses["split"] == int(dataset.split_ids[args.split]))
    candidate_audit = audit_native_aabb_candidates(dataset, world_aabbs, canonical_split_indices)
    canonical_candidate_digest = candidate_digest_for_pose_sequence(dataset, canonical_split_indices)
    representative_records: list[dict] = []
    if args.viewcell_dataset:
        viewcell_dir = Path(args.viewcell_dataset)
        required = [
            "viewcell_ids.bin", "viewcell_centers.bin", "subpose_offsets.bin",
            "subpose_pose_indices.bin", "subpose_camera_pos.bin", "subpose_camera_forward.bin",
        ]
        missing = [name for name in required if not (viewcell_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"view-cell dataset is missing {missing}")
        viewcell_ids = np.fromfile(viewcell_dir / "viewcell_ids.bin", dtype="<u4")
        centers = np.fromfile(viewcell_dir / "viewcell_centers.bin", dtype="<f4").reshape(-1, 3)
        offsets = np.fromfile(viewcell_dir / "subpose_offsets.bin", dtype="<u8")
        subpose_pose_indices = np.fromfile(viewcell_dir / "subpose_pose_indices.bin", dtype="<u4")
        subpose_positions = np.fromfile(viewcell_dir / "subpose_camera_pos.bin", dtype="<f4").reshape(-1, 3)
        subpose_forwards = np.fromfile(viewcell_dir / "subpose_camera_forward.bin", dtype="<f4").reshape(-1, 3)
        if centers.shape[0] != viewcell_ids.size or offsets.size != viewcell_ids.size + 1:
            raise ValueError("view-cell center/ID/offset counts are inconsistent")
        if (
            subpose_pose_indices.size != subpose_positions.shape[0]
            or subpose_forwards.shape[0] != subpose_positions.shape[0]
            or int(offsets[-1]) != subpose_pose_indices.size
        ):
            raise ValueError("view-cell subpose arrays are inconsistent")
        split_id = int(dataset.split_ids[args.split])
        canonical_set = set(int(value) for value in canonical_split_indices.tolist())
        if not np.array_equal(viewcell_ids, np.arange(viewcell_ids.size, dtype=np.uint32)):
            raise ValueError("formal view-cell source must use contiguous IDs matching its row order")

        # The source sampler stores a dense subpose ordinal in
        # subpose_pose_indices.bin.  It is not a PoseCSR row and must never be
        # used as one.  The canonical CSR row is resolved from the view-cell
        # row and then checked against the center/forward geometry.
        source_pose_by_viewcell: dict[int, int] = {}
        for row in range(viewcell_ids.size):
            if row >= dataset.poses.shape[0]:
                raise ValueError(f"view-cell row {row} has no PoseCSR row")
            pose = dataset.poses[row]
            pose_forward = np.asarray(pose["camera_forward"], dtype=np.float64)
            pose_forward /= max(float(np.linalg.norm(pose_forward)), 1e-12)
            center_delta = np.asarray(centers[row], dtype=np.float64) - np.asarray(pose["camera_world"], dtype=np.float64)
            forward_offset = float(np.dot(center_delta, pose_forward))
            residual = float(np.linalg.norm(center_delta - forward_offset * pose_forward))
            source_forward = np.asarray(subpose_forwards[int(offsets[row])], dtype=np.float64)
            source_forward /= max(float(np.linalg.norm(source_forward)), 1e-12)
            forward_error = 1.0 - float(np.dot(pose_forward, source_forward))
            if residual > 1e-2 or forward_offset <= 0.0 or forward_error > 1e-5:
                raise ValueError(
                    "view-cell/PoseCSR row alignment failed at row "
                    f"{row}: residual={residual:.6g}, offset={forward_offset:.6g}, "
                    f"forwardError={forward_error:.6g}"
                )
            source_pose_by_viewcell[row] = row

        selected_records: list[dict] = []
        for row in range(viewcell_ids.size):
            start, end = int(offsets[row]), int(offsets[row + 1])
            local_positions = subpose_positions[start:end]
            local = select_representative_subposes(
                centers[row],
                local_positions,
                count=int(args.representative_subposes_per_viewcell),
            )
            description = describe_selection(viewcell_ids[row], centers[row], local_positions, local)
            description["viewcellRow"] = int(row)
            selected_for_cell: list[dict] = []
            for local_index in local.tolist():
                flat_index = start + int(local_index)
                source_pose_index = int(source_pose_by_viewcell[row])
                if source_pose_index not in canonical_set or int(dataset.poses[source_pose_index]["split"]) != split_id:
                    continue
                selected_record = {
                    "renderPoseId": int(len(selected_records)),
                    "subposeIndex": int(flat_index),
                    "subposeLocalIndex": int(local_index),
                    "sourceSubposeOrdinal": int(subpose_pose_indices[flat_index]),
                    "sourcePoseIndex": source_pose_index,
                    "viewcellRow": int(row),
                    "viewcellId": int(viewcell_ids[row]),
                    "cameraWorld": subpose_positions[flat_index].astype(np.float32).tolist(),
                    "cameraForward": subpose_forwards[flat_index].astype(np.float32).tolist(),
                }
                selected_records.append(selected_record)
                selected_for_cell.append(selected_record)
            description["selectedPoseRecords"] = selected_for_cell
            representative_records.append(description)
        selected_render_records = selected_records
        representative_mode = "viewcell_center_plus_spatial_coverage"
    else:
        selected_render_records = [
            {
                "renderPoseId": int(index),
                "sourcePoseIndex": int(index),
                "cameraWorld": np.asarray(dataset.poses[int(index)]["camera_world"], dtype=np.float32).tolist(),
                "cameraForward": np.asarray(dataset.poses[int(index)]["camera_forward"], dtype=np.float32).tolist(),
            }
            for index in canonical_split_indices.tolist()
        ]
        representative_mode = "canonical_pose_fallback"
    if args.max_poses > 0:
        selected_render_records = selected_render_records[: int(args.max_poses)]
    pose_start = max(0, int(args.pose_start))
    if pose_start > len(selected_render_records):
        raise ValueError(f"pose-start {pose_start} exceeds selected split size {len(selected_render_records)}")
    pose_end = len(selected_render_records) if int(args.pose_count) <= 0 else min(len(selected_render_records), pose_start + int(args.pose_count))
    selected_render_records = selected_render_records[pose_start:pose_end]
    render_source_pose_sequence = np.asarray(
        [int(record["sourcePoseIndex"]) for record in selected_render_records], dtype=np.int64
    )
    render_candidate_digest = candidate_digest_for_pose_sequence(dataset, render_source_pose_sequence)
    glb_index = json.loads(Path(args.glb_index).read_text(encoding="utf-8"))
    entries = list(glb_index.get("entries", []))
    if args.max_glbs > 0: entries = entries[: int(args.max_glbs)]
    scene_min, scene_max, scene_size = scene_min_max(runtime_meta["sceneBounds"])
    camera_far = float(np.linalg.norm(scene_size) * 2.0)
    poses = []
    for ordinal, record in enumerate(selected_render_records):
        source_pose_index = int(record["sourcePoseIndex"])
        pose = dataset.poses[source_pose_index]
        camera_view = np.asarray(dataset.camera_view(source_pose_index), dtype=np.float32)
        poses.append({
            "renderPoseId": int(record.get("renderPoseId", ordinal)),
            "sourcePoseIndex": source_pose_index,
            "cameraWorld": np.asarray(record["cameraWorld"], dtype=np.float32).tolist(),
            "cameraForward": np.asarray(record["cameraForward"], dtype=np.float32).tolist(),
            "cameraView": camera_view.tolist(),
            "aspect": float(camera_view[3] / max(float(camera_view[4]), 1e-6)),
            "splitId": int(pose["split"]),
            **{
                key: record[key]
                for key in ("viewcellRow", "viewcellId", "subposeIndex", "subposeLocalIndex")
                if key in record
            },
        })
    payload = {
        "schema": "triangle-depth-layer-render-manifest-v2",
        "modelInputFovYDeg": 66.0,
        "frontendRenderFovYDeg": 60.0,
        "width": 320,
        "height": 180,
        "maxLayers": 6,
        "cameraFar": camera_far,
        "assetsDir": str(Path(args.glb_index).resolve().parent),
        "glbIndex": str(Path(args.glb_index).resolve()),
        "runtimeMeta": str(Path(args.runtime_meta).resolve()),
        "datasetDir": str(Path(args.dataset_dir).resolve()),
        "split": args.split,
        "formalReady": len(entries) == int(glb_index.get("total", len(entries))) and args.max_glbs <= 0 and args.pose_start == 0 and args.pose_count <= 0 and args.max_poses <= 0 and bool(args.viewcell_dataset),
        "poseOrdinalRange": {"split": args.split, "start": pose_start, "endExclusive": pose_end, "selectedRenderPoseCount": len(poses), "selectedSplitCount": int(np.count_nonzero(dataset.poses["split"] == int(dataset.split_ids[args.split])))},
        "representativeSubposeSelection": {
            "mode": representative_mode,
            "requestedPerViewcell": int(args.representative_subposes_per_viewcell),
            "viewcellDataset": str(Path(args.viewcell_dataset).resolve()) if args.viewcell_dataset else None,
            "viewcellCount": len(representative_records),
            "records": representative_records if args.viewcell_dataset else [],
        },
        "glbEntries": entries,
        "instanceBindings": source["instanceBindings"],
        "poses": poses,
        "sourceRenderManifest": str(Path(args.source_render_manifest).resolve()),
        "candidateSemantics": "sourcePoseIndex selects the native stored candidate CSR; browser renders each independent representative subpose and never unions GT IDs",
        "poseIdSemantics": "renderPoseId is unique per rendered subpose; sourcePoseIndex is the canonical PoseCSR row used only for candidate/GT lookup",
        "candidateIdentity": {
            "canonicalCandidateDigest": canonical_candidate_digest,
            "canonicalCandidateDigestScope": "all stored candidate CSR rows in the selected dataset split, canonical PoseCSR order",
            "renderCandidateDigest": render_candidate_digest,
            "renderCandidateDigestScope": "stored candidate CSR rows repeated in this manifest renderPose order",
            "canonicalPoseCount": int(canonical_split_indices.size),
            "renderPoseCount": int(render_source_pose_sequence.size),
        },
        "candidateAudit": candidate_audit,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "written", "output": str(output), "poseCount": len(poses), "glbCount": len(entries), "formalReady": payload["formalReady"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()
