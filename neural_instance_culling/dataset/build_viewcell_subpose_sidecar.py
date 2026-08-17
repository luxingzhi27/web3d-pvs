#!/usr/bin/env python3
"""Validate source view-cell supervision against a native candidate CSR.

The source directory stores one row per view-cell plus dense-subpose
supervision.  The candidate CSR stores the same visible union and the native
back-camera candidates.  They are validated together, but the main CSR is
never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np


SCHEMA = "pvs-viewcell-subpose-supervision-sidecar-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path, dtype: str) -> np.ndarray:
    if not path.is_file():
        raise ValueError(f"missing required file: {path}")
    itemsize = np.dtype(dtype).itemsize
    if path.stat().st_size % itemsize:
        raise ValueError(f"{path} has a byte length not divisible by {dtype}")
    return np.fromfile(path, dtype=dtype)


def check_offsets(name: str, offsets: np.ndarray, count: int, value_count: int | None = None) -> None:
    if (
        offsets.size != count + 1
        or offsets.size == 0
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"invalid {name}: expected {count + 1} monotonic offsets")
    if value_count is not None and int(offsets[-1]) != int(value_count):
        raise ValueError(
            f"invalid {name}: terminal offset {int(offsets[-1])} does not match "
            f"the referenced value count {int(value_count)}"
        )


def check_equal(name: str, left: np.ndarray, right: np.ndarray) -> None:
    if left.shape != right.shape or not np.array_equal(left, right):
        raise ValueError(f"{name} mismatch between source and main CSR")


def check_meta_count(meta: dict, key: str, actual: int, label: str) -> None:
    if key in meta and int(meta[key]) != int(actual):
        raise ValueError(f"{label} metadata {key} disagrees with binary data: {meta[key]} != {actual}")


def check_nonnegative_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} contains non-finite or negative values")


def file_digests(directory: Path, names: list[str]) -> dict[str, str]:
    return {name: sha256(directory / name) for name in names}


def same_tree(left: Path, right: Path) -> bool:
    left_files = sorted(path.relative_to(left) for path in left.rglob("*") if path.is_file())
    right_files = sorted(path.relative_to(right) for path in right.rglob("*") if path.is_file())
    if left_files != right_files:
        return False
    return all(sha256(left / path) == sha256(right / path) for path in left_files)


def build_sidecar(source_dir: Path, csr_dir: Path, output_dir: Path) -> dict:
    source_meta_path = source_dir / "dataset_meta.json"
    csr_meta_path = csr_dir / "dataset_meta.json"
    if not source_meta_path.is_file() or not csr_meta_path.is_file():
        raise ValueError("both source and main CSR require dataset_meta.json")
    source_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
    csr_meta = json.loads(csr_meta_path.read_text(encoding="utf-8"))

    pose_count = int(source_meta.get("viewcellCount", source_meta.get("poseCount", -1)))
    csr_pose_count = int(csr_meta.get("poseCount", -1))
    if pose_count <= 0 or csr_pose_count != pose_count:
        raise ValueError(f"source/CSR pose count mismatch: {pose_count} != {csr_pose_count}")
    source_viewcell_ids = read(source_dir / "viewcell_ids.bin", "<u4")
    if source_viewcell_ids.size != pose_count:
        raise ValueError("viewcell_ids length does not match source viewcell count")
    if not np.array_equal(source_viewcell_ids, np.arange(pose_count, dtype=np.uint32)):
        raise ValueError("source viewcell IDs are not the canonical row order")

    source_visible_offsets = read(source_dir / "visible_offsets.bin", "<u8")
    csr_visible_offsets = read(csr_dir / "visible_offsets.bin", "<u8")
    source_visible_ids = read(source_dir / "visible_ids.bin", "<u4")
    csr_visible_ids = read(csr_dir / "visible_ids.bin", "<u4")
    csr_candidate_offsets = read(csr_dir / "candidate_offsets.bin", "<u8")
    csr_candidate_ids = read(csr_dir / "candidate_ids.bin", "<u4")
    check_offsets("visible_offsets", source_visible_offsets, pose_count, source_visible_ids.size)
    check_offsets("csr_visible_offsets", csr_visible_offsets, pose_count, csr_visible_ids.size)
    check_offsets("candidate_offsets", csr_candidate_offsets, pose_count, csr_candidate_ids.size)
    check_meta_count(source_meta, "visibleCount", int(source_visible_ids.size), "source")
    check_meta_count(csr_meta, "visibleCount", int(csr_visible_ids.size), "CSR")
    check_meta_count(csr_meta, "candidateCount", int(csr_candidate_ids.size), "CSR")
    for name, left, right in (
        ("visible_offsets", source_visible_offsets, csr_visible_offsets),
        ("visible_ids", source_visible_ids, csr_visible_ids),
    ):
        check_equal(name, left, right)
    source_weights = read(source_dir / "visible_weights.bin", "<f4")
    csr_weights = read(csr_dir / "visible_weights.bin", "<f4")
    if source_weights.size != source_visible_ids.size or csr_weights.size != csr_visible_ids.size:
        raise ValueError("visible_weights length does not match visible_ids")
    check_nonnegative_finite("source visible_weights", source_weights)
    check_nonnegative_finite("CSR visible_weights", csr_weights)
    check_equal("visible_weights", source_weights, csr_weights)

    hits = read(source_dir / "visible_hit_counts.bin", "<u2")
    subpose_offsets = read(source_dir / "subpose_offsets.bin", "<u8")
    if hits.size != source_visible_ids.size:
        raise ValueError("visible_hit_counts length does not match visible_ids")
    check_offsets("subpose_offsets", subpose_offsets, pose_count)
    if subpose_offsets[-1] == 0:
        raise ValueError("subpose_offsets contains no successful subposes")
    raw_sampling = source_meta.get("rawSampling")
    if isinstance(raw_sampling, dict) and int(raw_sampling.get("sampleErrorRows", 0)) != 0:
        raise ValueError("source rawSampling reports failed subpose rows; the sidecar denominator is not valid")
    represented_subposes = np.diff(subpose_offsets).astype(np.uint64, copy=False)
    if int(represented_subposes.max(initial=0)) > np.iinfo(np.uint16).max:
        raise ValueError("subpose_offsets contains more than 65535 rows in one view-cell; uint16 hit counts cannot represent H")
    check_meta_count(source_meta, "subposeCount", int(subpose_offsets[-1]), "source")
    check_meta_count(csr_meta, "sourceSubposeCount", int(subpose_offsets[-1]), "CSR")
    if "stats" in csr_meta and isinstance(csr_meta["stats"], dict):
        check_meta_count(csr_meta["stats"], "subposeCount", int(subpose_offsets[-1]), "CSR stats")
        if "successSubposeCount" in csr_meta["stats"] and int(csr_meta["stats"]["successSubposeCount"]) != int(subpose_offsets[-1]):
            raise ValueError("CSR stats successSubposeCount disagrees with source subpose_offsets")
    # ``np.repeat`` requires a signed integer repeat-count array on the
    # NumPy versions used by the CUDA environment.  The offsets themselves
    # remain uint64 on disk; only this validation view is converted.
    visible_row_counts = np.diff(source_visible_offsets).astype(np.int64, copy=False)
    if np.any(hits.astype(np.uint64, copy=False) > np.repeat(represented_subposes, visible_row_counts)):
        raise ValueError("visible_hit_counts contains a hit count greater than the represented subpose count")

    if "sourceViewcellCount" in csr_meta and int(csr_meta["sourceViewcellCount"]) != pose_count:
        raise ValueError("main CSR sourceViewcellCount disagrees with poses")
    source_split_path = source_dir / "viewcell_split_ids.bin"
    source_split = read(source_split_path, "u1") if source_split_path.is_file() else np.full(pose_count, 255, dtype=np.uint8)
    if source_split.size != pose_count:
        raise ValueError("source viewcell split IDs do not match viewcell count")
    csr_pose_bytes = read(csr_dir / "poses.bin", "<u1")
    if csr_pose_bytes.size != pose_count * 64:
        raise ValueError("main CSR poses.bin is not a 64-byte pose array")
    if "poseStrideBytes" in csr_meta and int(csr_meta["poseStrideBytes"]) != 64:
        raise ValueError("main CSR poseStrideBytes is not the required 64-byte directional pose layout")
    csr_split = csr_pose_bytes.reshape(pose_count, 64)[:, 44].astype(np.uint8, copy=False)

    # The source sampler and the formal pose CSR may deliberately use
    # different split manifests.  In the current HKUST dataset the source
    # rows have train/val/test labels, while the formal CSR applies the
    # spatial train/validation/calibration/test/guard partition afterward.
    # Row order and visible/candidate identity are the invariants here; split
    # labels are copied for audit and are never used to rewrite supervision.
    split_identity_declared = isinstance(source_meta.get("splitIds"), dict) and isinstance(csr_meta.get("splitIds"), dict)
    split_identity_equal = bool(np.array_equal(source_split, csr_split))

    for row in range(pose_count):
        visible = source_visible_ids[source_visible_offsets[row]:source_visible_offsets[row + 1]]
        candidates = csr_candidate_ids[csr_candidate_offsets[row]:csr_candidate_offsets[row + 1]]
        if not np.isin(visible, candidates, assume_unique=False).all():
            raise ValueError(f"visible_ids is not a candidate subset at pose {row}")

    source_input_files = [
        "dataset_meta.json", "viewcell_ids.bin", "visible_offsets.bin", "visible_ids.bin",
        "visible_weights.bin", "visible_hit_counts.bin", "subpose_offsets.bin",
    ]
    if source_split_path.is_file():
        source_input_files.append("viewcell_split_ids.bin")
    csr_input_files = [
        "dataset_meta.json", "poses.bin", "visible_offsets.bin", "visible_ids.bin",
        "visible_weights.bin", "candidate_offsets.bin", "candidate_ids.bin",
    ]
    temp_parent = output_dir.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=temp_parent))
    try:
        np.asarray(hits, dtype="<u2").tofile(temporary_dir / "visible_hit_counts.bin")
        np.asarray(subpose_offsets, dtype="<u8").tofile(temporary_dir / "subpose_offsets.bin")
        np.arange(pose_count, dtype="<u4").tofile(temporary_dir / "pose_indices.bin")
        np.asarray(source_split, dtype="u1").tofile(temporary_dir / "source_viewcell_split_ids.bin")
        np.asarray(csr_split, dtype="u1").tofile(temporary_dir / "csr_pose_split_ids.bin")
        output_files = [
            "visible_hit_counts.bin", "subpose_offsets.bin", "pose_indices.bin",
            "source_viewcell_split_ids.bin", "csr_pose_split_ids.bin",
        ]
        output_digests = file_digests(temporary_dir, output_files)
        source_digest_files = file_digests(source_dir, source_input_files)
        csr_digest_files = file_digests(csr_dir, csr_input_files)
        manifest = {
            "schema": SCHEMA,
            "version": 2,
            "sourceDataset": str(source_dir.resolve()),
            "mainCsrDataset": str(csr_dir.resolve()),
            "poseCount": pose_count,
            "visibleCount": int(hits.size),
            "representedSubposeCount": int(subpose_offsets[-1]),
            "successfulSubposeCount": int(subpose_offsets[-1]),
            "subposeDenominatorSemantics": "H is the number of subpose rows represented by source subpose_offsets; no per-row success mask is available in this sidecar input",
            "sourceMetaSha256": sha256(source_meta_path),
            "mainCsrMetaSha256": sha256(csr_meta_path),
            "sourceSplitIdentityChecked": split_identity_equal,
            "sourceSplitIdentityCompared": split_identity_declared,
            "splitLabelsMayDiffer": bool(split_identity_declared and not split_identity_equal),
            "sourceSplitCounts": {str(int(x)): int(np.count_nonzero(source_split == x)) for x in np.unique(source_split)},
            "csrSplitCounts": {str(int(x)): int(np.count_nonzero(csr_split == x)) for x in np.unique(csr_split)},
            "semantics": {
                "label": "h > 0",
                "hitCount": "number of represented subpose rows seeing the visible instance",
                "hitRate": "h / H; supervision only, never a candidate or GT rewrite",
                "visibleWeights": "copied only for validation; this sidecar does not reinterpret weights as pixel coverage",
            },
            "validatedInputFiles": {"source": source_digest_files, "mainCsr": csr_digest_files},
            "validatedFiles": {
                "mainCsrPoses": csr_digest_files["poses.bin"],
                "sourceVisibleIds": source_digest_files["visible_ids.bin"],
                "mainCsrVisibleIds": csr_digest_files["visible_ids.bin"],
                "sourceVisibleWeights": source_digest_files["visible_weights.bin"],
                "mainCsrVisibleWeights": csr_digest_files["visible_weights.bin"],
                "mainCsrCandidateIds": csr_digest_files["candidate_ids.bin"],
                "mainCsrCandidateOffsets": csr_digest_files["candidate_offsets.bin"],
                "sourceVisibleHitCounts": source_digest_files["visible_hit_counts.bin"],
                "sourceSubposeOffsets": source_digest_files["subpose_offsets.bin"],
            },
            "fileSha256": output_digests,
            "files": {
                "visibleHitCounts": "visible_hit_counts.bin",
                "subposeOffsets": "subpose_offsets.bin",
                "poseIndices": "pose_indices.bin",
                "sourceViewcellSplitIds": "source_viewcell_split_ids.bin",
                "csrPoseSplitIds": "csr_pose_split_ids.bin",
            },
        }
        (temporary_dir / "sidecar_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_dir.exists():
            if not output_dir.is_dir():
                raise FileExistsError(f"sidecar output exists and is not a directory: {output_dir}")
            if not same_tree(temporary_dir, output_dir):
                raise FileExistsError(f"refusing to overwrite non-identical sidecar output: {output_dir}")
            return json.loads((output_dir / "sidecar_manifest.json").read_text(encoding="utf-8"))
        temporary_dir.replace(output_dir)
        return manifest
    finally:
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--csr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_sidecar(args.source_dir, args.csr_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
