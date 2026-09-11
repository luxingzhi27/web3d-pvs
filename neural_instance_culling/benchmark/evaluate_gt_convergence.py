#!/usr/bin/env python3
"""Measure nested view-cell GT convergence from a plan and raw JSONL pair.

The formal input contract is the generated 100 x 128 plan, its matching
Color-ID raw JSONL and the source Pose CSR dataset. A raw row is joined to a plan row by
``(viewcell_id, subpose_id, source_pose_index)``; ``pose_index`` is only an
optional consistency check and is never used as the join key.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


EXPECTED_VIEWCELL_COUNT = 100
EXPECTED_SUBPOSE_COUNT = 128
DEFAULT_SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128)
AlignmentKey = tuple[int, int, int]
Visibility = tuple[np.ndarray, np.ndarray]


def _jsonl_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jsonl"))
        if not files:
            raise ValueError(f"JSONL directory has no *.jsonl files: {path}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"JSONL input does not exist: {path}")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from one JSONL file or a sorted JSONL directory."""
    for file in _jsonl_files(Path(path)):
        with file.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {file}:{line_number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {file}:{line_number}")
                yield value


def _required_int(record: dict[str, Any], field: str, context: str) -> int:
    value = record.get(field)
    if value is None or isinstance(value, bool):
        raise ValueError(f"{context} is missing integer field {field!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{context} field {field!r} is not an integer: {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} field {field!r} is not an integer: {value!r}") from error


def _optional_int(record: dict[str, Any], field: str, context: str) -> int | None:
    if field not in record or record[field] is None:
        return None
    return _required_int(record, field, context)


def load_plan(
    path: Path,
    *,
    viewcell_count: int = EXPECTED_VIEWCELL_COUNT,
    subpose_count: int = EXPECTED_SUBPOSE_COUNT,
) -> dict[int, list[dict[str, Any]]]:
    """Load and validate the exact nested plan layout.

    The returned entries are ordered by ``subpose_id`` for every view-cell.
    A duplicate ``(viewcell_id, subpose_id)`` is rejected even when its source
    pose differs, so a source pose can never silently replace another row.
    """
    if viewcell_count <= 0 or subpose_count <= 0:
        raise ValueError("viewcell_count and subpose_count must be positive")

    slots: dict[tuple[int, int], dict[str, Any]] = {}
    row_count = 0
    for row_count, record in enumerate(iter_jsonl(Path(path)), start=1):
        context = f"plan row {row_count}"
        viewcell_id = _required_int(record, "viewcell_id", context)
        subpose_id = _required_int(record, "subpose_id", context)
        source_pose_index = _required_int(record, "source_pose_index", context)
        if not 0 <= viewcell_id < viewcell_count:
            raise ValueError(f"{context} viewcell_id={viewcell_id} is outside 0..{viewcell_count - 1}")
        if not 0 <= subpose_id < subpose_count:
            raise ValueError(f"{context} subpose_id={subpose_id} is outside 0..{subpose_count - 1}")
        if source_pose_index < 0:
            raise ValueError(f"{context} source_pose_index must be non-negative")
        slot = (viewcell_id, subpose_id)
        if slot in slots:
            raise ValueError(
                f"plan has duplicate viewcell_id={viewcell_id}, subpose_id={subpose_id}"
            )
        slots[slot] = {
            "key": (viewcell_id, subpose_id, source_pose_index),
            "poseIndex": _optional_int(record, "pose_index", context),
        }

    expected_rows = viewcell_count * subpose_count
    if row_count != expected_rows:
        raise ValueError(
            f"plan must contain exactly {viewcell_count}x{subpose_count}={expected_rows} rows; "
            f"found {row_count}"
        )

    plan: dict[int, list[dict[str, Any]]] = {}
    for viewcell_id in range(viewcell_count):
        missing = [
            subpose_id
            for subpose_id in range(subpose_count)
            if (viewcell_id, subpose_id) not in slots
        ]
        if missing:
            raise ValueError(
                f"plan viewcell_id={viewcell_id} is missing subpose_id values {missing[:8]}"
            )
        plan[viewcell_id] = [slots[(viewcell_id, subpose_id)] for subpose_id in range(subpose_count)]
    return plan


def _parse_visibility(record: dict[str, Any], context: str) -> Visibility:
    raw_ids = record.get("visible_component_ids")
    if raw_ids is None:
        raw_ids = []
    if not isinstance(raw_ids, (list, tuple)):
        raise ValueError(f"{context} visible_component_ids must be an array")

    ids: list[int] = []
    for index, value in enumerate(raw_ids):
        if isinstance(value, bool):
            raise ValueError(f"{context} visible_component_ids[{index}] is not an integer")
        try:
            component_id = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{context} visible_component_ids[{index}] is not an integer"
            ) from error
        if component_id < 0 or component_id > np.iinfo(np.uint32).max:
            raise ValueError(f"{context} component id is outside uint32: {component_id}")
        ids.append(component_id)

    if "component_weights" not in record or record["component_weights"] is None:
        weights = np.ones(len(ids), dtype=np.float64)
    else:
        raw_weights = record["component_weights"]
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError(f"{context} component_weights must be an array")
        if len(raw_weights) != len(ids):
            raise ValueError(
                f"{context} component_weights length {len(raw_weights)} does not match "
                f"visible_component_ids length {len(ids)}"
            )
        try:
            weights = np.asarray(raw_weights, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{context} component_weights are not numeric") from error
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"{context} component_weights must be finite and non-negative")

    return np.asarray(ids, dtype=np.uint32), weights


def load_raw(
    path: Path,
    plan: dict[int, list[dict[str, Any]]],
) -> dict[AlignmentKey, Visibility]:
    """Load raw rows and require an exact key-for-key match with ``plan``."""
    expected: dict[AlignmentKey, dict[str, Any]] = {
        entry["key"]: entry
        for entries in plan.values()
        for entry in entries
    }
    visibility: dict[AlignmentKey, Visibility] = {}
    for row_number, record in enumerate(iter_jsonl(Path(path)), start=1):
        context = f"raw row {row_number}"
        viewcell_id = _required_int(record, "viewcell_id", context)
        subpose_id = _required_int(record, "subpose_id", context)
        source_pose_index = _required_int(record, "source_pose_index", context)
        key = (viewcell_id, subpose_id, source_pose_index)
        if key not in expected:
            raise ValueError(
                f"{context} key {key} does not match the supplied plan; "
                "raw rows are not aligned by pose_index"
            )
        if key in visibility:
            raise ValueError(f"raw has duplicate alignment key {key}")

        plan_pose_index = expected[key].get("poseIndex")
        raw_pose_index = _optional_int(record, "pose_index", context)
        if plan_pose_index is not None and raw_pose_index is not None and plan_pose_index != raw_pose_index:
            raise ValueError(
                f"{context} pose_index={raw_pose_index} disagrees with plan pose_index={plan_pose_index} "
                f"for alignment key {key}"
            )
        visibility[key] = _parse_visibility(record, context)

    missing = sorted(set(expected) - set(visibility))
    if missing:
        raise ValueError(
            f"raw visibility is missing {len(missing)} plan rows; first missing key is {missing[0]}"
        )
    if len(visibility) != len(expected):
        raise ValueError(
            f"raw visibility has {len(visibility)} aligned rows but the plan has {len(expected)}"
        )
    return visibility


def load_source_visibility(
    dataset_dir: Path,
    plan: dict[int, list[dict[str, Any]]],
) -> dict[int, Visibility]:
    """Load the existing Pose CSR GT for each selected source view-cell."""

    dataset_dir = Path(dataset_dir).resolve()
    meta = json.loads((dataset_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    pose_count = int(meta.get("poseCount", -1))
    offsets = np.fromfile(dataset_dir / "visible_offsets.bin", dtype="<u8")
    ids = np.fromfile(dataset_dir / "visible_ids.bin", dtype="<u4")
    weights = np.fromfile(dataset_dir / "visible_weights.bin", dtype="<f4")
    if offsets.size != pose_count + 1 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("source dataset visible offsets do not match poseCount")
    if int(offsets[-1]) != ids.size or weights.size != ids.size:
        raise ValueError("source dataset visible IDs/weights do not match offsets")

    result: dict[int, Visibility] = {}
    for viewcell_id, entries in plan.items():
        source_indices = {int(entry["key"][2]) for entry in entries}
        if len(source_indices) != 1:
            raise ValueError(f"viewcell_id={viewcell_id} maps to multiple source poses")
        source_index = source_indices.pop()
        if not 0 <= source_index < pose_count:
            raise ValueError(f"source pose {source_index} is outside the Pose CSR dataset")
        start, end = int(offsets[source_index]), int(offsets[source_index + 1])
        result[viewcell_id] = (ids[start:end].copy(), weights[start:end].astype(np.float64))
    return result


def nested_union(
    ordered_pose_ids: Sequence[Any] | np.ndarray,
    visibility: dict[Any, Visibility],
    count: int,
) -> tuple[set[int], dict[int, float]]:
    """Return the union and per-component maximum weight for a prefix."""
    if count < 0 or count > len(ordered_pose_ids):
        raise ValueError(f"prefix count {count} is outside the ordered input")
    ids: set[int] = set()
    weights: dict[int, float] = {}
    for pose_id in list(ordered_pose_ids)[:count]:
        if pose_id not in visibility:
            raise ValueError(f"visibility is missing ordered row {pose_id!r}")
        pose_ids, pose_weights = visibility[pose_id]
        for component_id, weight in zip(pose_ids.tolist(), pose_weights.tolist()):
            component_id = int(component_id)
            ids.add(component_id)
            weights[component_id] = max(weights.get(component_id, 0.0), float(weight))
    return ids, weights


def parse_counts(value: str) -> list[int]:
    counts = sorted({int(item) for item in value.split(",") if item.strip()})
    if not counts or counts[0] <= 0:
        raise ValueError("sample counts must be positive")
    if any(count > EXPECTED_SUBPOSE_COUNT for count in counts):
        raise ValueError(f"sample counts cannot exceed {EXPECTED_SUBPOSE_COUNT}")
    if any(count & (count - 1) for count in counts):
        raise ValueError("sample counts must be powers of two")
    return counts


def _mass(weights: dict[int, float]) -> float:
    return float(sum(weights.values()))


def build_convergence_rows(
    scene: str,
    plan: dict[int, list[dict[str, Any]]],
    visibility: dict[AlignmentKey, Visibility],
    source_visibility: dict[int, Visibility],
    sample_counts: Sequence[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for viewcell_id in sorted(plan):
        ordered_keys = [entry["key"] for entry in plan[viewcell_id]]
        final_count = len(ordered_keys)
        if any(count < 1 or count > final_count for count in sample_counts):
            raise ValueError(
                f"sample counts must be in 1..{final_count} for viewcell_id={viewcell_id}"
            )
        snapshot_counts = {0, final_count, *sample_counts}
        snapshot_counts.update(count // 2 for count in sample_counts)
        prefix_ids: set[int] = set()
        prefix_weights: dict[int, float] = {}
        snapshots: dict[int, tuple[set[int], dict[int, float]]] = {0: (set(), {})}
        for count, pose_id in enumerate(ordered_keys, start=1):
            pose_ids, pose_weights = visibility[pose_id]
            for component_id, weight in zip(pose_ids.tolist(), pose_weights.tolist()):
                component_id = int(component_id)
                prefix_ids.add(component_id)
                prefix_weights[component_id] = max(
                    prefix_weights.get(component_id, 0.0), float(weight)
                )
            if count in snapshot_counts:
                snapshots[count] = (set(prefix_ids), dict(prefix_weights))
        final_ids, final_weights = snapshots[final_count]
        final_mass = _mass(final_weights)
        source_ids_array, source_weights_array = source_visibility[viewcell_id]
        source_ids = set(int(value) for value in source_ids_array.tolist())
        source_weights = {
            int(component_id): float(weight)
            for component_id, weight in zip(source_ids_array.tolist(), source_weights_array.tolist())
        }
        intersection = source_ids & final_ids
        union = source_ids | final_ids
        source_mass = _mass(source_weights)

        for count in sample_counts:
            current_ids, current_weights = snapshots[count]
            previous_count = count // 2
            previous_ids = snapshots[previous_count][0]
            new_ids = current_ids - previous_ids
            missing_ids = final_ids - current_ids
            current_mass = _mass(current_weights)
            final_visible_count = len(final_ids)
            rows.append({
                "scene": scene,
                "viewcellId": viewcell_id,
                "availableSubposes": len(ordered_keys),
                "sampleCount": count,
                "visibleCount": len(current_ids),
                "finalReferenceVisibleCount": final_visible_count,
                "visibleCoverage": (
                    len(current_ids) / final_visible_count if final_visible_count else 1.0
                ),
                "newInstanceCount": len(new_ids),
                "newInstanceRate": (
                    len(new_ids) / final_visible_count if final_visible_count else 0.0
                ),
                "remainingInstanceCount": len(missing_ids),
                "remainingInstanceRate": (
                    len(missing_ids) / final_visible_count if final_visible_count else 0.0
                ),
                "weightedConvergence": (
                    current_mass / final_mass if final_mass else 1.0
                ),
                "weightedCoverage": (
                    current_mass / final_mass if final_mass else 1.0
                ),
                "finalReferenceWeightMass": final_mass,
                "sourceVisibleCount": len(source_ids),
                "sourceCoverageOfReference": (
                    len(intersection) / final_visible_count if final_visible_count else 1.0
                ),
                "sourceCoverageOfReferenceWeight": (
                    sum(final_weights[component_id] for component_id in intersection) / final_mass
                    if final_mass else 1.0
                ),
                "referenceCoverageOfSource": (
                    len(intersection) / len(source_ids) if source_ids else 1.0
                ),
                "referenceCoverageOfSourceWeight": (
                    sum(source_weights[component_id] for component_id in intersection) / source_mass
                    if source_mass else 1.0
                ),
                "sourceReferenceJaccard": len(intersection) / len(union) if union else 1.0,
            })
    return rows


def _quantile(values: Sequence[float], probability: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def summarize_rows(
    scene: str,
    rows: Sequence[dict[str, Any]],
    sample_counts: Sequence[int],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for count in sample_counts:
        subset = [row for row in rows if int(row["sampleCount"]) == count]
        if not subset:
            continue

        def values(field: str) -> list[float]:
            return [float(row[field]) for row in subset]

        summary_rows.append({
            "scene": scene,
            "sampleCount": count,
            "viewcellCount": len(subset),
            "visibleCountMean": float(np.mean(values("visibleCount"))),
            "visibleCountP05": _quantile(values("visibleCount"), 0.05),
            "finalReferenceVisibleCountMean": float(np.mean(values("finalReferenceVisibleCount"))),
            "finalReferenceVisibleCountP05": _quantile(values("finalReferenceVisibleCount"), 0.05),
            "visibleCoverageMean": float(np.mean(values("visibleCoverage"))),
            "visibleCoverageP05": _quantile(values("visibleCoverage"), 0.05),
            "newInstanceCountMean": float(np.mean(values("newInstanceCount"))),
            "newInstanceRateMean": float(np.mean(values("newInstanceRate"))),
            "newInstanceRateP05": _quantile(values("newInstanceRate"), 0.05),
            "remainingInstanceRateMean": float(np.mean(values("remainingInstanceRate"))),
            "remainingInstanceRateP05": _quantile(values("remainingInstanceRate"), 0.05),
            "weightedConvergenceMean": float(np.mean(values("weightedConvergence"))),
            "weightedConvergenceP05": _quantile(values("weightedConvergence"), 0.05),
            "weightedCoverageMean": float(np.mean(values("weightedCoverage"))),
            "weightedCoverageP05": _quantile(values("weightedCoverage"), 0.05),
            "sourceCoverageOfReferenceMean": float(np.mean(values("sourceCoverageOfReference"))),
            "sourceCoverageOfReferenceP05": _quantile(values("sourceCoverageOfReference"), 0.05),
            "sourceCoverageOfReferenceWeightMean": float(np.mean(values("sourceCoverageOfReferenceWeight"))),
            "sourceCoverageOfReferenceWeightP05": _quantile(values("sourceCoverageOfReferenceWeight"), 0.05),
            "referenceCoverageOfSourceMean": float(np.mean(values("referenceCoverageOfSource"))),
            "referenceCoverageOfSourceWeightMean": float(np.mean(values("referenceCoverageOfSourceWeight"))),
            "sourceReferenceJaccardMean": float(np.mean(values("sourceReferenceJaccard"))),
        })
    return summary_rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    *,
    scene: str,
    plan_path: Path,
    raw_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    sample_counts: Sequence[int] = DEFAULT_SAMPLE_COUNTS,
    viewcell_count: int = EXPECTED_VIEWCELL_COUNT,
    subpose_count: int = EXPECTED_SUBPOSE_COUNT,
) -> dict[str, Any]:
    if subpose_count != EXPECTED_SUBPOSE_COUNT:
        raise ValueError(
            f"the formal GT convergence evaluator requires {EXPECTED_SUBPOSE_COUNT} subposes per view-cell"
        )
    requested_counts = sorted({int(value) for value in sample_counts})
    if not requested_counts or any(value <= 0 or value > subpose_count for value in requested_counts):
        raise ValueError(f"sample counts must be in 1..{subpose_count}")
    if any(value & (value - 1) for value in requested_counts):
        raise ValueError("sample counts must be powers of two")

    plan = load_plan(
        Path(plan_path),
        viewcell_count=viewcell_count,
        subpose_count=subpose_count,
    )
    visibility = load_raw(Path(raw_path), plan)
    source_visibility = load_source_visibility(Path(dataset_dir), plan)
    rows = build_convergence_rows(
        scene, plan, visibility, source_visibility, requested_counts
    )
    summary_rows = summarize_rows(scene, rows, requested_counts)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "per_viewcell.csv", rows)
    _write_csv(output / "summary.csv", summary_rows)
    manifest = {
        "schema": "pvs-viewcell-gt-convergence-v2",
        "scene": scene,
        "split": "validation",
        "formalEntry": "plan-jsonl-plus-raw-jsonl",
        "plan": str(Path(plan_path).resolve()),
        "raw": str(Path(raw_path).resolve()),
        "sourceDataset": str(Path(dataset_dir).resolve()),
        "viewcellCount": viewcell_count,
        "subposesPerViewcell": subpose_count,
        "requestedSampleCounts": requested_counts,
        "alignment": "exact (viewcell_id, subpose_id, source_pose_index) key; pose_index is only checked when present",
        "nestedOrder": "ascending subpose_id from the supplied plan",
        "reference": f"union of subpose_id 0..{subpose_count - 1} in each view-cell",
        "newInstanceRateDefinition": f"new IDs in G_N minus G_(N/2), divided by |G_{subpose_count}|; N=1 uses an empty prefix",
        "remainingInstanceRateDefinition": f"IDs in G_{subpose_count} not present in G_N, divided by |G_{subpose_count}|",
        "weightedConvergenceDefinition": f"sum of per-instance maximum component_weights in G_N divided by the corresponding G_{subpose_count} sum",
        "sourceCoverageDefinition": f"fraction of G_{subpose_count} IDs or G_{subpose_count} weight mass already present in the source Pose CSR GT",
        "summary": summary_rows,
        "testRead": False,
    }
    (output / "summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate nested GT convergence from the formal plan and raw JSONL pair."
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument(
        "--plan", "--plan-jsonl", dest="plan", type=Path, required=True,
        help="100x128 nested plan JSONL",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--raw", "--raw-jsonl", "--raw-dir", dest="raw", type=Path, required=True,
        help="matching raw JSONL file or shard directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-counts",
        default=",".join(str(value) for value in DEFAULT_SAMPLE_COUNTS),
    )
    args = parser.parse_args()
    evaluate(
        scene=args.scene,
        plan_path=args.plan,
        raw_path=args.raw,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        sample_counts=parse_counts(args.sample_counts),
    )


if __name__ == "__main__":
    main()
