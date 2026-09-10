#!/usr/bin/env python3
"""Run the paper's strict cold-cache GLB streaming benchmark.

All input paths are required.  The selected CSR candidate instances are
mapped to GLBs once per pose; every ranking method receives exactly that same
candidate GLB set.  Threshold filtering is emitted in a separate summary and
is never used to compute ranking curves.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

from glb_streaming import (  # noqa: E402
    BANDWIDTHS_MBPS,
    DECISION_MODE_SCHEDULER_REPLAY,
    DECISION_MODE_THRESHOLD_FILTERING,
    DECISION_MODE_THRESHOLD_FREE_RANKING,
    GlbAsset,
    NEURAL_COST_ALPHA_CANDIDATES,
    NEURAL_COST_METHOD,
    NEURAL_PROBABILITY_METHOD,
    PoseRecord,
    RANKING_METHODS,
    TARGET_COVERAGES,
    StreamingContractError,
    apply_neural_cost_alpha,
    build_fractional_utility_byte_lower_bound,
    coverage_metadata,
    neural_cost_method_for_alpha,
    select_neural_cost_alpha,
    simulate_ranked_pose,
    simulate_threshold_filter_pose,
    summarize_filter_results,
    summarize_ranking_results,
)
from glb_streaming_io import (  # noqa: E402
    attach_geometry_scores,
    load_neural_cost_manifest,
    load_glb_assets,
    load_pose_inputs,
    load_score_sidecar,
    write_neural_cost_manifest,
)
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


FORMAL_HZB_RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v2"
FORMAL_HZB_WORKLOAD_SCHEMA = "geometry-shell-hzb-browser-workload-v1"
FORMAL_HZB_MODE = "Region66"
FORMAL_HZB_REGION_FOV_DEG = 66.0
FORMAL_HZB_RENDER_FOV_DEG = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate strict cold-cache GLB streaming.")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True, help="continuous score sidecar directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation", "calibration", "train"], default="test")
    parser.add_argument("--pose-limit", type=int, default=0)
    parser.add_argument(
        "--utility-source",
        choices=["binary_gt", "visible_weights", "reference_frontmost_pixels"],
        default="visible_weights",
    )
    parser.add_argument(
        "--reference-frontmost",
        type=Path,
        default=None,
        help="JSONL sidecar from build_reference_frontmost_histogram.py",
    )
    parser.add_argument(
        "--hzb-region66-result",
        dest="hzb_region66_result",
        type=Path,
        default=None,
        help="formal Region66 test result JSON; visibleInstanceIds are mapped to GLBs",
    )
    parser.add_argument(
        "--methods",
        default=",".join(RANKING_METHODS),
        help="comma-separated ranking methods; defaults to the complete paper set",
    )
    parser.add_argument(
        "--neural-cost-manifest",
        type=Path,
        default=None,
        help="frozen validation-selected alpha manifest; required for test neural-cost ranking",
    )
    parser.add_argument(
        "--filter-thresholds",
        type=Path,
        default=None,
        help="optional JSON mapping of frozen threshold-filter fields to thresholds",
    )
    parser.add_argument("--include-runner-fallback-thresholds", action="store_true")
    parser.add_argument("--log-every", type=int, default=100, help="print progress after this many poses; zero disables progress")
    return parser.parse_args()


def _read_thresholds(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("thresholds", payload) if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        raise StreamingContractError("filter threshold file must be a method-to-threshold object")
    result: dict[str, float] = {}
    for name, value in values.items():
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise StreamingContractError(f"invalid filter threshold for {name}: {value}")
        result[str(name)] = threshold
    return result


def _load_manifest_thresholds(manifest: dict[str, Any], include_fallback: bool) -> dict[str, float]:
    thresholds = manifest.get("thresholds", {})
    sources = manifest.get("thresholdSources", {})
    score_sources = manifest.get("scoreSources", {})
    result: dict[str, float] = {}
    for name, value in thresholds.items():
        source = str(sources.get(name, ""))
        if name == "aabb":
            aabb_source = score_sources.get("aabb") if isinstance(score_sources, dict) else None
            if (
                not isinstance(aabb_source, dict)
                or aabb_source.get("kind") != "formal_aabb_test_sidecar"
                or aabb_source.get("split") != "test"
                or aabb_source.get("testRead") is not True
            ):
                continue
        if include_fallback or source == "checkpoint calibration":
            result[str(name)] = float(value)
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _coverage_source_name(utility_source: str) -> str:
    return coverage_metadata(utility_source)["source"]


def _formal_aabb_source(source: Any) -> bool:
    return (
        isinstance(source, dict)
        and source.get("kind") == "formal_aabb_test_sidecar"
        and source.get("split") == "test"
        and source.get("testRead") is True
    )


def require_complete_test_pose_selection(
    dataset: PoseCSRDataset,
    selected_pose_ids: Iterable[int],
    *,
    pose_limit: int = 0,
    label: str = "formal test mode",
) -> list[int]:
    """Reject a partial test selection before it can be reported as formal."""

    if pose_limit:
        raise StreamingContractError(
            f"{label} rejects pose_limit; use every pose in the test split"
        )
    selected = [int(value) for value in selected_pose_ids]
    expected = dataset.split("test").pose_indices.astype(np.int64, copy=False).tolist()
    if selected != expected:
        raise StreamingContractError(
            f"{label} requires the complete test split; a test subset is not reportable"
        )
    return expected


def _read_json_object(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StreamingContractError(f"missing {label}: {resolved}") from error
    except json.JSONDecodeError as error:
        raise StreamingContractError(f"invalid {label}: {resolved}") from error
    if not isinstance(value, dict):
        raise StreamingContractError(f"{label} must be a JSON object: {resolved}")
    return value


def _formal_integer_list(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise StreamingContractError(f"{label} must be a list of instance IDs")
    result: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise StreamingContractError(f"{label}[{index}] is not a non-negative integer")
        result.append(int(item))
    if len(result) != len(set(result)):
        raise StreamingContractError(f"{label} contains duplicate IDs")
    return tuple(result)


def _finite_equal(value: Any, expected: float, label: str) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise StreamingContractError(f"{label} must be {expected:g}") from error
    if not np.isfinite(numeric) or not np.isclose(numeric, expected, rtol=0.0, atol=1e-6):
        raise StreamingContractError(f"{label} must be {expected:g}, got {value!r}")


def load_formal_region66_test_result(
    path: str | Path,
    selected_pose_ids: Iterable[int],
    *,
    expected_scene: str | None = None,
    expected_candidate_ids: Mapping[int, Sequence[int]] | None = None,
) -> tuple[dict[int, tuple[int, ...]], dict[str, Any]]:
    """Load the formal Region66 visible-instance set without inventing scores."""

    resolved = Path(path).expanduser().resolve()
    result = _read_json_object(resolved, "Region66 HZB result")
    if result.get("schema") != FORMAL_HZB_RESULT_SCHEMA:
        raise StreamingContractError(
            "Region66 HZB input must use geometry-shell-hzb-browser-result-v2"
        )
    if result.get("mode") != FORMAL_HZB_MODE:
        raise StreamingContractError("HZB visible-first requires mode=Region66")
    if result.get("error"):
        raise StreamingContractError("Region66 HZB result records an execution error")
    if result.get("formalReady") is not True:
        raise StreamingContractError("Region66 HZB result is not formalReady=true")
    if result.get("executionClass") != "formal-hardware-gpu":
        raise StreamingContractError(
            "Region66 HZB result must declare executionClass=formal-hardware-gpu"
        )
    gpu_gate = result.get("gpuGate")
    if not isinstance(gpu_gate, dict) or gpu_gate.get("required") is not True or gpu_gate.get("hardware") is not True:
        raise StreamingContractError("Region66 HZB result did not pass its formal hardware GPU gate")

    workload = result.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("schema") != FORMAL_HZB_WORKLOAD_SCHEMA
        or workload.get("split") != "test"
    ):
        raise StreamingContractError("Region66 HZB result must contain a test browser workload")
    workload_scene = workload.get("scene")
    if not isinstance(workload_scene, str) or not workload_scene:
        raise StreamingContractError("Region66 HZB workload must declare a scene")
    if expected_scene is not None and workload_scene != str(expected_scene):
        raise StreamingContractError(
            f"Region66 HZB scene disagrees with runtime metadata: {workload_scene!r} != {expected_scene!r}"
        )
    _finite_equal(workload.get("fovYDeg"), FORMAL_HZB_REGION_FOV_DEG, "HZB workload fovYDeg")

    provenance = workload.get("provenance")
    if not isinstance(provenance, dict):
        raise StreamingContractError("Region66 HZB workload is missing provenance")
    configuration = provenance.get("configuration")
    if not isinstance(configuration, dict):
        raise StreamingContractError("Region66 HZB workload provenance is missing configuration")
    _finite_equal(configuration.get("fovYDeg"), FORMAL_HZB_RENDER_FOV_DEG, "HZB render fovYDeg")
    _finite_equal(configuration.get("regionFovYDeg"), FORMAL_HZB_REGION_FOV_DEG, "HZB regionFovYDeg")

    pose_selection = workload.get("poseSelection")
    if not isinstance(pose_selection, dict):
        pose_selection = provenance.get("poseSelection")
    if not isinstance(pose_selection, dict):
        raise StreamingContractError("Region66 HZB workload is missing pose selection provenance")
    if pose_selection.get("split") != "test":
        raise StreamingContractError("Region66 HZB pose selection must declare split=test")
    selected_in_result = pose_selection.get("selectedPoseIndices")
    if not isinstance(selected_in_result, list):
        raise StreamingContractError("Region66 HZB pose selection must list selectedPoseIndices")
    selected_in_result = _formal_integer_list(
        selected_in_result,
        "Region66 poseSelection.selectedPoseIndices",
    )
    if pose_selection.get("selectedPoseCount") != len(selected_in_result):
        raise StreamingContractError("Region66 HZB pose selection count is inconsistent")
    if pose_selection.get("limit") not in (None, 0):
        raise StreamingContractError("formal Region66 HZB result must not truncate test poses")
    if pose_selection.get("representative") is not True:
        raise StreamingContractError("formal Region66 HZB result must be representative of the test split")

    samples = result.get("samples")
    if not isinstance(samples, list) or not samples:
        raise StreamingContractError("Region66 HZB result has no pose samples")
    if workload.get("poseCount") != len(samples):
        raise StreamingContractError("Region66 HZB workload poseCount disagrees with result samples")
    candidate_rows: np.ndarray | None = None
    if expected_candidate_ids is not None:
        candidate_file = workload.get("candidateFile")
        if not isinstance(candidate_file, str) or not candidate_file or Path(candidate_file).name != candidate_file:
            raise StreamingContractError("Region66 HZB workload has an invalid candidateFile")
        if workload.get("candidateDtype") != "uint32-little-endian":
            raise StreamingContractError("Region66 HZB candidate file must use uint32-little-endian")
        candidate_path = resolved.parent / candidate_file
        try:
            candidate_rows = np.fromfile(candidate_path, dtype="<u4")
        except OSError as error:
            raise StreamingContractError(f"cannot read Region66 HZB candidate file: {candidate_path}") from error
        if workload.get("candidateCount") != int(candidate_rows.size):
            raise StreamingContractError("Region66 HZB workload candidateCount disagrees with its candidate file")
    visible_by_pose: dict[int, tuple[int, ...]] = {}
    candidate_cursor = 0
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise StreamingContractError(f"Region66 HZB sample {index} is not an object")
        pose_id = sample.get("poseId")
        if isinstance(pose_id, bool) or not isinstance(pose_id, int) or pose_id < 0:
            raise StreamingContractError(f"Region66 HZB sample {index} has an invalid poseId")
        if pose_id in visible_by_pose:
            raise StreamingContractError(f"Region66 HZB result has duplicate poseId {pose_id}")
        candidate_count = sample.get("candidateCount")
        if isinstance(candidate_count, bool) or not isinstance(candidate_count, int) or candidate_count < 0:
            raise StreamingContractError(f"Region66 HZB pose {pose_id} has an invalid candidateCount")
        if expected_candidate_ids is not None:
            expected_ids = expected_candidate_ids.get(int(pose_id))
            if expected_ids is None:
                raise StreamingContractError(
                    f"Region66 HZB pose {pose_id} is outside the expected CSR test split"
                )
            expected_row = np.asarray(expected_ids, dtype=np.uint32).reshape(-1)
            if candidate_count != int(expected_row.size):
                raise StreamingContractError(
                    f"Region66 HZB pose {pose_id} candidateCount {candidate_count} "
                    f"does not match CSR row length {expected_row.size}"
                )
            actual_row = candidate_rows[candidate_cursor : candidate_cursor + candidate_count]
            if not np.array_equal(actual_row, expected_row):
                raise StreamingContractError(
                    f"Region66 HZB pose {pose_id} candidate IDs do not match its CSR row"
                )
        candidate_cursor += candidate_count
        visible_by_pose[int(pose_id)] = _formal_integer_list(
            sample.get("visibleInstanceIds"),
            f"Region66 HZB pose {pose_id} visibleInstanceIds",
        )
    if set(visible_by_pose) != set(selected_in_result):
        raise StreamingContractError(
            "Region66 HZB samples do not exactly cover poseSelection.selectedPoseIndices"
        )
    if candidate_rows is not None and candidate_cursor != int(candidate_rows.size):
        raise StreamingContractError("Region66 HZB sample candidate counts do not consume its candidate file")

    requested = [int(value) for value in selected_pose_ids]
    if len(requested) != len(set(requested)):
        raise StreamingContractError("expected Region66 test pose IDs must be unique")
    missing = sorted(set(requested) - set(visible_by_pose))
    extra = sorted(set(visible_by_pose) - set(requested))
    if missing or extra:
        raise StreamingContractError(
            "formal Region66 HZB pose set does not equal the complete CSR test split: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    source = {
        "kind": "formal_region66_test_result",
        "schema": FORMAL_HZB_RESULT_SCHEMA,
        "path": str(resolved),
        "scene": workload_scene,
        "mode": FORMAL_HZB_MODE,
        "split": "test",
        "testRead": True,
        "formalReady": True,
        "executionClass": "formal-hardware-gpu",
        "visibleUnit": "instance_ids",
        "continuousScore": False,
        "poseCount": len(visible_by_pose),
        "selectedPoseCount": len(selected_in_result),
    }
    return visible_by_pose, source


def attach_formal_region66_visible_glbs(
    records: Sequence[PoseRecord],
    loaded_poses: Sequence[Any],
    visible_instance_ids_by_pose: Mapping[int, Sequence[int]],
    instance_to_glb: Mapping[int, int],
) -> list[PoseRecord]:
    """Map formal Region66 instance IDs to each pose's existing GLB candidate set."""

    loaded_by_pose = {int(loaded.record.pose_id): loaded for loaded in loaded_poses}
    result: list[PoseRecord] = []
    for pose in records:
        loaded = loaded_by_pose.get(int(pose.pose_id))
        if loaded is None:
            raise StreamingContractError(f"no CSR input row for pose {pose.pose_id}")
        if pose.pose_id not in visible_instance_ids_by_pose:
            raise StreamingContractError(f"Region66 result has no visible set for pose {pose.pose_id}")
        candidate_instances = set(int(value) for value in loaded.candidate_instance_ids.tolist())
        visible_instances = tuple(int(value) for value in visible_instance_ids_by_pose[pose.pose_id])
        unknown_instances = sorted(set(visible_instances) - set(instance_to_glb))
        if unknown_instances:
            raise StreamingContractError(
                f"Region66 pose {pose.pose_id} contains unknown instance IDs: {unknown_instances[:8]}"
            )
        outside = sorted(set(visible_instances) - candidate_instances)
        if outside:
            raise StreamingContractError(
                f"Region66 pose {pose.pose_id} visible instances are outside its CSR candidates: {outside[:8]}"
            )
        visible_glbs = tuple(sorted({int(instance_to_glb[instance_id]) for instance_id in visible_instances}))
        if not set(visible_glbs).issubset(set(pose.candidate_glb_ids)):
            raise StreamingContractError(
                f"Region66 pose {pose.pose_id} visible GLBs are outside its candidate GLB set"
            )
        result.append(replace(pose, hzb_visible_glb_ids=visible_glbs))
    return result


def ordered_hzb_glb_ids_for_pose(pose: PoseRecord, assets: Mapping[int, GlbAsset]) -> tuple[int, ...]:
    """Return the formal HZB visible-first permutation without HZB scores."""

    if pose.hzb_visible_glb_ids is None:
        raise StreamingContractError(f"pose {pose.pose_id} has no formal Region66 visible set")
    candidate = set(pose.candidate_glb_ids)
    visible = set(pose.hzb_visible_glb_ids)
    unknown = visible - candidate
    if unknown:
        raise StreamingContractError(
            f"pose {pose.pose_id} HZB visible set contains non-candidate GLBs: {sorted(unknown)[:8]}"
        )

    visible_order_rule = None
    visible_scores: Mapping[int, Any] = {}
    for rule in ("projected_area_per_byte", "projected_area"):
        values = pose.rank_scores.get(rule)
        finite_visible_scores = False
        if isinstance(values, Mapping) and visible.issubset(values):
            try:
                finite_visible_scores = all(
                    np.isfinite(float(values[glb_id])) for glb_id in visible
                )
            except (TypeError, ValueError):
                finite_visible_scores = False
        if (
            isinstance(values, Mapping)
            and visible.issubset(values)
            and finite_visible_scores
        ):
            visible_order_rule = rule
            visible_scores = values
            break
    if visible and visible_order_rule is None:
        raise StreamingContractError(
            f"pose {pose.pose_id} has no registered projected-area score for HZB visible-first ordering"
        )

    visible_order = sorted(
        visible,
        key=lambda glb_id: (-float(visible_scores.get(glb_id, 0.0)), int(glb_id)),
    )
    invisible_order = sorted(
        candidate - visible,
        key=lambda glb_id: (int(assets[glb_id].original_rank), int(glb_id)),
    )
    return tuple(visible_order + invisible_order)


def simulate_hzb_ranked_pose(
    pose: PoseRecord,
    assets: Mapping[int, GlbAsset],
    *,
    target_coverages: Sequence[float] = TARGET_COVERAGES,
    bandwidths_mbps: Sequence[int] = BANDWIDTHS_MBPS,
) -> dict[str, Any]:
    """Replay the formal HZB permutation using the common cold-cache metrics."""

    ordered = ordered_hzb_glb_ids_for_pose(pose, assets)
    # ``simulate_ranked_pose`` already implements the atomic-GLB accounting.
    # Reusing its original-order branch through a private asset view preserves
    # the exact permutation without manufacturing a continuous HZB score map.
    permutation_assets = {
        int(glb_id): replace(assets[int(glb_id)], original_rank=index)
        for index, glb_id in enumerate(ordered)
    }
    result = simulate_ranked_pose(
        pose,
        permutation_assets,
        "original",
        target_coverages=target_coverages,
        bandwidths_mbps=bandwidths_mbps,
    )
    result["method"] = "hzb_visible_first"
    result["rankingInput"] = hzb_ordering_metadata()
    return result


def hzb_ordering_metadata() -> dict[str, Any]:
    return {
        "kind": "formal_region66_visible_set",
        "continuousScore": False,
        "visibleGroup": "formal Region66 visibleInstanceIds mapped to GLBs",
        "visibleGroupOrder": "projected_area_per_byte descending, globalGlbId ascending; projected_area fallback",
        "nonVisibleGroupOrder": "original GLB index ascending, globalGlbId ascending",
    }


def main() -> None:
    args = parse_args()
    if args.pose_limit < 0:
        raise ValueError("--pose-limit must be non-negative")
    if args.split == "test" and args.pose_limit:
        raise StreamingContractError(
            "formal test mode rejects --pose-limit; a complete test split is required"
        )
    if args.log_every < 0:
        raise ValueError("--log-every must be non-negative")
    methods = tuple(name.strip() for name in args.methods.split(",") if name.strip())
    supported_methods = set(RANKING_METHODS) | {NEURAL_PROBABILITY_METHOD} | {
        neural_cost_method_for_alpha(alpha) for alpha in NEURAL_COST_ALPHA_CANDIDATES
    }
    unknown = [name for name in methods if name not in supported_methods]
    if unknown:
        raise ValueError(f"unknown ranking methods: {unknown}")
    if not methods:
        raise ValueError("--methods selected no methods")
    explicit_cost_methods = {
        neural_cost_method_for_alpha(alpha) for alpha in NEURAL_COST_ALPHA_CANDIDATES
    }
    if args.split == "test" and any(method in explicit_cost_methods for method in methods):
        raise StreamingContractError(
            "test ranking may use only neural_cost with a frozen validation manifest; "
            "alpha candidates are validation-only"
        )
    if args.utility_source == "reference_frontmost_pixels" and args.reference_frontmost is None:
        raise ValueError("reference-frontmost-pixels requires --reference-frontmost")

    assets, instance_to_glb, asset_meta = load_glb_assets(
        args.runtime_meta,
        args.glb_index,
        args.glb_root,
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_for_selection = PoseCSRDataset(
        args.dataset_dir,
        num_instances=len(instance_to_glb),
    )
    loaded_poses = load_pose_inputs(
        args.dataset_dir,
        args.runtime_meta,
        assets,
        instance_to_glb,
        split=args.split,
        pose_limit=args.pose_limit,
        utility_source=args.utility_source,
        reference_frontmost_path=args.reference_frontmost,
    )
    selected_pose_ids = [pose.record.pose_id for pose in loaded_poses]
    complete_test_pose_ids: list[int] | None = None
    if args.split == "test":
        complete_test_pose_ids = require_complete_test_pose_selection(
            dataset_for_selection,
            selected_pose_ids,
            pose_limit=args.pose_limit,
        )
    score_by_pose, score_manifest = load_score_sidecar(
        args.result_dir,
        args.dataset_dir,
        selected_pose_ids,
        require_complete_test=args.split == "test",
    )
    records = attach_geometry_scores(loaded_poses, assets, score_by_pose, instance_to_glb)

    neural_cost_manifest: dict[str, Any] | None = None
    neural_cost_requested = NEURAL_COST_METHOD in methods or bool(
        set(methods) & explicit_cost_methods
    )
    if neural_cost_requested:
        if args.split == "validation":
            if args.neural_cost_manifest is not None:
                raise StreamingContractError(
                    "validation alpha selection writes a new frozen manifest; it does not read a test manifest"
                )
            neural_cost_manifest = select_neural_cost_alpha(
                records,
                assets,
                selection_split=args.split,
            )
            manifest_path = write_neural_cost_manifest(
                output_dir / "neural_cost_ranking_manifest.json",
                neural_cost_manifest,
            )
            neural_cost_manifest = {
                **neural_cost_manifest,
                "path": str(manifest_path),
            }
        elif args.split == "test":
            manifest_path = args.neural_cost_manifest
            if manifest_path is None:
                raise StreamingContractError(
                    "test neural-cost ranking requires --neural-cost-manifest from validation"
                )
            neural_cost_manifest = load_neural_cost_manifest(manifest_path)
            test_coverage_sources = {pose.coverage_source for pose in records}
            if test_coverage_sources != {neural_cost_manifest["coverageSource"]}:
                raise StreamingContractError(
                    "test coverage definition does not match the validation alpha manifest"
                )
        else:
            raise StreamingContractError(
                "neural cost alpha must be selected on validation or read from a frozen manifest for test"
            )
        if NEURAL_COST_METHOD in methods:
            records = apply_neural_cost_alpha(
                records,
                neural_cost_manifest["selectedAlpha"],
            )

    hzb_source: dict[str, Any] | None = None
    hzb_unavailable_reason: str | None = None
    if args.hzb_region66_result is None:
        hzb_unavailable_reason = "formal Region66 test result was not supplied"
    elif args.split != "test":
        hzb_unavailable_reason = "formal Region66 visible-first input is test-only"
    else:
        try:
            if complete_test_pose_ids is None:
                raise StreamingContractError("complete test pose selection was not established")
            expected_candidate_ids = {
                int(pose_id): dataset_for_selection.candidate_slice(int(pose_id))
                for pose_id in complete_test_pose_ids
            }
            hzb_instances, hzb_source = load_formal_region66_test_result(
                args.hzb_region66_result,
                complete_test_pose_ids,
                expected_scene=str(asset_meta.get("sceneName", "")),
                expected_candidate_ids=expected_candidate_ids,
            )
            records = attach_formal_region66_visible_glbs(
                records,
                loaded_poses,
                hzb_instances,
                instance_to_glb,
            )
        except (FileNotFoundError, OSError, StreamingContractError, ValueError) as error:
            hzb_source = None
            hzb_unavailable_reason = f"formal Region66 result unavailable: {error}"

    ranking_rows: list[dict[str, Any]] = []
    unavailable_reasons: dict[str, str] = {
        str(name): str(reason)
        for name, reason in (score_manifest.get("unavailableSources", {}) or {}).items()
    }
    score_sources = score_manifest.get("scoreSources", {})
    if "aabb" in methods and not _formal_aabb_source(
        score_sources.get("aabb") if isinstance(score_sources, dict) else None
    ):
        unavailable_reasons["aabb"] = (
            "formal AABB test sidecar is unavailable; legacy runner/fallback scores are not accepted"
        )
    if hzb_unavailable_reason is not None:
        unavailable_reasons["hzb_visible_first"] = hzb_unavailable_reason
    fractional_bounds: dict[str, list[int | None]] = {
        str(target): [] for target in TARGET_COVERAGES
    }
    hzb_ordering_ready = hzb_source is not None
    if hzb_source is not None and "hzb_visible_first" in methods:
        try:
            for pose in records:
                ordered_hzb_glb_ids_for_pose(pose, assets)
        except StreamingContractError as error:
            hzb_ordering_ready = False
            unavailable_reasons["hzb_visible_first"] = f"formal Region66 ordering unavailable: {error}"

    required_score_methods = {
        "full",
        NEURAL_COST_METHOD,
        "aabb",
        "distance",
        "projected_area",
        "projected_area_per_byte",
    }
    for method in methods:
        if method == NEURAL_COST_METHOD:
            missing_neural_cost_input = any(
                pose.neural_cost_alpha is None
                or not any(
                    name in pose.rank_scores
                    for name in ("full", "neural_probability", "neural")
                )
                for pose in records
            )
        else:
            missing_neural_cost_input = False
        if (
            method in required_score_methods
            and (
                missing_neural_cost_input
                or (
                    method != NEURAL_COST_METHOD
                    and any(method not in pose.rank_scores for pose in records)
                )
            )
        ):
            unavailable_reasons.setdefault(
                method,
                f"continuous score input for {method} is unavailable for one or more selected poses",
            )
        if method == "hzb_visible_first" and not hzb_ordering_ready:
            unavailable_reasons.setdefault(
                method,
                hzb_unavailable_reason or "formal Region66 visible-first ordering is unavailable",
            )

    simulation_started = time.perf_counter()
    for ordinal, pose in enumerate(records, start=1):
        method_rows: dict[str, dict[str, Any]] = {}
        ranking_inputs: dict[str, dict[str, Any]] = {}
        for method in methods:
            if method in unavailable_reasons:
                continue
            try:
                method_rows[method] = (
                    simulate_hzb_ranked_pose(pose, assets)
                    if method == "hzb_visible_first"
                    else simulate_ranked_pose(pose, assets, method)
                )
                ranking_inputs[method] = method_rows[method].get("rankingInput")
                if method == "hzb_visible_first":
                    ranking_inputs[method] = hzb_ordering_metadata()
            except StreamingContractError as error:
                if method in required_score_methods or method == "hzb_visible_first":
                    unavailable_reasons[method] = str(error)
                    continue
                raise
        ranking_rows.append(
            {
                "poseId": int(pose.pose_id),
                "ordinal": int(pose.ordinal),
                "candidateGlbIds": list(pose.candidate_glb_ids),
                "gtGlbIds": list(pose.gt_glb_ids),
                "methods": method_rows,
                "rankingInputs": ranking_inputs,
            }
        )
        for target in TARGET_COVERAGES:
            fractional_bounds[str(target)].append(
                build_fractional_utility_byte_lower_bound(pose, assets, float(target))
            )
        if args.log_every > 0 and (ordinal % args.log_every == 0 or ordinal == len(records)):
            elapsed = time.perf_counter() - simulation_started
            print(
                f"[streaming-sim] {ordinal}/{len(records)} poses elapsed={elapsed:.1f}s",
                flush=True,
            )

    ranking_summary = summarize_ranking_results(
        ranking_rows,
        methods=methods,
        target_coverages=TARGET_COVERAGES,
        bandwidths_mbps=BANDWIDTHS_MBPS,
    )
    for method, reason in unavailable_reasons.items():
        method_summary = ranking_summary.get("methods", {}).get(method)
        if isinstance(method_summary, dict) and method_summary.get("status") != "available":
            method_summary["reason"] = reason
    ranking_summary["scene"] = asset_meta
    ranking_summary["split"] = args.split
    ranking_summary["testRead"] = args.split == "test"
    ranking_summary["poseCount"] = len(records)
    ranking_summary["testPoseCount"] = (
        len(complete_test_pose_ids) if complete_test_pose_ids is not None else None
    )
    ranking_summary["poseSelection"] = {
        "split": args.split,
        "selectedPoseCount": len(records),
        "limit": int(args.pose_limit),
        "completeTest": args.split == "test",
    }
    coverage = coverage_metadata(args.utility_source)
    ranking_summary["coverageSource"] = coverage["source"]
    ranking_summary["coverageMetric"] = coverage["metric"]
    ranking_summary["coverageUnit"] = coverage["unit"]
    ranking_summary["coverageSemantics"] = coverage["semantics"]
    ranking_summary["completeTestSplit"] = args.split == "test"
    ranking_summary["scoreSources"] = score_sources if isinstance(score_sources, dict) else {}
    formal_aabb_input = (
        score_sources.get("aabb")
        if isinstance(score_sources, dict) and _formal_aabb_source(score_sources.get("aabb"))
        else None
    )
    ranking_inputs_by_method = {
        str(name): value
        for name, value in {
            method: next(
                (
                    row.get("rankingInputs", {}).get(method)
                    for row in ranking_rows
                    if isinstance(row.get("rankingInputs", {}).get(method), dict)
                ),
                None,
            )
            for method in methods
        }.items()
        if value is not None
    }
    aabb_ranking_input: dict[str, Any]
    if formal_aabb_input is not None:
        aabb_ranking_input = {
            **(ranking_inputs_by_method.get("aabb") or {}),
            **formal_aabb_input,
        }
    else:
        aabb_ranking_input = {
            "kind": "unavailable",
            "reason": unavailable_reasons.get("aabb", "no formal AABB score input"),
        }
    hzb_ranking_input: dict[str, Any]
    if hzb_source is not None:
        hzb_ranking_input = {
            **(ranking_inputs_by_method.get("hzb_visible_first") or {}),
            **hzb_source,
        }
    else:
        hzb_ranking_input = {
            "kind": "unavailable",
            "reason": unavailable_reasons.get("hzb_visible_first", "no formal Region66 test result"),
        }
    ranking_summary["rankingInputs"] = {
        **ranking_inputs_by_method,
        "aabb": aabb_ranking_input,
        "hzb_visible_first": hzb_ranking_input,
    }
    if neural_cost_manifest is not None:
        ranking_summary["neuralCostManifest"] = neural_cost_manifest
    ranking_summary["source"] = {
        "datasetDir": str(Path(args.dataset_dir).expanduser().resolve()),
        "runtimeMeta": str(Path(args.runtime_meta).expanduser().resolve()),
        "glbIndex": str(Path(args.glb_index).expanduser().resolve()),
        "glbRoot": str(Path(args.glb_root).expanduser().resolve()),
        "scoreResultDir": str(Path(args.result_dir).expanduser().resolve()),
        "referenceFrontmost": str(args.reference_frontmost.expanduser().resolve()) if args.reference_frontmost else None,
        "hzbRegion66Result": str(args.hzb_region66_result.expanduser().resolve()) if args.hzb_region66_result else None,
    }
    ranking_summary["unavailableReasons"] = unavailable_reasons
    ranking_summary["fractionalUtilityByteLowerBound"] = {
        target: {
            "count": int(sum(value is not None for value in values)),
            "mean": float(np.mean([value for value in values if value is not None]))
            if any(value is not None for value in values)
            else None,
            "median": float(np.median([value for value in values if value is not None]))
            if any(value is not None for value in values)
            else None,
        }
        for target, values in fractional_bounds.items()
    }

    filter_thresholds = {}
    if args.filter_thresholds is not None:
        filter_thresholds = _read_thresholds(args.filter_thresholds)
    else:
        filter_thresholds = _load_manifest_thresholds(
            score_manifest,
            args.include_runner_fallback_thresholds,
        )
    filter_rows: list[dict[str, Any]] = []
    filter_methods: list[str] = list(filter_thresholds)
    filter_unavailable_reasons: dict[str, str] = {}
    by_pose_loaded = {pose.record.pose_id: pose for pose in loaded_poses}
    for method, threshold in filter_thresholds.items():
        if not any(method in fields for fields in score_by_pose.values()):
            filter_unavailable_reasons[method] = f"no continuous score field for {method}"
            continue
    for pose in records:
        loaded = by_pose_loaded[pose.pose_id]
        method_rows: dict[str, dict[str, Any]] = {}
        for method in filter_methods:
            scores = score_by_pose[pose.pose_id].get(method)
            if scores is None:
                continue
            method_rows[method] = simulate_threshold_filter_pose(
                pose,
                assets,
                method=method,
                instance_ids=loaded.candidate_instance_ids.tolist(),
                instance_scores=scores.tolist(),
                instance_to_glb=instance_to_glb,
                threshold=filter_thresholds[method],
            )
        filter_rows.append(
            {
                "poseId": int(pose.pose_id),
                "ordinal": int(pose.ordinal),
                "methods": method_rows,
            }
        )
    filtering_summary = summarize_filter_results(
        filter_rows,
        methods=filter_methods,
        target_coverages=TARGET_COVERAGES,
    )
    filtering_summary["scene"] = asset_meta
    filtering_summary["split"] = args.split
    filtering_summary["testRead"] = args.split == "test"
    filtering_summary["poseCount"] = len(records)
    filtering_summary["testPoseCount"] = (
        len(complete_test_pose_ids) if complete_test_pose_ids is not None else None
    )
    filtering_summary["poseSelection"] = ranking_summary["poseSelection"]
    filtering_summary["coverageSource"] = coverage["source"]
    filtering_summary["coverageMetric"] = coverage["metric"]
    filtering_summary["coverageUnit"] = coverage["unit"]
    filtering_summary["coverageSemantics"] = coverage["semantics"]
    filtering_summary["completeTestSplit"] = args.split == "test"
    filtering_summary["source"] = ranking_summary["source"]
    filtering_summary["thresholdSources"] = score_manifest.get("thresholdSources", {})
    filtering_summary["scoreSources"] = ranking_summary["scoreSources"]
    filtering_summary["rankingInputs"] = ranking_summary["rankingInputs"]
    filtering_summary["unavailableReasons"] = filter_unavailable_reasons
    filtering_summary["neuralCostManifest"] = neural_cost_manifest
    for method, reason in filter_unavailable_reasons.items():
        method_summary = filtering_summary.get("methods", {}).get(method)
        if isinstance(method_summary, dict) and method_summary.get("status") != "available":
            method_summary["reason"] = reason
    filtering_summary["note"] = (
        "Threshold filtering is a separate decision mode; its predicted subset is not used by ranking metrics."
    )

    _write_jsonl(output_dir / "ranking_per_pose.jsonl", ranking_rows)
    _write_jsonl(output_dir / "filtering_per_pose.jsonl", filter_rows)
    (output_dir / "ranking_summary.json").write_text(
        json.dumps(ranking_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "filtering_summary.json").write_text(
        json.dumps(filtering_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "pvs-glb-streaming-experiment-v1",
        "decisionModes": [
            DECISION_MODE_THRESHOLD_FREE_RANKING,
            DECISION_MODE_THRESHOLD_FILTERING,
            DECISION_MODE_SCHEDULER_REPLAY,
        ],
        "resultTypes": {
            "ranking": DECISION_MODE_THRESHOLD_FREE_RANKING,
            "filtering": DECISION_MODE_THRESHOLD_FILTERING,
            "schedulerReplay": DECISION_MODE_SCHEDULER_REPLAY,
        },
        "cacheMode": "strict_cold_cache_per_pose",
        "arrivalSemantics": "GLB bytes and reference-frontmost utility accumulate only after complete GLB arrival",
        "split": args.split,
        "testRead": args.split == "test",
        "poseCount": len(records),
        "testPoseCount": (
            len(complete_test_pose_ids) if complete_test_pose_ids is not None else None
        ),
        "poseSelection": ranking_summary["poseSelection"],
        "methods": list(methods),
        "filterMethods": filter_methods,
        "filterThresholds": filter_thresholds,
        "coverageSource": coverage["source"],
        "coverageMetric": coverage["metric"],
        "coverageUnit": coverage["unit"],
        "coverageSemantics": coverage["semantics"],
        "completeTestSplit": args.split == "test",
        "scoreSources": ranking_summary["scoreSources"],
        "rankingInputs": ranking_summary["rankingInputs"],
        "rankingSummary": "ranking_summary.json",
        "filteringSummary": "filtering_summary.json",
        "rankingPerPose": "ranking_per_pose.jsonl",
        "filteringPerPose": "filtering_per_pose.jsonl",
        "source": ranking_summary["source"],
        "scoreManifest": score_manifest,
        "neuralCostManifest": neural_cost_manifest,
        "unavailableReasons": unavailable_reasons,
    }
    (output_dir / "streaming_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "outputDir": str(output_dir),
                "split": args.split,
                "poses": len(records),
                "rankingMethods": list(methods),
                "filterMethods": filter_methods,
                "unavailableReasons": unavailable_reasons,
                "coverageSource": _coverage_source_name(args.utility_source),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
