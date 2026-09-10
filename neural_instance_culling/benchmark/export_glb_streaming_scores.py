#!/usr/bin/env python3
"""Export aligned continuous per-candidate scores for GLB streaming."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BENCHMARK_DIR = Path(__file__).resolve().parent
MODEL_DIR = BENCHMARK_DIR.parent / "model"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from model_runners import load_runner, load_runtime_meta, select_device  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402
from score_sidecar import SIDECAR_SCHEMA, read_score_sidecar  # noqa: E402


FORMAL_AABB_SIDECAR_KINDS = {
    "formal_test_sidecar",
    "formal_aabb_test_sidecar",
    "aabb_test_sidecar",
}
AABB_RUNNER_KINDS = {"aabb", "aabb_ray", "learned_aabb_ray"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export aligned continuous scores for strict cold-cache GLB streaming."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        help=(
            "name|kind|checkpoint|runtime_features|calibration_summary; use '-' "
            "for unused static-runner fields"
        ),
    )
    parser.add_argument(
        "--full-test-sidecar",
        type=Path,
        default=None,
        help="frozen Full-model test score sidecar or its test evaluation JSON",
    )
    parser.add_argument(
        "--aabb-test-sidecar",
        dest="aabb_test_sidecar",
        type=Path,
        default=None,
        help=(
            "formal AABB MLP test score sidecar directory/manifest or its test "
            "evaluation JSON; its continuous scores are copied into the aabb field"
        ),
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["test", "validation", "calibration", "train"], default="test")
    parser.add_argument("--pose-limit", type=int, default=0)
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def parse_model_spec(value: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 5 or any(not part for part in parts[:2]):
        raise ValueError(
            "--model-spec must use name|kind|checkpoint|runtime_features|calibration_summary"
        )
    name, kind, checkpoint, runtime_features, eval_summary = parts
    spec = {
        "kind": kind,
        "checkpoint": checkpoint,
        "runtime_features": runtime_features,
        "eval_summary": eval_summary,
    }
    return name, spec


def _clean_optional_path(value: str) -> str:
    return "" if value == "-" else value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing {label}: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_formal_score_manifest(
    path: str | Path,
    method: str,
) -> tuple[Path, Path | None, dict[str, Any]]:
    """Resolve a frozen-test sidecar or its formal evaluation summary."""

    source = Path(path).expanduser().resolve()
    manifest_path = source / "manifest.json" if source.is_dir() else source
    payload = _read_json_object(manifest_path, f"{method} sidecar manifest")
    evaluation_summary: Path | None = None
    if payload.get("schema") != SIDECAR_SCHEMA:
        sidecar_value = payload.get("scoreSidecar")
        if sidecar_value is None:
            raise ValueError(
                f"{method} input must be a pvs-typed-score-sidecar-v1 manifest "
                "or a formal test evaluation JSON with scoreSidecar"
            )
        if payload.get("split") != "test" or payload.get("testRead") is not True:
            raise ValueError(f"{method} evaluation summary must be a frozen test result")
        evaluation_summary = manifest_path
        candidate = Path(str(sidecar_value)).expanduser()
        manifest_path = (candidate if candidate.is_absolute() else manifest_path.parent / candidate)
        if manifest_path.is_dir():
            manifest_path = manifest_path / "manifest.json"
        payload = _read_json_object(manifest_path.resolve(), f"{method} score sidecar manifest")
    return manifest_path.resolve(), evaluation_summary, payload


def load_formal_test_sidecar(
    path: str | Path,
    dataset: Any,
    selected_pose_ids: np.ndarray,
    method: str,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Load checkpoint-specific continuous scores from a frozen test sidecar.

    The returned arrays remain aligned to the stored CSR candidate order.  No
    prediction IDs or thresholded values are used for streaming ranking.
    """

    manifest_path, evaluation_summary, manifest = _resolve_formal_score_manifest(path, method)
    if manifest.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"unsupported {method} score sidecar schema: {manifest.get('schema')!r}")
    if manifest.get("split") != "test" or manifest.get("testRead") is not True:
        raise ValueError(f"formal {method} score sidecar must declare split=test and testRead=true")
    arrays = read_score_sidecar(manifest_path)
    pose_indices = np.asarray(arrays["poseIndices"], dtype=np.int64)
    pose_offsets = np.asarray(arrays["poseOffsets"], dtype=np.int64)
    candidate_ids = np.asarray(arrays["candidateIds"], dtype=np.uint32)
    scores = np.asarray(arrays["scores"], dtype=np.float32)
    if pose_offsets.size != pose_indices.size + 1:
        raise ValueError(f"formal {method} sidecar pose offsets are invalid")
    if np.unique(pose_indices).size != pose_indices.size:
        raise ValueError(f"formal {method} sidecar contains duplicate pose IDs")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError(f"formal {method} sidecar scores must be finite probabilities in [0, 1]")
    if int(manifest.get("poseCount", -1)) != pose_indices.size:
        raise ValueError(f"formal {method} sidecar pose count disagrees with poseIndices")
    if int(manifest.get("candidateCount", -1)) != candidate_ids.size:
        raise ValueError(f"formal {method} sidecar candidate count disagrees with candidateIds")

    dataset_split = getattr(dataset, "split", None)
    if callable(dataset_split):
        expected_test_pose_ids = np.asarray(
            dataset_split("test").pose_indices,
            dtype=np.int64,
        )
        if set(int(value) for value in pose_indices.tolist()) != set(
            int(value) for value in expected_test_pose_ids.tolist()
        ):
            raise ValueError(f"formal {method} sidecar must cover the complete dataset test split")

    row_by_pose = {int(pose_id): index for index, pose_id in enumerate(pose_indices.tolist())}
    selected = [int(value) for value in np.asarray(selected_pose_ids, dtype=np.int64).tolist()]
    if len(selected) != len(set(selected)):
        raise ValueError(f"selected {method} test pose IDs must be unique")
    missing = [pose_id for pose_id in selected if pose_id not in row_by_pose]
    if missing:
        raise ValueError(f"formal {method} sidecar is missing selected test poses: {missing[:8]}")
    by_pose: dict[int, np.ndarray] = {}
    for pose_id in selected:
        row = row_by_pose[pose_id]
        start, end = int(pose_offsets[row]), int(pose_offsets[row + 1])
        expected = np.asarray(dataset.candidate_slice(pose_id), dtype=np.uint32)
        actual = candidate_ids[start:end]
        if not np.array_equal(actual, expected):
            raise ValueError(
                f"formal AABB sidecar candidate rows disagree with CSR at pose {pose_id}"
            )
        values = scores[start:end]
        if values.shape != expected.shape:
            raise ValueError(f"formal {method} sidecar scores are misaligned at pose {pose_id}")
        by_pose[pose_id] = values.copy()

    threshold = manifest.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("formal AABB sidecar threshold is invalid")
    source = {
        "kind": f"formal_{method}_test_sidecar",
        "schema": SIDECAR_SCHEMA,
        "manifest": str(manifest_path),
        "evaluationSummary": str(evaluation_summary) if evaluation_summary else None,
        "split": "test",
        "testRead": True,
        "continuousScoreField": "scores",
        "threshold": threshold,
        "checkpoint": manifest.get("checkpoint"),
        "calibration": manifest.get("calibration"),
    }
    return by_pose, source


def load_formal_aabb_test_sidecar(
    path: str | Path,
    dataset: Any,
    selected_pose_ids: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    return load_formal_test_sidecar(path, dataset, selected_pose_ids, "aabb")


def _model_sources(
    specs: list[tuple[str, dict[str, str]]],
    explicit_aabb_sidecar: Path | None,
) -> tuple[list[tuple[str, dict[str, str]]], Path | None, bool]:
    """Separate regular runners from the formal AABB artifact input.

    AABB runner specs are retained only as a request marker.  They never cause
    this exporter to run a fallback AABB implementation.
    """

    runner_specs: list[tuple[str, dict[str, str]]] = []
    aabb_sidecar = explicit_aabb_sidecar
    aabb_requested = explicit_aabb_sidecar is not None
    for name, spec in specs:
        kind = str(spec.get("kind", "")).strip().lower()
        if name == "aabb" or kind in AABB_RUNNER_KINDS or kind in FORMAL_AABB_SIDECAR_KINDS:
            aabb_requested = True
            if kind in FORMAL_AABB_SIDECAR_KINDS:
                if aabb_sidecar is not None:
                    raise ValueError("AABB formal sidecar was supplied twice")
                if not spec.get("checkpoint"):
                    raise ValueError("formal AABB sidecar model spec must put its path in checkpoint")
                aabb_sidecar = Path(spec["checkpoint"])
            continue
        runner_specs.append((name, spec))
    return runner_specs, aabb_sidecar, aabb_requested


def main() -> None:
    args = parse_args()
    if args.pose_limit < 0:
        raise ValueError("--pose-limit must be non-negative")
    if args.split == "test" and args.pose_limit != 0:
        raise ValueError("frozen test score export must cover the complete test split")
    if args.poses_per_batch < 1:
        raise ValueError("--poses-per-batch must be positive")
    if args.full_test_sidecar is not None and args.split != "test":
        raise ValueError("--full-test-sidecar is only valid for split=test")
    specs: list[tuple[str, dict[str, str]]] = [parse_model_spec(value) for value in args.model_spec]
    names = [name for name, _spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("--model-spec names must be unique")

    runtime_meta = args.runtime_meta.expanduser().resolve()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    world_aabbs, _instance_to_glb, runtime = load_runtime_meta(runtime_meta)
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(world_aabbs.shape[0]))
    split = dataset.split(args.split)
    pose_ids = split.pose_indices.astype(np.int64, copy=False)
    if args.pose_limit > 0:
        pose_ids = pose_ids[: args.pose_limit]
    if pose_ids.size == 0:
        raise ValueError(f"split {args.split} contains no selected poses")

    runner_specs, aabb_sidecar_path, aabb_requested = _model_sources(
        specs,
        args.aabb_test_sidecar,
    )
    if args.full_test_sidecar is not None and any(name == "full" for name, _spec in runner_specs):
        raise ValueError("Full scores were supplied both as a runner and a frozen test sidecar")
    if not runner_specs and aabb_sidecar_path is None and args.full_test_sidecar is None and not aabb_requested:
        raise ValueError("provide at least one model or frozen test sidecar")

    device = select_device(args.device) if runner_specs else None

    formal_full_scores: dict[int, np.ndarray] = {}
    formal_full_source: dict[str, Any] | None = None
    if args.full_test_sidecar is not None:
        formal_full_scores, formal_full_source = load_formal_test_sidecar(
            args.full_test_sidecar,
            dataset,
            pose_ids,
            "full",
        )

    formal_aabb_scores: dict[int, np.ndarray] = {}
    formal_aabb_source: dict[str, Any] | None = None
    unavailable_sources: dict[str, str] = {}
    if aabb_sidecar_path is not None:
        formal_aabb_scores, formal_aabb_source = load_formal_aabb_test_sidecar(
            aabb_sidecar_path,
            dataset,
            pose_ids,
        )
    else:
        unavailable_sources["aabb"] = "formal AABB test sidecar was not supplied"

    runners = []
    for name, parsed in runner_specs:
        spec = {key: _clean_optional_path(value) for key, value in parsed.items()}
        runners.append(
            (
                name,
                load_runner(
                    name,
                    spec,
                    runtime_meta,
                    device,
                    fallback_threshold=0.5,
                    dataset_dir=dataset_dir,
                ),
                spec,
            )
        )

    pose_offsets = [0]
    all_candidate_ids: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {name: [] for name, _runner, _spec in runners}
    if formal_full_source is not None:
        score_parts["full"] = []
    if formal_aabb_source is not None:
        score_parts["aabb"] = []
    score_thresholds: dict[str, float] = {}
    threshold_sources: dict[str, str] = {}
    score_sources: dict[str, dict[str, Any]] = {}
    for name, runner, spec in runners:
        score_sources[name] = {
            "kind": "model_runner",
            "runnerKind": str(runner.kind),
            "checkpoint": spec.get("checkpoint") or None,
            "runtimeFeatures": spec.get("runtime_features") or None,
            "calibrationSummary": spec.get("eval_summary") or None,
            "continuousScoreField": name,
        }
    if formal_aabb_source is not None:
        score_sources["aabb"] = formal_aabb_source
    if formal_full_source is not None:
        score_sources["full"] = formal_full_source
    start_time = time.perf_counter()
    rng = np.random.default_rng(20260909)
    for batch_start in range(0, pose_ids.size, args.poses_per_batch):
        batch_pose_ids = pose_ids[batch_start : batch_start + args.poses_per_batch]
        # No candidate cap or GT union is enabled here.  This is the exact
        # stored candidate set used by every later streaming method.
        batch = split.build_pose_set_batch(
            batch_pose_ids,
            world_aabbs,
            rng,
            include_empty=True,
        )
        actual_pose_ids = np.asarray(batch["pose_indices"], dtype=np.int64)
        if actual_pose_ids.tolist() != batch_pose_ids.tolist():
            raise ValueError("pose-set builder changed the selected pose order")
        ids = np.asarray(batch["instance"], dtype=np.uint32)
        all_candidate_ids.append(ids.copy())
        pose_offsets.extend(
            int(pose_offsets[-1] + (int(batch["pose_offsets"][index + 1]) - int(batch["pose_offsets"][index])))
            for index in range(batch_pose_ids.size)
        )
        for name, runner, spec in runners:
            if ids.size == 0:
                result_scores = np.zeros((0,), dtype=np.float32)
            else:
                result = runner.score_batch(batch)
                result_scores = np.asarray(result.scores, dtype=np.float32).reshape(-1)
                if result_scores.shape != ids.shape:
                    raise ValueError(
                        f"runner {name} returned {result_scores.size} scores for {ids.size} rows"
                    )
                if not np.all(np.isfinite(result_scores)):
                    raise ValueError(f"runner {name} returned non-finite scores")
            score_parts[name].append(result_scores.copy())
            if name not in score_thresholds:
                threshold = float(runner.threshold)
                score_thresholds[name] = threshold
                threshold_sources[name] = (
                    "checkpoint calibration" if runner.kind == "bounded_relation_survival_moment_v4" else "runner fallback; not a registered safety workpoint"
                )
        if formal_full_source is not None:
            formal_batch_scores = np.concatenate(
                [formal_full_scores[int(pose_id)] for pose_id in batch_pose_ids.tolist()]
            ) if batch_pose_ids.size else np.zeros((0,), dtype=np.float32)
            if formal_batch_scores.shape != ids.shape:
                raise ValueError("formal Full sidecar scores do not match the selected candidate rows")
            score_parts["full"].append(formal_batch_scores.astype(np.float32, copy=True))
            threshold = formal_full_source.get("threshold")
            if threshold is not None:
                score_thresholds["full"] = float(threshold)
                threshold_sources["full"] = "checkpoint calibration"
        if formal_aabb_source is not None:
            formal_batch_scores = np.concatenate(
                [formal_aabb_scores[int(pose_id)] for pose_id in batch_pose_ids.tolist()]
            ) if batch_pose_ids.size else np.zeros((0,), dtype=np.float32)
            if formal_batch_scores.shape != ids.shape:
                raise ValueError("formal AABB sidecar scores do not match the selected candidate rows")
            score_parts["aabb"].append(formal_batch_scores.astype(np.float32, copy=True))
            threshold = formal_aabb_source.get("threshold")
            if threshold is not None:
                score_thresholds["aabb"] = float(threshold)
                threshold_sources["aabb"] = "checkpoint calibration"
        completed = min(pose_ids.size, batch_start + batch_pose_ids.size)
        if args.log_every > 0 and (completed % args.log_every == 0 or completed == pose_ids.size):
            elapsed = time.perf_counter() - start_time
            print(
                f"[streaming-score] {completed}/{pose_ids.size} poses "
                f"rows={pose_offsets[-1]} elapsed={elapsed:.1f}s",
                flush=True,
            )

    candidate_ids = np.concatenate(all_candidate_ids).astype(np.uint32, copy=False)
    arrays: dict[str, Any] = {
        "pose_ids": pose_ids.astype(np.int64, copy=False),
        "pose_offsets": np.asarray(pose_offsets, dtype=np.int64),
        "candidate_ids": candidate_ids,
    }
    for name, parts in score_parts.items():
        values = np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros((0,), dtype=np.float32)
        if values.shape != candidate_ids.shape:
            raise ValueError(f"score field {name} is not aligned with candidate IDs")
        arrays[name] = values

    result_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = result_dir / "scores.npz"
    np.savez(arrays_path, **arrays)
    manifest = {
        "schema": "pvs-glb-streaming-score-sidecar-v1",
        "createdBy": "export_glb_streaming_scores.py",
        "split": args.split,
        "testRead": args.split == "test",
        "poseCount": int(pose_ids.size),
        "candidateRowCount": int(candidate_ids.size),
        "numInstances": int(world_aabbs.shape[0]),
        "scoreFields": list(score_parts),
        "arraysFile": arrays_path.name,
        "thresholds": score_thresholds,
        "thresholdSources": threshold_sources,
        "scoreSources": score_sources,
        "unavailableSources": unavailable_sources,
        "source": {
            "datasetDir": str(dataset_dir),
            "runtimeMeta": str(runtime_meta),
            "runtimeSchema": runtime.get("schemaVersion"),
            "candidateSemantics": dataset.meta.get("candidateSemantics"),
            "gtSemantics": dataset.meta.get("gtSemantics"),
        },
        "alignment": {
            "poseIds": "pose_ids array in selected split order",
            "candidateIds": "candidate_ids array equals PoseCSR candidate_ids for each pose",
            "offsets": "pose_offsets indexes candidate_ids and every score field",
        },
    }
    (result_dir / "score_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"resultDir": str(result_dir), **{key: manifest[key] for key in ("split", "poseCount", "candidateRowCount", "scoreFields")}}, indent=2))


if __name__ == "__main__":
    main()
