"""Export one V5 scene into the existing cold-cache streaming score contract."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import V5Run


STREAMING_SCORE_SCHEMA = "pvs-glb-streaming-score-sidecar-v1"


def _read_json(path: str | Path, label: str) -> Mapping[str, Any]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    result = np.empty_like(array)
    positive = array >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponential = np.exp(array[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    if array.ndim == 0:
        return float(result.item())
    return result.astype(np.float32)


def _frozen_threshold(
    summary: Mapping[str, Any],
    *,
    run: V5Run,
    scene: str,
    split: str,
) -> tuple[float, Mapping[str, Any]]:
    rows = summary.get("rows")
    if not isinstance(rows, list):
        raise ValueError("V5 evaluation summary must contain result rows")
    matches = [
        row for row in rows
        if isinstance(row, Mapping)
        and row.get("protocol") == run.protocol
        and row.get("variant") == run.variant
        and int(row.get("seed", -1)) == run.seed
        and row.get("scene") == scene
        and row.get("split") == split
        and row.get("threshold_mode") == "target_calibrated"
    ]
    if len(matches) != 1:
        raise ValueError("evaluation summary does not identify one matching frozen threshold row")
    row = matches[0]
    if (
        row.get("selection_split") != "calibration"
        or row.get("selection_test_read") is not False
        or row.get("test_read") is not (split == "test")
    ):
        raise ValueError("V5 streaming threshold is not calibration-frozen for the requested split")
    threshold = float(row.get("threshold"))
    if not np.isfinite(threshold):
        raise ValueError("V5 streaming threshold is non-finite")
    return threshold, row


def _expand_to_frozen_candidates(
    full_candidates: np.ndarray,
    retained_candidates: np.ndarray,
    retained_logits: np.ndarray,
    excluded: tuple[int, ...],
) -> np.ndarray:
    full = np.asarray(full_candidates, dtype=np.uint32).reshape(-1)
    retained = np.asarray(retained_candidates, dtype=np.uint32).reshape(-1)
    logits = np.asarray(retained_logits, dtype=np.float32).reshape(-1)
    if retained.size != logits.size:
        raise ValueError("V5 retained candidate/logit rows disagree")
    if full.size != np.unique(full).size or retained.size != np.unique(retained).size:
        raise ValueError("V5 streaming candidates must be unique per pose")
    lookup = {int(value): index for index, value in enumerate(full.tolist())}
    try:
        retained_positions = np.asarray([lookup[int(value)] for value in retained], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"V5 score contains a non-CSR candidate: {exc.args[0]}") from exc
    output = np.zeros((full.size,), dtype=np.float32)
    output[retained_positions] = np.asarray(_sigmoid(logits), dtype=np.float32)
    missing = set(int(value) for value in full.tolist()) - set(int(value) for value in retained.tolist())
    if missing != set(int(value) for value in excluded if int(value) in lookup):
        raise ValueError("V5 score omissions do not exactly match excluded non-renderable units")
    return output


def export_streaming_score_sidecar(
    *,
    bundle_manifest: str | Path,
    evaluation_summary: str | Path,
    scene: str,
    split: str,
    output_dir: str | Path,
    final_test: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write aligned sigmoid probabilities for one V5 scene and frozen split."""

    split_name = str(split)
    if split_name not in {"validation", "test"}:
        raise ValueError("V5 streaming export supports validation or test only")
    if split_name == "test" and not final_test:
        raise PermissionError("test streaming export requires explicit final-test permission")
    run = V5Run.from_manifest(bundle_manifest, allow_test_read=bool(final_test))
    if split_name == "validation" and run.test_read:
        raise PermissionError("test-tainted V5 bundles cannot select streaming parameters")
    scene_id = str(scene)
    if scene_id not in run.scenes:
        raise ValueError(f"V5 bundle has no scene {scene_id!r}")
    summary_path = Path(evaluation_summary).resolve()
    summary = _read_json(summary_path, "V5 evaluation summary")
    threshold_logit, threshold_row = _frozen_threshold(
        summary,
        run=run,
        scene=scene_id,
        split=split_name,
    )
    scene_scores = run.scenes[scene_id]
    sidecar = scene_scores._open_sidecar(split_name, allow_test=bool(final_test))
    csr = scene_scores._open_csr()
    expected_pose_ids = csr.split_indices(split_name)
    records = iter(scene_scores.records(split_name, allow_test=bool(final_test)))
    pose_offsets = [0]
    candidate_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    for expected_pose_id in expected_pose_ids.tolist():
        try:
            record = next(records)
        except StopIteration as exc:
            raise ValueError("V5 score stream ended before the frozen split") from exc
        if int(record.pose_index) != int(expected_pose_id):
            raise ValueError("V5 score stream changed frozen split order")
        candidates = np.asarray(
            csr.dataset.candidate_slice(int(expected_pose_id)), dtype=np.uint32
        )
        probabilities = _expand_to_frozen_candidates(
            candidates,
            record.candidate_ids,
            record.scores,
            sidecar.excluded_unit_ids,
        )
        candidate_parts.append(candidates.copy())
        probability_parts.append(probabilities)
        pose_offsets.append(pose_offsets[-1] + int(candidates.size))
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("V5 score stream contains poses outside the frozen split")

    candidates = np.concatenate(candidate_parts).astype(np.uint32, copy=False)
    probabilities = np.concatenate(probability_parts).astype(np.float32, copy=False)
    target = Path(output_dir).resolve()
    manifest_path = target / "score_manifest.json"
    arrays_path = target / "scores.npz"
    if not overwrite and (manifest_path.exists() or arrays_path.exists()):
        raise FileExistsError(f"refusing to overwrite V5 streaming scores: {target}")
    target.mkdir(parents=True, exist_ok=True)
    np.savez(
        arrays_path,
        pose_ids=expected_pose_ids.astype(np.int64, copy=False),
        pose_offsets=np.asarray(pose_offsets, dtype=np.int64),
        candidate_ids=candidates,
        full=probabilities,
    )
    threshold_probability = float(_sigmoid(threshold_logit))
    manifest = {
        "schema": STREAMING_SCORE_SCHEMA,
        "createdBy": "neural_instance_culling.benchmark.v5.streaming_export",
        "split": split_name,
        "testRead": split_name == "test",
        "poseCount": int(expected_pose_ids.size),
        "candidateRowCount": int(candidates.size),
        "numInstances": int(scene_scores.num_instances),
        "scoreFields": ["full"],
        "arraysFile": arrays_path.name,
        "thresholds": {"full": threshold_probability},
        "thresholdSources": {"full": "checkpoint calibration"},
        "scoreSources": {
            "full": {
                "kind": "formal_v5_columnar_score_bundle",
                "bundle": str(Path(bundle_manifest).resolve()),
                "evaluationSummary": str(summary_path),
                "protocol": run.protocol,
                "variant": run.variant,
                "seed": run.seed,
                "scene": scene_id,
                "sourceScore": "visibility_logit",
                "continuousScoreField": "sigmoid(visibility_logit)",
                "glbAggregation": "max_i_in_glb(instance_probability)",
                "thresholdLogit": threshold_logit,
                "thresholdProbability": threshold_probability,
                "qualification": threshold_row.get("qualification"),
            }
        },
        "unavailableSources": {},
        "source": {
            "datasetDir": str(scene_scores.pose_dataset.resolve()),
            "candidateSemantics": csr.dataset.meta.get("candidateSemantics"),
            "gtSemantics": csr.dataset.meta.get("gtSemantics"),
            "excludedUnitIds": list(sidecar.excluded_unit_ids),
            "excludedUnitFillProbability": 0.0,
        },
        "alignment": {
            "poseIds": "pose_ids array in frozen split order",
            "candidateIds": "candidate_ids exactly equal PoseCSR rows",
            "offsets": "pose_offsets index candidate_ids and full probabilities",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path


__all__ = ["STREAMING_SCORE_SCHEMA", "export_streaming_score_sidecar"]
