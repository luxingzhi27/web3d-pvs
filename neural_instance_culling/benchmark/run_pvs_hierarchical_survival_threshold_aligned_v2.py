#!/usr/bin/env python3
"""Execute the threshold-aligned full-innovation PVS experiment.

This runner is deliberately independent from the historical matrix runner.
It owns three stages from the registered plan: ``scan12``, ``refine24`` and
``formal80``.  Each training member is an independent single-GPU process. A
GPU receives the next member only after its previous process exits, so four
cards form one dynamic queue rather than a static assignment.

The runner never reads the test split and refuses to reuse a non-empty output
directory.  Safety status is recorded by the trainer and does not cancel a
numerically valid member.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from common.provenance import relation_artifact_digest  # noqa: E402
from common.runtime_meta import load_runtime_meta  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


PREFIX = "pvs_hierarchical_survival_integrated_spectral_threshold_aligned_utility_v2"
TRAIN = ROOT / "neural_instance_culling/model/train_hierarchical_relation_survival_integrated.py"
DEFAULT_DATASET = ROOT / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1"
DEFAULT_RELATION = ROOT / "neural_instance_culling/dataset/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/relation_csr"
DEFAULT_SIDECAR = ROOT / "neural_instance_culling/dataset/out/pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/subpose_quality_sidecar"
DEFAULT_RUNTIME_META = ROOT / "hkust-v3/assets/runtimeVisibilityMeta.json"
DEFAULT_GEO = ROOT / "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/instance_geo_features_fp16.bin"
DEFAULT_GLB_INDEX = ROOT / "hkust-v3/assets/glbIndex.json"
DEFAULT_GLB_ROOT = ROOT / "hkust-v3/assets"
DEFAULT_SEEDS = (20260801, 20260802, 20260803)
FORMAL_VARIANTS = (
    "full",
    "without_hierarchical_relation",
    "shuffled_relation_source",
    "without_viewcell_integration",
    "without_threshold_aligned_utility",
)
REQUEST_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _ensure_empty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty experiment directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _required_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.resolve()


def _required_dir(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path.resolve()


def _protocol_manifest(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = _required_file(Path(args.dataset_dir) / "dataset_meta.json", "dataset metadata").parent
    runtime_path = _required_file(Path(args.runtime_meta), "runtime metadata")
    relation_dir = Path(args.relation_dir).resolve()
    relation_meta_path = _required_file(relation_dir / "relation_csr_meta.json", "relation CSR metadata")
    geo_path = _required_file(Path(args.initial_geo_features), "fixed geometry table")
    sidecar_manifest = _required_file(Path(args.subpose_sidecar) / "sidecar_manifest.json", "subpose sidecar manifest")
    glb_index = _required_file(Path(args.glb_index), "GLB index")
    _required_dir(Path(args.glb_root), "GLB root")
    world_aabbs, instance_to_glb, runtime = load_runtime_meta(runtime_path)
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(world_aabbs.shape[0]))
    required_splits = ("train", "calibration", "validation")
    if any(name not in dataset.split_ids for name in required_splits):
        raise ValueError("dataset must contain train, calibration and validation; test is not a fallback")
    split_counts = {
        name: int(dataset.split(name).pose_indices.size) for name in required_splits
    }
    split_digests = {
        name: candidate_digest_for_pose_sequence(dataset, dataset.split(name).pose_indices)
        for name in required_splits
    }
    relation_meta = json.loads(relation_meta_path.read_text(encoding="utf-8"))
    if relation_meta.get("trainOnly") is not True or relation_meta.get("splitNames") != ["train"]:
        raise ValueError("relation CSR must be explicitly train-only")
    relation_candidate = relation_meta.get("candidateIdentity", {}).get("canonicalCandidateDigest")
    if relation_candidate != split_digests["train"]:
        raise ValueError("relation CSR train candidate digest does not match dataset")
    if geo_path.stat().st_size != int(world_aabbs.shape[0]) * 96 * 2:
        raise ValueError("fixed geometry table is not N x 96 FP16")
    relation_digest, relation_files = relation_artifact_digest(relation_dir, relation_meta)
    sidecar = json.loads(sidecar_manifest.read_text(encoding="utf-8"))
    if sidecar.get("schema") != "pvs-viewcell-subpose-supervision-sidecar-v1":
        raise ValueError("unexpected subpose sidecar schema")
    return {
        "schema": "pvs-full-innovation-threshold-aligned-matrix-manifest-v2",
        "experimentPrefix": PREFIX,
        "dataset": {"path": str(dataset_dir), "sha256": _sha256(dataset_dir / "dataset_meta.json")},
        "runtimeMeta": {"path": str(runtime_path), "sha256": _sha256(runtime_path), "numInstances": int(world_aabbs.shape[0]), "numGlbs": int(runtime.get("numGlbs", int(instance_to_glb.max()) + 1))},
        "geometry": {"path": str(geo_path), "sha256": _sha256(geo_path), "shape": [int(world_aabbs.shape[0]), 96], "dtype": "float16"},
        "glb": {"index": str(glb_index), "root": str(Path(args.glb_root).resolve()), "indexSha256": _sha256(glb_index)},
        "relation": {"path": str(relation_dir), "schema": relation_meta.get("schema"), "digest": relation_digest, "files": relation_files, "trainCandidateDigest": relation_candidate},
        "sidecar": {"path": str(Path(args.subpose_sidecar).resolve()), "manifestSha256": _sha256(sidecar_manifest), "schema": sidecar.get("schema"), "splitLabelsMayDiffer": bool(sidecar.get("splitLabelsMayDiffer", False))},
        "splitPoseCounts": split_counts,
        "splitCandidateDigests": split_digests,
        "fov": {"modelInputDegrees": 66.0, "frontendRenderDegrees": 60.0},
        "requestThresholds": list(REQUEST_THRESHOLDS),
        "testRead": False,
    }


def _anchor_configs() -> list[dict[str, float]]:
    return [
        {"lr": 1e-4, "weightDecay": 1e-5, "survival": 0.25, "tail": 0.35, "falseRequest": 0.015, "utility": 0.18, "download": 0.20, "temperature": 0.75, "tailFraction": 0.010, "rare": 1.0},
        {"lr": 2e-4, "weightDecay": 1e-5, "survival": 0.25, "tail": 0.35, "falseRequest": 0.015, "utility": 0.18, "download": 0.20, "temperature": 0.75, "tailFraction": 0.010, "rare": 1.0},
        {"lr": 3e-4, "weightDecay": 1e-5, "survival": 0.25, "tail": 0.35, "falseRequest": 0.015, "utility": 0.18, "download": 0.20, "temperature": 0.75, "tailFraction": 0.010, "rare": 1.0},
        {"lr": 2e-4, "weightDecay": 5e-5, "survival": 0.40, "tail": 0.60, "falseRequest": 0.030, "utility": 0.30, "download": 0.35, "temperature": 0.50, "tailFraction": 0.020, "rare": 2.0},
    ]


def _sobol_like_configs() -> list[dict[str, float]]:
    # A fixed low-discrepancy table.  The values are pre-registered in the
    # manifest; no result-dependent points are appended during execution.
    lr = [1e-4, 2e-4, 3e-4]
    wd = [1e-6, 1e-5, 5e-5]
    survival = [0.10, 0.25, 0.40]
    tail = [0.15, 0.35, 0.60]
    false_request = [0.005, 0.015, 0.030, 0.050]
    utility = [0.08, 0.18, 0.30]
    download = [0.10, 0.20, 0.35]
    temperature = [0.50, 0.75, 1.00]
    tail_fraction = [0.005, 0.010, 0.020]
    rare = [0.5, 1.0, 2.0]
    points = []
    for i in range(12):
        points.append({
            "lr": lr[(i * 2 + 1) % 3], "weightDecay": wd[(i * 5 + 1) % 3],
            "survival": survival[(i * 3 + 2) % 3], "tail": tail[(i * 7 + 1) % 3],
            "falseRequest": false_request[(i * 5 + 2) % 4], "utility": utility[(i * 2 + 2) % 3],
            "download": download[(i * 7 + 1) % 3], "temperature": temperature[(i * 5 + 2) % 3],
            "tailFraction": tail_fraction[(i * 11 + 1) % 3], "rare": rare[(i * 7 + 2) % 3],
        })
    return points


def scan_configs() -> list[dict[str, float]]:
    return _anchor_configs() + _sobol_like_configs()


def _variant_spec(name: str) -> dict[str, str]:
    if name == "full":
        return {"relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "threshold_aligned_utility"}
    if name == "without_hierarchical_relation":
        return {"relationSource": "geometry_only", "spectralMode": "integrated", "lossVariant": "threshold_aligned_utility"}
    if name == "shuffled_relation_source":
        return {"relationSource": "shuffled", "spectralMode": "integrated", "lossVariant": "threshold_aligned_utility"}
    if name == "without_viewcell_integration":
        return {"relationSource": "hierarchical", "spectralMode": "learned_point", "lossVariant": "threshold_aligned_utility"}
    if name == "without_threshold_aligned_utility":
        return {"relationSource": "hierarchical", "spectralMode": "integrated", "lossVariant": "rvl"}
    raise ValueError(f"unknown formal variant: {name}")


def _mode_settings(mode: str, requested_variants: list[str] | None, requested_seeds: list[int] | None) -> tuple[int, int, list[str], list[int]]:
    if mode == "scan12":
        return 12, 100, ["full"], [20260801]
    if mode == "refine24":
        seeds = requested_seeds or [20260801, 20260802]
        if len(seeds) != 2 or set(seeds) != {20260801, 20260802}:
            raise ValueError("refine24 must contain exactly seeds 20260801 and 20260802")
        return 24, 120, ["full"], seeds
    if mode == "formal80":
        names = requested_variants or list(FORMAL_VARIANTS)
        seeds = requested_seeds or list(DEFAULT_SEEDS)
        if set(names) != set(FORMAL_VARIANTS):
            raise ValueError(f"formal80 must contain exactly {list(FORMAL_VARIANTS)}")
        if len(seeds) != 3 or set(seeds) != set(DEFAULT_SEEDS):
            raise ValueError("formal80 must contain the three registered seeds")
        return 80, 150, names, seeds
    raise ValueError(f"unsupported training mode: {mode}")


def _config_for_job(job: Mapping[str, Any]) -> dict[str, float]:
    value = job.get("hyperparameters")
    if not isinstance(value, Mapping):
        raise ValueError("job has no hyperparameters")
    return {str(k): float(v) for k, v in value.items()}


def _train_command(args: argparse.Namespace, job: Mapping[str, Any], member: Path, epochs: int, steps: int, bootstrap: int) -> list[str]:
    hp = _config_for_job(job)
    spec = _variant_spec(str(job["variant"]))
    command = [
        sys.executable, str(TRAIN),
        "--dataset-dir", str(Path(args.dataset_dir).resolve()), "--relation-dir", str(Path(args.relation_dir).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()), "--initial-geo-features", str(Path(args.initial_geo_features).resolve()),
        "--subpose-sidecar", str(Path(args.subpose_sidecar).resolve()), "--glb-index", str(Path(args.glb_index).resolve()), "--glb-root", str(Path(args.glb_root).resolve()),
        "--output-dir", str(member), "--experiment-name", f"{PREFIX}_{job['variant']}_{job['configId']}_seed{job['seed']}_e{epochs}",
            "--variant", str(job["variant"]), "--relation-source", spec["relationSource"], "--spectral-mode", spec["spectralMode"], "--loss-variant", spec["lossVariant"],
        "--epochs", str(epochs), "--steps-per-epoch", str(steps), "--eval-every", "4", "--snapshot-every", "8",
        "--calibration-bootstrap-replicates", str(bootstrap), "--max-eval-poses", "0", "--seed", str(job["seed"]), "--device", "cuda",
        "--lr", str(hp["lr"]), "--weight-decay", str(hp["weightDecay"]), "--survival-loss-weight", str(hp["survival"]),
        "--quality-loss-weight", str(hp["tail"]), "--resource-loss-weight", str(hp["falseRequest"]),
        "--utility-loss-weight", str(hp["utility"]), "--download-loss-weight", str(hp["download"]),
        "--quality-temperature", str(hp["temperature"]), "--quality-tail-fraction", str(hp["tailFraction"]), "--rare-positive-weight", str(hp["rare"]),
    ]
    return command


def _run_one(args: argparse.Namespace, job: Mapping[str, Any], output_root: Path, log_root: Path, gpu: int, epochs: int, steps: int, bootstrap: int) -> dict[str, Any]:
    member = output_root / str(job["member"])
    _ensure_empty(member)
    stdout_path = log_root / f"{job['member']}.stdout.log"
    stderr_path = log_root / f"{job['member']}.stderr.log"
    command = _train_command(args, job, member, epochs, steps, bootstrap)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    result = {"job": dict(job), "gpu": int(gpu), "outputDir": str(member), "stdout": str(stdout_path), "stderr": str(stderr_path), "command": command, "returnCode": int(completed.returncode), "elapsedSeconds": time.time() - started, "testRead": False}
    if completed.returncode == 0:
        for required in ("last.pt", "model_meta.json", "calibration_ready_summary.json", "train_history.json"):
            if not (member / required).is_file():
                result["returnCode"] = 86
                result["artifactError"] = f"missing {required}"
                break
    return result


def _dynamic_queue(args: argparse.Namespace, jobs: list[dict[str, Any]], output_root: Path, log_root: Path, gpu_ids: list[int], epochs: int, steps: int, bootstrap: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    pending = iter(jobs)
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        active: dict[Future[dict[str, Any]], int] = {}
        for gpu in gpu_ids:
            try:
                job = next(pending)
            except StopIteration:
                break
            active[executor.submit(_run_one, args, job, output_root, log_root, gpu, epochs, steps, bootstrap)] = gpu
        while active:
            completed, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in completed:
                gpu = active.pop(future)
                result = future.result()
                results.append(result)
                try:
                    job = next(pending)
                except StopIteration:
                    continue
                active[executor.submit(_run_one, args, job, output_root, log_root, gpu, epochs, steps, bootstrap)] = gpu
    results.sort(key=lambda item: str(item["job"]["member"]))
    return results


def _scan_jobs(configs: list[dict[str, float]], epochs: int, seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "variant": "full",
            "variantSpec": _variant_spec("full"),
            "configId": f"scan{index:02d}",
            "seed": int(seeds[0]),
            "hyperparameters": config,
            "member": f"{index:02d}_full_{_canonical(config)[:8]}_seed{seeds[0]}_e{epochs}",
        }
        for index, config in enumerate(configs)
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _select_scan_configs(scan_root: Path, limit: int = 6) -> list[dict[str, float]]:
    summary = _load_json(scan_root / "scan_member_summary.json")
    candidates = []
    for row in summary.get("members", []):
        if int(row.get("returnCode", 1)) != 0:
            continue
        hp = row.get("job", {}).get("hyperparameters")
        if isinstance(hp, Mapping):
            calibration = row.get("calibration", {})
            selected = calibration.get("selected") if isinstance(calibration, Mapping) else None
            diagnostic = calibration.get("diagnostic") if isinstance(calibration, Mapping) else None
            # scan12 intentionally uses an early bootstrap and therefore does
            # not claim the formal safety gate.  Rank every numerically valid
            # member by its calibration-frozen diagnostic workpoint instead of
            # treating a missing safe workpoint as identical data.
            workpoint = selected if isinstance(selected, Mapping) else diagnostic
            workpoint = workpoint if isinstance(workpoint, Mapping) else {}
            score = (
                _metric_from_payload(workpoint, "aggregateWeightedRecall")
                if _metric_from_payload(workpoint, "aggregateWeightedRecall") is not None
                else -1.0,
                _metric_from_payload(workpoint, "aggregateWeightedRecallLowerConfidenceBound")
                if _metric_from_payload(workpoint, "aggregateWeightedRecallLowerConfidenceBound") is not None
                else -1.0,
                _metric_from_payload(workpoint, "balancedAccuracy")
                if _metric_from_payload(workpoint, "balancedAccuracy") is not None
                else 0.0,
                _metric_from_payload(workpoint, "usefulCull")
                if _metric_from_payload(workpoint, "usefulCull") is not None
                else 0.0,
                _metric_from_payload(workpoint, "precision")
                if _metric_from_payload(workpoint, "precision") is not None
                else 0.0,
                -(
                    _metric_from_payload(workpoint, "predictedGlbBytes")
                    if _metric_from_payload(workpoint, "predictedGlbBytes") is not None
                    else 1e30
                ),
                -(
                    _metric_from_payload(workpoint, "predictedCount")
                    if _metric_from_payload(workpoint, "predictedCount") is not None
                    else 1e30
                ),
            )
            candidates.append((score, {str(k): float(v) for k, v in hp.items()}))
    if not candidates:
        raise RuntimeError("scan produced no numerically valid configuration")
    candidates.sort(key=lambda item: item[0], reverse=True)
    unique: list[dict[str, float]] = []
    for _score, config in candidates:
        if config not in unique:
            unique.append(config)
        if len(unique) >= int(limit):
            break
    return unique


def _member_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(str(result["outputDir"]))
    calibration = _load_json(output / "calibration_ready_summary.json") if (output / "calibration_ready_summary.json").is_file() else {}
    return {**dict(result), "calibration": {"status": calibration.get("status"), "selected": calibration.get("selected"), "diagnostic": calibration.get("diagnostic"), "validationAtCalibration": calibration.get("validationAtCalibration")}}


REFINE_SAFETY_FLOOR = 0.99
REFINE_METRICS = (
    "aggregateWeightedRecall",
    "aggregateWeightedRecallLowerConfidenceBound",
    "balancedAccuracy",
    "usefulCull",
    "precision",
    "accuracy",
    "predictedCount",
    "predictedGlbCount",
    "predictedGlbBytes",
)


def _finite_number(value: Any, default: float | None = None) -> float | None:
    """Return a finite scalar, keeping missing metrics distinguishable."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _metric_from_payload(payload: Mapping[str, Any] | None, metric: str) -> float | None:
    """Read the trainer's flattened metric names without inventing a value.

    Validation rows currently use ``agg_*`` names while a few standalone
    summaries use camelCase aggregate names.  This small adapter keeps the
    refine logic about selection, not about a second evaluator schema.
    """

    if not isinstance(payload, Mapping):
        return None
    aliases = {
        "aggregateWeightedRecall": ("aggregateWeightedRecall", "agg_weighted_recall", "weightedRecall"),
        "aggregateWeightedRecallLowerConfidenceBound": (
            "aggregateWeightedRecallLowerConfidenceBound",
            "weightedRecallLowerConfidenceBound",
        ),
        "balancedAccuracy": ("agg_balanced_accuracy", "balancedAccuracy", "balanced_accuracy"),
        "usefulCull": ("agg_useful_cull", "usefulCull", "useful_cull"),
        "precision": ("agg_precision", "precision"),
        "accuracy": ("agg_accuracy", "accuracy"),
        "predictedCount": ("avg_pred_count", "avgPredCount", "predictedCount"),
        "predictedGlbCount": ("avg_pred_glb_count", "avgPredictedGlbCount", "predictedGlbCount"),
        "predictedGlbBytes": ("avg_pred_glb_bytes", "avgPredictedGlbBytes", "predictedGlbBytes"),
    }
    for key in aliases[metric]:
        value = _finite_number(payload.get(key))
        if value is not None:
            return value
    aggregate = payload.get("aggregate")
    if isinstance(aggregate, Mapping):
        nested_aliases = {
            "aggregateWeightedRecall": ("weightedRecall", "aggregateWeightedRecall", "agg_weighted_recall"),
            "aggregateWeightedRecallLowerConfidenceBound": (
                "weightedRecallLowerConfidenceBound",
                "aggregateWeightedRecallLowerConfidenceBound",
            ),
            "balancedAccuracy": ("balancedAccuracy", "balanced_accuracy"),
            "usefulCull": ("usefulCull", "useful_cull"),
            "precision": ("precision",),
            "accuracy": ("accuracy",),
            "predictedCount": ("avgPredCount", "avg_pred_count"),
            "predictedGlbCount": ("predictedGlbCount", "avgPredictedGlbCount"),
            "predictedGlbBytes": ("predictedGlbBytes", "avgPredictedGlbBytes"),
        }
        for key in nested_aliases[metric]:
            value = _finite_number(aggregate.get(key))
            if value is not None:
                return value
    return None


def _refine_member_metrics(member: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one 24-epoch member for deterministic two-seed selection."""

    calibration = member.get("calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    selected = calibration.get("selected")
    selected = selected if isinstance(selected, Mapping) else None
    validation = calibration.get("validationAtCalibration")
    validation = validation if isinstance(validation, Mapping) else None

    calibration_recall = _metric_from_payload(selected, "aggregateWeightedRecall")
    calibration_lcb = _metric_from_payload(selected, "aggregateWeightedRecallLowerConfidenceBound")
    calibration_safe = bool(
        selected is not None
        and calibration_recall is not None
        and calibration_lcb is not None
        and calibration_recall > REFINE_SAFETY_FLOOR
        and calibration_lcb > REFINE_SAFETY_FLOOR
    )

    # Selection must be judged on validation replay at the calibration-frozen
    # threshold.  A missing validation row is never considered safe, even if
    # calibration itself was safe.
    evaluation = validation if validation is not None else selected
    metrics = {metric: _metric_from_payload(evaluation, metric) for metric in REFINE_METRICS}
    validation_safe = bool(
        validation is not None
        and metrics["aggregateWeightedRecall"] is not None
        and metrics["aggregateWeightedRecallLowerConfidenceBound"] is not None
        and metrics["aggregateWeightedRecall"] > REFINE_SAFETY_FLOOR
        and metrics["aggregateWeightedRecallLowerConfidenceBound"] > REFINE_SAFETY_FLOOR
    )
    return {
        "seed": int(member.get("job", {}).get("seed", -1)),
        "member": str(member.get("job", {}).get("member", member.get("outputDir", ""))),
        "outputDir": member.get("outputDir"),
        "calibrationSafe": calibration_safe,
        "validationSafe": validation_safe,
        "safeForRefine": bool(calibration_safe and validation_safe),
        "metricSource": "validationAtCalibration" if validation is not None else "calibrationSelectedFallback",
        "calibration": {
            "aggregateWeightedRecall": calibration_recall,
            "aggregateWeightedRecallLowerConfidenceBound": calibration_lcb,
        },
        "metrics": metrics,
    }


def _configuration_key(hyperparameters: Mapping[str, Any]) -> str:
    return _canonical({str(key): float(value) for key, value in hyperparameters.items()})


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _variance_or_none(values: list[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.var(finite, ddof=0)) if len(finite) >= 2 else None


def _aggregate_refine_groups(
    members: list[Mapping[str, Any]],
    *,
    expected_seeds: tuple[int, int] = (20260801, 20260802),
) -> list[dict[str, Any]]:
    """Group refine members by hyperparameters and summarize both seeds."""

    grouped: dict[str, dict[str, Any]] = {}
    for member in members:
        if int(member.get("returnCode", 1)) != 0:
            continue
        job = member.get("job")
        if not isinstance(job, Mapping) or not isinstance(job.get("hyperparameters"), Mapping):
            continue
        hyperparameters = {str(key): float(value) for key, value in job["hyperparameters"].items()}
        key = _configuration_key(hyperparameters)
        group = grouped.setdefault(
            key,
            {
                "configurationKey": key,
                "configIds": [],
                "hyperparameters": hyperparameters,
                "members": [],
            },
        )
        config_id = str(job.get("configId", ""))
        if config_id not in group["configIds"]:
            group["configIds"].append(config_id)
        group["members"].append(_refine_member_metrics(member))

    output: list[dict[str, Any]] = []
    required = set(int(seed) for seed in expected_seeds)
    for group in grouped.values():
        seed_rows: dict[int, dict[str, Any]] = {}
        duplicate_seeds: list[int] = []
        for row in group["members"]:
            seed = int(row["seed"])
            if seed in seed_rows:
                duplicate_seeds.append(seed)
            seed_rows[seed] = row
        rows = [seed_rows[seed] for seed in sorted(seed_rows)]
        complete = not duplicate_seeds and set(seed_rows) == required and len(rows) == len(expected_seeds)
        means = {
            metric: _mean_or_none([row["metrics"].get(metric) for row in rows])
            for metric in REFINE_METRICS
        }
        variances = {
            metric: _variance_or_none([row["metrics"].get(metric) for row in rows])
            for metric in REFINE_METRICS
        }
        stddevs = {
            metric: None if variances[metric] is None else float(np.sqrt(variances[metric]))
            for metric in REFINE_METRICS
        }
        # Normalize resource variance before combining it with probabilities;
        # otherwise bytes would dominate the stability diagnostic.
        normalized_variances = []
        for metric in REFINE_METRICS:
            variance = variances[metric]
            mean = means[metric]
            if variance is None:
                continue
            scale = max(abs(float(mean or 0.0)), 1.0)
            normalized_variances.append(float(variance) / (scale * scale))
        stability_variance = float(np.mean(normalized_variances)) if normalized_variances else None
        worst = {
            metric: (min(
                float(row["metrics"][metric])
                for row in rows
                if row["metrics"].get(metric) is not None
            ) if any(row["metrics"].get(metric) is not None for row in rows) else None)
            for metric in REFINE_METRICS
        }
        safe_count = sum(bool(row["safeForRefine"]) for row in rows)
        group["members"] = rows
        group.update(
            {
                "configId": sorted(group["configIds"])[0] if group["configIds"] else "",
                "configIds": sorted(group["configIds"]),
                "expectedSeeds": sorted(required),
                "seeds": sorted(seed_rows),
                "seedCount": len(rows),
                "completeTwoSeed": bool(complete),
                "duplicateSeeds": sorted(duplicate_seeds),
                "safety": {
                    "calibrationSafeCount": sum(bool(row["calibrationSafe"]) for row in rows),
                    "validationSafeCount": sum(bool(row["validationSafe"]) for row in rows),
                    "safeForRefineCount": int(safe_count),
                    "safeAcrossBothSeeds": bool(complete and safe_count == len(expected_seeds)),
                },
                "mean": means,
                "variance": variances,
                "stddev": stddevs,
                "worstSeed": worst,
                "stabilityVariance": stability_variance,
            }
        )
        output.append(group)
    output.sort(key=lambda group: str(group["configurationKey"]))
    return output


def _rank_number(group: Mapping[str, Any], metric: str, *, missing: float, source: str = "mean") -> float:
    value = group.get(source, {}).get(metric)
    number = _finite_number(value)
    return missing if number is None else number


def _pareto_frontier(groups: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return nondominated groups for the registered efficiency objectives."""

    maximize = ("balancedAccuracy", "usefulCull", "precision", "accuracy")
    minimize = ("predictedCount", "predictedGlbCount", "predictedGlbBytes", "stabilityVariance")

    def value(group: Mapping[str, Any], metric: str) -> float:
        number = _finite_number(group.get("mean", {}).get(metric))
        if number is None:
            return -float("inf") if metric in maximize else float("inf")
        return number

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        left_values = [value(left, metric) for metric in maximize]
        right_values = [value(right, metric) for metric in maximize]
        left_min = [value(left, metric) for metric in minimize]
        right_min = [value(right, metric) for metric in minimize]
        no_worse = all(a >= b for a, b in zip(left_values, right_values)) and all(a <= b for a, b in zip(left_min, right_min))
        strictly_better = any(a > b for a, b in zip(left_values, right_values)) or any(a < b for a, b in zip(left_min, right_min))
        return no_worse and strictly_better

    return [candidate for candidate in groups if not any(dominates(other, candidate) for other in groups if other is not candidate)]


def _select_refine_configuration(groups: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze one deterministic configuration using the registered rules."""

    complete = [group for group in groups if bool(group.get("completeTwoSeed"))]
    if not complete:
        raise RuntimeError("refine has no complete two-seed configuration")
    safe = [group for group in complete if bool(group.get("safety", {}).get("safeAcrossBothSeeds"))]
    pool = safe if safe else complete
    selection_pool = "safe_candidate_pool" if safe else "diagnostic_candidate_pool"
    frontier = _pareto_frontier(pool) if safe else list(pool)

    def descending(group: Mapping[str, Any], metric: str, missing: float = -float("inf"), *, source: str = "mean") -> tuple[int, float]:
        value = _rank_number(group, metric, missing=missing, source=source)
        return (0 if np.isfinite(value) else 1, -value if np.isfinite(value) else 0.0)

    def ascending(group: Mapping[str, Any], metric: str, missing: float = float("inf"), *, source: str = "mean") -> tuple[int, float]:
        value = _rank_number(group, metric, missing=missing, source=source)
        return (0 if np.isfinite(value) else 1, value if np.isfinite(value) else 0.0)

    if safe:
        # The lexicographic order mirrors the plan after Pareto filtering.
        ordered = sorted(
            frontier,
            key=lambda group: (
                descending(group, "usefulCull"),
                descending(group, "balancedAccuracy"),
                descending(group, "precision"),
                descending(group, "accuracy"),
                ascending(group, "predictedCount"),
                ascending(group, "predictedGlbBytes"),
                ascending(group, "predictedGlbCount"),
                ascending(group, "stabilityVariance"),
                descending(group, "aggregateWeightedRecall"),
                descending(group, "aggregateWeightedRecallLowerConfidenceBound"),
                str(group["configurationKey"]),
            ),
        )
        rule = "safe: Pareto efficiency frontier, then useful cull, balanced accuracy, precision, accuracy, predicted count, predicted GLB bytes, predicted GLB count, stability variance"
    else:
        # Unsafe configurations still need a deterministic winner so the
        # mandatory long-train queue can proceed, but this winner is not safe.
        ordered = sorted(
            frontier,
            key=lambda group: (
                descending(group, "aggregateWeightedRecall", -float("inf"), source="worstSeed"),
                descending(group, "aggregateWeightedRecallLowerConfidenceBound", -float("inf"), source="worstSeed"),
                descending(group, "balancedAccuracy"),
                descending(group, "usefulCull"),
                descending(group, "precision"),
                ascending(group, "predictedGlbBytes"),
                ascending(group, "predictedGlbCount"),
                ascending(group, "predictedCount"),
                ascending(group, "stabilityVariance"),
                str(group["configurationKey"]),
            ),
        )
        rule = "diagnostic: worst available weighted recall/LCB, balanced accuracy, useful cull, precision, predicted GLB bytes, predicted GLB count, predicted count, stability variance"
    winner = ordered[0]
    ranking = [
        {
            "rank": index + 1,
            "configurationKey": group["configurationKey"],
            "configId": group.get("configId"),
            "hyperparameters": group["hyperparameters"],
            "selectionMetrics": group.get("mean"),
            "stabilityVariance": group.get("stabilityVariance"),
            "safeAcrossBothSeeds": group.get("safety", {}).get("safeAcrossBothSeeds", False),
        }
        for index, group in enumerate(ordered)
    ]
    return {
        "schema": "pvs-full-innovation-frozen-config-v2",
        "selectionPool": selection_pool,
        "selectionRule": rule,
        "expectedSeeds": [20260801, 20260802],
        "safetyFloor": REFINE_SAFETY_FLOOR,
        "configurationKey": winner["configurationKey"],
        "configId": winner.get("configId"),
        "hyperparameters": dict(winner["hyperparameters"]),
        "selection": winner,
        "ranking": ranking,
        "paretoFrontier": [group["configurationKey"] for group in frontier],
        "completeConfigurationCount": len(complete),
        "safeConfigurationCount": len(safe),
        "testRead": False,
    }


def _make_jobs(args: argparse.Namespace, mode: str, output_root: Path, frozen: Mapping[str, Any] | None) -> tuple[list[dict[str, Any]], int, int, int]:
    epochs, steps, variants, seeds = _mode_settings(mode, args.variants, args.seeds)
    if mode == "scan12":
        configs = scan_configs()
        return _scan_jobs(configs, epochs, seeds), epochs, steps, 2000
    if mode == "refine24":
        scan_root = Path(args.scan_root).resolve() if args.scan_root else output_root.parent / f"{PREFIX}_scan12_20260813"
        configs = _select_scan_configs(scan_root)
        jobs = []
        for index, config in enumerate(configs):
            for seed in seeds:
                jobs.append({"variant": "full", "variantSpec": _variant_spec("full"), "configId": f"refine{index:02d}", "seed": int(seed), "hyperparameters": config, "member": f"{index:02d}_full_seed{seed}_e{epochs}"})
        return jobs, epochs, steps, 10000
    if frozen is None:
        raise ValueError("formal80 requires --frozen-config or --refine-root")
    config = frozen.get("hyperparameters")
    if not isinstance(config, Mapping):
        raise ValueError("frozen config has no hyperparameters")
    jobs = []
    for variant in variants:
        for seed in seeds:
            jobs.append({"variant": variant, "variantSpec": _variant_spec(variant), "configId": "frozen", "seed": int(seed), "hyperparameters": dict(config), "member": f"{variant}_seed{seed}_e{epochs}"})
    if len(jobs) != 15:
        raise AssertionError("formal80 must create exactly 15 jobs")
    return jobs, epochs, steps, 10000


def _frozen_from_refine(refine_root: Path) -> dict[str, Any]:
    path = refine_root / "frozen_config.json"
    if path.is_file():
        frozen = _load_json(path)
        # A pre-v2 placeholder is not a valid freeze record.  Recompute from
        # the immutable refine summary instead of silently reusing its first
        # member.
        if frozen.get("selectionPool") and frozen.get("selectionRule") and isinstance(frozen.get("ranking"), list):
            return frozen
    summary = _load_json(refine_root / "refine_member_summary.json")
    rows = summary.get("members", [])
    if not isinstance(rows, list):
        raise ValueError("refine summary members must be a list")
    groups = _aggregate_refine_groups(rows)
    return _select_refine_configuration(groups) | {"source": str(refine_root)}


def _summarize_stage_artifacts(input_root: Path, mode: str) -> dict[str, Any]:
    """Read-only stage inspection for the runner's reserved modes.

    This intentionally does not run a model, choose a threshold, or read
    test data.  It provides a small hand-off contract until the independent
    validation/benchmark evaluator is wired into the formal stage.
    """

    manifest = _load_json(input_root / "matrix_manifest.json")
    stage_name = str(manifest.get("mode", ""))
    summary_name = {
        "scan12": "scan_member_summary.json",
        "refine24": "refine_member_summary.json",
        "formal80": "formal_member_summary.json",
    }.get(stage_name)
    summary = _load_json(input_root / summary_name) if summary_name and (input_root / summary_name).is_file() else None
    members = summary.get("members", []) if isinstance(summary, Mapping) else []
    if not isinstance(members, list):
        raise ValueError("stage summary members must be a list")
    result: dict[str, Any] = {
        "schema": f"{PREFIX}-{mode}-reserved-inspection-v1",
        "mode": mode,
        "inputRoot": str(input_root),
        "stageMode": stage_name,
        "memberCount": len(members),
        "completedMemberCount": sum(int(row.get("returnCode", 1)) == 0 for row in members if isinstance(row, Mapping)),
        "failedMemberCount": sum(int(row.get("returnCode", 1)) != 0 for row in members if isinstance(row, Mapping)),
        "testRead": False,
        "evaluationImplemented": False,
        "message": "read-only artifact inspection; full evaluator is intentionally outside this runner change",
    }
    if stage_name == "refine24" and (input_root / "refine_configuration_summary.json").is_file():
        result["refineConfigurationSummary"] = _load_json(input_root / "refine_configuration_summary.json")
    return result


def _write_stage_summary(output_root: Path, mode: str, results: list[dict[str, Any]]) -> None:
    wrapped = [_member_summary(row) if int(row.get("returnCode", 1)) == 0 else dict(row) for row in results]
    _write_json(output_root / ("scan_member_summary.json" if mode == "scan12" else "refine_member_summary.json" if mode == "refine24" else "formal_member_summary.json"), {"schema": f"{PREFIX}-{mode}-summary-v2", "mode": mode, "members": wrapped, "testRead": False}, overwrite=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("scan12", "refine24", "formal80", "evaluate", "summarize"))
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--relation-dir", default=str(DEFAULT_RELATION))
    parser.add_argument("--runtime-meta", default=str(DEFAULT_RUNTIME_META))
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_GEO))
    parser.add_argument("--subpose-sidecar", default=str(DEFAULT_SIDECAR))
    parser.add_argument("--glb-index", default=str(DEFAULT_GLB_INDEX))
    parser.add_argument("--glb-root", default=str(DEFAULT_GLB_ROOT))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--log-root", default="")
    parser.add_argument("--scan-root", default="")
    parser.add_argument("--refine-root", default="")
    parser.add_argument("--frozen-config", default="")
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode in {"evaluate", "summarize"}:
        if not args.output_root:
            raise ValueError(f"{args.mode} requires --output-root pointing to an existing stage")
        inspection = _summarize_stage_artifacts(Path(args.output_root).resolve(), args.mode)
        print(json.dumps(inspection, ensure_ascii=False, indent=2, sort_keys=True))
        return
    if len(args.gpu_ids) == 0 or len(set(args.gpu_ids)) != len(args.gpu_ids) or any(value < 0 for value in args.gpu_ids):
        raise ValueError("gpu-ids must be unique non-negative IDs")
    default_suffix = {"scan12": "scan12", "refine24": "refine24", "formal80": "formal80"}[args.mode]
    output_root = Path(args.output_root or ROOT / "neural_instance_culling/model/out" / f"{PREFIX}_{default_suffix}_20260813").resolve()
    log_root = Path(args.log_root or output_root / "logs").resolve()
    protocol = _protocol_manifest(args)
    frozen: dict[str, Any] | None = None
    if args.mode == "formal80":
        if args.frozen_config:
            frozen = _load_json(Path(args.frozen_config).resolve())
        elif args.refine_root:
            frozen = _frozen_from_refine(Path(args.refine_root).resolve())
        elif args.dry_run:
            # Contract-only formal dry-run: the numerical scan has not yet
            # frozen a value, so use the registered anchor solely to expand
            # and validate the 15-member queue. It is never used to train.
            frozen = {
                "schema": "pvs-full-innovation-frozen-config-placeholder-v2",
                "selection": "dry_run_only_pending_scan",
                "hyperparameters": _anchor_configs()[1],
                "testRead": False,
            }
        else:
            raise ValueError("formal80 requires --frozen-config or --refine-root")
    jobs, epochs, steps, bootstrap = _make_jobs(args, args.mode, output_root, frozen)
    manifest = {
        **protocol,
        "mode": args.mode,
        "outputRoot": str(output_root),
        "logRoot": str(log_root),
        "epochs": epochs,
        "stepsPerEpoch": steps,
        "calibrationBootstrapReplicates": bootstrap,
        "gpuIds": [int(value) for value in args.gpu_ids],
        "gpuPolicy": "one independent CUDA process per GPU with dynamic completion queue",
        "variants": list(FORMAL_VARIANTS) if args.mode == "formal80" else ["full"],
        "seedCount": len(set(int(job["seed"]) for job in jobs)),
        "plannedMemberCount": len(jobs),
        "jobs": jobs,
        "frozenConfig": frozen,
        "dryRun": bool(args.dry_run),
        "testRead": False,
    }
    _ensure_empty(output_root)
    _ensure_empty(log_root)
    _write_json(output_root / "matrix_manifest.json", manifest)
    if args.dry_run:
        print(json.dumps({"manifest": str(output_root / "matrix_manifest.json"), "plannedMemberCount": len(jobs), "gpuIds": args.gpu_ids, "testRead": False}, ensure_ascii=False, indent=2))
        return
    results = _dynamic_queue(args, jobs, output_root, log_root, [int(value) for value in args.gpu_ids], epochs, steps, bootstrap)
    _write_stage_summary(output_root, args.mode, results)
    _write_json(output_root / "matrix_run_summary.json", {"schema": f"{PREFIX}-{args.mode}-run-v2", "members": results, "testRead": False})
    if any(int(item.get("returnCode", 1)) != 0 for item in results):
        raise RuntimeError(f"one or more {args.mode} members failed; inspect {log_root}")
    if args.mode == "refine24":
        summary = _load_json(output_root / "refine_member_summary.json")
        groups = _aggregate_refine_groups(summary.get("members", []))
        frozen = _select_refine_configuration(groups)
        _write_json(
            output_root / "refine_configuration_summary.json",
            {
                "schema": "pvs-full-innovation-refine-configuration-summary-v2",
                "expectedSeeds": [20260801, 20260802],
                "groups": groups,
                "selected": frozen,
                "testRead": False,
            },
        )
        _write_json(output_root / "frozen_config.json", frozen)
    print(json.dumps({"mode": args.mode, "outputRoot": str(output_root), "members": len(results), "testRead": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
