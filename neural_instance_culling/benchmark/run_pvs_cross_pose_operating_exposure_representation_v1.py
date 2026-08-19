#!/usr/bin/env python3
"""Run the cross-pose operating-boundary and view-cell exposure experiment.

This runner owns orchestration only.  The model/trainer owns the new loss and
training-only exposure head; the existing evaluator owns validation metrics.
The scan and factor stages have independent output roots so this experiment
cannot overwrite another experiment's members or reports.

The registered execution is:

* S1-S4: one seed, eight epochs, four loss configurations in parallel;
* A-D: one seed, sixteen epochs, the complete 2x2 factor matrix in parallel.

The factor stage is never gated by the scan safety result.  A scan always
freezes one relative winner, including when no scan member reaches the safety
threshold.  Test data is never opened by this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Empty, Queue
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py"
EXPERIMENT = "pvs_cross_pose_operating_exposure_representation_v1"
OUTPUT_TAG = f"{EXPERIMENT}_20260819"

SCAN_SEED = 20260801
FACTOR_SEED = 20260801
SCAN_EPOCHS = 8
FACTOR_EPOCHS = 16
STEPS_PER_EPOCH = 100
TARGET_WEIGHTED_RECALL = 0.99
RUNTIME_FEATURE_DIM = 124
RUNTIME_QUERY_INPUT_DIM = 130
RUNTIME_FP16_BYTES = 2
HKUST_RUNTIME_FEATURE_BYTES = 4_670_088
MAX_NEURAL_ASSET_BYTES = 7 * 1024 * 1024
QUICK_VALIDATION_BOOTSTRAP_REPLICATES = 2000

SCAN_VARIANTS = ("S1", "S2", "S3", "S4")
FACTOR_VARIANTS = (
    "A_pose_frontier_no_exposure",
    "B_cross_pose_no_exposure",
    "C_pose_frontier_exposure",
    "D_cross_pose_exposure",
)

# The four points are registered before execution.  The runner must not add a
# result-dependent fifth point or stop because a point is unsafe.
SCAN_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "configId": "S1",
        "positiveClassFraction": 0.20,
        "weightedRecallTarget": 0.993,
    },
    {
        "configId": "S2",
        "positiveClassFraction": 0.20,
        "weightedRecallTarget": 0.995,
    },
    {
        "configId": "S3",
        "positiveClassFraction": 0.30,
        "weightedRecallTarget": 0.993,
    },
    {
        "configId": "S4",
        "positiveClassFraction": 0.30,
        "weightedRecallTarget": 0.995,
    },
)

# A/C preserve the previous pose-balanced frontier objective.  B/D replace
# that objective with the cross-pose operating-boundary loss.  The exposure
# head is training-only and therefore does not affect the runtime schema.
POSE_FRONTIER_CONFIG: dict[str, float] = {
    "frontierLossWeight": 0.20,
    "frontierPositiveMassFraction": 0.005,
    "frontierPositiveCountCap": 64,
    "frontierNegativeFraction": 0.01,
    "frontierNegativeCountCap": 256,
    "frontierMargin": 0.50,
    "frontierTemperature": 0.25,
    "frontierPositiveImportanceFloor": 0.50,
    "frontierPositiveImportancePower": 0.50,
}

COMMON_TRAINING_CONFIG: dict[str, float] = {
    "learningRate": 2e-4,
    "weightDecay": 1e-5,
    "survivalLossWeight": 0.25,
    "relationConsistencyWeight": 0.10,
    "relationGradientCap": 0.25,
    "instanceCalibrationRegularizationWeight": 0.02,
    "instanceCalibrationMaxAbs": 4.0,
    "sparseInstancePenalty": 3.0,
}


def _default_paths(root: Path) -> dict[str, Path]:
    """Discover the same registered resources as the formal HKUST runners."""
    shared_subpose = root / (
        "neural_instance_culling/dataset/out/"
        "pvs_v4_viewcell_extreme_support_scan_20260818/"
        "subpose_supervision_sidecar"
    )
    local_subpose = ROOT / (
        "neural_instance_culling/dataset/out/"
        "pvs_v4_viewcell_extreme_support_scan_20260818/"
        "subpose_supervision_sidecar"
    )
    return {
        "dataset": root / (
            "neural_instance_culling/dataset/out/"
            "pose_csr_hkust_v3_bounded_relation_moment_fov66_v3"
        ),
        "relation": root / (
            "neural_instance_culling/dataset/out/"
            "pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/"
            "bounded_relation_csr_v3"
        ),
        "runtimeMeta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root / (
            "neural_instance_culling/model/out/"
            "pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/"
            "instance_geo_features_fp16.bin"
        ),
        "glbIndex": root / "hkust-v3/assets/glbIndex.json",
        "glbRoot": root / "hkust-v3/assets",
        "subpose": shared_subpose if shared_subpose.is_dir() else local_subpose,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an immutable record, accepting an identical existing record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _jsonable(dict(payload))
    if path.exists():
        existing = _load_json(path)
        if existing != normalized:
            raise FileExistsError(f"refusing to overwrite an existing record: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(normalized, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _required_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _required_dir(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    defaults = _default_paths(Path(args.root).resolve())
    overrides = {
        "dataset": args.dataset_dir,
        "relation": args.relation_dir,
        "runtimeMeta": args.runtime_meta,
        "geometry": args.initial_geo_features,
        "glbIndex": args.glb_index,
        "glbRoot": args.glb_root,
        "subpose": args.subpose_sidecar,
    }
    resolved: dict[str, Path] = {}
    for key, value in overrides.items():
        resolved[key] = Path(value).resolve() if value else defaults[key].resolve()
    return resolved


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate registered resources without loading a test split."""
    paths = _resolve_paths(args)
    _required_dir(paths["dataset"], "PoseCSR dataset")
    _required_dir(paths["relation"], "train-only relation artifact")
    _required_file(paths["runtimeMeta"], "runtime metadata")
    _required_file(paths["geometry"], "fixed geometry feature table")
    _required_file(paths["glbIndex"], "GLB index")
    _required_dir(paths["glbRoot"], "GLB asset root")
    _required_dir(paths["subpose"], "view-cell subpose sidecar")
    _required_file(TRAIN, "training entry point")
    _required_file(EVALUATE, "validation evaluator")

    dataset_meta = paths["dataset"] / "dataset_meta.json"
    if not dataset_meta.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {dataset_meta}")
    metadata = _load_json(dataset_meta)
    if metadata.get("testRead") is True:
        raise ValueError("dataset metadata is marked as having read the test split")

    geometry_bytes = paths["geometry"].stat().st_size
    if geometry_bytes <= 0 or geometry_bytes % (96 * 2) != 0:
        raise ValueError("fixed geometry table is not a non-empty N x 96 FP16 file")
    instance_count = geometry_bytes // (96 * RUNTIME_FP16_BYTES)
    runtime_metadata = _load_json(paths["runtimeMeta"])
    if int(runtime_metadata.get("instanceCount", -1)) != instance_count:
        raise ValueError("runtime metadata and fixed geometry table instance counts differ")
    runtime_feature_bytes = (
        instance_count * RUNTIME_FEATURE_DIM * RUNTIME_FP16_BYTES
    )
    if runtime_feature_bytes != HKUST_RUNTIME_FEATURE_BYTES:
        raise ValueError(
            "HKUST runtime feature table contract changed: "
            f"{runtime_feature_bytes} bytes != {HKUST_RUNTIME_FEATURE_BYTES} bytes"
        )
    if runtime_feature_bytes >= MAX_NEURAL_ASSET_BYTES:
        raise ValueError("fixed runtime feature table alone exceeds the neural asset budget")
    return {
        "schema": "pvs-cross-pose-operating-exposure-preflight-v1",
        "experiment": EXPERIMENT,
        "scanVariants": list(SCAN_VARIANTS),
        "factorVariants": list(FACTOR_VARIANTS),
        "scanSeed": SCAN_SEED,
        "factorSeed": FACTOR_SEED,
        "scanEpochs": SCAN_EPOCHS,
        "factorEpochs": FACTOR_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "initialCheckpoint": None,
        "runtimeSchemaUnchanged": True,
        "fixedFeatureContract": "96D geometry + 28D survival = 124D FP16",
        "runtimeFeatureDimension": RUNTIME_FEATURE_DIM,
        "runtimeQueryInputDimension": RUNTIME_QUERY_INPUT_DIM,
        "runtimeFeatureByteLength": int(runtime_feature_bytes),
        "maximumNeuralAssetBytes": MAX_NEURAL_ASSET_BYTES,
        "trainingOnlyExposureHeadExported": False,
        "onlineQueriesPerViewcell": 1,
        "testRead": False,
        "paths": {key: str(value) for key, value in paths.items()},
        "datasetMetaSchema": metadata.get("schema"),
        "geometryByteLength": int(geometry_bytes),
    }


def _mode_config(mode: str) -> dict[str, Any]:
    mode = str(mode)
    if mode == "smoke":
        return {
            "mode": mode,
            "stage": "smoke",
            "variants": ["S1"],
            "seed": SCAN_SEED,
            "epochs": 1,
            "stepsPerEpoch": 1,
            "evalEvery": 1,
            "snapshotEvery": 1,
            "maxEvalPoses": 2,
            "calibrationBootstrapReplicates": 2,
            "observationBatchSize": 32768,
            "testRead": False,
        }
    if mode == "scan8":
        return {
            "mode": mode,
            "stage": "scan8",
            "variants": list(SCAN_VARIANTS),
            "seed": SCAN_SEED,
            "epochs": SCAN_EPOCHS,
            "stepsPerEpoch": STEPS_PER_EPOCH,
            "evalEvery": SCAN_EPOCHS,
            "snapshotEvery": 4,
            "maxEvalPoses": 0,
            "calibrationBootstrapReplicates": 1000,
            "observationBatchSize": 8192,
            "testRead": False,
        }
    if mode == "factor16":
        return {
            "mode": mode,
            "stage": "factor16",
            "variants": list(FACTOR_VARIANTS),
            "seed": FACTOR_SEED,
            "epochs": FACTOR_EPOCHS,
            "stepsPerEpoch": STEPS_PER_EPOCH,
            "evalEvery": 4,
            "snapshotEvery": 4,
            "maxEvalPoses": 0,
            "calibrationBootstrapReplicates": 2000,
            "observationBatchSize": 8192,
            "testRead": False,
        }
    if mode == "all":
        return {"mode": mode, "testRead": False}
    raise ValueError(f"unsupported runner mode: {mode!r}")


def _scan_config(config_id: str) -> dict[str, Any]:
    for config in SCAN_CONFIGS:
        if str(config["configId"]) == str(config_id):
            return dict(config)
    raise ValueError(f"unknown scan configuration: {config_id!r}")


def _factor_spec(name: str, selected_config: Mapping[str, Any]) -> dict[str, Any]:
    if name not in FACTOR_VARIANTS:
        raise ValueError(f"unknown factor variant: {name!r}")
    exposure = name in {"C_pose_frontier_exposure", "D_cross_pose_exposure"}
    cross_pose = name.startswith("B_") or name.startswith("D_")
    return {
        "variant": name,
        "trainingVariant": {
            "A_pose_frontier_no_exposure": "pose_balanced_frontier_without_exposure",
            "B_cross_pose_no_exposure": "cross_pose_operating_without_exposure",
            "C_pose_frontier_exposure": "pose_balanced_frontier_with_exposure",
            "D_cross_pose_exposure": "cross_pose_operating_with_exposure",
        }[name],
        "configId": str(selected_config["configId"]),
        "lossVariant": "cross_pose_operating" if cross_pose else "pose_balanced_frontier",
        "crossPoseEnabled": cross_pose,
        "exposureEnabled": exposure,
        "hyperparameters": dict(selected_config),
        "exposureHiddenDim": 16 if exposure else 0,
        "exposureLossWeight": 0.05 if exposure else 0.0,
        "testRead": False,
    }


def _make_jobs(mode: str, selected_config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    config = _mode_config(mode)
    if mode == "smoke":
        selected_config = _scan_config("S1")
        names = ["S1"]
        stage = "smoke"
        seed = SCAN_SEED
    elif mode == "scan8":
        names = list(SCAN_VARIANTS)
        stage = "scan8"
        seed = SCAN_SEED
    elif mode == "factor16":
        if selected_config is None:
            raise ValueError("factor16 requires the frozen scan configuration")
        names = list(FACTOR_VARIANTS)
        stage = "factor16"
        seed = FACTOR_SEED
    else:
        raise ValueError(f"job construction is not defined for mode {mode!r}")

    jobs: list[dict[str, Any]] = []
    for name in names:
        if stage == "scan8" or stage == "smoke":
            scan = dict(selected_config) if stage == "smoke" else _scan_config(name)
            spec = {
                "variant": name,
                "trainingVariant": "cross_pose_operating_without_exposure",
                "configId": str(scan["configId"]),
                "lossVariant": "cross_pose_operating",
                "crossPoseEnabled": True,
                "exposureEnabled": False,
                "hyperparameters": scan,
                "exposureHiddenDim": 0,
                "exposureLossWeight": 0.0,
            }
        else:
            spec = _factor_spec(name, selected_config)
        epochs = int(config["epochs"])
        jobs.append(
            {
                **spec,
                "stage": stage,
                "seed": int(seed),
                "epochs": epochs,
                "member": f"{stage}_{name}_seed{int(seed)}_e{epochs}",
                "testRead": False,
            }
        )
    return jobs


def _fraction_for_steps(numerator: int, *, epochs: int, steps_per_epoch: int) -> float:
    total = int(epochs) * int(steps_per_epoch)
    if total <= 0 or numerator < 0:
        raise ValueError("training step schedule is invalid")
    return min(1.0, float(numerator) / float(total))


def _member_dir(model_root: Path, job: Mapping[str, Any]) -> Path:
    return model_root.resolve() / str(job["stage"]) / str(job["member"])


def build_train_command(
    args: argparse.Namespace,
    job: Mapping[str, Any],
    member: Path,
    mode_config: Mapping[str, Any],
) -> list[str]:
    paths = _resolve_paths(args)
    epochs = int(mode_config["epochs"])
    steps_per_epoch = int(mode_config["stepsPerEpoch"])
    hp = dict(job["hyperparameters"])
    cross_pose = bool(job["crossPoseEnabled"])
    exposure = bool(job["exposureEnabled"])
    total_steps = epochs * steps_per_epoch
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"]),
        "--relation-dir", str(paths["relation"]),
        "--runtime-meta", str(paths["runtimeMeta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glbIndex"]),
        "--glb-root", str(paths["glbRoot"]),
        "--subpose-sidecar", str(paths["subpose"]),
        "--output-dir", str(member.resolve()),
        "--experiment-name", f"{EXPERIMENT}_{job['member']}",
        "--variant", str(job["trainingVariant"]),
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_envelope",
        "--instance-calibration-mode", "residual",
        "--loss-variant", str(job["lossVariant"]),
        "--refinement-scope", "all",
        "--epochs", str(epochs),
        "--steps-per-epoch", str(steps_per_epoch),
        "--poses-per-batch", str(int(args.poses_per_batch)),
        "--observation-batch-size", str(int(mode_config["observationBatchSize"])),
        "--eval-every", str(int(mode_config["evalEvery"])),
        "--snapshot-every", str(int(mode_config["snapshotEvery"])),
        "--max-eval-poses", str(int(mode_config["maxEvalPoses"])),
        "--calibration-bootstrap-replicates", str(int(mode_config["calibrationBootstrapReplicates"])),
        "--seed", str(int(job["seed"])),
        "--device", "cuda",
        "--learning-rate", str(COMMON_TRAINING_CONFIG["learningRate"]),
        "--weight-decay", str(COMMON_TRAINING_CONFIG["weightDecay"]),
        "--survival-loss-weight", str(COMMON_TRAINING_CONFIG["survivalLossWeight"]),
        "--relation-consistency-weight", str(COMMON_TRAINING_CONFIG["relationConsistencyWeight"]),
        "--utility-loss-weight", "0.0",
        "--download-loss-weight", "0.0",
        "--boundary-tail-weight", "0.0",
        "--negative-band-weight", "0.0",
        "--glb-resource-weight", "0.0",
        "--relation-gradient-cap", str(COMMON_TRAINING_CONFIG["relationGradientCap"]),
        "--schedule-gradient-cap", "0.0",
        "--efficiency-gradient-cap", "0.0",
        "--instance-calibration-regularization-weight", str(COMMON_TRAINING_CONFIG["instanceCalibrationRegularizationWeight"]),
        "--instance-calibration-max-abs", str(COMMON_TRAINING_CONFIG["instanceCalibrationMaxAbs"]),
        "--sparse-instance-penalty", str(COMMON_TRAINING_CONFIG["sparseInstancePenalty"]),
        "--instance-calibration-warmup-fraction", str(
            0.0
            if total_steps <= 400
            else _fraction_for_steps(400, epochs=epochs, steps_per_epoch=steps_per_epoch)
        ),
        "--instance-calibration-ramp-fraction", str(
            1.0
            if total_steps <= 400
            else _fraction_for_steps(800, epochs=epochs, steps_per_epoch=steps_per_epoch)
        ),
        "--rvl-count-weight", "0.0",
        "--rvl-rank-weight", "0.0",
        "--frontier-loss-weight", str(POSE_FRONTIER_CONFIG["frontierLossWeight"] if not cross_pose else 0.0),
        "--frontier-positive-mass-fraction", str(POSE_FRONTIER_CONFIG["frontierPositiveMassFraction"]),
        "--frontier-positive-count-cap", str(int(POSE_FRONTIER_CONFIG["frontierPositiveCountCap"])),
        "--frontier-negative-fraction", str(POSE_FRONTIER_CONFIG["frontierNegativeFraction"]),
        "--frontier-negative-count-cap", str(int(POSE_FRONTIER_CONFIG["frontierNegativeCountCap"])),
        "--frontier-margin", str(POSE_FRONTIER_CONFIG["frontierMargin"]),
        "--frontier-temperature", str(POSE_FRONTIER_CONFIG["frontierTemperature"]),
        "--frontier-positive-importance-floor", str(POSE_FRONTIER_CONFIG["frontierPositiveImportanceFloor"]),
        "--frontier-positive-importance-power", str(POSE_FRONTIER_CONFIG["frontierPositiveImportancePower"]),
        "--exposure-supervision-hidden-dim", str(int(job["exposureHiddenDim"])),
        "--exposure-supervision-loss-weight", str(float(job["exposureLossWeight"])),
        "--exposure-supervision-positive-class-fraction", str(float(hp["positiveClassFraction"])),
    ]
    if cross_pose:
        command.extend(
            [
                "--cross-pose-positive-class-fraction", str(float(hp["positiveClassFraction"])),
                "--cross-pose-operating-weight", "1.0",
                "--cross-pose-weighted-recall-target", str(float(hp["weightedRecallTarget"])),
                "--cross-pose-temperature", "0.25",
                "--cross-pose-hard-negative-fraction", "0.01",
                "--cross-pose-hard-negative-count-cap", "512",
                "--cross-pose-hard-negative-mix", "0.50",
                "--cross-pose-augmented-penalty", "10.0",
                "--cross-pose-initial-boundary-logit", "0.0",
                "--cross-pose-dual-initial", "1.0",
                "--cross-pose-dual-learning-rate", "0.05",
                "--cross-pose-dual-maximum", "20.0",
            ]
        )
    if args.allow_missing_glb_costs:
        command.append("--allow-missing-glb-costs")
    if "--initial-checkpoint" in command:
        raise RuntimeError("from-scratch command unexpectedly contains an initial checkpoint")
    if total_steps <= 0:
        raise ValueError("training command has no optimizer steps")
    return command


def _training_artifacts_complete(member: Path, expected_epochs: int) -> bool:
    required = (
        member / "last.pt",
        member / "model_meta.json",
        member / "calibration_ready_summary.json",
        member / "train_history.json",
    )
    if not all(path.is_file() for path in required):
        return False
    try:
        history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
        if not isinstance(history, list) or not history:
            return False
        epochs = [int(row["epoch"]) for row in history if isinstance(row, Mapping) and "epoch" in row]
        return bool(epochs) and max(epochs) >= int(expected_epochs)
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _completed_member(model_root: Path, job: Mapping[str, Any]) -> Path | None:
    base = _member_dir(model_root, job)
    candidates = [base]
    candidates.extend(sorted(base.parent.glob(f"{base.name}_attempt_*")))
    complete = [
        path for path in candidates
        if _training_artifacts_complete(path, int(job["epochs"]))
    ]
    return complete[-1] if complete else None


def _next_member(model_root: Path, job: Mapping[str, Any]) -> Path:
    base = _member_dir(model_root, job)
    if not base.exists() or not any(base.iterdir()):
        return base
    index = 2
    while True:
        candidate = base.parent / f"{base.name}_attempt_{index:03d}"
        if not candidate.exists():
            return candidate
        index += 1


def _run_logged(
    command: Sequence[str],
    *,
    gpu: int,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "command": list(command),
        "gpu": int(gpu),
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "testRead": False,
    }


def _run_queue(
    jobs: Sequence[tuple[str, Sequence[str], Path]],
    *,
    gpu_ids: Sequence[int],
    log_root: Path,
) -> list[dict[str, Any]]:
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    if len(set(int(value) for value in gpu_ids)) != len(gpu_ids) or any(int(value) < 0 for value in gpu_ids):
        raise ValueError("gpu IDs must be unique non-negative integers")
    pending: Queue[tuple[str, Sequence[str], Path]] = Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu: int) -> list[dict[str, Any]]:
        owned: list[dict[str, Any]] = []
        while True:
            try:
                name, command, output_dir = pending.get_nowait()
            except Empty:
                return owned
            result = {
                "member": name,
                "outputDir": str(output_dir),
                **_run_logged(
                    command,
                    gpu=gpu,
                    stdout_path=log_root / f"{name}.stdout.log",
                    stderr_path=log_root / f"{name}.stderr.log",
                ),
            }
            owned.append(result)
            pending.task_done()
            print(
                json.dumps(
                    {key: result[key] for key in ("member", "gpu", "returnCode", "elapsedSeconds")},
                    sort_keys=True,
                ),
                flush=True,
            )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, int(gpu)) for gpu in gpu_ids]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [row for row in results if int(row["returnCode"]) != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} jobs failed; inspect runner logs")
    return sorted(results, key=lambda row: str(row["member"]))


def _training_result_for_job(model_root: Path, job: Mapping[str, Any], *, reused: bool = False) -> dict[str, Any]:
    member = _completed_member(model_root, job)
    if member is None:
        raise RuntimeError(f"training member is incomplete: {job['member']}")
    return {
        "stage": str(job["stage"]),
        "variant": str(job["variant"]),
        "configId": str(job["configId"]),
        "seed": int(job["seed"]),
        "epochs": int(job["epochs"]),
        "member": str(job["member"]),
        "outputDir": str(member),
        "returnCode": 0,
        "reusedCompletedMember": bool(reused),
        "testRead": False,
    }


def _load_existing_training_results(
    path: Path,
    jobs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    payload = _load_json(path)
    rows = payload.get("results")
    if payload.get("testRead") is not False or not isinstance(rows, list):
        raise ValueError(f"invalid immutable training result record: {path}")
    expected = {str(job["member"]) for job in jobs}
    actual = {str(row.get("member")) for row in rows if isinstance(row, Mapping)}
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError(f"existing training result record is incomplete: {path}")
    if any(int(row.get("returnCode", 1)) != 0 for row in rows if isinstance(row, Mapping)):
        raise RuntimeError(f"existing training result record contains a failed member: {path}")
    return [dict(row) for row in rows]


def _run_training_stage(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
    mode_config: Mapping[str, Any],
    model_root: Path,
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    result_path = benchmark_root / f"{mode_config['stage']}_training_results.json"
    existing = _load_existing_training_results(result_path, jobs)
    if existing is not None:
        return sorted(existing, key=lambda row: str(row["member"]))
    pending: list[tuple[str, Sequence[str], Path]] = []
    reused: list[dict[str, Any]] = []
    for job in jobs:
        complete = _completed_member(model_root, job)
        if complete is not None:
            reused.append(_training_result_for_job(model_root, job, reused=True))
            continue
        member = _next_member(model_root, job)
        pending.append(
            (
                str(job["member"]),
                build_train_command(args, job, member, mode_config),
                member,
            )
        )
    new_results = _run_queue(
        pending,
        gpu_ids=args.gpu_ids,
        log_root=benchmark_root / "logs" / str(mode_config["stage"]) / "train",
    ) if pending else []
    for row in new_results:
        job = next(job for job in jobs if str(job["member"]) == str(row["member"]))
        if int(row["returnCode"]) != 0:
            raise RuntimeError(f"training failed for {row['member']}")
        if not _training_artifacts_complete(Path(row["outputDir"]), int(job["epochs"])):
            raise RuntimeError(f"training exited without complete artifacts: {row['member']}")
        row.update(
            {
                "stage": str(job["stage"]),
                "variant": str(job["variant"]),
                "configId": str(job["configId"]),
                "seed": int(job["seed"]),
                "epochs": int(job["epochs"]),
                "reusedCompletedMember": False,
                "testRead": False,
            }
        )
    results = sorted(reused + new_results, key=lambda row: str(row["member"]))
    _write_json_once(result_path, {"schema": f"{EXPERIMENT}-{mode_config['stage']}-training-v1", "results": results, "testRead": False})
    return results


def _checkpoint_for_evaluation(member: Path) -> tuple[Path, bool]:
    summary = _load_json(member / "calibration_ready_summary.json")
    if summary.get("testRead") is not False:
        raise ValueError(f"calibration summary opened test data: {member}")
    if summary.get("status") == "safe" and (member / "best_safe.pt").is_file():
        return member / "best_safe.pt", True
    diagnostic = member / "best_diagnostic.pt"
    if diagnostic.is_file():
        return diagnostic, False
    raise FileNotFoundError(f"no selected checkpoint exists in {member}")


def _evaluation_path(benchmark_root: Path, job: Mapping[str, Any]) -> Path:
    return benchmark_root / "members" / str(job["stage"]) / f"{job['member']}.json"


def build_evaluate_command(
    args: argparse.Namespace,
    job: Mapping[str, Any],
    member: Path,
    output: Path,
) -> list[str]:
    paths = _resolve_paths(args)
    checkpoint, _safe = _checkpoint_for_evaluation(member)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"]),
        "--runtime-meta", str(paths["runtimeMeta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glbIndex"]),
        "--glb-root", str(paths["glbRoot"]),
        "--relation-dir", str(paths["relation"]),
        "--model-meta", str((member / "model_meta.json").resolve()),
        "--calibration", str((member / "calibration_ready_summary.json").resolve()),
        "--output", str(output.resolve()),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--seed", str(int(job["seed"])),
        "--device", "cuda",
        "--bootstrap-replicates", str(QUICK_VALIDATION_BOOTSTRAP_REPLICATES),
        "--allow-unsafe-diagnostic",
        "--persist-ids",
    ]


def _run_evaluation_stage(
    args: argparse.Namespace,
    jobs: Sequence[Mapping[str, Any]],
    training_results: Sequence[Mapping[str, Any]],
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    result_path = benchmark_root / f"{jobs[0]['stage']}_evaluation_results.json"
    if result_path.is_file():
        payload = _load_json(result_path)
        rows = payload.get("results")
        if payload.get("testRead") is not False or not isinstance(rows, list):
            raise ValueError(f"invalid immutable evaluation result record: {result_path}")
        expected = {f"evaluate_{job['member']}" for job in jobs}
        actual = {str(row.get("member")) for row in rows if isinstance(row, Mapping)}
        if actual != expected or len(rows) != len(expected):
            raise RuntimeError(f"existing evaluation result record is incomplete: {result_path}")
        if any(int(row.get("returnCode", 1)) != 0 for row in rows if isinstance(row, Mapping)):
            raise RuntimeError(f"existing evaluation result record contains a failed member: {result_path}")
        return [dict(row) for row in sorted(rows, key=lambda row: str(row.get("member")))]
    by_member = {str(row["member"]): Path(str(row["outputDir"])) for row in training_results}
    pending: list[tuple[str, Sequence[str], Path]] = []
    reused: list[dict[str, Any]] = []
    for job in jobs:
        output = _evaluation_path(benchmark_root, job)
        if output.is_file():
            reused.append({
                "member": f"evaluate_{job['member']}",
                "stage": str(job["stage"]),
                "variant": str(job["variant"]),
                "configId": str(job["configId"]),
                "seed": int(job["seed"]),
                "output": str(output),
                "returnCode": 0,
                "reusedExisting": True,
                "testRead": False,
            })
            continue
        member = by_member.get(str(job["member"]))
        if member is None:
            raise RuntimeError(f"missing training result for {job['member']}")
        pending.append(
            (
                f"evaluate_{job['member']}",
                build_evaluate_command(args, job, member, output),
                output,
            )
        )
    new_results = _run_queue(
        pending,
        gpu_ids=args.gpu_ids,
        log_root=benchmark_root / "logs" / str(jobs[0]["stage"]) / "evaluate",
    ) if pending else []
    for row in new_results:
        if int(row["returnCode"]) != 0:
            raise RuntimeError(f"validation evaluation failed for {row['member']}")
        row["output"] = str(row["outputDir"])
        row["testRead"] = False
    results = sorted(reused + new_results, key=lambda row: str(row["member"]))
    stage = str(jobs[0]["stage"])
    _write_json_once(
        result_path,
        {"schema": f"{EXPERIMENT}-{stage}-evaluation-v1", "results": results, "testRead": False},
    )
    return results


def _metric(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric == numeric and abs(numeric) != float("inf") else None


def _validate_runtime_contract(runtime: Mapping[str, Any], evaluation_path: Path) -> None:
    feature_dim = int(runtime.get("runtimeFeatureDim", -1))
    feature_bytes = int(runtime.get("runtimeFeatureBytes", -1))
    input_dim = int(runtime.get("inferenceInputDim", -1))
    if feature_dim != RUNTIME_FEATURE_DIM:
        raise ValueError(
            f"runtime feature dimension changed in {evaluation_path}: "
            f"{feature_dim} != {RUNTIME_FEATURE_DIM}"
        )
    if feature_bytes != HKUST_RUNTIME_FEATURE_BYTES:
        raise ValueError(
            f"runtime feature table size changed in {evaluation_path}: "
            f"{feature_bytes} != {HKUST_RUNTIME_FEATURE_BYTES} bytes"
        )
    if input_dim != RUNTIME_QUERY_INPUT_DIM:
        raise ValueError(
            f"runtime query input dimension changed in {evaluation_path}: "
            f"{input_dim} != {RUNTIME_QUERY_INPUT_DIM}"
        )
    if feature_bytes >= MAX_NEURAL_ASSET_BYTES:
        raise ValueError(
            f"runtime feature table exceeds the neural asset budget in {evaluation_path}"
        )


def _evaluation_row(job: Mapping[str, Any], member: Path, evaluation_path: Path) -> dict[str, Any]:
    calibration = _load_json(member / "calibration_ready_summary.json")
    evaluation = _load_json(evaluation_path)
    if evaluation.get("split") != "validation" or evaluation.get("testRead") is not False:
        raise ValueError(f"evaluation is not validation-only: {evaluation_path}")
    aggregate = evaluation.get("aggregate")
    pose_macro = evaluation.get("poseMacro")
    if not isinstance(aggregate, Mapping) or not isinstance(pose_macro, Mapping):
        raise ValueError(f"evaluation lacks aggregate/poseMacro metrics: {evaluation_path}")
    runtime = evaluation.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"evaluation lacks runtime metrics: {evaluation_path}")
    _validate_runtime_contract(runtime, evaluation_path)
    selected_checkpoint = (
        calibration.get("bestSafe")
        if calibration.get("status") == "safe"
        else calibration.get("bestDiagnostic")
    )
    selected_epoch = (
        int(selected_checkpoint.get("epoch", 0))
        if isinstance(selected_checkpoint, Mapping)
        else 0
    )
    history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
    selected_history = next(
        (
            row
            for row in history
            if isinstance(row, Mapping) and int(row.get("epoch", -1)) == selected_epoch
        ),
        None,
    )
    training_summary = (
        selected_history.get("trainMetricSummary", {})
        if isinstance(selected_history, Mapping)
        else {}
    )

    def training_mean(name: str) -> float | None:
        value = training_summary.get(name)
        return _metric(value.get("mean")) if isinstance(value, Mapping) else None

    validation_weighted_recall = _metric(aggregate.get("weightedRecall"))
    validation_weighted_recall_lcb = _metric(
        aggregate.get("weightedRecallLowerConfidenceBound")
    )
    safe_on_validation = (
        validation_weighted_recall is not None
        and validation_weighted_recall > TARGET_WEIGHTED_RECALL
        and validation_weighted_recall_lcb is not None
        and validation_weighted_recall_lcb > TARGET_WEIGHTED_RECALL
    )
    return {
        "stage": str(job["stage"]),
        "variant": str(job["variant"]),
        "configId": str(job["configId"]),
        "seed": int(job["seed"]),
        "member": str(job["member"]),
        "threshold": _metric(evaluation.get("threshold")),
        "checkpointSafeOnCalibration": calibration.get("status") == "safe",
        "safeOnValidation": safe_on_validation,
        "eligibleSafe": calibration.get("status") == "safe" and safe_on_validation,
        "aggregate": {
            key: _metric(aggregate.get(key))
            for key in (
                "precision", "recall", "weightedRecall", "weightedRecallLowerConfidenceBound",
                "accuracy", "balancedAccuracy", "specificity", "f1", "usefulCull", "badCull",
                "avgCandidateCount", "avgGtCount", "avgPredCount",
                "predOverCandidate", "predOverGt", "predictedGlbCount", "candidateGlbCount",
                "glbCountReduction", "predictedGlbBytes", "candidateGlbBytes",
                "glbByteReduction", "glbBytesAtAchievedVisualUtility", "downloadUtilityRecall",
            )
        },
        "poseMacro": {
            key: _metric(pose_macro.get(key))
            for key in (
                "precision", "recall", "weightedRecall", "accuracy", "balancedAccuracy",
                "specificity", "f1", "usefulCull", "badCull",
            )
        },
        "runtime": {
            "device": runtime.get("device"),
            "elapsedSeconds": _metric(runtime.get("elapsedSeconds")),
            "runtimeFeatureBytes": int(runtime.get("runtimeFeatureBytes", -1)),
            "runtimeFeatureDim": int(runtime.get("runtimeFeatureDim", -1)),
            "inferenceInputDim": int(runtime.get("inferenceInputDim", -1)),
        },
        "trainingDiagnostics": {
            "selectedEpoch": selected_epoch,
            "viewcellExposureMae": training_mean("viewcellExposureMae"),
            "viewcellExposureBoundaryMae": training_mean(
                "viewcellExposureBoundaryMae"
            ),
            "viewcellExposureStableMae": training_mean(
                "viewcellExposureStableMae"
            ),
            "viewcellExposureInvisibleMae": training_mean(
                "viewcellExposureInvisibleMae"
            ),
            "crossPoseTrainBoundaryLogit": training_mean(
                "crossPoseBoundaryLogit"
            ),
            "crossPoseSoftWeightedRecall": training_mean(
                "crossPoseSoftWeightedRecall"
            ),
            "counterfactualViewGap": training_mean(
                "counterfactualViewGap"
            ),
            "counterfactualViewWorstGap": training_mean(
                "counterfactualViewWorstGap"
            ),
            "counterfactualViewViolationFraction": training_mean(
                "counterfactualViewViolationFraction"
            ),
            "counterfactualViewPairCount": training_mean(
                "counterfactualViewPairCount"
            ),
            "cudaPeakMemoryAllocatedBytes": training_mean(
                "cudaPeakMemoryAllocatedBytes"
            ),
            "boundaryCalibrationComparison": (
                selected_history.get("crossPoseOperatingDiagnostics")
                if isinstance(selected_history, Mapping)
                else None
            ),
        },
        "testRead": False,
    }


def _scan_rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    aggregate = row["aggregate"]
    if bool(row.get("eligibleSafe")):
        return (
            1.0,
            float(aggregate.get("balancedAccuracy") or float("-inf")),
            float(aggregate.get("precision") or float("-inf")),
            float(aggregate.get("usefulCull") or float("-inf")),
            float(aggregate.get("accuracy") or float("-inf")),
            -float(aggregate.get("avgPredCount") or float("inf")),
        )
    lcb = aggregate.get("weightedRecallLowerConfidenceBound")
    return (
        0.0,
        float(lcb if lcb is not None else aggregate.get("weightedRecall") or float("-inf")),
        float(aggregate.get("weightedRecall") or float("-inf")),
        float(aggregate.get("balancedAccuracy") or float("-inf")),
        float(aggregate.get("precision") or float("-inf")),
        float(aggregate.get("accuracy") or float("-inf")),
        float(aggregate.get("usefulCull") or float("-inf")),
    )


def select_scan_configuration(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Freeze one winner; safety gates ranking but never cancel factor16."""
    if len(rows) != len(SCAN_CONFIGS):
        raise ValueError(f"scan must contain exactly {len(SCAN_CONFIGS)} rows")
    ids = {str(row.get("configId")) for row in rows}
    if ids != {str(config["configId"]) for config in SCAN_CONFIGS}:
        raise ValueError("scan rows do not contain the registered S1-S4 IDs")
    safe_rows = [row for row in rows if bool(row.get("eligibleSafe"))]
    pool = safe_rows if safe_rows else list(rows)
    winner = max(pool, key=_scan_rank)
    return {
        "schema": f"{EXPERIMENT}-scan-selection-v1",
        "experiment": EXPERIMENT,
        "selectionSplit": "validation",
        "thresholdSource": "each checkpoint calibration split",
        "weightedRecallFloor": TARGET_WEIGHTED_RECALL,
        "selectionPool": "safe_candidate_pool" if safe_rows else "diagnostic_candidate_pool",
        "selectionRule": (
            "safe rows first; balanced accuracy, precision, useful cull, accuracy, "
            "and lower prediction count; if no safe row, weighted recall LCB/point, "
            "balanced accuracy, precision, accuracy, and useful cull"
        ),
        "selected": dict(winner),
        "rows": sorted((dict(row) for row in rows), key=lambda row: str(row["configId"])),
        "factorStageMustRun": True,
        "testRead": False,
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty factor matrix")
    metrics = (
        "precision", "recall", "weightedRecall", "accuracy", "balancedAccuracy", "specificity",
        "f1", "usefulCull", "badCull", "avgCandidateCount", "avgGtCount", "avgPredCount",
        "predOverCandidate", "predOverGt", "predictedGlbCount", "candidateGlbCount",
        "glbCountReduction", "predictedGlbBytes", "candidateGlbBytes",
        "glbByteReduction", "glbBytesAtAchievedVisualUtility", "downloadUtilityRecall",
    )
    summary: dict[str, Any] = {"memberCount": len(rows), "safeMemberCount": sum(bool(row.get("eligibleSafe")) for row in rows), "testRead": False}
    for key in metrics:
        values = [_metric(row.get("aggregate", {}).get(key)) for row in rows]
        finite = [value for value in values if value is not None]
        summary[key] = {
            "availableCount": len(finite),
            "mean": statistics.fmean(finite) if finite else None,
            "minimum": min(finite) if finite else None,
            "maximum": max(finite) if finite else None,
        }
    summary["rows"] = [dict(row) for row in rows]
    return summary


def _load_or_build_scan_rows(
    jobs: Sequence[Mapping[str, Any]],
    training_results: Sequence[Mapping[str, Any]],
    evaluation_results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    training_by_member = {str(row["member"]): Path(str(row["outputDir"])) for row in training_results}
    evaluation_by_member = {str(row["member"]).removeprefix("evaluate_"): Path(str(row.get("output") or row["outputDir"])) for row in evaluation_results}
    return [
        _evaluation_row(job, training_by_member[str(job["member"])], evaluation_by_member[str(job["member"])])
        for job in jobs
    ]


def _run_scan(args: argparse.Namespace, model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    mode_config = _mode_config("scan8")
    jobs = _make_jobs("scan8")
    training = _run_training_stage(args, jobs, mode_config, model_root, benchmark_root)
    evaluations = _run_evaluation_stage(args, jobs, training, benchmark_root)
    rows = _load_or_build_scan_rows(jobs, training, evaluations)
    selection = select_scan_configuration(rows)
    _write_json_once(benchmark_root / "selected_configuration.json", selection)
    _write_json_once(benchmark_root / "scan8_summary.json", {"schema": f"{EXPERIMENT}-scan8-summary-v1", "rows": rows, "selection": selection, "testRead": False})
    return selection


def _run_factor(args: argparse.Namespace, selected: Mapping[str, Any], model_root: Path, benchmark_root: Path) -> dict[str, Any]:
    selected_record = selected.get("selected")
    if not isinstance(selected_record, Mapping):
        raise ValueError("frozen scan selection has no selected configuration")
    selected_config = _scan_config(str(selected_record.get("configId")))
    mode_config = _mode_config("factor16")
    jobs = _make_jobs("factor16", selected_config)
    training = _run_training_stage(args, jobs, mode_config, model_root, benchmark_root)
    evaluations = _run_evaluation_stage(args, jobs, training, benchmark_root)
    rows = _load_or_build_scan_rows(jobs, training, evaluations)
    summary = _summarize_rows(rows)
    summary["selectedScanConfiguration"] = selected_config
    _write_json_once(benchmark_root / "factor16_summary.json", summary)
    return summary


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode not in {"smoke", "scan8", "factor16", "all"}:
        raise ValueError(f"unsupported mode: {args.mode}")
    if len(set(int(value) for value in args.gpu_ids)) != len(args.gpu_ids) or any(int(value) < 0 for value in args.gpu_ids):
        raise ValueError("gpu IDs must be unique non-negative integers")
    if args.poses_per_batch <= 0:
        raise ValueError("poses-per-batch must be positive")
    contract = preflight(args)
    model_root = Path(args.model_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json_once(benchmark_root / "preflight.json", contract)

    if args.dry_run:
        scan_jobs = _make_jobs("scan8")
        factor_jobs = _make_jobs("factor16", _scan_config("S1"))
        if args.mode == "all":
            jobs = scan_jobs + factor_jobs
        elif args.mode == "factor16":
            jobs = factor_jobs
        else:
            jobs = _make_jobs(args.mode)
        commands = []
        for job in jobs:
            stage_config = _mode_config(str(job["stage"]))
            commands.append(list(build_train_command(args, job, _member_dir(model_root, job), stage_config)))
        print(json.dumps({"mode": args.mode, "jobCount": len(commands), "commands": commands, "testRead": False}, ensure_ascii=False, indent=2))
        return {"status": "dry_run", "jobCount": len(commands), "testRead": False}

    if args.mode == "smoke":
        jobs = _make_jobs("smoke")
        result = _run_training_stage(args, jobs, _mode_config("smoke"), model_root, benchmark_root)
        return {"status": "smoke_complete", "members": result, "testRead": False}

    selected: dict[str, Any]
    if args.mode in {"scan8", "all"}:
        selected = _run_scan(args, model_root, benchmark_root)
    else:
        selection_path = benchmark_root / "selected_configuration.json"
        if not selection_path.is_file():
            raise FileNotFoundError("factor16 requires selected_configuration.json from scan8")
        selected = _load_json(selection_path)
    if args.mode in {"factor16", "all"}:
        summary = _run_factor(args, selected, model_root, benchmark_root)
        return {"status": "factor16_complete", "selection": selected, "summary": summary, "testRead": False}
    return {"status": "scan8_complete", "selection": selected, "factorStageMustRun": True, "testRead": False}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "scan8", "factor16", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset-dir", default="")
    parser.add_argument("--relation-dir", default="")
    parser.add_argument("--runtime-meta", default="")
    parser.add_argument("--initial-geo-features", default="")
    parser.add_argument("--subpose-sidecar", default="")
    parser.add_argument("--glb-index", default="")
    parser.add_argument("--glb-root", default="")
    parser.add_argument("--model-root", type=Path, default=ROOT / "neural_instance_culling/model/out" / OUTPUT_TAG)
    parser.add_argument("--benchmark-root", type=Path, default=ROOT / "neural_instance_culling/benchmark/out" / OUTPUT_TAG)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--poses-per-batch", type=int, default=4)
    parser.add_argument("--allow-missing-glb-costs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_experiment(args)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
