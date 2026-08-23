#!/usr/bin/env python3
"""Run the fixed V4 visibility-only mainline, scan, and 40-epoch ablations.

The runner never selects a threshold on test and never initializes from an
existing checkpoint.  It fixes the V4 relation/survival and view-cell moment
architecture, while the registered visibility objective contains
pose-balanced classification, a one-sided RVL recall guard, and shared-tail
logit/contrastive separation.  Visual utility, download priority, and GLB
resource objectives are disabled and excluded from configuration ranking.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_pvs.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs.py"
EXPERIMENT = "pvs_v4_integrated_visibility_mainline_v1"
OUTPUT_TAG = f"{EXPERIMENT}_20260821"
SCAN_SEED = 20260801
FORMAL_SEEDS = (20260801, 20260802, 20260803)
SCAN_EPOCHS = 10
FORMAL_EPOCHS = 40
STEPS_PER_EPOCH = 100
FORMAL_STEPS_PER_EPOCH = 900
FORMAL_BOOTSTRAP_REPLICATES = 10000
UPDATE_BUDGET_CHECK_EPOCHS = 12
UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH = 300
UPDATE_BUDGET_CHECK_STAGE = "update_budget_check12x300"
FORMAL_FULL_STAGE = "formal40_s02_full"
FORMAL_ABLATION_STAGE = "formal40_s02_ablation"
CORE_ABLATION_STAGE = "core_ablation"
EXPECTED_SPLITS = {
    "train": 5926,
    "calibration": 659,
    "validation": 730,
    "test": 684,
    "guard": 0,
}
WEIGHTED_RECALL_FLOOR = 0.99
EVALUATION_SCHEMA = (
    "pvs-bounded-relation-prior-instance-calibrated-moment-v4-evaluation-v1"
)


SCAN_CONFIGS: tuple[dict[str, Any], ...] = (
    {"name": "s00_guard015_sep010_mix015", "lr": 1e-4, "guard": 0.15, "separation": 0.10, "mix": 0.15, "margin": 0.30},
    {"name": "s01_guard030_sep015_mix015", "lr": 2e-4, "guard": 0.30, "separation": 0.15, "mix": 0.15, "margin": 0.50},
    {"name": "s02_guard030_sep020_mix025", "lr": 2e-4, "guard": 0.30, "separation": 0.20, "mix": 0.25, "margin": 0.50},
    {"name": "s03_guard045_sep020_mix025", "lr": 2e-4, "guard": 0.45, "separation": 0.20, "mix": 0.25, "margin": 0.50},
    {"name": "s04_guard030_sep030_mix035", "lr": 3e-4, "guard": 0.30, "separation": 0.30, "mix": 0.35, "margin": 0.50},
    {"name": "s05_guard045_sep030_mix035", "lr": 1e-4, "guard": 0.45, "separation": 0.30, "mix": 0.35, "margin": 0.75},
    {"name": "s06_guard020_sep020_mix000", "lr": 2e-4, "guard": 0.20, "separation": 0.20, "mix": 0.00, "margin": 0.50},
    {"name": "s07_guard045_sep015_mix050", "lr": 3e-4, "guard": 0.45, "separation": 0.15, "mix": 0.50, "margin": 0.50},
)

UPDATE_BUDGET_CHECK_CONFIGS: tuple[dict[str, Any], ...] = (
    SCAN_CONFIGS[1],
    SCAN_CONFIGS[2],
)

FORMAL_VARIANTS: dict[str, dict[str, Any]] = {
    "full": {
        "variant": "full_integrated_visibility_mainline",
        "relation": "bounded_hierarchical",
        "spectral": "moment_envelope",
    },
    "without_hierarchical_relation_prior": {
        "variant": "integrated_without_bounded_relation",
        "relation": "geometry_only",
        "spectral": "moment_envelope",
    },
    "without_viewcell_moment_envelope": {
        "variant": "integrated_without_viewcell_moment_envelope",
        "relation": "bounded_hierarchical",
        "spectral": "point",
    },
    "without_rvl_recall_guard": {
        "variant": "integrated_without_rvl_recall_guard",
        "relation": "bounded_hierarchical",
        "spectral": "moment_envelope",
        "guard": 0.0,
    },
    "without_contrastive_separation": {
        "variant": "integrated_without_contrastive_separation",
        "relation": "bounded_hierarchical",
        "spectral": "moment_envelope",
        "mix": 0.0,
    },
}
FORMAL_ABLATIONS = tuple(name for name in FORMAL_VARIANTS if name != "full")

CORE_CONFIG: dict[str, Any] = {
    "name": "final",
    "lr": 2e-4,
    "guard": 0.30,
    "separation": 0.20,
    "mix": 0.0,
    "margin": 0.50,
}
CORE_VARIANTS: dict[str, dict[str, Any]] = {
    "no_relation": {
        "variant": "core_no_relation",
        "representation": "survival",
        "relation": "geometry_only",
        "calibration": "residual",
        "spectral": "moment_envelope",
    },
    "no_survival": {
        "variant": "core_no_survival",
        "representation": "none",
        "relation": "none",
        "calibration": "disabled",
        "spectral": "moment_envelope",
    },
    "generic28": {
        "variant": "core_generic28",
        "representation": "generic28",
        "relation": "none",
        "calibration": "disabled",
        "spectral": "moment_envelope",
    },
    "no_moment": {
        "variant": "core_no_moment",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "point",
    },
    "no_recall_guard": {
        "variant": "core_no_recall_guard",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "moment_envelope",
        "guard": 0.0,
    },
    "no_tail_margin": {
        "variant": "core_no_tail_margin",
        "representation": "survival",
        "relation": "bounded_hierarchical",
        "calibration": "residual",
        "spectral": "moment_envelope",
        "separation": 0.0,
    },
}


def _variant_spec(name: str) -> Mapping[str, Any]:
    if name in CORE_VARIANTS:
        return CORE_VARIANTS[name]
    return FORMAL_VARIANTS[name]


def _paths(data_root: Path) -> dict[str, Path]:
    root = data_root.resolve()
    return {
        "dataset": root / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_main_stratified_calibration_fov66_v1",
        "relation": root / "neural_instance_culling/dataset/out/pvs_v4_integrated_visibility_mainline_v1/bounded_relation_csr_v3",
        "runtime_meta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root / "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin",
        "glb_index": root / "hkust-v3/assets/glbIndex.json",
        "glb_root": root / "hkust-v3/assets",
        "subpose": root / "neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar",
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(data_root: Path) -> dict[str, Any]:
    paths = _paths(data_root)
    missing = [str(path) for path in paths.values() if not path.exists()]
    missing.extend(str(path) for path in (TRAIN, EVALUATE) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"missing registered mainline inputs: {missing}")
    dataset_meta = _load_json(paths["dataset"] / "dataset_meta.json")
    if dataset_meta.get("schema") != "pose-csr-explicit-four-way-split-v1":
        raise ValueError("mainline requires the explicit four-way split dataset")
    split_counts = {
        str(key): int(value)
        for key, value in (dataset_meta.get("splitCounts") or {}).items()
    }
    if split_counts != EXPECTED_SPLITS:
        raise ValueError(
            f"mainline split counts changed: expected={EXPECTED_SPLITS}, actual={split_counts}"
        )
    files = dataset_meta.get("files") or {}
    for field in ("queryCenterWorld", "candidateCameraWorld", "viewcellRadiusM", "candidateIds", "visibleIds", "visibleWeights"):
        if field not in files or not (paths["dataset"] / str(files[field])).is_file():
            raise ValueError(f"mainline dataset is missing {field}")
    relation_meta = _load_json(paths["relation"] / "relation_csr_meta.json")
    if relation_meta.get("schema") != "pvs-viewcell-train-observed-relation-csr-v3":
        raise ValueError("mainline relation artifact has the wrong schema")
    return {
        "schema": "pvs-v4-integrated-visibility-mainline-preflight-v1",
        "experiment": EXPERIMENT,
        "splitCounts": split_counts,
        "initialCheckpoint": None,
        "relationSource": "bounded_hierarchical",
        "spectralMode": "moment_envelope",
        "instanceCalibrationMode": "residual",
        "lossVariant": "pose_balanced_rvl_contrastive",
        "disabledTrainingObjectives": [
            "visual utility",
            "download priority",
            "GLB resource budget",
            "historical RVL count and duplicate rank terms",
        ],
        "scanConfigCount": len(SCAN_CONFIGS),
        "formalVariants": list(FORMAL_VARIANTS),
        "coreAblationVariants": list(CORE_VARIANTS),
        "coreAblationNewMemberCount": len(CORE_VARIANTS) * len(FORMAL_SEEDS),
        "formalSeeds": list(FORMAL_SEEDS),
        "formalEpochs": FORMAL_EPOCHS,
        "formalStepsPerEpoch": FORMAL_STEPS_PER_EPOCH,
        "testRead": False,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
    }


def _member_name(stage: str, variant: str, config: str, seed: int, epochs: int) -> str:
    return f"{stage}_{variant}_{config}_seed{int(seed)}_e{int(epochs)}"


def _member_dir(model_root: Path, stage: str, variant: str, config: str, seed: int, epochs: int) -> Path:
    return model_root.resolve() / _member_name(stage, variant, config, seed, epochs)


def _effective_config(base: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key in ("guard", "separation", "mix", "margin", "lr"):
        if key in variant:
            result[key] = variant[key]
    return result


def build_train_command(
    data_root: Path,
    member: Path,
    config: Mapping[str, Any],
    variant_name: str,
    *,
    seed: int,
    epochs: int,
    smoke: bool = False,
    steps_per_epoch: int = STEPS_PER_EPOCH,
    eval_every: int = 4,
) -> list[str]:
    paths = _paths(data_root)
    spec = _variant_spec(variant_name)
    hp = _effective_config(config, spec)
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"]),
        "--relation-dir", str(paths["relation"]),
        "--runtime-meta", str(paths["runtime_meta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glb_index"]),
        "--glb-root", str(paths["glb_root"]),
        "--subpose-sidecar", str(paths["subpose"]),
        "--output-dir", str(member.resolve()),
        "--experiment-name", f"{EXPERIMENT}_{member.name}",
        "--variant", str(spec["variant"]),
        "--occlusion-representation", str(spec.get("representation", "survival")),
        "--relation-source", str(spec["relation"]),
        "--spectral-mode", str(spec["spectral"]),
        "--instance-calibration-mode", str(spec.get("calibration", "residual")),
        "--loss-variant", "pose_balanced_rvl_contrastive",
        "--refinement-scope", "all",
        "--epochs", "1" if smoke else str(int(epochs)),
        "--steps-per-epoch", "1" if smoke else str(int(steps_per_epoch)),
        "--poses-per-batch", "4",
        "--observation-batch-size", "32768" if smoke else "8192",
        "--eval-every", "1" if smoke else str(int(eval_every)),
        "--snapshot-every", "1" if smoke else str(int(eval_every)),
        "--max-eval-poses", "2" if smoke else "0",
        "--calibration-bootstrap-replicates", "2" if smoke else (str(FORMAL_BOOTSTRAP_REPLICATES) if int(epochs) == FORMAL_EPOCHS else "2000"),
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--learning-rate", str(float(hp["lr"])),
        "--weight-decay", "0.00001",
        "--survival-loss-weight", (
            "0.25" if spec.get("representation", "survival") == "survival" else "0.0"
        ),
        "--relation-consistency-weight", (
            "0.10" if spec.get("representation", "survival") == "survival" else "0.0"
        ),
        "--instance-calibration-regularization-weight", (
            "0.02" if spec.get("representation", "survival") == "survival" else "0.0"
        ),
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--relation-gradient-cap", "0.25",
        "--integrated-rvl-recall-guard-weight", str(float(hp["guard"])),
        "--integrated-rvl-recall-target", "0.99",
        "--integrated-rvl-recall-temperature", "0.05",
        "--integrated-rvl-pose-cvar-fraction", "0.25",
        "--integrated-rvl-pose-cvar-weight", "0.25",
        "--integrated-separation-weight", str(float(hp["separation"])),
        "--integrated-contrastive-mix", str(float(hp["mix"])),
        "--integrated-tail-ramp-fraction", "0.15",
        "--integrated-contrastive-hidden-dim", "64",
        "--integrated-contrastive-projection-dim", "32",
        "--integrated-contrastive-temperature", "0.10",
        "--frontier-positive-mass-fraction", "0.005",
        "--frontier-positive-count-cap", "64",
        "--frontier-negative-fraction", "0.01",
        "--frontier-negative-count-cap", "256",
        "--frontier-margin", str(float(hp["margin"])),
        "--frontier-temperature", "0.25",
        "--frontier-positive-importance-floor", "0.5",
        "--frontier-positive-importance-power", "0.5",
    ]
    if "--initial-checkpoint" in command:
        raise RuntimeError("mainline command unexpectedly contains an initial checkpoint")
    return command


def _member_contract(
    member: Path,
    spec: tuple[str, str, Mapping[str, Any], int, int],
) -> dict[str, Any]:
    _stage, variant, config, seed, epochs = spec
    variant_spec = _variant_spec(variant)
    hp = _effective_config(config, variant_spec)
    return {
        "experiment_name": f"{EXPERIMENT}_{member.name}",
        "variant": str(variant_spec["variant"]),
        "occlusion_representation": str(
            variant_spec.get("representation", "survival")
        ),
        "relation_source": str(variant_spec["relation"]),
        "spectral_mode": str(variant_spec["spectral"]),
        "instance_calibration_mode": str(
            variant_spec.get("calibration", "residual")
        ),
        "loss_variant": "pose_balanced_rvl_contrastive",
        "epochs": int(epochs),
        "steps_per_epoch": _steps_per_epoch(_stage),
        "seed": int(seed),
        "learning_rate": float(hp["lr"]),
        "integrated_rvl_recall_guard_weight": float(hp["guard"]),
        "integrated_separation_weight": float(hp["separation"]),
        "integrated_contrastive_mix": float(hp["mix"]),
        "frontier_margin": float(hp["margin"]),
        "utility_loss_weight": 0.0,
        "download_loss_weight": 0.0,
        "glb_resource_weight": 0.0,
    }


def _steps_per_epoch(stage: str) -> int:
    if stage == UPDATE_BUDGET_CHECK_STAGE:
        return UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH
    if stage in {FORMAL_FULL_STAGE, FORMAL_ABLATION_STAGE}:
        return FORMAL_STEPS_PER_EPOCH
    if stage == CORE_ABLATION_STAGE:
        return FORMAL_STEPS_PER_EPOCH
    return STEPS_PER_EPOCH


def _eval_every(stage: str) -> int:
    if stage == UPDATE_BUDGET_CHECK_STAGE:
        return 2
    return 4


def _member_complete(
    member: Path,
    spec: tuple[str, str, Mapping[str, Any], int, int],
) -> bool:
    epochs = int(spec[-1])
    required = (member / "last.pt", member / "model_meta.json", member / "calibration_ready_summary.json", member / "train_history.json")
    if not all(path.is_file() for path in required):
        return False
    try:
        history = json.loads((member / "train_history.json").read_text(encoding="utf-8"))
        manifest = _load_json(member / "run_manifest.json")
        arguments = manifest.get("arguments")
        if not isinstance(arguments, Mapping):
            return False
        expected = _member_contract(member, spec)
        if any(arguments.get(key) != value for key, value in expected.items()):
            return False
        return (
            isinstance(history, list)
            and len(history) >= epochs
            and max(int(row["epoch"]) for row in history) >= epochs
            and manifest.get("testRead") is False
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _evaluation_checkpoint(member: Path) -> Path:
    summary = _load_json(member / "calibration_ready_summary.json")
    if summary.get("testRead") is not False:
        raise ValueError(f"calibration summary opened test: {member}")
    safe = summary.get("status") == "safe" and summary.get("bestSafe") is not None
    path = member / ("best_safe.pt" if safe else "best_diagnostic.pt")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _calibration_bootstrap_replicates(member: Path) -> int:
    summary = _load_json(member / "calibration_ready_summary.json")
    calibration = summary.get("calibration")
    if not isinstance(calibration, Mapping):
        return 0
    return int(calibration.get("bootstrapReplicates", 0))


def _require_formal_bootstrap_protocol(
    member: Path,
    spec: tuple[str, str, Mapping[str, Any], int, int],
) -> None:
    stage = str(spec[0])
    if stage not in {
        FORMAL_FULL_STAGE,
        FORMAL_ABLATION_STAGE,
        CORE_ABLATION_STAGE,
    }:
        return
    replicates = _calibration_bootstrap_replicates(member)
    if replicates < FORMAL_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"formal member {member.name} used {replicates} calibration bootstrap "
            "replicates; run reaudit_pvs.py "
            "instead of accepting or regenerating a mixed-protocol summary"
        )


def build_evaluate_command(data_root: Path, member: Path, output: Path, *, seed: int) -> list[str]:
    paths = _paths(data_root)
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(_evaluation_checkpoint(member)),
        "--dataset-dir", str(paths["dataset"]),
        "--runtime-meta", str(paths["runtime_meta"]),
        "--initial-geo-features", str(paths["geometry"]),
        "--glb-index", str(paths["glb_index"]),
        "--glb-root", str(paths["glb_root"]),
        "--relation-dir", str(paths["relation"]),
        "--model-meta", str(member / "model_meta.json"),
        "--calibration", str(member / "calibration_ready_summary.json"),
        "--output", str(output),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--seed", str(int(seed)),
        "--device", "cuda",
        "--allow-unsafe-diagnostic",
        "--persist-ids",
    ]


def _run_logged(command: Sequence[str], gpu: int, stdout: Path, stderr: Path) -> dict[str, Any]:
    stdout.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    started = time.time()
    with stdout.open("w", encoding="utf-8") as out_handle, stderr.open("w", encoding="utf-8") as err_handle:
        result = subprocess.run(list(command), cwd=ROOT, env=environment, stdout=out_handle, stderr=err_handle, check=False)
    return {
        "gpu": int(gpu),
        "returnCode": int(result.returncode),
        "elapsedSeconds": float(time.time() - started),
        "stdout": str(stdout),
        "stderr": str(stderr),
        "testRead": False,
    }


def _run_queue(jobs: Sequence[tuple[str, Sequence[str]]], gpu_ids: Sequence[int], log_root: Path) -> list[dict[str, Any]]:
    pending: Queue[tuple[str, Sequence[str]]] = Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        while True:
            try:
                name, command = pending.get_nowait()
            except Empty:
                return rows
            row = {
                "member": name,
                **_run_logged(command, gpu, log_root / f"{name}.stdout.log", log_root / f"{name}.stderr.log"),
            }
            rows.append(row)
            pending.task_done()
            print(json.dumps({key: row[key] for key in ("member", "gpu", "returnCode", "elapsedSeconds")}), flush=True)

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, gpu) for gpu in gpu_ids]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [row for row in results if row["returnCode"] != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} mainline jobs failed; inspect logs")
    return sorted(results, key=lambda row: row["member"])


def _specs(stage: str, variants: Sequence[str], configs: Sequence[Mapping[str, Any]], seeds: Sequence[int], epochs: int) -> list[tuple[str, str, Mapping[str, Any], int, int]]:
    return [(stage, variant, config, seed, epochs) for variant in variants for config in configs for seed in seeds]


def _member_for_spec(model_root: Path, spec: tuple[str, str, Mapping[str, Any], int, int]) -> Path:
    stage, variant, config, seed, epochs = spec
    return _member_dir(model_root, stage, variant, str(config["name"]), seed, epochs)


def _evaluation_path(benchmark_root: Path, member: Path) -> Path:
    return benchmark_root / "members" / member.name / "validation_evaluation.json"


def _evaluation_matches_member(
    output: Path,
    member: Path,
    spec: tuple[str, str, Mapping[str, Any], int, int],
) -> bool:
    try:
        payload = _load_json(output)
        checkpoint = _evaluation_checkpoint(member).resolve()
        calibration = _load_json(member / "calibration_ready_summary.json")
        selected = (
            calibration.get("bestSafe")
            if calibration.get("status") == "safe"
            else calibration.get("bestDiagnostic")
        )
        if not isinstance(selected, Mapping):
            return False
        return bool(
            payload.get("schema") == EVALUATION_SCHEMA
            and payload.get("split") == "validation"
            and payload.get("testRead") is False
            and Path(str(payload.get("checkpoint"))).resolve() == checkpoint
            and Path(str(payload.get("modelMeta"))).resolve()
            == (member / "model_meta.json").resolve()
            and Path(str(payload.get("calibrationSummary"))).resolve()
            == (member / "calibration_ready_summary.json").resolve()
            and int(payload.get("checkpointSeed", -1)) == int(spec[-2])
            and int(payload.get("epoch", -1)) == int(selected["epoch"])
            and abs(float(payload.get("threshold", -1.0)) - float(selected["threshold"]))
            <= 1e-7
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _run_specs(data_root: Path, model_root: Path, benchmark_root: Path, specs: Sequence[tuple[str, str, Mapping[str, Any], int, int]], gpu_ids: Sequence[int]) -> None:
    train_jobs: list[tuple[str, Sequence[str]]] = []
    for spec in specs:
        stage, variant, config, seed, epochs = spec
        member = _member_for_spec(model_root, spec)
        if not _member_complete(member, spec):
            if member.exists():
                shutil.rmtree(member)
            train_jobs.append(
                (
                    member.name,
                    build_train_command(
                        data_root,
                        member,
                        config,
                        variant,
                        seed=seed,
                        epochs=epochs,
                        steps_per_epoch=_steps_per_epoch(stage),
                        eval_every=_eval_every(stage),
                    ),
                )
            )
    if train_jobs:
        _run_queue(train_jobs, gpu_ids, benchmark_root / "logs" / "train")
    eval_jobs: list[tuple[str, Sequence[str]]] = []
    for spec in specs:
        member = _member_for_spec(model_root, spec)
        if not _member_complete(member, spec):
            raise RuntimeError(f"training member is incomplete: {member}")
        _require_formal_bootstrap_protocol(member, spec)
        output = _evaluation_path(benchmark_root, member)
        if not _evaluation_matches_member(output, member, spec):
            if output.exists():
                output.unlink()
            output.parent.mkdir(parents=True, exist_ok=True)
            eval_jobs.append((f"evaluate_{member.name}", build_evaluate_command(data_root, member, output, seed=spec[-2])))
    if eval_jobs:
        _run_queue(eval_jobs, gpu_ids, benchmark_root / "logs" / "evaluate")


def _result_row(model_root: Path, benchmark_root: Path, spec: tuple[str, str, Mapping[str, Any], int, int]) -> dict[str, Any]:
    stage, variant, config, seed, epochs = spec
    member = _member_for_spec(model_root, spec)
    _require_formal_bootstrap_protocol(member, spec)
    calibration = _load_json(member / "calibration_ready_summary.json")
    evaluation = _load_json(_evaluation_path(benchmark_root, member))
    if evaluation.get("split") != "validation" or evaluation.get("testRead") is not False:
        raise ValueError(f"mainline evaluation used the wrong split: {member}")
    metrics = evaluation["aggregate"]
    calibration_safe = calibration.get("status") == "safe" and calibration.get("bestSafe") is not None
    validation_lcb = float(
        metrics.get("aggregateWeightedRecallLowerConfidenceBound") or -1.0
    )
    validation_safe = (
        float(metrics["weightedRecall"]) > WEIGHTED_RECALL_FLOOR
        and validation_lcb > WEIGHTED_RECALL_FLOOR
    )
    keys = (
        "precision", "recall", "weightedRecall", "accuracy", "balancedAccuracy",
        "specificity", "f1", "usefulCull", "badCull", "avgCandidateCount",
        "avgGtCount", "avgPredCount",
    )
    return {
        "member": member.name,
        "stage": stage,
        "variant": variant,
        "config": dict(config),
        "seed": int(seed),
        "epochs": int(epochs),
        "calibrationSafe": bool(calibration_safe),
        "validationSafe": bool(validation_safe),
        "eligibleSafe": bool(calibration_safe and validation_safe),
        "threshold": float(evaluation["threshold"]),
        "validationWeightedRecallLowerConfidenceBound": validation_lcb,
        "aggregate": {key: float(metrics[key]) for key in keys},
        "testRead": False,
    }


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    metrics = row["aggregate"]
    if row["eligibleSafe"]:
        return (
            1.0,
            float(metrics["balancedAccuracy"]),
            float(metrics["precision"]),
            float(metrics["accuracy"]),
            float(metrics["usefulCull"]),
            -float(metrics["avgPredCount"]),
        )
    return (
        0.0,
        float(metrics["weightedRecall"]),
        float(metrics["balancedAccuracy"]),
        float(metrics["precision"]),
        float(metrics["accuracy"]),
    )


def _summarize(model_root: Path, benchmark_root: Path, specs: Sequence[tuple[str, str, Mapping[str, Any], int, int]], output: Path) -> dict[str, Any]:
    rows = [_result_row(model_root, benchmark_root, spec) for spec in specs]
    selected = max(rows, key=_rank)
    payload = {
        "schema": "pvs-v4-integrated-visibility-mainline-summary-v1",
        "selectionSplit": "validation",
        "thresholdSource": "each checkpoint's calibration split",
        "selectionRule": (
            "calibration and validation weighted-recall safe first; then balanced "
            "accuracy, precision, accuracy, useful cull, and lower prediction count"
        ),
        "selected": selected,
        "rows": sorted(rows, key=lambda row: row["member"]),
        "testRead": False,
    }
    _write_json(output, payload)
    return payload


def _validate_scan_summary(
    summary: Mapping[str, Any],
    model_root: Path,
    benchmark_root: Path,
    specs: Sequence[tuple[str, str, Mapping[str, Any], int, int]],
) -> dict[str, Any]:
    if (
        summary.get("schema")
        != "pvs-v4-integrated-visibility-mainline-summary-v1"
        or summary.get("selectionSplit") != "validation"
        or summary.get("testRead") is not False
    ):
        raise ValueError("scan summary does not satisfy the current mainline protocol")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != len(specs):
        raise ValueError("scan validation requires the complete registered scan matrix")
    expected = {
        _member_for_spec(model_root, spec).name: spec for spec in specs
    }
    actual = {str(row.get("member")): row for row in rows if isinstance(row, Mapping)}
    if set(actual) != set(expected):
        raise ValueError("scan summary member set differs from the registered matrix")
    for name, spec in expected.items():
        stage, variant, config, seed, epochs = spec
        row = actual[name]
        if (
            row.get("stage") != stage
            or row.get("variant") != variant
            or row.get("config") != dict(config)
            or int(row.get("seed", -1)) != int(seed)
            or int(row.get("epochs", -1)) != int(epochs)
            or row.get("testRead") is not False
        ):
            raise ValueError(f"scan summary member contract changed: {name}")
        member = _member_for_spec(model_root, spec)
        if not _member_complete(member, spec):
            raise ValueError(f"scan training artifact is incomplete: {name}")
        if not _evaluation_matches_member(
            _evaluation_path(benchmark_root, member), member, spec
        ):
            raise ValueError(f"scan validation artifact is stale: {name}")
    selected = summary.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("scan summary has no selected configuration")
    selected_name = str(selected.get("member"))
    if selected_name not in actual or dict(selected) != dict(actual[selected_name]):
        raise ValueError("selected scan row is not one of the complete matrix rows")
    selected_config = selected.get("config")
    if selected_config not in [dict(config) for config in SCAN_CONFIGS]:
        raise ValueError("selected scan configuration is not registered")
    return dict(selected_config)


def _formal_seed_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in FORMAL_VARIANTS:
        members = [row for row in rows if row["variant"] == variant]
        result[variant] = {
            "memberCount": len(members),
            "safeMemberCount": sum(bool(row["eligibleSafe"]) for row in members),
            "metrics": {
                key: {
                    "mean": float(statistics.fmean(row["aggregate"][key] for row in members)),
                    "populationStd": float(statistics.pstdev(row["aggregate"][key] for row in members)),
                }
                for key in members[0]["aggregate"]
            } if members else {},
        }
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "preflight",
            "smoke",
            "core-smoke",
            "core-ablation",
            "update-budget-check12x300",
            "formal40-s02-full",
            "formal40-s02-ablation",
            "scan",
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/sda/rhyang/slm"))
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    core_mode = args.mode in {"core-smoke", "core-ablation"}
    if args.model_root is None:
        args.model_root = ROOT / "neural_instance_culling/model/out" / (
            "pvs_mainline_core_ablation" if core_mode else OUTPUT_TAG
        )
    if args.benchmark_root is None:
        args.benchmark_root = ROOT / "neural_instance_culling/benchmark/out" / (
            "pvs_mainline_core_ablation" if core_mode else OUTPUT_TAG
        )

    contract = preflight(args.data_root)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.benchmark_root / "preflight.json", contract)
    if args.mode == "preflight":
        print(json.dumps(contract, ensure_ascii=False, indent=2))
        return
    if args.mode == "core-smoke":
        smoke_specs = _specs(
            "core_smoke",
            ("full", "generic28", "no_survival"),
            (CORE_CONFIG,),
            (SCAN_SEED,),
            1,
        )
        commands = [
            build_train_command(
                args.data_root,
                _member_for_spec(args.model_root, spec),
                spec[2],
                spec[1],
                seed=spec[3],
                epochs=1,
                smoke=True,
            )
            for spec in smoke_specs
        ]
        if args.dry_run:
            print(json.dumps({"commands": commands, "testRead": False}, indent=2))
            return
        for spec in smoke_specs:
            member = _member_for_spec(args.model_root, spec)
            if member.exists():
                shutil.rmtree(member)
        _run_queue(
            [
                (_member_for_spec(args.model_root, spec).name, command)
                for spec, command in zip(smoke_specs, commands)
            ],
            args.gpu_ids[: min(3, len(args.gpu_ids))],
            args.benchmark_root / "logs/core_smoke",
        )
        return

    if args.mode == "core-ablation":
        specs = _specs(
            CORE_ABLATION_STAGE,
            tuple(CORE_VARIANTS),
            (CORE_CONFIG,),
            FORMAL_SEEDS,
            FORMAL_EPOCHS,
        )
        if args.dry_run:
            commands = [
                build_train_command(
                    args.data_root,
                    _member_for_spec(args.model_root, spec),
                    spec[2],
                    spec[1],
                    seed=spec[3],
                    epochs=spec[4],
                    steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
                    eval_every=4,
                )
                for spec in specs
            ]
            print(json.dumps({"commands": commands, "testRead": False}, indent=2))
            return
        _run_specs(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            specs,
            args.gpu_ids,
        )
        summary = _summarize(
            args.model_root,
            args.benchmark_root,
            specs,
            args.benchmark_root / "core_ablation_summary.json",
        )
        summary["fullReference"] = "existing three-seed no-contrastive formal members"
        summary["epochs"] = FORMAL_EPOCHS
        summary["seeds"] = list(FORMAL_SEEDS)
        _write_json(args.benchmark_root / "core_ablation_summary.json", summary)
        return
    if args.mode == "smoke":
        config = SCAN_CONFIGS[2]
        member = _member_dir(args.model_root / "smoke", "smoke", "full", config["name"], SCAN_SEED, 1)
        command = build_train_command(args.data_root, member, config, "full", seed=SCAN_SEED, epochs=1, smoke=True)
        if args.dry_run:
            print(json.dumps({"command": command, "testRead": False}, indent=2))
            return
        if member.exists():
            shutil.rmtree(member)
        _run_queue([(member.name, command)], args.gpu_ids[:1], args.benchmark_root / "logs/smoke")
        return

    if args.mode == "update-budget-check12x300":
        specs = _specs(
            UPDATE_BUDGET_CHECK_STAGE,
            ("full",),
            UPDATE_BUDGET_CHECK_CONFIGS,
            (SCAN_SEED,),
            UPDATE_BUDGET_CHECK_EPOCHS,
        )
        if args.dry_run:
            commands = [
                build_train_command(
                    args.data_root,
                    _member_for_spec(args.model_root, spec),
                    spec[2],
                    spec[1],
                    seed=spec[3],
                    epochs=spec[4],
                    steps_per_epoch=UPDATE_BUDGET_CHECK_STEPS_PER_EPOCH,
                    eval_every=2,
                )
                for spec in specs
            ]
            print(json.dumps({"commands": commands, "testRead": False}, indent=2))
            return
        _run_specs(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            specs,
            args.gpu_ids,
        )
        _summarize(
            args.model_root,
            args.benchmark_root,
            specs,
            args.benchmark_root / "update_budget_check_12x300_summary.json",
        )
        return

    if args.mode == "formal40-s02-full":
        specs = _specs(
            FORMAL_FULL_STAGE,
            ("full",),
            (SCAN_CONFIGS[2],),
            FORMAL_SEEDS,
            FORMAL_EPOCHS,
        )
        if args.dry_run:
            commands = [
                build_train_command(
                    args.data_root,
                    _member_for_spec(args.model_root, spec),
                    spec[2],
                    spec[1],
                    seed=spec[3],
                    epochs=spec[4],
                    steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
                    eval_every=4,
                )
                for spec in specs
            ]
            print(json.dumps({"commands": commands, "testRead": False}, indent=2))
            return
        _run_specs(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            specs,
            args.gpu_ids,
        )
        summary = _summarize(
            args.model_root,
            args.benchmark_root,
            specs,
            args.benchmark_root / "formal40_s02_full_summary.json",
        )
        summary["epochs"] = FORMAL_EPOCHS
        summary["seeds"] = list(FORMAL_SEEDS)
        summary["seedAggregate"] = _formal_seed_summary(summary["rows"])
        _write_json(
            args.benchmark_root / "formal40_s02_full_summary.json",
            summary,
        )
        return

    if args.mode == "formal40-s02-ablation":
        specs = _specs(
            FORMAL_ABLATION_STAGE,
            FORMAL_ABLATIONS,
            (SCAN_CONFIGS[2],),
            FORMAL_SEEDS,
            FORMAL_EPOCHS,
        )
        if args.dry_run:
            commands = [
                build_train_command(
                    args.data_root,
                    _member_for_spec(args.model_root, spec),
                    spec[2],
                    spec[1],
                    seed=spec[3],
                    epochs=spec[4],
                    steps_per_epoch=FORMAL_STEPS_PER_EPOCH,
                    eval_every=4,
                )
                for spec in specs
            ]
            print(json.dumps({"commands": commands, "testRead": False}, indent=2))
            return
        _run_specs(
            args.data_root,
            args.model_root,
            args.benchmark_root,
            specs,
            args.gpu_ids,
        )
        summary = _summarize(
            args.model_root,
            args.benchmark_root,
            specs,
            args.benchmark_root / "formal40_s02_ablation_summary.json",
        )
        summary["selectedScanConfiguration"] = dict(SCAN_CONFIGS[2])
        summary["epochs"] = FORMAL_EPOCHS
        summary["seeds"] = list(FORMAL_SEEDS)
        summary["seedAggregate"] = _formal_seed_summary(summary["rows"])
        _write_json(
            args.benchmark_root / "formal40_s02_ablation_summary.json",
            summary,
        )
        return

    scan_specs = _specs("scan", ("full",), SCAN_CONFIGS, (SCAN_SEED,), SCAN_EPOCHS)
    if args.dry_run:
        commands = [
            build_train_command(
                args.data_root,
                _member_for_spec(args.model_root, spec),
                spec[2],
                spec[1],
                seed=spec[3],
                epochs=spec[4],
            )
            for spec in scan_specs
        ]
        print(json.dumps({"commands": commands, "testRead": False}, indent=2))
        return
    _run_specs(
        args.data_root,
        args.model_root,
        args.benchmark_root,
        scan_specs,
        args.gpu_ids,
    )
    _summarize(
        args.model_root,
        args.benchmark_root,
        scan_specs,
        args.benchmark_root / "selected_configuration.json",
    )


if __name__ == "__main__":
    main()
