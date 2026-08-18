#!/usr/bin/env python3
"""Run the three-seed long-training and same-checkpoint posterior ablation.

The neural base is trained once per seed.  Afterwards, no posterior, a fixed
linear dual probe, and the selected hinge-plus-small-MLP dual probe are replayed
on that same checkpoint.  Probe parameters and certificates are train-owned;
calibration freezes the operating threshold; validation is replay-only.  This
entry point has no test-split argument.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py"
FIT_LINEAR = ROOT / "neural_instance_culling/benchmark/analyze_pvs_difficult_tail_feature_separability.py"
FIT_NONLINEAR = ROOT / "neural_instance_culling/benchmark/fit_pvs_train_owned_tail_nonlinear_probe.py"
SCAN = ROOT / "neural_instance_culling/benchmark/scan_pvs_train_owned_tail_residual.py"

EXPERIMENT = "pvs_train_owned_nonlinear_tail_posterior_longtrain_v1"
FORMAL_SEEDS = (20260801, 20260802, 20260803)
FORMAL_EPOCHS = 80
STEPS_PER_EPOCH = 100
FIXED_OPERATING_THRESHOLD_SEED = 20260818


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _required_file(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_file():
        raise FileNotFoundError(f"missing {label}: {result}")
    return result


def _required_dir(path: Path, label: str) -> Path:
    result = path.resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"missing {label}: {result}")
    return result


def _shared_paths(shared_root: Path) -> dict[str, Path]:
    root = shared_root.resolve()
    return {
        "dataset": root
        / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3",
        "relation": root
        / "neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3",
        "runtime_meta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root
        / "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin",
        "glb_index": root / "hkust-v3/assets/glbIndex.json",
        "glb_root": root / "hkust-v3/assets",
        "initial_root": root
        / "neural_instance_culling/model/out/pvs_bounded_relation_prior_instance_calibrated_moment_safety_reserve_v4_formal80",
    }


def _initial_checkpoint(paths: Mapping[str, Path], seed: int) -> Path:
    return paths["initial_root"] / f"full_seed{int(seed)}_e80/checkpoint_epoch_024.pt"


def _member_dir(model_output_root: Path, seed: int) -> Path:
    return model_output_root.resolve() / f"base_seed{int(seed)}_e80"


def _member_complete(member: Path) -> bool:
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
        return (
            isinstance(history, list)
            and bool(history)
            and max(int(row["epoch"]) for row in history if isinstance(row, Mapping))
            >= FORMAL_EPOCHS
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def preflight(shared_root: Path) -> dict[str, Any]:
    paths = _shared_paths(shared_root)
    _required_dir(paths["dataset"], "pose CSR dataset")
    _required_dir(paths["relation"], "bounded relation CSR")
    _required_file(paths["runtime_meta"], "runtime visibility metadata")
    _required_file(paths["geometry"], "96D geometry table")
    _required_file(paths["glb_index"], "GLB index")
    _required_dir(paths["glb_root"], "GLB asset root")
    _required_file(TRAIN, "training entry")
    _required_file(EVALUATE, "evaluation entry")
    _required_file(FIT_LINEAR, "linear probe entry")
    _required_file(FIT_NONLINEAR, "nonlinear probe entry")
    _required_file(SCAN, "posterior scan entry")
    sidecar = _required_dir(
        ROOT
        / "neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar",
        "train-only subpose supervision sidecar",
    )
    initial: dict[str, Any] = {}
    for seed in FORMAL_SEEDS:
        checkpoint = _required_file(_initial_checkpoint(paths, seed), f"seed {seed} epoch-24 checkpoint")
        manifest_path = _required_file(checkpoint.parent / "run_manifest.json", "source run manifest")
        manifest = _load_json(manifest_path)
        arguments = manifest.get("arguments")
        if not isinstance(arguments, Mapping) or int(arguments.get("seed", -1)) != seed:
            raise ValueError(f"source checkpoint seed mismatch: {checkpoint}")
        initial[str(seed)] = {
            "checkpoint": str(checkpoint),
            "sourceEpoch": 24,
            "sourceSeed": int(seed),
        }
    return {
        "schema": "pvs-train-owned-nonlinear-tail-posterior-longtrain-preflight-v1",
        "experiment": EXPERIMENT,
        "seeds": list(FORMAL_SEEDS),
        "epochs": FORMAL_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "subposeSidecar": str(sidecar),
        "initialCheckpoints": initial,
        "candidateSetChanged": False,
        "groundTruthChanged": False,
        "testRead": False,
    }


def build_train_command(
    shared_root: Path,
    model_output_root: Path,
    seed: int,
) -> list[str]:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    paths = _shared_paths(shared_root)
    member = _member_dir(model_output_root, seed)
    sidecar = (
        ROOT
        / "neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar"
    )
    return [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--relation-dir", str(paths["relation"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--initial-checkpoint", str(_initial_checkpoint(paths, seed).resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--subpose-sidecar", str(sidecar.resolve()),
        "--output-dir", str(member),
        "--experiment-name", f"{EXPERIMENT}_seed{seed}_e80",
        "--variant", "bounded_tail_residual_nonlinear_posterior_base",
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_extrema",
        "--instance-calibration-mode", "residual",
        "--loss-variant", "safety_reserve",
        "--refinement-scope", "boundary_tail_residual",
        "--epochs", str(FORMAL_EPOCHS),
        "--steps-per-epoch", str(STEPS_PER_EPOCH),
        "--poses-per-batch", "4",
        "--observation-batch-size", "8192",
        "--eval-every", "4",
        "--snapshot-every", "4",
        "--max-eval-poses", "0",
        "--calibration-bootstrap-replicates", "10000",
        "--seed", str(seed),
        "--device", "cuda",
        "--learning-rate", "0.0005",
        "--weight-decay", "0.00001",
        "--survival-loss-weight", "0.25",
        "--relation-consistency-weight", "0.1",
        "--utility-loss-weight", "0.1",
        "--download-loss-weight", "0.1",
        "--boundary-tail-weight", "0.3",
        "--negative-band-weight", "0.03",
        "--negative-band-shape", "sigmoid",
        "--negative-band-temperature", "0.1",
        "--glb-resource-weight", "0.015",
        "--relation-gradient-cap", "0.25",
        "--schedule-gradient-cap", "0.25",
        "--efficiency-gradient-cap", "0.25",
        "--efficiency-primary-fraction", "0.0",
        "--efficiency-warmup-fraction", "0.1",
        "--instance-calibration-regularization-weight", "0.02",
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.1",
        "--instance-calibration-ramp-fraction", "0.2",
        "--boundary-tail-residual-hidden-dim", "32",
        "--boundary-tail-residual-projection-dim", "24",
        "--boundary-tail-residual-max-abs", "0.5",
        "--boundary-tail-residual-centering", "pose_mean",
        "--boundary-tail-residual-shortcut", "none",
        "--boundary-tail-residual-fusion", "product",
        "--boundary-tail-residual-output-init-std", "0.0",
        "--boundary-tail-residual-loss-weight", "1.0",
        "--boundary-tail-residual-rare-threshold", "0.05",
        "--boundary-tail-residual-positive-tail-mass-fraction", "0.005",
        "--boundary-tail-residual-positive-tail-count-cap", "64",
        "--boundary-tail-residual-min-positives", "2",
        "--boundary-tail-residual-negative-tail-fraction", "0.04",
        "--boundary-tail-residual-min-negatives", "64",
        "--boundary-tail-residual-max-negatives", "256",
        "--boundary-tail-residual-margin", "0.0",
        "--boundary-tail-residual-temperature", "0.25",
        "--boundary-tail-residual-pose-cvar-fraction", "0.25",
        "--boundary-tail-residual-pose-cvar-weight", "0.5",
        "--boundary-tail-residual-negative-positive-weight", "0.25",
        "--boundary-tail-residual-positive-negative-weight", "0.25",
        "--boundary-tail-residual-all-positive-negative-weight", "0.0",
        "--boundary-tail-residual-classification-weight", "0.0",
        "--boundary-tail-residual-classification-margin", "0.2",
        "--boundary-tail-residual-regularization-weight", "0.02",
        "--boundary-tail-residual-positive-importance-transform", "power",
        "--boundary-tail-residual-positive-importance-power", "0.5",
        "--boundary-tail-residual-cross-pose-pair-weight", "0.4",
        "--boundary-tail-residual-global-tail-pair-weight", "0.25",
        "--boundary-tail-residual-objective", "partial_auc",
        "--boundary-tail-residual-selection-source", "frozen_base",
        "--rvl-bce-positive-weight", "14.0",
        "--rvl-count-weight", "0.1",
        "--rvl-rank-weight", "0.45",
        "--rvl-tversky-fn-weight", "7.0",
        "--rvl-rank-negative-top-k", "256",
        "--rvl-fp-normalization", "positive",
        "--operating-threshold-seed", str(FIXED_OPERATING_THRESHOLD_SEED),
    ]


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
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
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
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "testRead": False,
    }


def run_training(
    shared_root: Path,
    model_output_root: Path,
    train_gpus: Sequence[int],
) -> list[dict[str, Any]]:
    if len(train_gpus) != len(FORMAL_SEEDS) or len(set(train_gpus)) != len(train_gpus):
        raise ValueError("training requires exactly three distinct GPU ids")
    model_output_root.mkdir(parents=True, exist_ok=True)
    logs = model_output_root / "logs"

    def one(seed: int, gpu: int) -> dict[str, Any]:
        member = _member_dir(model_output_root, seed)
        if _member_complete(member):
            return {
                "seed": seed,
                "gpu": gpu,
                "outputDir": str(member),
                "status": "reused_complete",
                "returnCode": 0,
                "testRead": False,
            }
        if member.exists() and any(member.iterdir()):
            raise FileExistsError(f"refusing to overwrite incomplete member: {member}")
        member.mkdir(parents=True, exist_ok=True)
        result = _run_logged(
            build_train_command(shared_root, model_output_root, seed),
            gpu=gpu,
            stdout_path=logs / f"base_seed{seed}.stdout.log",
            stderr_path=logs / f"base_seed{seed}.stderr.log",
        )
        result.update({"seed": seed, "outputDir": str(member), "status": "completed"})
        if result["returnCode"] == 0 and not _member_complete(member):
            result["returnCode"] = 86
            result["status"] = "incomplete_artifacts"
        return result

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(one, seed, gpu): (seed, gpu)
            for seed, gpu in zip(FORMAL_SEEDS, train_gpus)
        }
        for future in as_completed(futures):
            seed, gpu = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                results.append(
                    {
                        "seed": seed,
                        "gpu": gpu,
                        "status": "exception",
                        "returnCode": 1,
                        "error": repr(exc),
                        "testRead": False,
                    }
                )
    results.sort(key=lambda row: int(row["seed"]))
    _write_json_once(
        model_output_root / "training_summary.json",
        {
            "schema": "pvs-train-owned-nonlinear-tail-posterior-training-summary-v1",
            "experiment": EXPERIMENT,
            "members": results,
            "testRead": False,
        },
    )
    failed = [row for row in results if int(row.get("returnCode", 1)) != 0]
    if failed:
        raise RuntimeError(f"long-training member failure: {failed}")
    return results


def _selected_checkpoint(member: Path) -> tuple[Path, bool]:
    calibration = _load_json(member / "calibration_ready_summary.json")
    if calibration.get("testRead") is not False:
        raise ValueError(f"calibration summary is not test-free: {member}")
    if calibration.get("status") == "safe" and (member / "best_safe.pt").is_file():
        return member / "best_safe.pt", True
    if (member / "best_diagnostic.pt").is_file():
        return member / "best_diagnostic.pt", False
    return _required_file(member / "last.pt", "long-training checkpoint"), False


def _capture_command(
    paths: Mapping[str, Path],
    member: Path,
    checkpoint: Path,
    split: str,
    output: Path,
    seed: int,
    safe: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--output", str(output.resolve()),
        "--split", split,
        "--poses-per-batch", "2",
        "--seed", str(seed),
        "--device", "cuda",
        "--persist-scores",
        "--relation-dir", str(paths["relation"].resolve()),
        "--model-meta", str((member / "model_meta.json").resolve()),
        "--calibration", str((member / "calibration_ready_summary.json").resolve()),
    ]
    if not safe:
        command.append("--allow-unsafe-diagnostic")
    return command


def _probe_common(
    paths: Mapping[str, Path],
    captures: Mapping[str, Path],
    checkpoint: Path,
    output: Path,
) -> list[str]:
    return [
        "--train-capture", str(captures["train"].resolve()),
        "--calibration-capture", str(captures["calibration"].resolve()),
        "--validation-capture", str(captures["validation"].resolve()),
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--output", str(output.resolve()),
        "--positive-tail-mode", "weighted_mass",
        "--positive-tail-quantile", "0.01",
        "--negative-tail-quantile", "0.99",
        "--seed", str(FIXED_OPERATING_THRESHOLD_SEED),
        "--device", "cuda",
        "--batch-size", "8192",
    ]


def _scan_command(
    paths: Mapping[str, Path],
    captures: Mapping[str, Path],
    checkpoint: Path,
    primary_probe: Path,
    coverage_probe: Path,
    output: Path,
    *,
    include_zero: bool,
) -> list[str]:
    return [
        sys.executable,
        str(SCAN),
        "--train-capture", str(captures["train"].resolve()),
        "--calibration-capture", str(captures["calibration"].resolve()),
        "--validation-capture", str(captures["validation"].resolve()),
        "--probe", str(primary_probe.resolve()),
        "--probe-family", "combined",
        "--coverage-probe", str(coverage_probe.resolve()),
        "--coverage-probe-family", "combined",
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--output", str(output.resolve()),
        "--device", "cuda",
        "--batch-size", "8192",
        "--alphas", "0,0.5" if include_zero else "0.5",
        "--temperatures", "0.05",
        "--gate-temperatures", "none",
        "--residual-modes", "dual_rescue",
        "--rescue-risk", "0.05",
        "--coverage-rescue-risk", "0.05",
        "--coverage-rescue-weight", "1.0",
        "--bootstrap-replicates", "10000",
        "--seed", str(FIXED_OPERATING_THRESHOLD_SEED),
    ]


def _run_stage_once(command: Sequence[str], output: Path, logs: Path, name: str, gpu: int) -> None:
    if output.is_file():
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    result = _run_logged(
        command,
        gpu=gpu,
        stdout_path=logs / f"{name}.stdout.log",
        stderr_path=logs / f"{name}.stderr.log",
    )
    if result["returnCode"] != 0 or not output.is_file():
        raise RuntimeError(f"stage failed or output missing: {name}; {result}")


def run_postprocess(
    shared_root: Path,
    model_output_root: Path,
    benchmark_output_root: Path,
    gpu: int,
) -> list[dict[str, Any]]:
    paths = _shared_paths(shared_root)
    results: list[dict[str, Any]] = []
    for seed in FORMAL_SEEDS:
        member = _member_dir(model_output_root, seed)
        if not _member_complete(member):
            raise RuntimeError(f"long-training member is incomplete: {member}")
        checkpoint, safe = _selected_checkpoint(member)
        seed_root = benchmark_output_root.resolve() / f"seed{seed}"
        logs = seed_root / "logs"
        captures = {
            split: seed_root / "captures" / f"{split}_scores.json"
            for split in ("train", "calibration", "validation")
        }
        for split, output in captures.items():
            _run_stage_once(
                _capture_command(paths, member, checkpoint, split, output, seed, safe),
                output,
                logs,
                f"capture_{split}",
                gpu,
            )

        probes = {
            "linear_primary": seed_root / "probes/linear_primary_p05.json",
            "linear_coverage": seed_root / "probes/linear_coverage_p0.json",
            "hinge_primary": seed_root / "probes/hinge_primary_p05.json",
            "mlp_coverage": seed_root / "probes/mlp8_coverage_p0.json",
        }
        for role, power in (("linear_primary", "0.5"), ("linear_coverage", "0.0")):
            command = [sys.executable, str(FIT_LINEAR)] + _probe_common(
                paths, captures, checkpoint, probes[role]
            )
            command.extend(
                [
                    "--probe-fit-split", "train",
                    "--probe-fit-tail-source", "train",
                    "--ridge", "0.001",
                    "--max-probe-samples", "-1",
                    "--probe-sample-weight-power", power,
                ]
            )
            _run_stage_once(command, probes[role], logs, f"fit_{role}", gpu)

        hinge_command = [sys.executable, str(FIT_NONLINEAR)] + _probe_common(
            paths, captures, checkpoint, probes["hinge_primary"]
        )
        hinge_command.extend(
            [
                "--probe-type", "hinge",
                "--sample-weight-power", "0.5",
                "--max-probe-samples", "30000",
                "--ridge", "0.001",
                "--hinge-knots=-1,0,1",
            ]
        )
        _run_stage_once(
            hinge_command, probes["hinge_primary"], logs, "fit_hinge_primary", gpu
        )

        mlp_command = [sys.executable, str(FIT_NONLINEAR)] + _probe_common(
            paths, captures, checkpoint, probes["mlp_coverage"]
        )
        mlp_command.extend(
            [
                "--probe-type", "mlp",
                "--sample-weight-power", "0.0",
                "--max-probe-samples", "30000",
                "--ridge", "0.0001",
                "--hidden-dim", "8",
                "--epochs", "40",
                "--learning-rate", "0.001",
            ]
        )
        _run_stage_once(
            mlp_command, probes["mlp_coverage"], logs, "fit_mlp_coverage", gpu
        )

        linear_scan = seed_root / "linear_dual_probe_formal.json"
        nonlinear_scan = seed_root / "nonlinear_dual_probe_formal.json"
        _run_stage_once(
            _scan_command(
                paths,
                captures,
                checkpoint,
                probes["linear_primary"],
                probes["linear_coverage"],
                linear_scan,
                include_zero=False,
            ),
            linear_scan,
            logs,
            "scan_linear_dual_probe",
            gpu,
        )
        _run_stage_once(
            _scan_command(
                paths,
                captures,
                checkpoint,
                probes["hinge_primary"],
                probes["mlp_coverage"],
                nonlinear_scan,
                include_zero=True,
            ),
            nonlinear_scan,
            logs,
            "scan_nonlinear_dual_probe",
            gpu,
        )
        results.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint.resolve()),
                "baseCheckpointSafetyStatus": "safe" if safe else "diagnostic_only",
                "captures": {key: str(value.resolve()) for key, value in captures.items()},
                "linearScan": str(linear_scan.resolve()),
                "nonlinearScan": str(nonlinear_scan.resolve()),
                "testRead": False,
            }
        )
    _write_json_once(
        benchmark_output_root / "postprocess_summary.json",
        {
            "schema": "pvs-train-owned-nonlinear-tail-posterior-postprocess-summary-v1",
            "experiment": EXPERIMENT,
            "members": results,
            "candidateSetChanged": False,
            "groundTruthChanged": False,
            "testRead": False,
        },
    )
    return results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train", "postprocess", "all"), default="all")
    parser.add_argument("--shared-root", type=Path, default=ROOT)
    parser.add_argument(
        "--model-output-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / f"{EXPERIMENT}_20260819",
    )
    parser.add_argument(
        "--benchmark-output-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / f"{EXPERIMENT}_20260819",
    )
    parser.add_argument("--gpus", type=int, nargs=4, default=(0, 1, 2, 3))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if len(set(args.gpus)) != 4:
        raise ValueError("--gpus must contain four distinct GPU ids")
    preflight_payload = preflight(args.shared_root)
    _write_json_once(args.model_output_root.resolve() / "preflight.json", preflight_payload)
    if args.phase in {"train", "all"}:
        run_training(args.shared_root, args.model_output_root, args.gpus[:3])
    if args.phase in {"postprocess", "all"}:
        run_postprocess(
            args.shared_root,
            args.model_output_root,
            args.benchmark_output_root,
            args.gpus[3],
        )
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT,
                "phase": args.phase,
                "modelOutputRoot": str(args.model_output_root.resolve()),
                "benchmarkOutputRoot": str(args.benchmark_output_root.resolve()),
                "testRead": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
