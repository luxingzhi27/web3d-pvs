#!/usr/bin/env python3
"""Run the from-scratch three-seed 108D query-tail separator ablation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "neural_instance_culling/model/train_bounded_relation_survival_moment_safety.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_bounded_relation_survival_moment_v4.py"
SUMMARIZE = ROOT / "neural_instance_culling/benchmark/summarize_pvs_joint_108d_query_tail_separator_from_scratch_v1.py"
EXPERIMENT = "pvs_joint_108d_query_tail_separator_from_scratch_v1"
OUTPUT_TAG = f"{EXPERIMENT}_20260819"
FORMAL_SEEDS = (20260801, 20260802, 20260803)
FORMAL_EPOCHS = 80
STEPS_PER_EPOCH = 100
VARIANT_FAMILIES: tuple[tuple[str, str], ...] = (
    ("without_query_tail_separator", "disabled"),
    ("linear_query_tail_separator", "linear"),
    ("hinge_query_tail_separator", "hinge"),
    ("mlp8_query_tail_separator", "mlp"),
)


def _paths(root: Path) -> dict[str, Path]:
    shared_subpose = root / "neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar"
    local_subpose = ROOT / "neural_instance_culling/dataset/out/pvs_v4_viewcell_extreme_support_scan_20260818/subpose_supervision_sidecar"
    return {
        "dataset": root / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_bounded_relation_moment_fov66_v3",
        "relation": root / "neural_instance_culling/dataset/out/pvs_bounded_relation_survival_moment_envelope_safety_reserve_v3/bounded_relation_csr_v3",
        "runtime_meta": root / "hkust-v3/assets/runtimeVisibilityMeta.json",
        "geometry": root / "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_w042_full40/instance_geo_features_fp16.bin",
        "glb_index": root / "hkust-v3/assets/glbIndex.json",
        "glb_root": root / "hkust-v3/assets",
        "subpose": shared_subpose if shared_subpose.is_dir() else local_subpose,
    }


def _member_name(variant: str, seed: int, epochs: int = FORMAL_EPOCHS) -> str:
    return f"{variant}_seed{int(seed)}_e{int(epochs)}"


def _member_dir(model_root: Path, variant: str, seed: int, epochs: int = FORMAL_EPOCHS) -> Path:
    return model_root.resolve() / _member_name(variant, seed, epochs)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def preflight(root: Path) -> dict[str, Any]:
    paths = _paths(root.resolve())
    missing = [str(path) for path in paths.values() if not path.exists()]
    for entry in (TRAIN, EVALUATE, SUMMARIZE):
        if not entry.is_file():
            missing.append(str(entry))
    if missing:
        raise FileNotFoundError(f"missing registered inputs: {missing}")
    return {
        "schema": "pvs-joint-108d-query-tail-separator-from-scratch-preflight-v1",
        "experiment": EXPERIMENT,
        "variants": [variant for variant, _ in VARIANT_FAMILIES],
        "families": {variant: family for variant, family in VARIANT_FAMILIES},
        "seeds": list(FORMAL_SEEDS),
        "epochs": FORMAL_EPOCHS,
        "stepsPerEpoch": STEPS_PER_EPOCH,
        "initialCheckpoint": None,
        "refinementScope": "all",
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "testRead": False,
    }


def build_train_command(
    root: Path,
    model_root: Path,
    variant: str,
    seed: int,
    *,
    smoke: bool = False,
) -> list[str]:
    families = dict(VARIANT_FAMILIES)
    if variant not in families:
        raise ValueError(f"unregistered variant: {variant}")
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"unregistered seed: {seed}")
    family = families[variant]
    paths = _paths(root.resolve())
    epochs = 1 if smoke else FORMAL_EPOCHS
    member = _member_dir(model_root, variant, seed, epochs)
    command = [
        sys.executable,
        str(TRAIN),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--relation-dir", str(paths["relation"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--subpose-sidecar", str(paths["subpose"].resolve()),
        "--output-dir", str(member),
        "--experiment-name", f"{EXPERIMENT}_{variant}_seed{seed}_e{epochs}",
        "--variant", variant,
        "--relation-source", "bounded_hierarchical",
        "--spectral-mode", "moment_envelope",
        "--instance-calibration-mode", "residual",
        "--loss-variant", "safety_reserve",
        "--refinement-scope", "all",
        "--epochs", str(epochs),
        "--steps-per-epoch", "1" if smoke else str(STEPS_PER_EPOCH),
        "--poses-per-batch", "4",
        "--observation-batch-size", "32768" if smoke else "8192",
        "--eval-every", "1" if smoke else "4",
        "--snapshot-every", "1" if smoke else "4",
        "--max-eval-poses", "2" if smoke else "0",
        "--calibration-bootstrap-replicates", "2" if smoke else "10000",
        "--seed", str(seed),
        "--device", "cuda",
        "--learning-rate", "0.0002",
        "--weight-decay", "0.00001",
        "--survival-loss-weight", "0.25",
        "--relation-consistency-weight", "0.10",
        "--utility-loss-weight", "0.10",
        "--download-loss-weight", "0.10",
        "--boundary-tail-weight", "0.30",
        "--negative-band-weight", "0.03",
        "--negative-band-shape", "sigmoid",
        "--negative-band-temperature", "0.10",
        "--glb-resource-weight", "0.015",
        "--relation-gradient-cap", "0.25",
        "--schedule-gradient-cap", "0.25",
        "--efficiency-gradient-cap", "0.25",
        "--instance-calibration-regularization-weight", "0.02",
        "--instance-calibration-max-abs", "4.0",
        "--sparse-instance-penalty", "3.0",
        "--instance-calibration-warmup-fraction", "0.10",
        "--instance-calibration-ramp-fraction", "0.20",
        "--rvl-bce-positive-weight", "14.0",
        "--rvl-count-weight", "0.10",
        "--rvl-rank-weight", "0.45",
        "--rvl-tversky-fn-weight", "7.0",
        "--rvl-rank-negative-top-k", "256",
        "--rvl-fp-normalization", "positive",
        "--query-tail-separator-family", family,
        "--query-tail-separator-hidden-dim", "8",
        "--query-tail-separator-max-abs", "0.5",
        "--query-tail-separator-centering", "pose_mean",
        "--query-tail-separation-loss-weight", "0.0" if family == "disabled" else "0.30",
        "--query-tail-positive-mass-fraction", "0.005",
        "--query-tail-positive-count-cap", "64",
        "--query-tail-min-positives", "2",
        "--query-tail-negative-fraction", "0.04",
        "--query-tail-min-negatives", "64",
        "--query-tail-max-negatives", "256",
        "--query-tail-margin", "0.0",
        "--query-tail-temperature", "0.25",
        "--query-tail-pose-cvar-fraction", "0.25",
        "--query-tail-pose-cvar-weight", "0.5",
        "--query-tail-negative-positive-weight", "0.25",
        "--query-tail-positive-negative-weight", "0.25",
        "--query-tail-regularization-weight", "0.02",
        "--query-tail-positive-importance-power", "0.5",
        "--query-tail-cross-pose-pair-weight", "0.4",
        "--query-tail-global-pair-weight", "0.25",
    ]
    if "--initial-checkpoint" in command:
        raise RuntimeError("from-scratch command unexpectedly contains an initial checkpoint")
    return command


def _member_complete(member: Path, epochs: int = FORMAL_EPOCHS) -> bool:
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
            and max(int(row["epoch"]) for row in history if isinstance(row, Mapping)) >= int(epochs)
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False


def _selected_checkpoint(member: Path) -> tuple[Path, bool]:
    summary = _load_json(member / "calibration_ready_summary.json")
    if summary.get("testRead") is not False:
        raise ValueError(f"invalid calibration summary: {member}")
    if summary.get("status") == "safe":
        checkpoint = member / "best_safe.pt"
        safe = True
    else:
        checkpoint = member / "best_diagnostic.pt"
        safe = False
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing selected checkpoint: {checkpoint}")
    return checkpoint, safe


def build_evaluate_command(
    root: Path,
    model_root: Path,
    benchmark_root: Path,
    variant: str,
    seed: int,
) -> list[str]:
    paths = _paths(root.resolve())
    member = _member_dir(model_root, variant, seed)
    checkpoint, _ = _selected_checkpoint(member)
    output = benchmark_root.resolve() / "members" / _member_name(variant, seed) / "validation_evaluation.json"
    return [
        sys.executable,
        str(EVALUATE),
        "--checkpoint", str(checkpoint.resolve()),
        "--dataset-dir", str(paths["dataset"].resolve()),
        "--runtime-meta", str(paths["runtime_meta"].resolve()),
        "--initial-geo-features", str(paths["geometry"].resolve()),
        "--glb-index", str(paths["glb_index"].resolve()),
        "--glb-root", str(paths["glb_root"].resolve()),
        "--relation-dir", str(paths["relation"].resolve()),
        "--model-meta", str((member / "model_meta.json").resolve()),
        "--calibration", str((member / "calibration_ready_summary.json").resolve()),
        "--output", str(output),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--seed", str(seed),
        "--device", "cuda",
        "--allow-unsafe-diagnostic",
        "--persist-ids",
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
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "testRead": False,
    }


def _jobs() -> list[tuple[str, str, int]]:
    return [
        (_member_name(variant, seed), variant, seed)
        for variant, _ in VARIANT_FAMILIES
        for seed in FORMAL_SEEDS
    ]


def _run_queue(
    jobs: Sequence[tuple[str, Sequence[str]]],
    *,
    gpu_ids: Sequence[int],
    log_root: Path,
) -> list[dict[str, Any]]:
    if not gpu_ids:
        raise ValueError("at least one GPU is required")
    results: list[dict[str, Any]] = []
    pending: Queue[tuple[str, Sequence[str]]] = Queue()
    for job in jobs:
        pending.put(job)

    def worker(gpu: int) -> list[dict[str, Any]]:
        owned: list[dict[str, Any]] = []
        while True:
            try:
                name, command = pending.get_nowait()
            except Empty:
                return owned
            result = {
                "member": name,
                **_run_logged(
                    command,
                    gpu=int(gpu),
                    stdout_path=log_root / f"{name}.stdout.log",
                    stderr_path=log_root / f"{name}.stderr.log",
                ),
            }
            owned.append(result)
            print(json.dumps({key: result[key] for key in ("member", "gpu", "returnCode", "elapsedSeconds")}), flush=True)
            pending.task_done()

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(worker, int(gpu)) for gpu in gpu_ids]
        for future in as_completed(futures):
            results.extend(future.result())
    failures = [result for result in results if int(result["returnCode"]) != 0]
    if failures:
        raise RuntimeError(f"{len(failures)} experiment members failed; inspect logs")
    return sorted(results, key=lambda row: str(row["member"]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "formal80", "evaluate", "all"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=ROOT / "neural_instance_culling/model/out" / OUTPUT_TAG,
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=ROOT / "neural_instance_culling/benchmark/out" / OUTPUT_TAG,
    )
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    contract = preflight(args.root)
    args.benchmark_root.mkdir(parents=True, exist_ok=True)
    _write_json(args.benchmark_root / "preflight.json", contract)

    if args.mode == "smoke":
        selected = [
            (
                f"smoke_{variant}",
                build_train_command(
                    args.root,
                    args.model_root / "smoke",
                    variant,
                    FORMAL_SEEDS[0],
                    smoke=True,
                ),
            )
            for variant, _ in VARIANT_FAMILIES
        ]
        log_root = args.benchmark_root / "logs/smoke"
    else:
        selected = []
        log_root = args.benchmark_root / "logs/formal80"
        if args.mode in {"formal80", "all"}:
            for name, variant, seed in _jobs():
                member = _member_dir(args.model_root, variant, seed)
                if _member_complete(member):
                    continue
                selected.append(
                    (name, build_train_command(args.root, args.model_root, variant, seed))
                )

    if args.dry_run:
        print(json.dumps({"mode": args.mode, "jobCount": len(selected), "commands": [list(command) for _, command in selected], "testRead": False}, indent=2))
        return
    if selected:
        results = _run_queue(selected, gpu_ids=args.gpu_ids, log_root=log_root)
        _write_json(args.benchmark_root / f"{args.mode}_training_results.json", {"results": results, "testRead": False})

    if args.mode in {"evaluate", "all"}:
        evaluation_jobs: list[tuple[str, Sequence[str]]] = []
        for name, variant, seed in _jobs():
            member = _member_dir(args.model_root, variant, seed)
            if not _member_complete(member):
                raise RuntimeError(f"formal member is incomplete: {member}")
            output = args.benchmark_root / "members" / name / "validation_evaluation.json"
            if output.is_file():
                continue
            evaluation_jobs.append(
                (f"evaluate_{name}", build_evaluate_command(args.root, args.model_root, args.benchmark_root, variant, seed))
            )
        if evaluation_jobs:
            results = _run_queue(
                evaluation_jobs,
                gpu_ids=args.gpu_ids,
                log_root=args.benchmark_root / "logs/evaluate",
            )
            _write_json(args.benchmark_root / "evaluation_results.json", {"results": results, "testRead": False})
        summary_result = _run_logged(
            [
                sys.executable,
                str(SUMMARIZE),
                "--benchmark-root",
                str(args.benchmark_root.resolve()),
            ],
            gpu=int(args.gpu_ids[0]),
            stdout_path=args.benchmark_root / "logs/summarize.stdout.log",
            stderr_path=args.benchmark_root / "logs/summarize.stderr.log",
        )
        if int(summary_result["returnCode"]) != 0:
            raise RuntimeError("formal summary failed; inspect summarize logs")


if __name__ == "__main__":
    main()
