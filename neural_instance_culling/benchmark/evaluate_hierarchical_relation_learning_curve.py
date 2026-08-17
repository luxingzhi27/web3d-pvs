#!/usr/bin/env python3
"""Replay every saved learning-curve checkpoint on validation.

The training matrix runner intentionally evaluates only ``best.pt``.  This
entry point handles the saved epoch snapshots without changing the training
output root: each epoch gets an independent matrix-shaped evaluation directory
so the existing paired-bootstrap summarizer can validate it unchanged.

Validation thresholds still come only from each checkpoint's calibration
artifact.  ``--allow-unsafe-diagnostic`` is used for early snapshots that have
no qualified calibration workpoint; those results remain diagnostic and are
never considered safe workpoints.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "neural_instance_culling/benchmark/run_pvs_hierarchical_relation_survival_integrated_matrix.py"
EVALUATOR = ROOT / "neural_instance_culling/benchmark/evaluate_pvs_hierarchical_relation_survival_integrated.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "matrix_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "learning_curve32":
        raise ValueError(f"expected learning_curve32 manifest, got {payload.get('mode')!r}")
    if payload.get("testRead") is not False:
        raise ValueError("training matrix manifest is not test-free")
    variants = payload.get("variants")
    seeds = payload.get("seeds")
    if not isinstance(variants, Mapping) or not variants:
        raise ValueError("learning curve manifest has no variants")
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("learning curve manifest has no seeds")
    return payload


def _evaluation_command(
    *,
    checkpoint: Path,
    output: Path,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint", str(checkpoint),
        "--dataset-dir", str(Path(args.dataset_dir).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--initial-geo-features", str(Path(args.initial_geo_features).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--output", str(output),
        "--split", "validation",
        "--poses-per-batch", str(args.poses_per_batch),
        "--device", str(args.device),
    ]
    if args.allow_unsafe_diagnostic:
        command.append("--allow-unsafe-diagnostic")
    return command


def _run_one(
    *,
    source_root: Path,
    output_root: Path,
    log_root: Path,
    variant: str,
    seed: int,
    epoch: int,
    source_epochs: int,
    gpu: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    source_member = source_root / f"{variant}_seed{seed}_e{int(source_epochs)}"
    checkpoint = source_member / f"checkpoint_epoch_{epoch:03d}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing snapshot: {checkpoint}")
    epoch_root = output_root / f"e{epoch:03d}"
    member = epoch_root / f"{variant}_seed{seed}_e{epoch}"
    output = member / "validation_evaluation.json"
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema") != "pvs-hierarchical-relation-survival-integrated-evaluation-v1":
            raise ValueError(f"existing snapshot output has unexpected schema: {output}")
        if payload.get("split") != "validation" or payload.get("testRead") is not False:
            raise ValueError(f"existing snapshot output has invalid split provenance: {output}")
        if str(payload.get("variant")) != str(variant) or int(payload.get("checkpointSeed", payload.get("seed", -1))) != int(seed):
            raise ValueError(f"existing snapshot output has mismatched member identity: {output}")
        if int(payload.get("poseCount", -1)) <= 0:
            raise ValueError(f"existing snapshot output has no poses: {output}")
        if str(payload.get("checkpoint")) != str(checkpoint.resolve()):
            raise ValueError(f"existing snapshot output points to another checkpoint: {output}")
        expected_sha = _sha256(checkpoint)
        if str(payload.get("checkpointSha256")) != expected_sha:
            raise ValueError(f"existing snapshot output checkpoint hash mismatch: {output}")
        return {
            "variant": variant,
            "seed": int(seed),
            "epoch": int(epoch),
            "checkpoint": str(checkpoint),
            "output": str(output),
            "gpu": int(gpu),
            "reusedExisting": True,
            "checkpointSha256": expected_sha,
            "returnCode": 0,
            "testRead": False,
        }
    member.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"e{epoch:03d}_{variant}_seed{seed}_stdout.log"
    stderr_path = log_root / f"e{epoch:03d}_{variant}_seed{seed}_stderr.log"
    command = _evaluation_command(checkpoint=checkpoint, output=output, args=args)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    result = {
        "variant": variant,
        "seed": int(seed),
        "epoch": int(epoch),
        "checkpoint": str(checkpoint),
        "checkpointSha256": _sha256(checkpoint),
        "output": str(output),
        "gpu": int(gpu),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "testRead": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"snapshot validation failed for {checkpoint}; see {stderr_path}")
    return result


def _epoch_manifest(
    source_manifest: Mapping[str, Any],
    *,
    epoch: int,
    epoch_root: Path,
    variants: list[str],
    seeds: list[int],
    allow_unsafe_diagnostic: bool,
) -> dict[str, Any]:
    return {
        "schema": "pvs-hierarchical-relation-survival-integrated-snapshot-validation-manifest-v1",
        "mode": "learning_curve_snapshot_validation",
        "sourceMatrixManifest": str((Path(source_manifest["_path"]).resolve())),
        "sourceMatrixManifestSha256": _sha256(Path(source_manifest["_path"])),
        "sourceMode": source_manifest.get("mode"),
        "variants": {name: source_manifest["variants"][name] for name in variants},
        "seeds": seeds,
        "epochs": int(epoch),
        "stepsPerEpoch": int(source_manifest.get("stepsPerEpoch", 0)),
        "snapshotEpoch": int(epoch),
        "splitPoseCounts": source_manifest.get("splitPoseCounts"),
        "splitCandidateDigests": source_manifest.get("splitCandidateDigests"),
        "candidateSemantics": "stored native back-camera candidate CSR; no GT union, cap, or frontend whitelist",
        "allowUnsafeDiagnostic": bool(allow_unsafe_diagnostic),
        "testRead": False,
        "epochRoot": str(epoch_root.resolve()),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, required=True)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--runtime-meta", required=True)
    parser.add_argument("--initial-geo-features", required=True)
    parser.add_argument("--glb-index", required=True)
    parser.add_argument("--glb-root", required=True)
    parser.add_argument("--poses-per-batch", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument(
        "--allow-unsafe-diagnostic",
        action="store_true",
        help="allow snapshots without a calibration-safe workpoint as diagnostic-only replays",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.epochs or any(epoch <= 0 for epoch in args.epochs):
        raise ValueError("epochs must be positive")
    if not args.gpu_ids or any(gpu < 0 for gpu in args.gpu_ids):
        raise ValueError("gpu-ids must be non-empty and non-negative")
    source_root = args.matrix_root.resolve()
    output_root = args.output_root.resolve()
    log_root = args.log_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output root: {output_root}")
    if log_root.exists() and any(log_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty log root: {log_root}")
    source_manifest = _load_manifest(source_root)
    source_manifest = dict(source_manifest)
    source_manifest["_path"] = str(source_root / "matrix_manifest.json")
    variants = [str(name) for name in source_manifest["variants"]]
    seeds = [int(seed) for seed in source_manifest["seeds"]]
    epochs = sorted(set(int(epoch) for epoch in args.epochs))
    if any(epoch > int(source_manifest.get("epochs", 0)) for epoch in epochs):
        raise ValueError("requested snapshot epoch exceeds training epochs")
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (variant, seed, epoch, args.gpu_ids[index % len(args.gpu_ids)])
        for index, (epoch, variant, seed) in enumerate(
            (epoch, variant, seed)
            for epoch in epochs
            for variant in variants
            for seed in seeds
        )
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(args.gpu_ids)) as executor:
        futures = [
            executor.submit(
                _run_one,
                source_root=source_root,
                output_root=output_root,
                log_root=log_root,
                variant=variant,
                seed=seed,
                epoch=epoch,
                source_epochs=int(source_manifest["epochs"]),
                gpu=gpu,
                args=args,
            )
            for variant, seed, epoch, gpu in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (int(item["epoch"]), str(item["variant"]), int(item["seed"])))
    for epoch in epochs:
        epoch_root = output_root / f"e{epoch:03d}"
        payload = _epoch_manifest(
            source_manifest,
            epoch=epoch,
            epoch_root=epoch_root,
            variants=variants,
            seeds=seeds,
            allow_unsafe_diagnostic=bool(args.allow_unsafe_diagnostic),
        )
        _write_json(epoch_root / "matrix_manifest.json", payload)
    top = {
        "schema": "pvs-hierarchical-relation-survival-integrated-learning-curve-replay-v1",
        "sourceMatrixRoot": str(source_root),
        "sourceMatrixManifestSha256": _sha256(source_root / "matrix_manifest.json"),
        "variants": variants,
        "seeds": seeds,
        "epochs": epochs,
        "allowUnsafeDiagnostic": bool(args.allow_unsafe_diagnostic),
        "members": results,
        "testRead": False,
    }
    _write_json(output_root / "learning_curve_replay_manifest.json", top)
    print(json.dumps({"outputRoot": str(output_root), "epochs": epochs, "memberCount": len(results), "testRead": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
