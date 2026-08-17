#!/usr/bin/env python3
"""Run the isolated hierarchical-survival/spectral/RVL experiment matrix.

This is the only orchestration entry point for the 2026-08-13 experiment.  It
launches independent single-GPU processes, writes an immutable manifest before
training, and refuses to reuse a member directory.  ``test`` is never loaded
by this file; test evaluation belongs to a later explicitly frozen command.

The modes are intentionally small and composable:

``smoke``      one member, one epoch, two poses;
``pilot``      four relation-source members, one seed, eight epochs;
``learning_curve32`` R0/R2, two seeds, thirty-two epochs;
``confirm``    selected members, two seeds, sixteen epochs;
``module_recheck16`` R1/R3/S1/S2/L1, two seeds, sixteen epochs and full calibration;
``spectral_pilot`` three query controls, one seed, eight epochs;
``loss_pilot`` four loss controls, one seed, eight epochs;
``formal80``   seven registered members, three seeds, eighty epochs;
``evaluate``   replay validation at each checkpoint's frozen calibration threshold;
``summarize``  collect member summaries and optionally run the formal bootstrap.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from validate_pvs_hierarchical_relation_survival_integrated import (  # noqa: E402
    FORMAL_SEEDS,
    FORMAL_VARIANTS,
    LOSS_PILOT_VARIANTS,
    MODULE_RECHECK_VARIANTS,
    PILOT_VARIANTS,
    SPECTRAL_PILOT_VARIANTS,
    validate_evaluation_result,
    validate_training_artifact,
)


TRAIN = ROOT / "neural_instance_culling/model/train_hierarchical_relation_survival_integrated.py"
EXPORT = ROOT / "neural_instance_culling/model/export_hierarchical_relation_survival_integrated.py"
PREFIX = "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1"
DEFAULT_DATASET = ROOT / "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1"
DEFAULT_RELATION = ROOT / "neural_instance_culling/dataset/out" / PREFIX / "relation_csr"
DEFAULT_RUNTIME_META = ROOT / "hkust-v3/assets/runtimeVisibilityMeta.json"
DEFAULT_GLB_INDEX = ROOT / "hkust-v3/assets/glbIndex.json"
DEFAULT_GLB_ROOT = ROOT / "hkust-v3/assets"
DEFAULT_GEO = ROOT / (
    "neural_instance_culling/model/out/"
    "pvs_m4_ablation_geometry_context_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/"
    "instance_geo_features_fp16.bin"
)
DEFAULT_SIDECAR = ROOT / "neural_instance_culling/dataset/out" / PREFIX / "subpose_quality_sidecar"

SIDECAR_SCHEMA = "pvs-viewcell-subpose-supervision-sidecar-v1"
RELATION_SOURCES = ("hierarchical", "single_scale", "free", "shuffled")
SPECTRAL_MODES = ("integrated", "learned_point", "fourier117")
LOSS_VARIANTS = ("rvl", "quality", "quality_resource")

# These are the registered quality/resource defaults.  Keeping them in the
# matrix manifest makes a run reproducible even when the training CLI defaults
# change later.
QUALITY_RESOURCE_DEFAULTS = {
    "qualityLossWeight": 0.5,
    "resourceLossWeight": 0.05,
    "qualityTemperature": 0.10,
    "qualityTailFraction": 0.01,
    "rarePositiveWeight": 1.0,
    "resourceWarmupSteps": 0,
    "resourceGateQuality": 0.0,
    "resourceGradientProjection": True,
    "allowMissingGlbCosts": False,
}

LOSS_MATRIX_VARIANTS = LOSS_PILOT_VARIANTS
REGISTERED_VARIANTS = {**PILOT_VARIANTS, **FORMAL_VARIANTS, **LOSS_MATRIX_VARIANTS}
DEFAULT_SUBPOSE_SIDECAR = ROOT / (
    "neural_instance_culling/dataset/out/"
    "pvs_hierarchical_relation_survival_integrated_spectral_quality_rvl_v1/"
    "subpose_quality_sidecar"
)


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_sidecar_file(sidecar_dir: Path, relative: Any, key: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"sidecar files.{key} must be a non-empty relative path")
    candidate = (sidecar_dir / relative).resolve()
    try:
        candidate.relative_to(sidecar_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"sidecar files.{key} escapes sidecar directory") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"missing sidecar file: {candidate}")
    return candidate


def _validate_quality_sidecar(dataset_dir: Path, sidecar_dir: Path) -> dict[str, Any]:
    """Validate the sidecar the training entry point reads by fixed filename.

    The trainer accepts the validated sidecar explicitly and uses the fixed
    ``visible_hit_counts.bin`` and ``subpose_offsets.bin`` names inside it.
    The matrix runner validates the source before passing it to each member.
    """
    manifest_path = sidecar_dir / "sidecar_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing quality sidecar manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SIDECAR_SCHEMA:
        raise ValueError(f"unexpected quality sidecar schema: {manifest.get('schema')!r}")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("quality sidecar manifest has no files mapping")
    visible_hit_counts = _resolve_sidecar_file(sidecar_dir, files.get("visibleHitCounts"), "visibleHitCounts")
    subpose_offsets = _resolve_sidecar_file(sidecar_dir, files.get("subposeOffsets"), "subposeOffsets")

    dataset_meta_path = dataset_dir / "dataset_meta.json"
    if not dataset_meta_path.is_file():
        raise FileNotFoundError(f"missing dataset metadata: {dataset_meta_path}")
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    pose_count = int(dataset_meta.get("poseCount", dataset_meta.get("viewcellCount", -1)))
    if pose_count <= 0:
        raise ValueError("dataset metadata has no positive poseCount/viewcellCount")
    if int(manifest.get("poseCount", -1)) != pose_count:
        raise ValueError("quality sidecar poseCount does not match dataset")
    if int(subpose_offsets.stat().st_size) != (pose_count + 1) * 8:
        raise ValueError("quality sidecar subposeOffsets has an invalid uint64 length")
    if visible_hit_counts.stat().st_size % 2:
        raise ValueError("quality sidecar visibleHitCounts has an invalid uint16 length")
    visible_count = int(visible_hit_counts.stat().st_size // 2)
    if int(manifest.get("visibleCount", -1)) != visible_count:
        raise ValueError("quality sidecar visibleCount does not match visibleHitCounts")

    expected_meta_sha = manifest.get("mainCsrMetaSha256")
    actual_meta_sha = _sha256_file(dataset_meta_path)
    if expected_meta_sha is not None and str(expected_meta_sha) != actual_meta_sha:
        raise ValueError("quality sidecar was built for a different dataset_meta.json")
    validated = manifest.get("validatedFiles", {})
    if isinstance(validated, Mapping):
        for key, relative_name in (
            ("mainCsrPoses", "poses.bin"),
            ("mainCsrVisibleIds", "visible_ids.bin"),
            ("mainCsrVisibleWeights", "visible_weights.bin"),
            ("mainCsrCandidateIds", "candidate_ids.bin"),
            ("mainCsrCandidateOffsets", "candidate_offsets.bin"),
        ):
            expected = validated.get(key)
            source = dataset_dir / relative_name
            if expected is not None:
                if not source.is_file() or _sha256_file(source) != str(expected):
                    raise ValueError(f"quality sidecar validation digest mismatch for dataset {relative_name}")
    return {
        "path": str(sidecar_dir.resolve()),
        "manifest": str(manifest_path.resolve()),
        "manifestSha256": _sha256_file(manifest_path),
        "schema": SIDECAR_SCHEMA,
        "poseCount": pose_count,
        "visibleCount": visible_count,
        "files": {
            "visibleHitCounts": {
                "source": str(visible_hit_counts),
                "sha256": _sha256_file(visible_hit_counts),
                "bytes": int(visible_hit_counts.stat().st_size),
            },
            "subposeOffsets": {
                "source": str(subpose_offsets),
                "sha256": _sha256_file(subpose_offsets),
                "bytes": int(subpose_offsets.stat().st_size),
            },
        },
    }


def _resolve_split_counts(dataset: PoseCSRDataset) -> dict[str, int]:
    result: dict[str, int] = {}
    for split in ("train", "calibration", "validation"):
        if split not in dataset.split_ids:
            raise ValueError(f"dataset has no required split {split!r}")
        result[split] = int(dataset.split(split).pose_indices.size)
    return result


def _protocol_manifest(dataset_dir: Path, runtime_meta: Path, relation_dir: Path) -> dict[str, Any]:
    world_aabbs, _instance_to_glb, runtime = load_runtime_meta(runtime_meta)
    dataset = PoseCSRDataset(dataset_dir, num_instances=int(world_aabbs.shape[0]))
    counts = _resolve_split_counts(dataset)
    split_digests = {
        split: candidate_digest_for_pose_sequence(dataset, dataset.split(split).pose_indices)
        for split in ("train", "calibration", "validation")
    }
    relation_meta_path = relation_dir / "relation_csr_meta.json"
    relation_meta = json.loads(relation_meta_path.read_text(encoding="utf-8"))
    relation_digest, relation_files = relation_artifact_digest(relation_dir, relation_meta)
    train_relation_digest = relation_meta["candidateIdentity"]["canonicalCandidateDigest"]
    if train_relation_digest != split_digests["train"]:
        raise ValueError("relation CSR canonical candidate digest does not match native train candidates")
    return {
        "schema": "pvs-hierarchical-relation-survival-integrated-matrix-manifest-v1",
        "experimentPrefix": PREFIX,
        "datasetDir": str(dataset_dir.resolve()),
        "datasetMetaSha256": hashlib.sha256((dataset_dir / "dataset_meta.json").read_bytes()).hexdigest(),
        "runtimeMeta": {"path": str(runtime_meta.resolve()), "sha256": hashlib.sha256(runtime_meta.read_bytes()).hexdigest()},
        "numInstances": int(world_aabbs.shape[0]),
        "numGlbs": int(runtime.get("numGlbs", int(_instance_to_glb.max()) + 1)),
        "splitPoseCounts": counts,
        "splitCandidateDigests": split_digests,
        "relation": {
            "path": str(relation_dir.resolve()),
            "schema": relation_meta.get("schema"),
            "canonicalCandidateDigest": train_relation_digest,
            "renderCandidateDigest": relation_meta["candidateIdentity"]["renderCandidateDigest"],
            "artifactDigest": relation_digest,
            "files": relation_files,
        },
        "testRead": False,
    }


def _variant_arguments(name: str, spec: Mapping[str, Any], args: argparse.Namespace) -> list[str]:
    loss = str(spec["lossVariant"])
    return [
        "--variant", name,
        "--relation-source", str(spec["relationSource"]),
        "--spectral-mode", str(spec["spectralMode"]),
        "--loss-variant", loss,
        "--quality-loss-weight", str(
            spec.get("qualityWeight", args.quality_loss_weight)
            if loss in {"quality", "quality_resource"} else 0.0
        ),
        "--resource-loss-weight", str(spec.get("effectiveResourceWeight", 0.0)),
        "--quality-temperature", str(args.quality_temperature),
        "--quality-tail-fraction", str(args.quality_tail_fraction),
        "--rare-positive-weight", str(args.rare_positive_weight),
        "--resource-warmup-steps", str(args.resource_warmup_steps),
        "--resource-gate-quality", str(args.resource_gate_quality),
        "--subpose-sidecar", str(args.subpose_sidecar),
    ]


def _default_eval_every(mode: str) -> int:
    """Return the evaluation cadence without changing historical modes."""

    return 4 if mode in {"learning_curve32", "formal80"} else 1


def _resolve_eval_every(mode: str, requested: int | None) -> int:
    value = _default_eval_every(mode) if requested is None else int(requested)
    if value <= 0:
        raise ValueError("eval-every must be positive")
    return value


def _mode_config(mode: str, requested_variants: list[str] | None, requested_seeds: list[int] | None) -> tuple[dict[str, dict[str, Any]], list[int], int, int, int, bool]:
    if mode == "pilot":
        variants = {name: PILOT_VARIANTS[name] for name in (requested_variants or list(PILOT_VARIANTS))}
        return variants, requested_seeds or [20260801], 8, 75, 32, True
    if mode == "spectral_pilot":
        variants = {name: SPECTRAL_PILOT_VARIANTS[name] for name in (requested_variants or list(SPECTRAL_PILOT_VARIANTS))}
        return variants, requested_seeds or [20260801], 8, 75, 32, True
    if mode == "loss_pilot":
        variants = {name: LOSS_PILOT_VARIANTS[name] for name in (requested_variants or list(LOSS_PILOT_VARIANTS))}
        return variants, requested_seeds or [20260801], 8, 75, 32, True
    if mode == "learning_curve32":
        names = requested_variants or ["R0_free_survival", "R2_hierarchical_relation"]
        variants = {name: (PILOT_VARIANTS | FORMAL_VARIANTS)[name] for name in names}
        return variants, requested_seeds or [20260801, 20260802], 32, 100, 0, False
    if mode == "confirm":
        names = requested_variants or ["R0_free_survival", "R2_hierarchical_relation"]
        variants = {name: (PILOT_VARIANTS | FORMAL_VARIANTS)[name] for name in names}
        return variants, requested_seeds or [20260801, 20260802], 16, 100, 0, False
    if mode == "module_recheck16":
        names = requested_variants or list(MODULE_RECHECK_VARIANTS)
        variants = {name: MODULE_RECHECK_VARIANTS[name] for name in names}
        return variants, requested_seeds or [20260801, 20260802], 16, 100, 0, False
    if mode == "formal80":
        variants = {name: FORMAL_VARIANTS[name] for name in (requested_variants or list(FORMAL_VARIANTS))}
        return variants, requested_seeds or list(FORMAL_SEEDS), 80, 150, 0, False
    if mode == "smoke":
        names = requested_variants or ["R2_hierarchical_relation"]
        variants = {name: (PILOT_VARIANTS | FORMAL_VARIANTS).get(name, PILOT_VARIANTS["R2_hierarchical_relation"]) for name in names}
        return variants, requested_seeds or [20260801], 1, 1, 2, True
    raise ValueError(f"unsupported training mode {mode!r}")


def _member_command(
    *,
    name: str,
    spec: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
    member: Path,
    epochs: int,
    steps: int,
    max_eval_poses: int,
    eval_every: int,
    allow_unsafe: bool,
) -> list[str]:
    command = [
        sys.executable, str(TRAIN),
        "--dataset-dir", str(args.dataset_dir),
        "--relation-dir", str(args.relation_dir),
        "--runtime-meta", str(args.runtime_meta),
        "--initial-geo-features", str(args.initial_geo_features),
        "--glb-index", str(args.glb_index),
        "--glb-root", str(args.glb_root),
        "--output-dir", str(member),
        "--experiment-name", f"{PREFIX}_{name}_seed{seed}_e{epochs}",
        *_variant_arguments(name, spec, args),
        "--epochs", str(epochs),
        "--steps-per-epoch", str(steps),
        "--max-eval-poses", str(max_eval_poses),
        "--eval-every", str(eval_every),
        "--snapshot-every", str(eval_every),
        "--calibration-bootstrap-replicates", str(args.calibration_bootstrap_replicates),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    if allow_unsafe:
        command.append("--allow-unsafe-final")
    return command


def _run_member(
    *,
    name: str,
    spec: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
    output_root: Path,
    log_root: Path,
    epochs: int,
    steps: int,
    max_eval_poses: int,
    eval_every: int,
    allow_unsafe: bool,
    gpu: int,
) -> dict[str, Any]:
    member = output_root / f"{name}_seed{seed}_e{epochs}"
    if member.exists() and any(member.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty member directory: {member}")
    log_root.mkdir(parents=True, exist_ok=True)
    stdout_path = log_root / f"{name}_seed{seed}_stdout.log"
    stderr_path = log_root / f"{name}_seed{seed}_stderr.log"
    command = _member_command(
        name=name,
        spec=spec,
        seed=seed,
        args=args,
        member=member,
        epochs=epochs,
        steps=steps,
        max_eval_poses=max_eval_poses,
        eval_every=eval_every,
        allow_unsafe=allow_unsafe,
    )
    environment = dict(__import__("os").environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    result = {
        "variant": name,
        "seed": int(seed),
        "gpu": int(gpu),
        "outputDir": str(member),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "command": command,
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "testRead": False,
    }
    if completed.returncode == 0 and (member / "last.pt").exists():
        result["lastCheckpoint"] = str(member / "last.pt")
        result["calibration"] = str(member / "calibration_ready_summary.json")
    return result


def run_training_matrix(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).resolve()
    relation_dir = Path(args.relation_dir).resolve()
    runtime_meta = Path(args.runtime_meta).resolve()
    output_root = Path(args.output_root).resolve()
    log_root = Path(args.log_root).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty matrix output root: {output_root}")
    if log_root.exists() and any(log_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty matrix log root: {log_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = _protocol_manifest(dataset_dir, runtime_meta, relation_dir)
    variants, seeds, epochs, steps, max_eval_poses, allow_unsafe = _mode_config(args.mode, args.variants, args.seeds)
    eval_every = _resolve_eval_every(args.mode, args.eval_every)
    quality_variants = {
        name for name, spec in variants.items()
        if str(spec.get("lossVariant", "")) in {"quality", "quality_resource"}
    }
    sidecar_info = None
    if quality_variants:
        sidecar_dir = Path(args.subpose_sidecar).resolve() if str(args.subpose_sidecar).strip() else None
        if sidecar_dir is None:
            raise ValueError(
                "quality-aware matrix members require --subpose-sidecar; "
                "the runner refuses to start without validated dense-subpose supervision"
            )
        sidecar_info = _validate_quality_sidecar(dataset_dir, sidecar_dir)
    manifest.update({
        "mode": args.mode,
        "variants": variants,
        "seeds": seeds,
        "epochs": epochs,
        "stepsPerEpoch": steps,
        "maxEvalPoses": max_eval_poses,
        "evalEvery": eval_every,
        "dryRun": bool(args.dry_run),
        "calibrationBootstrapReplicates": int(args.calibration_bootstrap_replicates),
        "qualityLossConfig": {
            "qualityWeight": float(args.quality_loss_weight),
            "qualityTemperature": float(args.quality_temperature),
            "qualityTailFraction": float(args.quality_tail_fraction),
            "rarePositiveWeight": float(args.rare_positive_weight),
            "resourceWarmupSteps": int(args.resource_warmup_steps),
            "resourceGateQuality": float(args.resource_gate_quality),
            "resourceGradientProjection": True,
            "subposeSidecar": str(Path(args.subpose_sidecar).resolve()) if args.subpose_sidecar else None,
        },
        "effectiveResourceWeights": {
            name: float(spec.get("effectiveResourceWeight", 0.0))
            for name, spec in variants.items()
        },
        "qualitySidecar": sidecar_info,
        "gpuPolicy": "one independent CUDA process per visible GPU; no distributed training",
        "gpuIds": [int(value) for value in (args.gpu_ids or list(range(int(args.gpu_count))))],
        "testRead": False,
    })
    gpu_ids = [int(value) for value in (args.gpu_ids or list(range(int(args.gpu_count))))]
    jobs = [
        (name, spec, seed, gpu_ids[index % len(gpu_ids)])
        for index, (name, spec, seed) in enumerate(
            (name, spec, seed) for name, spec in variants.items() for seed in seeds
        )
    ]
    planned_members = []
    for name, spec, seed, gpu in jobs:
        member = output_root / f"{name}_seed{seed}_e{epochs}"
        planned_members.append({
            "variant": name,
            "seed": int(seed),
            "gpu": int(gpu),
            "outputDir": str(member),
            "evalEvery": int(eval_every),
            "allowUnsafeFinal": bool(allow_unsafe),
            "command": _member_command(
                name=name,
                spec=spec,
                seed=seed,
                args=args,
                member=member,
                epochs=epochs,
                steps=steps,
                max_eval_poses=max_eval_poses,
                eval_every=eval_every,
                allow_unsafe=allow_unsafe,
            ),
        })
    manifest["plannedMembers"] = planned_members
    _write_json(output_root / "matrix_manifest.json", manifest)
    if args.dry_run:
        return {
            "schema": manifest["schema"],
            "manifest": str(output_root / "matrix_manifest.json"),
            "dryRun": True,
            "members": planned_members,
            "testRead": False,
        }
    log_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.gpu_count))) as executor:
        futures = [executor.submit(_run_member, name=name, spec=spec, seed=seed, args=args, output_root=output_root, log_root=log_root, epochs=epochs, steps=steps, max_eval_poses=max_eval_poses, eval_every=eval_every, allow_unsafe=allow_unsafe, gpu=gpu) for name, spec, seed, gpu in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda item: (item["variant"], item["seed"]))
    payload = {"schema": manifest["schema"], "manifest": str(output_root / "matrix_manifest.json"), "members": results, "testRead": False}
    _write_json(output_root / "matrix_run_summary.json", payload)
    if any(int(item["returnCode"]) != 0 for item in results):
        raise RuntimeError(f"one or more matrix members failed; see {log_root}")
    return payload


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    return value


def _summarize_training_root(root: Path) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for checkpoint in sorted(root.glob("*/last.pt")):
        payload = _load_torch_checkpoint(checkpoint)
        meta_path = checkpoint.parent / "model_meta.json"
        calibration_path = checkpoint.parent / "calibration_ready_summary.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None
        checked = validate_training_artifact(payload, model_meta=meta, calibration_summary=calibration, allow_unsafe=True)
        selected = (calibration or {}).get("selected")
        validation = (calibration or {}).get("validationAtCalibration")
        members.append({
            "variant": checked["variant"], "seed": checked["seed"],
            "status": (calibration or {}).get("status"),
            "threshold": None if not isinstance(selected, Mapping) else selected.get("threshold"),
            "validationAtCalibration": validation,
            "checkpoint": str(checkpoint), "testRead": False,
        })
    return {"schema": "pvs-hierarchical-relation-survival-integrated-training-summary-v1", "members": members, "testRead": False}


def _evaluation_command(
    *,
    checkpoint: Path,
    output: Path,
    args: argparse.Namespace,
    allow_unsafe: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "neural_instance_culling/benchmark/evaluate_pvs_hierarchical_relation_survival_integrated.py"),
        "--checkpoint", str(checkpoint),
        "--dataset-dir", str(Path(args.dataset_dir).resolve()),
        "--runtime-meta", str(Path(args.runtime_meta).resolve()),
        "--initial-geo-features", str(Path(args.initial_geo_features).resolve()),
        "--glb-index", str(Path(args.glb_index).resolve()),
        "--glb-root", str(Path(args.glb_root).resolve()),
        "--output", str(output),
        "--split", "validation",
        "--poses-per-batch", "2",
        "--device", "cuda",
    ]
    if allow_unsafe:
        command.append("--allow-unsafe-diagnostic")
    return command


def _evaluate_member(
    *,
    member: Path,
    args: argparse.Namespace,
    gpu: int,
    allow_unsafe: bool,
) -> dict[str, Any]:
    checkpoint = member / "best.pt"
    if not checkpoint.is_file():
        checkpoint = member / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"member has no checkpoint: {member}")
    output = member / "validation_evaluation.json"
    if output.exists():
        # Incremental evaluation is allowed to reuse an existing replay, but
        # it must be checked before it enters the matrix summary.  Never
        # overwrite a prior replay as part of a later matrix invocation.
        payload = json.loads(output.read_text(encoding="utf-8"))
        validate_evaluation_result(
            payload,
            expected_variant=str(member.name.rsplit("_seed", 1)[0]),
            expected_seed=int(member.name.split("_seed", 1)[1].split("_e", 1)[0]),
            expected_split="validation",
            allow_test=False,
        )
        return {
            "member": str(member),
            "checkpoint": str(checkpoint),
            "output": str(output),
            "gpu": int(gpu),
            "reusedExisting": True,
            "returnCode": 0,
            "elapsedSeconds": 0.0,
            "testRead": False,
        }
    command = _evaluation_command(
        checkpoint=checkpoint,
        output=output,
        args=args,
        allow_unsafe=allow_unsafe,
    )
    log_root = Path(args.log_root).resolve()
    stdout_path = log_root / f"{member.name}_evaluation_stdout.log"
    stderr_path = log_root / f"{member.name}_evaluation_stderr.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    result = {
        "member": str(member),
        "checkpoint": str(checkpoint),
        "output": str(output),
        "gpu": int(gpu),
        "command": command,
        "returnCode": int(completed.returncode),
        "elapsedSeconds": float(time.time() - started),
        "testRead": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"validation replay failed for {member}; see {stderr_path}")
    return result


def evaluate_training_root(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    manifest_path = root / "matrix_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing matrix manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    members = []
    for variant in manifest.get("variants", {}):
        for seed in manifest.get("seeds", []):
            member = root / f"{variant}_seed{int(seed)}_e{int(manifest['epochs'])}"
            if not member.is_dir():
                raise FileNotFoundError(f"missing matrix member: {member}")
            members.append(member)
    if not members:
        raise ValueError("matrix manifest contains no members")
    allow_unsafe = str(manifest.get("mode")) in {"pilot", "smoke"}
    gpu_ids = [int(value) for value in (args.gpu_ids or list(range(int(args.gpu_count))))]
    if not gpu_ids or any(value < 0 for value in gpu_ids):
        raise ValueError("gpu-ids must be a non-empty list of non-negative IDs")
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(
                _evaluate_member,
                member=member,
                args=args,
                gpu=gpu_ids[index % len(gpu_ids)],
                allow_unsafe=allow_unsafe,
            )
            for index, member in enumerate(members)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda value: value["member"])
    payload = {
        "schema": "pvs-hierarchical-relation-survival-integrated-validation-replay-v1",
        "manifest": str(manifest_path),
        "members": results,
        "testRead": False,
    }
    _write_json(root / "validation_replay_summary.json", payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=(
            "smoke", "pilot", "spectral_pilot", "loss_pilot", "confirm",
            "module_recheck16", "formal80", "evaluate", "summarize",
            "learning_curve32",
        ),
    )
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--relation-dir", default=str(DEFAULT_RELATION))
    parser.add_argument("--runtime-meta", default=str(DEFAULT_RUNTIME_META))
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_GEO))
    parser.add_argument("--subpose-sidecar", default=str(DEFAULT_SUBPOSE_SIDECAR))
    parser.add_argument("--glb-index", default=str(DEFAULT_GLB_INDEX))
    parser.add_argument("--glb-root", default=str(DEFAULT_GLB_ROOT))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--variants", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--gpu-count", type=int, default=4)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Physical CUDA device IDs to assign to independent members; defaults to 0..gpu-count-1.",
    )
    # Safety selection is a registered 10,000-replicate calibration protocol;
    # callers may lower this only for explicitly named smoke/pilot runs.
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--quality-loss-weight", type=float, default=0.5)
    parser.add_argument("--quality-temperature", type=float, default=0.10)
    parser.add_argument("--quality-tail-fraction", type=float, default=0.01)
    parser.add_argument("--rare-positive-weight", type=float, default=1.0)
    parser.add_argument("--resource-warmup-steps", type=int, default=0)
    parser.add_argument("--resource-gate-quality", type=float, default=0.0)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=None,
        help="Evaluate/calibrate every N epochs; defaults to 4 for learning_curve32/formal80 and 1 otherwise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and write only the matrix manifest; do not start training processes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gpu_count <= 0:
        raise ValueError("gpu-count must be positive")
    if args.gpu_ids is not None:
        if not args.gpu_ids or len(set(args.gpu_ids)) != len(args.gpu_ids):
            raise ValueError("gpu-ids must be a non-empty list of unique IDs")
        if any(value < 0 for value in args.gpu_ids):
            raise ValueError("gpu-ids must be non-negative")
        args.gpu_count = len(args.gpu_ids)
    if args.mode == "evaluate":
        summary = evaluate_training_root(args)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    if args.mode == "summarize":
        summary = _summarize_training_root(Path(args.output_root).resolve())
        _write_json(Path(args.output_root).resolve() / "training_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return
    result = run_training_matrix(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
