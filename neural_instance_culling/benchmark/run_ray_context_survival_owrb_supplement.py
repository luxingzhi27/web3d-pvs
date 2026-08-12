#!/usr/bin/env python3
"""Run the registered offline-only supplement comparisons.

The supplement keeps the formal PoseCSR candidate/GT protocol fixed and
changes exactly one registered mechanism at a time.  It never exports these
variants to the browser: the deployable exporter accepts only the default
32-D context, direct-nine ray, monotone-survival configuration.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "neural_instance_culling" / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

TRAIN = ROOT / "neural_instance_culling/model/train_ray_context_survival_owrb.py"
EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_ray_context_survival_owrb.py"
IMAGE_EVALUATE = ROOT / "neural_instance_culling/benchmark/evaluate_ray_context_survival_owrb_image.py"
SUMMARY = ROOT / "neural_instance_culling/benchmark/summarize_ray_context_survival_owrb.py"

DEFAULT_INITIAL_GEO = ROOT / (
    "neural_instance_culling/model/out/"
    "pvs_m4_ablation_geometry_context_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/"
    "instance_geo_features_fp16.bin"
)


SUPPLEMENT_VARIANTS: dict[str, dict[str, Any]] = {
    "triangle_context32_direct9_monotone": {
        "evidence": "triangle",
        "contextQueryDim": 8,
        "rayMode": "direct9",
        "survivalParameterization": "monotone",
        "description": "registered reference configuration using triangle-depth relation evidence",
    },
    "triangle_context64_direct9_monotone": {
        "evidence": "triangle",
        "contextQueryDim": 16,
        "rayMode": "direct9",
        "survivalParameterization": "monotone",
        "description": "same relation evidence with 64-D fixed context coefficients",
    },
    "triangle_context32_fourier117_monotone": {
        "evidence": "triangle",
        "contextQueryDim": 8,
        "rayMode": "fourier117",
        "survivalParameterization": "monotone",
        "description": "same relation evidence with the historical 117-D Fourier ray input",
    },
    "triangle_context32_direct9_unconstrained28": {
        "evidence": "triangle",
        "contextQueryDim": 8,
        "rayMode": "direct9",
        "survivalParameterization": "unconstrained28",
        "description": "same 28 coefficients without the monotone survival construction",
    },
    "aabb_context32_direct9_monotone": {
        "evidence": "aabb",
        "contextQueryDim": 8,
        "rayMode": "direct9",
        "survivalParameterization": "monotone",
        "description": "coarse AABB projection relation evidence with all other settings fixed",
    },
}


def _candidate_digest(dataset_dir: str, runtime_meta: str) -> str:
    import sys as _sys

    from common.candidate_identity import candidate_digest_for_pose_sequence
    from common.runtime_meta import load_runtime_meta
    from pose_csr_dataset import PoseCSRDataset

    num_instances = int(load_runtime_meta(runtime_meta)[0].shape[0])
    dataset = PoseCSRDataset(dataset_dir, num_instances=num_instances)
    return candidate_digest_for_pose_sequence(dataset, dataset.split("train").pose_indices)


def _validate_evidence(path: str, dataset_dir: str, runtime_meta: str, expected_kind: str) -> dict[str, Any]:
    evidence_path = Path(path).resolve()
    meta_path = evidence_path / "evidence_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing supplement evidence metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "ray-context-relation-evidence-v2":
        raise ValueError(f"{meta_path}: unexpected relation evidence schema")
    if float(meta.get("modelInputFovYDeg", 0.0)) != 66.0:
        raise ValueError(f"{meta_path}: relation evidence must use model-input FOV 66")
    if [str(value) for value in meta.get("splits", [])] != ["train"]:
        raise ValueError(f"{meta_path}: supplement relation evidence must be train-only")
    expected_digest = _candidate_digest(dataset_dir, runtime_meta)
    if str(meta.get("candidateDigest", "")) != expected_digest:
        raise ValueError(f"{meta_path}: candidate digest does not match current train CSR")
    if (meta.get("survivalObservationChecksums") or {}).get("byteIdentical") is not True:
        raise ValueError(f"{meta_path}: survival observations are not byte-identical")
    variant = str(meta.get("relationBuildVariant", ""))
    if expected_kind == "triangle" and variant != "surface_fallback_merged_v1":
        raise ValueError(f"{meta_path}: expected the registered triangle-depth surface-fallback relation table")
    if expected_kind == "aabb" and variant != "aabb_projection_supplement_v1":
        raise ValueError(f"{meta_path}: expected the AABB projection supplement relation table")
    return {
        "path": str(meta_path),
        "kind": expected_kind,
        "candidateDigest": expected_digest,
        "relationBuildVariant": variant,
        "nonzeroRelationCells": int(meta.get("stats", {}).get("nonzeroRelationCells", 0)),
    }


def _command_for_member(
    args: argparse.Namespace,
    variant: str,
    spec: dict[str, Any],
    seed: int,
    model_dir: Path,
    evidence_dir: str,
    gpu: str,
) -> tuple[list[str], dict[str, str]]:
    command = [
        "conda", "run", "--no-capture-output", "-n", args.environment,
        "python", "-u", str(TRAIN),
        "--dataset-dir", args.dataset_dir,
        "--evidence-dir", evidence_dir,
        "--runtime-meta", args.runtime_meta,
        "--glb-points", args.glb_points,
        "--glb-index", args.glb_index,
        "--glb-root", args.glb_root,
        "--initial-geo-features", args.initial_geo_features,
        "--output-dir", str(model_dir),
        "--experiment-name", f"{args.experiment_name}_{variant}_seed{seed}_e{args.epochs}",
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--pose-set-batch-size", str(args.pose_set_batch_size),
        "--eval-every", str(args.eval_every),
        "--context-mode", "directional",
        "--context-query-dim", str(spec["contextQueryDim"]),
        "--ray-mode", str(spec["rayMode"]),
        "--survival-parameterization", str(spec["survivalParameterization"]),
        "--survival-input-mode", "semantic",
        "--loss-mode", str(args.loss_mode),
        "--gamma", str(args.gamma),
        "--context-pretrain-steps", str(args.context_pretrain_steps),
        "--calibration-bootstrap-replicates", str(args.calibration_bootstrap_replicates),
        "--seed", str(seed),
        "--device", "cuda",
        "--allow-unsafe-final",
    ]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, environment


def _run_member(
    args: argparse.Namespace,
    variant: str,
    spec: dict[str, Any],
    seed: int,
    gpu: str,
    evidence_dirs: dict[str, str],
) -> tuple[str, int, Path, Path]:
    model_dir = Path(args.model_root) / f"{variant}_seed{seed}_e{args.epochs}"
    benchmark_dir = Path(args.benchmark_root) / f"{variant}_seed{seed}_e{args.epochs}"
    model_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = evidence_dirs[str(spec["evidence"])]
    command, environment = _command_for_member(
        args, variant, spec, seed, model_dir, evidence_dir, gpu
    )
    train_stdout = model_dir / "train_stdout.log"
    train_stderr = model_dir / "train_stderr.log"
    with train_stdout.open("w", encoding="utf-8") as stdout, train_stderr.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"training failed for {variant}/seed{seed}; see {train_stderr}")
    checkpoint = model_dir / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"training produced no best checkpoint for {variant}/seed{seed}")
    evaluation = benchmark_dir / "validation.json"
    evaluate_command = [
        "conda", "run", "--no-capture-output", "-n", args.environment,
        "python", "-u", str(EVALUATE),
        "--checkpoint", str(checkpoint),
        "--evidence-dir", evidence_dir,
        "--dataset-dir", args.dataset_dir,
        "--runtime-meta", args.runtime_meta,
        "--glb-index", args.glb_index,
        "--glb-root", args.glb_root,
        "--split", "validation",
        "--output", str(evaluation),
        "--device", "cuda",
        "--seed", str(seed),
        "--persist-ids",
        "--allow-unsafe-diagnostic",
    ]
    evaluate_stdout = benchmark_dir / "evaluate_stdout.log"
    evaluate_stderr = benchmark_dir / "evaluate_stderr.log"
    with evaluate_stdout.open("w", encoding="utf-8") as stdout, evaluate_stderr.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(evaluate_command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed for {variant}/seed{seed}; see {evaluate_stderr}")
    if args.run_image_evaluation:
        image_dir = benchmark_dir / "image_validation"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_command = [
            "conda", "run", "--no-capture-output", "-n", args.environment,
            "python", "-u", str(IMAGE_EVALUATE),
            "--evaluation", str(evaluation),
            "--dataset-dir", args.dataset_dir,
            "--runtime-meta", args.runtime_meta,
            "--glb-index", args.glb_index,
            "--glb-root", args.glb_root,
            "--output-dir", str(image_dir),
            "--render-timeout-ms", str(args.image_timeout_ms),
        ]
        if args.chrome_exe:
            image_command.extend(["--chrome-exe", args.chrome_exe])
        if args.image_max_poses > 0:
            image_command.extend(["--max-poses", str(args.image_max_poses)])
        with (image_dir / "image_stdout.log").open("w", encoding="utf-8") as stdout, (image_dir / "image_stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(image_command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"image evaluation failed for {variant}/seed{seed}; see {image_dir}")
    return variant, seed, checkpoint, evaluation


def _comparisons(variants: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    pairs = [
        ("context64_minus_context32", "triangle_context64_direct9_monotone", "triangle_context32_direct9_monotone"),
        ("fourier117_minus_direct9", "triangle_context32_fourier117_monotone", "triangle_context32_direct9_monotone"),
        ("unconstrained28_minus_monotone", "triangle_context32_direct9_unconstrained28", "triangle_context32_direct9_monotone"),
        ("aabb_minus_triangle_relation", "aabb_context32_direct9_monotone", "triangle_context32_direct9_monotone"),
    ]
    return [
        {"name": name, "left": left, "right": right}
        for name, left, right in pairs
        if left in variants and right in variants
    ]


def _write_manifest(args: argparse.Namespace, results: dict[str, dict[str, Path]], evidence_audit: dict[str, Any], path: Path) -> None:
    payload = {
        "schema": "ray-context-survival-owrb-matrix-manifest-v1",
        "experiment": args.experiment_name,
        "mode": args.mode,
        "supplement": True,
        "seeds": args.seeds,
        "variants": {
            variant: {str(seed): str(value.resolve()) for seed, value in members.items()}
            for variant, members in results.items()
        },
        "comparisons": _comparisons({name: SUPPLEMENT_VARIANTS[name] for name in results}),
        "data": {
            "datasetDir": str(Path(args.dataset_dir).resolve()),
            "runtimeMeta": str(Path(args.runtime_meta).resolve()),
            "modelInputFovYDeg": 66.0,
            "frontendRenderFovYDeg": 60.0,
            "candidatePolicy": "native back-camera CSR; no GT-visible union",
            "evidenceAudit": evidence_audit,
            "variantDefinitions": {name: SUPPLEMENT_VARIANTS[name] for name in results},
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--environment", default="slm_pvs")
    parser.add_argument("--dataset-dir", default="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1")
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--triangle-evidence-dir", required=True)
    parser.add_argument("--aabb-evidence-dir", required=True)
    parser.add_argument("--glb-points", default="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin")
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_INITIAL_GEO))
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--experiment-name", default="pvs_ray_context_survival_owrb_supplement_v1")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--pose-set-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=12)
    parser.add_argument("--context-pretrain-steps", type=int, default=64)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--loss-mode", choices=("rvl_strong_v2", "owrb", "safety_constraint"), default="rvl_strong_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260801])
    parser.add_argument("--variants", nargs="+", choices=tuple(SUPPLEMENT_VARIANTS), default=None)
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--run-summary", action="store_true")
    parser.add_argument("--run-image-evaluation", action="store_true")
    parser.add_argument("--chrome-exe", default="")
    parser.add_argument("--image-timeout-ms", type=int, default=86_400_000)
    parser.add_argument("--image-max-poses", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "formal":
        if len(args.seeds) != 3:
            raise ValueError("formal supplement requires the three registered seeds")
        args.epochs = 40
        if args.bootstrap_replicates < 10000:
            raise ValueError("formal supplement requires at least 10,000 paired bootstrap replicates")
    selected_names = list(args.variants or SUPPLEMENT_VARIANTS)
    variants = {name: SUPPLEMENT_VARIANTS[name] for name in selected_names}
    if not variants:
        raise ValueError("no supplement variants selected")
    args.initial_geo_features = str(Path(args.initial_geo_features).resolve())
    evidence_audit = {
        "triangle": _validate_evidence(args.triangle_evidence_dir, args.dataset_dir, args.runtime_meta, "triangle"),
        "aabb": _validate_evidence(args.aabb_evidence_dir, args.dataset_dir, args.runtime_meta, "aabb"),
    }
    if evidence_audit["triangle"]["candidateDigest"] != evidence_audit["aabb"]["candidateDigest"]:
        raise ValueError("triangle and AABB supplement evidence use different candidate digests")
    if args.steps_per_epoch <= 0:
        from common.runtime_meta import load_runtime_meta
        from pose_csr_dataset import PoseCSRDataset
        num_instances = int(load_runtime_meta(args.runtime_meta)[0].shape[0])
        train_count = int(PoseCSRDataset(args.dataset_dir, num_instances=num_instances).split("train").pose_indices.size)
        args.steps_per_epoch = max(1, (train_count + args.pose_set_batch_size - 1) // args.pose_set_batch_size)
    evidence_dirs = {"triangle": args.triangle_evidence_dir, "aabb": args.aabb_evidence_dir}
    tasks = [
        (variant, spec, seed, args.gpus[index % len(args.gpus)])
        for index, (variant, spec, seed) in enumerate(
            (item for variant in variants for item in [(variant, variants[variant], seed) for seed in args.seeds])
        )
    ]
    results: dict[str, dict[str, Path]] = {variant: {} for variant in variants}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(int(args.jobs), len(tasks))) as executor:
        futures = {
            executor.submit(_run_member, args, variant, spec, seed, gpu, evidence_dirs): (variant, seed)
            for variant, spec, seed, gpu in tasks
        }
        for future in as_completed(futures):
            variant, seed = futures[future]
            try:
                _variant, _seed, _checkpoint, evaluation = future.result()
                results[variant][str(seed)] = evaluation
                print(json.dumps({"status": "supplement_member_complete", "variant": variant, "seed": seed, "evaluation": str(evaluation)}, ensure_ascii=False), flush=True)
            except Exception as error:  # noqa: BLE001
                failures.append(f"{variant}/seed{seed}: {error}")
                print(json.dumps({"status": "supplement_member_failed", "variant": variant, "seed": seed, "error": str(error)}, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError("supplement failures: " + " | ".join(failures))
    manifest = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_matrix_manifest.json"
    _write_manifest(args, results, evidence_audit, manifest)
    summary_path = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_summary.json"
    report_path = ROOT / "docs/evaluation" / f"{args.experiment_name}_{args.mode}_validation.md"
    if args.run_summary:
        subprocess.run([
            "conda", "run", "--no-capture-output", "-n", args.environment,
            "python", "-u", str(SUMMARY),
            "--manifest", str(manifest), "--output", str(summary_path),
            "--report", str(report_path), "--bootstrap-replicates", str(args.bootstrap_replicates),
        ], cwd=ROOT, check=True)
    print(json.dumps({
        "status": "supplement_complete",
        "manifest": str(manifest.resolve()),
        "variantCount": len(results),
        "seedCount": len(args.seeds),
        "summary": str(summary_path.resolve()) if args.run_summary else None,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
