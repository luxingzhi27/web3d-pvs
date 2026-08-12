#!/usr/bin/env python3
"""Run the independent ray-context/survival/OWRB pilot or formal matrix."""
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
DECIDE = ROOT / "neural_instance_culling/benchmark/decide_ray_context_survival_owrb_route.py"
SELECT_FINAL = ROOT / "neural_instance_culling/benchmark/select_ray_context_survival_owrb_final.py"
DEFAULT_INITIAL_GEO = ROOT / (
    "neural_instance_culling/model/out/"
    "pvs_m4_ablation_geometry_context_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/"
    "instance_geo_features_fp16.bin"
)
EXPECTED_GEO_DIM = 96


FORMAL_VARIANTS: dict[str, dict[str, Any]] = {
    # Registered 2 x 2 x 2 design: context off/on, survival field off/on,
    # and RVL control/safety-constrained utility loss.
    "context_off_survival_off_rvl": {
        "contextMode": "pooled", "survivalInputMode": "semantic",
        "lossMode": "rvl_strong_v2", "disableSurvival": True,
    },
    "context_on_survival_off_rvl": {
        "contextMode": "directional", "survivalInputMode": "semantic",
        "lossMode": "rvl_strong_v2", "disableSurvival": True,
    },
    "context_off_survival_on_rvl": {
        "contextMode": "pooled", "survivalInputMode": "semantic",
        "lossMode": "rvl_strong_v2",
    },
    "context_on_survival_on_rvl": {
        "contextMode": "directional", "survivalInputMode": "semantic",
        "lossMode": "rvl_strong_v2",
    },
    "context_off_survival_off_safety": {
        "contextMode": "pooled", "survivalInputMode": "semantic",
        "lossMode": "safety_constraint", "disableSurvival": True,
    },
    "context_on_survival_off_safety": {
        "contextMode": "directional", "survivalInputMode": "semantic",
        "lossMode": "safety_constraint", "disableSurvival": True,
    },
    "context_off_survival_on_safety": {
        "contextMode": "pooled", "survivalInputMode": "semantic",
        "lossMode": "safety_constraint",
    },
    "context_on_survival_on_safety": {
        "contextMode": "directional", "survivalInputMode": "semantic",
        "lossMode": "safety_constraint",
    },
}


def _variant_specs(mode: str, requested: list[str] | None = None) -> dict[str, dict[str, Any]]:
    if mode == "formal":
        if requested and set(requested) != set(FORMAL_VARIANTS):
            raise ValueError("formal matrix must run the complete registered 2x2x2 variant set")
        return dict(FORMAL_VARIANTS)
    # The pilot may restrict itself to the full model so gamma is selected from
    # the resource-bearing route that will be used in the formal matrix.
    if not requested:
        return dict(FORMAL_VARIANTS)
    unknown = [name for name in requested if name not in FORMAL_VARIANTS]
    if unknown:
        raise ValueError(f"unknown OWRB variant(s): {unknown}")
    return {name: FORMAL_VARIANTS[name] for name in requested}


def _validate_fixed_geometry(args: argparse.Namespace) -> str:
    """Require the audited frozen 96-D geometry table for every member."""
    import numpy as np

    path = Path(args.initial_geo_features)
    if not path.is_file():
        raise FileNotFoundError(
            "the OWRB matrix requires the audited fixed geometry table: "
            f"{path}"
        )
    from common.runtime_meta import load_runtime_meta

    world_aabbs, _instance_to_glb, _runtime_meta = load_runtime_meta(args.runtime_meta)
    instance_count = int(world_aabbs.shape[0])
    expected_bytes = instance_count * EXPECTED_GEO_DIM * 2
    actual_bytes = int(path.stat().st_size)
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"fixed geometry table has {actual_bytes} bytes; expected "
            f"{expected_bytes} ({instance_count} instances x "
            f"{EXPECTED_GEO_DIM} FP16 values)"
        )
    values = np.fromfile(path, dtype=np.float16)
    if values.size != instance_count * EXPECTED_GEO_DIM:
        raise ValueError("fixed geometry table value count does not match runtime metadata")
    if not bool(np.isfinite(values).all()):
        raise ValueError("fixed geometry table contains non-finite values")
    return str(path.resolve())


def _validate_formal_evidence(args: argparse.Namespace) -> dict[str, Any]:
    """Fail closed unless formal evidence provenance is complete and aligned."""
    import numpy as np

    evidence_path = Path(args.evidence_dir)
    meta_path = evidence_path / "evidence_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"formal relation evidence metadata is missing: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != "ray-context-relation-evidence-v2":
        raise ValueError("formal OWRB matrix requires ray-context-relation-evidence-v2")
    if meta.get("relationBuildVariant") != "surface_fallback_merged_v1" or meta.get("surfaceFallbackMerged") is not True:
        raise ValueError(
            "formal OWRB matrix requires a relation table with per-render surface-point fallback merged; "
            "use an independent relation_evidence_v3 output"
        )
    if float(meta.get("modelInputFovYDeg", 0.0)) != 66.0:
        raise ValueError("formal OWRB relation evidence must use the 66 degree model-input FOV")
    if meta.get("splits") != ["train"]:
        raise ValueError("formal OWRB relation evidence must be train-only")
    checksums = meta.get("survivalObservationChecksums") or {}
    if checksums.get("byteIdentical") is not True:
        raise ValueError("formal OWRB relation evidence must preserve byte-identical survival observations")
    from common.runtime_meta import load_runtime_meta
    from pose_csr_dataset import PoseCSRDataset
    from common.candidate_identity import candidate_digest_for_pose_sequence

    world_aabbs, _instance_to_glb, _runtime = load_runtime_meta(args.runtime_meta)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=int(world_aabbs.shape[0]))
    digest = candidate_digest_for_pose_sequence(dataset, dataset.split("train").pose_indices)
    if str(meta.get("candidateDigest")) != digest:
        raise ValueError("formal OWRB relation evidence candidate digest does not match native train CSR")
    if int(meta.get("stats", {}).get("surfaceFallbackRelationCount", 0)) <= 0:
        raise ValueError("formal OWRB relation evidence contains no merged surface fallback relations")
    return {
        "relationEvidence": str(meta_path.resolve()),
        "candidateDigest": digest,
        "surfaceFallbackRelationCount": int(meta["stats"]["surfaceFallbackRelationCount"]),
        "surfaceFallbackTargetCount": int(meta["stats"].get("surfaceFallbackTargetCount", 0)),
        "surfaceFallbackPoseCount": int(meta["stats"].get("surfaceFallbackPoseCount", 0)),
    }


def _command_for_member(
    args: argparse.Namespace,
    variant: str,
    spec: dict[str, Any],
    seed: int,
    model_dir: Path,
    benchmark_dir: Path,
    gpu: str,
) -> tuple[list[str], dict[str, str]]:
    command = [
        "conda", "run", "--no-capture-output", "-n", args.environment,
        "python", "-u", str(TRAIN),
        "--dataset-dir", args.dataset_dir,
        "--evidence-dir", args.evidence_dir,
        "--runtime-meta", args.runtime_meta,
        "--glb-points", args.glb_points,
        "--glb-index", args.glb_index,
        "--glb-root", args.glb_root,
        "--output-dir", str(model_dir),
        "--experiment-name", f"{args.experiment_name}_{variant}_seed{seed}_{args.epochs}epoch",
        "--epochs", str(args.epochs),
        "--steps-per-epoch", str(args.steps_per_epoch),
        "--pose-set-batch-size", str(args.pose_set_batch_size),
        "--eval-every", str(args.eval_every),
        "--context-mode", str(spec["contextMode"]),
        "--survival-input-mode", str(spec["survivalInputMode"]),
        "--loss-mode", str(spec["lossMode"]),
        "--gamma", str(args.gamma),
        "--context-pretrain-steps", str(args.context_pretrain_steps),
        "--calibration-bootstrap-replicates", str(args.calibration_bootstrap_replicates),
        "--seed", str(seed),
        "--device", "cuda",
    ]
    command.extend(["--initial-geo-features", args.initial_geo_features])
    if bool(spec.get("disableSurvival", False)):
        command.append("--disable-survival")
    # A formal member without a safe calibration row still needs a diagnostic
    # checkpoint so the matrix can report it.  This flag does not relax the
    # safety rule; the checkpoint and evaluator mark the workpoint unsafe.
    command.append("--allow-unsafe-final")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return command, env


def _run_member(
    args: argparse.Namespace,
    variant: str,
    spec: dict[str, Any],
    seed: int,
    gpu: str,
) -> tuple[str, int, Path, Path]:
    model_dir = Path(args.model_root) / f"{variant}_seed{seed}_e{args.epochs}"
    benchmark_dir = Path(args.benchmark_root) / f"{variant}_seed{seed}_e{args.epochs}"
    model_dir.mkdir(parents=True, exist_ok=True)
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    command, env = _command_for_member(args, variant, spec, seed, model_dir, benchmark_dir, gpu)
    stdout = model_dir / "train_stdout.log"
    stderr = model_dir / "train_stderr.log"
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"training failed for {variant}, seed {seed}; see {stderr}")
    checkpoint = model_dir / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError(f"training produced no best checkpoint for {variant}, seed {seed}")
    evaluation = benchmark_dir / "validation.json"
    eval_command = [
        "conda", "run", "--no-capture-output", "-n", args.environment,
        "python", "-u", str(EVALUATE),
        "--checkpoint", str(checkpoint),
        "--evidence-dir", args.evidence_dir,
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
    eval_stdout = benchmark_dir / "evaluate_stdout.log"
    eval_stderr = benchmark_dir / "evaluate_stderr.log"
    with eval_stdout.open("w", encoding="utf-8") as out, eval_stderr.open("w", encoding="utf-8") as err:
        completed = subprocess.run(eval_command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"evaluation failed for {variant}, seed {seed}; see {eval_stderr}")
    if args.mode == "formal" or args.run_image_evaluation:
        image_dir = benchmark_dir / "image_validation"
        image_stdout = image_dir / "image_stdout.log"
        image_stderr = image_dir / "image_stderr.log"
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
        with image_stdout.open("w", encoding="utf-8") as out, image_stderr.open("w", encoding="utf-8") as err:
            completed = subprocess.run(image_command, cwd=ROOT, env=env, stdout=out, stderr=err, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"image evaluation failed for {variant}, seed {seed}; see {image_stderr}")
    return variant, seed, checkpoint, evaluation


def _write_manifest(
    args: argparse.Namespace,
    variants: dict[str, dict[str, Path]],
    comparison_names: list[dict[str, str]],
    path: Path,
) -> None:
    # Store absolute artifact paths.  The summary loader still accepts the
    # older repository-relative pilot manifests so completed pilots remain
    # auditable, while new formal manifests are location-independent.
    absolute_variants = {
        variant: {str(seed): str(Path(value).resolve()) for seed, value in members.items()}
        for variant, members in variants.items()
    }
    payload = {
        "schema": "ray-context-survival-owrb-matrix-manifest-v1",
        "experiment": args.experiment_name,
        "mode": args.mode,
        "seeds": args.seeds,
        "variants": absolute_variants,
        "comparisons": comparison_names,
        "data": {
            "datasetDir": str(Path(args.dataset_dir).resolve()),
            "evidenceDir": str(Path(args.evidence_dir).resolve()),
            "runtimeMeta": str(Path(args.runtime_meta).resolve()),
            "modelInputFovYDeg": 66.0,
            "frontendRenderFovYDeg": 60.0,
            "candidatePolicy": "native back-camera CSR; no visible union",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _comparisons(specs: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    values: list[dict[str, str]] = []
    for loss in ("rvl", "safety"):
        for survival in ("off", "on"):
            left = f"context_on_survival_{survival}_{loss}"
            right = f"context_off_survival_{survival}_{loss}"
            if left in specs and right in specs:
                values.append({
                    "name": f"context_on_minus_off_{loss}_survival_{survival}",
                    "left": left,
                    "right": right,
                })
        for context in ("off", "on"):
            left = f"context_{context}_survival_on_{loss}"
            right = f"context_{context}_survival_off_{loss}"
            if left in specs and right in specs:
                values.append({
                    "name": f"survival_on_minus_off_{loss}_context_{context}",
                    "left": left,
                    "right": right,
                })
    for context in ("off", "on"):
        for survival in ("off", "on"):
            left = f"context_{context}_survival_{survival}_safety"
            right = f"context_{context}_survival_{survival}_rvl"
            if left in specs and right in specs:
                values.append({
                    "name": f"safety_minus_rvl_context_{context}_survival_{survival}",
                    "left": left,
                    "right": right,
                })
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--environment", default="slm_pvs")
    parser.add_argument("--dataset-dir", default="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_fov66_v1")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--runtime-meta", default="hkust-v3/assets/runtimeVisibilityMeta.json")
    parser.add_argument("--glb-points", default="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin")
    parser.add_argument("--initial-geo-features", default=str(DEFAULT_INITIAL_GEO))
    parser.add_argument("--glb-index", default="hkust-v3/assets/glbIndex.json")
    parser.add_argument("--glb-root", default="hkust-v3/assets")
    parser.add_argument(
        "--model-root",
        default="neural_instance_culling/model/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811",
    )
    parser.add_argument(
        "--benchmark-root",
        default="neural_instance_culling/benchmark/out/pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811",
    )
    parser.add_argument(
        "--experiment-name",
        default="pvs_direction_depth_relation_survival_constrained_v1_surface_fallback_owrb_formal40_pooled_context_20260811",
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument(
        "--pose-set-batch-size",
        type=int,
        default=8,
        help="number of pose candidate sets per optimization step; formal runs use the same value for every variant",
    )
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--context-pretrain-steps", type=int, default=64)
    parser.add_argument("--calibration-bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.50)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260801])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="pilot-only subset; formal mode always requires all eight registered variants",
    )
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--run-summary", action="store_true")
    parser.add_argument(
        "--run-image-evaluation",
        action="store_true",
        help="run the real component-ID browser image evaluation after each member; formal mode always enables it",
    )
    parser.add_argument("--chrome-exe", default="", help="Chrome executable for formal component-ID image evaluation")
    parser.add_argument("--image-timeout-ms", type=int, default=86_400_000)
    parser.add_argument("--image-max-poses", type=int, default=0, help="diagnostic cap; formal full evaluation uses 0")
    parser.add_argument("--run-test", action="store_true", help="after validation summary/selection, evaluate only the authorized member once on test")
    parser.add_argument("--test-only", action="store_true", help="run the already-written validation-authorized selection on test without training")
    return parser.parse_args()


def _run_one_shot_test(args: argparse.Namespace) -> None:
    selection_path = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_final_selection.json"
    test_output = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_final_test.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"missing validation-only final selection: {selection_path}")
    selection_payload = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection_payload.get("selection")
    if not selected:
        test_output.parent.mkdir(parents=True, exist_ok=True)
        test_output.write_text(json.dumps({
            "schema": "ray-context-survival-owrb-final-test-v1",
            "split": "test", "testRead": False,
            "status": "not_authorized_no_validation_safe_final_member",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "test_not_authorized", "output": str(test_output.resolve()), "testRead": False}, ensure_ascii=False))
        return
    checkpoint = Path(str(selected.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
    command = [
        "conda", "run", "--no-capture-output", "-n", args.environment,
        "python", "-u", str(EVALUATE),
        "--checkpoint", str(checkpoint), "--evidence-dir", args.evidence_dir,
        "--dataset-dir", args.dataset_dir, "--runtime-meta", args.runtime_meta,
        "--glb-index", args.glb_index, "--glb-root", args.glb_root,
        "--split", "test", "--output", str(test_output), "--device", "cuda",
        "--seed", str(selected.get("seed", 20260810)),
    ]
    runtime = selected.get("runtimeFeatures")
    if runtime:
        command.extend(["--runtime-features", str(runtime)])
    test_output.parent.mkdir(parents=True, exist_ok=True)
    if test_output.is_file():
        print(json.dumps({"status": "test_already_exists", "output": str(test_output.resolve()), "testRead": True}, ensure_ascii=False))
        return
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.test_only:
        _run_one_shot_test(args)
        return
    args.initial_geo_features = _validate_fixed_geometry(args)
    specs = _variant_specs(args.mode, args.variants)
    if args.steps_per_epoch == 0:
        if args.mode == "formal":
            from pose_csr_dataset import PoseCSRDataset
            from common.runtime_meta import load_runtime_meta
            num_instances = int(load_runtime_meta(args.runtime_meta)[0].shape[0])
            train_count = int(PoseCSRDataset(args.dataset_dir, num_instances=num_instances).split("train").pose_indices.size)
            args.steps_per_epoch = max(1, int((train_count + max(1, args.pose_set_batch_size) - 1) // max(1, args.pose_set_batch_size)))
        else:
            args.steps_per_epoch = 75
    if args.mode == "formal":
        if len(args.seeds) != 3:
            raise ValueError("formal matrix requires exactly three registered seeds")
        if len(args.gpus) < 4:
            raise ValueError("formal matrix requires four GPU slots")
        args.epochs = 40
        args.steps_per_epoch = max(args.steps_per_epoch, 0)
        if args.bootstrap_replicates < 10000:
            raise ValueError("formal matrix requires at least 10,000 paired bootstrap replicates")
        # The 10,000-resample requirement applies to validation comparisons;
        # calibration only needs an independent frozen safety lower bound.
        args.calibration_bootstrap_replicates = max(args.calibration_bootstrap_replicates, 2000)
        formal_evidence = _validate_formal_evidence(args)
    else:
        formal_evidence = None
    raw_tasks = [(variant, spec, seed) for variant, spec in specs.items() for seed in args.seeds]
    tasks = [(*task, args.gpus[index % len(args.gpus)]) for index, task in enumerate(raw_tasks)]
    results: dict[str, dict[str, Path]] = {variant: {} for variant in specs}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(int(args.jobs), len(tasks))) as executor:
        futures = {executor.submit(_run_member, args, variant, spec, seed, gpu): (variant, seed) for variant, spec, seed, gpu in tasks}
        for future in as_completed(futures):
            variant, seed = futures[future]
            try:
                _variant, _seed, _checkpoint, evaluation = future.result()
                results[variant][str(seed)] = str(evaluation)
                print(json.dumps({"status": "member_complete", "variant": variant, "seed": seed, "evaluation": str(evaluation)}, ensure_ascii=False), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{variant}/seed{seed}: {exc}")
                print(json.dumps({"status": "member_failed", "variant": variant, "seed": seed, "error": str(exc)}, ensure_ascii=False), flush=True)
    if failures:
        raise RuntimeError("matrix failures: " + " | ".join(failures))
    manifest_path = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_matrix_manifest.json"
    _write_manifest(args, results, _comparisons(specs), manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if formal_evidence is not None:
        manifest_payload["data"]["formalEvidenceAudit"] = formal_evidence
        manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    output = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_summary.json"
    report = ROOT / "docs/evaluation" / f"{args.experiment_name}_{args.mode}_validation.md"
    if args.run_summary:
        command = [
            "conda", "run", "--no-capture-output", "-n", args.environment,
            "python", "-u", str(SUMMARY), "--manifest", str(manifest_path), "--output", str(output),
            "--report", str(report), "--bootstrap-replicates", str(args.bootstrap_replicates),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        # Route decisions and final checkpoint authorization require the full
        # registered 2x2x2 factor matrix.  A gamma pilot intentionally has a
        # single safety member and therefore has no factor effects to decide.
        if args.mode == "formal":
            route = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_route.json"
            subprocess.run([
                "conda", "run", "--no-capture-output", "-n", args.environment, "python", "-u", str(DECIDE),
                "--summary", str(output), "--output", str(route),
                "--context-effect", "context_main_effect",
                "--survival-effect", "survival_main_effect",
                "--loss-effect", "loss_main_effect",
            ], cwd=ROOT, check=True)
            selection = Path(args.benchmark_root) / f"{args.experiment_name}_{args.mode}_final_selection.json"
            subprocess.run([
                "conda", "run", "--no-capture-output", "-n", args.environment, "python", "-u", str(SELECT_FINAL),
                "--summary", str(output), "--output", str(selection),
            ], cwd=ROOT, check=True)
    if args.run_test:
        _run_one_shot_test(args)
    print(json.dumps({"status": "matrix_complete", "manifest": str(manifest_path), "variantCount": len(specs), "seedCount": len(args.seeds), "summary": str(output) if args.run_summary else None}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
