#!/usr/bin/env python3
"""Run the formal component-ID image evaluation for one OWRB evaluation.

The set evaluator is the source of truth for the candidate rows and the
calibration-frozen prediction IDs.  This entry point only converts those rows
to the existing real-GLB browser manifest, renders the complete local scene at
the real 60 degree camera FOV, and merges the resulting miss/wrong/extra pixel
statistics back into the evaluation JSON.  It never rebuilds candidates and
never changes a GLB into a visibility unit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runtime_meta import load_runtime_meta  # noqa: E402
from common.candidate_identity import candidate_digest_for_pose_sequence  # noqa: E402
from evaluate_viewcell_image_per import load_glb_aabbs, load_glb_index  # noqa: E402
from instance_id_render_schema import build_instance_binding_preflight  # noqa: E402
from pose_csr_dataset import PoseCSRDataset  # noqa: E402


IMAGE_SCHEMA = "ray-context-survival-owrb-image-evaluation-v1"
FORMAL_MANIFEST_SCHEMA = "local-true-component-id-formal-render-manifest-v1"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _pose_as_sample(
    row: dict[str, Any],
    pose: np.void,
    ordinal: int,
) -> dict[str, Any]:
    camera_view = np.asarray(pose["camera_view"], dtype=np.float32).reshape(-1)
    if camera_view.size != 2:
        raise ValueError("formal OWRB image evaluation requires directional camera_view with two tangents")
    tan_x, tan_y = (float(camera_view[0]), float(camera_view[1]))
    if tan_x <= 0.0 or tan_y <= 0.0:
        raise ValueError("pose camera_view contains a non-positive FOV tangent")
    candidate_ids = [int(value) for value in row.get("candidateIds", [])]
    predicted_ids = [int(value) for value in row.get("predictedIds", [])]
    if not set(predicted_ids).issubset(set(candidate_ids)):
        raise ValueError(f"pose {row.get('poseIndex')} prediction is not a subset of its candidate IDs")
    return {
        "sampleId": f"owrb_pose_{int(row['poseIndex']):06d}_{ordinal:06d}",
        "poseIndex": int(row["poseIndex"]),
        "cameraPosition": [float(value) for value in np.asarray(pose["camera_world"], dtype=np.float32)],
        "cameraForward": [float(value) for value in np.asarray(pose["camera_forward"], dtype=np.float32)],
        "renderFovYDeg": 60.0,
        "modelInputFovYDeg": 66.0,
        "aspect": tan_x / tan_y,
        "predictionComponentIds": predicted_ids,
        "candidateComponentIds": candidate_ids,
        "referenceMode": "full_scene_renderable_instances",
    }


def _build_manifest(
    evaluation: dict[str, Any],
    dataset: PoseCSRDataset,
    runtime_meta_path: Path,
    glb_index: Path,
    glb_root: Path,
    output_dir: Path,
    max_poses: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    world_aabbs, _instance_to_glb, runtime_meta = load_runtime_meta(runtime_meta_path)
    paths, local_check = load_glb_index(glb_index, glb_root)
    if local_check.get("missingLocalFiles", 0) or local_check.get("outsideRootFiles", 0):
        raise ValueError(f"local GLB inventory is incomplete: {local_check}")
    bindings = build_instance_binding_preflight(runtime_meta, glb_index, glb_root)
    glb_aabbs, glb_aabb_meta = load_glb_aabbs(runtime_meta)
    rows = list(evaluation.get("perPose") or [])
    if not rows:
        raise ValueError("evaluation contains no perPose rows")
    if max_poses > 0:
        rows = rows[: int(max_poses)]
    pose_lookup = {int(index): index for index in range(int(dataset.poses.shape[0]))}
    samples: list[dict[str, Any]] = []
    for ordinal, row in enumerate(rows):
        pose_index = int(row["poseIndex"])
        if pose_index not in pose_lookup:
            raise ValueError(f"evaluation pose {pose_index} is outside the dataset")
        samples.append(_pose_as_sample(row, dataset.poses[pose_index], ordinal))
    component_aabbs = {
        str(component_id): {
            "min": [float(value) for value in world_aabbs[component_id, :3]],
            "max": [float(value) for value in world_aabbs[component_id, 3:]],
        }
        for component_id in range(int(world_aabbs.shape[0]))
    }
    manifest = {
        "schema": FORMAL_MANIFEST_SCHEMA,
        "createdBy": IMAGE_SCHEMA,
        "width": 320,
        "height": 180,
        "renderFovYDeg": 60.0,
        "modelInputFovYDeg": 66.0,
        "glbRoot": str(glb_root.resolve()),
        "glbIndex": str(glb_index.resolve()),
        "runtimeMeta": str(runtime_meta_path.resolve()),
        "selectedGlbs": sorted(int(gid) for gid in paths),
        "reference": {
            "mode": "full_scene_renderable_instances",
            "idSource": "componentGlobalId",
            "geometrySource": "original_local_glb_meshes",
            "completeInventory": True,
        },
        "prediction": {
            "field": "predictionComponentIds",
            "postFilter": "component_visibility_mask_after_conservative_render_submission",
        },
        "instanceBindings": bindings,
        "glbAabbs": {
            str(global_id): {
                "min": [float(value) for value in glb_aabbs[global_id, :3]],
                "max": [float(value) for value in glb_aabbs[global_id, 3:]],
            }
            for global_id in sorted(int(gid) for gid in paths)
            if np.any(glb_aabbs[global_id, 3:] > glb_aabbs[global_id, :3])
        },
        "componentAabbs": component_aabbs,
        "spatialCulling": {
            "schema": "aabb-frustum-conservative-v1",
            "source": "runtimeMeta.componentRecords[].bounds",
            "purpose": "render_submission_only",
            "renderFovYDeg": 60.0,
            "near": 0.05,
            "far": 20000.0,
            "completeInventoryRetained": True,
            "componentLevelMask": True,
        },
        "samples": samples,
        "previewSamples": 0,
        "saveIdBuffers": False,
        "localOnly": True,
        "idEncoding": "componentGlobalId + 1, RGB24, 0 background",
        "formalImageEvaluationReady": True,
        "formalRequirements": {
            "requiresHardwareWebGL": True,
            "syntheticSmokeAllowed": False,
            "completeGlbInventory": True,
            "renderFovYDeg": 60.0,
            "modelInputFovYDeg": 66.0,
        },
        "evaluationIdentity": {
            "checkpoint": evaluation.get("checkpoint"),
            "checkpointSha256": evaluation.get("checkpointSha256"),
            "split": evaluation.get("split"),
            "candidateDigest": evaluation.get("candidateDigest"),
            "poseCount": len(samples),
            "poseOrder": [int(sample["poseIndex"]) for sample in samples],
            "candidateHashDigest": candidate_digest_for_pose_sequence(
                dataset, [int(sample["poseIndex"]) for sample in samples]
            ),
        },
        "localInventoryAudit": local_check,
        "glbAabbAudit": glb_aabb_meta,
        "outputDir": str(output_dir.resolve()),
    }
    return manifest, samples


def _merge_browser_metrics(
    evaluation_path: Path,
    evaluation: dict[str, Any],
    image_output: Path,
    manifest_path: Path,
    render_summary_path: Path,
    sample_metrics_path: Path,
    samples: list[dict[str, Any]],
) -> dict[str, Any]:
    render_summary = _read(render_summary_path)
    sample_rows = _read(sample_metrics_path)
    if not isinstance(sample_rows, list) or len(sample_rows) != len(samples):
        raise ValueError("browser sample image result count does not match the formal manifest")
    by_pose = {int(row["poseIndex"]): row for row in sample_rows}
    if len(by_pose) != len(samples):
        raise ValueError("browser image results contain duplicate pose indices")
    evaluation_rows = list(evaluation.get("perPose") or [])
    if len(evaluation_rows) < len(samples):
        raise ValueError("evaluation has fewer set rows than rendered image samples")
    for row, sample in zip(evaluation_rows, samples, strict=False):
        pose_index = int(row["poseIndex"])
        if pose_index != int(sample["poseIndex"]):
            raise ValueError("image sample pose order differs from the set evaluation")
        browser_row = by_pose.get(pose_index)
        if browser_row is None or not isinstance(browser_row.get("imageMetrics"), dict):
            raise ValueError(f"browser produced no image metrics for pose {pose_index}")
        row["imageMetrics"] = browser_row["imageMetrics"]
    image_metrics = dict(render_summary.get("imageMetrics") or {})
    image_metrics.update({
        "status": "formal_ready" if render_summary.get("formalImageEvaluationReady") is True else "failed_formal_gate",
        "schema": "color-id-per-v1-aggregate",
        "rendererSummary": str(render_summary_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "gpuGate": render_summary.get("gpuGate"),
        "perPoseAvailable": True,
    })
    payload = {
        "schema": IMAGE_SCHEMA,
        "evaluation": str(evaluation_path.resolve()),
        "imageMetrics": image_metrics,
        "perPose": [
            {"poseIndex": int(row["poseIndex"]), "imageMetrics": row["imageMetrics"]}
            for row in evaluation_rows[: len(samples)]
        ],
        "rendererSummary": str(render_summary_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "sampleMetrics": str(sample_metrics_path.resolve()),
        "formalImageEvaluationReady": bool(render_summary.get("formalImageEvaluationReady") is True),
        "gpuGate": render_summary.get("gpuGate"),
    }
    if not payload["formalImageEvaluationReady"]:
        raise RuntimeError("browser image evaluation did not pass formalImageEvaluationReady")
    evaluation["imageMetrics"] = image_metrics
    evaluation["imageEvaluation"] = payload
    evaluation["perPose"] = evaluation_rows
    _write(evaluation_path, evaluation)
    _write(image_output, payload)
    return payload


def run(args: argparse.Namespace) -> dict[str, Any]:
    evaluation_path = args.evaluation.resolve()
    evaluation = _read(evaluation_path)
    if evaluation.get("schema") != "ray-context-survival-owrb-evaluation-v1":
        raise ValueError("unexpected OWRB evaluation schema")
    if evaluation.get("split") != "validation":
        raise ValueError("formal OWRB image evaluation is validation-only")
    if evaluation.get("thresholdSource", {}).get("protocol") != "calibration_ready_pre_test":
        raise ValueError("image evaluation requires a calibration-frozen threshold")
    if int(evaluation.get("thresholdSource", {}).get("testEvaluationCount", -1)) != 0:
        raise ValueError("image evaluation cannot use a threshold selected after test")
    if not all("candidateIds" in row and "predictedIds" in row for row in evaluation.get("perPose", [])):
        raise ValueError("set evaluation must be run with --persist-ids before image evaluation")

    world_aabbs, _instance_to_glb, _runtime_meta = load_runtime_meta(args.runtime_meta)
    dataset = PoseCSRDataset(args.dataset_dir, num_instances=int(world_aabbs.shape[0]))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest, samples = _build_manifest(
        evaluation, dataset, args.runtime_meta, args.glb_index, args.glb_root,
        output_dir, args.max_poses,
    )
    manifest_path = output_dir / "component_id_formal_render_manifest.json"
    _write(manifest_path, manifest)
    render_dir = output_dir / "browser_render"
    command = [
        "node", str(args.renderer_script),
        "--manifest", str(manifest_path),
        "--output-dir", str(render_dir),
        "--timeout-ms", str(args.render_timeout_ms),
        "--require-hardware-gpu",
    ]
    if args.chrome_exe:
        command.extend(["--chrome-exe", str(args.chrome_exe)])
    stdout_path = output_dir / "browser_stdout.log"
    stderr_path = output_dir / "browser_stderr.log"
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=ROOT.parent, stdout=stdout, stderr=stderr, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"formal component-ID browser render failed; see {stderr_path}")
    render_summary_path = render_dir / "render_summary.json"
    sample_metrics_path = render_dir / "sample_image_metrics.json"
    if not render_summary_path.is_file() or not sample_metrics_path.is_file():
        raise FileNotFoundError("browser render did not produce render_summary.json and sample_image_metrics.json")
    return _merge_browser_metrics(
        evaluation_path,
        evaluation,
        output_dir / "image_evaluation.json",
        manifest_path,
        render_summary_path,
        sample_metrics_path,
        samples,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--runtime-meta", type=Path, required=True)
    parser.add_argument("--glb-index", type=Path, required=True)
    parser.add_argument("--glb-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-script", type=Path, default=ROOT / "benchmark/render_local_glb_color_id_browser.mjs")
    parser.add_argument("--chrome-exe", type=Path, default=None)
    parser.add_argument("--max-poses", type=int, default=0)
    parser.add_argument("--render-timeout-ms", type=int, default=86_400_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_poses < 0:
        raise ValueError("--max-poses must be non-negative")
    payload = run(args)
    print(json.dumps({
        "status": "formal_image_evaluated",
        "output": str((args.output_dir / "image_evaluation.json").resolve()),
        "poseCount": len(payload["perPose"]),
        "missPixelRate": payload["imageMetrics"].get("missPixelRate"),
        "p95MissPixelRate": payload["imageMetrics"].get("p95MissPixelRate"),
        "testRead": False,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
