#!/usr/bin/env python3
"""Run bounded-size hardware triangle-depth manifests with bounded concurrency."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / "neural_instance_culling/benchmark/build_triangle_depth_layer_evidence_browser.mjs"
COMPACTOR = ROOT / "neural_instance_culling/dataset/compact_triangle_depth_relation_shard.py"
FORMAL_EVIDENCE_SCHEMA = "triangle-depth-layer-gpu-evidence-v1"
SPARSE_SCHEMA = "triangle-depth-relation-sparse-shard-v1"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def completed_formal_cache(output_dir: Path) -> bool:
    meta_path = output_dir / "layer_cache_meta.json"
    evidence_path = output_dir / "triangle_depth.bin.gpu_evidence.json"
    if not meta_path.is_file() or not evidence_path.is_file():
        return False
    meta = read_json(meta_path)
    evidence = read_json(evidence_path)
    files = meta.get("files") or {}
    payloads = [
        output_dir / str(files.get("instanceIds", "")),
        output_dir / str(files.get("linearDepth", "")),
    ]
    return bool(
        meta.get("gpuGate", {}).get("hardware") is True
        and evidence.get("schema") == FORMAL_EVIDENCE_SCHEMA
        and evidence.get("formalReady") is True
        and evidence.get("gpuGate", {}).get("hardware") is True
        and all(path.is_file() and path.stat().st_size > 0 for path in payloads)
    )


def valid_sparse_directory(sparse_dir: Path) -> bool:
    meta_path = sparse_dir / "sparse_relation_meta.json"
    if not meta_path.is_file():
        return False
    meta = read_json(meta_path)
    files = meta.get("files") or {}
    required = [
        sparse_dir / str(files.get("relationMoments", "")),
        sparse_dir / str(files.get("relationDepthOffsets", "")),
        sparse_dir / str(files.get("relationDepthValues", "")),
        sparse_dir / str(files.get("survivalObservations", "")),
        sparse_dir / str(files.get("renderPoseIds", "")),
        sparse_dir / str(files.get("sourcePoseIndices", "")),
    ]
    return bool(
        meta.get("schema") == SPARSE_SCHEMA
        and meta.get("formalReady") is True
        and meta.get("trainOnly") is True
        and int(meta.get("poseCount", 0)) > 0
        and all(path.is_file() for path in required)
    )


def completed_sparse_cache(output_dir: Path) -> bool:
    return valid_sparse_directory(output_dir / "sparse")


def dense_payload_paths(output_dir: Path) -> list[Path]:
    meta_path = output_dir / "layer_cache_meta.json"
    if not meta_path.is_file():
        return []
    files = read_json(meta_path).get("files") or {}
    paths = []
    for key in ("instanceIds", "linearDepth"):
        name = files.get(key)
        if name:
            paths.append(output_dir / str(name))
    return paths


def remove_dense_payload(output_dir: Path) -> int:
    if not completed_sparse_cache(output_dir):
        raise ValueError("dense payload may only be removed after sparse validation")
    removed = 0
    for path in dense_payload_paths(output_dir):
        if path.is_file():
            removed += path.stat().st_size
            path.unlink()
    return removed


def command_for_manifest(
    manifest: Path,
    output_dir: Path,
    chrome_exe: Path,
    width: int,
    height: int,
    max_layers: int,
    timeout_ms: int,
    require_hardware_gpu: bool,
    resume_existing: bool,
) -> list[str]:
    command = [
        "node",
        str(RENDERER),
        "--manifest",
        str(manifest),
        "--output",
        str(output_dir / "triangle_depth.bin"),
        "--chrome-exe",
        str(chrome_exe),
        "--width",
        str(width),
        "--height",
        str(height),
        "--max-layers",
        str(max_layers),
        "--timeout-ms",
        str(timeout_ms),
        "--require-hardware-gpu" if require_hardware_gpu else "--allow-software-gpu",
    ]
    partial_index = output_dir / "triangle_depth.bin.pose_indices.partial"
    if resume_existing and partial_index.is_file() and partial_index.stat().st_size > 0:
        command.append("--resume")
    return command


def run_one(
    manifest: Path,
    output_dir: Path,
    command: list[str],
    environment: dict[str, str],
    *,
    compact_only: bool,
    keep_dense_layers: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    render_seconds = 0.0
    render_return_code = 0
    if not completed_formal_cache(output_dir):
        if compact_only:
            raise FileNotFoundError(f"compact-only requires a complete dense shard: {output_dir}")
        render_started = time.time()
        with (output_dir / "run_stdout.log").open("w", encoding="utf-8") as stdout, (
            output_dir / "run_stderr.log"
        ).open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        render_return_code = int(result.returncode)
        render_seconds = float(time.time() - render_started)
        if render_return_code != 0 or not completed_formal_cache(output_dir):
            return {
                "manifest": str(manifest.resolve()),
                "outputDir": str(output_dir.resolve()),
                "returnCode": render_return_code or 1,
                "elapsedSeconds": float(time.time() - started),
                "renderElapsedSeconds": render_seconds,
                "compactElapsedSeconds": 0.0,
                "formalReady": False,
                "stage": "render",
            }

    compact_started = time.time()
    sparse_dir = output_dir / "sparse"
    pending_sparse_dir = output_dir / ".sparse.pending"
    if pending_sparse_dir.exists():
        shutil.rmtree(pending_sparse_dir)
    compact_command = [
        sys.executable,
        str(COMPACTOR),
        "--cache-dir",
        str(output_dir),
        "--output-dir",
        str(pending_sparse_dir),
    ]
    with (output_dir / "compact_stdout.log").open("w", encoding="utf-8") as stdout, (
        output_dir / "compact_stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        compact_result = subprocess.run(
            compact_command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    compact_seconds = float(time.time() - compact_started)
    sparse_ready = int(compact_result.returncode) == 0 and valid_sparse_directory(
        pending_sparse_dir
    )
    if sparse_ready:
        if sparse_dir.exists():
            shutil.rmtree(sparse_dir)
        pending_sparse_dir.replace(sparse_dir)
        sparse_ready = completed_sparse_cache(output_dir)
    removed_dense_bytes = 0
    if sparse_ready and not keep_dense_layers:
        removed_dense_bytes = remove_dense_payload(output_dir)
    return {
        "manifest": str(manifest.resolve()),
        "outputDir": str(output_dir.resolve()),
        "returnCode": 0 if sparse_ready else int(compact_result.returncode or 1),
        "elapsedSeconds": float(time.time() - started),
        "renderElapsedSeconds": render_seconds,
        "compactElapsedSeconds": compact_seconds,
        "formalReady": sparse_ready,
        "stage": "complete" if sparse_ready else "compact",
        "removedDenseBytes": int(removed_dense_bytes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--chrome-exe", type=Path, default=Path("/usr/bin/google-chrome"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--max-layers", type=int, default=6)
    parser.add_argument("--timeout-ms", type=int, default=3_600_000)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--allow-software-gpu", action="store_true")
    parser.add_argument("--require-hardware-gpu", action="store_true")
    parser.add_argument(
        "--compact-only",
        action="store_true",
        help="Convert existing complete dense shards without launching Chrome.",
    )
    parser.add_argument(
        "--keep-dense-layers",
        action="store_true",
        help="Retain dense ID/depth payloads after verified sparse compaction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if args.allow_software_gpu and args.require_hardware_gpu:
        raise ValueError("choose either --require-hardware-gpu or --allow-software-gpu")
    manifests = sorted(args.manifest_dir.resolve().glob("shard_*.json"))
    if not manifests:
        raise FileNotFoundError(f"no shard_*.json manifests in {args.manifest_dir}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    numeric_threads = max(1, int(os.cpu_count() or 1) // int(args.jobs))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = str(numeric_threads)
    nvidia_icd = Path("/etc/vulkan/icd.d/nvidia_icd.json")
    if nvidia_icd.is_file():
        environment["VK_ICD_FILENAMES"] = str(nvidia_icd)

    rows: list[dict[str, Any]] = []
    jobs: list[tuple[Path, Path, list[str]]] = []
    for manifest in manifests:
        output_dir = args.output_root / manifest.stem
        if completed_sparse_cache(output_dir):
            removed_dense_bytes = (
                0 if args.keep_dense_layers else remove_dense_payload(output_dir)
            )
            rows.append(
                {
                    "manifest": str(manifest.resolve()),
                    "outputDir": str(output_dir.resolve()),
                    "returnCode": 0,
                    "elapsedSeconds": 0.0,
                    "formalReady": True,
                    "skipped": True,
                    "removedDenseBytes": int(removed_dense_bytes),
                }
            )
            continue
        jobs.append(
            (
                manifest,
                output_dir,
                command_for_manifest(
                    manifest,
                    output_dir,
                    args.chrome_exe.resolve(),
                    args.width,
                    args.height,
                    args.max_layers,
                    args.timeout_ms,
                    not args.allow_software_gpu,
                    args.resume_existing,
                ),
            )
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_one,
                manifest,
                output_dir,
                command,
                environment,
                compact_only=args.compact_only,
                keep_dense_layers=args.keep_dense_layers,
            ): manifest
            for manifest, output_dir, command in jobs
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    rows.sort(key=lambda row: row["manifest"])
    failed = [row for row in rows if row["returnCode"] != 0 or not row["formalReady"]]
    summary = {
        "schema": "triangle-depth-sparse-relation-shard-run-summary-v1",
        "manifestDir": str(args.manifest_dir.resolve()),
        "outputRoot": str(args.output_root.resolve()),
        "renderer": str(RENDERER.resolve()),
        "jobs": int(args.jobs),
        "requireHardwareGpu": not args.allow_software_gpu,
        "compactOnly": bool(args.compact_only),
        "keepDenseLayers": bool(args.keep_dense_layers),
        "numericThreadsPerJob": int(numeric_threads),
        "shardCount": len(rows),
        "completedCount": len(rows) - len(failed),
        "failedCount": len(failed),
        "formalReady": not failed and bool(rows),
        "rows": rows,
    }
    (args.output_root / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if failed:
        names = [Path(row["manifest"]).name for row in failed]
        raise RuntimeError(f"triangle-depth shard failures: {names}")
    print(json.dumps({key: summary[key] for key in ("shardCount", "completedCount", "formalReady")}, indent=2))


if __name__ == "__main__":
    main()
