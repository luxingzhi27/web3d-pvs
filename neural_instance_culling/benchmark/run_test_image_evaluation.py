#!/usr/bin/env python3
"""Run one frozen-test formal-v2 image manifest through the existing renderer."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .instance_id_render_schema import (  # type: ignore[import-not-found]
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
    )
except ImportError:
    from instance_id_render_schema import (  # type: ignore[no-redef]
        FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RENDERER = ROOT / "benchmark" / "render_local_glb_color_id_browser.mjs"
GPU_EVIDENCE_SCHEMA = "pvs-browser-hardware-gpu-evidence-v1"
HZB_RESULT_SCHEMA = "geometry-shell-hzb-browser-result-v2"
SOFTWARE_GPU_PATTERN = re.compile(
    r"swiftshader|llvmpipe|softpipe|swrast|software|no-webgl",
    re.IGNORECASE,
)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"missing JSON input: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON input {path}: {error}") from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _capture_gpu_snapshot(evidence_dir: Path, phase: str) -> dict[str, Any]:
    """Capture both host GPU commands into one formal render window."""

    evidence_dir.mkdir(parents=True, exist_ok=True)
    commands = {
        "nvidiaSmi": (
            ["nvidia-smi"],
            evidence_dir / f"nvidia_smi_{phase}.txt",
        ),
        "nvidiaSmiPmon": (
            ["nvidia-smi", "pmon", "-c", "1", "-s", "um"],
            evidence_dir / f"nvidia_smi_pmon_{phase}.txt",
        ),
    }
    snapshot: dict[str, Any] = {
        "schema": "pvs-gpu-host-snapshot-v1",
        "phase": phase,
        "capturedAt": _utc_now(),
        "complete": True,
        "observations": {},
    }
    for name, (command, path) in commands.items():
        try:
            completed = subprocess.run(
                command,
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            path.write_text(output, encoding="utf-8")
            available = completed.returncode == 0 and bool(output.strip())
            observation: dict[str, Any] = {
                "command": command,
                "path": str(path.resolve()),
                "returnCode": int(completed.returncode),
                "available": bool(available),
            }
            if name == "nvidiaSmiPmon":
                observation["sampleCount"] = sum(
                    1
                    for line in output.splitlines()
                    if re.match(r"^\s*\d+\s+\S+", line)
                )
            snapshot["observations"][name] = observation
            snapshot["complete"] = bool(snapshot["complete"] and available)
        except (OSError, subprocess.SubprocessError) as error:
            path.write_text(str(error), encoding="utf-8")
            snapshot["observations"][name] = {
                "command": command,
                "path": str(path.resolve()),
                "returnCode": None,
                "available": False,
                "error": str(error),
            }
            snapshot["complete"] = False
    return snapshot


def _classify_gpu_backend(backend: Any) -> dict[str, Any]:
    values = backend if isinstance(backend, dict) else {}
    text = " ".join(str(values.get(key) or "") for key in ("vendor", "renderer", "version"))
    marker = SOFTWARE_GPU_PATTERN.search(text)
    return {
        "vendor": str(values.get("vendor") or ""),
        "renderer": str(values.get("renderer") or ""),
        "version": str(values.get("version") or ""),
        "hardware": bool(text.strip()) and marker is None,
        "softwareMarkers": marker.group(0) if marker else None,
    }


def _host_evidence_complete(evidence: dict[str, Any]) -> bool:
    phases = evidence.get("phases")
    if not isinstance(phases, dict) or set(phases) != {"before", "during", "after"}:
        return False
    for phase_name, phase in phases.items():
        if not isinstance(phase, dict) or phase.get("complete") is not True:
            return False
        observations = phase.get("observations")
        if not isinstance(observations, dict):
            return False
        if any(
            not isinstance(observations.get(name), dict)
            or observations[name].get("available") is not True
            for name in ("nvidiaSmi", "nvidiaSmiPmon")
        ):
            return False
        if phase_name == "during" and phase.get("rendererProcessAlive") is not True:
            return False
        try:
            sample_count = int(observations["nvidiaSmiPmon"].get("sampleCount", 0))
        except (TypeError, ValueError):
            return False
        if sample_count <= 0:
            return False
    return True


def _validate_formal_render_summary(
    summary: dict[str, Any], manifest: dict[str, Any], gpu_evidence: dict[str, Any]
) -> None:
    if summary.get("error"):
        raise RuntimeError(f"formal image renderer reported an error: {summary['error']}")
    if summary.get("formalImageEvaluationReady") is not True:
        raise RuntimeError("formal image renderer did not report formalImageEvaluationReady=true")
    if "formalReady" in summary and summary.get("formalReady") is not True:
        raise RuntimeError("formal image renderer reported formalReady=false")
    if summary.get("componentIdShaderImplemented") is not True:
        raise RuntimeError("formal image renderer did not report component-ID shader completion")
    if summary.get("renderStatus") != "rendered_component_id_buffers":
        raise RuntimeError("formal image renderer did not report complete component-ID buffers")
    if int(summary.get("sampleCount", -1)) != len(manifest.get("samples") or []):
        raise RuntimeError("formal image renderer did not evaluate every manifest sample")

    backend = _classify_gpu_backend(summary.get("gpuBackend"))
    raw_backend = summary.get("gpuBackend")
    if (
        not isinstance(raw_backend, dict)
        or raw_backend.get("api") != "WebGL"
        or not backend["vendor"]
        or not backend["renderer"]
        or backend["hardware"] is not True
    ):
        raise RuntimeError(
            "formal image evaluation requires a hardware WebGL backend; "
            f"reported renderer={backend['renderer'] or 'missing'}"
        )
    gate = summary.get("gpuGate")
    if (
        not isinstance(gate, dict)
        or gate.get("required") is not True
        or gate.get("hardware") is not True
    ):
        raise RuntimeError("formal image renderer did not pass its hardware GPU gate")
    if gpu_evidence.get("complete") is not True or not _host_evidence_complete(gpu_evidence):
        raise RuntimeError(
            "formal image evaluation requires complete before/during/after nvidia-smi/pmon evidence"
        )


def run_formal_renderer(
    command: list[str],
    output_dir: Path,
    manifest: dict[str, Any],
    timeout_ms: int | None,
) -> dict[str, Any]:
    """Run the browser and attach host evidence from this exact process window."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "render_summary.json"
    summary_path.unlink(missing_ok=True)
    evidence_dir = output_dir / "gpu_evidence"
    evidence: dict[str, Any] = {
        "schema": GPU_EVIDENCE_SCHEMA,
        "required": True,
        "formalReady": False,
        "command": command,
        "phases": {},
        "startedAt": _utc_now(),
    }
    evidence["phases"]["before"] = _capture_gpu_snapshot(evidence_dir, "before")
    process: subprocess.Popen[str] | None = None
    timed_out = False
    process_error: Exception | None = None
    stdout_log = (output_dir / "renderer_stdout.log").open("w", encoding="utf-8")
    stderr_log = (output_dir / "renderer_stderr.log").open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT.parent,
            stdout=stdout_log,
            stderr=stderr_log,
            text=True,
        )
        evidence["pid"] = int(process.pid)
        # A formal render is long enough to observe the browser while this
        # Node process is still alive. The alive bit makes the window auditable.
        time.sleep(0.5)
        during = _capture_gpu_snapshot(evidence_dir, "during")
        during["rendererProcessAlive"] = process.poll() is None
        during["complete"] = bool(during["complete"] and during["rendererProcessAlive"])
        evidence["phases"]["during"] = during
        communicate_timeout = None if timeout_ms is None else max(1.0, timeout_ms / 1000.0 + 10.0)
        try:
            process.communicate(timeout=communicate_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
    except Exception as error:
        process_error = error
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        stdout_log.close()
        stderr_log.close()
        evidence["phases"]["after"] = _capture_gpu_snapshot(evidence_dir, "after")
        evidence["finishedAt"] = _utc_now()
        evidence["returnCode"] = int(process.returncode) if process is not None and process.returncode is not None else None
        evidence["timedOut"] = timed_out
        evidence["phaseComplete"] = {
            phase: bool(
                details.get("complete", False)
                and (phase != "during" or details.get("rendererProcessAlive") is True)
            )
            for phase, details in evidence["phases"].items()
            if isinstance(details, dict)
        }
        evidence["complete"] = _host_evidence_complete(evidence)
        for alias, phase in (
            ("hostGpuBefore", "before"),
            ("hostGpuDuring", "during"),
            ("hostGpuAfter", "after"),
        ):
            snapshot = evidence["phases"].get(phase) or {}
            evidence[alias] = {
                "schema": snapshot.get("schema"),
                "phase": phase,
                "capturedAt": snapshot.get("capturedAt"),
                "complete": snapshot.get("complete") is True,
                "nvidiaSmi": (snapshot.get("observations") or {}).get("nvidiaSmi"),
                "nvidiaSmiPmon": (snapshot.get("observations") or {}).get("nvidiaSmiPmon"),
            }
        evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
        (evidence_dir / "hardware_gpu_evidence.json").write_text(evidence_json, encoding="utf-8")
        (output_dir / "gpu_evidence.json").write_text(evidence_json, encoding="utf-8")

    if summary_path.is_file():
        summary_value = read_json(summary_path)
        if not isinstance(summary_value, dict):
            raise ValueError(f"renderer summary must be an object: {summary_path}")
        summary = summary_value
        summary["hardwareEvidence"] = evidence
        gate = summary.get("gpuGate")
        if isinstance(gate, dict):
            summary["gpuGate"] = {
                **gate,
                "hostEvidenceRequired": True,
                "hostEvidenceComplete": bool(evidence["complete"]),
            }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        summary = None

    if process_error is not None:
        raise RuntimeError(f"failed to run formal image renderer: {process_error}") from process_error
    if timed_out or process is None or process.returncode != 0:
        raise RuntimeError(
            "formal image renderer failed; see renderer_stdout.log and renderer_stderr.log "
            f"(return code: {None if process is None else process.returncode})"
        )
    if summary is None:
        raise FileNotFoundError(f"formal image renderer did not write {summary_path}")
    _validate_formal_render_summary(summary, manifest, evidence)
    evidence["formalReady"] = True
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
    (evidence_dir / "hardware_gpu_evidence.json").write_text(evidence_json, encoding="utf-8")
    (output_dir / "gpu_evidence.json").write_text(evidence_json, encoding="utf-8")
    summary["hardwareEvidence"] = evidence
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def validate_test_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Check only frozen-test fields; the renderer owns the full schema check."""
    if manifest.get("schema") != FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA:
        raise ValueError(
            "test image input must use "
            f"{FORMAL_INSTANCE_RENDER_MANIFEST_SCHEMA}"
        )
    if manifest.get("split") != "test":
        raise ValueError("formal test image manifest must declare split=test")
    if manifest.get("testRead") is not True:
        raise ValueError("formal test image manifest must declare testRead=true")
    if manifest.get("testEvaluationCount") != 1:
        raise ValueError("formal test image manifest must declare testEvaluationCount=1")

    has_threshold = "thresholdSelection" in manifest
    has_baseline = "baselineSelection" in manifest
    if has_threshold == has_baseline:
        raise ValueError(
            "formal test image manifest must contain exactly one of "
            "thresholdSelection or baselineSelection"
        )
    if has_threshold:
        threshold = manifest.get("thresholdSelection")
        if not isinstance(threshold, dict):
            raise ValueError("formal test image manifest is missing thresholdSelection")
        if threshold.get("selectionSplit") != "calibration":
            raise ValueError("formal test image threshold must come from calibration")
        if threshold.get("testRead") is not False:
            raise ValueError("calibration threshold provenance must be recorded before test")
        provenance = manifest.get("thresholdProvenance")
        if not isinstance(provenance, dict):
            raise ValueError("formal test image manifest is missing thresholdProvenance")
        if provenance.get("selectionSplit") != "calibration" or provenance.get("testRead") is not False:
            raise ValueError("thresholdProvenance must identify pre-test calibration")
        selection_method = "threshold"
    else:
        if any(field in manifest for field in ("threshold", "thresholdProvenance")):
            raise ValueError("baseline test image manifest cannot contain neural threshold fields")
        baseline = manifest.get("baselineSelection")
        if not isinstance(baseline, dict):
            raise ValueError("formal test image manifest is missing baselineSelection")
        if baseline.get("method") != "geometry-shell-hzb":
            raise ValueError("formal test image baseline must use geometry-shell-hzb")
        if baseline.get("selectionSplit") != "calibration":
            raise ValueError("formal test image baseline must come from calibration")
        if baseline.get("testRead") is not False:
            raise ValueError("calibration baseline provenance must be recorded before test")
        for field in ("assetVariant", "resolution", "depthBiasM", "regionSampleCount", "sourceResult"):
            if field not in baseline:
                raise ValueError(f"formal test image baseline is missing {field}")
        if not isinstance(baseline["assetVariant"], str) or not baseline["assetVariant"]:
            raise ValueError("formal test image baseline assetVariant must be non-empty")
        resolution = baseline["resolution"]
        if isinstance(resolution, dict):
            resolution = [resolution.get("width"), resolution.get("height")]
        if (
            not isinstance(resolution, (list, tuple))
            or len(resolution) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolution)
        ):
            raise ValueError("formal test image baseline resolution must contain positive width and height")
        try:
            depth_bias = float(baseline["depthBiasM"])
        except (TypeError, ValueError) as error:
            raise ValueError("formal test image baseline depthBiasM must be finite and non-negative") from error
        if not math.isfinite(depth_bias) or depth_bias < 0:
            raise ValueError("formal test image baseline depthBiasM must be finite and non-negative")
        if (
            isinstance(baseline["regionSampleCount"], bool)
            or not isinstance(baseline["regionSampleCount"], int)
            or baseline["regionSampleCount"] < 0
        ):
            raise ValueError("formal test image baseline regionSampleCount must be non-negative")
        source_result = baseline["sourceResult"]
        if not isinstance(source_result, str) or not source_result:
            raise ValueError("formal test image baseline sourceResult must be non-empty")
        source_path = Path(source_result).expanduser()
        if not source_path.is_absolute():
            raise ValueError("formal test image baseline sourceResult must be an absolute path")
        try:
            source_payload = read_json(source_path)
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                "formal test image baseline sourceResult must point to parseable JSON"
            ) from error
        if (
            not isinstance(source_payload, dict)
            or source_payload.get("schema") != HZB_RESULT_SCHEMA
            or source_payload.get("mode") != "Region66"
            or source_payload.get("formalReady") is not True
            or source_payload.get("executionClass") != "formal-hardware-gpu"
            or not isinstance(source_payload.get("gpuGate"), dict)
            or source_payload["gpuGate"].get("required") is not True
            or source_payload["gpuGate"].get("hardware") is not True
        ):
            raise ValueError(
                "formal test image baseline sourceResult must be a formal Region66 HZB result"
            )
        selection_method = "geometry-shell-hzb"

    coverage = manifest.get("testCoverage")
    if not isinstance(coverage, dict):
        raise ValueError("formal test image manifest is missing testCoverage")
    if coverage.get("split") != "test":
        raise ValueError("testCoverage must declare split=test")
    if coverage.get("selection") != "all_unique_test_viewcells":
        raise ValueError("formal test image evaluation must use all unique test view-cells")
    if coverage.get("sampledWithReplacement") is not False:
        raise ValueError("formal test image evaluation cannot sample with replacement")
    if coverage.get("maxViewcells") != 0:
        raise ValueError("formal test image evaluation cannot truncate view-cells")
    if coverage.get("subposesPerViewcell") != 0:
        raise ValueError("formal test image evaluation must use all dense subposes")

    subposes = manifest.get("subposeSelection")
    if not isinstance(subposes, dict) or subposes.get("mode") != "all":
        raise ValueError("formal test image manifest must select all dense subposes")
    if subposes.get("requestedPerViewcell") != 0:
        raise ValueError("formal test image manifest cannot select a subpose subset")
    return {
        "schema": "pvs-frozen-test-image-selection-v1",
        "manifestSchema": manifest["schema"],
        "split": "test",
        "testRead": True,
        "testEvaluationCount": 1,
        "selectionSplit": "calibration",
        "selectionMethod": selection_method,
        "viewcellCount": coverage.get("viewcellCount"),
        "sampleCount": coverage.get("sampleCount"),
    }


def renderer_command(
    manifest: Path,
    output_dir: Path,
    renderer: Path,
    chrome_exe: Path | None,
    schema_only: bool,
    timeout_ms: int | None,
) -> list[str]:
    command = [
        "node",
        str(renderer.resolve()),
        "--manifest",
        str(manifest.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--require-hardware-gpu",
    ]
    if schema_only:
        command.append("--validate-only")
    if timeout_ms is not None:
        command.extend(["--timeout-ms", str(timeout_ms)])
    if chrome_exe is not None:
        command.extend(["--chrome-exe", str(chrome_exe.resolve())])
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--renderer-script", type=Path, default=DEFAULT_RENDERER)
    parser.add_argument("--chrome-exe", type=Path, default=None)
    parser.add_argument("--timeout-ms", type=int, default=None)
    parser.add_argument(
        "--require-hardware-gpu",
        action="store_true",
        help="pass the formal renderer's hardware-GPU gate",
    )
    parser.add_argument(
        "--render-schema-only",
        action="store_true",
        help="run the existing renderer validator without starting Chrome",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.timeout_ms is not None and args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if not args.require_hardware_gpu:
        raise ValueError("formal test image evaluation requires --require-hardware-gpu")
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"formal test image manifest must be an object: {manifest_path}")
    validation = validate_test_manifest(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = renderer_command(
        manifest_path,
        output_dir,
        args.renderer_script,
        args.chrome_exe,
        args.render_schema_only,
        args.timeout_ms,
    )
    print(
        json.dumps(
            {"manifest": str(manifest_path), "validation": validation, "command": command},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.render_schema_only:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"formal image renderer failed with exit code {result.returncode}")
        return
    run_formal_renderer(command, output_dir, manifest, args.timeout_ms)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"run_test_image_evaluation: {error}", file=sys.stderr)
        raise
