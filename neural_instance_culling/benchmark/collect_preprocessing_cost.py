#!/usr/bin/env python3
"""Read the registered paper-scene preprocessing evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
UNAVAILABLE = "unavailable"
SCHEMA = "pvs-preprocessing-cost-v1"
STAGES = ("sampling", "relation", "fixed_geometry", "training", "calibration", "export")
COST_FIELDS = (
    "scene", "stage", "status", "timeSeconds", "timeAggregation",
    "timeRecordSelection", "timeCompleteness", "timeScope", "timeEvidenceCount",
    "timeSource", "device", "deviceSource", "outputBytes", "outputMiB",
    "outputFileCount", "outputSource", "peakVramMiB", "peakVramSource",
    "peakCpuRamMiB", "peakCpuRamSource", "sourcePaths", "missingSources",
    "unavailableReasons", "notes",
)

SEEDS = ("20260801", "20260802", "20260803")
HKUST_MODEL = "neural_instance_culling/model/out/pvs_v4_integrated_visibility_mainline_v1"
IFC_MODEL = "neural_instance_culling/model/out/pvs_mainline_v4_ifcbench_fantasy_metropolis_v1"
IFC_BENCH = "neural_instance_culling/benchmark/out/pvs_mainline_v4_ifcbench_fantasy_metropolis_v1"
STANDARD_MODEL = "neural_instance_culling/model/out/pvs_mainline_v4_standard_graphics_v1"
STANDARD_BENCH = "neural_instance_culling/benchmark/out/paper_results/standard_graphics/test_metrics"
STANDARD_PREPROCESSING = "neural_instance_culling/benchmark/out/paper_results/standard_graphics/preprocessing"


def _seed_paths(pattern: str) -> tuple[str, ...]:
    return tuple(pattern.format(seed=seed) for seed in SEEDS)


# This is deliberately a small, fixed source table.  A stage without an explicit
# field stays unavailable; no duration or memory value is inferred from files.
SCENE_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "hkust-v3": {
        "sampling": {
            "sources": ("neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/color_id/logs/*_stdout.log",),
            "elapsedSources": ("neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/color_id/logs/*_stdout.log",),
            "deviceSources": ("neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/color_id/logs/*_stdout.log",),
            "elapsed": ("elapsedMs", 0.001, "text"), "device": "sampler",
            "outputs": ("neural_instance_culling/sampler/out/hkust_v3_viewcell_fov66/color_id/*.jsonl",),
            "completeness": "partial", "scope": "sum of formal sampler shard elapsedMs; parallel wall time was not recorded",
        },
        "relation": {
            "sources": ("neural_instance_culling/dataset/out/pvs_v4_integrated_visibility_mainline_v1/bounded_relation_csr_v3/relation_csr_meta.json",),
            "outputs": ("neural_instance_culling/dataset/out/pvs_v4_integrated_visibility_mainline_v1/bounded_relation_csr_v3",),
            "notes": "No relation-build elapsed or resource record was registered for HKUST.",
        },
        "fixed_geometry": {
            "outputs": ("neural_instance_culling/dataset/out/fixed_geometry_features_hkust_v3/instance_geo_features_fp16.bin", "neural_instance_culling/dataset/out/fixed_geometry_features_hkust_v3/geometry_encoder.pt"),
        },
        "training": {
            "sources": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/run_manifest.json") + _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/train_history.json"),
            "elapsedSources": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/train_history.json"),
            "deviceSources": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/run_manifest.json"),
            "elapsed": ("elapsedSeconds", 1.0, "json"), "lastPerSource": True,
            "deviceField": "device.cudaName",
            "outputs": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/best_safe.pt"),
            "completeness": "complete", "scope": "sum of the final cumulative elapsedSeconds value from each seed history",
        },
        "calibration": {
            "sources": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/calibration_ready_summary.json"),
            "outputs": _seed_paths(HKUST_MODEL + "/paper_full_seed{seed}_e40/calibration_ready_summary.json"),
        },
        "export": {
            "sources": ("neural_instance_culling/model/out/hkust_v4_full_seed20260802_epoch036_image_eval_bundle_20260902/model_meta.json",),
            "outputs": ("neural_instance_culling/model/out/hkust_v4_full_seed20260802_epoch036_image_eval_bundle_20260902",),
        },
    },
    "ifcbench_fantasy_metropolis_instanced_v2": {
        "sampling": {
            "sources": ("neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/viewcell_colorid_k4/logs/*_stdout.log",),
            "elapsedSources": ("neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/viewcell_colorid_k4/logs/*_stdout.log",),
            "deviceSources": ("neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/viewcell_colorid_k4/logs/*_stdout.log",),
            "elapsed": ("elapsedMs", 0.001, "text"), "device": "sampler",
            "outputs": ("neural_instance_culling/sampler/out/ifcbench_fantasy_metropolis_instanced_v2/viewcell_colorid_k4/*.jsonl",),
            "completeness": "partial", "scope": "sum of formal sampler shard elapsedMs; parallel wall time was not recorded",
        },
        "relation": {
            "sources": ("neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_v4_bounded_relation_csr_v1/relation_csr_meta.json", "neural_instance_culling/benchmark/out/ifcbench_fantasy_metropolis_v4_depth_layers_v1/sparse_compact_run.stdout.log", "neural_instance_culling/benchmark/out/ifcbench_fantasy_metropolis_v4_depth_layers_v1/sparse_compact_run_retry.stdout.log"),
            "elapsedSources": ("neural_instance_culling/benchmark/out/ifcbench_fantasy_metropolis_v4_depth_layers_v1/sparse_compact_run.stdout.log", "neural_instance_culling/benchmark/out/ifcbench_fantasy_metropolis_v4_depth_layers_v1/sparse_compact_run_retry.stdout.log"),
            "elapsed": ("elapsedSeconds", 1.0, "json"), "successOnly": True,
            "outputs": ("neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_v4_bounded_relation_csr_v1",),
            "completeness": "partial", "scope": "sum of successful sparse compaction shard elapsedSeconds; full relation wall time was not recorded",
        },
        "fixed_geometry": {
            "outputs": ("neural_instance_culling/dataset/out/fixed_geometry_features_metropolis_v2/instance_geo_features_fp16.bin", "neural_instance_culling/dataset/out/fixed_geometry_features_metropolis_v2/geometry_encoder.pt"),
        },
        "training": {
            "sources": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/run_manifest.json") + _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/train_history.json") + _seed_paths(IFC_BENCH + "/logs/train/train_seed{seed}.stdout.log"),
            "elapsedSources": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/train_history.json"),
            "deviceSources": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/run_manifest.json"),
            "vramSources": _seed_paths(IFC_BENCH + "/logs/train/train_seed{seed}.stdout.log"),
            "elapsed": ("elapsedSeconds", 1.0, "json"), "lastPerSource": True,
            "deviceField": "device.cudaName", "vramField": "cudaPeakMemoryAllocatedMiB",
            "outputs": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/best_safe.pt"),
            "completeness": "complete", "scope": "sum of the final cumulative elapsedSeconds value from each seed history",
        },
        "calibration": {
            "sources": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/calibration_ready_summary.json"),
            "outputs": _seed_paths(IFC_MODEL + "/full_seed{seed}_e40/calibration_ready_summary.json"),
        },
        "export": {
            "sources": (IFC_BENCH + "/logs/export_seed20260802_runtime.stdout.log",),
            "elapsedSources": (IFC_BENCH + "/logs/export_seed20260802_runtime.stdout.log",),
            "elapsed": ("elapsedSeconds", 1.0, "json"),
            "outputs": (IFC_MODEL + "/runtime_seed20260802_best_safe_v1",),
        },
    },
}


def _standard_scene_spec(scene: str, sampler_name: str, relation_name: str, depth_name: str) -> dict[str, dict[str, Any]]:
    model = f"{STANDARD_MODEL}/{scene}"
    sampler = f"neural_instance_culling/sampler/out/{sampler_name}"
    relation = f"neural_instance_culling/dataset/out/{relation_name}"
    geometry = f"neural_instance_culling/dataset/out/standard_graphics_scenes/{scene}"
    depth_log = f"{STANDARD_PREPROCESSING}/{depth_name}/run_shards_stdout.log"
    return {
        "sampling": {
            "sources": (f"{sampler}/logs/*_stdout.log", f"{sampler}/gpu_execution_summary.json"),
            "elapsedSources": (f"{sampler}/logs/*_stdout.log",),
            "deviceSources": (f"{sampler}/logs/*_stdout.log",),
            "elapsed": ("elapsedMs", 0.001, "text"), "device": "sampler",
            "outputs": (f"{sampler}/*.jsonl",),
            "completeness": "partial",
            "scope": "sum of formal hardware sampler shard elapsedMs; parallel wall time was not recorded",
        },
        "relation": {
            "sources": (f"{relation}/relation_csr_meta.json", depth_log),
            "elapsedSources": (depth_log,),
            "elapsed": ("elapsedSeconds", 1.0, "json"), "successOnly": True,
            "outputs": (relation,),
            "completeness": "partial",
            "scope": "sum of successful hardware depth-and-compaction shard elapsedSeconds; parallel wall time and final merge time were not recorded",
        },
        "fixed_geometry": {
            "sources": (f"{geometry}/instance_geo_features_fp16.json",),
            "outputs": (f"{geometry}/instance_geo_features_fp16.bin", f"{geometry}/instance_geo_features_fp16.json"),
            "notes": "The fixed feature output is registered, but its standalone encoding time was not recorded.",
        },
        "training": {
            "sources": _seed_paths(model + "/full_seed{seed}_e40/run_manifest.json") + _seed_paths(model + "/full_seed{seed}_e40/train_metrics.jsonl") + _seed_paths(STANDARD_BENCH + f"/{scene}/logs/train/train_seed{{seed}}.stdout.log"),
            "elapsedSources": _seed_paths(model + "/full_seed{seed}_e40/train_metrics.jsonl"),
            "deviceSources": _seed_paths(model + "/full_seed{seed}_e40/run_manifest.json"),
            "vramSources": _seed_paths(STANDARD_BENCH + f"/{scene}/logs/train/train_seed{{seed}}.stdout.log"),
            "elapsed": ("elapsedSeconds", 1.0, "json"), "lastPerSource": True,
            "deviceField": "arguments.device", "vramField": "cudaPeakMemoryAllocatedMiB",
            "outputs": _seed_paths(model + "/full_seed{seed}_e40/best_safe.pt"),
            "completeness": "complete",
            "scope": "sum of the final cumulative elapsedSeconds value from each completed seed history",
            "notes": "The recorded device field identifies CUDA; GPU model evidence is reported with the experiment execution logs.",
        },
        "calibration": {
            "sources": _seed_paths(model + "/full_seed{seed}_e40/calibration_ready_summary.json"),
            "outputs": _seed_paths(model + "/full_seed{seed}_e40/calibration_ready_summary.json"),
        },
        "export": {
            "sources": (f"{STANDARD_BENCH}/{scene}/runtime_export.json",),
            "outputs": (model + "/runtime_selected_v1",),
            "notes": "Runtime export is registered after validation model selection; no export time is inferred before that stage completes.",
        },
    }


SCENE_SPECS.update(
    {
        "sponza_128k": _standard_scene_spec(
            "sponza_128k",
            "sponza_standard_graphics_viewcell_fov66",
            "sponza_standard_graphics_v4_bounded_relation_csr_v1",
            "sponza_triangle_depth_train",
        ),
        "viking_village_128k": _standard_scene_spec(
            "viking_village_128k",
            "viking_village_standard_graphics_viewcell_fov66",
            "viking_village_standard_graphics_v4_bounded_relation_csr_v1",
            "viking_triangle_depth_train",
        ),
        "bigcity_128k": _standard_scene_spec(
            "bigcity_128k",
            "bigcity_standard_graphics_viewcell_fov66",
            "bigcity_standard_graphics_v4_bounded_relation_csr_v1",
            "bigcity_triangle_depth_train",
        ),
    }
)

_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SAMPLER_DEVICE = re.compile(r'WebGL backend vendor="([^"]+)" renderer="([^"]+)"')


def _expand(root: Path, names: Sequence[str]) -> tuple[list[Path], list[str]]:
    found: list[Path] = []
    missing: list[str] = []
    for name in names:
        matches = sorted(root.glob(name))
        if matches:
            found.extend(path.resolve() for path in matches)
        else:
            missing.append(str((root / name).resolve()))
    return sorted(set(found)), sorted(set(missing))


def _records(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        value: Any = json.loads(text)
    except json.JSONDecodeError:
        value = []
        for line in text.splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, Mapping):
                value.append(item)
    if isinstance(value, Mapping):
        return [value]
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def _lookup(record: Mapping[str, Any], field: str) -> Any:
    value: Any = record
    for part in field.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def _numbers(paths: Sequence[Path], field: str, kind: str, *, last: bool = False, success_only: bool = False, sum_per_file: bool = True) -> list[float]:
    values: list[float] = []
    for path in paths:
        if kind == "text":
            raw = re.findall(rf'["\']{re.escape(field)}["\']\s*:\s*({_NUMBER})', path.read_text(encoding="utf-8", errors="replace"))
            current = [float(item) for item in raw]
            if any(not math.isfinite(value) for value in current):
                raise ValueError(f"non-finite {field} in {path}")
        else:
            current = []
            for record in _records(path):
                if success_only and record.get("returnCode") != 0:
                    continue
                value = _lookup(record, field)
                if isinstance(value, bool) or value is None:
                    continue
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(number):
                    raise ValueError(f"non-finite {field} in {path}")
                current.append(number)
        if current:
            if last:
                values.append(current[-1])
            elif sum_per_file:
                values.append(sum(current))
            else:
                values.extend(current)
    return values


def _devices(paths: Sequence[Path], spec: Mapping[str, Any]) -> list[str]:
    if spec.get("device") == "sampler":
        values = []
        for path in paths:
            text = path.read_text(encoding="utf-8", errors="replace")
            values.extend(f"vendor={vendor}; renderer={renderer}" for vendor, renderer in _SAMPLER_DEVICE.findall(text))
        return list(dict.fromkeys(values))
    field = spec.get("deviceField")
    if not field:
        return []
    values = []
    for path in paths:
        for record in _records(path):
            value = _lookup(record, str(field))
            if isinstance(value, str) and value:
                values.append(value)
    return list(dict.fromkeys(values))


def _output_size(paths: Sequence[Path]) -> tuple[int, int]:
    files: set[Path] = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(child.resolve() for child in path.rglob("*") if child.is_file())
    return sum(path.stat().st_size for path in files), len(files)


def _stage(root: Path, scene: str, stage: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    sources, missing = _expand(root, spec.get("sources", ()))
    elapsed_sources, _ = _expand(root, spec.get("elapsedSources", ()))
    device_sources, _ = _expand(root, spec.get("deviceSources", ()))
    vram_sources, _ = _expand(root, spec.get("vramSources", ()))
    output_paths, output_missing = _expand(root, spec.get("outputs", ()))
    missing = sorted(set(missing + output_missing))
    reasons: dict[str, str] = {}

    elapsed = spec.get("elapsed")
    time_values: list[float] = []
    if elapsed:
        field, scale, kind = elapsed
        time_values = [value * float(scale) for value in _numbers(elapsed_sources, field, kind, last=bool(spec.get("lastPerSource")), success_only=bool(spec.get("successOnly")))]
        if not time_values:
            reasons["timeSeconds"] = f"no finite {field} value in the registered sources"
    else:
        reasons["timeSeconds"] = "no explicit elapsed field was registered"
    time_seconds: Any = sum(time_values) if time_values else UNAVAILABLE

    devices = _devices(device_sources, spec)
    device: Any = devices[0] if len(devices) == 1 else UNAVAILABLE
    if not devices:
        reasons["device"] = "no explicit device value was found"
    elif len(devices) > 1:
        reasons["device"] = "registered device evidence contains conflicting values"

    if output_paths:
        output_bytes, output_count = _output_size(output_paths)
        output_mib: Any = output_bytes / (1024 * 1024)
    else:
        output_bytes = output_count = output_mib = UNAVAILABLE
        reasons["outputBytes"] = "no registered output path exists"

    vram_field = spec.get("vramField")
    vram_values = _numbers(vram_sources, str(vram_field), "json", sum_per_file=False) if vram_field else []
    peak_vram: Any = max(vram_values) if vram_values else UNAVAILABLE
    if not vram_values:
        reasons["peakVramMiB"] = "no explicit peak VRAM value was found"
    cpu_field = spec.get("cpuRamField")
    cpu_values = _numbers(vram_sources, str(cpu_field), "json", sum_per_file=False) if cpu_field else []
    peak_cpu: Any = max(cpu_values) if cpu_values else UNAVAILABLE
    if not cpu_values:
        reasons["peakCpuRamMiB"] = "no explicit peak CPU RAM value was registered"
    values = (time_seconds, device, output_bytes, peak_vram, peak_cpu)
    available = sum(value != UNAVAILABLE for value in values)
    status = "unavailable" if available == 0 else "partial" if missing or available < len(values) or spec.get("completeness") == "partial" else "available"
    return {
        "scene": scene, "stage": stage, "status": status,
        "timeSeconds": time_seconds, "timeAggregation": "sum" if elapsed else UNAVAILABLE,
        "timeRecordSelection": "last_per_source" if elapsed and spec.get("lastPerSource") else "all" if elapsed else UNAVAILABLE,
        "timeCompleteness": spec.get("completeness", UNAVAILABLE), "timeScope": spec.get("scope", UNAVAILABLE),
        "timeEvidenceCount": len(time_values), "timeSource": [str(path) for path in elapsed_sources],
        "device": device, "deviceSource": [str(path) for path in device_sources],
        "outputBytes": output_bytes, "outputMiB": output_mib, "outputFileCount": output_count,
        "outputSource": [str(path) for path in output_paths], "peakVramMiB": peak_vram,
        "peakVramSource": [str(path) for path in vram_sources], "peakCpuRamMiB": peak_cpu,
        "peakCpuRamSource": [str(path) for path in vram_sources] if cpu_values else [], "sourcePaths": [str(path) for path in sources],
        "missingSources": missing, "unavailableReasons": reasons, "notes": spec.get("notes", ""),
    }


def collect_cost_rows(source_root: str | Path, scenes: Sequence[str] | None = None) -> list[dict[str, Any]]:
    root = Path(source_root).expanduser().resolve()
    selected = list(scenes) if scenes is not None else list(SCENE_SPECS)
    return [
        _stage(root, scene, stage, SCENE_SPECS[scene].get(stage, {}))
        for scene in selected for stage in STAGES
    ]


def _cell(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def write_cost_bundle(source_root: str | Path, output_dir: str | Path, scenes: Sequence[str] | None = None) -> dict[str, Any]:
    root, output = Path(source_root).resolve(), Path(output_dir).expanduser().resolve()
    rows = collect_cost_rows(root, scenes)
    payload = {
        "schema": SCHEMA, "generatedBy": "collect_preprocessing_cost.py", "unavailableValue": UNAVAILABLE,
        "sourceRoot": str(root), "scenes": list(dict.fromkeys(row["scene"] for row in rows)),
        "stages": list(STAGES), "rows": rows,
        "rowStatusCounts": {status: sum(row["status"] == status for row in rows) for status in ("available", "partial", "unavailable")},
        "fieldSemantics": {"timeSeconds": "registered explicit elapsed values only", "outputBytes": "recursive stat bytes at registered paths", "peakVramMiB": "maximum explicit VRAM value", "peakCpuRamMiB": "explicit CPU RAM high-water value; never inferred"},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cost.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "cost.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        writer.writeheader()
        writer.writerows({field: _cell(row.get(field, "")) for field in COST_FIELDS} for row in rows)
    return {"schema": SCHEMA, "unavailableValue": UNAVAILABLE, "sourceRoot": str(root), "costCsv": str(output / "cost.csv"), "costJson": str(output / "cost.json"), "sceneCount": len(payload["scenes"]), "stageCount": len(STAGES), "rowCount": len(rows), "rowStatusCounts": payload["rowStatusCounts"]}


def collect_preprocessing_cost(source_root: str | Path, output_dir: str | Path, scenes: Sequence[str] | None = None) -> dict[str, Any]:
    return write_cost_bundle(source_root, output_dir, scenes)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out" / "paper_results" / "preprocessing")
    parser.add_argument("--scenes", default="")
    args = parser.parse_args(argv)
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()] or None
    print(json.dumps(collect_preprocessing_cost(args.source_root, args.output_dir, scenes), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
