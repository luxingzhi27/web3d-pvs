#!/usr/bin/env python3
"""Build the paper bundle and register its known artifact paths."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import sys
from typing import Any, Iterable, Mapping, Sequence


BENCHMARK_DIR = Path(__file__).resolve().parent
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
from collect_preprocessing_cost import collect_preprocessing_cost  # noqa: E402


RELATIVE_INPUTS = {
    "ablation": Path("neural_instance_culling/benchmark/out/pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json"),
    "rank": Path("neural_instance_culling/benchmark/out/pvs_survival_rank_capacity_sweep_v1/capacity_summary.json"),
    "runtime": Path("neural_instance_culling/benchmark/out/paper_results/mobile_runtime/runtime_paper_summary.json"),
    "hkust_image": Path("neural_instance_culling/benchmark/out/pvs_v4_image_validation_hkust_seed20260802_v1/summary.json"),
    "ifcbench_image": Path("neural_instance_culling/benchmark/out/pvs_v4_image_validation_ifcbench_seed20260802_v1/summary.json"),
}

ARTIFACT_REGISTRY_SCHEMA = "pvs-paper-artifact-registry-v1"
ARTIFACT_SECTIONS = (
    "sceneStatistics", "testMetrics", "imageMetrics", "ablation", "rankSweep",
    "runtime", "hzb", "thresholdCurves", "streaming", "gtConvergence",
    "preprocessing", "figures",
)
HZB_SHELL_ARTIFACTS = (
    ("hkust_lossless_shell", "geometry_shell_hzb_lossless_hkust/shell_meta.json"),
    ("hkust_equal_asset_shell", "geometry_shell_hzb_equal_asset_hkust/shell_meta.json"),
    ("ifcbench_lossless_shell", "geometry_shell_hzb_lossless_ifcbench/shell_meta.json"),
    ("ifcbench_equal_asset_shell", "geometry_shell_hzb_equal_asset_ifcbench/shell_meta.json"),
)
ABLATION_METRICS = (
    "posePrecision", "poseRecall", "poseWeightedRecall", "poseAccuracy",
    "poseBalancedAccuracy", "poseSpecificity", "aggregatePrecision",
    "aggregateRecall", "aggregateWeightedRecall", "aggregateAccuracy",
    "aggregateBalancedAccuracy", "aggregateSpecificity", "aggregateUsefulCull",
    "aggregateBadCull", "avgCandidateCount", "avgGtCount", "avgPredCount",
    "predictedGlbCount", "predictedGlbBytes", "glbByteReduction",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_ablation(source: Path, output: Path) -> dict[str, Any]:
    summary = load_json(source)
    if summary.get("split") != "validation" or summary.get("testRead") is not False:
        raise ValueError("core ablation input must be test-free validation output")
    source_rows = summary.get("rows")
    if not isinstance(source_rows, dict):
        raise ValueError("core ablation summary has no variant rows")
    rows: list[dict[str, Any]] = []
    for variant, members in source_rows.items():
        if not isinstance(members, dict) or len(members) < 2:
            raise ValueError(f"ablation variant {variant} has fewer than two seeds")
        values = list(members.values())
        row: dict[str, Any] = {
            "variant": variant, "split": "validation", "seedCount": len(values),
            "safeSeedCount": sum(float(item["validationWeightedRecallLowerConfidenceBound"]) > 0.99 for item in values),
            "runtimeFeatureMiB": mean(float(item["runtimeFeatureBytes"]) for item in values) / (1024 * 1024),
        }
        for metric in ABLATION_METRICS:
            numbers = [float(item["metrics"][metric]) for item in values]
            row[f"{metric}Mean"], row[f"{metric}Std"] = mean(numbers), stdev(numbers)
        rows.append(row)
    write_csv(output / "ablation/core_ablation.csv", rows, list(rows[0]))
    write_json(output / "ablation/paired_bootstrap.json", {
        "schema": summary.get("schema"), "split": "validation", "reference": summary.get("reference"),
        "bootstrap": summary.get("bootstrap"), "comparisons": summary.get("pairedComparisons"),
    })
    return {"variantCount": len(rows), "seedCountPerVariant": rows[0]["seedCount"]}


def build_rank(source: Path, output: Path) -> dict[str, Any]:
    summary = load_json(source)
    if summary.get("selectionSplit") != "validation" or summary.get("testRead") is not False:
        raise ValueError("rank input must be test-free validation output")
    rows = []
    for rank_text, item in sorted(summary["byRank"].items(), key=lambda pair: int(pair[0])):
        aggregate, pose = item["aggregateMean"], item["poseMacroMean"]
        rows.append({
            "rank": int(rank_text), "survivalDim": int(item["survivalDim"]),
            "runtimeFeatureDim": int(item["runtimeFeatureDim"]),
            "runtimeFeatureMiB": float(item["runtimeFeatureBytes"]) / (1024 * 1024),
            "modelStateParameterCount": int(item["modelStateParameterCount"]),
            "seedCount": int(item["memberCount"]), "safeSeedCount": int(item["safeMemberCount"]),
            "posePrecision": float(pose["precision"]), "poseWeightedRecall": float(pose["weightedRecall"]),
            "aggregatePrecision": float(aggregate["precision"]), "aggregateWeightedRecall": float(aggregate["weightedRecall"]),
            "aggregateBalancedAccuracy": float(aggregate["balancedAccuracy"]),
            "aggregateUsefulCull": float(aggregate["usefulCull"]), "aggregateBadCull": float(aggregate["badCull"]),
            "avgPredCount": float(aggregate["avgPredCount"]),
        })
    write_csv(output / "rank_sweep/rank_capacity.csv", rows, list(rows[0]))
    return {"rankCount": len(rows), "ranks": [row["rank"] for row in rows]}


def build_runtime(source: Path, output: Path) -> dict[str, Any]:
    summary = load_json(source)
    rows = list(summary.get("groups") or [])
    if not rows:
        raise ValueError("runtime summary has no groups")
    formal = [item for item in rows if item.get("status") == "formal"]
    if any(not item.get("scene") for item in formal):
        raise ValueError("formal runtime groups must identify their scene")
    if any(int(item.get("formalSessions", 0)) < 5 for item in formal):
        raise ValueError("formal runtime groups require at least five sessions")
    fields = (
        "status", "scene", "device", "deviceModel", "backend", "adapter",
        "timingSource", "formalUploadCount", "formalSessions", "sampleCount",
        "poseCount", "candidateTotal", "candidateMean", "candidateP95",
        "candidateMin", "candidateMax", "runtimeAssetMiB", "latencyMeanMs",
        "latencyMeanCi95LowMs", "latencyMeanCi95HighMs", "latencyP50Ms",
        "latencyP50Ci95LowMs", "latencyP50Ci95HighMs", "latencyP95Ms",
        "latencyP95Ci95LowMs", "latencyP95Ci95HighMs", "fitInterceptMs",
        "fitSlopeMsPerCandidate", "fitRmseMs", "fitMaeMs", "fitR2",
        "fitMaxAbsErrorMs", "reason",
    )
    serializable = [{field: item.get(field) for field in fields} for item in rows]
    runtime_output = output / "mobile_runtime/runtime_summary.csv"
    write_csv(runtime_output, serializable, fields)
    return {
        "groupCount": len(rows),
        "formalGroupCount": len(formal),
        "scenes": sorted({str(item["scene"]) for item in formal}),
        "backends": sorted({str(item.get("backend")) for item in formal}),
        "unavailableGroups": [
            {
                "scene": item.get("scene"),
                "device": item.get("device"),
                "backend": item.get("backend"),
                "reason": item.get("reason"),
            }
            for item in rows
            if item.get("status") != "formal"
        ],
        "output": str(runtime_output.resolve()),
    }


def build_images(inputs: Mapping[str, Path], output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scene, source in inputs.items():
        summary = load_json(source)
        if summary.get("formalImageEvaluationReady") is not True:
            raise ValueError(f"image input is not formal-ready: {source}")
        split = str(summary.get("split"))
        write_json(output / f"image_metrics/{scene}_{split}.json", {
            "schema": "pvs-paper-image-metrics-v1", "scene": scene, "split": split,
            "modelName": summary.get("modelName"), "threshold": summary.get("threshold"),
            "viewcellCount": summary.get("viewcellCount"), "subposesPerViewcell": summary.get("subposesPerViewcell"),
            "imageResolution": summary.get("imageResolution"), "componentSetMetrics": summary.get("componentSetMetrics"),
            "glbSetMetrics": summary.get("glbSetMetrics"), "weightedRecall": summary.get("weightedRecall"),
            "imageMetrics": summary.get("imageMetrics"), "formalImageEvaluationReady": True,
            "gpuGate": summary["renderer"]["gpuGate"], "source": str(source.resolve()),
        })
        result[scene] = {"split": split, "viewcellCount": int(summary["viewcellCount"]), "subposeCount": int(summary["imageMetrics"]["evaluatedSubposeCount"]), "formalHardware": bool(summary["renderer"]["gpuGate"]["hardware"])}
    return result


def _optional_root(value: Path | None, data_root: Path, relative: str) -> Path:
    return (value if value is not None else data_root / relative).expanduser().resolve()


def _artifact_specs(data_root: Path, output: Path, *, hzb: Path, curves: Path, streaming: Path, gt: Path, figures: Path, test_image: Path) -> list[tuple[str, str, Path]]:
    paper = data_root / "neural_instance_culling/benchmark/out/paper_results"
    specs: list[tuple[str, str, Path]] = [
        ("sceneStatistics", "scene_statistics", paper / "scene_statistics.csv"),
        ("testMetrics", "test_metrics", paper / "test_metrics"),
        ("imageMetrics", "validation_image_metrics", output / "image_metrics"),
        ("imageMetrics", "test_image_metrics", test_image if test_image.is_file() else test_image / "*test*.json"),
        ("ablation", "core_ablation", output / "ablation/core_ablation.csv"),
        ("rankSweep", "rank_capacity", output / "rank_sweep/rank_capacity.csv"),
        ("runtime", "runtime_summary", output / "mobile_runtime/runtime_summary.csv"),
    ]
    specs.extend(("hzb", name, hzb / relative) for name, relative in HZB_SHELL_ARTIFACTS)
    specs.append((
        "hzb",
        "formal_execution",
        hzb / "geometry_shell_hzb_paper_2026-09-09/execution_summary.json",
    ))
    specs.extend([
        ("thresholdCurves", "threshold_curves", curves),
        ("streaming", "formal_streaming", streaming / "figures/paper_output_manifest.json"),
        ("gtConvergence", "hkust_128", gt / "hkust_128_eval/summary.json"),
        ("gtConvergence", "ifcbench_128", gt / "ifcbench_128_eval/summary.json"),
        ("preprocessing", "preprocessing", output / "preprocessing"),
        ("figures", "figures", figures),
    ])
    return specs


def _artifact_exists(path: Path) -> bool:
    return bool(list(path.parent.glob(path.name))) if "*" in path.name else path.exists()


def build_artifact_registry(specs: Sequence[tuple[str, str, Path]]) -> dict[str, Any]:
    artifacts = []
    for section, artifact_id, path in specs:
        available = _artifact_exists(path)
        artifacts.append({
            "section": section, "artifactId": artifact_id, "path": str(path.resolve()),
            "status": "available" if available else "unavailable",
            "reason": "" if available else "registered path does not exist",
        })
    return {"schema": ARTIFACT_REGISTRY_SCHEMA, "sections": list(ARTIFACT_SECTIONS), "artifacts": artifacts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "out" / "paper_results")
    parser.add_argument("--hzb-dir", type=Path, default=None)
    parser.add_argument("--threshold-curves-dir", type=Path, default=None)
    parser.add_argument("--streaming-dir", type=Path, default=None)
    parser.add_argument("--gt-convergence-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--test-image-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root, output = args.data_root.resolve(), args.output_dir.resolve()
    sources = {name: data_root / relative for name, relative in RELATIVE_INPUTS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed paper input(s): {missing}")
    roots = {
        "hzb": _optional_root(args.hzb_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/hzb"),
        "curves": _optional_root(args.threshold_curves_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/threshold_curves"),
        "streaming": _optional_root(args.streaming_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/streaming_formal"),
        "gt": _optional_root(args.gt_convergence_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/gt_convergence"),
        "figures": _optional_root(args.figures_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/figures"),
        "test_image": _optional_root(args.test_image_dir, data_root, "neural_instance_culling/benchmark/out/paper_results/image_metrics"),
    }
    preprocessing = collect_preprocessing_cost(data_root, output / "preprocessing")
    ablation = build_ablation(sources["ablation"], output)
    rank = build_rank(sources["rank"], output)
    runtime = build_runtime(sources["runtime"], output)
    images = build_images({"hkust": sources["hkust_image"], "ifcbench": sources["ifcbench_image"]}, output)
    registry = build_artifact_registry(_artifact_specs(data_root, output, **roots))
    pending = [
        {
            "section": item["section"],
            "artifactId": item["artifactId"],
            "reason": item["reason"],
        }
        for item in registry["artifacts"]
        if item["status"] != "available"
    ]
    pending.extend(
        {"section": "runtime", "artifactId": "runtime_group", **item}
        for item in runtime["unavailableGroups"]
    )
    manifest = {
        "schema": "pvs-paper-result-bundle-v1", "scope": "completed evidence plus explicit unavailable registrations",
        "generatedBy": "build_paper_result_bundle.py", "artifactRegistry": registry,
        "ablation": ablation, "rankSweep": rank, "runtime": runtime, "imageMetrics": images,
        "preprocessing": preprocessing,
        "pending": pending,
    }
    write_json(output / "bundle_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
