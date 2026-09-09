#!/usr/bin/env python3
"""Collect completed paper evidence into a stable, table-oriented bundle."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping


RELATIVE_INPUTS = {
    "ablation": Path(
        "neural_instance_culling/benchmark/out/"
        "pvs_v4_integrated_visibility_mainline_v1/paper_core_ablation_summary.json"
    ),
    "rank": Path(
        "neural_instance_culling/benchmark/out/"
        "pvs_survival_rank_capacity_sweep_v1/capacity_summary.json"
    ),
    "runtime": Path(
        "neural_instance_culling/benchmark/out/"
        "pvs_v4_frontend_inference_latency_v1/summary.json"
    ),
    "hkust_image": Path(
        "neural_instance_culling/benchmark/out/"
        "pvs_v4_image_validation_hkust_seed20260802_v1/summary.json"
    ),
    "ifcbench_image": Path(
        "neural_instance_culling/benchmark/out/"
        "pvs_v4_image_validation_ifcbench_seed20260802_v1/summary.json"
    ),
}

ABLATION_METRICS = (
    "posePrecision",
    "poseRecall",
    "poseWeightedRecall",
    "poseAccuracy",
    "poseBalancedAccuracy",
    "poseSpecificity",
    "aggregatePrecision",
    "aggregateRecall",
    "aggregateWeightedRecall",
    "aggregateAccuracy",
    "aggregateBalancedAccuracy",
    "aggregateSpecificity",
    "aggregateUsefulCull",
    "aggregateBadCull",
    "avgCandidateCount",
    "avgGtCount",
    "avgPredCount",
    "predictedGlbCount",
    "predictedGlbBytes",
    "glbByteReduction",
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
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
        member_values = list(members.values())
        row: dict[str, Any] = {
            "variant": variant,
            "split": "validation",
            "seedCount": len(member_values),
            "safeSeedCount": sum(
                float(item["validationWeightedRecallLowerConfidenceBound"]) > 0.99
                for item in member_values
            ),
            "runtimeFeatureMiB": mean(float(item["runtimeFeatureBytes"]) for item in member_values)
            / (1024.0 * 1024.0),
        }
        for metric in ABLATION_METRICS:
            values = [float(item["metrics"][metric]) for item in member_values]
            row[f"{metric}Mean"] = mean(values)
            row[f"{metric}Std"] = stdev(values)
        rows.append(row)

    fields = list(rows[0])
    write_csv(output / "ablation" / "core_ablation.csv", rows, fields)
    write_json(output / "ablation" / "paired_bootstrap.json", {
        "schema": summary.get("schema"),
        "split": "validation",
        "reference": summary.get("reference"),
        "bootstrap": summary.get("bootstrap"),
        "comparisons": summary.get("pairedComparisons"),
    })
    return {"variantCount": len(rows), "seedCountPerVariant": rows[0]["seedCount"]}


def build_rank(source: Path, output: Path) -> dict[str, Any]:
    summary = load_json(source)
    if summary.get("selectionSplit") != "validation" or summary.get("testRead") is not False:
        raise ValueError("rank input must be test-free validation output")
    rows = []
    for rank_text, item in sorted(summary["byRank"].items(), key=lambda pair: int(pair[0])):
        aggregate = item["aggregateMean"]
        pose = item["poseMacroMean"]
        rows.append({
            "rank": int(rank_text),
            "survivalDim": int(item["survivalDim"]),
            "runtimeFeatureDim": int(item["runtimeFeatureDim"]),
            "runtimeFeatureMiB": float(item["runtimeFeatureBytes"]) / (1024.0 * 1024.0),
            "modelStateParameterCount": int(item["modelStateParameterCount"]),
            "seedCount": int(item["memberCount"]),
            "safeSeedCount": int(item["safeMemberCount"]),
            "posePrecision": float(pose["precision"]),
            "poseWeightedRecall": float(pose["weightedRecall"]),
            "aggregatePrecision": float(aggregate["precision"]),
            "aggregateWeightedRecall": float(aggregate["weightedRecall"]),
            "aggregateBalancedAccuracy": float(aggregate["balancedAccuracy"]),
            "aggregateUsefulCull": float(aggregate["usefulCull"]),
            "aggregateBadCull": float(aggregate["badCull"]),
            "avgPredCount": float(aggregate["avgPredCount"]),
        })
    write_csv(output / "rank_sweep" / "rank_capacity.csv", rows, list(rows[0]))
    return {"rankCount": len(rows), "ranks": [row["rank"] for row in rows]}


def build_runtime(source: Path, output: Path) -> dict[str, Any]:
    summary = load_json(source)
    rows = list(summary.get("groups") or [])
    if not rows:
        raise ValueError("runtime summary has no groups")
    fields = [
        "device", "backend", "timingSource", "sessionCount", "sampleCount",
        "runtimeAssetMiB", "candidateMean", "candidateP95", "modelInferenceP50Ms",
        "modelInferenceP50Ci95", "modelInferenceP95Ms", "modelInferenceP95Ci95",
        "near10kSampleCount", "near10kP95Ms",
    ]
    serializable = []
    for item in rows:
        row = {field: item.get(field) for field in fields}
        for field in ("modelInferenceP50Ci95", "modelInferenceP95Ci95"):
            row[field] = json.dumps(row[field], separators=(",", ":"))
        serializable.append(row)
    write_csv(output / "mobile_runtime" / "hkust_webgpu_summary.csv", serializable, fields)
    return {"groupCount": len(rows), "scene": "hkust", "backend": "webgpu"}


def build_images(inputs: Mapping[str, Path], output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scene, source in inputs.items():
        summary = load_json(source)
        if summary.get("formalImageEvaluationReady") is not True:
            raise ValueError(f"image input is not formal-ready: {source}")
        split = str(summary.get("split"))
        target = output / "image_metrics" / f"{scene}_{split}.json"
        write_json(target, {
            "schema": "pvs-paper-image-metrics-v1",
            "scene": scene,
            "split": split,
            "modelName": summary.get("modelName"),
            "threshold": summary.get("threshold"),
            "viewcellCount": summary.get("viewcellCount"),
            "subposesPerViewcell": summary.get("subposesPerViewcell"),
            "imageResolution": summary.get("imageResolution"),
            "componentSetMetrics": summary.get("componentSetMetrics"),
            "glbSetMetrics": summary.get("glbSetMetrics"),
            "weightedRecall": summary.get("weightedRecall"),
            "imageMetrics": summary.get("imageMetrics"),
            "formalImageEvaluationReady": True,
            "gpuGate": summary["renderer"]["gpuGate"],
            "source": str(source.resolve()),
        })
        result[scene] = {
            "split": split,
            "viewcellCount": int(summary["viewcellCount"]),
            "subposeCount": int(summary["imageMetrics"]["evaluatedSubposeCount"]),
            "formalHardware": bool(summary["renderer"]["gpuGate"]["hardware"]),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "out" / "paper_results",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output = args.output_dir.resolve()
    sources = {name: data_root / relative for name, relative in RELATIVE_INPUTS.items()}
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing completed paper input(s): {missing}")

    manifest = {
        "schema": "pvs-paper-result-bundle-v1",
        "scope": "completed evidence only",
        "ablation": build_ablation(sources["ablation"], output),
        "rankSweep": build_rank(sources["rank"], output),
        "runtime": build_runtime(sources["runtime"], output),
        "imageMetrics": build_images({
            "hkust": sources["hkust_image"],
            "ifcbench": sources["ifcbench_image"],
        }, output),
        "pending": [
            "frozen test metrics and images",
            "IFCBench runtime and WASM runtime",
            "geometry-shell HZB",
            "cold-cache streaming",
            "scene statistics and preprocessing costs",
            "GT convergence",
        ],
    }
    write_json(output / "bundle_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
