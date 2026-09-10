#!/usr/bin/env python3
"""Generate the compact asset/runtime Table 4 and rank-capacity Pareto plot."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
PAPER_RESULTS = ROOT / "neural_instance_culling/benchmark/out/paper_results"
SCENES = ("hkust", "ifcbench")
VARIANTS = ("lossless", "equal-asset")
LATENCY_FIELDS = (
    "latencyMeanMs", "latencyMeanCi95LowMs", "latencyMeanCi95HighMs",
    "latencyP50Ms", "latencyP50Ci95LowMs", "latencyP50Ci95HighMs",
    "latencyP95Ms", "latencyP95Ci95LowMs", "latencyP95Ci95HighMs",
)
TABLE_FIELDS = (
    "row_type", "scene", "method", "variant", "status", "device", "backend",
    "timing_source", "transfer_bytes", "transfer_mib", "expanded_geometry_bytes",
    "expanded_memory_bytes", "instance_count", "glb_count", "prototype_count",
    "prototype_triangles", "expanded_triangles", "occluder_instance_count",
    "transfer_to_lossless_ratio", "neural_to_lossless_ratio", "formal_sessions",
    "sample_count", "candidate_mean", "candidate_p95",
    "latency_mean_ms", "latency_mean_ci95_low_ms", "latency_mean_ci95_high_ms",
    "latency_p50_ms", "latency_p50_ci95_low_ms", "latency_p50_ci95_high_ms",
    "latency_p95_ms", "latency_p95_ci95_low_ms", "latency_p95_ci95_high_ms",
    "reason", "source",
)
RANK_FIELDS = (
    "rank", "survival_dim", "runtime_feature_dim", "runtime_feature_mib",
    "aggregate_useful_cull", "aggregate_balanced_accuracy", "aggregate_weighted_recall",
    "aggregate_precision", "aggregate_bad_cull", "avg_pred_count", "seed_count",
    "safe_seed_count", "safety_status", "pareto_optimal", "source",
)

def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value

def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    return rows

def number(value: Any, field: str, path: Path, *, integer: bool = False) -> int | float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"missing {field}: {path}")
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} in {path}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {field} in {path}: {value!r}")
    return parsed

def scene_key(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"hkust", "hkust-v3"}:
        return "hkust"
    if text in {"ifcbench", "ifcbench_fantasy_metropolis_instanced_v2"}:
        return "ifcbench"
    raise ValueError(f"unsupported scene: {value!r}")

def scene_label(scene: str) -> str:
    return "HKUST" if scene == "hkust" else "IFCBench / Metropolis"

def actual_bytes(directory: Path) -> int:
    if not directory.is_dir():
        raise FileNotFoundError(f"asset directory does not exist: {directory}")
    return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

def _require(mapping: Mapping[str, Any], key: str, path: Path) -> Any:
    if key not in mapping:
        raise ValueError(f"missing {key} in {path}")
    return mapping[key]

def read_neural_asset(scene: str, directory: Path) -> dict[str, Any]:
    meta_path = directory / "model_meta.json"
    meta = load_json(meta_path)
    if not meta.get("schema"):
        raise ValueError(f"neural model_meta has no schema: {meta_path}")
    transfer = actual_bytes(directory)
    return {
        "row_type": "asset", "scene": scene, "method": "neural",
        "variant": "pvs-mainline-v4", "status": "available", "device": "",
        "backend": "", "timing_source": "", "transfer_bytes": transfer,
        "transfer_mib": transfer / 2**20, "expanded_geometry_bytes": None,
        "expanded_memory_bytes": None, "instance_count": meta.get("numInstances"),
        "glb_count": meta.get("numGlbs"), "prototype_count": None,
        "prototype_triangles": None, "expanded_triangles": None,
        "occluder_instance_count": None, "source": str(directory.resolve()),
    }

def read_hzb_asset(scene: str, variant: str, hzb_root: Path) -> dict[str, Any]:
    stem = f"geometry_shell_hzb_{variant.replace('-', '_')}_{scene}"
    directory = hzb_root / stem
    shell_path = directory / "shell_meta.json"
    offline_path = hzb_root / f"{stem}.offline.json"
    meta = load_json(shell_path)
    offline = load_json(offline_path)
    if meta.get("schema") != "geometry-shell-hzb-v2":
        raise ValueError(f"unexpected HZB shell schema: {shell_path}")
    if offline.get("schema") != "geometry-shell-hzb-offline-report-v1":
        raise ValueError(f"unexpected HZB offline schema: {offline_path}")
    if scene_key(meta.get("sceneName")) != scene or scene_key(offline.get("sceneName")) != scene:
        raise ValueError(f"HZB scene mismatch: {shell_path}")
    if meta.get("variant") != variant or offline.get("variant") != variant:
        raise ValueError(f"HZB variant mismatch: {shell_path}")
    if offline.get("runtimeAsset") is not False:
        raise ValueError(f"offline HZB report must not be a runtime asset: {offline_path}")

    stats = _require(meta, "stats", shell_path)
    streams = _require(meta, "streams", shell_path)
    if not isinstance(stats, Mapping) or not isinstance(streams, Mapping):
        raise ValueError(f"HZB stats/streams must be objects: {shell_path}")
    decoded_geometry = 0
    compressed_geometry = 0
    for name in ("positions", "indices", "transforms"):
        stream = _require(streams, name, shell_path)
        segments = _require(stream, "segments", shell_path)
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"HZB stream has no segments: {shell_path} ({name})")
        compressed_geometry += sum(int(_require(item, "byteLength", shell_path)) for item in segments)
        decoded_geometry += sum(int(_require(item, "decodedByteLength", shell_path)) for item in segments)
    if compressed_geometry != int(_require(stats, "compressedGeometryBytes", shell_path)):
        raise ValueError(f"HZB compressed stream total disagrees with stats: {shell_path}")

    runtime_files = _require(meta, "files", shell_path)
    binary_paths = [
        directory / str(_require(streams[name], "file", shell_path))
        for name in ("positions", "indices", "transforms")
    ]
    binary_paths.extend(
        directory / str(_require(descriptor, "file", shell_path))
        for descriptor in runtime_files.values()
    )
    binary_bytes = sum(path.stat().st_size for path in binary_paths)
    if binary_bytes != int(_require(stats, "binaryPayloadBytes", shell_path)):
        raise ValueError(f"HZB binary payload disagrees with stats: {shell_path}")
    transfer = actual_bytes(directory)
    if transfer != binary_bytes + shell_path.stat().st_size:
        raise ValueError(f"HZB directory contains unexpected transfer bytes: {directory}")

    runtime_payload = int(_require(stats, "runtimePayloadBytes", shell_path))
    if binary_bytes - compressed_geometry != runtime_payload:
        raise ValueError(f"HZB runtime payload disagrees with files: {shell_path}")
    selection = meta.get("selection")
    if variant == "equal-asset":
        if not isinstance(selection, Mapping) or int(selection.get("totalAssetBytes", -1)) != transfer:
            raise ValueError(f"equal-asset selection total disagrees with files: {shell_path}")
    elif selection is not None:
        raise ValueError(f"lossless HZB unexpectedly has a selection: {shell_path}")

    aggregate = offline.get("aggregate") or {}
    for key in ("sourceGlbCount", "opaquePrimitiveCount"):
        if key in aggregate and int(aggregate[key]) != int(stats[key]):
            raise ValueError(f"offline aggregate disagrees with shell stats: {shell_path} ({key})")
    offline_selection = offline.get("selection")
    if isinstance(offline_selection, Mapping) and variant == "equal-asset" and int(offline_selection.get("totalAssetBytes", -1)) != transfer:
        raise ValueError(f"offline equal-asset selection total disagrees with files: {offline_path}")
    return {
        "row_type": "asset", "scene": scene, "method": "geometry-shell-hzb",
        "variant": variant, "status": "available", "device": "", "backend": "",
        "timing_source": "", "transfer_bytes": transfer, "transfer_mib": transfer / 2**20,
        "expanded_geometry_bytes": decoded_geometry,
        "expanded_memory_bytes": decoded_geometry + runtime_payload,
        "instance_count": meta.get("instanceCount"), "glb_count": meta.get("globalGlbCount"),
        "prototype_count": meta.get("prototypeCount"),
        "prototype_triangles": stats.get("prototypeTriangles"),
        "expanded_triangles": stats.get("rasterizedTriangleInstanceCount"),
        "occluder_instance_count": meta.get("occluderInstanceCount"),
        "source": f"{shell_path.resolve()} + {offline_path.resolve()}",
    }

def classify_latency(row: Mapping[str, str], path: Path) -> tuple[str, str]:
    reason = str(row.get("reason") or "").strip()
    raw_status = str(row.get("status") or "").strip().lower()
    if "smoke-excluded" in reason.lower() or raw_status == "smoke-excluded":
        return "smoke-excluded", reason or "smoke-excluded: source row is marked smoke"
    if raw_status != "formal":
        return "unavailable", reason or "unavailable: source row is not formal"
    if any(str(row.get(field) or "").strip() == "" for field in LATENCY_FIELDS):
        return "unavailable", "unavailable: formal row has incomplete latency fields"
    if int(number(row.get("formalSessions"), "formalSessions", path, integer=True)) != 5:
        return "unavailable", "unavailable: formal latency requires five sessions"
    return "formal", ""

def read_runtime_rows(summary_path: Path, neural_rows: Mapping[str, Mapping[str, Any]], lossless_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = read_csv(summary_path)
    result = []
    for row in rows:
        scene = scene_key(row.get("scene"))
        status, reason = classify_latency(row, summary_path)
        neural = neural_rows[scene]
        lossless = lossless_rows[scene]
        output: dict[str, Any] = {
            "row_type": "runtime", "scene": scene, "method": "neural-runtime",
            "variant": "pvs-mainline-v4", "status": status,
            "device": row.get("device", ""), "backend": row.get("backend", ""),
            "timing_source": row.get("timingSource", "") if status == "formal" else "",
            "transfer_bytes": neural["transfer_bytes"], "transfer_mib": neural["transfer_mib"],
            "instance_count": neural.get("instance_count"), "glb_count": neural.get("glb_count"),
            "transfer_to_lossless_ratio": neural["transfer_bytes"] / lossless["transfer_bytes"],
            "neural_to_lossless_ratio": neural["transfer_bytes"] / lossless["transfer_bytes"],
            "formal_sessions": int(row["formalSessions"]) if status == "formal" else None,
            "sample_count": int(row["sampleCount"]) if status == "formal" else None,
            "candidate_mean": float(row["candidateMean"]) if status == "formal" else None,
            "candidate_p95": float(row["candidateP95"]) if status == "formal" else None,
            "reason": reason, "source": str(summary_path.resolve()),
        }
        for source_name, output_name in (
            ("latencyMeanMs", "latency_mean_ms"),
            ("latencyMeanCi95LowMs", "latency_mean_ci95_low_ms"),
            ("latencyMeanCi95HighMs", "latency_mean_ci95_high_ms"),
            ("latencyP50Ms", "latency_p50_ms"),
            ("latencyP50Ci95LowMs", "latency_p50_ci95_low_ms"),
            ("latencyP50Ci95HighMs", "latency_p50_ci95_high_ms"),
            ("latencyP95Ms", "latency_p95_ms"),
            ("latencyP95Ci95LowMs", "latency_p95_ci95_low_ms"),
            ("latencyP95Ci95HighMs", "latency_p95_ci95_high_ms"),
        ):
            output[output_name] = float(row[source_name]) if status == "formal" else None
        result.append(output)
    return result

def read_rank_rows(path: Path) -> list[dict[str, Any]]:
    source_rows = read_csv(path)
    result = []
    seen = set()
    for source in source_rows:
        rank = int(number(source.get("rank"), "rank", path, integer=True))
        if rank in seen:
            raise ValueError(f"duplicate rank: {rank}")
        seen.add(rank)
        seeds = int(number(source.get("seedCount"), "seedCount", path, integer=True))
        safe_seeds = int(number(source.get("safeSeedCount"), "safeSeedCount", path, integer=True))
        weighted = float(number(source.get("aggregateWeightedRecall"), "aggregateWeightedRecall", path))
        useful = float(number(source.get("aggregateUsefulCull"), "aggregateUsefulCull", path))
        safety = "safe" if seeds > 0 and safe_seeds == seeds and weighted > 0.99 else "unsafe"
        result.append({
            "rank": rank, "survival_dim": int(number(source.get("survivalDim"), "survivalDim", path, integer=True)),
            "runtime_feature_dim": int(number(source.get("runtimeFeatureDim"), "runtimeFeatureDim", path, integer=True)),
            "runtime_feature_mib": float(number(source.get("runtimeFeatureMiB"), "runtimeFeatureMiB", path)),
            "aggregate_useful_cull": useful,
            "aggregate_balanced_accuracy": float(number(source.get("aggregateBalancedAccuracy"), "aggregateBalancedAccuracy", path)),
            "aggregate_weighted_recall": weighted,
            "aggregate_precision": float(number(source.get("aggregatePrecision"), "aggregatePrecision", path)),
            "aggregate_bad_cull": float(number(source.get("aggregateBadCull"), "aggregateBadCull", path)),
            "avg_pred_count": float(number(source.get("avgPredCount"), "avgPredCount", path)),
            "seed_count": seeds, "safe_seed_count": safe_seeds,
            "safety_status": safety, "source": str(path.resolve()),
        })
    safe = [row for row in result if row["safety_status"] == "safe"]
    for row in result:
        row["pareto_optimal"] = row["safety_status"] == "safe" and not any(
            other["rank"] != row["rank"]
            and other["runtime_feature_mib"] <= row["runtime_feature_mib"]
            and other["aggregate_useful_cull"] >= row["aggregate_useful_cull"]
            and (
                other["runtime_feature_mib"] < row["runtime_feature_mib"]
                or other["aggregate_useful_cull"] > row["aggregate_useful_cull"]
            )
            for other in safe
        )
    return sorted(result, key=lambda row: row["rank"])

def complete_asset_rows(neural: list[dict[str, Any]], hzb: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = neural + hzb
    lossless = {row["scene"]: row for row in hzb if row["variant"] == "lossless"}
    neural_by_scene = {row["scene"]: row for row in neural}
    for row in all_rows:
        baseline = lossless[row["scene"]]
        row["transfer_to_lossless_ratio"] = row["transfer_bytes"] / baseline["transfer_bytes"]
        row["neural_to_lossless_ratio"] = neural_by_scene[row["scene"]]["transfer_bytes"] / baseline["transfer_bytes"]
    return sorted(all_rows, key=lambda row: (SCENES.index(row["scene"]), {"neural": 0, "geometry-shell-hzb": 1}[row["method"]], VARIANTS.index(row["variant"]) if row["method"] != "neural" else 0))

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if row.get(field) is None else row.get(field, "") for field in fields})

def fmt(value: Any, *, digits: int = 2) -> str:
    if value is None or value == "":
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

def fmt_bytes(value: Any) -> str:
    return "unavailable" if value is None or value == "" else f"{int(value):,}"

def fmt_ratio(value: Any) -> str:
    return "unavailable" if value is None or value == "" else f"{float(value) * 100:.2f}%"

def write_table_markdown(path: Path, asset_rows: Sequence[Mapping[str, Any]], runtime_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Table 4. 资产与运行时开销", "",
        "日期：2026-09-09。字节来自输入目录内的实际文件 `stat`；HZB 展开运行时内存为三条 meshopt 几何流解码后字节加固定 runtime payload。",
        "", "## 资产", "",
        "| 场景 | 方案 | 变体 | 传输字节 | 传输 MiB | 展开几何 MiB | 展开运行时 MiB | prototype | prototype 三角形 | 展开三角形 | occluder 实例 | 方案/ lossless | neural/ lossless |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in asset_rows:
        neural_ratio = fmt_ratio(row["neural_to_lossless_ratio"])
        if row["method"] == "neural":
            neural_ratio = f"**{neural_ratio}**"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            scene_label(row["scene"]), row["method"], row["variant"], fmt_bytes(row["transfer_bytes"]),
            fmt(row["transfer_mib"]), fmt(None if row["expanded_geometry_bytes"] is None else row["expanded_geometry_bytes"] / 2**20),
            fmt(None if row["expanded_memory_bytes"] is None else row["expanded_memory_bytes"] / 2**20),
            fmt(row["prototype_count"], digits=0), fmt(row["prototype_triangles"], digits=0),
            fmt(row["expanded_triangles"], digits=0), fmt(row["occluder_instance_count"], digits=0),
            fmt_ratio(row["transfer_to_lossless_ratio"]), neural_ratio,
        ))
    lines.extend([
        "", "神经资产相对同场景 lossless shell 的比例以最后一列粗体突出；HZB `equal-asset` 仍单独列出，它只表示按神经资产预算删除完整 primitive 的资产敏感性对照，不替代 lossless shell。神经资产的几何展开内存和三角形/occluder 统计不适用，保留为 `unavailable`。",
        "", "## 正式运行延迟", "",
        "| 场景 | 设备 | 后端 | 状态 | session | 样本 | 候选 mean/p95 | mean ms (95% CI) | p50 ms (95% CI) | p95 ms (95% CI) | 计时字段 | 原因 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ])
    for row in runtime_rows:
        def interval(prefix: str) -> str:
            value = row.get(prefix)
            low = row.get(prefix.replace("_ms", "_ci95_low_ms"))
            high = row.get(prefix.replace("_ms", "_ci95_high_ms"))
            return "unavailable" if value is None else f"{fmt(value)} ({fmt(low)}, {fmt(high)})"
        candidate = "unavailable" if row.get("candidate_mean") is None else f"{fmt(row['candidate_mean'])}/{fmt(row['candidate_p95'])}"
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            scene_label(row["scene"]), row["device"] or "unavailable", row["backend"] or "unavailable",
            row["status"], fmt(row["formal_sessions"], digits=0), fmt(row["sample_count"], digits=0), candidate,
            interval("latency_mean_ms"), interval("latency_p50_ms"), interval("latency_p95_ms"),
            row["timing_source"] or "unavailable", row["reason"] or "",
        ))
    lines.extend([
        "", "只有 `formal` 行填入 latency 数值，规则是完整 test workload、5 个 session、每 session 50 次 warmup 且后端门通过。并发 A6000 记录为 `smoke-excluded`，即使原始记录含 timing 也不进入正式统计；没有正式上传的组合为 `unavailable`。",
        "", "来源：rank/runtime CSV、四个 HZB `shell_meta.json` 与 `.offline.json`、以及两场景实际神经 runtime asset 目录，均由生成命令参数显式指定。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")

def plot_rank_pareto(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe = [row for row in rows if row["safety_status"] == "safe"]
    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.scatter(
        [row["runtime_feature_mib"] for row in rows],
        [row["aggregate_useful_cull"] for row in rows],
        c=["#0b7285" if row["safety_status"] == "safe" else "#adb5bd" for row in rows],
        s=[90 if row["pareto_optimal"] else 58 for row in rows],
        edgecolors="#212529", linewidths=0.7, zorder=3,
    )
    frontier = sorted((row for row in safe if row["pareto_optimal"]), key=lambda row: row["runtime_feature_mib"])
    if len(frontier) > 1:
        axis.plot([row["runtime_feature_mib"] for row in frontier], [row["aggregate_useful_cull"] for row in frontier], color="#0b7285", linewidth=1.5, zorder=2)
    for row in rows:
        axis.annotate(f"R={row['rank']}", (row["runtime_feature_mib"], row["aggregate_useful_cull"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    axis.set_xlabel("Fixed runtime feature table (MiB)")
    axis.set_ylabel("Aggregate useful cull")
    axis.set_title("Survival-field rank capacity Pareto")
    axis.set_ylim(bottom=min(row["aggregate_useful_cull"] for row in rows) - 0.01)
    axis.grid(axis="both", alpha=0.22)
    axis.text(0.02, 0.03, "Teal: source safety gate passed; line: Pareto frontier", transform=axis.transAxes, fontsize=8, color="#495057")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)

def build_outputs(
    paper_results: Path = PAPER_RESULTS,
    output_dir: Path | None = None,
    neural_assets: Mapping[str, Path] | None = None,
    hzb_dir: Path | None = None,
) -> dict[str, Any]:
    paper_results = paper_results.resolve()
    output_dir = (output_dir or paper_results / "figures").resolve()
    hzb_dir = (hzb_dir or paper_results / "hzb").resolve()
    neural_assets = neural_assets or {
        "hkust": ROOT / "slm2viewer/assets/neural_instance_culling/pvs_mainline_v4",
        "ifcbench": ROOT / "slm2viewer/assets/neural_instance_culling/pvs_mainline_v4_ifcbench",
    }
    if set(neural_assets) != set(SCENES):
        raise ValueError(f"neural assets must specify exactly {SCENES}")
    neural = [read_neural_asset(scene, Path(neural_assets[scene]).resolve()) for scene in SCENES]
    hzb = [read_hzb_asset(scene, variant, hzb_dir) for scene in SCENES for variant in VARIANTS]
    assets = complete_asset_rows(neural, hzb)
    neural_by_scene = {row["scene"]: row for row in neural}
    lossless_by_scene = {row["scene"]: row for row in hzb if row["variant"] == "lossless"}
    runtime = read_runtime_rows(paper_results / "mobile_runtime/runtime_paper_summary.csv", neural_by_scene, lossless_by_scene)
    rank = read_rank_rows(paper_results / "rank_sweep/rank_capacity.csv")
    combined = assets + runtime
    write_csv(output_dir / "table4_asset_runtime.csv", combined, TABLE_FIELDS)
    write_table_markdown(output_dir / "table4_asset_runtime.md", assets, runtime)
    write_csv(output_dir / "rank_capacity_pareto.csv", rank, RANK_FIELDS)
    plot_rank_pareto(output_dir / "rank_capacity_pareto", rank)
    return {
        "output_dir": str(output_dir), "asset_rows": len(assets), "runtime_rows": len(runtime),
        "formal_latency_rows": sum(row["status"] == "formal" for row in runtime),
        "smoke_excluded_rows": sum(row["status"] == "smoke-excluded" for row in runtime),
        "unavailable_rows": sum(row["status"] == "unavailable" for row in runtime),
        "pareto_ranks": [row["rank"] for row in rank if row["pareto_optimal"]],
    }

def parse_scene_assets(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--neural-asset must be SCENE=PATH: {value}")
        scene, raw_path = value.split("=", 1)
        key = scene_key(scene)
        if key in result:
            raise ValueError(f"duplicate neural asset scene: {key}")
        result[key] = Path(raw_path).expanduser()
    return result

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-results-dir", type=Path, default=PAPER_RESULTS)
    parser.add_argument("--hzb-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--neural-asset", action="append", default=[], metavar="SCENE=PATH")
    args = parser.parse_args(argv)
    neural_assets = parse_scene_assets(args.neural_asset) if args.neural_asset else None
    print(json.dumps(build_outputs(args.paper_results_dir, args.output_dir, neural_assets, args.hzb_dir), indent=2))

if __name__ == "__main__":
    main()
