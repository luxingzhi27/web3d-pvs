#!/usr/bin/env python3
"""Freeze and report the independent M5 subpose-robust image evaluation.

The renderer writes the raw aggregate image summary.  This entry point adds
the experiment-level audit around that output: strict calibration provenance,
the hardware-GPU gate, dense-subpose counts, and the pre-registered image
quality decision.  It never predicts, scans thresholds, reads test, or edits
the renderer output.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_OUTPUT = (
    ROOT
    / "neural_instance_culling"
    / "benchmark"
    / "out"
    / "m5_subpose_robust_image_dense_hw_20260804"
)
DEFAULT_MODEL_ROOT = ROOT / "neural_instance_culling" / "model" / "out"
DEFAULT_VARIANT = "pvs_m5_subpose_robust_v1_hkust_spatial_fov66"

IMAGE_COUNT_FIELDS = (
    "totalPixels",
    "validReferencePixels",
    "backgroundReferencePixels",
    "errorPixels",
    "missPixels",
    "wrongInstancePixels",
    "extraPixels",
)
BATCH_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_(?P<split>validation|calibration)$")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _rate(totals: dict[str, int], numerator: str, denominator: str) -> float:
    return float(totals[numerator]) / max(1, int(totals[denominator]))


def _merge(target: dict[str, int], metrics: dict[str, Any]) -> None:
    for field in IMAGE_COUNT_FIELDS:
        target[field] += int(metrics.get(field, 0))


def _empty_totals() -> dict[str, int]:
    return {field: 0 for field in IMAGE_COUNT_FIELDS}


def _finalize(totals: dict[str, int], sample_count: int) -> dict[str, Any]:
    return {
        **totals,
        "sampleCount": int(sample_count),
        "PER": _rate(totals, "errorPixels", "validReferencePixels"),
        "missPixelRate": _rate(totals, "missPixels", "validReferencePixels"),
        "wrongInstancePixelRate": _rate(totals, "wrongInstancePixels", "validReferencePixels"),
        "extraPixelRateOverImage": _rate(totals, "extraPixels", "totalPixels"),
    }


def _require_strict_calibration(model_root: Path, variant: str, seed: int) -> dict[str, Any]:
    directory = model_root / f"{variant}_seed{int(seed)}_full40_strict_calibration"
    path = directory / "calibration_ready_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = read_json(path)
    selected = ((payload.get("calibration") or {}).get("selected") or {})
    checks = {
        "protocol": payload.get("protocol") == "calibration_ready_pre_test",
        "testEvaluationCount": int(payload.get("testEvaluationCount", -1)) == 0,
        "selectionStatus": (payload.get("calibration") or {}).get("selectionStatus") == "safe",
        "poseRecall": float(selected.get("pose_recall", -1.0)) >= 0.95,
        "weightedRecallPoint": float(selected.get("pose_weighted_recall", -1.0)) >= 0.9925,
        "weightedRecallLcb": float(selected.get("weighted_recall_lower_confidence_bound", -1.0)) > 0.99,
    }
    if not all(checks.values()):
        raise ValueError(f"strict calibration safety audit failed for seed {seed}: {checks}")
    return {
        "seed": int(seed),
        "directory": str(directory),
        "checkpoint": str(directory / "best.pt"),
        "threshold": float(payload.get("frozenThreshold")),
        "poseRecall": float(selected["pose_recall"]),
        "weightedRecall": float(selected["pose_weighted_recall"]),
        "weightedRecallLowerConfidenceBound": float(selected["weighted_recall_lower_confidence_bound"]),
        "testEvaluationCount": int(payload["testEvaluationCount"]),
        "checks": checks,
    }


def _audit_hardware(image_output: Path, payload: dict[str, Any]) -> dict[str, Any]:
    gate = payload.get("browserGpuGate") or payload.get("gpuGate") or {}
    backend = payload.get("browserGpuBackend") or payload.get("gpuBackend") or {}
    if not bool(gate.get("required")) or not bool(gate.get("hardware")):
        raise ValueError(f"formal M5-v2 requires a passing hardware GPU gate: {gate}")
    if gate.get("softwareMarkers"):
        raise ValueError(f"formal M5-v2 reported a software renderer marker: {gate}")
    renderer = " ".join(str(backend.get(key, "")) for key in ("vendor", "renderer", "version"))
    if not renderer.strip() or re.search(r"swiftshader|llvmpipe|softpipe|swrast", renderer, re.I):
        raise ValueError(f"formal M5-v2 reported an invalid renderer: {renderer!r}")
    snapshots = sorted(image_output.glob("nvidia_smi_snapshot_*.csv"))
    pmon = sorted(image_output.glob("nvidia_smi_pmon_snapshot_*.txt"))
    if not snapshots or not pmon:
        raise FileNotFoundError("formal M5-v2 requires nvidia-smi and nvidia-smi pmon evidence")
    return {
        "backend": backend,
        "gate": gate,
        "nvidiaSmiSnapshot": str(snapshots[-1]),
        "nvidiaSmiPmonSnapshot": str(pmon[-1]),
    }


def summarize(
    image_output: Path,
    model_root: Path = DEFAULT_MODEL_ROOT,
    variant: str = DEFAULT_VARIANT,
    seeds: tuple[int, ...] = (20260801, 20260802, 20260803),
) -> dict[str, Any]:
    image_output = image_output.resolve()
    base = read_json(image_output / "summary.json")
    if base.get("testRead") is not False:
        raise ValueError("M5-v2 image summary must explicitly state testRead=false")
    if base.get("formalImageEvaluationReady"):
        raise ValueError("M5-v2 image output cannot be promoted by the renderer itself")
    if base.get("renderFovYDeg") != 60 or base.get("modelInputFovYDeg") != 66:
        raise ValueError("M5-v2 image output violates the 60°/66° camera contract")
    hardware = _audit_hardware(image_output, base)

    calibration = [_require_strict_calibration(model_root, variant, seed) for seed in seeds]
    expected_ids = {
        f"{variant}_seed{seed}_{split}"
        for seed in seeds
        for split in ("validation", "calibration")
    }
    rows = base.get("batches")
    if not isinstance(rows, list) or {str(row.get("batchId")) for row in rows} != expected_ids:
        raise ValueError("M5-v2 batch IDs do not match the registered variant/seed/split matrix")

    grouped: dict[tuple[int, str], dict[str, Any]] = {}
    variant_totals = _empty_totals()
    variant_samples = 0
    split_totals: dict[str, dict[str, int]] = defaultdict(_empty_totals)
    split_samples: dict[str, int] = defaultdict(int)
    per_seed: dict[int, dict[str, Any]] = {}
    viewcell_p95: list[float] = []
    viewcell_mean: list[float] = []
    for row in rows:
        batch_id = str(row.get("batchId"))
        match = BATCH_RE.match(batch_id)
        if not match or match.group("variant") != variant:
            raise ValueError(f"unexpected M5-v2 batch id: {batch_id}")
        seed = int(match.group("seed"))
        split = match.group("split")
        sample_count = int(row.get("sampleCount", 0))
        metrics = row.get("imageMetrics") or {}
        if sample_count <= 0 or int(metrics.get("evaluatedSubposeCount", -1)) != sample_count:
            raise ValueError(f"sample count mismatch in {batch_id}")
        if int(metrics.get("renderFailedSubposeCount", -1)) != 0 or int(metrics.get("missingGlbSubposeCount", -1)) != 0:
            raise ValueError(f"render failure in {batch_id}")
        _merge(variant_totals, metrics)
        variant_samples += sample_count
        _merge(split_totals[split], metrics)
        split_samples[split] += sample_count
        seed_result = per_seed.setdefault(seed, {"seed": seed, "splits": {}})
        seed_totals = seed_result["splits"].setdefault(split, {"totals": _empty_totals(), "sampleCount": 0})
        _merge(seed_totals["totals"], metrics)
        seed_totals["sampleCount"] += sample_count
        view = row.get("viewCellImageMetrics") or {}
        viewcell_p95.append(float(view.get("viewCellMissPixelRateP95", 0.0)))
        viewcell_mean.append(float(view.get("viewCellMissPixelRateMean", 0.0)))
        grouped[(seed, split)] = row

    for seed_result in per_seed.values():
        seed_result["splits"] = {
            split: _finalize(value["totals"], value["sampleCount"])
            for split, value in seed_result["splits"].items()
        }
        merged = _empty_totals()
        count = 0
        for value in seed_result["splits"].values():
            _merge(merged, value)
            count += int(value["sampleCount"])
        seed_result["all"] = _finalize(merged, count)

    image_metrics = _finalize(variant_totals, variant_samples)
    split_metrics = {
        split: _finalize(split_totals[split], split_samples[split])
        for split in sorted(split_totals)
    }
    mean_miss = image_metrics["missPixelRate"]
    p95_miss = sorted(viewcell_p95)[int(0.95 * (len(viewcell_p95) - 1))] if viewcell_p95 else 0.0
    quality = {
        "meanMissPixelRate": mean_miss,
        "meanMissPixelRateMax": 0.005,
        "viewCellMissPixelRateP95": p95_miss,
        "viewCellMissPixelRateP95Max": 0.01,
        "meanGatePassed": mean_miss < 0.005,
        "p95GatePassed": p95_miss < 0.01,
        "status": "Go" if mean_miss < 0.005 and p95_miss < 0.01 else "No-Go",
    }
    return {
        "schema": "m5-subpose-robust-image-dense-hw-summary-v1",
        "created": datetime.now().isoformat(timespec="seconds"),
        "imageOutput": str(image_output),
        "variant": variant,
        "seeds": [int(seed) for seed in seeds],
        "splits": ["validation", "calibration"],
        "testRead": False,
        "cameraContract": {"renderFovYDeg": 60, "modelInputFovYDeg": 66},
        "denseSubpose": {
            "batchCount": len(rows),
            "sampleCount": variant_samples,
            "viewCellBatchP95Count": len(viewcell_p95),
            "viewCellMissPixelRateMeanOfBatchMeans": sum(viewcell_mean) / max(1, len(viewcell_mean)),
        },
        "hardwareGpu": hardware,
        "strictCalibration": calibration,
        "imageMetrics": image_metrics,
        "splitMetrics": split_metrics,
        "perSeed": [per_seed[seed] for seed in sorted(per_seed)],
        "qualityGate": quality,
        "thresholdsUntouched": True,
        "candidateSetChanged": False,
        "gtChanged": False,
        "note": "Calibration-frozen validation/calibration image evidence; test remains sealed.",
    }


def _percent(value: float) -> str:
    return f"{100.0 * float(value):.4f}%"


def write_report(summary: dict[str, Any], path: Path) -> None:
    image = summary["imageMetrics"]
    quality = summary["qualityGate"]
    calibration_text = ", ".join(
        f"seed {row['seed']} threshold {row['threshold']:.8g}"
        for row in summary["strictCalibration"]
    )
    lines = [
        "# M5-v2 View-Cell Subpose 鲁棒性图像评价",
        "",
        "日期：2026-08-04",
        f"状态：`{quality['status']}`",
        "",
        "## 评价边界",
        "",
        "本报告只汇总 validation/calibration 的完整 dense subpose。三个 seed 使用各自独立的 calibration 冻结阈值；未重新扫描阈值、未改变候选集合、未补入 GT 可见实例、未使用前端白名单，且 `testRead=false`。真实渲染相机为 60°，模型后退相机和输入为 66°。",
        "",
        f"- 变体：`{summary['variant']}`；seed 数：{len(summary['seeds'])}；批次：{summary['denseSubpose']['batchCount']}；dense subpose：{summary['denseSubpose']['sampleCount']:,}；",
        f"- 严格校准：{calibration_text}；",
        "- 评价阶段未读取 test；预测阈值由 calibration 单独冻结。",
        "",
        "## 硬件 GPU 证据",
        "",
        "正式浏览器路径通过硬件门。页面实际回报的后端、`gpuGate`、Chrome 日志和同时间段的 `nvidia-smi`/`pmon` 快照必须同时存在；SwiftShader、llvmpipe、softpipe、swrast 或空后端会使汇总失败。",
        "",
        f"- renderer：`{summary['hardwareGpu']['backend'].get('renderer', '')}`；",
        f"- `gpuGate.hardware`：`{summary['hardwareGpu']['gate'].get('hardware')}`；",
        f"- `nvidia-smi`：`{summary['hardwareGpu']['nvidiaSmiSnapshot']}`；`pmon`：`{summary['hardwareGpu']['nvidiaSmiPmonSnapshot']}`。",
        "",
        "以后正式采样、图像评价或 HZB 构建不得因为能创建 WebGL context 就判定为 GPU 执行；必须复用 `docs/current/hardware_gpu_execution_policy.md` 的硬件门。显式软件回退只允许小规模语义调试，不能生成或覆盖正式结果。",
        "",
        "## 图像指标",
        "",
        "| 指标 | 全部 validation/calibration | 含义 |",
        "|---|---:|---|",
        f"| 样本数 | {image['sampleCount']:,} | dense subpose 图像数 |",
        f"| miss-pixel rate | {_percent(image['missPixelRate'])} | 预测漏掉参考可见实例造成的像素比例 |",
        f"| wrong-ID pixel rate | {_percent(image['wrongInstancePixelRate'])} | 被错误实例覆盖的像素比例 |",
        f"| PER | {_percent(image['PER'])} | 有效参考像素中的错误比例 |",
        f"| extra-pixel rate | {_percent(image['extraPixelRateOverImage'])} | 预测额外像素占整幅图的比例 |",
        "",
        "| Split | Samples | Miss-pixel rate | Wrong-ID pixel rate | PER |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, metrics in summary["splitMetrics"].items():
        lines.append(
            f"| {split} | {metrics['sampleCount']:,} | {_percent(metrics['missPixelRate'])} | "
            f"{_percent(metrics['wrongInstancePixelRate'])} | {_percent(metrics['PER'])} |"
        )
    lines.extend([
        "",
        "## 质量门结论",
        "",
        f"预登记门槛为 mean miss-pixel rate `<0.5%`，并要求 view-cell dense subpose miss-pixel rate 的 p95 `<1%`。本轮全量 mean 为 `{_percent(quality['meanMissPixelRate'])}`，p95 为 `{_percent(quality['viewCellMissPixelRateP95'])}`，因此结论为 `{quality['status']}`。",
        "",
        "该结论描述视点区域位置扰动下的画面安全性，不等同于普通集合 recall，也不能由 useful cull 或较高 precision 单独替代。当前若为 No-Go，应继续改进视点区域鲁棒监督或保守查询，而不能调低安全要求或改候选集合。",
        "",
        "## 可复现产物",
        "",
        f"- 原始图像输出：`{summary['imageOutput']}`；",
        "- 本汇总：`m5_v2_summary.json`；",
        "- 硬件 GPU 政策：`docs/current/hardware_gpu_execution_policy.md`；",
        "- 训练与严格校准目录保留各自 checkpoint、日志和 `testEvaluationCount=0` 摘要。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def self_test() -> None:
    assert _finalize({**_empty_totals(), "validReferencePixels": 10, "missPixels": 2}, 3)["missPixelRate"] == 0.2
    assert BATCH_RE.match("demo_seed20260801_validation")
    print("m5 subpose robust image summarizer self-test: OK")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-output", type=Path, default=DEFAULT_IMAGE_OUTPUT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--variant", default=DEFAULT_VARIANT)
    parser.add_argument("--seeds", default="20260801,20260802,20260803")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Markdown output path; defaults to m5_v2_summary.md beside --image-output.",
    )
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    seeds = tuple(int(value) for value in str(args.seeds).split(",") if value.strip())
    summary = summarize(args.image_output, args.model_root, args.variant, seeds)
    output = args.image_output / "m5_v2_summary.json"
    report = args.report or (args.image_output / "m5_v2_summary.md")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(summary, report)
    print(json.dumps({"summary": str(output), "report": str(report), "status": summary["qualityGate"]["status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
