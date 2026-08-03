#!/usr/bin/env python3
"""Render the checked M4-v2 JSON artifacts as a reproducible Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from m4_formal_matrix_v2_utils import ALL_VARIANTS, DIAGNOSTIC_METRICS, FACTOR_EFFECTS, FACTOR_VARIANTS, SEEDS, STAGED_EFFECTS, UNAVAILABLE_METRICS


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def f(value: Any, digits: int = 6) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, bool):
        return "是" if value else "否"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def member_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    return [summary["members"][f"{variant}:{seed}"] for variant in ALL_VARIANTS for seed in SEEDS]


def case_text(summary: dict[str, Any]) -> str:
    records = member_rows(summary)
    unsafe = [record for record in records if not record.get("safety", {}).get("qualifiedSafetyWorkpoint", False)]
    candidates = unsafe or records
    record = max(candidates, key=lambda item: float(item["poseMacro"]["useful_cull"]))
    pose = record["poseMacro"]
    aggregate = record["aggregate"]
    return (
        f"例如 `{record['variant']}` seed `{record['seed']}` 在其记录中 pose 宏平均 useful cull 为 "
        f"{f(pose['useful_cull'])}，但 pose recall 为 {f(pose['recall'])}、weighted recall 为 "
        f"{f(pose['weighted_recall'])}，bad cull 为 {f(pose['bad_cull'])}。合并所有 pose 后，"
        f"aggregate useful cull 为 {f(aggregate['useful_cull'])}、bad cull 为 {f(aggregate['bad_cull'])}，"
        "这说明大量 TN 可能掩盖 FN；useful cull 的升高不能解释为更好的模型，必须同时满足召回和画面安全约束。"
    )


def render(summary: dict[str, Any], route: dict[str, Any]) -> str:
    route_name = route.get("route", "unknown")
    lines = [
        "# M4-v2 完整因子消融评价",
        "",
        "日期：2026-08-03  ",
        "评价阶段：validation-only；test split 未读取",
        "",
        "## 摘要",
        "",
        f"本轮将方向遮挡代理和显式遮挡抑制头作为两个独立因子，形成 A/B/C/D 四个正式变体，并保留 AABB 与离线几何基线作为逐级参考。新的路线判定为 `{route_name}`。判定顺序是安全性、分类/剔除效果、系统效果，useful cull 不再单独决定路线。",
        "",
        "方向代理的独立作用通过 `B-A` 和 `D-C` 同时观察；显式抑制通过 `C-A` 和 `D-B` 观察；交互项为 `D-B-C+A`。所有成员使用相同的 validation pose、后退相机候选、GT 和候选哈希。",
        "",
        "## 数据与阈值",
        "",
        f"场景为 HKUST v3，validation pose 数为 `{summary['poseCount']}`，候选身份摘要为 `{summary['candidateIdentity']['candidateDigest']}`。模型与后退候选相机使用 66°，真实渲染约定为 60°。每个 checkpoint 的阈值只从自己的 calibration split 冻结；安全工作点要求 pose recall >= 0.95、weighted recall > 0.99 且 weighted recall 单侧 95% 下界 > 0.99。",
        "",
        "## 指标口径",
        "",
        "对每个 pose，候选集合记为 C，真实可见集合记为 G，预测集合记为 P。TP=P∩G，FP=P-G，FN=G-P，TN=C-(P∪G)。pose 宏平均先逐 pose 计算指标再平均；aggregate 先合并全部 pose 的计数再计算比例。",
        "",
        "- 画面安全：recall、weighted recall、bad cull。weighted recall 使用 `visible_weights`，它是可见重要性证据，不是真实像素覆盖率。",
        "- 分类诊断：precision、F1、Jaccard、accuracy、balanced accuracy、specificity。",
        "- 剔除效率：useful cull=TN/candidate，只统计正确剔除的不可见候选；bad cull=FN/candidate，统计错误剔除的真实可见候选。",
        "- 资源诊断：平均预测数、预测/候选、预测/GT。GLB 字节、像素和浏览器延迟必须有同位姿的额外证据，不能由 useful cull 推断。",
        "",
        "## 成员结果：pose 宏平均",
        "",
        "| 变体 | seed | 阈值 | pose recall | weighted recall | weighted LCB | pose precision | pose F1 | pose accuracy | balanced accuracy | useful cull | bad cull | 平均预测数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in member_rows(summary):
        macro = record["poseMacro"]
        lines.append(
            f"| `{record['variant']}` | {record['seed']} | {f(record['threshold'], 8)} | {f(macro['recall'])} | "
            f"{f(macro['weighted_recall'])} | {f(macro['weighted_recall_lower_confidence_bound'])} | {f(macro['precision'])} | "
            f"{f(macro['f1'])} | {f(macro['accuracy'])} | {f(macro['balanced_accuracy'])} | {f(macro['useful_cull'])} | "
            f"{f(macro['bad_cull'])} | {f(macro['avg_pred_count'], 2)} |"
        )
    lines.extend([
        "",
        "## 成员结果：aggregate",
        "",
        "aggregate 先合并全部 validation pose 的 TP、FP、FN、TN 和可见重要性权重，再计算比例；它不是 pose 宏平均的替代口径。",
        "",
        "| 变体 | seed | 阈值 | aggregate recall | aggregate weighted recall | aggregate weighted LCB | aggregate precision | aggregate F1 | aggregate accuracy | aggregate balanced accuracy | aggregate useful cull | aggregate bad cull | 平均预测数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in member_rows(summary):
        aggregate = record["aggregate"]
        lines.append(
            f"| `{record['variant']}` | {record['seed']} | {f(record['threshold'], 8)} | {f(aggregate['recall'])} | "
            f"{f(aggregate['weighted_recall'])} | {f(aggregate['weighted_recall_lower_confidence_bound'])} | {f(aggregate['precision'])} | "
            f"{f(aggregate['f1'])} | {f(aggregate['accuracy'])} | {f(aggregate['balanced_accuracy'])} | {f(aggregate['useful_cull'])} | "
            f"{f(aggregate['bad_cull'])} | {f(aggregate['avg_pred_count'], 2)} |"
        )
    lines.extend([
        "",
        "## 诊断工作点",
        "",
        "每个 checkpoint 的 best-F1、最高 precision 和原始冻结阈值都来自该 checkpoint 自己的 calibration 记录。它们只用于诊断阈值敏感性，不参与安全主工作点排名，也没有读取 validation/test 重新选阈值。",
        "",
        "| 变体 | seed | 安全工作点状态 | 安全阈值 | best-F1 阈值 | best-F1 precision | best-F1 recall | best-F1 weighted recall | best-F1 F1 | 最高 precision 阈值 | 最高 precision | 对应 recall | 对应 weighted recall | 原始冻结阈值 |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in member_rows(summary):
        workpoint = record.get("calibrationWorkpoint") or {}
        diagnostics = workpoint.get("diagnosticWorkpoints") or {}
        best = diagnostics.get("bestF1") or {}
        highest = diagnostics.get("highestPrecision") or {}
        original = diagnostics.get("originalFrozenThreshold") or {}
        lines.append(
            f"| `{record['variant']}` | {record['seed']} | {workpoint.get('status', 'not_available')} | {f(record['threshold'], 8)} | "
            f"{f(best.get('threshold'), 8)} | {f(best.get('pose_precision'))} | {f(best.get('pose_recall'))} | "
            f"{f(best.get('pose_weighted_recall'))} | {f(best.get('pose_f1'))} | {f(highest.get('threshold'), 8)} | "
            f"{f(highest.get('pose_precision'))} | {f(highest.get('pose_recall'))} | {f(highest.get('pose_weighted_recall'))} | "
            f"{f(original.get('threshold'), 8)} |"
        )
    lines.extend([
        "",
        "每个成员的平均 TP、FP、FN、TN、预测/候选和预测/GT 已同时保存在汇总 JSON 的 `poseMacro` 与 `aggregate` 对象中；GLB、像素和浏览器成本若为 `not_available`，不参与路线排名。",
        "",
    ])

    lines.extend([
        "## 完整成员诊断",
        "",
        "下表把每个成员的两种统计口径和完整分类/剔除指标直接列出。`visual utility recall` 没有统一视觉效用监督，因此对所有成员明确记为 `not_available`，不能用 weighted recall 替代。",
        "",
        "| 口径 | 变体 | seed | precision | recall | weighted recall | visual utility recall | F1 | Jaccard | accuracy | balanced accuracy | specificity | useful cull | bad cull | avg TP | avg FP | avg FN | avg TN | avg pred | pred/candidate | pred/GT |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for record in member_rows(summary):
        for scope_name, scope_label in (("poseMacro", "pose_macro"), ("aggregate", "aggregate")):
            metrics = record[scope_name]
            lines.append(
                f"| {scope_label} | `{record['variant']}` | {record['seed']} | {f(metrics['precision'])} | "
                f"{f(metrics['recall'])} | {f(metrics['weighted_recall'])} | not_available | {f(metrics['f1'])} | "
                f"{f(metrics['jaccard'])} | {f(metrics['accuracy'])} | {f(metrics['balanced_accuracy'])} | "
                f"{f(metrics['specificity'])} | {f(metrics['useful_cull'])} | {f(metrics['bad_cull'])} | "
                f"{f(metrics['avg_tp_count'], 2)} | {f(metrics['avg_fp_count'], 2)} | {f(metrics['avg_fn_count'], 2)} | "
                f"{f(metrics['avg_tn_count'], 2)} | {f(metrics['avg_pred_count'], 2)} | "
                f"{f(metrics['pred_over_candidate'])} | {f(metrics['pred_over_gt'])} |"
            )
    lines.extend([
        "",
        "## 运行与资源成本",
        "",
        "固定特征表字节数和 checkpoint/运行时特征文件来源保存在每个成员的 JSON 记录中；当前 intervention 没有保存逐 pose 前向计时、GLB 集合/字节曲线、同位姿图像或浏览器采样，因此下列系统指标对所有成员均为不可用，不能推断为零收益：",
        "",
        "| 指标 | 状态 | 原因 |",
        "|---|---|---|",
    ])
    for name in ("glb_count_reduction", "glb_byte_reduction", "equal_visual_utility_glb_bytes", "forward_latency_ms", "webgpu_latency_ms", "main_thread_ms", "runtime_memory_bytes", "miss_pixel_rate", "wrong_id_pixel_rate", "extra_pixel_rate"):
        lines.append(f"| `{name}` | not_available | {UNAVAILABLE_METRICS[name]} |")
    lines.extend([
        "",
        "固定特征表大小仍按成员分别记录；它是离线资产体积，不等同于前向延迟或移动设备内存峰值。",
        "",
    ])

    lines.extend(["## 因子差值与置信区间", "", "以下每项均为右侧因子组合减去左侧因子组合；区间由按 seed 聚类、seed 内按 pose 重采样的 10,000 次 paired bootstrap 得到。", ""])
    for effect_name in FACTOR_EFFECTS:
        lines.extend([f"### `{effect_name}`", "", "| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |", "|---|---|---:|---|---|---|"])
        effect = summary["factorEffects"][effect_name]["comparisons"]
        for scope in ("pose_macro", "aggregate"):
            for metric in DIAGNOSTIC_METRICS:
                item = effect[scope][metric]
                lines.append(
                    f"| {scope} | `{metric}` | {f(item['mean_delta'])} | [{f(item['ci95'][0])}, {f(item['ci95'][1])}] | "
                    f"{item['direction']} | {f(item['crosses_zero'])} |"
                )
        lines.append("")

    lines.extend([
        "## 逐级输入表征增益",
        "",
        "以下比较只用于展示从 AABB 到离线几何、再到上下文表征的逐级变化，不参与方向代理的三层路线判定。它们仍使用相同的 validation pose、候选集合和 paired bootstrap。",
        "",
    ])
    for effect_name in STAGED_EFFECTS:
        lines.extend([f"### `{effect_name}`", "", "| 统计口径 | 指标 | 差值 | 95% CI | 方向 | 是否跨零 |", "|---|---|---:|---|---|---|"])
        effect = summary["stagedEffects"][effect_name]["comparisons"]
        for scope in ("pose_macro", "aggregate"):
            for metric in DIAGNOSTIC_METRICS:
                item = effect[scope][metric]
                lines.append(
                    f"| {scope} | `{metric}` | {f(item['mean_delta'])} | [{f(item['ci95'][0])}, {f(item['ci95'][1])}] | "
                    f"{item['direction']} | {f(item['crosses_zero'])} |"
                )
        lines.append("")

    safety = route["safetyLayer"]
    lines.extend([
        "## 路线三层判定",
        "",
        f"安全层：`{'通过' if safety['passed'] else '未通过'}`。B-A 和 D-C 都要求 recall、weighted recall 没有超过预登记容忍范围的下降，bad cull 区间上界不超过预登记安全增量，并且所有因子成员都存在合格安全工作点。",
        f"分类/剔除层：`{'通过' if route['classificationCullLayer']['passed'] else '未通过'}`。只有 precision、balanced accuracy、F1、useful cull 的区间下界为正，或平均预测数量的区间上界为负，才算有预测贡献。",
        f"系统层：`{'通过' if route['systemLayer']['passed'] else '未实现/未通过'}`。当前 intervention 记录没有同位姿图像或 GLB 资源曲线，因此不能将资源收益写成已证实结论。",
        "",
        "## 方向代理的五个独立问题",
        "",
        "路线结论把机制使用、分类效果、安全剔除、画面质量和资源成本分开记录。一个输入改变了 logits，只能说明模型读到了它，不能自动说明它改善了最终系统。",
        "",
        "| 问题 | 本轮判定 | 证据边界 |",
        "|---|---|---|",
        "| 方向代理是否被模型使用 | 是，B/D 启用、A/C 关闭；B-A 与 D-C 是对应的配对比较 | 这是因子结构和 intervention 诊断的机制证据，不等同于效果提升 |",
        f"| 是否改善分类准确性 | {'是' if route['classificationCullLayer']['passed'] else '未证明'} | 只有 precision、balanced accuracy、F1、useful cull 或平均预测数的 paired bootstrap 区间满足预注册方向，才算通过 |",
        f"| 是否改善安全约束下的剔除效率 | {'是' if route['interpretation'].get('directionProxySafetyConstrainedCullContribution') else '未证明'} | 必须先通过 B-A 和 D-C 的 recall、weighted recall、bad cull 安全层，再看分类/剔除层 |",
        f"| 是否改善图像质量 | {'是' if route['interpretation'].get('directionProxyImageContribution') else '未实现或未证明'} | 当前矩阵没有同位姿 miss-pixel、wrong-ID 或 p95 图像结果 |",
        f"| 是否改善下载和运行时成本 | {'是' if route['interpretation'].get('directionProxyResourceContribution') else '未实现或未证明'} | 当前 intervention 没有 GLB 字节、首屏时间、WebGPU 或主线程配对记录 |",
        "",
        "## 反例解释",
        "",
        case_text(summary),
        "",
        "## 系统指标边界",
        "",
    ])
    for name, reason in UNAVAILABLE_METRICS.items():
        lines.append(f"- `{name}`：{reason}")
    lines.extend([
        "",
        "## 可复现产物",
        "",
        f"- 汇总：`{summary.get('schema')}`，文件为 benchmark 输出目录中的 `summary.json`。",
        "- 路线 JSON 和 Markdown 使用独立的 `m4_formal_route_decision_v2` 名称。",
        "- 旧 M4 summary、旧 route JSON 和旧报告未修改；本报告不改变默认模型、默认前端资产或旧 Route B 结论。",
        "",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    route = json.loads(args.route.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(summary, route), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "route": route.get("route")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
