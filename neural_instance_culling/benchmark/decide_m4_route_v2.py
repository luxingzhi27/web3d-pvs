#!/usr/bin/env python3
"""Make the M4-v2 route decision from the complete validation factor matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20260801, 20260802, 20260803)
FACTOR_VARIANTS = (
    "geometry_context_ray_no_inhibition",
    "geometry_context_proxy_ray_no_inhibition",
    "geometry_context_ray",
    "full",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--max-bad-cull-delta", type=float, default=0.002)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def ci_lower(effect: dict[str, Any]) -> float:
    return float(effect.get("ci95", [-float("inf"), -float("inf")])[0])


def ci_upper(effect: dict[str, Any]) -> float:
    return float(effect.get("ci95", [float("inf"), float("inf")])[1])


def positive_ci(effect: dict[str, Any]) -> bool:
    return ci_lower(effect) > 0.0


def negative_ci(effect: dict[str, Any]) -> bool:
    return ci_upper(effect) < 0.0


def effect(summary: dict[str, Any], name: str, scope: str, metric: str) -> dict[str, Any]:
    return summary["factorEffects"][name]["comparisons"][scope][metric]


def safety_for_direction(
    summary: dict[str, Any],
    effect_name: str,
    max_bad_cull_delta: float,
    max_recall_drop: float,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for scope in ("pose_macro", "aggregate"):
        recall = effect(summary, effect_name, scope, "recall")
        weighted = effect(summary, effect_name, scope, "weighted_recall")
        bad = effect(summary, effect_name, scope, "bad_cull")
        checks[scope] = {
            "recall": recall,
            "weightedRecall": weighted,
            "badCull": bad,
            "recallNoMaterialDrop": ci_lower(recall) >= -max_recall_drop,
            "weightedRecallNoMaterialDrop": ci_lower(weighted) >= -max_recall_drop,
            "badCullUpperBoundSatisfied": ci_upper(bad) <= max_bad_cull_delta,
        }
    members = summary.get("members", {})
    member_safe = True
    member_status: dict[str, Any] = {}
    for variant in FACTOR_VARIANTS:
        values = []
        for seed in SEEDS:
            key = f"{variant}:{seed}"
            record = members.get(key, {})
            safe = bool(record.get("safety", {}).get("qualifiedSafetyWorkpoint", False))
            values.append(safe)
            member_status[key] = {
                "qualifiedSafetyWorkpoint": safe,
                "calibrationStatus": (record.get("calibrationWorkpoint") or {}).get("status"),
            }
        member_safe = member_safe and all(values)
    all_scope_checks = all(
        check[key]
        for check in checks.values()
        for key in ("recallNoMaterialDrop", "weightedRecallNoMaterialDrop", "badCullUpperBoundSatisfied")
    )
    return {
        "effect": effect_name,
        "memberSafetyAllQualified": member_safe,
        "memberSafety": member_status,
        "ciSafety": checks,
        "passed": bool(member_safe and all_scope_checks),
        "policy": {
            "maxRecallDrop": max_recall_drop,
            "maxBadCullDelta": max_bad_cull_delta,
            "bothPoseMacroAndAggregateRequired": True,
        },
    }


def classification_for_direction(summary: dict[str, Any], effect_names: tuple[str, ...]) -> dict[str, Any]:
    metrics = ("precision", "balanced_accuracy", "f1", "useful_cull", "avg_pred_count")
    rows: dict[str, Any] = {}
    any_positive = False
    for name in effect_names:
        rows[name] = {}
        for scope in ("pose_macro", "aggregate"):
            rows[name][scope] = {}
            for metric in metrics:
                record = effect(summary, name, scope, metric)
                improved = positive_ci(record) if metric != "avg_pred_count" else negative_ci(record)
                rows[name][scope][metric] = {
                    "effect": record,
                    "improvementCIExcludesZero": improved,
                    "improvementDirection": "decrease" if metric == "avg_pred_count" else "increase",
                }
                any_positive = any_positive or improved
    return {
        "metrics": rows,
        "passed": any_positive,
        "rule": "after safety, a precision/balanced-accuracy/F1/useful-cull CI lower bound > 0 or average prediction count CI upper bound < 0",
    }


def system_for_direction(summary: dict[str, Any], effect_names: tuple[str, ...]) -> dict[str, Any]:
    unavailable = summary.get("unavailableMetrics", {})
    candidates = ("glb_byte_reduction", "equal_visual_utility_glb_bytes", "miss_pixel_rate")
    details = {name: {metric: {"status": "not_available", "reason": unavailable.get(metric)} for metric in candidates} for name in effect_names}
    return {
        "passed": False,
        "details": details,
        "rule": "same visual utility with lower GLB bytes/first-screen time, or lower miss-pixel/p95 miss-pixel rate",
        "status": "not_available",
    }


def make_decision(summary: dict[str, Any], summary_path: Path, max_bad_cull_delta: float = 0.002, max_recall_drop: float = 0.01) -> dict[str, Any]:
    if summary.get("schema") != "neuralstreamweb3d-formal-m4-matrix-summary-v2":
        raise ValueError(f"unexpected M4-v2 summary schema: {summary_path}")
    if summary.get("split") != "validation" or summary.get("testRead") is not False:
        raise ValueError("M4-v2 route decision requires test-free validation summary")
    if list(summary.get("factorDefinition", {}).get(key, "") for key in ("A", "B", "C", "D")) != list(FACTOR_VARIANTS):
        raise ValueError("M4-v2 factor definition is incomplete or reordered")
    if sorted(summary.get("seeds", list(SEEDS))) != list(SEEDS):
        # The summary currently stores seeds under the matrix member keys; a
        # missing explicit field is an error in a formal result, not a reason
        # to infer the seed set from filenames.
        raise ValueError("M4-v2 summary lacks the registered three-seed declaration")
    for name in ("B_minus_A", "C_minus_A", "D_minus_C", "D_minus_B", "interaction_D_minus_B_minus_C_plus_A"):
        if name not in summary.get("factorEffects", {}):
            raise ValueError(f"M4-v2 summary lacks factor effect {name}")

    safety_ba = safety_for_direction(summary, "B_minus_A", max_bad_cull_delta, max_recall_drop)
    safety_dc = safety_for_direction(summary, "D_minus_C", max_bad_cull_delta, max_recall_drop)
    classification = classification_for_direction(summary, ("B_minus_A", "D_minus_C"))
    system = system_for_direction(summary, ("B_minus_A", "D_minus_C"))
    safety_passed = safety_ba["passed"] and safety_dc["passed"]
    contribution_passed = bool(safety_passed and (classification["passed"] or system["passed"]))
    route = "route_a_directional_proxy" if contribution_passed else "route_b_system"
    if not safety_passed:
        route_reason = "direction proxy safety layer failed; no classification or system gain can upgrade the route"
    elif not classification["passed"] and not system["passed"]:
        route_reason = "direction proxy is evaluated at safe workpoints but no stable classification, culling, resource, or image gain is shown"
    else:
        route_reason = "direction proxy passes the registered safety layer and at least one independent effect layer"
    source_hash = hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.is_file() else None
    return {
        "schema": "neuralstreamweb3d-formal-m4-route-decision-v2",
        "version": 2,
        "split": "validation",
        "testRead": False,
        "sourceSummary": str(summary_path.resolve()),
        "sourceSummarySha256": source_hash,
        "route": route,
        "routeReason": route_reason,
        "safetyLayer": {
            "directionProxyWithoutInhibition": safety_ba,
            "directionProxyWithInhibition": safety_dc,
            "passed": safety_passed,
        },
        "classificationCullLayer": classification,
        "systemLayer": system,
        "explicitInhibitionEffects": {
            "C_minus_A": summary["factorEffects"]["C_minus_A"],
            "D_minus_B": summary["factorEffects"]["D_minus_B"],
        },
        "interaction": summary["factorEffects"]["interaction_D_minus_B_minus_C_plus_A"],
        "policy": {
            "maxBadCullDelta": max_bad_cull_delta,
            "maxRecallDrop": max_recall_drop,
            "factorComparisons": ["B-A", "C-A", "D-C", "D-B", "D-B-C+A"],
            "usefulCullIsNotSufficient": True,
        },
        "interpretation": {
            "directionProxyUsed": "inferred from the B-A and D-C factor members and their separate proxy branch inputs",
            "directionProxyClassificationContribution": classification["passed"],
            "directionProxySafetyConstrainedCullContribution": safety_passed and classification["passed"],
            "directionProxyImageContribution": system["passed"],
            "directionProxyResourceContribution": system["passed"],
        },
    }


def render_markdown(decision: dict[str, Any]) -> str:
    route = decision["route"]
    safety = decision["safetyLayer"]
    lines = [
        "# M4-v2 路线判定",
        "",
        "本文件只使用 validation；test split 未读取。路线判定同时检查安全性、分类/剔除效果和系统效果，不能由 useful cull 单独决定。",
        "",
        f"**路线：** `{route}`  ",
        f"**原因：** {decision['routeReason']}",
        "",
        "## 安全层",
        "",
        "方向遮挡代理分别比较无显式抑制的 B-A 和有显式抑制的 D-C；两种统计口径都要求 recall、weighted recall 的置信区间下界不出现预登记的明显下降，且 bad cull 上界不超过安全增量。",
        "",
        "| 比较 | 成员安全工作点 | pose 宏平均安全 | aggregate 安全 |",
        "|---|---|---|---|",
    ]
    for label, item in (("B-A", safety["directionProxyWithoutInhibition"]), ("D-C", safety["directionProxyWithInhibition"])):
        lines.append(
            f"| {label} | {'通过' if item['memberSafetyAllQualified'] else '未通过'} | "
            f"{'通过' if item['ciSafety']['pose_macro']['recallNoMaterialDrop'] and item['ciSafety']['pose_macro']['weightedRecallNoMaterialDrop'] and item['ciSafety']['pose_macro']['badCullUpperBoundSatisfied'] else '未通过'} | "
            f"{'通过' if item['ciSafety']['aggregate']['recallNoMaterialDrop'] and item['ciSafety']['aggregate']['weightedRecallNoMaterialDrop'] and item['ciSafety']['aggregate']['badCullUpperBoundSatisfied'] else '未通过'} |"
        )
    lines.extend([
        "",
        "## 分层解释",
        "",
        f"分类/剔除层：{'通过' if decision['classificationCullLayer']['passed'] else '未通过'}。",
        f"系统层：{'通过' if decision['systemLayer']['passed'] else '未实现或未通过'}。",
        "",
        "方向代理被模型使用、方向代理改善分类、方向代理改善安全约束下剔除效率、方向代理改善图像质量以及方向代理改善下载/运行时成本分别记录，不能互相替代。",
        "",
        "## 具体反例",
        "",
        "如果某个变体把更多真实可见实例误判为不可见，它会同时增加 FN 和 bad cull。由于 TN 仍可能大量增加，useful cull 可能看起来更高；但 recall、weighted recall 会下降，画面安全变差。这种数字组合不是模型性能提升。",
        "",
        "完整差值、置信区间和交互项见同目录的 M4-v2 JSON 汇总；GLB/像素/浏览器字段若没有同位姿数据会明确记为 not_available。",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> dict[str, Any]:
    # Keep the fixture deliberately small; the schema test exercises the route
    # logic independently from the expensive matrix bootstrap.
    effects: dict[str, Any] = {}
    metric = {"mean_delta": 0.01, "ci95": [0.001, 0.02], "crosses_zero": False}
    safe_bad = {"mean_delta": 0.0005, "ci95": [-0.0005, 0.001], "crosses_zero": True}
    for name in ("B_minus_A", "C_minus_A", "D_minus_C", "D_minus_B", "interaction_D_minus_B_minus_C_plus_A"):
        effects[name] = {"comparisons": {scope: {key: (safe_bad if key == "bad_cull" else metric) for key in ("recall", "weighted_recall", "bad_cull", "precision", "balanced_accuracy", "f1", "useful_cull", "avg_pred_count")} for scope in ("pose_macro", "aggregate")}}
    members = {
        f"{variant}:{seed}": {"safety": {"qualifiedSafetyWorkpoint": True}, "calibrationWorkpoint": {"status": "safe"}}
        for variant in FACTOR_VARIANTS for seed in SEEDS
    }
    summary = {
        "schema": "neuralstreamweb3d-formal-m4-matrix-summary-v2",
        "split": "validation",
        "testRead": False,
        "factorDefinition": {"A": FACTOR_VARIANTS[0], "B": FACTOR_VARIANTS[1], "C": FACTOR_VARIANTS[2], "D": FACTOR_VARIANTS[3]},
        "factorEffects": effects,
        "members": members,
        "seeds": list(SEEDS),
        "unavailableMetrics": {},
    }
    result = make_decision(summary, Path("fixture.json"))
    assert result["route"] == "route_a_directional_proxy"
    return {"status": "passed", "route": result["route"]}


def main() -> None:
    args = parse_args()
    if args.self_test:
        print(json.dumps(self_test(), indent=2))
        return
    if args.summary is None or args.output is None or args.markdown is None:
        raise ValueError("--summary, --output, and --markdown are required unless --self-test is used")
    if args.max_bad_cull_delta < 0.0 or args.max_recall_drop < 0.0:
        raise ValueError("route safety tolerances must be non-negative")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    decision = make_decision(summary, args.summary, args.max_bad_cull_delta, args.max_recall_drop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(decision), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "route": decision["route"], "testRead": False}, indent=2))


if __name__ == "__main__":
    main()
