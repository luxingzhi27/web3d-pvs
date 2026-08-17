#!/usr/bin/env python3
"""Apply the registered safety-first decision policy to the v2 matrix.

The input is the validation-only summary produced by
``summarize_pvs_hierarchical_relation_survival_integrated.py``.  Comparisons
are named ``full_minus_<ablation>`` and therefore a positive delta means an
improvement for the complete model.  This entry point is intentionally
separate from the historical M4 route decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
FORMAL_VARIANTS = (
    "full",
    "without_hierarchical_relation",
    "shuffled_relation_source",
    "without_viewcell_integration",
    "without_threshold_aligned_utility",
)
ABLATIONS = FORMAL_VARIANTS[1:]
SEEDS = (20260801, 20260802, 20260803)


def _effect(summary: Mapping[str, Any], name: str, scope: str, metric: str) -> dict[str, Any]:
    try:
        value = summary["comparisons"][name]["metrics"][f"{scope}.{metric}"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing comparison effect {name}/{scope}.{metric}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison effect is not an object: {name}/{scope}.{metric}")
    return dict(value)


def _lower(effect: Mapping[str, Any]) -> float:
    ci = effect.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("effect has no two-sided 95% interval")
    return float(ci[0])


def _upper(effect: Mapping[str, Any]) -> float:
    ci = effect.get("ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("effect has no two-sided 95% interval")
    return float(ci[1])


def _positive(effect: Mapping[str, Any]) -> bool:
    return _lower(effect) > 0.0


def _negative(effect: Mapping[str, Any]) -> bool:
    return _upper(effect) < 0.0


def _member_safety(summary: Mapping[str, Any], variant: str, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    members = summary.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("summary has no member records")
    rows: dict[str, Any] = {}
    for seed in seeds:
        key = f"{variant}:{seed}"
        record = members.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"summary is missing member {key}")
        rows[key] = {
            "safeWorkpoint": bool(record.get("safeWorkpoint", False)),
            "status": record.get("safety", {}),
        }
    return {
        "allSafe": all(value["safeWorkpoint"] for value in rows.values()),
        "members": rows,
    }


def _safety_for(
    summary: Mapping[str, Any],
    ablation: str,
    max_bad_cull_delta: float,
    max_recall_drop: float,
    seeds: tuple[int, ...] = SEEDS,
) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    scopes: dict[str, Any] = {}
    passed = True
    for scope in ("poseMacro", "aggregate"):
        recall = _effect(summary, name, scope, "recall")
        weighted = _effect(summary, name, scope, "weightedRecall")
        bad = _effect(summary, name, scope, "badCull")
        current = {
            "recall": recall,
            "weightedRecall": weighted,
            "badCull": bad,
            "recallNoMaterialDrop": _lower(recall) >= -float(max_recall_drop),
            "weightedRecallNoMaterialDrop": _lower(weighted) >= -float(max_recall_drop),
            "badCullUpperBoundSatisfied": _upper(bad) <= float(max_bad_cull_delta),
        }
        scopes[scope] = current
        passed = passed and all(current[key] for key in (
            "recallNoMaterialDrop",
            "weightedRecallNoMaterialDrop",
            "badCullUpperBoundSatisfied",
        ))
    full_safe = _member_safety(summary, "full", seeds)
    ablation_safe = _member_safety(summary, ablation, seeds)
    passed = passed and full_safe["allSafe"] and ablation_safe["allSafe"]
    return {
        "comparison": name,
        "full": full_safe,
        "ablation": ablation_safe,
        "scopes": scopes,
        "passed": bool(passed),
        "policy": {
            "maxRecallDrop": float(max_recall_drop),
            "maxBadCullDelta": float(max_bad_cull_delta),
            "gateMetric": "aggregate weighted recall and its calibration lower confidence bound",
        },
    }


def _classification_for(summary: Mapping[str, Any], ablation: str) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    metrics = ("precision", "balancedAccuracy", "f1", "usefulCull", "avgPredCount")
    rows: dict[str, Any] = {}
    passed = False
    for scope in ("poseMacro", "aggregate"):
        rows[scope] = {}
        for metric in metrics:
            item = _effect(summary, name, scope, metric)
            improved = _negative(item) if metric == "avgPredCount" else _positive(item)
            rows[scope][metric] = {
                "effect": item,
                "improvedWith95Ci": bool(improved),
                "desiredDirection": "decrease" if metric == "avgPredCount" else "increase",
            }
            passed = passed or improved
    return {
        "metrics": rows,
        "passed": bool(passed),
        "rule": "precision, balanced accuracy, F1, useful cull increase or average prediction count decreases with a 95% CI excluding zero",
    }


def _resource_for(summary: Mapping[str, Any], ablation: str) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    metrics = ("predictedGlbBytes", "glbByteReduction", "downloadUtilityRecall")
    rows: dict[str, Any] = {}
    passed = False
    for scope in ("poseMacro", "aggregate"):
        rows[scope] = {}
        for metric in metrics:
            item = _effect(summary, name, scope, metric)
            if metric == "downloadUtilityRecall":
                improved = _positive(item)
            elif metric == "predictedGlbBytes":
                improved = _negative(item)
            else:
                improved = _positive(item)
            rows[scope][metric] = {"effect": item, "improvedWith95Ci": bool(improved)}
            passed = passed or improved
    return {
        "metrics": rows,
        "passed": bool(passed),
        "rule": "lower predicted GLB bytes or higher download utility recall with a 95% CI excluding zero",
    }


def make_decision(summary: Mapping[str, Any], summary_path: Path, max_bad_cull_delta: float = 0.002, max_recall_drop: float = 0.01) -> dict[str, Any]:
    if summary.get("schema") != "pvs-hierarchical-relation-survival-integrated-validation-summary-v1":
        raise ValueError("unexpected v2 validation summary schema")
    if summary.get("split") != "validation" or summary.get("testRead") is not False:
        raise ValueError("v2 route decision requires test-free validation")
    if tuple(summary.get("variants", ())) != FORMAL_VARIANTS:
        raise ValueError("v2 summary variant order is incomplete")
    if tuple(int(value) for value in summary.get("seeds", ())) != SEEDS:
        raise ValueError("v2 summary does not contain the registered three seeds")
    required_comparisons = {f"full_minus_{ablation}" for ablation in ABLATIONS}
    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, Mapping) or not required_comparisons.issubset(comparisons):
        raise ValueError("v2 summary is missing a current full-minus-ablation comparison")
    bootstrap = summary.get("bootstrap", {})
    if not isinstance(bootstrap, Mapping) or int(bootstrap.get("replicates", 0)) < 10000:
        raise ValueError("v2 route decision requires at least 10000 paired bootstrap replicates")

    rows: dict[str, Any] = {}
    all_safety = True
    any_classification = False
    any_resource = False
    for ablation in ABLATIONS:
        safety = _safety_for(summary, ablation, max_bad_cull_delta, max_recall_drop)
        classification = _classification_for(summary, ablation)
        resource = _resource_for(summary, ablation)
        rows[ablation] = {"safety": safety, "classification": classification, "resource": resource}
        all_safety = all_safety and safety["passed"]
        any_classification = any_classification or (safety["passed"] and classification["passed"])
        any_resource = any_resource or (safety["passed"] and resource["passed"])

    unavailable = summary.get("unavailableMetrics", {})
    image_available = not any(name in unavailable for name in ("missPixelRate", "wrongIdPixelRate", "extraPixelRate"))
    route = "full_innovation_candidate" if all_safety and (any_classification or any_resource) else "baseline_system"
    if not all_safety:
        reason = "at least one complete-versus-ablation comparison failed the pre-registered safety layer"
    elif not any_classification and not any_resource:
        reason = "the complete model has no stable classification, culling, or resource gain at safe workpoints"
    else:
        reason = "the complete model passes safety and has at least one independent classification or resource gain"
    return {
        "schema": "pvs-full-innovation-v2-route-decision-v1",
        "version": 1,
        "split": "validation",
        "testRead": False,
        "sourceSummary": str(summary_path.resolve()),
        "sourceSummarySha256": hashlib.sha256(summary_path.read_bytes()).hexdigest() if summary_path.is_file() else None,
        "route": route,
        "routeReason": reason,
        "innovationComparisons": rows,
        "layers": {
            "safety": all_safety,
            "classification": any_classification,
            "resource": any_resource,
            "image": {"available": image_available, "status": "available" if image_available else "not_available"},
        },
        "policy": {
            "maxBadCullDelta": float(max_bad_cull_delta),
            "maxRecallDrop": float(max_recall_drop),
            "pairedBootstrapReplicates": int(bootstrap["replicates"]),
            "usefulCullIsNotSufficient": True,
        },
    }


def render_markdown(decision: Mapping[str, Any]) -> str:
    title = "Full Innovation v2 路线判定"
    lines = [
        f"# {title}",
        "",
        "本判定只读取 validation。比较均为 `full - ablation`，正值表示完整组合相对消融的提升；useful cull 不单独决定路线。",
        "",
        f"**路线：** `{decision['route']}`  ",
        f"**原因：** {decision['routeReason']}",
        "",
        "| 消融 | 安全层 | 分类/剔除 | 下载/资源 |",
        "|---|---|---|---|",
    ]
    for variant, row in decision["innovationComparisons"].items():
        lines.append(
            f"| `{variant}` | {'通过' if row['safety']['passed'] else '未通过'} | "
            f"{'有稳定改善' if row['classification']['passed'] else '无稳定改善'} | "
            f"{'有稳定改善' if row['resource']['passed'] else '无稳定改善'} |"
        )
    lines.extend([
        "",
        "图像 Color-ID 和浏览器 WebGPU 指标若未接入，只能标记为未实现，不能填零或推断为无差异。",
        "",
        "具体差值、95% 置信区间、weighted recall、bad cull、accuracy 和 balanced accuracy 保存在同一 JSON 汇总中。",
    ])
    return "\n".join(lines) + "\n"


def self_test() -> dict[str, Any]:
    effect = {"meanDelta": 0.01, "ci95": [0.001, 0.02], "crossesZero": False}
    safe = {"meanDelta": 0.0001, "ci95": [-0.0005, 0.001], "crossesZero": True}
    comparisons = {}
    for variant in ABLATIONS:
        comparisons[f"full_minus_{variant}"] = {
            "metrics": {
                f"{scope}.{metric}": (safe if metric in ("recall", "weightedRecall", "badCull") else effect)
                for scope in ("poseMacro", "aggregate")
                for metric in ("recall", "weightedRecall", "badCull", "precision", "balancedAccuracy", "f1", "usefulCull", "avgPredCount", "predictedGlbBytes", "glbByteReduction", "downloadUtilityRecall")
            }
        }
    members = {
        f"{variant}:{seed}": {"safeWorkpoint": True, "safety": {}}
        for variant in FORMAL_VARIANTS for seed in SEEDS
    }
    summary = {
        "schema": "pvs-hierarchical-relation-survival-integrated-validation-summary-v1",
        "split": "validation", "testRead": False,
        "variants": list(FORMAL_VARIANTS), "seeds": list(SEEDS),
        "bootstrap": {"replicates": 10000}, "members": members,
        "comparisons": comparisons, "unavailableMetrics": {"missPixelRate": "not attached"},
    }
    result = make_decision(summary, Path("fixture.json"))
    assert result["route"] == "full_innovation_candidate"
    return {"status": "passed", "route": result["route"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--max-bad-cull-delta", type=float, default=0.002)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.summary is None or args.output is None or args.markdown is None:
        raise ValueError("--summary, --output and --markdown are required")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    decision = make_decision(summary, args.summary, args.max_bad_cull_delta, args.max_recall_drop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(decision), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "route": decision["route"], "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
