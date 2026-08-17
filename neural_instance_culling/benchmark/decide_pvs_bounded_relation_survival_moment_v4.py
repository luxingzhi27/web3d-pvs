#!/usr/bin/env python3
"""Apply the safety-first route policy to the v4 validation summary.

This route consumes only the calibrated relation-prior v4 summary
records.  Legacy v2 summaries are rejected instead of being interpreted by a
compatibility branch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


SUMMARY_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-v4-validation-summary-v1"
ROUTE_SCHEMA = "pvs-bounded-relation-prior-instance-calibrated-v4-route-decision-v1"
FORMAL_VARIANTS = (
    "full",
    "without_bounded_relation",
    "without_viewcell_moment_envelope",
    "without_safety_reserve_utility",
    "without_instance_calibration_residual",
)
SEEDS = (20260801, 20260802, 20260803)
V4_SUMMARY_SCHEMA = SUMMARY_SCHEMA
V4_FORMAL_VARIANTS = FORMAL_VARIANTS
V4_SEEDS = SEEDS


def _effect(summary: Mapping[str, Any], name: str, scope: str, metric: str) -> dict[str, Any]:
    try:
        value = summary["comparisons"][name]["metrics"][f"{scope}.{metric}"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing v4 comparison effect {name}/{scope}.{metric}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"v4 comparison effect is not an object: {name}/{scope}.{metric}")
    return dict(value)


def _lower(effect: Mapping[str, Any]) -> float:
    interval = effect.get("ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("v4 effect has no two-sided 95% interval")
    return float(interval[0])


def _upper(effect: Mapping[str, Any]) -> float:
    interval = effect.get("ci95")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("v4 effect has no two-sided 95% interval")
    return float(interval[1])


def _positive(effect: Mapping[str, Any]) -> bool:
    return _lower(effect) > 0.0


def _negative(effect: Mapping[str, Any]) -> bool:
    return _upper(effect) < 0.0


def _member_safety(summary: Mapping[str, Any], variant: str) -> dict[str, Any]:
    members = summary.get("members")
    if not isinstance(members, Mapping):
        raise ValueError("v4 summary has no member records")
    rows: dict[str, Any] = {}
    for seed in SEEDS:
        key = f"{variant}:{seed}"
        record = members.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"v4 summary is missing member {key}")
        aggregate = record.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise ValueError(f"v4 summary member has no validation aggregate metrics: {key}")
        try:
            weighted_recall_raw = (
                aggregate["aggregateWeightedRecall"]
                if "aggregateWeightedRecall" in aggregate
                else aggregate["weightedRecall"]
            )
            weighted_recall_lcb_raw = (
                aggregate["aggregateWeightedRecallLowerConfidenceBound"]
                if "aggregateWeightedRecallLowerConfidenceBound" in aggregate
                else aggregate["weightedRecallLowerConfidenceBound"]
            )
            weighted_recall = float(weighted_recall_raw)
            weighted_recall_lcb = float(weighted_recall_lcb_raw)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"v4 summary member has incomplete validation safety metrics: {key}") from exc
        calibration_safe = bool(
            record.get("calibrationSafeWorkpoint", record.get("safeWorkpoint", False))
        )
        validation_point_safe = weighted_recall > 0.99
        validation_lcb_safe = weighted_recall_lcb > 0.99
        rows[key] = {
            "calibrationSafeWorkpoint": calibration_safe,
            "validationAggregateWeightedRecall": weighted_recall,
            "validationAggregateWeightedRecallLowerConfidenceBound": weighted_recall_lcb,
            "validationWeightedRecallSatisfied": validation_point_safe,
            "validationWeightedRecallLowerConfidenceBoundSatisfied": validation_lcb_safe,
            "safe": bool(calibration_safe and validation_point_safe and validation_lcb_safe),
            "status": record.get("safety", {}),
        }
    return {
        "allCalibrationSafe": all(value["calibrationSafeWorkpoint"] for value in rows.values()),
        "allValidationWeightedRecallSafe": all(
            value["validationWeightedRecallSatisfied"] for value in rows.values()
        ),
        "allValidationWeightedRecallLowerBoundSafe": all(
            value["validationWeightedRecallLowerConfidenceBoundSatisfied"]
            for value in rows.values()
        ),
        "allSafe": all(value["safe"] for value in rows.values()),
        "members": rows,
        "gate": {
            "calibrationSafeRequired": True,
            "validationAggregateWeightedRecall": "> 0.99",
            "validationAggregateWeightedRecallLowerConfidenceBound": "> 0.99",
        },
    }


def _safety_for(summary: Mapping[str, Any], ablation: str, max_weighted_recall_drop: float) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    scopes: dict[str, Any] = {}
    for scope in ("poseMacro", "aggregate"):
        recall = _effect(summary, name, scope, "recall")
        weighted = _effect(summary, name, scope, "weightedRecall")
        bad = _effect(summary, name, scope, "badCull")
        scopes[scope] = {
            "recall": recall,
            "weightedRecall": weighted,
            "badCull": bad,
            "recallDiagnosticOnly": True,
            "badCullDiagnosticOnly": True,
            "weightedRecallNoMaterialDrop": (
                _lower(weighted) >= -float(max_weighted_recall_drop)
                if scope == "aggregate"
                else None
            ),
        }
    full_safe = _member_safety(summary, "full")
    ablation_safe = _member_safety(summary, ablation)
    passed = bool(
        full_safe["allSafe"]
        and scopes["aggregate"]["weightedRecallNoMaterialDrop"]
    )
    return {
        "comparison": name,
        "full": full_safe,
        "ablation": ablation_safe,
        "scopes": scopes,
        "passed": bool(passed),
        "policy": {
            "maxAggregateWeightedRecallDrop": float(max_weighted_recall_drop),
            "gateMetric": "full member aggregate weighted recall and its lower confidence bound",
            "ablationSafeWorkpointRequired": False,
            "poseRecallDiagnosticOnly": True,
            "badCullConfidenceIntervalDiagnosticOnly": True,
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
            effect = _effect(summary, name, scope, metric)
            improved = _negative(effect) if metric == "avgPredCount" else _positive(effect)
            rows[scope][metric] = {
                "effect": effect,
                "improvedWith95Ci": bool(improved),
                "desiredDirection": "decrease" if metric == "avgPredCount" else "increase",
            }
            passed = passed or improved
    return {
        "metrics": rows,
        "passed": bool(passed),
        "rule": "precision, balanced accuracy, F1, useful cull, or average prediction count must improve with a 95% CI excluding zero",
    }


def _resource_for(summary: Mapping[str, Any], ablation: str) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    metrics = ("predictedGlbBytes", "glbByteReduction", "downloadUtilityRecall")
    rows: dict[str, Any] = {}
    passed = False
    for scope in ("poseMacro", "aggregate"):
        rows[scope] = {}
        for metric in metrics:
            effect = _effect(summary, name, scope, metric)
            improved = _negative(effect) if metric == "predictedGlbBytes" else _positive(effect)
            rows[scope][metric] = {"effect": effect, "improvedWith95Ci": bool(improved)}
            passed = passed or improved
    return {
        "metrics": rows,
        "passed": bool(passed),
        "rule": "lower predicted GLB bytes or higher download utility recall/byte reduction with a 95% CI excluding zero",
    }


def _image_for(summary: Mapping[str, Any], ablation: str) -> dict[str, Any]:
    name = f"full_minus_{ablation}"
    metrics = (
        "meanMissPixelRate", "p95MissPixelRate",
        "meanWrongInstancePixelRate", "p95WrongInstancePixelRate",
        "meanExtraPixelRateOverImage", "p95ExtraPixelRateOverImage",
    )
    rows: dict[str, Any] = {}
    improved = False
    worsened = False
    for metric in metrics:
        effect = _effect(summary, name, "image", metric)
        metric_improved = _negative(effect)
        metric_worsened = _positive(effect)
        rows[metric] = {
            "effect": effect,
            "improvedWith95Ci": bool(metric_improved),
            "worsenedWith95Ci": bool(metric_worsened),
            "desiredDirection": "decrease",
        }
        improved = improved or metric_improved
        worsened = worsened or metric_worsened
    return {
        "metrics": rows,
        "improved": bool(improved),
        "noStableWorsening": not worsened,
        "passed": bool(not worsened),
        "rule": "mean/p95 miss, wrong-ID and extra-pixel rates must not increase with a 95% CI excluding zero",
    }


def make_decision(
    summary: Mapping[str, Any],
    summary_path: Path,
    max_weighted_recall_drop: float = 0.01,
) -> dict[str, Any]:
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise ValueError("v4 route accepts only the bounded-relation survival moment summary schema")
    if summary.get("split") != "validation" or summary.get("testRead") is not False:
        raise ValueError("v4 route decision requires test-free validation")
    if tuple(summary.get("variants", ())) != FORMAL_VARIANTS:
        raise ValueError("v4 summary variant order is incomplete")
    if tuple(int(value) for value in summary.get("seeds", ())) != SEEDS:
        raise ValueError("v4 summary does not contain the registered three seeds")
    comparisons = summary.get("comparisons")
    required = {f"full_minus_{variant}" for variant in FORMAL_VARIANTS[1:]}
    if not isinstance(comparisons, Mapping) or not required.issubset(comparisons):
        raise ValueError("v4 summary is missing a complete full-minus-ablation comparison set")
    bootstrap = summary.get("bootstrap")
    if not isinstance(bootstrap, Mapping) or int(bootstrap.get("replicates", 0)) < 10000 or bootstrap.get("paired") is not True:
        raise ValueError("v4 route decision requires at least 10000 paired bootstrap replicates")
    unavailable = summary.get("unavailableMetrics", {})
    if not isinstance(unavailable, Mapping):
        raise ValueError("v4 summary unavailableMetrics is not an object")
    image_available = not any(
        name in unavailable for name in ("missPixelRate", "wrongIdPixelRate", "extraPixelRate")
    )
    rows: dict[str, Any] = {}
    all_safety = True
    any_classification = False
    any_resource = False
    all_image_safe = bool(image_available)
    any_image = False
    for ablation in FORMAL_VARIANTS[1:]:
        safety = _safety_for(summary, ablation, max_weighted_recall_drop)
        classification = _classification_for(summary, ablation)
        resource = _resource_for(summary, ablation)
        image = _image_for(summary, ablation) if image_available else {
            "passed": False,
            "improved": False,
            "noStableWorsening": None,
            "status": "not_available",
        }
        rows[ablation] = {
            "safety": safety,
            "classification": classification,
            "resource": resource,
            "image": image,
        }
        all_safety = all_safety and safety["passed"]
        any_classification = any_classification or (safety["passed"] and classification["passed"])
        any_resource = any_resource or (safety["passed"] and resource["passed"])
        all_image_safe = all_image_safe and bool(image["passed"])
        any_image = any_image or (safety["passed"] and bool(image["improved"]))
    route = (
        "full_innovation_candidate"
        if all_safety
        and image_available
        and all_image_safe
        and (any_classification or any_resource or any_image)
        else "baseline_system"
    )
    if not all_safety:
        reason = "at least one v4 full-versus-ablation comparison failed the registered safety layer"
    elif not image_available:
        reason = "formal hardware image metrics have not been backfilled"
    elif not all_image_safe:
        reason = "the v4 complete model has a stable mean/p95 image-error regression"
    elif not any_classification and not any_resource and not any_image:
        reason = "the v4 complete model has no stable classification, culling, resource, or image gain at safe workpoints"
    else:
        reason = "the v4 complete model passes set/image safety and has at least one stable classification, resource, or image gain"
    return {
        "schema": ROUTE_SCHEMA,
        "version": 1,
        "split": "validation",
        "testRead": False,
        "sourceSummary": str(summary_path.resolve()),
        "route": route,
        "routeReason": reason,
        "innovationComparisons": rows,
        "layers": {
            "safety": all_safety,
            "classification": any_classification,
            "resource": any_resource,
            "image": {
                "available": image_available,
                "noStableWorsening": all_image_safe if image_available else None,
                "improvement": any_image if image_available else None,
                "status": "available" if image_available else "not_available",
            },
        },
        "policy": {
            "maxAggregateWeightedRecallDrop": float(max_weighted_recall_drop),
            "pairedBootstrapReplicates": int(bootstrap["replicates"]),
            "bootstrapClusterUnit": bootstrap.get("clusterUnit"),
            "usefulCullIsNotSufficient": True,
            "poseRecallDiagnosticOnly": True,
            "badCullConfidenceIntervalDiagnosticOnly": True,
        },
        "inputProvenance": summary.get("inputProvenance", {}),
    }


def render_markdown(decision: Mapping[str, Any]) -> str:
    lines = [
        "# v4 有界关系生存矩包络路线判定",
        "",
        "本判定只读取 validation。比较均为 `full - ablation`，正值表示完整模型相对消融的提升；useful cull 不单独决定路线。",
        "",
        f"**路线：** `{decision['route']}`  ",
        f"**原因：** {decision['routeReason']}",
        "",
        "| 消融 | 安全层 | 分类/剔除 | 下载/资源 | 图像 |",
        "|---|---|---|---|---|",
    ]
    for variant, row in decision["innovationComparisons"].items():
        lines.append(
            f"| `{variant}` | {'通过' if row['safety']['passed'] else '未通过'} | "
            f"{'有稳定改善' if row['classification']['passed'] else '无稳定改善'} | "
            f"{'有稳定改善' if row['resource']['passed'] else '无稳定改善'} | "
            f"{'无稳定恶化' if row['image']['passed'] else '缺失或恶化'} |"
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
    safety = {"meanDelta": 0.0001, "ci95": [-0.0005, 0.001], "crossesZero": True}
    metrics = {}
    for scope in ("poseMacro", "aggregate"):
        for metric in ("recall", "weightedRecall", "badCull"):
            metrics[f"{scope}.{metric}"] = safety
        for metric in ("precision", "balancedAccuracy", "f1", "usefulCull", "avgPredCount", "predictedGlbBytes", "glbByteReduction", "downloadUtilityRecall"):
            metrics[f"{scope}.{metric}"] = {"meanDelta": -0.01, "ci95": [-0.02, -0.001]} if metric == "avgPredCount" or metric == "predictedGlbBytes" else effect
    summary = {
        "schema": SUMMARY_SCHEMA,
        "split": "validation",
        "testRead": False,
        "variants": list(FORMAL_VARIANTS),
        "seeds": list(SEEDS),
        "bootstrap": {"replicates": 10000, "paired": True},
        "members": {
            f"{variant}:{seed}": {
                "safeWorkpoint": True,
                "aggregate": {
                    "weightedRecall": 0.995,
                    "weightedRecallLowerConfidenceBound": 0.991,
                },
                "safety": {},
            }
            for variant in FORMAL_VARIANTS
            for seed in SEEDS
        },
        "comparisons": {f"full_minus_{variant}": {"metrics": metrics} for variant in FORMAL_VARIANTS[1:]},
        "unavailableMetrics": {"browserWebGpuLatency": "not attached"},
    }
    for comparison in summary["comparisons"].values():
        for metric in (
            "meanMissPixelRate", "p95MissPixelRate",
            "meanWrongInstancePixelRate", "p95WrongInstancePixelRate",
            "meanExtraPixelRateOverImage", "p95ExtraPixelRateOverImage",
        ):
            comparison["metrics"][f"image.{metric}"] = {
                "meanDelta": -0.001,
                "ci95": [-0.002, 0.0],
                "crossesZero": True,
            }
    decision = make_decision(summary, Path("v4_fixture.json"))
    if decision["route"] != "full_innovation_candidate":
        raise AssertionError("v4 route self-test failed")
    return {"status": "passed", "schema": ROUTE_SCHEMA, "route": decision["route"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--max-weighted-recall-drop", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    if args.summary is None or args.output is None or args.markdown is None:
        raise ValueError("--summary, --output and --markdown are required")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    decision = make_decision(summary, args.summary, args.max_weighted_recall_drop)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(decision), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "route": decision["route"], "testRead": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
