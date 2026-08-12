#!/usr/bin/env python3
"""Apply the registered layered route rule to the ray-context matrix.

The matrix has three independent factors: the offline context representation,
the monotone occlusion-survival field, and the visibility loss.  This script
keeps those effects separate; it never treats a useful-cull increase alone as
evidence of a better model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONTEXT_EFFECT = "context_main_effect"
DEFAULT_SURVIVAL_EFFECT = "survival_main_effect"
DEFAULT_LOSS_EFFECT = "loss_main_effect"


def _find_effect(
    summary: dict[str, Any],
    name: str | None,
    contains: str,
) -> tuple[str, dict[str, Any]]:
    # Pairwise comparisons and registered factor contrasts are deliberately
    # stored separately in the summary.  The three default route factors are
    # full-factorial contrasts, so route selection must inspect both stores.
    effect_sources = (
        summary.get("comparisons", {}),
        summary.get("factorEffects", {}),
    )
    if name:
        for effects in effect_sources:
            if name in effects:
                return name, effects[name]
        if not any(isinstance(effects, dict) for effects in effect_sources):
            raise ValueError(f"missing comparison {name}")
        raise ValueError(f"missing comparison or factor effect {name}")
    matches = [
        (key, value)
        for effects in effect_sources
        if isinstance(effects, dict)
        for key, value in effects.items()
        if contains in key
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one comparison containing {contains!r}, found {[key for key, _ in matches]}"
        )
    return matches[0]


def _ci(effect: dict[str, Any], metric: str) -> tuple[float, float]:
    item = effect.get("metrics", {}).get(metric)
    if not isinstance(item, dict) or len(item.get("ci95", [])) != 2:
        raise ValueError(f"comparison has no ci95 for {metric}")
    return float(item["ci95"][0]), float(item["ci95"][1])


def _positive(effect: dict[str, Any], metric: str) -> bool:
    return _ci(effect, metric)[0] > 0.0


def _negative(effect: dict[str, Any], metric: str) -> bool:
    return _ci(effect, metric)[1] < 0.0


def _factor_decision(effect: dict[str, Any] | None, weighted_drop: float) -> dict[str, Any]:
    if effect is None:
        return {"available": False, "hasContribution": False}

    pose_weighted = _ci(effect, "pose_weighted_recall")
    aggregate_weighted = _ci(effect, "aggregate_weighted_recall")
    pose_recall = _ci(effect, "pose_recall")
    bad_cull = _ci(effect, "pose_bad_cull")
    weighted_lower = min(pose_weighted[0], aggregate_weighted[0])
    safety = {
        "poseWeightedRecallDeltaCI95": list(pose_weighted),
        "aggregateWeightedRecallDeltaCI95": list(aggregate_weighted),
        "weightedRecallLowerDelta": weighted_lower,
        "weightedRecallSafe": weighted_lower >= -float(weighted_drop),
        "poseRecallDiagnosticCI95": list(pose_recall),
        "badCullDeltaCI95": list(bad_cull),
        "badCullUpperBoundReportedOnly": True,
    }

    classification_metrics = (
        "pose_precision",
        "aggregate_precision",
        "pose_accuracy",
        "aggregate_accuracy",
        "pose_balanced_accuracy",
        "aggregate_balanced_accuracy",
        "pose_specificity",
        "aggregate_specificity",
        "pose_f1",
        "aggregate_f1",
        "pose_useful_cull",
        "aggregate_useful_cull",
    )
    classification = {
        metric: {"ci95": list(_ci(effect, metric)), "stablePositive": _positive(effect, metric)}
        for metric in classification_metrics
    }
    resource_metrics = (
        "pose_avg_pred_count",
        "aggregate_avg_pred_count",
        "pose_predicted_glb_bytes",
        "aggregate_predicted_glb_bytes",
        "pose_predicted_glb_count",
        "aggregate_predicted_glb_count",
        "pose_glb_bytes_at_achieved_visual_utility",
        "aggregate_glb_bytes_at_achieved_visual_utility",
    )
    resource = {
        metric: {"ci95": list(_ci(effect, metric)), "stableReduction": _negative(effect, metric)}
        for metric in resource_metrics
    }
    download_utility_metrics = (
        "pose_download_utility_recall",
        "aggregate_download_utility_recall",
    )
    download_utility = {
        metric: {"ci95": list(_ci(effect, metric)), "stableImprovement": _positive(effect, metric)}
        for metric in download_utility_metrics
    }
    image_metrics = (
        "pose_image_per",
        "aggregate_image_per",
        "pose_image_miss_pixel_rate",
        "aggregate_image_miss_pixel_rate",
        "pose_image_wrong_id_pixel_rate",
        "aggregate_image_wrong_id_pixel_rate",
        "pose_image_extra_pixel_rate",
        "aggregate_image_extra_pixel_rate",
    )
    image = {
        "status": "available" if all(metric in effect.get("metrics", {}) for metric in image_metrics) else "not_available",
        "metrics": {},
    }
    if image["status"] == "available":
        image["metrics"] = {
            metric: {"ci95": list(_ci(effect, metric)), "stableReduction": _negative(effect, metric)}
            for metric in image_metrics
        }
    safe_workpoints = effect.get("safeWorkpoints", {})
    validation_safe_workpoints = effect.get("validationSafeWorkpoints", {"bothAllSafe": True})
    safety["validationWeightedRecallSafe"] = bool(validation_safe_workpoints.get("bothAllSafe", True))
    safety_ok = bool(
        safety["weightedRecallSafe"]
        and safe_workpoints.get("bothAllSafe", False)
        and validation_safe_workpoints.get("bothAllSafe", True)
    )
    classification_gain = [metric for metric, item in classification.items() if item["stablePositive"]]
    resource_gain = [metric for metric, item in resource.items() if item["stableReduction"]]
    download_utility_gain = [metric for metric, item in download_utility.items() if item["stableImprovement"]]
    image_gain = [metric for metric, item in image.get("metrics", {}).items() if item["stableReduction"]]
    system_gain = bool(resource_gain or download_utility_gain or image_gain)
    return {
        "available": True,
        "safety": safety,
        "safeWorkpoints": safe_workpoints,
        "validationSafeWorkpoints": validation_safe_workpoints,
        "classification": classification,
        "resource": resource,
        "downloadUtility": download_utility,
        "image": image,
        "classificationImprovementMetrics": classification_gain if safety_ok else [],
        "resourceImprovementMetrics": (resource_gain + download_utility_gain) if safety_ok else [],
        "imageImprovementMetrics": image_gain if safety_ok else [],
        "hasPredictionContribution": bool(safety_ok and (classification_gain or resource_gain or download_utility_gain or image_gain)),
        "hasSystemContribution": bool(safety_ok and system_gain),
        "hasContribution": bool(safety_ok and (classification_gain or resource_gain or download_utility_gain or image_gain or system_gain)),
    }


def make_decision(
    summary: dict[str, Any],
    context_effect_name: str | None = None,
    survival_effect_name: str | None = None,
    loss_effect_name: str | None = None,
    max_weighted_recall_drop: float = 0.01,
) -> dict[str, Any]:
    if summary.get("testRead") is not False:
        raise ValueError("route decision requires a test-free validation summary")
    context_name, context_effect = _find_effect(
        summary, context_effect_name, DEFAULT_CONTEXT_EFFECT
    )
    survival_name, survival_effect = _find_effect(
        summary, survival_effect_name, DEFAULT_SURVIVAL_EFFECT
    )
    loss_name, loss_effect = _find_effect(
        summary, loss_effect_name, DEFAULT_LOSS_EFFECT
    )

    factors = {
        "context": {
            "comparison": context_name,
            **_factor_decision(context_effect, max_weighted_recall_drop),
        },
        "survivalField": {
            "comparison": survival_name,
            **_factor_decision(survival_effect, max_weighted_recall_drop),
        },
        "loss": {
            "comparison": loss_name,
            **_factor_decision(loss_effect, max_weighted_recall_drop),
        },
    }
    retained = [name for name, item in factors.items() if item.get("hasContribution")]
    route = "retain_ray_context_survival_owrb_candidate" if retained else "degrade_to_auxiliary_or_failed_ablation"
    return {
        "schema": "ray-context-survival-owrb-route-decision-v2",
        "summary": summary.get("manifest"),
        "testRead": False,
        "route": route,
        "retainedFactors": retained,
        "factors": factors,
        "policy": {
            "primarySafetyMetric": "weighted_recall",
            "weightedRecallLowerDeltaRequirement": -float(max_weighted_recall_drop),
            "poseRecallRole": "diagnostic_only",
            "badCullUpperBoundRouteVetoEnabled": False,
            "usefulCullNotSufficient": True,
            "testThresholdSelection": False,
        },
        "defaultModelChanged": False,
        "defaultFrontendChanged": False,
    }


def self_test() -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    all_metrics = (
        "pose_weighted_recall", "aggregate_weighted_recall", "pose_recall", "pose_bad_cull",
        "pose_precision", "aggregate_precision", "pose_accuracy", "aggregate_accuracy",
        "pose_balanced_accuracy", "aggregate_balanced_accuracy", "pose_specificity", "aggregate_specificity",
        "pose_f1", "aggregate_f1", "pose_useful_cull", "aggregate_useful_cull",
        "pose_avg_pred_count", "aggregate_avg_pred_count", "pose_predicted_glb_bytes", "aggregate_predicted_glb_bytes",
        "pose_predicted_glb_count", "aggregate_predicted_glb_count",
        "pose_glb_bytes_at_achieved_visual_utility", "aggregate_glb_bytes_at_achieved_visual_utility",
        "pose_download_utility_recall", "aggregate_download_utility_recall",
        "pose_image_per", "aggregate_image_per", "pose_image_miss_pixel_rate", "aggregate_image_miss_pixel_rate",
        "pose_image_wrong_id_pixel_rate", "aggregate_image_wrong_id_pixel_rate", "pose_image_extra_pixel_rate", "aggregate_image_extra_pixel_rate",
    )
    for metric in all_metrics:
        metrics[metric] = {"meanDelta": 0.01, "ci95": [0.001, 0.02], "crossesZero": False}
    summary = {
        "manifest": "synthetic",
        "testRead": False,
        "comparisons": {
            DEFAULT_CONTEXT_EFFECT: {"metrics": metrics, "safeWorkpoints": {"bothAllSafe": True}},
            DEFAULT_SURVIVAL_EFFECT: {"metrics": metrics, "safeWorkpoints": {"bothAllSafe": True}},
            DEFAULT_LOSS_EFFECT: {"metrics": metrics, "safeWorkpoints": {"bothAllSafe": True}},
        },
    }
    decision = make_decision(summary)
    assert decision["route"] == "retain_ray_context_survival_owrb_candidate"
    assert set(decision["retainedFactors"]) == {"context", "survivalField", "loss"}
    return {"status": "ok", "route": decision["route"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--context-effect")
    parser.add_argument("--survival-effect")
    parser.add_argument("--loss-effect")
    parser.add_argument("--max-weighted-recall-drop", type=float, default=0.01)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False))
        return
    if args.summary is None or args.output is None:
        parser.error("--summary and --output are required unless --self-test is used")
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    result = make_decision(
        summary,
        args.context_effect,
        args.survival_effect,
        args.loss_effect,
        args.max_weighted_recall_drop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "decided", "output": str(args.output.resolve()), "route": result["route"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
