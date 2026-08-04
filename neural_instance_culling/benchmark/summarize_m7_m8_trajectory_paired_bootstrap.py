#!/usr/bin/env python3
"""Summarize paired M7/M8 trajectory replays with hierarchical bootstrap.

The replay files already contain the pose demands and decode/upload completion
events.  This tool reconstructs per-pose weak-utility service from those
events, pairs the current cascade with the independent ranker on the same
scene/track, and reports a trajectory-cluster bootstrap confidence interval.

The clusters are navigation tracks, not training seeds.  The output therefore
improves the paired uncertainty accounting without pretending to satisfy a
three-seed model-training requirement.  Utility remains the explicitly weak
``log1p(visible_weights)`` signal used by the offline replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPLAY_SCHEMA = "neuralstreamweb3d-download-trajectory-replay-v1"
SUMMARY_SCHEMA = "neuralstreamweb3d-m7-m8-trajectory-paired-bootstrap-v2"
EPS = 1e-7


def _model_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != REPLAY_SCHEMA:
        raise ValueError(f"{path} is not a {REPLAY_SCHEMA} replay")
    models = payload.get("models")
    if not isinstance(models, list) or len(models) != 1 or not isinstance(models[0], dict):
        raise ValueError(f"{path} must contain exactly one model replay")
    return models[0]


def _utility_map(pose: dict[str, Any]) -> dict[int, float]:
    raw = pose.get("weakUtilityByGlb") or {}
    if not isinstance(raw, dict):
        raise ValueError("weakUtilityByGlb must be an object")
    result = {int(key): float(value) for key, value in raw.items()}
    if any(not np.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError("weakUtilityByGlb contains an invalid value")
    return result


def per_pose_metrics(model: dict[str, Any]) -> list[dict[str, float | int]]:
    """Reconstruct pose-window service from stored completion events."""

    trajectory = model.get("trajectory") or {}
    replay = model.get("replay") or {}
    poses = trajectory.get("poses")
    if not isinstance(poses, list) or not poses:
        raise ValueError("replay trajectory has no poses")
    horizon = float(replay.get("replayHorizonMs", -1.0))
    if not np.isfinite(horizon):
        raise ValueError("replay horizon is not finite")
    initial = {
        int(value)
        for value in ((replay.get("cache") or {}).get("initialGlbIds") or [])
    }
    events = []
    for event in ((replay.get("events") or {}).get("decodeUploads") or []):
        complete = float(event.get("completeMs", -1.0))
        glb = int(event.get("glbId", -1))
        if not np.isfinite(complete) or glb < 0:
            raise ValueError("invalid decode/upload event")
        events.append((complete, glb))
    events.sort(key=lambda item: (item[0], item[1]))

    rows: list[dict[str, float | int]] = []
    for index, pose in enumerate(poses):
        start = float(pose.get("timeMs", -1.0))
        next_time = float(poses[index + 1]["timeMs"]) if index + 1 < len(poses) else horizon
        end = min(next_time, horizon)
        if not np.isfinite(start) or not np.isfinite(end) or end + EPS < start:
            raise ValueError("pose times are not monotonic or exceed replay horizon")

        utility = _utility_map(pose)
        total = float(pose.get("weakUtilityTotal", sum(utility.values())))
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("pose weakUtilityTotal is invalid")

        completed = set(initial)
        interval_events: list[tuple[float, int]] = []
        for complete, glb in events:
            if complete <= start + EPS:
                completed.add(glb)
            elif complete <= end + EPS:
                interval_events.append((complete, glb))

        covered = float(sum(value for glb, value in utility.items() if glb in completed))
        missing_integral = 0.0
        time_cursor = start
        event_index = 0
        while event_index < len(interval_events):
            event_time = max(start, min(end, interval_events[event_index][0]))
            duration = max(0.0, event_time - time_cursor)
            missing = max(0.0, total - covered)
            missing_integral += missing * duration
            time_cursor = event_time
            while event_index < len(interval_events) and interval_events[event_index][0] <= event_time + EPS:
                _complete, glb = interval_events[event_index]
                if glb not in completed:
                    covered += float(utility.get(glb, 0.0))
                    completed.add(glb)
                event_index += 1
        final_duration = max(0.0, end - time_cursor)
        missing_integral += max(0.0, total - covered) * final_duration
        duration = max(0.0, end - start)
        rows.append(
            {
                "poseIndex": int(pose.get("poseIndex", -1)),
                "timeMs": start,
                "windowMs": duration,
                "candidateCount": int(pose.get("candidateCount", 0)),
                "candidateGlbCount": int(pose.get("candidateGlbCount", 0)),
                "weakUtilityTotal": total,
                "deadlineUtilityRecall": float(covered / total) if total > EPS else 1.0,
                "meanMissingUtilityRatio": float(missing_integral / duration / total)
                if duration > EPS and total > EPS
                else 0.0,
                "missingWeakUtilityIntegralMs": float(missing_integral),
            }
        )
    return rows


def validate_pair(current: dict[str, Any], ranker: dict[str, Any]) -> None:
    """Ensure the two replay files represent the same trajectory and demand."""

    current_poses = (current.get("trajectory") or {}).get("poses") or []
    ranker_poses = (ranker.get("trajectory") or {}).get("poses") or []
    if len(current_poses) != len(ranker_poses):
        raise ValueError("paired replays have different pose counts")
    for index, (left, right) in enumerate(zip(current_poses, ranker_poses)):
        for key in ("poseIndex", "timeMs", "candidateCount", "candidateGlbCount"):
            if left.get(key) != right.get(key):
                raise ValueError(f"paired replay mismatch at pose {index}, field {key}")
        left_utility = _utility_map(left)
        right_utility = _utility_map(right)
        if set(left_utility) != set(right_utility):
            raise ValueError(f"paired replay weak utility GLB demand mismatch at pose {index}")
        for glb in left_utility:
            if abs(left_utility[glb] - right_utility[glb]) > 1e-6:
                raise ValueError(f"paired replay weak utility mismatch at pose {index}, GLB {glb}")


def _cluster_bootstrap(
    values: list[np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if not values or any(array.size == 0 for array in values):
        raise ValueError("bootstrap needs non-empty clusters")
    if replicates < 10000:
        raise ValueError("formal M7/M8 bootstrap requires at least 10000 replicates")
    cluster_count = len(values)
    result = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected_clusters = rng.integers(0, cluster_count, size=cluster_count)
        cluster_means = []
        for cluster_index in selected_clusters.tolist():
            array = values[cluster_index]
            sample_indices = rng.integers(0, array.size, size=array.size)
            cluster_means.append(float(array[sample_indices].mean()))
        result[replicate] = float(np.mean(cluster_means))
    return result


def _effect(
    values: list[np.ndarray],
    metric: str,
    replicates: int,
    rng: np.random.Generator,
    *,
    lower_is_better: bool = False,
) -> dict[str, Any]:
    bootstrap = _cluster_bootstrap(values, replicates, rng)
    raw = float(np.mean([array.mean() for array in values]))
    improvement = -raw if lower_is_better else raw
    ci = np.quantile(bootstrap, [0.025, 0.975]).astype(float).tolist()
    improvement_ci = [-float(ci[1]), -float(ci[0])] if lower_is_better else ci
    return {
        "metric": metric,
        "rawDifferenceRanknetMinusCurrent": raw,
        "rawCi95": ci,
        "improvementPositive": improvement,
        "improvementCi95": improvement_ci,
        "improvementCiCrossesZero": bool(improvement_ci[0] <= 0.0 <= improvement_ci[1]),
        "bootstrapReplicates": int(replicates),
        "bootstrapClusters": int(len(values)),
        "bootstrapUnit": "outer trajectory cluster, inner pose resample",
    }


def _scalar(model: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = model
    for key in path:
        value = value[key]
    return float(value)


def summarize_pairs(
    pairs: Iterable[tuple[str, Path, Path]],
    *,
    replicates: int = 10000,
    seed: int = 20260804,
) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    pose_values: dict[str, list[np.ndarray]] = {
        "deadlineUtilityRecall": [],
        "meanMissingUtilityRatio": [],
        "missingWeakUtilityIntegralMs": [],
    }
    scalar_values: dict[str, list[np.ndarray]] = {
        "finalTrajectoryUtilityRecall": [],
        "meanMissingUtilityRatio": [],
        "downloadedBytes": [],
        "requestedBytes": [],
        "invalidDownloadBytes": [],
        "lateUsefulDownloadBytes": [],
    }
    first_useful_values: list[np.ndarray] = []
    for label, current_path, ranker_path in pairs:
        current = _model_payload(current_path)
        ranker = _model_payload(ranker_path)
        validate_pair(current, ranker)
        current_pose = per_pose_metrics(current)
        ranker_pose = per_pose_metrics(ranker)
        if len(current_pose) != len(ranker_pose):
            raise ValueError(f"{label}: per-pose reconstruction mismatch")
        cluster: dict[str, Any] = {
            "label": label,
            "currentFile": str(current_path),
            "rankerFile": str(ranker_path),
            "poseCount": len(current_pose),
            "poseRows": current_pose,
            "current": {},
            "ranker": {},
        }
        for metric in pose_values:
            current_array = np.asarray([float(row[metric]) for row in current_pose], dtype=np.float64)
            ranker_array = np.asarray([float(row[metric]) for row in ranker_pose], dtype=np.float64)
            pose_values[metric].append(ranker_array - current_array)
            cluster["current"][metric] = float(current_array.mean())
            cluster["ranker"][metric] = float(ranker_array.mean())
        for metric, path in {
            "finalTrajectoryUtilityRecall": ("replay", "finalTrajectoryUtilityRecall"),
            "meanMissingUtilityRatio": ("replay", "missingUtility", "meanMissingUtilityRatio"),
            "downloadedBytes": ("replay", "resourceAccounting", "downloadedBytes"),
            "requestedBytes": ("replay", "resourceAccounting", "requestedBytes"),
            "invalidDownloadBytes": ("replay", "resourceAccounting", "invalidDownloadBytes"),
            "lateUsefulDownloadBytes": ("replay", "resourceAccounting", "lateUsefulDownloadBytes"),
        }.items():
            current_value = _scalar(current, path)
            ranker_value = _scalar(ranker, path)
            scalar_values[metric].append(np.asarray([ranker_value - current_value], dtype=np.float64))
            cluster["current"][metric] = current_value
            cluster["ranker"][metric] = ranker_value
        current_first = (current.get("replay") or {}).get("firstUsefulFrame") or {}
        ranker_first = (ranker.get("replay") or {}).get("firstUsefulFrame") or {}
        if current_first.get("status") == "available" and ranker_first.get("status") == "available":
            current_value = float(current_first["timeMs"])
            ranker_value = float(ranker_first["timeMs"])
            first_useful_values.append(np.asarray([ranker_value - current_value], dtype=np.float64))
            cluster["current"]["firstUsefulFrameMs"] = current_value
            cluster["ranker"]["firstUsefulFrameMs"] = ranker_value
        else:
            cluster["current"]["firstUsefulFrameMs"] = None
            cluster["ranker"]["firstUsefulFrameMs"] = None
        clusters.append(cluster)

    rng = np.random.default_rng(int(seed))
    effects: dict[str, Any] = {}
    for metric, values in pose_values.items():
        effects[f"pose.{metric}"] = _effect(
            values,
            f"pose.{metric}",
            replicates,
            rng,
            lower_is_better=metric in {"meanMissingUtilityRatio", "missingWeakUtilityIntegralMs"},
        )
    for metric, values in scalar_values.items():
        effects[f"trajectory.{metric}"] = _effect(
            values,
            f"trajectory.{metric}",
            replicates,
            rng,
            lower_is_better=metric
            in {"meanMissingUtilityRatio", "downloadedBytes", "requestedBytes", "invalidDownloadBytes", "lateUsefulDownloadBytes"},
        )
    if first_useful_values:
        effects["trajectory.firstUsefulFrameMs"] = _effect(
            first_useful_values,
            "trajectory.firstUsefulFrameMs",
            replicates,
            rng,
            lower_is_better=True,
        )
    return {
        "schema": SUMMARY_SCHEMA,
        "protocol": {
            "pairing": "same scene and same navigation track; pose index/time/candidate counts and weak demand must match",
            "bootstrap": "outer trajectory-cluster resampling followed by inner pose resampling",
            "bootstrapReplicates": int(replicates),
            "bootstrapSeed": int(seed),
            "clusterCount": len(clusters),
            "clusterUnit": "trajectory track, not training seed",
            "candidateHash": "not_available_in_legacy_replay_schema; candidate counts and pose demand were checked",
            "testRead": False,
            "utilityBasis": "weak_log1p_visible_weights_not_pixel_coverage",
        },
        "clusters": clusters,
        "effects": effects,
        "limitations": [
            "The six registered trajectories are navigation clusters, not three independent training seeds.",
            "The replay uses a deterministic equal-share bandwidth model rather than a captured HTTP/2 or HTTP/3 trace.",
            "The utility signal is visible_weights, not pixel coverage; M5 image metrics remain authoritative for image safety.",
            "No mobile device decode/upload or frame-time evidence is inferred from this offline summary.",
        ],
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# M7/M8 轨迹配对 bootstrap 汇总",
        "",
        "状态：离线轨迹配对统计完成；不替代真实网络、移动设备或像素级效用质量门。",
        "",
        "## 口径",
        "",
        f"current cascade 与 independent ranker 只在相同场景、相同导航轨迹和相同 pose 顺序上比较。bootstrap 外层按轨迹聚类重采样，内层按该轨迹的 pose 重采样，共 `{summary['protocol']['bootstrapReplicates']:,}` 次。",
        "候选集合哈希在旧 replay schema 中没有保存，因此本次只校验 pose index、时间、候选数量、GLB 候选数量和弱效用需求；结果不能声称完成候选哈希审计。",
        "",
        "## 配对差值",
        "",
        "`improvementPositive` 已按指标方向统一：召回提高为正，缺失效用、时间和字节减少为正。",
        "",
        "| 指标 | 改善点估计 | 95% CI | 是否跨零 |",
        "|---|---:|---:|---|",
    ]
    for key, effect in summary["effects"].items():
        ci = effect["improvementCi95"]
        lines.append(
            f"| `{key}` | {effect['improvementPositive']:.6g} | [{ci[0]:.6g}, {ci[1]:.6g}] | {'是' if effect['improvementCiCrossesZero'] else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "该汇总只说明在固定离线网络假设下，两种排序路线在相同轨迹上的配对差异；它不把独立 ranker 的优势解释成联合可见性头的因果贡献，也不把弱可见权重解释成真实像素覆盖率。M7/M8 的真实网络、设备解码/上传和移动端 p95 质量门仍需独立证据。",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_pairs(args: argparse.Namespace) -> list[tuple[str, Path, Path]]:
    root = Path(args.input_root)
    scenes = tuple(value.strip() for value in args.scenes.split(",") if value.strip())
    tracks = tuple(value.strip() for value in args.tracks.split(",") if value.strip())
    pairs = []
    for scene in scenes:
        for track in tracks:
            current = root / args.current_template.format(scene=scene, track=track)
            ranker = root / args.ranker_template.format(scene=scene, track=track)
            if not current.is_file() or not ranker.is_file():
                raise FileNotFoundError(f"missing paired replay for {scene}/{track}: {current}, {ranker}")
            pairs.append((f"{scene}/{track}", current, ranker))
    return pairs


def _self_test() -> dict[str, Any]:
    def fake(model_name: str, first_ms: float) -> dict[str, Any]:
        return {
            "schema": REPLAY_SCHEMA,
            "models": [
                {
                    "model": model_name,
                    "trajectory": {"poses": [
                        {"poseIndex": 1, "timeMs": 0.0, "candidateCount": 2, "candidateGlbCount": 2, "weakUtilityByGlb": {"0": 1.0}, "weakUtilityTotal": 1.0},
                        {"poseIndex": 2, "timeMs": 100.0, "candidateCount": 2, "candidateGlbCount": 2, "weakUtilityByGlb": {"1": 1.0}, "weakUtilityTotal": 1.0},
                    ]},
                    "replay": {
                        "replayHorizonMs": 200.0,
                        "cache": {"initialGlbIds": []},
                        "events": {"decodeUploads": [{"glbId": 0, "completeMs": first_ms}, {"glbId": 1, "completeMs": first_ms + 10.0}]},
                        "firstUsefulFrame": {"status": "available", "timeMs": first_ms},
                        "finalTrajectoryUtilityRecall": 1.0,
                        "missingUtility": {"meanMissingUtilityRatio": 0.0},
                        "resourceAccounting": {"downloadedBytes": 100.0, "requestedBytes": 100.0, "invalidDownloadBytes": 0.0, "lateUsefulDownloadBytes": 0.0},
                    },
                }
            ],
        }

    left = fake("current", 50.0)
    right = fake("ranker", 25.0)
    validate_pair(left["models"][0], right["models"][0])
    summary = summarize_pairs_from_models([("self", left["models"][0], right["models"][0])], replicates=10000, seed=1)
    assert summary["protocol"]["clusterCount"] == 1
    assert summary["effects"]["trajectory.firstUsefulFrameMs"]["improvementPositive"] == 25.0
    return {"status": "passed", "checks": ["same_pose_pairing", "event_reconstruction", "bootstrap_floor"]}


def summarize_pairs_from_models(
    pairs: Iterable[tuple[str, dict[str, Any], dict[str, Any]]],
    *,
    replicates: int = 10000,
    seed: int = 20260804,
) -> dict[str, Any]:
    """In-memory equivalent used by self-test and unit tests."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for index, (label, current, ranker) in enumerate(pairs):
            current_path = Path(directory) / f"current_{index}.json"
            ranker_path = Path(directory) / f"ranker_{index}.json"
            current_path.write_text(json.dumps({"schema": REPLAY_SCHEMA, "models": [current]}), encoding="utf-8")
            ranker_path.write_text(json.dumps({"schema": REPLAY_SCHEMA, "models": [ranker]}), encoding="utf-8")
            paths.append((label, current_path, ranker_path))
        return summarize_pairs(paths, replicates=replicates, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-root", default="neural_instance_culling/benchmark/out")
    parser.add_argument("--scenes", default="hkust,metropolis")
    parser.add_argument("--tracks", default="a,b,c")
    parser.add_argument("--current-template", default="m8_formal_current_{scene}_track_{track}_20260802_retry1.json")
    parser.add_argument("--ranker-template", default="m8_formal_ranknet_{scene}_track_{track}_20260802_retry1.json")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--output-json", required=False)
    parser.add_argument("--output-md", required=False)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2))
        return
    pairs = _parse_pairs(args)
    summary = summarize_pairs(pairs, replicates=args.bootstrap_replicates, seed=args.seed)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps({"schema": SUMMARY_SCHEMA, "clusters": len(summary["clusters"]), "effects": len(summary["effects"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
