from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def load_directional_occlusion_edges(edge_dir: str | Path, num_instances: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = Path(edge_dir)
    meta = json.loads((path / "edge_meta.json").read_text(encoding="utf-8"))
    n = int(num_instances or meta["numInstances"])
    bins = int(meta["directionBins"])
    k = int(meta["edgeK"])
    dim = int(meta["edgeFeatureDim"])
    ids = np.fromfile(path / meta["files"]["edgeIds"], dtype=np.uint16).reshape(n, bins, k).astype(np.int64)
    feats = np.fromfile(path / meta["files"]["edgeStaticFeatures"], dtype=np.float16).reshape(n, bins, k, dim).astype(np.float32)
    return ids, feats, meta


def load_directional_occlusion_edge_ids(edge_dir: str | Path, num_instances: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    path = Path(edge_dir)
    meta = json.loads((path / "edge_meta.json").read_text(encoding="utf-8"))
    n = int(num_instances or meta["numInstances"])
    bins = int(meta["directionBins"])
    k = int(meta["edgeK"])
    ids = np.fromfile(path / meta["files"]["edgeIds"], dtype=np.uint16).reshape(n, bins, k).astype(np.int64)
    return ids, meta


def build_occlusion_pool_from_directional_edges(edge_ids: np.ndarray, pool_k: int) -> np.ndarray:
    edge_ids = np.asarray(edge_ids, dtype=np.int64)
    n = int(edge_ids.shape[0])
    pool = np.zeros((n, int(pool_k)), dtype=np.uint16)
    for i in range(n):
        values, counts = np.unique(edge_ids[i].reshape(-1), return_counts=True)
        keep = values != i
        values = values[keep]
        counts = counts[keep]
        if values.size == 0:
            values = np.asarray([(i + 1) % n], dtype=np.int64)
            counts = np.asarray([1], dtype=np.int64)
        order = np.argsort(-counts, kind="stable")
        chosen = values[order][: int(pool_k)]
        if chosen.size < int(pool_k):
            chosen = np.pad(chosen, (0, int(pool_k) - chosen.size), mode="edge")
        pool[i] = chosen.astype(np.uint16, copy=False)
    return pool


def load_glb_costs(
    glb_index: str | Path | None,
    glb_root: str | Path | None,
    num_glbs: int,
    instance_to_glb: np.ndarray,
    allow_missing: bool = False,
) -> np.ndarray:
    costs = np.zeros((num_glbs,), dtype=np.float32)
    if glb_index and Path(glb_index).exists():
        path = Path(glb_index)
        root = Path(glb_root or path.parent)
        obj = json.loads(path.read_text(encoding="utf-8"))
        for entry in obj.get("entries", []):
            gid = int(entry.get("globalId", -1))
            if gid < 0 or gid >= num_glbs:
                continue
            glb_path = root / str(entry.get("path", ""))
            if glb_path.exists() and glb_path.stat().st_size > 0:
                costs[gid] = float(glb_path.stat().st_size)
    used_glbs = np.unique(np.asarray(instance_to_glb, dtype=np.int64))
    missing = used_glbs[(used_glbs < 0) | (used_glbs >= num_glbs) | (costs[np.clip(used_glbs, 0, max(0, num_glbs - 1))] <= 0)]
    if missing.size and not allow_missing:
        raise FileNotFoundError(
            "Missing GLB byte costs for runtime ids "
            f"{missing[:20].tolist()}" + (" ..." if missing.size > 20 else "") + ". "
            "Formal training refuses synthetic cost imputation; use the exploratory escape hatch only for diagnostics."
        )
    if not np.any(costs > 0):
        if not allow_missing:
            raise FileNotFoundError("No usable GLB byte costs were found for formal training.")
        counts = np.bincount(np.clip(instance_to_glb.astype(np.int64), 0, num_glbs - 1), minlength=num_glbs).astype(np.float32)
        costs = np.maximum(counts, 1.0)
    else:
        fallback = np.median(costs[costs > 0])
        if allow_missing:
            costs[costs <= 0] = float(max(1.0, fallback))
    return np.log1p(costs) / max(1e-6, float(np.log1p(costs).max()))


def glb_priority_loss(
    instance_download_logits: torch.Tensor,
    instance_ids: torch.Tensor,
    target: torch.Tensor,
    visible_weights: torch.Tensor,
    instance_to_glb: torch.Tensor,
    glb_cost_norm: torch.Tensor,
    pose_offsets: torch.Tensor,
    rank_margin: float = 0.15,
    required_weight: float = 0.20,
    rank_weight: float = 0.15,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = instance_download_logits.float().view(-1)
    ids = instance_ids.long().view(-1)
    target = target.to(device=logits.device, dtype=torch.float32).view(-1)
    weights = visible_weights.to(device=logits.device, dtype=torch.float32).view(-1)
    glb_ids = torch.clamp(instance_to_glb[ids].long(), min=0)
    glb_cost_norm = glb_cost_norm.to(device=logits.device, dtype=torch.float32)
    required_terms = []
    rank_terms = []
    for pose_id in range(max(0, pose_offsets.numel() - 1)):
        start = int(pose_offsets[pose_id].item())
        end = int(pose_offsets[pose_id + 1].item())
        if end <= start:
            continue
        local_glb, inverse = torch.unique(glb_ids[start:end], sorted=True, return_inverse=True)
        g = int(local_glb.numel())
        if g == 0:
            continue
        score = torch.full((g,), -1e4, device=logits.device, dtype=logits.dtype)
        score = score.scatter_reduce(0, inverse, logits[start:end], reduce="amax", include_self=True)
        pos = target[start:end] > 0.5
        required = torch.zeros((g,), device=logits.device, dtype=logits.dtype)
        if pos.any():
            required = required.scatter_reduce(0, inverse[pos], target[start:end][pos], reduce="amax", include_self=True)
        required_terms.append(F.binary_cross_entropy_with_logits(score, required))
        util_raw = torch.zeros((g,), device=logits.device, dtype=logits.dtype)
        if pos.any():
            util_raw = util_raw.scatter_add(0, inverse[pos], torch.log1p(torch.clamp(weights[start:end][pos], min=0.0)))
            count = torch.zeros((g,), device=logits.device, dtype=logits.dtype)
            count = count.scatter_add(0, inverse[pos], torch.ones_like(weights[start:end][pos]))
            cost = torch.clamp(glb_cost_norm[local_glb], min=0.05)
            utility = (util_raw + 0.25 * torch.log1p(count)) / cost
            pos_glb = utility > 0
            neg_glb = ~pos_glb
            if pos_glb.any() and neg_glb.any():
                pos_scores = score[pos_glb]
                neg_scores = score[neg_glb]
                pos_utility = utility[pos_glb]
                if pos_scores.numel() > 32:
                    top = torch.topk(pos_utility, k=32).indices
                    pos_scores = pos_scores[top]
                    pos_utility = pos_utility[top]
                neg_scores = torch.topk(neg_scores, k=min(128, neg_scores.numel())).values
                rank = F.softplus(neg_scores[:, None] - pos_scores[None, :] + rank_margin)
                weight = torch.clamp(pos_utility[None, :] / torch.clamp(pos_utility.max(), min=1e-6), 0.1, 1.0)
                rank_terms.append((rank * weight).mean())
    required_loss = torch.stack(required_terms).mean() if required_terms else torch.zeros((), device=logits.device)
    rank_loss = torch.stack(rank_terms).mean() if rank_terms else torch.zeros((), device=logits.device)
    return float(required_weight) * required_loss + float(rank_weight) * rank_loss, {
        "lossGlbRequired": float(required_loss.detach().cpu()),
        "lossGlbPriorityRank": float(rank_loss.detach().cpu()),
    }
