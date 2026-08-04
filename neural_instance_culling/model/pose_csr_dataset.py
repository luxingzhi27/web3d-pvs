from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np


def _project_aabb_features_numpy(aabbs: np.ndarray, mvp: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """用 numpy 计算 AABB 的屏幕矩形、面积和深度。

    这里用于 pose-set 训练时的候选裁剪。裁剪不能随机丢负样本，否则模型会少见到
    真正困难的遮挡负例；因此根据当前 MVP 选择屏幕面积大、深度近、与 GT 投影重叠的 hard negative。
    """
    if aabbs.size == 0:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.ones((0,), dtype=np.float32),
            np.zeros((0,), dtype=bool),
        )
    mins = aabbs[:, :3].astype(np.float32, copy=False)
    maxs = aabbs[:, 3:].astype(np.float32, copy=False)
    corners = np.stack(
        [
            np.stack([mins[:, 0], mins[:, 1], mins[:, 2]], axis=-1),
            np.stack([mins[:, 0], mins[:, 1], maxs[:, 2]], axis=-1),
            np.stack([mins[:, 0], maxs[:, 1], mins[:, 2]], axis=-1),
            np.stack([mins[:, 0], maxs[:, 1], maxs[:, 2]], axis=-1),
            np.stack([maxs[:, 0], mins[:, 1], mins[:, 2]], axis=-1),
            np.stack([maxs[:, 0], mins[:, 1], maxs[:, 2]], axis=-1),
            np.stack([maxs[:, 0], maxs[:, 1], mins[:, 2]], axis=-1),
            np.stack([maxs[:, 0], maxs[:, 1], maxs[:, 2]], axis=-1),
        ],
        axis=1,
    )
    e = np.asarray(mvp, dtype=np.float32).reshape(16)
    x = corners[..., 0] * e[0] + corners[..., 1] * e[4] + corners[..., 2] * e[8] + e[12]
    y = corners[..., 0] * e[1] + corners[..., 1] * e[5] + corners[..., 2] * e[9] + e[13]
    z = corners[..., 0] * e[2] + corners[..., 1] * e[6] + corners[..., 2] * e[10] + e[14]
    w = corners[..., 0] * e[3] + corners[..., 1] * e[7] + corners[..., 2] * e[11] + e[15]
    front = w > 1e-4
    safe_w = np.where(front, w, 1.0)
    ndc_x = np.clip(x / safe_w, -8.0, 8.0)
    ndc_y = np.clip(y / safe_w, -8.0, 8.0)
    valid_count = front.sum(axis=1)
    valid = valid_count > 0
    rect_min_x = np.where(valid, np.where(front, ndc_x, 8.0).min(axis=1), 0.0)
    rect_max_x = np.where(valid, np.where(front, ndc_x, -8.0).max(axis=1), 0.0)
    rect_min_y = np.where(valid, np.where(front, ndc_y, 8.0).min(axis=1), 0.0)
    rect_max_y = np.where(valid, np.where(front, ndc_y, -8.0).max(axis=1), 0.0)
    rect = np.stack(
        [
            np.clip(rect_min_x, -1.0, 1.0),
            np.clip(rect_min_y, -1.0, 1.0),
            np.clip(rect_max_x, -1.0, 1.0),
            np.clip(rect_max_y, -1.0, 1.0),
        ],
        axis=-1,
    ).astype(np.float32, copy=False)
    wh = np.maximum(rect[:, 2:] - rect[:, :2], 0.0)
    area = np.clip(wh[:, 0] * wh[:, 1] * 0.25, 0.0, 4.0).astype(np.float32, copy=False)
    depth = np.where(front, w, 0.0).sum(axis=1) / np.maximum(valid_count, 1)
    depth = np.maximum(depth.astype(np.float32, copy=False), 1e-3)
    return rect, area, depth, valid


def _rect_iou_to_gt(candidate_rect: np.ndarray, gt_rect: np.ndarray) -> np.ndarray:
    if candidate_rect.size == 0 or gt_rect.size == 0:
        return np.zeros((candidate_rect.shape[0],), dtype=np.float32)
    x0 = np.maximum(candidate_rect[:, None, 0], gt_rect[None, :, 0])
    y0 = np.maximum(candidate_rect[:, None, 1], gt_rect[None, :, 1])
    x1 = np.minimum(candidate_rect[:, None, 2], gt_rect[None, :, 2])
    y1 = np.minimum(candidate_rect[:, None, 3], gt_rect[None, :, 3])
    inter = np.maximum(x1 - x0, 0.0) * np.maximum(y1 - y0, 0.0)
    cand_area = np.maximum(candidate_rect[:, 2] - candidate_rect[:, 0], 0.0) * np.maximum(candidate_rect[:, 3] - candidate_rect[:, 1], 0.0)
    gt_area = np.maximum(gt_rect[:, 2] - gt_rect[:, 0], 0.0) * np.maximum(gt_rect[:, 3] - gt_rect[:, 1], 0.0)
    union = cand_area[:, None] + gt_area[None, :] - inter
    return (inter / np.maximum(union, 1e-8)).max(axis=1).astype(np.float32, copy=False)


def _cap_candidates_projection_aware(
    candidate_ids: np.ndarray,
    visible_unique: np.ndarray,
    world_aabbs: np.ndarray,
    mvp: np.ndarray | None,
    max_candidates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """裁剪超大候选集合，但保留 GT 和最像遮挡物的 hard negatives。"""
    if max_candidates <= 0 or candidate_ids.size <= max_candidates:
        return candidate_ids
    if visible_unique.size >= max_candidates:
        return visible_unique[:max_candidates].astype(np.uint32, copy=False)
    negatives = np.setdiff1d(candidate_ids, visible_unique, assume_unique=True)
    neg_budget = max_candidates - visible_unique.size
    if negatives.size <= neg_budget:
        return np.union1d(visible_unique, negatives.astype(np.uint32, copy=False))
    if mvp is None:
        selected = rng.choice(negatives, size=neg_budget, replace=False).astype(np.uint32, copy=False)
        return np.union1d(visible_unique, selected)

    neg_rect, neg_area, neg_depth, neg_valid = _project_aabb_features_numpy(world_aabbs[negatives], mvp)
    gt_rect, gt_area, gt_depth, gt_valid = _project_aabb_features_numpy(world_aabbs[visible_unique], mvp)
    overlap = _rect_iou_to_gt(neg_rect, gt_rect[gt_valid]) if bool(gt_valid.any()) else np.zeros((negatives.size,), dtype=np.float32)
    if bool(gt_valid.any()):
        gt_depth_ref = float(np.median(gt_depth[gt_valid]))
    else:
        gt_depth_ref = float(np.median(neg_depth))
    depth_score = 1.0 / (1.0 + np.abs(np.log1p(neg_depth / 100.0) - np.log1p(gt_depth_ref / 100.0)) * 4.0)
    near_score = 1.0 / (1.0 + neg_depth / 250.0)
    area_score = np.log1p(np.clip(neg_area, 0.0, 4.0) * 2048.0) / np.log1p(4.0 * 2048.0)
    random_tie = rng.random(negatives.size, dtype=np.float32) * 1e-4
    score = overlap * 3.0 + area_score * 1.25 + depth_score * 0.75 + near_score * 0.5 + neg_valid.astype(np.float32) * 0.1 + random_tie
    top = np.argpartition(score, -neg_budget)[-neg_budget:]
    selected = negatives[top].astype(np.uint32, copy=False)
    return np.union1d(visible_unique, selected)


POSE_DTYPE = np.dtype(
    {
        "names": ["camera_norm", "camera_world", "split", "category", "reserved0", "reserved1"],
        "formats": [("<f4", (3,)), ("<f4", (3,)), "u1", "u1", "<u2", "<u4"],
        "offsets": [0, 12, 24, 25, 26, 28],
        "itemsize": 32,
    }
)

# directional 数据集是当前正式格式：每个 pose 保存相机位置、forward 和 tanHalfFovX/Y。
# 这样训练时可以复现某一个具体视角，而不是把同位置多个方向粗暴合并。
DIRECTIONAL_POSE_DTYPE = np.dtype(
    {
        "names": [
            "camera_norm",
            "camera_world",
            "camera_forward",
            "camera_view",
            "split",
            "category",
            "reserved0",
            "reserved1",
        ],
        "formats": [
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (2,)),
            "u1",
            "u1",
            "<u2",
            "<u4",
        ],
        "offsets": [0, 12, 24, 36, 44, 45, 46, 48],
        "itemsize": 64,
    }
)


class PoseCSRSplit:
    """某个 split(train/val/test) 的轻量访问器。

    数据本体都在 PoseCSRDataset 的 memmap 中，这里只保存 pose index。
    训练时按需从 CSR 中切出 visible/frustum 实例，避免把巨大 `(pose, instance)` 表加载进内存。
    """

    def __init__(
        self,
        dataset: "PoseCSRDataset",
        split_name: str,
        pose_indices: np.ndarray | None = None,
        split_id: int | None = None,
    ):
        self.dataset = dataset
        self.split_name = split_name
        if pose_indices is None:
            self.split_id = dataset.split_ids[split_name] if split_id is None else int(split_id)
            selected = np.flatnonzero(dataset.poses["split"] == self.split_id)
        else:
            self.split_id = int(split_id) if split_id is not None else None
            selected = np.asarray(pose_indices, dtype=np.int64).reshape(-1)
        self.pose_indices = np.unique(selected.astype(np.int64, copy=False))
        self.pose_indices_with_visible = self.pose_indices[dataset.visible_counts[self.pose_indices] > 0]
        self.total_positive_refs = int(dataset.visible_counts[self.pose_indices].sum())
        self.total_frustum_refs = int(dataset.frustum_counts[self.pose_indices].sum())

    def pose_batches(
        self,
        batch_size: int,
        negative_multiplier: int,
        rng: np.random.Generator,
        max_steps: int | None,
        positives_per_pose: int = 16,
    ):
        examples_per_pose = max(1, int(positives_per_pose)) * max(1, negative_multiplier + 1)
        poses_per_batch = max(1, batch_size // examples_per_pose)
        if self.pose_indices_with_visible.size == 0:
            return
        order = rng.permutation(self.pose_indices_with_visible)
        total_steps = math.ceil(order.size / poses_per_batch)
        if max_steps is not None and max_steps > total_steps:
            for _step in range(max_steps):
                replace = self.pose_indices_with_visible.size < poses_per_batch
                yield rng.choice(self.pose_indices_with_visible, size=poses_per_batch, replace=replace)
            return
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        for step in range(total_steps):
            start = step * poses_per_batch
            end = min(order.size, start + poses_per_batch)
            yield order[start:end]

    def pose_set_batches(
        self,
        poses_per_batch: int,
        rng: np.random.Generator,
        max_steps: int | None,
        include_empty: bool = False,
    ):
        poses_per_batch = max(1, int(poses_per_batch))
        source_indices = self.pose_indices if include_empty else self.pose_indices_with_visible
        if source_indices.size == 0:
            return
        if max_steps is not None:
            replace = source_indices.size < poses_per_batch
            for _step in range(max(1, int(max_steps))):
                yield rng.choice(source_indices, size=poses_per_batch, replace=replace)
            return
        order = rng.permutation(source_indices)
        for start in range(0, order.size, poses_per_batch):
            yield order[start:min(order.size, start + poses_per_batch)]

    def build_training_batch(
        self,
        pose_indices: np.ndarray,
        negative_multiplier: int,
        rng: np.random.Generator,
        importance_pixels: float,
        batch_size: int,
        positives_per_pose: int = 16,
    ) -> dict[str, np.ndarray]:
        # sample 模式：每个 pose 抽若干正样本，再配 hard negative 和随机负样本。
        # 这个模式适合快速二分类训练，但最终效果不能只看它的 sample precision/recall。
        max_examples = max(
            1,
            min(batch_size, pose_indices.size * max(1, int(positives_per_pose)) * max(1, negative_multiplier + 1)),
        )
        camera = np.zeros((max_examples, 3), dtype=np.float32)
        camera_world = np.zeros((max_examples, 3), dtype=np.float32)
        camera_view = np.zeros((max_examples, 5), dtype=np.float32)
        instance = np.zeros((max_examples,), dtype=np.int64)
        target = np.zeros((max_examples,), dtype=np.float32)
        importance = np.zeros((max_examples,), dtype=np.float32)
        cursor = 0
        for pose_index in pose_indices:
            if cursor >= max_examples:
                break
            visible_ids, visible_pixels = self.dataset.visible_slice(int(pose_index))
            if visible_ids.size == 0:
                continue
            frustum_ids = self.dataset.frustum_slice(int(pose_index))
            camera_norm = self.dataset.poses["camera_norm"][pose_index]
            camera_world_pose = self.dataset.poses["camera_world"][pose_index]
            view = self.dataset.camera_view(int(pose_index))
            hard_negatives = np.setdiff1d(frustum_ids, visible_ids, assume_unique=True)
            visible_set = set(int(v) for v in visible_ids.tolist())
            positive_samples = max(1, min(int(positives_per_pose), int(visible_ids.size)))

            for _ in range(positive_samples):
                if cursor >= max_examples:
                    break
                pixel_idx = int(rng.integers(0, visible_ids.size))
                pos_pick = int(visible_ids[pixel_idx])
                camera[cursor] = camera_norm
                camera_world[cursor] = camera_world_pose
                camera_view[cursor] = view
                instance[cursor] = pos_pick
                target[cursor] = 1.0
                importance[cursor] = min(1.0, float(visible_pixels[pixel_idx]) / max(1.0, importance_pixels))
                cursor += 1
                if cursor >= max_examples:
                    break

                if hard_negatives.size > 0:
                    count = min(negative_multiplier, int(hard_negatives.size), max_examples - cursor)
                    if count > 0:
                        picks = rng.choice(hard_negatives.size, size=count, replace=False)
                        selected = hard_negatives[picks]
                        next_cursor = cursor + count
                        camera[cursor:next_cursor] = camera_norm
                        camera_world[cursor:next_cursor] = camera_world_pose
                        camera_view[cursor:next_cursor] = view
                        instance[cursor:next_cursor] = selected.astype(np.int64)
                        cursor = next_cursor
                        continue

                for _neg_index in range(negative_multiplier):
                    if cursor >= max_examples:
                        break
                    neg = self.dataset.random_non_visible_instance(visible_set, rng)
                    camera[cursor] = camera_norm
                    camera_world[cursor] = camera_world_pose
                    camera_view[cursor] = view
                    instance[cursor] = neg
                    cursor += 1

        return {
            "camera": camera[:cursor],
            "camera_world": camera_world[:cursor],
            "camera_view": camera_view[:cursor],
            "instance": instance[:cursor],
            "target": target[:cursor],
            "importance": importance[:cursor],
        }

    def build_pose_set_batch(
        self,
        pose_indices: np.ndarray,
        world_aabbs: np.ndarray,
        rng: np.random.Generator,
        max_candidates_per_pose: int = 0,
        allow_candidate_visible_union: bool = False,
        include_empty: bool = False,
    ) -> dict[str, np.ndarray]:
        # pose-set 模式：一个 pose 内所有候选实例共同组成一个集合监督问题。
        # 这更接近前端真实需求：一次预测输出的整个可见实例集合要和 GT 对齐。
        cameras: list[np.ndarray] = []
        cameras_world: list[np.ndarray] = []
        views: list[np.ndarray] = []
        mvps: list[np.ndarray] = []
        instances: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        visible_weight_targets: list[np.ndarray] = []
        visible_hit_rate_targets: list[np.ndarray] = []
        batch_pose_indices: list[int] = []
        offsets = [0]
        visible_counts = []
        candidate_counts = []

        max_candidates = int(max_candidates_per_pose or 0)
        for pose_index in pose_indices:
            pose_index = int(pose_index)
            visible_ids, visible_weights = self.dataset.visible_slice(pose_index)
            visible_hit_counts = self.dataset.visible_hit_count_slice(pose_index)
            subpose_count = self.dataset.subpose_count(pose_index)
            if visible_ids.size == 0 and not include_empty:
                continue

            camera_world = np.asarray(self.dataset.poses["camera_world"][pose_index], dtype=np.float32)
            forward = np.asarray(self.dataset.poses["camera_forward"][pose_index], dtype=np.float32)
            tan_x, tan_y = np.asarray(self.dataset.poses["camera_view"][pose_index], dtype=np.float32)
            # v3 CSR 已经把“与前端一致的候选集合”写进 frustum_ids/candidate_ids。
            # 正式运行只允许使用这份存储候选；旧数据缺失候选时的几何重算只能通过
            # allow_candidate_visible_union 显式开启，避免把候选生成错误隐藏在训练器里。
            stored_candidates = self.dataset.frustum_slice(pose_index).astype(np.uint32, copy=False)
            candidate_ids = stored_candidates
            if candidate_ids.size == 0 and allow_candidate_visible_union:
                candidate_ids = frustum_candidate_ids_for_pose(
                    camera_world, forward, float(tan_x), float(tan_y), world_aabbs
                )

            visible_unique = np.unique(visible_ids.astype(np.uint32, copy=False))
            missing_visible = np.setdiff1d(visible_unique, candidate_ids, assume_unique=False)
            if missing_visible.size and not allow_candidate_visible_union:
                raise ValueError(
                    "Formal pose-set candidate semantics failed at pose "
                    f"{pose_index}: {missing_visible.size} visible instances are absent from the stored "
                    "back-camera candidate set. Rebuild candidates or explicitly enable the exploratory "
                    "candidate-visible union."
                )
            if allow_candidate_visible_union:
                # This branch is retained only for legacy/exploratory datasets. It must never be
                # enabled for a formal benchmark because it changes the denominator and hides
                # candidate-generation errors.
                candidate_ids = np.union1d(candidate_ids, visible_unique)

            if max_candidates > 0 and candidate_ids.size > max_candidates:
                if visible_unique.size > max_candidates and not allow_candidate_visible_union:
                    raise ValueError(
                        f"max_candidates_per_pose={max_candidates} would drop "
                        f"{visible_unique.size} GT visible instances at pose {pose_index}."
                    )
                pose_mvp = self.dataset.mvp_slice(pose_index) if self.dataset.mvp is not None else None
                candidate_ids = _cap_candidates_projection_aware(
                    candidate_ids,
                    visible_unique,
                    world_aabbs,
                    pose_mvp,
                    max_candidates,
                    rng,
                )

            if candidate_ids.size == 0:
                # Evaluation needs an explicit zero-length row for empty
                # candidate poses so that the pose is counted rather than
                # silently disappearing from the denominator. Training keeps
                # the historical positive-only default via include_empty=False.
                offsets.append(offsets[-1])
                visible_counts.append(int(visible_unique.size))
                candidate_counts.append(0)
                batch_pose_indices.append(pose_index)
                continue
            label = np.isin(candidate_ids, visible_unique, assume_unique=True).astype(np.float32)
            weight_map = {int(idx): float(weight) for idx, weight in zip(visible_ids.tolist(), visible_weights.tolist())}
            weight_target = np.asarray([weight_map.get(int(idx), 0) for idx in candidate_ids.tolist()], dtype=np.float32)
            if visible_hit_counts.size:
                hit_map = {
                    int(idx): float(hit) / max(1, subpose_count)
                    for idx, hit in zip(visible_ids.tolist(), visible_hit_counts.tolist())
                }
                hit_rate_target = np.asarray(
                    [hit_map.get(int(idx), 0.0) for idx in candidate_ids.tolist()],
                    dtype=np.float32,
                )
            else:
                hit_rate_target = np.zeros((candidate_ids.size,), dtype=np.float32)
            count = int(candidate_ids.size)
            cameras.append(np.repeat(self.dataset.poses["camera_norm"][pose_index][None, :], count, axis=0).astype(np.float32, copy=False))
            cameras_world.append(np.repeat(camera_world[None, :], count, axis=0).astype(np.float32, copy=False))
            views.append(np.repeat(self.dataset.camera_view(pose_index)[None, :], count, axis=0).astype(np.float32, copy=False))
            if self.dataset.mvp is not None:
                mvps.append(np.repeat(self.dataset.mvp_slice(pose_index)[None, :], count, axis=0).astype(np.float32, copy=False))
            instances.append(candidate_ids.astype(np.int64, copy=False))
            targets.append(label)
            visible_weight_targets.append(weight_target)
            visible_hit_rate_targets.append(hit_rate_target)
            offsets.append(offsets[-1] + count)
            visible_counts.append(int(visible_unique.size))
            candidate_counts.append(count)
            batch_pose_indices.append(pose_index)

        if not instances:
            return {
                "camera": np.zeros((0, 3), dtype=np.float32),
                "camera_world": np.zeros((0, 3), dtype=np.float32),
                "camera_view": np.zeros((0, 5), dtype=np.float32),
                "instance": np.zeros((0,), dtype=np.int64),
                "target": np.zeros((0,), dtype=np.float32),
                "visible_weights": np.zeros((0,), dtype=np.float32),
                "visible_hit_rates": np.zeros((0,), dtype=np.float32),
                "visible_pixels": np.zeros((0,), dtype=np.float32),
                "pose_offsets": np.asarray(offsets, dtype=np.int64),
                "visible_counts": np.asarray(visible_counts, dtype=np.int64),
                "candidate_counts": np.asarray(candidate_counts, dtype=np.int64),
                "pose_indices": np.asarray(batch_pose_indices, dtype=np.int64),
            }

        out = {
            "camera": np.concatenate(cameras, axis=0),
            "camera_world": np.concatenate(cameras_world, axis=0),
            "camera_view": np.concatenate(views, axis=0),
            "instance": np.concatenate(instances, axis=0),
            "target": np.concatenate(targets, axis=0),
            "visible_weights": np.concatenate(visible_weight_targets, axis=0),
            "visible_hit_rates": np.concatenate(visible_hit_rate_targets, axis=0),
            # 兼容旧训练脚本。新模型不要把这个字段当真实像素覆盖。
            "visible_pixels": np.concatenate(visible_weight_targets, axis=0),
            "pose_offsets": np.asarray(offsets, dtype=np.int64),
            "visible_counts": np.asarray(visible_counts, dtype=np.int64),
            "candidate_counts": np.asarray(candidate_counts, dtype=np.int64),
            "pose_indices": np.asarray(batch_pose_indices, dtype=np.int64),
        }
        if mvps:
            out["mvp"] = np.concatenate(mvps, axis=0)
        return out

    def eval_batches(self, batch_size: int, importance_pixels: float):
        for pose_index in self.pose_indices:
            visible_ids, visible_pixels = self.dataset.visible_slice(int(pose_index))
            frustum_ids = self.dataset.frustum_slice(int(pose_index))
            hard_negatives = np.setdiff1d(frustum_ids, visible_ids, assume_unique=True)
            if hard_negatives.size == 0 and visible_ids.size > 0:
                rng = np.random.default_rng(0x5EED + int(pose_index))
                visible_set = set(int(v) for v in visible_ids.tolist())
                random_negatives = []
                target_count = min(max(visible_ids.size * 4, 32), 4096)
                while len(random_negatives) < target_count:
                    candidate = self.dataset.random_non_visible_instance(visible_set, rng)
                    if candidate not in visible_set:
                        random_negatives.append(candidate)
                        visible_set.add(candidate)
                hard_negatives = np.array(random_negatives, dtype=np.uint32)
            total = visible_ids.size + hard_negatives.size
            if total == 0:
                continue
            candidate_ids = np.concatenate([visible_ids, hard_negatives]).astype(np.int64, copy=False)
            targets = np.concatenate(
                [
                    np.ones(visible_ids.size, dtype=np.float32),
                    np.zeros(hard_negatives.size, dtype=np.float32),
                ]
            )
            importance = np.concatenate(
                [
                    np.clip(visible_pixels.astype(np.float32) / max(1.0, importance_pixels), 0.0, 1.0),
                    np.zeros(hard_negatives.size, dtype=np.float32),
                ]
            )
            camera_norm = self.dataset.poses["camera_norm"][pose_index]
            camera_world = self.dataset.poses["camera_world"][pose_index]
            view = self.dataset.camera_view(int(pose_index))
            for start in range(0, total, batch_size):
                end = min(total, start + batch_size)
                count = end - start
                yield {
                    "camera": np.repeat(camera_norm[None, :], count, axis=0).astype(np.float32, copy=False),
                    "camera_world": np.repeat(camera_world[None, :], count, axis=0).astype(np.float32, copy=False),
                    "camera_view": np.repeat(view[None, :], count, axis=0).astype(np.float32, copy=False),
                    "instance": candidate_ids[start:end],
                    "target": targets[start:end],
                    "importance": importance[start:end],
                }


class PoseCSRDataset:
    """pose 级 CSR 数据集读取器。

    文件组织方式：
    - poses.bin 存每个 pose 的固定相机信息。
    - visible_offsets/ids/weights 存 GT 可见实例和 rvcServer 重要性权重。
    - frustum_offsets/ids 存当前视锥附近候选实例，用于 hard negative。

    注意：旧 CSR 数据集中 `visible_pixels.bin` 实际来自 rvcServer
    `component_weights`，不是严格像素数。这里为了兼容旧脚本会继续提供
    `visible_pixels` 别名，但新模型必须按 `visible_weights` 解释。
    """

    def __init__(self, dataset_dir: str | Path, num_instances: int):
        self.dataset_dir = Path(dataset_dir)
        self.meta = self._read_json("dataset_meta.json")
        self.split_ids = self.meta.get("splitIds", {"train": 0, "val": 1, "test": 2})
        self.num_instances = int(num_instances)
        pose_stride = int(self.meta.get("poseStrideBytes", POSE_DTYPE.itemsize))
        self.pose_dtype = DIRECTIONAL_POSE_DTYPE if pose_stride == DIRECTIONAL_POSE_DTYPE.itemsize else POSE_DTYPE
        self.poses = np.memmap(self.dataset_dir / "poses.bin", dtype=self.pose_dtype, mode="r")
        self.visible_offsets = np.memmap(self.dataset_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
        self.visible_ids = np.memmap(self.dataset_dir / "visible_ids.bin", dtype=np.uint32, mode="r")
        weights_path = self.dataset_dir / "visible_weights.bin"
        legacy_pixels_path = self.dataset_dir / "visible_pixels.bin"
        self.visible_weight_source = "visible_weights.bin"
        self.visible_weight_semantics = self.meta.get(
            "visibleWeightSemantics",
            "rvcServer component_weights, not pixel count",
        )
        if weights_path.exists():
            weight_dtype_name = str(self.meta.get("visibleWeightDtype", "float32")).lower()
            weight_dtype = np.float32 if weight_dtype_name in {"float", "float32", "f4", "<f4"} else np.uint32
            self.visible_weights = np.memmap(weights_path, dtype=weight_dtype, mode="r")
        elif legacy_pixels_path.exists():
            self.visible_weights = np.memmap(legacy_pixels_path, dtype=np.uint32, mode="r")
            self.visible_weight_source = "visible_pixels.bin"
            self.visible_weight_semantics = "legacy visible_pixels.bin interpreted as rvcServer component_weights, not pixel count"
            warnings.warn(
                f"{legacy_pixels_path} is a legacy field; interpreting it as visible_weights, not pixel coverage.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise FileNotFoundError(f"Expected visible_weights.bin or legacy visible_pixels.bin in {self.dataset_dir}")
        self.visible_pixels = self.visible_weights
        hit_counts_path = self.dataset_dir / "visible_hit_counts.bin"
        subpose_offsets_path = self.dataset_dir / "subpose_offsets.bin"
        self.visible_hit_counts = None
        self.subpose_offsets = None
        self.subpose_counts = None
        if hit_counts_path.exists():
            if hit_counts_path.stat().st_size % np.dtype(np.uint16).itemsize:
                raise ValueError(f"{hit_counts_path} has a non-integral uint16 length")
            self.visible_hit_counts = np.memmap(hit_counts_path, dtype=np.uint16, mode="r")
            if self.visible_hit_counts.size != self.visible_ids.size:
                raise ValueError(
                    f"{hit_counts_path} has {self.visible_hit_counts.size} entries, "
                    f"but visible_ids.bin has {self.visible_ids.size}"
                )
        if subpose_offsets_path.exists():
            self.subpose_offsets = np.memmap(subpose_offsets_path, dtype=np.uint64, mode="r")
            if self.subpose_offsets.size != self.poses.size + 1:
                raise ValueError(
                    f"{subpose_offsets_path} has {self.subpose_offsets.size} entries, "
                    f"expected {self.poses.size + 1}"
                )
            self.subpose_counts = np.diff(self.subpose_offsets).astype(np.int64, copy=False)
        if self.visible_hit_counts is not None and self.subpose_counts is None:
            raise ValueError("visible_hit_counts.bin requires subpose_offsets.bin")
        self.has_subpose_robust_labels = self.visible_hit_counts is not None and self.subpose_counts is not None
        self.subpose_robust_label_semantics = (
            "visible_hit_counts divided by successful dense subpose count; max-pooled screen weight remains separate"
            if self.has_subpose_robust_labels
            else "unavailable"
        )
        mvp_path = self.dataset_dir / "mvp.bin"
        self.mvp = None
        if mvp_path.exists() and mvp_path.stat().st_size > 0:
            self.mvp = np.memmap(mvp_path, dtype=np.float32, mode="r").reshape(-1, 16)
        self.frustum_offsets = np.memmap(self.dataset_dir / "frustum_offsets.bin", dtype=np.uint64, mode="r")
        frustum_path = self.dataset_dir / "frustum_ids.bin"
        if frustum_path.stat().st_size == 0:
            self.frustum_ids = np.zeros((0,), dtype=np.uint32)
        else:
            self.frustum_ids = np.memmap(frustum_path, dtype=np.uint32, mode="r")
        self.visible_counts = np.diff(self.visible_offsets).astype(np.int64, copy=False)
        self.frustum_counts = np.diff(self.frustum_offsets).astype(np.int64, copy=False)

    def _read_json(self, name: str) -> dict:
        import json

        return json.loads((self.dataset_dir / name).read_text(encoding="utf-8"))

    def split(self, split_name: str) -> PoseCSRSplit:
        return PoseCSRSplit(self, split_name)

    def subset(self, subset_name: str, pose_indices: np.ndarray) -> PoseCSRSplit:
        """Return a deterministic view over an explicit pose-index subset.

        The underlying CSR arrays remain shared and read-only.  This is used by
        the formal training protocol to keep a held-out calibration subset
        disjoint from the training poses without rewriting the raw dataset.
        """
        return PoseCSRSplit(self, subset_name, pose_indices=np.asarray(pose_indices, dtype=np.int64))

    def visible_slice(self, pose_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.visible_offsets[pose_index])
        end = int(self.visible_offsets[pose_index + 1])
        return self.visible_ids[start:end], self.visible_weights[start:end]

    def visible_weight_slice(self, pose_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return visible ids and rvcServer importance weights for one pose."""
        return self.visible_slice(pose_index)

    def visible_hit_count_slice(self, pose_index: int) -> np.ndarray:
        """Return per-instance dense-subpose hit counts for one view-cell."""
        if self.visible_hit_counts is None:
            return np.zeros((0,), dtype=np.uint16)
        start = int(self.visible_offsets[pose_index])
        end = int(self.visible_offsets[pose_index + 1])
        return self.visible_hit_counts[start:end]

    def subpose_count(self, pose_index: int) -> int:
        """Return the number of successful dense subposes represented by a pose."""
        if self.subpose_counts is None:
            return 0
        return int(self.subpose_counts[pose_index])

    def frustum_slice(self, pose_index: int) -> np.ndarray:
        start = int(self.frustum_offsets[pose_index])
        end = int(self.frustum_offsets[pose_index + 1])
        return self.frustum_ids[start:end]

    def camera_view(self, pose_index: int) -> np.ndarray:
        if "camera_forward" in self.poses.dtype.names and "camera_view" in self.poses.dtype.names:
            forward = np.asarray(self.poses["camera_forward"][pose_index], dtype=np.float32)
            tan_fovs = np.asarray(self.poses["camera_view"][pose_index], dtype=np.float32)
            return np.concatenate([forward, tan_fovs], axis=0).astype(np.float32, copy=False)
        # Backward-compatible neutral 66deg/16:9 view for old position-only datasets.
        tan_y = np.float32(math.tan(math.radians(66.0) * 0.5))
        tan_x = np.float32(tan_y * (16.0 / 9.0))
        return np.asarray([0.0, 0.0, -1.0, tan_x, tan_y], dtype=np.float32)

    def mvp_slice(self, pose_index: int) -> np.ndarray:
        if self.mvp is None:
            return np.zeros((16,), dtype=np.float32)
        return np.asarray(self.mvp[pose_index], dtype=np.float32)

    def random_non_visible_instance(self, visible_set: set[int], rng: np.random.Generator) -> int:
        for _attempt in range(128):
            candidate = int(rng.integers(0, self.num_instances))
            if candidate not in visible_set:
                return candidate
        return 0


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.astype(np.float32, copy=True)
    return (v / n).astype(np.float32, copy=False)


def frustum_candidate_ids_for_pose(
    camera_world: np.ndarray,
    forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    world_aabbs: np.ndarray,
    near: float = 0.05,
) -> np.ndarray:
    """训练/评估阶段的保守 AABB-vs-camera-frustum 候选测试。

    它近似前端 Three.js Frustum.intersectsBox 的作用：保留投影区间可能和水平/垂直
    FOV 重叠的 AABB。这里故意偏保守，宁愿多保留候选，也不要漏掉 GT 正样本。
    漏正样本会直接破坏 set-level 训练，多几个负样本只会增加一点训练压力。
    """
    if world_aabbs.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    f = _normalize(np.asarray(forward, dtype=np.float32), np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_ref))) > 0.98:
        up_ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(f, up_ref), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = _normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    mins = world_aabbs[:, :3]
    maxs = world_aabbs[:, 3:]
    centers = (mins + maxs) * 0.5
    extents = np.maximum((maxs - mins) * 0.5, 0.0)
    d = centers - np.asarray(camera_world, dtype=np.float32)[None, :]
    z = d @ f
    x = d @ right
    y = d @ up
    rz = extents @ np.abs(f)
    rx = extents @ np.abs(right)
    ry = extents @ np.abs(up)
    front = z + rz >= float(near)
    x_inside = (np.abs(x) - rx) <= ((z + rz) * max(float(tan_x), 1e-4))
    y_inside = (np.abs(y) - ry) <= ((z + rz) * max(float(tan_y), 1e-4))
    return np.flatnonzero(front & x_inside & y_inside).astype(np.uint32)
