#!/usr/bin/env python3
"""计算 Color-ID / instance-id buffer 的图像级 PER。

本脚本只负责像素差异统计，不负责渲染。正式图像差异管线应先用 rvcServer
或离屏渲染器生成同一真实相机下的 reference_id_buffer 和 test_id_buffer：

  reference: 完整场景实例 ID 图
  test: 使用模型预测 PVS 剔除后的实例 ID 图

ID 约定：0 表示背景；componentGlobalId 使用 id + 1 编码。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute image-level PER from reference/test instance id buffers.")
    parser.add_argument("--reference", type=Path, required=True, help=".npy or raw uint32 .bin reference id buffer")
    parser.add_argument("--test", type=Path, required=True, help=".npy or raw uint32 .bin test id buffer")
    parser.add_argument("--width", type=int, default=0, help="Required for raw .bin")
    parser.add_argument("--height", type=int, default=0, help="Required for raw .bin")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--write-diff-raw", action="store_true")
    return parser.parse_args()


def load_id_buffer(path: Path, width: int, height: int) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"{path} must be a 2D id buffer, got shape {arr.shape}")
        return arr.astype(np.uint32, copy=False)
    if width <= 0 or height <= 0:
        raise ValueError("--width and --height are required for raw .bin id buffers")
    arr = np.fromfile(path, dtype="<u4")
    expected = int(width) * int(height)
    if arr.size != expected:
        raise ValueError(f"{path} has {arr.size} uint32 values, expected {expected}")
    return arr.reshape(height, width)


def compute_metrics(reference: np.ndarray, test: np.ndarray) -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    if reference.shape != test.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape}, test {test.shape}")
    valid = reference != 0
    correct = test == reference
    error = valid & ~correct
    miss = valid & (test == 0)
    wrong_instance = valid & (test != 0) & (test != reference)
    extra = (reference == 0) & (test != 0)
    valid_count = int(valid.sum())
    total = int(reference.size)
    metrics = {
        "schema": "color-id-per-v1",
        "width": int(reference.shape[1]),
        "height": int(reference.shape[0]),
        "totalPixels": total,
        "validReferencePixels": valid_count,
        "backgroundReferencePixels": int((reference == 0).sum()),
        "errorPixels": int(error.sum()),
        "missPixels": int(miss.sum()),
        "wrongInstancePixels": int(wrong_instance.sum()),
        "extraPixels": int(extra.sum()),
        "PER": float(error.sum() / max(1, valid_count)),
        "missPixelRate": float(miss.sum() / max(1, valid_count)),
        "wrongInstancePixelRate": float(wrong_instance.sum() / max(1, valid_count)),
        "extraPixelRateOverImage": float(extra.sum() / max(1, total)),
        "backgroundReferenceRatio": float((reference == 0).sum() / max(1, total)),
    }
    missed_ids, missed_counts = np.unique(reference[miss], return_counts=True)
    per_instance = {str(int(cid - 1)): int(count) for cid, count in zip(missed_ids.tolist(), missed_counts.tolist()) if int(cid) > 0}
    diff = np.zeros(reference.shape, dtype=np.uint8)
    diff[miss] = 1
    diff[wrong_instance] = 2
    diff[extra] = 3
    return metrics, diff, per_instance


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = load_id_buffer(args.reference, args.width, args.height)
    test = load_id_buffer(args.test, args.width, args.height)
    metrics, diff, per_instance = compute_metrics(reference, test)
    (args.output_dir / "summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "per_instance_error.json").write_text(json.dumps(per_instance, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.write_diff_raw:
        diff.tofile(args.output_dir / "diff_mask_u8.bin")
    lines = [
        "# Color-ID PER Summary",
        "",
        f"- Resolution: `{metrics['width']} x {metrics['height']}`",
        f"- PER: `{metrics['PER']:.6f}`",
        f"- Miss pixel rate: `{metrics['missPixelRate']:.6f}`",
        f"- Wrong-instance pixel rate: `{metrics['wrongInstancePixelRate']:.6f}`",
        f"- Extra pixel rate over image: `{metrics['extraPixelRateOverImage']:.6f}`",
        f"- Valid reference pixels: `{metrics['validReferencePixels']}`",
        "",
        "说明：PER 使用真实相机下的 reference/test ID buffer；后退扩大视锥只用于模型输入，不能作为 PER 渲染相机。",
    ]
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
