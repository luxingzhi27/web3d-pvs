#!/usr/bin/env python3
"""Plot the paper safety-efficiency curves from generated metric JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "Full": "#1f4e79",
    "No survival field": "#d55e00",
    "Generic-28 memory": "#7f7f7f",
    "No hierarchical relation": "#009e73",
    "No view-cell moment": "#cc79a7",
    "No recall protection": "#e69f00",
}


def parse_curve(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--curve must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("--curve must use a non-empty LABEL=PATH")
    return label.strip(), Path(path).resolve()


def load_validation_curve(path: Path) -> list[dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("testRead") is not False:
        raise ValueError(f"threshold curve is not test-free: {path}")
    rows = [row for row in payload.get("rows", []) if row.get("split") == "validation"]
    if not rows:
        raise ValueError(f"threshold curve has no validation rows: {path}")
    return rows


def plot(curves: list[tuple[str, Path]], output_prefix: Path) -> dict[str, object]:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for label, path in curves:
        rows = load_validation_curve(path)
        rows.sort(key=lambda row: float(row["useful_cull"]))
        recall = np.asarray([float(row["weighted_recall"]) * 100.0 for row in rows])
        useful = np.asarray([float(row["useful_cull"]) * 100.0 for row in rows])
        glb_mib = np.asarray([float(row["avg_pred_glb_bytes"]) / (1024.0 * 1024.0) for row in rows])
        color = COLORS.get(label)
        width = 2.4 if label == "Full" else 1.35
        axes[0].plot(useful, recall, label=label, color=color, linewidth=width)
        order = np.argsort(glb_mib)
        axes[1].plot(glb_mib[order], recall[order], label=label, color=color, linewidth=width)
    axes[0].set_xlabel("Useful cull (%)")
    axes[0].set_ylabel("Weighted recall (%)")
    axes[1].set_xlabel("Predicted GLB bytes per view-cell (MiB)")
    axes[1].set_ylabel("Weighted recall (%)")
    for axis in axes:
        axis.axhline(99.0, color="#222222", linestyle="--", linewidth=1.0, label="99% safety target")
        axis.set_ylim(98.0, 100.05)
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower left")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Safety-efficiency trade-off on HKUST validation")
    fig.tight_layout()
    output_prefix = output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in (".png", ".pdf", ".svg"):
        path = output_prefix.with_suffix(suffix)
        fig.savefig(path, dpi=180 if suffix == ".png" else None)
        outputs.append(str(path))
    plt.close(fig)
    manifest = {
        "schema": "pvs-threshold-curve-figure-v1",
        "split": "validation",
        "testRead": False,
        "curves": [{"label": label, "source": str(path)} for label, path in curves],
        "figures": outputs,
    }
    output_prefix.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curve", action="append", type=parse_curve, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.curve, args.output_prefix), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
