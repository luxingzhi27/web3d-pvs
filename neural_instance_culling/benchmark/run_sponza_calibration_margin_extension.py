#!/usr/bin/env python3
"""Extend the Sponza calibration-margin scan to 0.999 and 0.9995."""
from __future__ import annotations

try:
    from . import run_sponza_calibration_margin_refinement as base
except ImportError:
    import run_sponza_calibration_margin_refinement as base  # type: ignore


base.EXPERIMENT = "pvs_v4_sponza_calibration_margin_extension_v1"
base.MARGINS = (
    base.Margin("margin999", 0.999),
    base.Margin("margin9995", 0.9995),
)
base.MODEL_ROOT = base.ROOT / "neural_instance_culling/model/out" / base.EXPERIMENT
base.RESULT_ROOT = (
    base.ROOT / "neural_instance_culling/benchmark/out/paper_results" / base.EXPERIMENT
)


if __name__ == "__main__":
    base.main()
