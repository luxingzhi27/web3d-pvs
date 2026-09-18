"""GCOF-PVS V5 evaluation, calibration, and result aggregation."""

from .calibration import (
    CalibrationSelection,
    calibrate_source_global,
    calibrate_target,
    select_calibration_workpoint,
)
from .contracts import (
    ALL_VARIANTS,
    LOSO_VARIANTS,
    V5Run,
    PoseRecord,
    SceneScores,
    validate_matrix,
)
from .metrics import METRIC_FIELDS, evaluate_scene
from .evaluation import evaluate_loso_matrix, evaluate_shared_matrix
from .summary import summarize_loso, summarize_shared

__all__ = [
    "ALL_VARIANTS",
    "LOSO_VARIANTS",
    "METRIC_FIELDS",
    "CalibrationSelection",
    "PoseRecord",
    "SceneScores",
    "V5Run",
    "calibrate_source_global",
    "calibrate_target",
    "evaluate_loso_matrix",
    "evaluate_scene",
    "evaluate_shared_matrix",
    "select_calibration_workpoint",
    "summarize_loso",
    "summarize_shared",
    "validate_matrix",
]
