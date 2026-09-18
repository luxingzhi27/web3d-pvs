"""GCOF-PVS V5 columnar score export, calibration and evaluation."""

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
    SceneScores,
    validate_matrix,
)
from .evaluation import evaluate_loso_matrix, evaluate_shared_matrix
from .inference import (
    InferenceSceneSpec,
    compile_geometry,
    export_score_bundle,
    infer_split,
    load_checkpoint,
    registered_scene_specs,
)
from .metrics import METRIC_FIELDS, evaluate_scene
from .score_bundle import (
    BUNDLE_SCHEMA,
    ColumnarScoreSidecar,
    ColumnarScoreSidecarWriter,
    FrozenPoseCSR,
    PoseScores,
    SIDECAR_SCHEMA,
)
from .summary import summarize_loso, summarize_shared

__all__ = [
    "ALL_VARIANTS",
    "BUNDLE_SCHEMA",
    "LOSO_VARIANTS",
    "METRIC_FIELDS",
    "SIDECAR_SCHEMA",
    "CalibrationSelection",
    "ColumnarScoreSidecar",
    "ColumnarScoreSidecarWriter",
    "FrozenPoseCSR",
    "InferenceSceneSpec",
    "PoseScores",
    "SceneScores",
    "V5Run",
    "calibrate_source_global",
    "calibrate_target",
    "compile_geometry",
    "evaluate_loso_matrix",
    "evaluate_scene",
    "evaluate_shared_matrix",
    "export_score_bundle",
    "infer_split",
    "load_checkpoint",
    "registered_scene_specs",
    "select_calibration_workpoint",
    "summarize_loso",
    "summarize_shared",
    "validate_matrix",
]
