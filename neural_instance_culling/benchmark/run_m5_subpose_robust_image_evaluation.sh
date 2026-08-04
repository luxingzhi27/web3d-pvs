#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# This is a separate M5-v2 output. The shared evaluator's default four-variant
# matrix remains unchanged; this entry waits only for the three registered
# subpose-robust checkpoints and never reads the test split. The source
# training processes were started before the ordinary pose-recall calibration
# floor was added, so this wrapper first creates independent strict-calibration
# bundles with the current evaluator before launching image evaluation.
VARIANT="pvs_m5_subpose_robust_v1_hkust_spatial_fov66"
SEEDS=(20260801 20260802 20260803)
GPUS=(${SLM_M5_SUBPOSE_GPUS:-0 1 2})
MODEL_ROOT="neural_instance_culling/model/out"
DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_spatial_fov66_v1"
GLB_POINTS="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"
GLB_INDEX="hkust-v3/assets/glbIndex.json"
GLB_ROOT="hkust-v3/assets"
ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"

wait_for_training() {
  while true; do
    local missing=0
    for seed in "${SEEDS[@]}"; do
      local source_dir="$MODEL_ROOT/${VARIANT}_seed${seed}_full40"
      if [[ -f "$source_dir/best.pt" && -f "$source_dir/calibration_ready_summary.json" ]]; then
        continue
      fi
      missing=$((missing + 1))
      local stderr_log="$source_dir/train_stderr.log"
      if [[ -f "$stderr_log" ]] && rg -q "Traceback|RuntimeError:|ValueError:|CUDA out of memory|non-finite" "$stderr_log"; then
        echo "[m5-image] source training failed for seed $seed; see $stderr_log" >&2
        return 1
      fi
      # A missing ready summary with no live training process is an error,
      # not a reason to wait forever.  The source command contains the
      # immutable output directory, so this check does not match this wrapper.
      if [[ -f "$source_dir/train_stdout.log" ]] && ! pgrep -f -- "--output-dir $source_dir" >/dev/null 2>&1; then
        echo "[m5-image] source training exited without calibration summary for seed $seed: $source_dir" >&2
        return 1
      fi
    done
    (( missing == 0 )) && return 0
    echo "[m5-image] waiting for $missing/${#SEEDS[@]} source checkpoints"
    sleep 60
  done
}

strict_export_one() {
  local gpu="$1"
  local seed="$2"
  local source_dir="$MODEL_ROOT/${VARIANT}_seed${seed}_full40"
  local strict_name="${VARIANT}_seed${seed}_full40_strict_calibration"
  local strict_dir="$MODEL_ROOT/$strict_name"
  local ready="$strict_dir/calibration_ready_summary.json"

  if [[ -f "$ready" ]]; then
    echo "[m5-image] keeping completed strict calibration $strict_name"
    return 0
  fi
  if [[ -e "$strict_dir" ]]; then
    echo "[m5-image] refusing to reuse incomplete strict output directory: $strict_dir" >&2
    return 1
  fi
  [[ -f "$source_dir/best.pt" ]] || { echo "missing source checkpoint: $source_dir/best.pt" >&2; return 1; }
  mkdir -p "$strict_dir"
  echo "[m5-image] strict calibration export $strict_name on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
    --dataset-dir "$DATASET" \
    --evidence-dir "$EVIDENCE" \
    --glb-points "$GLB_POINTS" \
    --runtime-meta "$RUNTIME_META" \
    --glb-index "$GLB_INDEX" \
    --glb-root "$GLB_ROOT" \
    --output-dir "$strict_dir" \
    --experiment-name "$strict_name" \
    --pose-set-batch-size 2 \
    --loss-profile m5_subpose_robust_v1 \
    --target-weighted-recall 0.99 \
    --calibration-pose-recall-floor 0.95 \
    --calibration-point-floor 0.9925 \
    --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 \
    --seed "$seed" --device cuda \
    --export-eval-checkpoint "$source_dir/best.pt" \
    --checkpoint-alias best.pt --skip-final-test \
    >"$strict_dir/strict_calibration_stdout.log" \
    2>"$strict_dir/strict_calibration_stderr.log"
  echo "[m5-image] completed strict calibration $strict_name"
}

validate_strict_one() {
  local seed="$1"
  local strict_dir="$MODEL_ROOT/${VARIANT}_seed${seed}_full40_strict_calibration"
  conda run --no-capture-output -n "$ENV_NAME" python -c '
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selected = payload.get("calibration", {}).get("selected") or {}
checks = [
    payload.get("protocol") == "calibration_ready_pre_test",
    int(payload.get("testEvaluationCount", -1)) == 0,
    payload.get("calibration", {}).get("selectionStatus") == "safe",
    float(selected.get("pose_recall", -1.0)) >= 0.95,
    float(selected.get("pose_weighted_recall", -1.0)) >= 0.9925,
    float(selected.get("weighted_recall_lower_confidence_bound", -1.0)) > 0.99,
]
if not all(checks):
    raise SystemExit("strict calibration summary violates a registered safety floor")
print(json.dumps({
    "threshold": payload.get("frozenThreshold"),
    "poseRecall": selected.get("pose_recall"),
    "weightedRecall": selected.get("pose_weighted_recall"),
    "weightedRecallLcb": selected.get("weighted_recall_lower_confidence_bound"),
}, ensure_ascii=False))
' "$strict_dir/calibration_ready_summary.json"
}

wait_for_training

jobs=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  strict_export_one "$gpu" "$seed" &
  jobs+=("$!")
done
for pid in "${jobs[@]}"; do wait "$pid"; done
for seed in "${SEEDS[@]}"; do validate_strict_one "$seed"; done

OUTPUT_NAME="m5_subpose_robust_image_dense_hw_20260804"
SCHEMA_ROOT="neural_instance_culling/benchmark/out/${OUTPUT_NAME}_manifests"
OUTPUT_DIR="neural_instance_culling/benchmark/out/${OUTPUT_NAME}"
EVIDENCE_DIR="neural_instance_culling/benchmark/out/.${OUTPUT_NAME}_hardware_evidence"
EVIDENCE_TAG="$(date +%Y%m%d_%H%M%S)"
if [[ -e "$EVIDENCE_DIR" ]]; then
  echo "refusing to reuse incomplete hardware evidence directory: $EVIDENCE_DIR" >&2
  exit 1
fi
mkdir -p "$EVIDENCE_DIR"
PMON_PID=""
cleanup_pmon() {
  if [[ -n "$PMON_PID" ]] && kill -0 "$PMON_PID" 2>/dev/null; then
    kill "$PMON_PID" 2>/dev/null || true
    wait "$PMON_PID" 2>/dev/null || true
  fi
}
trap cleanup_pmon EXIT
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv \
  >"$EVIDENCE_DIR/nvidia_smi_snapshot_${EVIDENCE_TAG}.csv"
nvidia-smi pmon -s um -d 5 >"$EVIDENCE_DIR/nvidia_smi_pmon_snapshot_${EVIDENCE_TAG}.txt" 2>&1 &
PMON_PID=$!

conda run --no-capture-output -n "$ENV_NAME" \
  python -u neural_instance_culling/benchmark/run_m5_visual_safety_image_evaluation.py \
  --variants "$VARIANT" \
  --experiment-template '{variant}_seed{seed}_full40_strict_calibration' \
  --minimum-pose-recall 0.95 \
  --subposes-per-viewcell 0 \
  --chunk-samples 512 \
  --max-chunk-manifest-bytes 400000000 \
  --output-name "$OUTPUT_NAME" \
  --schema-root "$SCHEMA_ROOT"

nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu --format=csv \
  >>"$EVIDENCE_DIR/nvidia_smi_snapshot_${EVIDENCE_TAG}.csv"
cleanup_pmon
PMON_PID=""
mkdir -p "$OUTPUT_DIR"
mv "$EVIDENCE_DIR"/nvidia_smi_snapshot_*.csv "$OUTPUT_DIR/"
mv "$EVIDENCE_DIR"/nvidia_smi_pmon_snapshot_*.txt "$OUTPUT_DIR/"
rmdir "$EVIDENCE_DIR"

conda run --no-capture-output -n "$ENV_NAME" \
  python -u neural_instance_culling/benchmark/summarize_m5_subpose_robust_image.py \
  --image-output "neural_instance_culling/benchmark/out/$OUTPUT_NAME" \
  --model-root "$MODEL_ROOT" \
  --variant "$VARIANT" \
  --report "docs/evaluation/m5_subpose_robust_image_dense_hw_2026-08-04.md"
