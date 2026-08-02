#!/usr/bin/env bash
# Run the registered 1/5/10 percent target-scene adaptations sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
GPU="${SLM_M11_FEWSHOT_GPU:-3}"
SEED="${SLM_M11_FEWSHOT_SEED:-20260801}"
SOURCE="neural_instance_culling/model/out/pvs_m11_directional_yaw20_rvl_strong_v2_full40_hkust_fov66_seed20260801/best.pt"
SOURCE_DIR="$(dirname "$SOURCE")"
DATASET="neural_instance_culling/dataset/out/pose_csr_metropolis_directional_yaw20_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_metropolis_directional_yaw20_fov66_v1"
RUNTIME_META="ifcbench_fantasy_metropolis_source/assets/runtimeVisibilityMeta.json"
GLB_INDEX="ifcbench_fantasy_metropolis_source/assets/glbIndex.json"
GLB_ROOT="ifcbench_fantasy_metropolis_source/assets"
MODEL_ROOT="neural_instance_culling/model/out"
BENCH_ROOT="neural_instance_culling/benchmark/out"
RUN_SUFFIX="${SLM_M11_FEWSHOT_SUFFIX:-retry3}"
POSE_SET_BATCH_SIZE="${SLM_M11_FEWSHOT_POSE_SET_BATCH_SIZE:-1}"
FEATURE_EXPORT_BATCH_SIZE="${SLM_M11_FEWSHOT_FEATURE_EXPORT_BATCH_SIZE:-128}"
USE_AMP="${SLM_M11_FEWSHOT_AMP:-1}"
CUDA_ALLOC_CONF="${SLM_M11_FEWSHOT_CUDA_ALLOC_CONF:-expandable_segments:True}"
EVAL_POSES_PER_BATCH="${SLM_M11_FEWSHOT_EVAL_POSES_PER_BATCH:-1}"
FEWSHOT_LABELS="${SLM_M11_FEWSHOT_LABELS:-1pct,5pct,10pct}"

TRAIN_MEMORY_ARGS=(
  --pose-set-batch-size "$POSE_SET_BATCH_SIZE"
  --feature-export-batch-size "$FEATURE_EXPORT_BATCH_SIZE"
)
if [[ "$USE_AMP" == "1" ]]; then
  TRAIN_MEMORY_ARGS+=(--amp)
fi

# The directional Metropolis dataset is built from the source scene's
# 41,298 component records. The 3,669-row cache belongs to the separate
# instanced-v2 runtime and must not be mixed into this experiment.
GLB_POINTS="neural_instance_culling/dataset/out/ifcbench_fantasy_metropolis_instanced_v2_glb_points_v3.bin"

while [[ ! -f "$SOURCE_DIR/calibration_ready_summary.json" ]]; do
  printf '[m11-fewshot] waiting for completed source calibration: %s\n' "$SOURCE_DIR/calibration_ready_summary.json"
  sleep 60
done
[[ -f "$SOURCE" ]]

run_fraction() {
  local label="$1"
  local fraction="$2"
  local experiment="pvs_m11_fewshot_${label}_metropolis_yaw20_rvl_strong_v2_full40_seed${SEED}_${RUN_SUFFIX}"
  local output_dir="$MODEL_ROOT/$experiment"
  local stdout="$output_dir/train_stdout.log"
  local stderr="$output_dir/train_stderr.log"
  local manifest="$BENCH_ROOT/m11_${experiment}_frozen_manifest.json"
  local test_out="$BENCH_ROOT/m11_${experiment}_frozen_test"

  if [[ -f "$test_out/summary.json" ]]; then
    printf '[m11-fewshot] keeping completed %s\n' "$experiment"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    printf '[m11-fewshot] refusing to reuse incomplete output directory: %s\n' "$output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  printf '[m11-fewshot] training %s (fraction=%s) on GPU %s\n' "$experiment" "$fraction" "$GPU"
  CUDA_VISIBLE_DEVICES="$GPU" PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/model/train_directional_occlusion_proxy_encoder.py \
    --dataset-dir "$DATASET" \
    --evidence-dir "$EVIDENCE" \
    --glb-points "$GLB_POINTS" \
    --runtime-meta "$RUNTIME_META" \
    --glb-index "$GLB_INDEX" \
    --glb-root "$GLB_ROOT" \
    --output-dir "$output_dir" \
    --experiment-name "$experiment" \
    --epochs 40 --steps-per-epoch 900 "${TRAIN_MEMORY_ARGS[@]}" --eval-every 2 \
    --loss-profile rvl_strong_v2 \
    --init-checkpoint "$SOURCE" --allow-scene-transfer --train-pose-fraction "$fraction" \
    --target-weighted-recall 0.99 --calibration-point-floor 0.9925 \
    --calibration-lcb-floor 0.99 --calibration-bootstrap-replicates 10000 \
    --seed "$SEED" --device cuda --skip-final-test \
    >"$stdout" 2>"$stderr"

  printf '[m11-fewshot] calibrating and evaluating %s\n' "$experiment"
  if CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_frozen_test.py prepare \
    --model-checkpoint "$experiment=$output_dir/best.pt" \
    --runtime-features "$experiment=$output_dir/instance_runtime_features_fp16.bin" \
    --dataset-dir "$DATASET" --target-weighted-recall 0.99 --output "$manifest" \
    >"$BENCH_ROOT/${experiment}_prepare_stdout.log" 2>"$BENCH_ROOT/${experiment}_prepare_stderr.log"; then
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 \
      conda run --no-capture-output -n "$ENV_NAME" python -u \
      neural_instance_culling/benchmark/evaluate_frozen_test.py evaluate \
      --models "$experiment" --manifest "$manifest" --dataset-dir "$DATASET" \
      --runtime-meta "$RUNTIME_META" --glb-index "$GLB_INDEX" --glb-root "$GLB_ROOT" \
      --output-dir "$test_out" --poses-per-batch "$EVAL_POSES_PER_BATCH" --device cuda \
      >"$BENCH_ROOT/${experiment}_test_stdout.log" 2>"$BENCH_ROOT/${experiment}_test_stderr.log"
  else
    printf '[m11-fewshot] no safe calibration workpoint for %s; continuing\n' "$experiment"
  fi
}

IFS=',' read -r -a requested_labels <<< "$FEWSHOT_LABELS"
for label in "${requested_labels[@]}"; do
  case "$label" in
    1pct) run_fraction 1pct 0.01 ;;
    5pct) run_fraction 5pct 0.05 ;;
    10pct) run_fraction 10pct 0.10 ;;
    *)
      printf '[m11-fewshot] unsupported label: %s\n' "$label" >&2
      exit 2
      ;;
  esac
done
printf '[m11-fewshot] registered adaptations completed\n'
