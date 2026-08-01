#!/usr/bin/env bash
# Evaluate all completed M4 checkpoints on the same validation split, then
# summarize their paired pose records. This script never reads test.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M4_EVAL_POLL_SECONDS:-60}"
GPU_LIST="${SLM_M4_EVAL_GPUS:-0 1 2}"
MODEL_ROOT="neural_instance_culling/model/out"
BENCH_ROOT="neural_instance_culling/benchmark/out"
DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"

variants=(aabb_ray geometry_ray geometry_context_ray geometry_context_proxy_ray_no_inhibition full)
seeds=(20260801 20260802 20260803)

experiment_name() {
  printf 'pvs_m4_ablation_%s_rvl_strong_v2_hkust_spatial_fov66_seed%s_full40' "$1" "$2"
}

wait_for_matrix() {
  for seed in "${seeds[@]}"; do
    for variant in "${variants[@]}"; do
      local experiment="$(experiment_name "$variant" "$seed")"
      while [[ ! -f "$MODEL_ROOT/$experiment/calibration_ready_summary.json" ]]; do
        printf '[formal-m4-eval] waiting for %s calibration-ready output\n' "$experiment"
        sleep "$POLL_SECONDS"
      done
    done
  done
}

wait_for_m3() {
  local paths=(
    "$BENCH_ROOT/m3_formal_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2_validation/interventions.json"
    "$BENCH_ROOT/m3_formal_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2_validation/interventions.json"
  )
  for path in "${paths[@]}"; do
    while [[ ! -f "$path" ]]; do
      printf '[formal-m4-eval] waiting for formal M3 completion: %s\n' "$path"
      sleep "$POLL_SECONDS"
    done
  done
}

run_one() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local experiment="$(experiment_name "$variant" "$seed")"
  local model_dir="$MODEL_ROOT/$experiment"
  local output_dir="$BENCH_ROOT/m4_formal_baseline_${experiment}_validation"
  local output="$output_dir/interventions.json"
  mkdir -p "$output_dir/logs"
  if [[ -f "$output" ]]; then
    printf '[formal-m4-eval] keeping existing %s\n' "$output"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
    --checkpoint "$model_dir/best.pt" \
    --runtime-features "$model_dir/instance_runtime_features_fp16.bin" \
    --dataset-dir "$DATASET" \
    --runtime-meta "$RUNTIME_META" \
    --split validation --interventions baseline --poses-per-batch 2 \
    --bootstrap-replicates 1 --seed 20260801 --device cuda --output "$output" \
    >"$output_dir/logs/run_stdout.log" 2>"$output_dir/logs/run_stderr.log"
}

wait_for_matrix
wait_for_m3
gpu_array=($GPU_LIST)
if (( ${#gpu_array[@]} == 0 )); then
  printf '[formal-m4-eval] SLM_M4_EVAL_GPUS must contain at least one GPU id\n' >&2
  exit 1
fi
jobs=()
index=0
for seed in "${seeds[@]}"; do
  for variant in "${variants[@]}"; do
    gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
    run_one "$gpu" "$variant" "$seed" &
    jobs+=("$!")
    index=$((index + 1))
    if (( ${#jobs[@]} >= ${#gpu_array[@]} )); then
      for pid in "${jobs[@]}"; do wait "$pid"; done
      jobs=()
    fi
  done
done
for pid in "${jobs[@]}"; do wait "$pid"; done

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
  neural_instance_culling/benchmark/summarize_formal_m4_matrix.py \
  --output "$BENCH_ROOT/m4_formal_matrix_validation_summary.json" \
  --bootstrap-replicates 10000 \
  >"$BENCH_ROOT/m4_formal_matrix_validation_summary.log" 2>&1
printf '[formal-m4-eval] validation matrix summary completed\n'
