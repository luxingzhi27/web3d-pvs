#!/usr/bin/env bash
# Run the registered M3 inference-time interventions after M0 has sealed the
# calibration threshold and one-shot test.  This script never opens the test
# split itself; evaluate_proxy_interventions.py reads the pre-test calibration
# record and evaluates the complete validation split with paired candidates.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M3_POLL_SECONDS:-60}"

wait_for_path() {
  local path="$1"
  local label="$2"
  local error_log="${3:-}"
  while [[ ! -e "$path" ]]; do
    if [[ -n "$error_log" && -f "$error_log" ]] && rg -q "Traceback|RuntimeError:|ValueError:" "$error_log"; then
      printf '[formal-m3] %s stopped after an error; see %s\n' "$label" "$error_log" >&2
      return 1
    fi
    printf '[formal-m3] waiting for %s: %s\n' "$label" "$path"
    sleep "$POLL_SECONDS"
  done
}

run_scene() {
  local label="$1"
  local visible_gpu="$2"
  local dataset_dir="$3"
  local runtime_meta="$4"
  local checkpoint_dir="$5"
  local experiment_name="$6"
  local m0_dir="$7"

  local ready="$checkpoint_dir/calibration_ready_summary.json"
  local m0_summary="$m0_dir/test/summary.json"
  local output_dir="neural_instance_culling/benchmark/out/m3_formal_${experiment_name}_validation"
  local output="$output_dir/interventions.json"
  local log_dir="$output_dir/logs"
  mkdir -p "$log_dir"

  wait_for_path "$ready" "$label calibration-ready checkpoint" "$checkpoint_dir/train_stderr.log"
  wait_for_path "$m0_summary" "$label M0 one-shot test" "$m0_dir/logs/evaluate_stderr.log"
  if [[ -e "$output" ]]; then
    printf '[formal-m3] refusing to overwrite existing output for %s: %s\n' "$label" "$output" >&2
    return 1
  fi

  CUDA_VISIBLE_DEVICES="$visible_gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
    --checkpoint "$checkpoint_dir/best.pt" \
    --runtime-features "$checkpoint_dir/instance_runtime_features_fp16.bin" \
    --dataset-dir "$dataset_dir" \
    --runtime-meta "$runtime_meta" \
    --split validation \
    --poses-per-batch 2 \
    --bootstrap-replicates 10000 \
    --seed 20260801 \
    --device cuda \
    --output "$output" \
    >"$log_dir/run_stdout.log" 2>"$log_dir/run_stderr.log"

  printf '[formal-m3] %s completed: %s\n' "$label" "$output"
}

run_scene "HKUST" "0" \
  "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1" \
  "hkust-v3/assets/runtimeVisibilityMeta.json" \
  "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2" \
  "pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2" \
  "neural_instance_culling/benchmark/out/m0_frozen_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2" &
hkust_pid=$!

run_scene "Metropolis" "2" \
  "neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json" \
  "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2" \
  "pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2" \
  "neural_instance_culling/benchmark/out/m0_frozen_pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2" &
metropolis_pid=$!

wait "$hkust_pid"
wait "$metropolis_pid"
printf '[formal-m3] all registered scenes completed\n'
