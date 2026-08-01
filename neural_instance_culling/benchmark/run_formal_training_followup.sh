#!/usr/bin/env bash
# Wait for the registered full-training jobs, then perform the strict M0
# calibration-manifest and one-shot test protocol for each scene.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_FOLLOWUP_POLL_SECONDS:-60}"

wait_for_ready() {
  local output_dir="$1"
  local label="$2"
  local ready="$output_dir/calibration_ready_summary.json"
  local stderr_log="$output_dir/train_stderr.log"
  while [[ ! -f "$ready" ]]; do
    if [[ -f "$stderr_log" ]] && rg -q "Traceback|RuntimeError:|ValueError:" "$stderr_log"; then
      printf '[formal-followup] %s stopped after a training error; see %s\n' "$label" "$stderr_log" >&2
      return 1
    fi
    printf '[formal-followup] waiting for %s: %s\n' "$label" "$ready"
    sleep "$POLL_SECONDS"
  done
}

run_scene() {
  local label="$1"
  local visible_gpu="$2"
  local dataset_dir="$3"
  local runtime_meta="$4"
  local glb_index="$5"
  local glb_root="$6"
  local experiment_name="$7"
  local output_dir="$8"

  local protocol_dir="neural_instance_culling/benchmark/out/m0_frozen_${experiment_name}"
  local manifest="$protocol_dir/frozen_manifest.json"
  local test_dir="$protocol_dir/test"
  local log_dir="$protocol_dir/logs"
  mkdir -p "$log_dir"

  wait_for_ready "$output_dir" "$label"
  if [[ -f "$manifest" && -f "$test_dir/summary.json" ]]; then
    printf '[formal-followup] %s already completed; keeping immutable output: %s\n' "$label" "$test_dir/summary.json"
    return 0
  fi
  if [[ -e "$manifest" || -e "$test_dir" ]]; then
    printf '[formal-followup] refusing to reuse incomplete frozen output for %s: %s\n' "$label" "$protocol_dir" >&2
    return 1
  fi

  local model_name="${experiment_name}_formal"
  CUDA_VISIBLE_DEVICES="$visible_gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_frozen_test.py prepare \
    --model-checkpoint "${model_name}=${output_dir}/best.pt" \
    --runtime-features "${model_name}=${output_dir}/instance_runtime_features_fp16.bin" \
    --dataset-dir "$dataset_dir" \
    --output "$manifest" \
    >"$log_dir/prepare_stdout.log" 2>"$log_dir/prepare_stderr.log"

  CUDA_VISIBLE_DEVICES="$visible_gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_frozen_test.py evaluate \
    --models "$model_name" \
    --manifest "$manifest" \
    --dataset-dir "$dataset_dir" \
    --runtime-meta "$runtime_meta" \
    --glb-index "$glb_index" \
    --glb-root "$glb_root" \
    --output-dir "$test_dir" \
    --device cuda \
    >"$log_dir/evaluate_stdout.log" 2>"$log_dir/evaluate_stderr.log"

  printf '[formal-followup] %s frozen test completed: %s\n' "$label" "$test_dir/summary.json"
}

run_scene "HKUST" "0" \
  "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1" \
  "hkust-v3/assets/runtimeVisibilityMeta.json" \
  "hkust-v3/assets/glbIndex.json" \
  "hkust-v3/assets" \
  "pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2" \
  "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2" &
hkust_pid=$!

run_scene "Metropolis" "3" \
  "neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets" \
  "pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2" \
  "neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2" &
metropolis_pid=$!

status=0
wait "$hkust_pid" || status=1
wait "$metropolis_pid" || status=1
if (( status != 0 )); then
  printf '[formal-followup] one or more registered scenes failed; inspect per-scene logs\n' >&2
  exit "$status"
fi
printf '[formal-followup] all registered scenes completed\n'
