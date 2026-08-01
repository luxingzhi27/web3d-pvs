#!/usr/bin/env bash
# Run the registered M7 mainline/ranker comparison after formal checkpoints
# become calibration-ready. This script never opens the test split.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M7_POLL_SECONDS:-60}"
DEVICE="${SLM_M7_DEVICE:-cpu}"

wait_for_ready() {
  local path="$1"
  local label="$2"
  local error_log="${3:-}"
  while [[ ! -f "$path" ]]; do
    if [[ -n "$error_log" && -f "$error_log" ]] && rg -q "Traceback|RuntimeError:|ValueError:" "$error_log"; then
      printf '[formal-m7] %s stopped after an error; see %s\n' "$label" "$error_log" >&2
      return 1
    fi
    printf '[formal-m7] waiting for %s: %s\n' "$label" "$path"
    sleep "$POLL_SECONDS"
  done
}

run_split() {
  local label="$1"
  local scene_name="$2"
  local dataset_dir="$3"
  local runtime_meta="$4"
  local glb_index="$5"
  local glb_root="$6"
  local checkpoint_dir="$7"
  local ranker_checkpoint="$8"
  local split="$9"

  local experiment_name="${checkpoint_dir##*/}"
  local ready="$checkpoint_dir/calibration_ready_summary.json"
  local output_dir="neural_instance_culling/benchmark/out/m7_formal_${scene_name}_${split}_20260801"
  local log_dir="$output_dir/logs"
  wait_for_ready "$ready" "$label calibration-ready checkpoint" "$checkpoint_dir/train_stderr.log"
  if [[ -f "$output_dir/summary.json" ]]; then
    printf '[formal-m7] %s %s already completed; keeping immutable summary\n' "$label" "$split"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    printf '[formal-m7] refusing to reuse incomplete output path: %s\n' "$output_dir" >&2
    return 1
  fi
  mkdir -p "$log_dir"

  local model_spec
  model_spec="${experiment_name}|${checkpoint_dir}/best.pt|${checkpoint_dir}/instance_runtime_features_fp16.bin|${ready}"
  local ranker_spec="m7_independent_ranknet_${scene_name}|${ranker_checkpoint}"
  printf '[formal-m7] starting %s %s on %s\n' "$label" "$split" "$DEVICE"
  CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_visual_utility_metrics.py \
    --models baseline_keep_all,baseline_camera_distance,baseline_aabb_ray \
    --learned-model-spec "$model_spec" \
    --independent-ranker-spec "$ranker_spec" \
    --dataset-dir "$dataset_dir" \
    --runtime-meta "$runtime_meta" \
    --glb-index "$glb_index" \
    --glb-root "$glb_root" \
    --output-dir "$output_dir" \
    --split "$split" \
    --poses-per-batch 4 \
    --max-candidates-per-pose 0 \
    --score-modes visibility-only,current-cascade,visibility-gated,independent-utility \
    --glb-aggregations max,sum,top-k,noisy-or \
    --aggregation-top-k 2 \
    --byte-budgets 1048576,5242880,10485760,20971520 \
    --device "$DEVICE" \
    >"$log_dir/run_stdout.log" 2>"$log_dir/run_stderr.log"
  printf '[formal-m7] completed %s %s: %s\n' "$label" "$split" "$output_dir/summary.json"
}

run_scene() {
  local label="$1"
  local scene_name="$2"
  local dataset_dir="$3"
  local runtime_meta="$4"
  local glb_index="$5"
  local glb_root="$6"
  local checkpoint_dir="$7"
  local ranker_checkpoint="$8"

  run_split "$label" "$scene_name" "$dataset_dir" "$runtime_meta" "$glb_index" "$glb_root" "$checkpoint_dir" "$ranker_checkpoint" validation &
  local validation_pid=$!
  run_split "$label" "$scene_name" "$dataset_dir" "$runtime_meta" "$glb_index" "$glb_root" "$checkpoint_dir" "$ranker_checkpoint" calibration &
  local calibration_pid=$!
  wait "$validation_pid"
  wait "$calibration_pid"
}

HKUST_CHECKPOINT="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2"
METRO_CHECKPOINT="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2"

run_scene "HKUST" "hkust" \
  "neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1" \
  "hkust-v3/assets/runtimeVisibilityMeta.json" \
  "hkust-v3/assets/glbIndex.json" \
  "hkust-v3/assets" \
  "$HKUST_CHECKPOINT" \
  "neural_instance_culling/model/out/m7_independent_ranknet_hkust_spatial_fov66_seed20260801/best.pt"

run_scene "Metropolis" "metropolis" \
  "neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json" \
  "ifcbench_fantasy_metropolis_instanced_v2/assets" \
  "$METRO_CHECKPOINT" \
  "neural_instance_culling/model/out/m7_independent_ranknet_metropolis_spatial_fov66_seed20260801/best.pt"

printf '[formal-m7] all registered validation/calibration comparisons completed\n'
