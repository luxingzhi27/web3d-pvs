#!/usr/bin/env bash
# Queue the pre-registered HKUST M4 input/inhibition ablations. The first
# AABB+ray seed is already registered separately; this script waits for the
# three active training jobs to finish before using GPUs 0, 1, and 2.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M4_POLL_SECONDS:-60}"
GPU_LIST="${SLM_M4_GPUS:-0 1 2}"
SEEDS="${SLM_M4_SEEDS:-20260801 20260802 20260803}"

DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1"
GLB_POINTS="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"
GLB_INDEX="hkust-v3/assets/glbIndex.json"
GLB_ROOT="hkust-v3/assets"
OUTPUT_ROOT="neural_instance_culling/model/out"

wait_for_optional_pid_exit() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]]; then
    return 0
  fi
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    printf '[formal-m4] invalid PID for %s: %s\n' "$label" "$pid" >&2
    return 1
  fi
  while kill -0 "$pid" 2>/dev/null; do
    printf '[formal-m4] waiting for %s (pid=%s)\n' "$label" "$pid"
    sleep "$POLL_SECONDS"
  done
}

wait_for_existing_aabb() {
  local path="$OUTPUT_ROOT/pvs_m4_ablation_aabb_ray_rvl_strong_v2_hkust_spatial_fov66_seed20260801_full40/calibration_ready_summary.json"
  while [[ ! -f "$path" ]]; do
    printf '[formal-m4] waiting for the registered AABB+ray seed: %s\n' "$path"
    sleep "$POLL_SECONDS"
  done
}

variant_ablation() {
  case "$1" in
    aabb_ray) printf '%s' 'geo_context_proxy_zero' ;;
    geometry_ray) printf '%s' 'context_proxy_zero' ;;
    geometry_context_ray) printf '%s' 'proxy_zero' ;;
    geometry_context_proxy_ray_no_inhibition) printf '%s' 'none' ;;
    full) printf '%s' 'none' ;;
    *) printf '[formal-m4] unknown variant: %s\n' "$1" >&2; return 1 ;;
  esac
}

run_variant() {
  local gpu="$1"
  local seed="$2"
  local variant="$3"
  local ablation="$(variant_ablation "$variant")"
  local experiment="pvs_m4_ablation_${variant}_rvl_strong_v2_hkust_spatial_fov66_seed${seed}_full40"
  local output_dir="$OUTPUT_ROOT/$experiment"
  local ready="$output_dir/calibration_ready_summary.json"
  local stdout="$output_dir/train_stdout.log"
  local stderr="$output_dir/train_stderr.log"

  if [[ -f "$ready" ]]; then
    printf '[formal-m4] already complete, keeping %s\n' "$experiment"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    printf '[formal-m4] refusing to reuse incomplete output directory: %s\n' "$output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  printf '[formal-m4] starting %s on GPU %s\n' "$experiment" "$gpu"
  local extra=()
  if [[ "$variant" == "geometry_context_proxy_ray_no_inhibition" ]]; then
    extra+=(--disable-explicit-inhibition)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
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
    --epochs 40 --steps-per-epoch 900 --pose-set-batch-size 2 --eval-every 2 \
    --loss-profile rvl_strong_v2 \
    --runtime-feature-ablation "$ablation" \
    "${extra[@]}" \
    --target-weighted-recall 0.99 --calibration-point-floor 0.9925 --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 --seed "$seed" --device cuda --skip-final-test \
    >"$stdout" 2>"$stderr"
  printf '[formal-m4] completed %s\n' "$experiment"
}

# These are intentionally opt-in.  Hard-coding PIDs from a previous machine
# makes the queue non-reproducible and can accidentally wait on an unrelated
# process after PID reuse.  The registered AABB+ray output remains a required
# artifact gate below; external jobs may be supplied only by the launcher that
# knows their current PIDs.
wait_for_optional_pid_exit "${SLM_M4_HKUST_MAINLINE_PID:-}" "HKUST mainline"
wait_for_optional_pid_exit "${SLM_M4_METROPOLIS_MAINLINE_PID:-}" "Metropolis mainline"
wait_for_optional_pid_exit "${SLM_M4_AABB_RAY_PID:-}" "registered AABB+ray ablation"
wait_for_existing_aabb

variants=(aabb_ray geometry_ray geometry_context_ray geometry_context_proxy_ray_no_inhibition full)
jobs=()
index=0
for seed in $SEEDS; do
  for variant in "${variants[@]}"; do
    if [[ "$seed" == "20260801" && "$variant" == "aabb_ray" ]]; then
      continue
    fi
    gpu_array=($GPU_LIST)
    gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
    run_variant "$gpu" "$seed" "$variant" &
    jobs+=("$!")
    index=$((index + 1))
    if (( ${#jobs[@]} >= ${#gpu_array[@]} )); then
      for pid in "${jobs[@]}"; do wait "$pid"; done
      jobs=()
    fi
  done
done
for pid in "${jobs[@]}"; do wait "$pid"; done
printf '[formal-m4] registered HKUST matrix completed\n'
