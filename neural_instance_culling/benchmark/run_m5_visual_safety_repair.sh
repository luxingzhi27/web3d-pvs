#!/usr/bin/env bash
# Queue the pre-registered M5 visual-safety repair variants.
# The queue waits for the currently running formal M4/M11 sessions so that
# the repair runs do not change their GPU scheduling or resource measurements.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M5_POLL_SECONDS:-60}"
GPU_LIST="${SLM_M5_GPUS:-0 1 2 3}"
SEEDS="${SLM_M5_SEEDS:-20260801 20260802 20260803}"

DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1"
GLB_POINTS="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"
GLB_INDEX="hkust-v3/assets/glbIndex.json"
GLB_ROOT="hkust-v3/assets"
OUTPUT_ROOT="neural_instance_culling/model/out"

wait_for_session_exit() {
  local session="$1"
  while tmux has-session -t "$session" 2>/dev/null; do
    printf '[m5-queue] waiting for session %s\n' "$session"
    sleep "$POLL_SECONDS"
  done
}

wait_for_formal_jobs() {
  local session
  # The retry suffix is part of the current registered M11 run. Waiting on
  # the old unsuffixed name would silently release GPU 3 while M11 is active.
  for session in formal_m4_matrix formal_m4_eval m11_hkust_directional m11_hkust_eval m11_cross_scene_transfer m11_metropolis_fewshot_retry3 m11_metropolis_fewshot_5_10; do
    wait_for_session_exit "$session"
  done
}

variant_args() {
  case "$1" in
    m5_visual_mass_linear)
      printf '%s\n' --visual-safety-loss-weight 0.20 --visual-safety-weight-power 1.0 --visual-safety-tail-k 0 --visual-safety-tail-margin 1.0 --visual-safety-tail-weight 0.0
      ;;
    m5_visual_mass_tail)
      printf '%s\n' --visual-safety-loss-weight 0.20 --visual-safety-weight-power 1.0 --visual-safety-tail-k 8 --visual-safety-tail-margin 1.0 --visual-safety-tail-weight 0.5
      ;;
    m5_visual_mass_soft)
      printf '%s\n' --visual-safety-loss-weight 0.20 --visual-safety-weight-power 0.5 --visual-safety-tail-k 8 --visual-safety-tail-margin 1.0 --visual-safety-tail-weight 0.5
      ;;
    m5_control_log1p)
      printf '%s\n' --visual-safety-loss-weight 0.0 --visual-safety-weight-power 1.0 --visual-safety-tail-k 8 --visual-safety-tail-margin 1.0 --visual-safety-tail-weight 0.5
      ;;
    *)
      printf '[m5-queue] unknown variant: %s\n' "$1" >&2
      return 1
      ;;
  esac
}

run_variant() {
  local gpu="$1"
  local seed="$2"
  local variant="$3"
  local experiment="${variant}_rvl_strong_v2_hkust_spatial_fov66_seed${seed}_full40"
  local output_dir="$OUTPUT_ROOT/$experiment"
  local ready="$output_dir/calibration_ready_summary.json"
  local stdout="$output_dir/train_stdout.log"
  local stderr="$output_dir/train_stderr.log"
  local visual_args=()
  while IFS= read -r value; do
    visual_args+=("$value")
  done < <(variant_args "$variant")

  if [[ -f "$ready" ]]; then
    printf '[m5-queue] already complete, keeping %s\n' "$experiment"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    printf '[m5-queue] refusing to reuse incomplete output directory: %s\n' "$output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  printf '[m5-queue] starting %s on GPU %s\n' "$experiment" "$gpu"
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
    "${visual_args[@]}" \
    --target-weighted-recall 0.99 --calibration-point-floor 0.9925 --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 --seed "$seed" --device cuda --skip-final-test \
    >"$stdout" 2>"$stderr"
  printf '[m5-queue] completed %s\n' "$experiment"
}

wait_for_formal_jobs

variants=(m5_visual_mass_linear m5_visual_mass_tail m5_visual_mass_soft m5_control_log1p)
gpu_array=($GPU_LIST)
if (( ${#gpu_array[@]} == 0 )); then
  printf '[m5-queue] SLM_M5_GPUS must contain at least one GPU id\n' >&2
  exit 1
fi

jobs=()
index=0
for seed in $SEEDS; do
  for variant in "${variants[@]}"; do
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
printf '[m5-queue] pre-registered visual-safety repair matrix completed\n'
