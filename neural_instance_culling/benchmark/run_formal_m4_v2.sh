#!/usr/bin/env bash
# Execute the independent M4-v2 matrix. Existing M4 outputs are read-only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
POLL_SECONDS="${SLM_M4_V2_POLL_SECONDS:-60}"
GPU_LIST="${SLM_M4_V2_GPUS:-0 1 2 3}"
SEEDS="${SLM_M4_V2_SEEDS:-20260801 20260802 20260803}"

DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_v3_spatial_raw_fov66_v1"
GLB_POINTS="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"
GLB_INDEX="hkust-v3/assets/glbIndex.json"
GLB_ROOT="hkust-v3/assets"
MODEL_ROOT="neural_instance_culling/model/out"
BENCH_ROOT="neural_instance_culling/benchmark/out"
V2_ROOT="$BENCH_ROOT/m4_formal_matrix_validation_v2"

wait_for_pid_exit() {
  local pid="$1"
  local label="$2"
  if [[ -z "$pid" ]]; then return 0; fi
  if [[ ! "$pid" =~ ^[0-9]+$ ]]; then
    printf '[m4-v2] invalid PID for %s: %s\n' "$label" "$pid" >&2
    return 1
  fi
  while kill -0 "$pid" 2>/dev/null; do
    printf '[m4-v2] waiting for %s (pid=%s)\n' "$label" "$pid"
    sleep "$POLL_SECONDS"
  done
}

old_experiment() {
  printf 'pvs_m4_ablation_%s_rvl_strong_v2_hkust_spatial_fov66_seed%s_full40' "$1" "$2"
}

a_experiment() {
  printf 'pvs_m4_v2_ablation_geometry_context_ray_no_inhibition_rvl_strong_v2_hkust_spatial_fov66_seed%s_full40' "$1"
}

run_a_seed() {
  local gpu="$1"
  local seed="$2"
  local experiment
  experiment="$(a_experiment "$seed")"
  local output_dir="$MODEL_ROOT/$experiment"
  local ready="$output_dir/calibration_ready_summary.json"
  if [[ -f "$ready" ]]; then
    printf '[m4-v2] keeping completed A checkpoint %s\n' "$experiment"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    printf '[m4-v2] refusing to reuse incomplete A directory: %s\n' "$output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
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
    --runtime-feature-ablation proxy_zero \
    --disable-explicit-inhibition \
    --target-weighted-recall 0.99 \
    --calibration-point-floor 0.9925 \
    --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 \
    --seed "$seed" --device cuda --skip-final-test \
    >"$output_dir/train_stdout.log" 2>"$output_dir/train_stderr.log"
}

run_eval() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local experiment
  local model_dir
  local output_dir
  local output
  local workpoint
  if [[ "$variant" == "geometry_context_ray_no_inhibition" ]]; then
    experiment="$(a_experiment "$seed")"
  else
    experiment="$(old_experiment "$variant" "$seed")"
  fi
  model_dir="$MODEL_ROOT/$experiment"
  output_dir="$V2_ROOT/members/$variant/seed$seed"
  output="$output_dir/interventions.json"
  workpoint="$V2_ROOT/workpoints/${variant}_seed${seed}.json"
  if [[ -f "$output" ]]; then
    printf '[m4-v2] keeping completed validation evaluation %s/%s\n' "$variant" "$seed"
    return 0
  fi
  if [[ ! -f "$workpoint" ]]; then
    printf '[m4-v2] missing calibration workpoint: %s\n' "$workpoint" >&2
    return 1
  fi
  local threshold
  threshold="$(python - "$workpoint" <<'PY'
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
print(format(float(payload["threshold"]), ".17g"))
PY
)"
  mkdir -p "$output_dir/logs"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONDONTWRITEBYTECODE=1 \
    conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_proxy_interventions.py \
    --checkpoint "$model_dir/best.pt" \
    --runtime-features "$model_dir/instance_runtime_features_fp16.bin" \
    --dataset-dir "$DATASET" \
    --runtime-meta "$RUNTIME_META" \
    --split validation --interventions baseline --poses-per-batch 2 \
    --bootstrap-replicates 1 --seed 20260801 --device cuda \
    --threshold "$threshold" --threshold-source "$workpoint" \
    --output "$output" \
    >"$output_dir/logs/run_stdout.log" 2>"$output_dir/logs/run_stderr.log"
}

wait_for_pid_exit "${SLM_M4_V2_WAIT_PID_1:-}" "queued training job 1"
wait_for_pid_exit "${SLM_M4_V2_WAIT_PID_2:-}" "queued training job 2"
wait_for_pid_exit "${SLM_M4_V2_WAIT_PID_3:-}" "queued training job 3"

gpu_array=($GPU_LIST)
if (( ${#gpu_array[@]} < 3 )); then
  printf '[m4-v2] at least three GPU slots are required for the three A seeds\n' >&2
  exit 1
fi

jobs=()
index=0
for seed in $SEEDS; do
  gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  run_a_seed "$gpu" "$seed" &
  jobs+=("$!")
  index=$((index + 1))
done
for pid in "${jobs[@]}"; do wait "$pid"; done

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
  neural_instance_culling/benchmark/select_m4_v2_calibration_workpoints.py \
  --model-root "$MODEL_ROOT" \
  --output "$V2_ROOT/calibration_workpoints.json" \
  >"$V2_ROOT/calibration_workpoints_stdout.log" 2>"$V2_ROOT/calibration_workpoints_stderr.log"

jobs=()
index=0
for seed in $SEEDS; do
  for variant in aabb_ray geometry_ray geometry_context_ray_no_inhibition geometry_context_proxy_ray_no_inhibition geometry_context_ray full; do
    gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
    run_eval "$gpu" "$variant" "$seed" &
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
  neural_instance_culling/benchmark/summarize_formal_m4_matrix_v2.py \
  --benchmark-root "$BENCH_ROOT" \
  --v2-root "$V2_ROOT" \
  --calibration-workpoints "$V2_ROOT/calibration_workpoints.json" \
  --bootstrap-replicates 10000 \
  --output "$V2_ROOT/summary.json" \
  >"$V2_ROOT/summary_stdout.log" 2>"$V2_ROOT/summary_stderr.log"

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
  neural_instance_culling/benchmark/decide_m4_route_v2.py \
  --summary "$V2_ROOT/summary.json" \
  --output "$BENCH_ROOT/m4_formal_route_decision_v2.json" \
  --markdown "$BENCH_ROOT/m4_formal_route_decision_v2.md" \
  --max-bad-cull-delta 0.002 \
  --max-recall-drop 0.01 \
  >"$BENCH_ROOT/m4_formal_route_decision_v2.log" 2>&1

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
  neural_instance_culling/benchmark/validate_m4_formal_matrix_v2.py \
  --summary "$V2_ROOT/summary.json" \
  --route "$BENCH_ROOT/m4_formal_route_decision_v2.json" \
  >"$V2_ROOT/schema_validation.log" 2>&1

PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
  neural_instance_culling/benchmark/write_m4_formal_matrix_v2_report.py \
  --summary "$V2_ROOT/summary.json" \
  --route "$BENCH_ROOT/m4_formal_route_decision_v2.json" \
  --output "docs/evaluation/m4_formal_matrix_validation_v2_2026-08-03.md" \
  >"$V2_ROOT/report_stdout.log" 2>"$V2_ROOT/report_stderr.log"

printf '[m4-v2] complete: %s\n' "$V2_ROOT/summary.json"
