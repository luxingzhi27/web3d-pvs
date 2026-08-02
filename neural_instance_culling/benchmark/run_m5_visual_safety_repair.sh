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
BENCHMARK_ROOT="neural_instance_culling/benchmark/out"
M4_SUMMARY="$BENCHMARK_ROOT/m4_formal_matrix_validation_summary.json"
M4_ROUTE_DECISION="$BENCHMARK_ROOT/m4_formal_route_decision.json"
M11_FEWSHOT_SESSION="${SLM_M11_FEWSHOT_SESSION:-m11_metropolis_fewshot_5_10}"
M11_FEWSHOT_LOG="$BENCHMARK_ROOT/m11_metropolis_fewshot_5_10_queue.log"
M11_ONE_PERCENT_SUMMARY="${SLM_M11_ONE_PERCENT_SUMMARY:-$BENCHMARK_ROOT/m11_formal_metropolis_fewshot_1pct_frozen_test_20260802_protocolfix/summary.json}"

# Filled after the validation-only M4 route decision.  Route B uses the
# registered geometry+context+ray reference by zeroing only the directional
# proxy input; it does not silently change the model name or candidate data.
M5_RUNTIME_FEATURE_ABLATION=""
M5_ROUTE=""

wait_for_m4_route() {
  while true; do
    if [[ -f "$M4_SUMMARY" && -f "$M4_ROUTE_DECISION" ]]; then
      local route
      route="$(python - "$M4_SUMMARY" "$M4_ROUTE_DECISION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1]).resolve()
decision_path = Path(sys.argv[2]).resolve()
summary = json.loads(summary_path.read_text(encoding="utf-8"))
decision = json.loads(decision_path.read_text(encoding="utf-8"))
if summary.get("status") != "validation_summary_only; formal test not read":
    raise SystemExit("M4 summary is not validation-only")
if decision.get("status") != "route_selected_from_validation_only" or decision.get("testRead") is not False:
    raise SystemExit("M4 route decision is not validation-only")
if Path(str(decision.get("sourceSummary", ""))).resolve() != summary_path:
    raise SystemExit("M4 route decision points to a different summary")
expected = str(decision.get("sourceSummarySha256", ""))
actual = hashlib.sha256(summary_path.read_bytes()).hexdigest()
if not expected or expected != actual:
    raise SystemExit("M4 route decision summary hash does not match")
route = decision.get("route")
if route not in {"route_a_directional_proxy", "route_b_system"}:
    raise SystemExit(f"unsupported M4 route: {route!r}")
print(route)
PY
      )" || true
      if [[ "$route" == "route_a_directional_proxy" ]]; then
        M5_ROUTE="$route"
        M5_RUNTIME_FEATURE_ABLATION="none"
        printf '[m5-queue] M4 route A accepted; using full registered architecture\n'
        return 0
      fi
      if [[ "$route" == "route_b_system" ]]; then
        M5_ROUTE="$route"
        M5_RUNTIME_FEATURE_ABLATION="proxy_zero"
        printf '[m5-queue] M4 route B accepted; using geometry+context+ray reference\n'
        return 0
      fi
      printf '[m5-queue] M4 artifacts exist but route/hash validation is not ready\n'
    else
      printf '[m5-queue] waiting for validation-only M4 summary and route decision\n'
    fi
    sleep "$POLL_SECONDS"
  done
}

wait_for_m11_fewshot() {
  # The active 5%/10% runner is in m11_metropolis_fewshot_5_10. A separate
  # watcher session named m11_metropolis_fewshot_retry3 is not a completion
  # signal and is intentionally ignored here.
  while tmux has-session -t "$M11_FEWSHOT_SESSION" 2>/dev/null || \
      pgrep -f 'train_directional_occlusion_proxy_encoder.py.*pvs_m11_fewshot_(5pct|10pct)' >/dev/null 2>&1; do
    printf '[m5-queue] waiting for active M11 few-shot 5%%/10%% run\n'
    sleep "$POLL_SECONDS"
  done
  if [[ ! -f "$M11_FEWSHOT_LOG" ]] || ! rg -q '\[m11-fewshot\] registered adaptations completed' "$M11_FEWSHOT_LOG"; then
    printf '[m5-queue] M11 few-shot queue exited without its completion marker\n' >&2
    return 1
  fi
  if [[ ! -f "$M11_ONE_PERCENT_SUMMARY" ]]; then
    printf '[m5-queue] missing audited 1%% M11 frozen-test summary: %s\n' "$M11_ONE_PERCENT_SUMMARY" >&2
    return 1
  fi
  printf '[m5-queue] M11 few-shot queue completed with audited 1%% summary\n'
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
  cat > "$output_dir/m5_route.json" <<EOF
{
  "m4Route": "$M5_ROUTE",
  "runtimeFeatureAblation": "$M5_RUNTIME_FEATURE_ABLATION",
  "sourceSummary": "$M4_SUMMARY",
  "sourceRouteDecision": "$M4_ROUTE_DECISION",
  "testReadBeforeTraining": false
}
EOF
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
    --runtime-feature-ablation "$M5_RUNTIME_FEATURE_ABLATION" \
    "${visual_args[@]}" \
    --target-weighted-recall 0.99 --calibration-point-floor 0.9925 --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 --seed "$seed" --device cuda --skip-final-test \
    >"$stdout" 2>"$stderr"
  printf '[m5-queue] completed %s\n' "$experiment"
}

wait_for_m4_route
wait_for_m11_fewshot

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
