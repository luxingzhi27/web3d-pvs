#!/usr/bin/env bash
# Run the registered M8 offline trajectory replay with compatible runner/mode pairs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
DATE_TAG="${SLM_M8_DATE_TAG:-20260802_retry1}"
OUT_ROOT="neural_instance_culling/benchmark/out"
TRAJ_ROOT="$OUT_ROOT/m8_trajectories_20260802"

HKUST_DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
HKUST_RUNTIME="hkust-v3/assets/runtimeVisibilityMeta.json"
HKUST_INDEX="hkust-v3/assets/glbIndex.json"
HKUST_ROOT="hkust-v3/assets"
HKUST_COST="$OUT_ROOT/m7_glb_decode_upload_costs_hkust_20260802.json"
HKUST_MODEL="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/best.pt"
HKUST_FEATURES="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/instance_runtime_features_fp16.bin"
HKUST_SUMMARY="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_hkust_spatial_fov66_seed20260801_protocolfix_retry2/calibration_ready_summary.json"
HKUST_RANKER="neural_instance_culling/model/out/m7_independent_ranknet_hkust_spatial_fov66_seed20260801/best.pt"

METRO_DATASET="neural_instance_culling/dataset/out/pose_csr_metropolis_spatial_dense_subpose_union_fov66_v2"
METRO_RUNTIME="ifcbench_fantasy_metropolis_instanced_v2/assets/runtimeVisibilityMeta.json"
METRO_INDEX="ifcbench_fantasy_metropolis_instanced_v2/assets/glbIndex.json"
METRO_ROOT="ifcbench_fantasy_metropolis_instanced_v2/assets"
METRO_COST="$OUT_ROOT/m8_glb_decode_upload_costs_metropolis_20260802.json"
METRO_MODEL="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2/best.pt"
METRO_FEATURES="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2/instance_runtime_features_fp16.bin"
METRO_SUMMARY="neural_instance_culling/model/out/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_metropolis_spatial_fov66_seed20260801_protocolfix_bs1_gpu2/calibration_ready_summary.json"
METRO_RANKER="neural_instance_culling/model/out/m7_independent_ranknet_metropolis_spatial_fov66_seed20260801/best.pt"

run_case() {
  local scene="$1"
  local track="$2"
  local mode="$3"
  local dataset runtime index root cost trajectory output prefix
  local model features summary ranker

  if [[ "$scene" == "hkust" ]]; then
    dataset="$HKUST_DATASET"
    runtime="$HKUST_RUNTIME"
    index="$HKUST_INDEX"
    root="$HKUST_ROOT"
    cost="$HKUST_COST"
    model="$HKUST_MODEL"
    features="$HKUST_FEATURES"
    summary="$HKUST_SUMMARY"
    ranker="$HKUST_RANKER"
  else
    dataset="$METRO_DATASET"
    runtime="$METRO_RUNTIME"
    index="$METRO_INDEX"
    root="$METRO_ROOT"
    cost="$METRO_COST"
    model="$METRO_MODEL"
    features="$METRO_FEATURES"
    summary="$METRO_SUMMARY"
    ranker="$METRO_RANKER"
  fi

  trajectory="$TRAJ_ROOT/${scene}_track_${track}_wifi.json"
  prefix="$OUT_ROOT/m8_formal_${mode}_${scene}_track_${track}_${DATE_TAG}"
  output="$prefix.json"
  if [[ -f "$output" ]]; then
    printf '[m8-replay] keeping completed %s\n' "$output"
    return 0
  fi
  if [[ -e "$output" || -e "$prefix.stdout.log" || -e "$prefix.stderr.log" ]]; then
    printf '[m8-replay] refusing to reuse incomplete output prefix: %s\n' "$prefix" >&2
    return 1
  fi

  local -a runner_args
  if [[ "$mode" == "current" ]]; then
    runner_args=(--learned-model-spec "m8_mainline_${scene}|$model|$features|$summary" --score-mode current-cascade)
  else
    runner_args=(--independent-ranker-spec "m8_ranknet_${scene}|$ranker" --score-mode independent-utility)
  fi

  printf '[m8-replay] starting %s %s %s\n' "$scene" "$track" "$mode"
  PYTHONDONTWRITEBYTECODE=1 conda run --no-capture-output -n "$ENV_NAME" python -u \
    neural_instance_culling/benchmark/evaluate_download_trajectory.py \
    --trajectory "$trajectory" \
    --dataset-dir "$dataset" \
    --runtime-meta "$runtime" \
    --glb-index "$index" \
    --glb-root "$root" \
    --glb-time-index "$cost" \
    --models "" \
    "${runner_args[@]}" \
    --glb-aggregation max \
    --aggregation-top-k 2 \
    --device cpu \
    --byte-budgets 1048576,5242880,10485760,20971520 \
    --time-budgets-ms 100,250,500,1000 \
    --output "$output" \
    >"$prefix.stdout.log" 2>"$prefix.stderr.log"
  printf '[m8-replay] completed %s\n' "$output"
}

for mode in current ranknet; do
  for scene in hkust metropolis; do
    for track in a b c; do
      run_case "$scene" "$track" "$mode" &
    done
  done
  wait
done

printf '[m8-replay] all compatible runner/mode pairs completed\n'
