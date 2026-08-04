#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

ENV_NAME="${SLM_CONDA_ENV:-slm_pvs}"
GPU_LIST="${SLM_M5_SUBPOSE_GPUS:-0 1 2}"
SEEDS="${SLM_M5_SUBPOSE_SEEDS:-20260801 20260802 20260803}"

DATASET="neural_instance_culling/dataset/out/pose_csr_hkust_v3_spatial_raw_subpose_aabb_fov66_v1"
EVIDENCE="neural_instance_culling/dataset/out/directional_occlusion_evidence_hkust_spatial_fov66_v1"
GLB_POINTS="neural_instance_culling/dataset/out/glb_points_v3_formal_hkust_fov66.bin"
RUNTIME_META="hkust-v3/assets/runtimeVisibilityMeta.json"
GLB_INDEX="hkust-v3/assets/glbIndex.json"
GLB_ROOT="hkust-v3/assets"
OUTPUT_ROOT="neural_instance_culling/model/out"
RUN_TAG="${SLM_M5_SUBPOSE_RUN_TAG:-}"
if [[ "$RUN_TAG" == */* || "$RUN_TAG" == *..* ]]; then
  echo "SLM_M5_SUBPOSE_RUN_TAG must not contain '/' or '..'" >&2
  exit 1
fi
VARIANT="pvs_m5_subpose_robust_v1_hkust_spatial_fov66${RUN_TAG}"

[[ -f "$DATASET/visible_hit_counts.bin" ]] || { echo "missing dense subpose hit-count labels" >&2; exit 1; }
[[ -f "$DATASET/subpose_offsets.bin" ]] || { echo "missing dense subpose offsets" >&2; exit 1; }
[[ -f "$DATASET/dataset_meta.json" ]] || { echo "missing dataset metadata" >&2; exit 1; }

run_one() {
  local gpu="$1"
  local seed="$2"
  local experiment="${VARIANT}_seed${seed}_full40"
  local output_dir="$OUTPUT_ROOT/$experiment"
  local ready="$output_dir/calibration_ready_summary.json"

  if [[ -f "$ready" ]]; then
    echo "[m5-subpose] keeping completed $experiment"
    return 0
  fi
  if [[ -e "$output_dir" ]]; then
    echo "[m5-subpose] refusing to reuse incomplete output directory: $output_dir" >&2
    return 1
  fi
  mkdir -p "$output_dir"
  echo "[m5-subpose] starting $experiment on physical GPU $gpu"
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
    --loss-profile m5_subpose_robust_v1 \
    --target-weighted-recall 0.99 \
    --calibration-pose-recall-floor 0.95 \
    --calibration-point-floor 0.9925 \
    --calibration-lcb-floor 0.99 \
    --calibration-bootstrap-replicates 10000 \
    --seed "$seed" --device cuda --skip-final-test \
    >"$output_dir/train_stdout.log" 2>"$output_dir/train_stderr.log"
  echo "[m5-subpose] completed $experiment"
}

gpu_array=($GPU_LIST)
(( ${#gpu_array[@]} >= 1 )) || { echo "SLM_M5_SUBPOSE_GPUS is empty" >&2; exit 1; }

jobs=()
index=0
for seed in $SEEDS; do
  gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
  run_one "$gpu" "$seed" &
  jobs+=("$!")
  index=$((index + 1))
done
for pid in "${jobs[@]}"; do
  wait "$pid"
done
echo "[m5-subpose] all registered seeds completed"
