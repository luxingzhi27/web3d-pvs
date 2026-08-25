#!/usr/bin/env bash
set -euo pipefail

# Verify that neural runtime assets are served from the deployed viewer origin.
#
# Usage:
#   bash verify_deploy_assets.sh http://SERVER_IP:8080 hkust-v3

BASE_URL="${1:-http://127.0.0.1:8080}"
BASE_URL="${BASE_URL%/}"
SCENE="${2:-hkust-v3}"

command -v curl >/dev/null 2>&1 || {
  echo "curl is required." >&2
  exit 1
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

failures=0

paths=(
  "/assets/config.json"
  "/assets/scenes/${SCENE}/glbIndex.json"
  "/assets/scenes/${SCENE}/runtimeVisibilityMeta.json"
)
if [[ "${SCENE}" == "hkust-v3" ]]; then
  paths+=(
    "/assets/neural_instance_culling/pvs_mainline_v4/model_meta.json"
    "/assets/neural_instance_culling/pvs_mainline_v4/instance_runtime_features_fp16.bin"
    "/assets/neural_instance_culling/pvs_mainline_v4/instance_aabb_fp32.bin"
    "/assets/neural_instance_culling/pvs_mainline_v4/instance_to_glb_uint32.bin"
    "/assets/neural_instance_culling/pvs_mainline_v4/query_weights_fp16.bin"
    "/assets/neural_instance_culling/pvs_mainline_v4/frequency_cycles_fp32.bin"
    "/assets/neural_instance_culling/pvs_mainline_v4/chi_table_fp32.bin"
  )
elif [[ "${SCENE}" != "ifcbench_fantasy_metropolis_instanced_v2" ]]; then
  echo "Unsupported scene: ${SCENE}" >&2
  exit 2
fi

for path in "${paths[@]}"; do
  url="${BASE_URL}${path}"
  out="${tmp_dir}/asset"
  headers="${tmp_dir}/headers"

  status="$(curl -L --compressed -sS -D "${headers}" -o "${out}" -w "%{http_code}" "${url}" || true)"
  # curl may fail before creating the output file (DNS/TLS/connection error).
  # Keep the diagnostic path below usable and report the HTTP failure cleanly.
  [[ -f "${out}" ]] || : > "${out}"
  [[ -f "${headers}" ]] || : > "${headers}"
  bytes="$(wc -c < "${out}" | tr -d ' ')"
  content_type="$(awk 'BEGIN{IGNORECASE=1} /^content-type:/ {sub(/\r$/, ""); print $0}' "${headers}" | tail -n 1)"
  content_encoding="$(awk 'BEGIN{IGNORECASE=1} /^content-encoding:/ {sub(/\r$/, ""); print $0}' "${headers}" | tail -n 1)"
  head_text="$(head -c 32 "${out}" | LC_ALL=C tr -cd '[:print:]' || true)"

  echo "== ${path}"
  echo "status=${status} bytes=${bytes}"
  echo "${content_type:-content-type: -}"
  echo "${content_encoding:-content-encoding: -}"
  echo "head=${head_text}"

  if [[ "${status}" != "200" ]]; then
    echo "ERROR: expected HTTP 200" >&2
    failures=$((failures + 1))
  fi

  if [[ "${path}" == *.json && "${head_text}" != \{* && "${head_text}" != \[* ]]; then
    echo "ERROR: JSON asset does not look like JSON. It may be index.html fallback." >&2
    failures=$((failures + 1))
  fi

  if [[ "${path}" == *.bin && "${bytes}" -lt 32 ]]; then
    echo "ERROR: runtime binary is unexpectedly small. It may be missing or served as HTML." >&2
    failures=$((failures + 1))
  fi

  echo ""
done

if [[ "${failures}" -gt 0 ]]; then
  echo "Asset verification failed: ${failures} issue(s)." >&2
  exit 1
fi

echo "All neural deployment assets look readable from ${BASE_URL}."
