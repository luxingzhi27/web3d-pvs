#!/usr/bin/env bash
set -euo pipefail

# Verify that neural runtime assets are served from the deployed viewer origin.
#
# Usage:
#   bash verify_deploy_assets.sh http://SERVER_IP:8080

BASE_URL="${1:-http://127.0.0.1:8080}"
BASE_URL="${BASE_URL%/}"

command -v curl >/dev/null 2>&1 || {
  echo "curl is required." >&2
  exit 1
}

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

failures=0

while IFS= read -r path; do
  [[ -n "${path}" ]] || continue
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

  if [[ "${path}" == *.bin && "${bytes}" -lt 1000000 ]]; then
    echo "ERROR: pvs_assets.bin is unexpectedly small. It may be missing or served as HTML." >&2
    failures=$((failures + 1))
  fi

  echo ""
done <<'EOF'
/assets/config.json
/assets/scenes/hkust-v3/glbIndex.json
/assets/scenes/hkust-v3/runtimeVisibilityMeta.json
/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2/glbIndex.json
/assets/scenes/ifcbench_fantasy_metropolis_instanced_v2/runtimeVisibilityMeta.json
/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best/instance_pvs_assets.bin
/assets/neural_instance_culling/pvs_directional_occlusion_proxy_encoder_rvl_strong_v2_full40_best/instance_model_meta.json
/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best/instance_pvs_assets.bin
/assets/neural_instance_culling/pvs_directional_occlusion_proxy_ifcbench_fantasy_metropolis_instanced_v2_k4_full40_best/instance_model_meta.json
EOF

if [[ "${failures}" -gt 0 ]]; then
  echo "Asset verification failed: ${failures} issue(s)." >&2
  exit 1
fi

echo "All neural deployment assets look readable from ${BASE_URL}."
