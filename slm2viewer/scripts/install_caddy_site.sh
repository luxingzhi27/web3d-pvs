#!/usr/bin/env bash
set -euo pipefail

# Install the SLM2Viewer neural-culling Caddy site without touching existing
# FileBrowser/domain sites. The viewer is served by IP on a high port by default.
#
# Usage:
#   sudo bash install_caddy_site.sh /var/www/slm2viewer/public_deploy 8080
#
# Then visit:
#   http://SERVER_IP:8080

SITE_ROOT="${1:-/var/www/slm2viewer/public_deploy}"
LISTEN_PORT="${2:-8080}"
SITE_NAME="${3:-slm2viewer}"
CADDYFILE="/etc/caddy/Caddyfile"
CONF_DIR="/etc/caddy/conf.d"
SITE_CONF="${CONF_DIR}/${SITE_NAME}.caddy"
ASSET_PROXY_PATH="/hkust-v3-assets"
REMOTE_ASSET_ORIGIN="https://smart3d.hkust-gz.edu.cn"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root, for example: sudo bash $0 ${SITE_ROOT} ${LISTEN_PORT}" >&2
  exit 1
fi

if ! command -v caddy >/dev/null 2>&1; then
  echo "caddy is not installed. Please install Caddy first." >&2
  exit 1
fi

if [[ ! -f "${SITE_ROOT}/index.html" ]]; then
  cat >&2 <<EOF
ERROR: ${SITE_ROOT}/index.html not found.

Upload the generated slm2viewer/public_deploy directory to ${SITE_ROOT} first.
Example:
  rsync -avz slm2viewer/public_deploy/ user@server:${SITE_ROOT}/
EOF
  exit 1
fi

mkdir -p "${CONF_DIR}"

cat > "${SITE_CONF}" <<EOF
:${LISTEN_PORT} {
    root * ${SITE_ROOT}
    encode zstd gzip

    handle_path ${ASSET_PROXY_PATH}/* {
        rewrite * /proxy/hkust-v3/assets{uri}
        reverse_proxy ${REMOTE_ASSET_ORIGIN} {
            header_up Host smart3d.hkust-gz.edu.cn
        }
    }

    @html path /index.html
    header @html Cache-Control "no-cache"

    @static path *.js *.css *.json *.bin *.wasm *.mjs *.jpg *.png *.ico
    header @static Cache-Control "public, max-age=31536000, immutable"

    handle /assets/* {
        file_server {
            precompressed br gzip
        }
    }

    handle /favicon* {
        file_server {
            precompressed br gzip
        }
    }

    handle {
        try_files {path} /index.html
        file_server {
            precompressed br gzip
        }
    }
}
EOF

if [[ ! -f "${CADDYFILE}" ]]; then
  mkdir -p "$(dirname "${CADDYFILE}")"
  touch "${CADDYFILE}"
fi

if ! grep -qF "import ${CONF_DIR}/*.caddy" "${CADDYFILE}"; then
  {
    echo ""
    echo "import ${CONF_DIR}/*.caddy"
  } >> "${CADDYFILE}"
fi

caddy fmt --overwrite "${SITE_CONF}"
caddy fmt --overwrite "${CADDYFILE}"
caddy validate --config "${CADDYFILE}"
systemctl reload caddy || systemctl restart caddy

echo "Caddy site installed:"
echo "  config: ${SITE_CONF}"
echo "  root:   ${SITE_ROOT}"
echo "  url:    http://SERVER_IP:${LISTEN_PORT}"
echo "  proxy:  ${ASSET_PROXY_PATH}/* -> ${REMOTE_ASSET_ORIGIN}/proxy/hkust-v3/assets/*"
