#!/usr/bin/env bash
set -euo pipefail

# Install an nginx static site for the SLM2Viewer neural-culling build.
#
# Usage:
#   sudo bash install_nginx_site.sh /var/www/slm2viewer/public_deploy your.domain.com
#
# If domain is omitted, "_" is used as the catch-all server_name.
# The script detects nginx Brotli support. If Brotli is unavailable, it keeps
# gzip_static enabled so the browser can still receive precompressed .gz files.

SITE_ROOT="${1:-/var/www/slm2viewer/public_deploy}"
SERVER_NAME="${2:-_}"
SITE_NAME="${3:-slm2viewer-neural}"
NGINX_CONF="/etc/nginx/sites-available/${SITE_NAME}"
NGINX_ENABLED="/etc/nginx/sites-enabled/${SITE_NAME}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run as root, for example: sudo bash $0 ${SITE_ROOT} ${SERVER_NAME}" >&2
  exit 1
fi

if ! command -v nginx >/dev/null 2>&1; then
  echo "nginx is not installed. Installing nginx with apt..."
  apt-get update
  apt-get install -y nginx
fi

mkdir -p "${SITE_ROOT}"

if [[ ! -f "${SITE_ROOT}/index.html" ]]; then
  cat >&2 <<EOF
ERROR: ${SITE_ROOT}/index.html not found.

Upload the generated slm2viewer/public_deploy directory to ${SITE_ROOT} first.
Example:
  rsync -avz slm2viewer/public_deploy/ user@server:${SITE_ROOT}/
EOF
  exit 1
fi

ensure_brotli_modules_loaded() {
  local module_dir=""
  local static_module=""
  local filter_module=""

  for candidate in /usr/lib/nginx/modules /usr/share/nginx/modules /etc/nginx/modules; do
    if [[ -d "${candidate}" ]]; then
      module_dir="${candidate}"
      break
    fi
  done

  if [[ -n "${module_dir}" ]]; then
    static_module="${module_dir}/ngx_http_brotli_static_module.so"
    filter_module="${module_dir}/ngx_http_brotli_filter_module.so"
  fi

  if nginx -V 2>&1 | grep -qi "brotli"; then
    return 0
  fi

  if [[ -f "${static_module}" ]]; then
    mkdir -p /etc/nginx/modules-enabled
    {
      [[ -f "${filter_module}" ]] && echo "load_module ${filter_module};"
      echo "load_module ${static_module};"
    } > /etc/nginx/modules-enabled/50-brotli.conf
    return 0
  fi

  return 1
}

BROTLI_ENABLED=0
if ensure_brotli_modules_loaded; then
  BROTLI_ENABLED=1
fi

COMMON_STATIC_BLOCK='
    location / {
        try_files $uri $uri/ /index.html;
    }

    location ~* \.(js|css|html|json|bin|wasm|mjs)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }

    location = /index.html {
        add_header Cache-Control "no-cache";
    }

    location ~* \.(jpg|jpeg|png|ico|hdr)$ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        try_files $uri =404;
    }
'

if [[ "${BROTLI_ENABLED}" -eq 1 ]]; then
  cat > "${NGINX_CONF}" <<EOF
server {
    listen 80;
    server_name ${SERVER_NAME};
    root ${SITE_ROOT};
    index index.html;

    brotli_static on;
    gzip_static on;
    gzip_vary on;

${COMMON_STATIC_BLOCK}
}
EOF
else
  cat > "${NGINX_CONF}" <<EOF
server {
    listen 80;
    server_name ${SERVER_NAME};
    root ${SITE_ROOT};
    index index.html;

    gzip_static on;
    gzip_vary on;

${COMMON_STATIC_BLOCK}
}
EOF
fi

ln -sfn "${NGINX_CONF}" "${NGINX_ENABLED}"

if [[ -f /etc/nginx/sites-enabled/default ]]; then
  rm -f /etc/nginx/sites-enabled/default
fi

nginx -t
systemctl reload nginx || systemctl restart nginx

echo "nginx site installed:"
echo "  config: ${NGINX_CONF}"
echo "  root:   ${SITE_ROOT}"
echo "  host:   ${SERVER_NAME}"
if [[ "${BROTLI_ENABLED}" -eq 1 ]]; then
  echo "  compression: brotli_static + gzip_static"
else
  echo "  compression: gzip_static only"
  echo "  note: Brotli nginx module was not found. Install nginx Brotli modules if you want .br files to be served."
fi
