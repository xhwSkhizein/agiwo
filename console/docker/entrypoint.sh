#!/usr/bin/env bash
set -euo pipefail

export AGIWO_ROOT_PATH="${AGIWO_ROOT_PATH:-/data/root}"
export BROWSER_CLI_HOME="${BROWSER_CLI_HOME:-${AGIWO_ROOT_PATH}/browser-cli}"
BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-18080}"

if [[ ! -d /data ]]; then
  echo "/data mount is required" >&2
  exit 1
fi

if [[ ! -w /data ]]; then
  echo "/data must be writable" >&2
  exit 1
fi

mkdir -p "${AGIWO_ROOT_PATH}" "${AGIWO_ROOT_PATH}/skills" "${BROWSER_CLI_HOME}" /data/runtime

if command -v browser-cli >/dev/null 2>&1; then
  if ! browser-cli install-skills --target "${AGIWO_ROOT_PATH}/skills"; then
    echo "warning: failed to install Browser CLI skills into ${AGIWO_ROOT_PATH}/skills" >&2
  fi
  if ! browser-cli reload >/data/runtime/browser-cli-reload.log 2>&1; then
    echo "warning: failed to start Browser CLI daemon; see /data/runtime/browser-cli-reload.log" >&2
  fi
fi

backend_pid=""
web_pid=""
nginx_pid=""

shutdown() {
  for pid in "$backend_pid" "$web_pid" "$nginx_pid"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}

trap shutdown EXIT INT TERM

agiwo-console serve --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" &
backend_pid=$!

(
  cd /opt/agiwo-console/web
  HOSTNAME=127.0.0.1 PORT=13000 node server.js
) &
web_pid=$!

nginx -g 'daemon off;' &
nginx_pid=$!

wait -n "$backend_pid" "$web_pid" "$nginx_pid"
exit $?
