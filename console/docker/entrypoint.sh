#!/usr/bin/env bash
set -euo pipefail

export AGIWO_ROOT_PATH="${AGIWO_ROOT_PATH:-/data/root}"
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

mkdir -p "${AGIWO_ROOT_PATH}" "${AGIWO_ROOT_PATH}/skills" /data/runtime

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
