#!/usr/bin/env bash
set -euo pipefail

export AGIWO_ROOT_PATH="${AGIWO_ROOT_PATH:-/data/root}"
export AGIWO_CONSOLE_HOST="${AGIWO_CONSOLE_HOST:-127.0.0.1}"
export AGIWO_CONSOLE_PORT="${AGIWO_CONSOLE_PORT:-18080}"

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

agiwo-console serve --host "${AGIWO_CONSOLE_HOST}" --port "${AGIWO_CONSOLE_PORT}" &
backend_pid=$!

(
  cd /opt/agiwo-console/web
  npm run start -- --hostname 127.0.0.1 --port 13000
) &
web_pid=$!

nginx -g 'daemon off;' &
nginx_pid=$!

wait -n "$backend_pid" "$web_pid" "$nginx_pid"
exit $?
