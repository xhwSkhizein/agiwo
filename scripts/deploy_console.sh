#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/deploy_console.sh --env-file PATH --data-dir PATH [options]

Options:
  --env-file PATH       Path to the env file passed to agiwo-console container up
  --env KEY=VALUE       Extra env passed to agiwo-console container up; may be repeated
  --env-name NAME       Forward host env NAME into the container; may be repeated
  --data-dir PATH       Host data directory mounted to /data
  --name NAME           Container name (default: agiwo-console)
  --image TAG           Local image tag (default: agiwo-console:local)
  --publish HOST:PORT   Port mapping passed to container up (default: 8422:8422)
  --network-mode MODE   Docker network mode passed to container up (bridge|host; default: bridge)
  --mount SRC:ALIAS     Additional host mount; may be repeated
  --pull                Forward --pull to container up
  --no-build            Skip docker build and use the existing image tag
  --help                Show this help text
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found in PATH: $1" >&2
    exit 1
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=""
DATA_DIR=""
NAME="agiwo-console"
IMAGE="agiwo-console:local"
PUBLISH="8422:8422"
NETWORK_MODE="bridge"
DO_BUILD=1
PULL=0
MOUNTS=()
EXTRA_ENVS=()
FORWARD_ENV_NAMES=()
AUTO_FORWARD_ENV_NAMES=(
  SERPER_API_KEY
  AGIWO_TOOL_WEB_SEARCH_SERPER_API_KEY
  HTTP_PROXY
  HTTPS_PROXY
  ALL_PROXY
  http_proxy
  https_proxy
  all_proxy
  NO_PROXY
  no_proxy
)
DISCOVERED_ENV_NAMES=()
BUILD_PROXY_ARGS=(
  --build-arg HTTP_PROXY=
  --build-arg HTTPS_PROXY=
  --build-arg ALL_PROXY=
  --build-arg http_proxy=
  --build-arg https_proxy=
  --build-arg all_proxy=
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file)
      ENV_FILE="${2:-}"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="${2:-}"
      shift 2
      ;;
    --env-name)
      FORWARD_ENV_NAMES+=("${2:-}")
      shift 2
      ;;
    --env)
      EXTRA_ENVS+=("${2:-}")
      shift 2
      ;;
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --publish)
      PUBLISH="${2:-}"
      shift 2
      ;;
    --network-mode)
      NETWORK_MODE="${2:-}"
      shift 2
      ;;
    --mount)
      MOUNTS+=("${2:-}")
      shift 2
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$ENV_FILE" ]]; then
  echo "--env-file is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Env file does not exist: $ENV_FILE" >&2
  exit 1
fi

if [[ -z "$DATA_DIR" ]]; then
  echo "--data-dir is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$ROOT/console/Dockerfile" ]]; then
  echo "Missing Dockerfile: $ROOT/console/Dockerfile" >&2
  exit 1
fi

require_cmd docker
require_cmd uv

mkdir -p "$DATA_DIR"
ENV_FILE="$(cd "$(dirname "$ENV_FILE")" && pwd)/$(basename "$ENV_FILE")"
DATA_DIR="$(cd "$DATA_DIR" && pwd)"

while IFS='=' read -r raw_key raw_value; do
  key="${raw_key#"${raw_key%%[![:space:]]*}"}"
  key="${key%"${key##*[![:space:]]}"}"
  value="${raw_value#"${raw_value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  if [[ -z "$key" || "$key" == \#* ]]; then
    continue
  fi
  if [[ "$key" != *_API_KEY_ENV_NAME ]]; then
    continue
  fi
  if [[ -n "$value" ]]; then
    DISCOVERED_ENV_NAMES+=("$value")
  fi
done < "$ENV_FILE"

if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[deploy_console] building image: $IMAGE"
  (
    cd "$ROOT"
    DOCKER_BUILDKIT=1 docker build "${BUILD_PROXY_ARGS[@]}" -f console/Dockerfile -t "$IMAGE" .
  )
  echo "[deploy_console] build complete"
fi

CMD=(
  uv
  run
  --project
  console
  agiwo-console
  container
  up
  --name "$NAME"
  --image "$IMAGE"
  --data-dir "$DATA_DIR"
  --env-file "$ENV_FILE"
  --publish "$PUBLISH"
  --network-mode "$NETWORK_MODE"
  --replace
)

if [[ "$PULL" -eq 1 ]]; then
  CMD+=(--pull)
fi

for mount_spec in "${MOUNTS[@]}"; do
  CMD+=(--mount "$mount_spec")
done

for env_spec in "${EXTRA_ENVS[@]}"; do
  CMD+=(--env "$env_spec")
done

for env_name in "${FORWARD_ENV_NAMES[@]}"; do
  if [[ -z "$env_name" ]]; then
    echo "--env-name requires a non-empty variable name" >&2
    exit 1
  fi
  if [[ -z "${!env_name+x}" ]]; then
    echo "Host environment variable is not set: $env_name" >&2
    exit 1
  fi
  CMD+=(--env "$env_name=${!env_name}")
done

for env_name in "${AUTO_FORWARD_ENV_NAMES[@]}"; do
  if [[ -z "${!env_name+x}" ]]; then
    continue
  fi
  CMD+=(--env "$env_name=${!env_name}")
done

for env_name in "${DISCOVERED_ENV_NAMES[@]}"; do
  if [[ -z "${!env_name+x}" ]]; then
    continue
  fi
  CMD+=(--env "$env_name=${!env_name}")
done

echo "[deploy_console] deploying container: $NAME"
(
  cd "$ROOT"
  "${CMD[@]}"
)
echo "[deploy_console] deployment complete"

if [[ "$NETWORK_MODE" == "host" ]]; then
  HOST_PORT="8422"
else
  HOST_PORT="${PUBLISH%%:*}"
fi
echo "Console is available at: http://localhost:${HOST_PORT}"
echo "Status: uv run --project console agiwo-console container status --name \"$NAME\""
echo "Logs:   uv run --project console agiwo-console container logs --name \"$NAME\""
echo "Down:   uv run --project console agiwo-console container down --name \"$NAME\""
