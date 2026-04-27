#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/update_redeploy_console_docker.sh [options]

Update source code, rebuild the Console Docker image, and replace the existing
Console container while preserving its envs, bind mounts, user, workdir, entrypoint,
and restart policy.

Options:
  --name NAME           Container name to replace (default: agiwo-console)
  --image TAG           Image tag to build and run (default: agiwo-console:local)
  --network-mode MODE   Docker network mode for replacement (default: host)
  --browser-cli-source PATH
                        Build and install Browser CLI from a local source checkout
  --remote NAME         Git remote used with --branch (default: current upstream)
  --branch NAME         Git branch to pull (default: current upstream)
  --no-pull             Do not run git pull before building
  --no-build            Do not rebuild the Docker image
  --keep-backup         Keep the previous container after a successful redeploy
  --help                Show this help text

Examples:
  scripts/update_redeploy_console_docker.sh
  scripts/update_redeploy_console_docker.sh --no-pull
  scripts/update_redeploy_console_docker.sh --remote origin --branch main
  scripts/update_redeploy_console_docker.sh --browser-cli-source "$HOME/workspace/browser-cli"
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found in PATH: $1" >&2
    exit 1
  fi
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="agiwo-console"
IMAGE="agiwo-console:local"
NETWORK_MODE="host"
BROWSER_CLI_SOURCE=""
BROWSER_CLI_WHEEL_DIR="$ROOT/console/docker/browser-cli-wheels"
BROWSER_CLI_STAGED_WHEEL=""
DO_PULL=1
DO_BUILD=1
KEEP_BACKUP=0
REMOTE=""
BRANCH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      NAME="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --network-mode)
      NETWORK_MODE="${2:-}"
      shift 2
      ;;
    --browser-cli-source)
      BROWSER_CLI_SOURCE="${2:-}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:-}"
      shift 2
      ;;
    --branch)
      BRANCH="${2:-}"
      shift 2
      ;;
    --no-pull)
      DO_PULL=0
      shift
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --keep-backup)
      KEEP_BACKUP=1
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

if [[ -z "$NAME" || -z "$IMAGE" || -z "$NETWORK_MODE" ]]; then
  echo "--name, --image, and --network-mode require non-empty values" >&2
  exit 1
fi

require_cmd docker
require_cmd git
require_cmd jq
if [[ -n "$BROWSER_CLI_SOURCE" ]]; then
  require_cmd uv
fi

stage_browser_cli_wheel() {
  mkdir -p "$BROWSER_CLI_WHEEL_DIR"
  rm -f "$BROWSER_CLI_WHEEL_DIR"/*.whl

  if [[ -z "$BROWSER_CLI_SOURCE" ]]; then
    return 0
  fi

  if [[ ! -d "$BROWSER_CLI_SOURCE" ]]; then
    echo "Browser CLI source directory does not exist: $BROWSER_CLI_SOURCE" >&2
    exit 1
  fi

  local source_dir
  source_dir="$(cd "$BROWSER_CLI_SOURCE" && pwd)"
  if [[ ! -f "$source_dir/pyproject.toml" ]]; then
    echo "Browser CLI source must contain pyproject.toml: $source_dir" >&2
    exit 1
  fi

  echo "[update_redeploy] building Browser CLI wheel from: $source_dir"
  (
    cd "$source_dir"
    uv build --wheel --out-dir "$BROWSER_CLI_WHEEL_DIR" --no-create-gitignore
  )

  BROWSER_CLI_STAGED_WHEEL="$(find "$BROWSER_CLI_WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
  if [[ -z "$BROWSER_CLI_STAGED_WHEEL" || ! -f "$BROWSER_CLI_STAGED_WHEEL" ]]; then
    echo "Browser CLI wheel build did not produce a wheel" >&2
    exit 1
  fi
  echo "[update_redeploy] staged Browser CLI wheel: $BROWSER_CLI_STAGED_WHEEL"
}

cleanup_browser_cli_wheel() {
  if [[ -n "$BROWSER_CLI_STAGED_WHEEL" ]]; then
    rm -f "$BROWSER_CLI_STAGED_WHEEL"
  fi
}

trap cleanup_browser_cli_wheel EXIT

if [[ ! -f "$ROOT/console/Dockerfile" ]]; then
  echo "Missing Dockerfile: $ROOT/console/Dockerfile" >&2
  exit 1
fi

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Container does not exist: $NAME" >&2
  echo "Create it first with scripts/deploy_console.sh or agiwo-console container up." >&2
  exit 1
fi

if [[ "$DO_PULL" -eq 1 ]]; then
  echo "[update_redeploy] checking worktree before git pull"
  (
    cd "$ROOT"
    if ! git diff --quiet || ! git diff --cached --quiet; then
      echo "Worktree has uncommitted changes; refusing to pull." >&2
      echo "Commit/stash them first, or pass --no-pull to deploy local changes." >&2
      exit 1
    fi

    pull_args=(pull --ff-only)
    if [[ -n "$REMOTE" || -n "$BRANCH" ]]; then
      if [[ -z "$REMOTE" || -z "$BRANCH" ]]; then
        echo "--remote and --branch must be provided together" >&2
        exit 1
      fi
      pull_args+=("$REMOTE" "$BRANCH")
    fi

    echo "[update_redeploy] git ${pull_args[*]}"
    git "${pull_args[@]}"
  )
fi

if [[ "$DO_BUILD" -eq 1 ]]; then
  stage_browser_cli_wheel
  echo "[update_redeploy] building image: $IMAGE"
  (
    cd "$ROOT"
    DOCKER_BUILDKIT=1 docker build \
      --build-arg HTTP_PROXY= \
      --build-arg HTTPS_PROXY= \
      --build-arg ALL_PROXY= \
      --build-arg http_proxy= \
      --build-arg https_proxy= \
      --build-arg all_proxy= \
      -f console/Dockerfile \
      -t "$IMAGE" \
      .
  )
  echo "[update_redeploy] build complete"
else
  if [[ -n "$BROWSER_CLI_SOURCE" ]]; then
    echo "[update_redeploy] --browser-cli-source ignored because --no-build was provided" >&2
  fi
fi

BACKUP="${NAME}-prev-$(date +%Y%m%d%H%M%S)"
ENV_FILE="$(mktemp)"
RESTORE_NEEDED=0

cleanup() {
  rm -f "$ENV_FILE"
  cleanup_browser_cli_wheel
}

restore_on_error() {
  status=$?
  if [[ "$RESTORE_NEEDED" -eq 1 ]]; then
    echo "[update_redeploy] redeploy failed; restoring previous container" >&2
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if docker inspect "$BACKUP" >/dev/null 2>&1; then
      docker rename "$BACKUP" "$NAME" >/dev/null
      docker start "$NAME" >/dev/null
    fi
  fi
  cleanup
  exit "$status"
}

trap restore_on_error ERR

CURRENT_IMAGE="$(docker inspect "$NAME" --format "{{.Config.Image}}")"
USER_SPEC="$(docker inspect "$NAME" --format "{{.Config.User}}")"
WORKDIR="$(docker inspect "$NAME" --format "{{.Config.WorkingDir}}")"
ENTRYPOINT="$(docker inspect "$NAME" | jq -r '.[0].Config.Entrypoint[0] // empty')"
RESTART_NAME="$(docker inspect "$NAME" | jq -r '.[0].HostConfig.RestartPolicy.Name // empty')"
RESTART_COUNT="$(docker inspect "$NAME" | jq -r '.[0].HostConfig.RestartPolicy.MaximumRetryCount // 0')"
mapfile -t BINDS < <(docker inspect "$NAME" | jq -r '.[0].HostConfig.Binds[]?')
mapfile -t CMD_ARGS < <(docker inspect "$NAME" | jq -r '.[0].Config.Cmd[]?')
docker inspect "$NAME" --format "{{range .Config.Env}}{{println .}}{{end}}" > "$ENV_FILE"

echo "[update_redeploy] replacing $NAME from $CURRENT_IMAGE with $IMAGE"
docker stop "$NAME" >/dev/null
docker rename "$NAME" "$BACKUP"
RESTORE_NEEDED=1

RUN_ARGS=(run -d --name "$NAME" --network "$NETWORK_MODE")
if [[ -n "$RESTART_NAME" && "$RESTART_NAME" != "no" ]]; then
  if [[ "$RESTART_NAME" == "on-failure" && "$RESTART_COUNT" != "0" ]]; then
    RUN_ARGS+=(--restart "${RESTART_NAME}:${RESTART_COUNT}")
  else
    RUN_ARGS+=(--restart "$RESTART_NAME")
  fi
fi
RUN_ARGS+=(--env-file "$ENV_FILE")
for bind in "${BINDS[@]}"; do
  RUN_ARGS+=(-v "$bind")
done
if [[ -n "$USER_SPEC" ]]; then
  RUN_ARGS+=(--user "$USER_SPEC")
fi
if [[ -n "$WORKDIR" ]]; then
  RUN_ARGS+=(--workdir "$WORKDIR")
fi
if [[ -n "$ENTRYPOINT" ]]; then
  RUN_ARGS+=(--entrypoint "$ENTRYPOINT")
fi
RUN_ARGS+=("$IMAGE")
for arg in "${CMD_ARGS[@]}"; do
  RUN_ARGS+=("$arg")
done

docker "${RUN_ARGS[@]}" >/dev/null

for _ in $(seq 1 60); do
  status="$(docker inspect "$NAME" --format "{{.State.Status}}")"
  health="$(docker inspect "$NAME" --format "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}")"
  if [[ "$health" == "healthy" || ( "$health" == "none" && "$status" == "running" ) ]]; then
    echo "[update_redeploy] new container status=$status health=$health"
    if [[ "$KEEP_BACKUP" -eq 0 ]]; then
      docker rm "$BACKUP" >/dev/null
    else
      echo "[update_redeploy] kept backup container: $BACKUP"
    fi
    RESTORE_NEEDED=0
    cleanup
    trap - ERR
    docker ps --filter "name=^/${NAME}$" --format "{{.ID}} {{.Names}} {{.Image}} {{.Status}}"
    exit 0
  fi

  if [[ "$status" == "exited" || "$health" == "unhealthy" ]]; then
    echo "[update_redeploy] new container failed: status=$status health=$health" >&2
    docker logs --tail 100 "$NAME" >&2 || true
    exit 1
  fi

  sleep 2
done

echo "[update_redeploy] new container did not become healthy in time" >&2
docker logs --tail 100 "$NAME" >&2 || true
exit 1
