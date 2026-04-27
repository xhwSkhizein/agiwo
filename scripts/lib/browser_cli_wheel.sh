#!/usr/bin/env bash

BROWSER_CLI_SOURCE="${BROWSER_CLI_SOURCE:-}"
BROWSER_CLI_WHEEL_DIR="${BROWSER_CLI_WHEEL_DIR:-$ROOT/console/docker/browser-cli-wheels}"
BROWSER_CLI_STAGED_WHEEL="${BROWSER_CLI_STAGED_WHEEL:-}"

stage_browser_cli_wheel() {
  local log_prefix="${1:-browser_cli_wheel}"

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

  echo "[$log_prefix] building Browser CLI wheel from: $source_dir"
  (
    cd "$source_dir"
    uv build --wheel --out-dir "$BROWSER_CLI_WHEEL_DIR" --no-create-gitignore
  )

  local wheels=()
  mapfile -t wheels < <(find "$BROWSER_CLI_WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' | sort)
  if [[ "${#wheels[@]}" -ne 1 ]]; then
    echo "Browser CLI wheel build must produce exactly one wheel; found ${#wheels[@]}" >&2
    exit 1
  fi

  BROWSER_CLI_STAGED_WHEEL="${wheels[0]}"
  echo "[$log_prefix] staged Browser CLI wheel: $BROWSER_CLI_STAGED_WHEEL"
}

cleanup_browser_cli_wheel() {
  if [[ -n "$BROWSER_CLI_STAGED_WHEEL" ]]; then
    rm -f "$BROWSER_CLI_STAGED_WHEEL"
  fi
}

trap cleanup_browser_cli_wheel EXIT
