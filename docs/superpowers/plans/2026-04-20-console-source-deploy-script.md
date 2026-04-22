# Console Source Deploy Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repo-local `scripts/deploy_console.sh` shortcut that builds the Console Docker image from the current source tree and deploys it through `agiwo-console container up` with explicit `--env-file` support.

**Architecture:** Keep the new behavior in a thin Bash script that validates inputs, builds `console/Dockerfile`, and then delegates container lifecycle management to the existing `uv run --project console agiwo-console container up` flow. Update the Docker deployment docs to point source-tree operators at the new shortcut instead of making them remember the raw build and startup commands.

**Tech Stack:** Bash, Docker, `uv`, existing `agiwo-console` CLI, Markdown docs, repo lint scripts

---

### Task 1: Add the Source Deploy Script

**Files:**
- Create: `scripts/deploy_console.sh`
- Test: manual shell smoke from the repo root using a temporary env file and data directory

- [ ] **Step 1: Create the shell script skeleton and usage contract**

Use `scripts/deploy_console.sh` with this initial structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/deploy_console.sh --env-file PATH --data-dir PATH [options]

Options:
  --env-file PATH       Path to the env file passed to agiwo-console container up
  --data-dir PATH       Host data directory mounted to /data
  --name NAME           Container name (default: agiwo-console)
  --image TAG           Local image tag (default: agiwo-console:local)
  --publish HOST:PORT   Port mapping passed to container up (default: 8422:8422)
  --mount SRC:ALIAS     Additional host mount; may be repeated
  --pull                Forward --pull to container up
  --no-build            Skip docker build and use the existing image tag
  --help                Show this help text
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE=""
DATA_DIR=""
NAME="agiwo-console"
IMAGE="agiwo-console:local"
PUBLISH="8422:8422"
DO_BUILD=1
PULL=0
MOUNTS=()
```

- [ ] **Step 2: Parse arguments and reject unsupported input**

Extend the script with this option parser:

```bash
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
```

- [ ] **Step 3: Add prerequisite and input validation**

Add validation helpers and checks after argument parsing:

```bash
require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found in PATH: $1" >&2
    exit 1
  fi
}

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
```

- [ ] **Step 4: Add docker build and deployment command assembly**

Append the build and deploy logic:

```bash
if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "[deploy_console] building image: $IMAGE"
  (
    cd "$ROOT"
    DOCKER_BUILDKIT=1 docker build -f console/Dockerfile -t "$IMAGE" .
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
  --replace
)

if [[ "$PULL" -eq 1 ]]; then
  CMD+=(--pull)
fi

for mount_spec in "${MOUNTS[@]}"; do
  CMD+=(--mount "$mount_spec")
done

echo "[deploy_console] deploying container: $NAME"
(
  cd "$ROOT"
  "${CMD[@]}"
)
echo "[deploy_console] deployment complete"
```

- [ ] **Step 5: Print operator follow-up commands**

Finish the script with a host-port summary and lifecycle hints:

```bash
HOST_PORT="${PUBLISH%%:*}"
echo "Console is available at: http://localhost:${HOST_PORT}"
echo "Status: uv run --project console agiwo-console container status --name \"$NAME\""
echo "Logs:   uv run --project console agiwo-console container logs --name \"$NAME\""
echo "Down:   uv run --project console agiwo-console container down --name \"$NAME\""
```

- [ ] **Step 6: Make the script executable and inspect the final file**

Run:

```bash
chmod +x scripts/deploy_console.sh
sed -n '1,240p' scripts/deploy_console.sh
```

Expected: the script is executable and contains the usage, validation, build, deploy, and follow-up output blocks above.

- [ ] **Step 7: Run the script help output**

Run:

```bash
scripts/deploy_console.sh --help
```

Expected: PASS and the usage text lists `--env-file`, `--data-dir`, `--name`, `--image`, `--publish`, `--mount`, `--pull`, and `--no-build`.

- [ ] **Step 8: Commit the deploy script**

Run:

```bash
git add scripts/deploy_console.sh
git commit -m "feat: add console source deploy script"
```

Expected: commit succeeds with only the new deploy script staged for this task.

### Task 2: Document the Source-Repo Deploy Flow

**Files:**
- Modify: `docs/console/docker.md`

- [ ] **Step 1: Add a source-repo quick-start section**

Update `docs/console/docker.md` so it contains this additional section after the package-install quick start:

````md
## From a Cloned Repository

If you are deploying from this source repository instead of an installed `agiwo-console`
package, use the repo shortcut script:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data"
```

The script builds the current `console/Dockerfile` image and then starts the managed
container through `uv run --project console agiwo-console container up`.
````

- [ ] **Step 2: Add follow-up lifecycle examples for source deployments**

Append these commands in the lifecycle section or immediately after the new source-repo section:

````md
For a source-repo deployment, use the existing CLI for follow-up operations:

```bash
uv run --project console agiwo-console container status
uv run --project console agiwo-console container logs
uv run --project console agiwo-console container restart
uv run --project console agiwo-console container down
```
````

- [ ] **Step 3: Review the rendered documentation content**

Run:

```bash
sed -n '1,220p' docs/console/docker.md
```

Expected: the doc still describes the package-installed flow, now also includes the repo-local shortcut, and clearly states that lifecycle commands remain on `agiwo-console container ...`.

- [ ] **Step 4: Commit the documentation update**

Run:

```bash
git add docs/console/docker.md
git commit -m "docs: add console source deploy instructions"
```

Expected: commit succeeds with only the Docker deployment doc staged for this task.

### Task 3: Validate the Shortcut End to End

**Files:**
- Verify: `scripts/deploy_console.sh`
- Verify: `docs/console/docker.md`

- [ ] **Step 1: Prepare a temporary env file and data directory**

Run:

```bash
tmpdir="$(mktemp -d)"
cat > "$tmpdir/.env" <<'EOF'
OPENAI_API_KEY=test-key
EOF
mkdir -p "$tmpdir/data"
printf '%s\n' "$tmpdir"
```

Expected: PASS and prints the temporary working directory path.

- [ ] **Step 2: Run the deploy script against the temporary paths**

Run:

```bash
scripts/deploy_console.sh \
  --env-file "$tmpdir/.env" \
  --data-dir "$tmpdir/data" \
  --name agiwo-console-smoke \
  --image agiwo-console:smoke \
  --publish 18422:8422
```

Expected: PASS after the image build and container health check. The output should end with `http://localhost:18422` and the `status/logs/down` follow-up commands.

- [ ] **Step 3: Verify the public health endpoint**

Run:

```bash
curl --fail http://127.0.0.1:18422/api/health
```

Expected: PASS with `{"status":"ok"}` or equivalent JSON health output.

- [ ] **Step 4: Verify the mounted data root was initialized**

Run:

```bash
test -d "$tmpdir/data/root"
```

Expected: PASS and no output, confirming the managed container initialized `/data/root`.

- [ ] **Step 5: Tear down the smoke container**

Run:

```bash
uv run --project console agiwo-console container down --name agiwo-console-smoke
rm -rf "$tmpdir"
```

Expected: PASS and the temporary container is removed.

- [ ] **Step 6: Run repository lint**

Run:

```bash
uv run python scripts/lint.py ci
```

Expected: PASS.

- [ ] **Step 7: Commit the validated final state**

Run:

```bash
git add scripts/deploy_console.sh docs/console/docker.md
git commit -m "feat: add console source deploy shortcut"
```

Expected: commit succeeds after the script and doc changes are fully validated.
