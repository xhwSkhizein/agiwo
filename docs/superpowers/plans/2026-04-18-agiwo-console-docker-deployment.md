# Agiwo Console Docker Deployment Implementation Plan

> **For agentic workers:** `writing-plans` is unavailable in this session. Follow this repo-native plan directly and implement one task at a time. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a first-class Docker deployment path for `agiwo-console` that starts a complete Console instance, including the FastAPI backend, Web UI, Agent runtime, Bash execution, and persistence, through a single CLI-managed container workflow.

**Architecture:** Keep `agiwo-console serve` intact as the host-mode path. Add a parallel `agiwo-console container ...` command group that manages one official Docker image and one managed container. Inside that container, run the backend, Web UI, and a reverse proxy behind one public port. Root all default persistence under `/data`, expose host work directories only through explicit mounts under `/mnt/host/<alias>`, and keep the existing `AGIWO_*` / `AGIWO_CONSOLE_*` configuration model canonical.

**Tech Stack:** Python 3.10+, FastAPI, Next.js, Docker, uvicorn, uv, Hatchling, pytest, Vitest, shell entrypoint tooling, lightweight reverse proxy

---

## File Structure

- Modify: `console/server/cli.py`
  - Add the `container` command group and keep `serve` behavior unchanged.
- Create: `console/server/docker_runtime.py`
  - Centralize Docker CLI argument normalization, preflight checks, command construction, and lifecycle helpers.
- Create: `console/server/tests/test_cli.py`
  - Add unit coverage for CLI parsing and Docker command dispatch.
- Create: `console/server/tests/test_docker_runtime.py`
  - Add focused tests for mount validation, alias validation, network-mode handling, and health-check behavior.
- Create: `console/Dockerfile`
  - Build the official single-image Console runtime with backend, frontend, and runtime tools.
- Create: `console/docker/entrypoint.sh`
  - Start the backend, Web UI, and reverse proxy inside the container.
- Create: `console/docker/nginx.conf`
  - Route `/api/*` and SSE traffic to FastAPI and route all other requests to the Web UI.
- Create: `console/docker/healthcheck.sh`
  - Provide a stable in-container health probe.
- Modify: `console/web/src/lib/api.ts`
  - Switch the frontend API base strategy from an absolute localhost default to same-origin-first behavior.
- Modify: `console/web/next.config.ts`
  - Add any runtime-facing configuration needed to keep the production container path stable.
- Modify: `console/pyproject.toml`
  - Ensure the package includes Docker runtime assets in the built artifact if needed by the CLI-managed flow.
- Modify: `README.md`
  - Document the Docker deployment path.
- Modify: `docs/console/overview.md`
  - Describe host mode versus container mode and the one-port deployment shape.
- Create: `docs/console/docker.md`
  - Add the dedicated Docker deployment guide.
- Modify: `AGENTS.md`
  - Update repository-level startup or deployment guidance if the stable Console startup surface changes materially.
- Modify: `.github/workflows/ci.yml`
  - Add Docker build and smoke coverage for the supported container path.

### Task 1: Add The Docker Container Lifecycle Surface To `agiwo-console`

**Files:**
- Modify: `console/server/cli.py`
- Create: `console/server/docker_runtime.py`
- Create: `console/server/tests/test_cli.py`
- Create: `console/server/tests/test_docker_runtime.py`
- Test: `console/server/tests/test_cli.py`
- Test: `console/server/tests/test_docker_runtime.py`

- [ ] **Step 1: Capture the current CLI behavior with thin regression coverage before expanding the command surface**

Add tests that prove:

- `agiwo-console serve` still dispatches to `uvicorn.run`
- `--host`, `--port`, `--env-file`, and `--reload` still pass through unchanged

- [ ] **Step 2: Introduce a dedicated Docker runtime module instead of embedding container logic directly in `cli.py`**

Create a small module, for example `console/server/docker_runtime.py`, that owns:

- Docker executable discovery
- daemon reachability checks
- `--data-dir` creation and writability checks
- mount alias validation
- Docker command construction
- post-start health polling
- `up`, `down`, `status`, `logs`, and `restart` execution helpers

This keeps `cli.py` as a command parser and dispatch layer rather than a process-orchestration file.

- [ ] **Step 3: Add the `container` command group in `console/server/cli.py`**

Recommended subcommands:

- `container up`
- `container down`
- `container status`
- `container logs`
- `container restart`

Recommended `container up` flags:

- `--name`
- `--image`
- `--data-dir`
- repeated `--mount <source>:<alias>`
- `--env-file`
- repeated `--env KEY=VALUE`
- `--publish`
- `--network-mode`
- `--pull`
- `--replace`

- [ ] **Step 4: Define and test strict preflight validation rules**

Enforce these behaviors:

- missing `docker` binary fails early
- unreachable daemon fails early
- missing `--data-dir` is created automatically, then verified writable
- missing `--mount` source path fails
- duplicate or invalid aliases fail
- `--network-mode=host` fails on unsupported platforms
- existing container name fails unless `--replace` is set

- [ ] **Step 5: Define the managed `docker run` contract**

The runtime helper should translate the supported flags into a controlled `docker run` invocation with defaults equivalent to:

```bash
docker run -d \
  --name agiwo-console \
  --restart unless-stopped \
  -p 8422:8422 \
  -v <data-dir>:/data \
  -e AGIWO_ROOT_PATH=/data/root \
  <image>
```

Each explicit mount should map to:

```bash
-v <source>:/mnt/host/<alias>
```

- [ ] **Step 6: Require health-based startup completion**

`container up` should not return success just because `docker run` returned zero. It should poll `GET /api/health` on the public endpoint and only succeed once the container is actually serving traffic.

On failure:

- return a non-zero exit code
- print a short diagnostic summary
- include recent container logs or a targeted log tail
- leave the failed container intact for debugging

- [ ] **Step 7: Add CLI unit tests for success and failure paths**

Cover at least:

- command parsing
- Docker argument construction
- `--replace` behavior
- `status` output for missing/running/stopped containers
- health-check timeout handling
- unsupported host-network selection

- [ ] **Step 8: Run the focused backend test slice and commit**

Run:

```bash
cd console
uv run pytest server/tests/test_cli.py server/tests/test_docker_runtime.py -v
```

Then commit:

```bash
git add console/server/cli.py console/server/docker_runtime.py console/server/tests/test_cli.py console/server/tests/test_docker_runtime.py
git commit -m "feat: add console docker lifecycle cli"
```

### Task 2: Build The Official Complete Console Runtime Image

**Files:**
- Create: `console/Dockerfile`
- Create: `console/docker/entrypoint.sh`
- Create: `console/docker/nginx.conf`
- Create: `console/docker/healthcheck.sh`
- Modify: `console/pyproject.toml`
- Test: Docker image build and runtime smoke

- [ ] **Step 1: Add a multi-stage Dockerfile for the full Console deployment unit**

The Dockerfile should:

- build the Web UI in a dedicated Node stage
- install the Console backend package and runtime dependencies in the final stage
- copy the built frontend assets or production app bundle into the final image
- install the required runtime tools:
  - Python
  - Node.js / npm
  - git
  - bash / coreutils
  - `curl` / `wget`
  - `ripgrep`
  - `jq`
  - `ffmpeg`
  - `yt-dlp`

- [ ] **Step 2: Add a fixed container entrypoint**

`console/docker/entrypoint.sh` should:

- verify `/data` is writable
- create `/data/root`, `/data/root/skills`, and `/data/runtime`
- apply required default environment values such as `AGIWO_ROOT_PATH=/data/root`
- start the backend, Web UI, and reverse proxy
- terminate the container if any required subprocess exits unexpectedly

- [ ] **Step 3: Add the reverse-proxy contract**

Use a lightweight reverse proxy configuration so that:

- `/api/*` goes to FastAPI
- `/api/health` goes to FastAPI
- SSE routes continue to work through the proxy
- all other requests go to the Web UI

The container should expose one public port, `8422`, and no separate documented frontend port.

- [ ] **Step 4: Add a stable healthcheck**

The image should include a healthcheck script that probes the same in-container public entrypoint the CLI depends on. Keep the health semantics aligned with `container up`.

- [ ] **Step 5: Ensure packaging includes any runtime assets the CLI depends on**

If the Console wheel needs to ship Docker templates or runtime assets for the CLI-managed path, update `console/pyproject.toml` build configuration so those files are included in the package.

- [ ] **Step 6: Add a reproducible local smoke command**

Verify the image can be built and run locally with a temporary data directory and that:

- `GET /api/health` succeeds
- the root browser route responds on the same port
- `/data/root` is created

- [ ] **Step 7: Commit the Docker runtime artifact**

```bash
git add console/Dockerfile console/docker/entrypoint.sh console/docker/nginx.conf console/docker/healthcheck.sh console/pyproject.toml
git commit -m "feat: add official console docker image"
```

### Task 3: Make The Web UI Work Behind The Same-Origin Container Entry Point

**Files:**
- Modify: `console/web/src/lib/api.ts`
- Modify: `console/web/next.config.ts`
- Test: `console/web` lint, test, and build

- [ ] **Step 1: Remove the hard-coded absolute localhost default from the frontend API layer**

The current API helper defaults to `http://localhost:8422`. Replace that with same-origin-first behavior so the production container path works correctly behind the reverse proxy.

Preferred behavior:

- when `NEXT_PUBLIC_API_URL` is provided, use it
- otherwise, use a relative API base such as `""` and fetch `/api/...` from the current origin

- [ ] **Step 2: Review streaming and browser-only usage to ensure the new API base is safe**

Confirm that:

- ordinary JSON API calls still work in development
- SSE/chat streaming calls still resolve correctly from the browser
- the change does not break local frontend development with `NEXT_PUBLIC_API_URL=http://localhost:8422`

- [ ] **Step 3: Add or update frontend tests if needed**

Add targeted coverage if the API helper behavior is currently untested and the same-origin switch could regress client behavior.

- [ ] **Step 4: Run the frontend verification path**

Run:

```bash
cd console/web
npm run lint
npm test
npm run build
```

- [ ] **Step 5: Commit the frontend deployment adjustment**

```bash
git add console/web/src/lib/api.ts console/web/next.config.ts
git commit -m "fix: support same-origin console api in docker mode"
```

### Task 4: Document The Docker Deployment Model And Operator Rules

**Files:**
- Modify: `README.md`
- Modify: `docs/console/overview.md`
- Create: `docs/console/docker.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the root README to introduce the supported Docker path**

Document:

- `agiwo-console serve` as host mode
- `agiwo-console container up` as container mode
- the one-port access story
- the unified data-root model
- explicit host mounts for agent-visible directories

- [ ] **Step 2: Expand `docs/console/overview.md` with the two startup modes**

Clarify:

- host mode versus container mode
- what runs inside the container
- why host directories are not visible by default
- why `host` networking is advanced-only

- [ ] **Step 3: Add a dedicated Docker guide**

Create `docs/console/docker.md` with:

- prerequisites
- example `container up` command
- data-root explanation
- mount examples
- environment variable guidance
- troubleshooting for image pull failures, health-check failures, and mount mistakes

- [ ] **Step 4: Update AGENTS.md only if stable startup or deployment guidance materially changes**

Keep AGENTS at the directory/API boundary level. Only add Docker-related notes if they are truly stable repository rules rather than implementation details.

- [ ] **Step 5: Commit the documentation pass**

```bash
git add README.md docs/console/overview.md docs/console/docker.md AGENTS.md
git commit -m "docs: add console docker deployment guide"
```

### Task 5: Add Continuous Validation For The Official Docker Path

**Files:**
- Modify: `.github/workflows/ci.yml`
- Test: CI-equivalent local checks where feasible

- [ ] **Step 1: Add Docker image build validation to CI**

At minimum, CI should prove the official image builds successfully.

- [ ] **Step 2: Add a lightweight runtime smoke path**

The smoke path should:

1. build the image
2. start the container with a temporary data directory
3. wait for the health endpoint
4. verify the root route is reachable on the same port
5. verify the container exits cleanly on teardown

- [ ] **Step 3: Keep the coverage bounded**

Do not make CI responsible for validating every optional runtime tool in an expensive end-to-end job. Keep the smoke path focused on the public deployment contract, and reserve deeper tool-presence checks for a smaller dedicated build-validation step if needed.

- [ ] **Step 4: Run the repository checks that match the touched surface**

Run:

```bash
uv run python scripts/lint.py ci
cd console && uv run pytest server/tests/test_cli.py server/tests/test_docker_runtime.py -v
cd console/web && npm run lint && npm test && npm run build
```

If the Docker smoke path is scriptable locally, run that too before finalizing.

- [ ] **Step 5: Commit the CI coverage**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: validate console docker deployment path"
```

## Final Verification Checklist

- [ ] `agiwo-console serve` still works as before
- [ ] `agiwo-console container up` starts the managed container and waits for health
- [ ] backend and frontend are reachable through the same public origin
- [ ] Agent persistence defaults under `/data/root`
- [ ] explicit mounts appear under `/mnt/host/<alias>`
- [ ] undeclared host directories remain inaccessible
- [ ] frontend API calls work with same-origin deployment
- [ ] docs match the supported command surface
