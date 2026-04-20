# Console Source Deploy Script Design

**Goal:** Add a repo-local deployment shortcut for `agiwo-console` that builds the Console Docker image from the current source tree and starts the managed container with an explicit `--env-file <path>` input.

**Architecture:** Keep image build ownership in a new repo script and reuse the existing `agiwo-console container up` lifecycle surface for container creation, replacement, health checks, and follow-up operations. The new script is a thin operator shortcut, not a replacement for the Console CLI or Docker runtime layer.

**Tech Stack:** Bash, Docker, `uv`, existing `agiwo-console` CLI, existing Console Docker image at `console/Dockerfile`

## Scope

This design covers a source-repo deployment shortcut for Docker mode only.

It includes:

1. a new `scripts/deploy_console.sh` helper
2. local image build from `console/Dockerfile`
3. deployment through `uv run --project console agiwo-console container up`
4. explicit `--env-file <path>` support on the script surface
5. minimal documentation for the source-repo deployment path

It does not include:

1. changing the stable `agiwo-console serve` host-mode path
2. replacing `agiwo-console container status|logs|restart|down`
3. adding a second deployment implementation that shells out to raw `docker run`
4. supporting ad hoc inline env overrides beyond the existing `--env-file` path

## Current State

The repository already has:

1. a stable Console Docker image definition at `console/Dockerfile`
2. a stable managed container lifecycle through `agiwo-console container ...`
3. Docker runtime health-check logic in `console/server/docker_runtime.py`
4. documentation for package-installed Docker deployment in `docs/console/docker.md`

What is still missing is a repo-local shortcut that an operator can run directly from the checked-out source tree without manually remembering the build command and the follow-up `container up` invocation.

## Product Decision

Add a single repo script at `scripts/deploy_console.sh`.

That script should:

1. validate required local prerequisites
2. build a local image from the current repository state unless told not to
3. invoke `uv run --project console agiwo-console container up`
4. pass through deployment intent such as `--env-file`, `--data-dir`, `--name`, `--publish`, and repeated `--mount`
5. print the follow-up lifecycle commands the operator should use after deployment

The script is intentionally narrow. It should act as a convenient deployment entrypoint for the source tree, not a new generic deployment framework.

## Interface

The script should expose this shape:

```bash
scripts/deploy_console.sh \
  --env-file /path/to/.env \
  --data-dir /path/to/data \
  [--name agiwo-console] \
  [--image agiwo-console:local] \
  [--publish 8422:8422] \
  [--mount /host/path:alias] \
  [--mount /another/path:alias] \
  [--no-build] \
  [--pull]
```

### Required arguments

- `--env-file <path>`: path to the environment file consumed by the managed container
- `--data-dir <path>`: persistent host directory mounted to `/data`

### Optional arguments

- `--name <container-name>`: defaults to `agiwo-console`
- `--image <tag>`: defaults to a local build tag such as `agiwo-console:local`
- `--publish <host:container>`: defaults to `8422:8422`
- `--mount <source:alias>`: may be repeated and is forwarded unchanged
- `--no-build`: skip image build and use the existing local tag
- `--pull`: forward to `container up` for image refresh behavior

### Behavioral default

The script should always pass `--replace` to `container up`.

Reasoning:

1. this script represents "deploy the current source tree"
2. redeploying the same container name should update the running instance by default
3. avoiding `--replace` would make repeat deployments fail on a pre-existing container, which is the wrong operator experience for a shortcut deploy command

## Flow

The script should execute this sequence:

1. resolve the repository root relative to the script path
2. verify `console/Dockerfile` exists
3. verify `docker` is available in `PATH`
4. verify `uv` is available in `PATH`
5. verify `--env-file` was provided and the file exists
6. verify `--data-dir` was provided
7. create `--data-dir` with `mkdir -p`
8. build the image with `docker build -f console/Dockerfile -t <image> .` unless `--no-build` is set
9. run `uv run --project console agiwo-console container up ... --replace`
10. print the public URL and follow-up lifecycle commands

## Ownership Boundaries

### `scripts/deploy_console.sh`

Owns:

1. operator-facing convenience defaults
2. local prerequisite checks
3. local image build from source
4. forwarding validated arguments to the existing CLI

Does not own:

1. Docker container lifecycle semantics
2. managed health-check polling
3. mount validation rules
4. container restart, logs, status, or teardown behavior

### Existing `agiwo-console container up`

Continues to own:

1. container creation and replacement
2. Docker availability checks
3. mount parsing and validation
4. publish/network handling
5. startup health verification

This split preserves the current architecture boundary: the new script is a repo convenience layer above the existing stable container interface.

## Error Handling

The script should fail fast in these cases:

1. `docker` is missing
2. `uv` is missing
3. `console/Dockerfile` is missing
4. `--env-file` is omitted
5. `--env-file` points to a non-existent path
6. `--data-dir` is omitted
7. `docker build` fails
8. `agiwo-console container up` fails

The script should not:

1. guess fallback env file paths
2. silently continue after a failed build
3. swallow `container up` errors

When `container up` fails, the script should preserve the non-zero exit code and let the existing CLI error text surface directly.

## Output

The script output should stay terse and operator-focused.

Expected high-signal messages:

1. build start
2. build success
3. deployment start
4. deployment success
5. public URL such as `http://localhost:8422`
6. follow-up commands for:
   - `container status`
   - `container logs`
   - `container down`

The script should not print a verbose derived config dump unless there is a failure.

## Documentation

Update `docs/console/docker.md` with a new source-repo deployment example that uses:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data"
```

The documentation should make two points explicit:

1. this path is for operators deploying from a cloned repository
2. post-deploy lifecycle commands still use `uv run --project console agiwo-console container ...`

## Testing

Minimum verification for this change:

1. `uv run python scripts/lint.py ci`
2. a script-level smoke run against a temporary env file and temporary data directory
3. verify the managed container serves `GET /api/health`

The smoke run may reuse the existing local Docker runtime and does not require introducing a new long-lived test fixture.

## Acceptance Criteria

1. Running `scripts/deploy_console.sh --env-file <path> --data-dir <path>` from the repo root builds the Console image and starts a healthy managed container.
2. Re-running the same command replaces the existing container instead of failing on name collision.
3. The script does not implement its own `docker run` deployment path.
4. The script works with repeated `--mount <source:alias>` options.
5. The script prints the URL and the follow-up lifecycle commands after a successful deployment.
6. `docs/console/docker.md` documents the source-repo shortcut.
