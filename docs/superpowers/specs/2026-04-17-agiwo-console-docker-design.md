# Agiwo Console Docker Deployment Design

**Goal:** Add a first-class Docker deployment path for `agiwo-console` so users can start a complete Console instance, with the API server, Web UI, Agent runtime, Bash execution, and persistence all running inside a managed container through a single CLI workflow.

## Scope

This design covers the first supported Docker deployment experience for `agiwo-console`.

It includes:

1. a full Console container image that serves both the FastAPI backend and the Web UI
2. a new `agiwo-console container ...` CLI surface that manages the Docker container lifecycle
3. a fixed container filesystem contract with one persistent data root and explicit host-directory mounts
4. an official runtime image that includes the tools needed for common Agent execution paths
5. Docker-focused documentation, validation, and operational guardrails

It does not include:

- replacing the existing host-mode `agiwo-console serve` flow
- adding Docker lifecycle controls to the Web UI
- introducing `docker compose` as the primary launch path
- allowing implicit access to arbitrary host directories from inside the container
- building multiple runtime image flavors in the first release

## Context

The current Console startup model is intentionally thin:

- `agiwo-console serve` starts the FastAPI app through `uvicorn.run(...)`
- the Web UI is started separately from `console/web`
- runtime configuration is still driven by the existing `AGIWO_*`, `AGIWO_CONSOLE_*`, and provider credential environment variables
- Agent execution, Bash tool execution, workspace state, and SQLite-backed persistence all currently run in the local process environment

That is workable for source-based development, but it is not a one-command deployment story. It leaves users to assemble their own runtime environment, install system tools, decide where state lives, and manually keep the backend and frontend aligned.

The requested Docker support is not just "put the server in a container". The intended behavior is:

- users launch Console through a single deployment entrypoint
- the Agent runtime lives inside the container
- Bash and related tool execution stay inside the container boundary
- persistence lives under a user-owned mounted data directory
- host filesystem access is opt-in through explicit mounts only
- Web UI access is included in the same Dockerized deployment unit

That makes this a deployment-shape feature, not just a packaging tweak.

## Decision

Ship a first-party Docker deployment mode for `agiwo-console` with these product-level rules:

1. `agiwo-console serve` remains the host-mode startup path
2. `agiwo-console container ...` becomes the supported Docker lifecycle interface
3. the official Docker deployment unit is one image and one container that serves both backend and Web UI
4. all default persistent state lives under one mounted data root inside the container
5. host directory access is denied by default and only enabled through explicit user-provided mounts
6. the first official image is a single full-feature runtime image rather than a minimal base image

## Alternatives Considered

### 1. Backend-Only Containerization

Containerize only `console/server` and keep the Web UI outside the first release.

Why not chosen:

- it falls short of the stated one-command deployment goal
- it still exposes frontend startup and API wiring details to the user
- it creates a hybrid deployment story instead of a complete supported shape

### 2. CLI Wrapper Around External Docker Files Only

Ship Dockerfile and docs, but keep the CLI as a thin helper around user-managed Docker invocations.

Why not chosen:

- it creates two competing "official" startup paths
- container naming, path mapping, health checks, and diagnostics become inconsistent
- it weakens the product-level ergonomics of the new deployment mode

### 3. `docker compose` As The First-Class Launch Path

Generate or manage Compose files and use them as the default deployment interface.

Why not chosen for the first release:

- it adds another artifact format and lifecycle to maintain
- the requested feature can be satisfied cleanly with direct `docker run`
- the current Console CLI already has a command-oriented startup model, so extending that surface is simpler

## Deployment Unit

The first Docker deployment should be a complete Console unit, not a backend-only runtime.

The official image should include:

- the Python Console server
- the production-built Next.js Web UI
- the Agent runtime dependencies and common CLI tools
- a container entrypoint that starts the required internal processes

The official container should expose one external port and one browser entrypoint. Users should not have to think in terms of separate frontend and backend ports during the supported Docker flow.

Internally, the container should run three logical pieces:

- the FastAPI backend
- the Web UI server
- a lightweight reverse proxy that exposes one public port and routes traffic to the correct internal service

Routing contract:

- `/api/*`, health checks, and SSE traffic go to the FastAPI backend
- all non-API browser routes go to the Web UI

This keeps the public surface simple while preserving the existing application boundaries.

## CLI Command Surface

The Docker path should extend the existing `agiwo-console` CLI rather than creating a new executable.

Recommended commands:

- `agiwo-console serve`
- `agiwo-console container up`
- `agiwo-console container down`
- `agiwo-console container status`
- `agiwo-console container logs`
- `agiwo-console container restart`

`serve` remains the host-mode path.

`container up` is the primary deployment command. It should accept a tightly scoped set of arguments that describe deployment intent rather than exposing arbitrary Docker pass-through behavior.

Recommended `up` arguments:

- `--name` for the managed container name, defaulting to something stable such as `agiwo-console`
- `--image` to override the image tag
- `--data-dir` for the required persistent host data root
- repeated `--mount <source>:<alias>` options for explicit host directory mounts
- `--env-file` for provider credentials and runtime configuration
- repeated `--env KEY=VALUE` overrides
- `--publish` for port mapping, defaulting to `8422:8422`
- `--network-mode` with `bridge` as the default and `host` as an advanced option
- `--pull` to optionally refresh the image before startup
- `--replace` to remove and recreate an existing managed container of the same name

The CLI should translate those arguments into a controlled `docker run` invocation.

Default behavior for the first release:

- detached startup
- `--restart unless-stopped`
- port mapping `8422:8422`
- host data directory mounted to `/data`
- each declared host mount mapped to `/mnt/host/<alias>`

The CLI should remain opinionated. It should not become a generic Docker argument tunnel in the first release.

## Filesystem And Path Model

The Docker design should split filesystem visibility into two explicit classes.

### 1. Persistent Console-Owned Data Root

The user provides one host data directory. The CLI mounts it to `/data` inside the container.

All default persistent state should derive from that mounted root, including:

- SDK root path
- SQLite-backed persistence
- trace and metadata state
- agent workspace state
- runtime logs
- a default location for user-managed custom skills

Recommended internal layout:

- `/data/root` as the effective `AGIWO_ROOT_PATH`
- `/data/root/skills` as the default custom skills location
- `/data/runtime` for deployment-owned runtime outputs such as logs or auxiliary state

This keeps all first-party persistent state under one obvious host-owned directory.

### 2. Explicit Host Workspace Mounts

Host filesystem access for Agent work must be opt-in.

Rules:

- the container does not automatically see the current working directory
- the container does not automatically see the user home directory
- every host directory that should be visible to Agent tools must be declared explicitly through `--mount`

Each explicit mount should appear inside the container as:

- `/mnt/host/<alias>`

This gives the runtime a stable and documented container-path convention while keeping the host boundary explicit.

### Path Semantics

In Docker mode, Console users are working with container paths, not raw host absolute paths.

That must be reflected consistently in:

- docs
- CLI help text
- error messages
- any Console UI surfaces that display configured skill or workspace paths

The design should avoid ambiguous mixed semantics such as showing a host path in one place and requiring a container path elsewhere.

## Runtime Image

The first official image should prioritize deployability over minimal size.

The image should include at least:

- Python
- Node.js and npm
- git
- bash and common shell utilities
- `curl` and `wget`
- `ripgrep`
- `jq`
- `ffmpeg`
- `yt-dlp`

This set matches the stated requirement that Agent execution inside the container should feel like a usable runtime rather than a stripped API appliance.

The first release should ship one full-feature official image. It should not split into `base`, `media`, or similar variants yet.

## Container Process Model

The container should use a multi-stage build and a fixed startup entrypoint.

Recommended internal service shape:

- FastAPI backend listens on an internal loopback port, for example `127.0.0.1:18080`
- Web UI listens on an internal loopback port, for example `127.0.0.1:13000`
- a lightweight reverse proxy listens on the public container port `8422`

The entrypoint should:

1. verify that `/data` exists and is writable
2. create required subdirectories such as `/data/root` and `/data/runtime`
3. apply the container-mode default environment contract
4. start backend, frontend, and reverse proxy
5. exit the container if any required subprocess dies unexpectedly

The reverse proxy should be the only externally visible listener. That keeps the public contract simple and decouples browser access from the internal service topology.

## Configuration Model

Docker mode should not invent a second business configuration system.

### Existing Config Remains Canonical

The application inside the container should continue to read:

- `AGIWO_*` for SDK settings
- `AGIWO_CONSOLE_*` for Console settings
- provider credentials such as `OPENAI_API_KEY`

The backend and runtime should not care whether they were started through host mode or Docker mode.

### CLI Deployment Parameters Stay Separate

Deployment parameters such as:

- `--data-dir`
- `--mount`
- `--publish`
- `--network-mode`
- `--name`
- `--replace`

belong to the host-side CLI orchestration layer. They should influence Docker startup, not become new application configuration objects.

### Minimal Bridge Defaults

The CLI may inject a small number of container-specific defaults when starting the official container.

The main required default is:

- `AGIWO_ROOT_PATH=/data/root`

Beyond that, the design should minimize new bridge variables. Docker mode is a deployment carrier for the existing config system, not a new settings hierarchy.

### Frontend API Base Handling

The Web UI should prefer same-origin API access in Docker mode instead of relying on a hard-coded absolute default such as `http://localhost:8422`.

This change is important even outside Docker packaging, because the supported container deployment exposes frontend and backend behind one external origin. The browser-facing API strategy should align with that deployment contract.

## Error Handling

The CLI should validate deployment preconditions before attempting `docker run`.

Required preflight checks:

- `docker` binary exists
- the Docker daemon is reachable
- the declared data directory is created automatically when missing, then verified writable
- the data directory is writable
- each `--mount` source path exists
- each mount alias is unique and matches a restricted safe pattern such as `[A-Za-z0-9._-]+`
- `--network-mode=host` is rejected when unsupported on the current platform
- container name conflicts fail clearly unless `--replace` is set

Startup success for `container up` should mean more than "Docker returned exit code 0". The CLI should wait for the managed container to become healthy through the public health endpoint.

Recommended success condition:

- container is running
- `GET /api/health` succeeds within a bounded timeout

On health failure, the CLI should:

- return a non-zero exit code
- print a short diagnostic summary
- include recent container logs or a targeted tail
- leave the failed container in place for inspection rather than silently removing it

## Platform Behavior

The default network mode should be bridge plus explicit port publishing.

That means the first supported path is effectively:

- `-p 8422:8422`

`host` networking should remain available as an advanced option, but not the default. This avoids making the primary Docker path depend on Linux-specific network behavior and keeps the one-command deployment more portable.

## Testing And Validation

Validation should be split across unit, integration, and smoke layers.

### CLI Tests

Add Python tests that cover:

- parser behavior for `container` subcommands
- `docker run` argument construction
- validation of `--data-dir`, `--mount`, `--replace`, and `--network-mode`
- health-check success and failure behavior
- stable error handling when Docker or the target container is unavailable

### Image Build Validation

The repository should validate that the Docker image builds and contains the required runtime tools, including:

- Python
- Node.js
- `ffmpeg`
- `yt-dlp`
- git
- `ripgrep`
- `jq`

### Runtime Integration Validation

At least one automated integration path should verify:

1. `agiwo-console container up` starts the official image
2. `GET /api/health` succeeds through the public port
3. the browser-facing root route is reachable from the same public port
4. default persistence resolves under `/data/root`
5. explicit host mounts appear under `/mnt/host/<alias>`
6. undeclared host paths are not visible inside the container

### Documentation Smoke

At least one documented Docker startup command should be exercised in CI or a release smoke workflow so the public docs do not drift from the supported path.

## Documentation Changes

The Docker deployment mode should update the live documentation set, at minimum:

- `README.md`
- `docs/console/overview.md`
- a dedicated Docker deployment guide for Console

The docs should describe:

- the difference between host mode and container mode
- the single data-root model
- the explicit host-mount model
- the one-port browser access contract
- the official CLI commands for container lifecycle management
- platform caveats for `--network-mode=host`

## Acceptance Criteria

The design is complete when the implementation delivers all of the following:

1. users can start a complete Console deployment through one `agiwo-console container up ...` command
2. the deployment exposes one browser-facing origin and one default public port
3. Agent and Bash execution run inside the container rather than on the host
4. all default persistent state is rooted under the user-provided data directory
5. host filesystem access requires explicit user-provided mounts
6. the existing `agiwo-console serve` host-mode startup path still works

## Implementation Outline

The implementation should proceed in three stages.

### Stage 1: Docker Runtime Artifact

- add the official Dockerfile and container entrypoint
- build and serve the Web UI inside the image
- add the reverse proxy and internal process wiring
- establish the `/data` and `/mnt/host/*` path conventions

### Stage 2: CLI Container Lifecycle

- extend `console/server/cli.py` with the `container` command group
- implement `up`, `down`, `status`, `logs`, and `restart`
- add deployment preflight validation and post-start health checks
- keep `serve` unchanged as the host-mode path

### Stage 3: Documentation And Validation

- add Docker deployment documentation
- update existing Console startup docs to explain both modes clearly
- add CLI tests, image validation, and runtime smoke coverage
- ensure the repository can continuously validate the public Docker path
