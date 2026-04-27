# Console Docker Deployment

`agiwo-console container up` starts a complete Console deployment in one managed Docker container.

That container includes:

- the FastAPI backend
- the Web UI
- the Agent runtime
- Bash execution and related runtime tools

The default public entrypoint is `http://localhost:8422`.

## Quick Start

```bash
pip install agiwo-console
cat > .env <<'EOF'
OPENAI_API_KEY=...
EOF
agiwo-console container up \
  --data-dir "$HOME/agiwo-data" \
  --env-file .env
```

## From a Cloned Repository

If you are deploying from this source repository instead of an installed
`agiwo-console` package, use the repo shortcut script:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data"
```

The script builds the current `console/Dockerfile` image and then starts the managed
container through `uv run --project console agiwo-console container up`.

It forwards common container options such as `--mount`, `--env`, `--env-name`,
`--publish`, and `--network-mode`.

## Local Browser CLI Development Build

For deployments that should use a local Browser CLI checkout instead of the
published package version, pass `--browser-cli-source` when using the source
deployment script:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data" \
  --network-mode host \
  --browser-cli-source "$HOME/workspace/browser-cli"
```

The script builds a Browser CLI wheel from that checkout and installs it into
the Console image after Agiwo's normal dependencies. Browser CLI packaged
skills are installed into the default Agent skills directory,
`/data/root/skills`, and refreshed at container startup.

Use `--network-mode host` on Linux when Agents in the Console container need to
operate a host-side browser or Browser CLI extension-connected runtime.

## Data Root

`--data-dir` is the single persistent host directory for the managed container.

The container maps it to `/data` and roots default runtime state under `/data/root`.
When launched through `agiwo-console container up`, the container runs with the invoking host UID/GID so `/data` stays writable without switching back to root.

That includes:

- SQLite-backed Console state
- default agent workspace state
- logs and runtime-owned files
- the default custom skills location under `/data/root/skills`

## Host Directory Mounts

Host directories are not visible to the Agent runtime by default.

To make a host directory available inside the container, pass `--mount <source>:<alias>`:

```bash
agiwo-console container up \
  --data-dir "$HOME/agiwo-data" \
  --env-file .env \
  --mount "$HOME/projects:projects" \
  --mount "$HOME/media:media"
```

Inside the container, those paths appear as:

- `/mnt/host/projects`
- `/mnt/host/media`

## Lifecycle Commands

```bash
agiwo-console container status
agiwo-console container logs
agiwo-console container restart
agiwo-console container down
```

For a source-repo deployment, use the existing CLI for follow-up operations:

```bash
uv run --project console agiwo-console container status
uv run --project console agiwo-console container logs
uv run --project console agiwo-console container restart
uv run --project console agiwo-console container down
```

## Notes

- `--network-mode host` is treated as an advanced option and is only supported as a first-class path on Linux.
- `NEXT_PUBLIC_API_URL` is optional in Docker mode because the Web UI uses same-origin API access by default.
- If `container up` fails its health check, inspect `agiwo-console container logs` before removing the container.
