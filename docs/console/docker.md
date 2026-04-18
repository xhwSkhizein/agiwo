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

## Notes

- `--network-mode host` is treated as an advanced option and is only supported as a first-class path on Linux.
- `NEXT_PUBLIC_API_URL` is optional in Docker mode because the Web UI uses same-origin API access by default.
- If `container up` fails its health check, inspect `agiwo-console container logs` before removing the container.
