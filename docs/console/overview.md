# Console Overview

The Console is a self-hosted control plane for managing Agiwo agents. It consists of a FastAPI backend and a Next.js frontend, is currently best suited for internal deployments, and should not yet be treated as a production-ready end-user product.

## Architecture

```
Console
├── FastAPI Server (console/server/)
│   ├── REST API — Agent config CRUD, scheduler state, runs, sessions
│   ├── SSE API — Real-time chat and trace streaming
│   ├── Channel Runtime — Feishu (and extensible to others)
│   └── Services — Agent lifecycle, registry, storage
└── Next.js Web UI (console/web/)
    ├── Agent Management — Create, edit, view agents
    ├── Chat — Interactive chat with agents
    ├── Scheduler — View and manage scheduled agents
    └── Traces — Inspect execution traces
```

## Quick Start

### Start the API Server (Host Mode)

```bash
pip install agiwo-console
cat > .env <<'EOF'
OPENAI_API_KEY=...
EOF
agiwo-console serve --env-file .env
```

If you are running from the source repository instead, you can still start from `console/.env.example.full`.

The server starts at `http://localhost:8422`.

### Start the Complete Console in Docker

```bash
pip install agiwo-console
cat > .env <<'EOF'
OPENAI_API_KEY=...
EOF
agiwo-console container up \
  --data-dir "$HOME/agiwo-data" \
  --env-file .env
```

The Docker path starts the FastAPI backend, Web UI, Agent runtime, and Bash execution in one managed container. The public entrypoint is still `http://localhost:8422`, but the runtime now lives inside the container instead of on the host.

If the Agent should see host directories, declare them explicitly:

```bash
agiwo-console container up \
  --data-dir "$HOME/agiwo-data" \
  --env-file .env \
  --mount "$HOME/projects:projects"
```

Mounted host directories appear inside the container as `/mnt/host/<alias>`. Undeclared host paths are not visible to the Agent runtime.

### Start the Web UI

```bash
cd console/web
npm install

# Point to the API
echo 'NEXT_PUBLIC_API_URL=http://localhost:8422' > .env.local

npm run dev
```

The UI starts at `http://localhost:3000`.

## Current Positioning

- Recommended use: internal/self-hosted operator workflows
- Built-in channel integrations today: Feishu only
- Production readiness: not yet production-ready

## Health Check

```bash
curl http://localhost:8422/api/health
# {"status": "ok"}
```

## Configuration

Console-specific settings use the `AGIWO_CONSOLE_*` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGIWO_CONSOLE_HOST` | `0.0.0.0` | Server bind address |
| `AGIWO_CONSOLE_PORT` | `8422` | Server port |
| `AGIWO_CONSOLE_*` | — | See `console/server/config.py` |

SDK settings (`AGIWO_*`) and provider credentials (`OPENAI_API_KEY`, etc.) are also read by the Console.

## Features

### Agent Management

- Create and configure agents via API or UI
- Store agent configs in SQLite, MongoDB, or memory
- Full replace semantics (not patch/merge)
- List available built-in tools via `GET /api/agents/tools/available`

### Chat

- Interactive chat with agents over SSE
- Real-time streaming responses
- Session management and continuity
- Cancel running conversations

### Scheduler

- View all scheduled agent states
- Monitor running, waiting, and queued agents
- Cancel, steer, or resume agents from the UI
- Submit new tasks via API

### Traces

- Browse execution traces
- Drill into individual steps
- View token usage and costs

### Runs & Sessions

- List runs with filtering by user and session
- View session summaries with aggregated metrics
- Inspect individual steps within sessions

### Channels

- **Feishu**: Built-in integration with Feishu messaging
  - WebSocket connection for real-time messages
  - Command parsing and routing (`/status`, `/new`, `/switch`, etc.)
  - Group and direct message support
  - Message history storage in SQLite or memory

## Extending the Console

### Adding a Channel

1. Create a new package in `console/server/channels/your_channel/`
2. Implement a channel service following the `feishu/service.py` pattern
3. Initialize and register in `console/server/app.py`

### Adding an API Route

1. Define request/response models in `console/server/models/`
2. Create a router in `console/server/routers/your_router.py`
3. Include it in `console/server/app.py`

See the existing Feishu implementation for reference.
