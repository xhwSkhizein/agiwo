# Console Overview

The Console is Agiwo's self-hosted control plane. It consists of a FastAPI backend plus a web UI, and is designed for operator-facing workflows rather than end-user SaaS usage.

## Architecture

```text
Console
├── FastAPI Server (console/server/)
│   ├── REST API — agents, sessions, runs, traces, runtime config
│   ├── SSE API — session input streaming
│   ├── Channel Runtime — Feishu
│   └── Services — agent factory/cache, session services, registry, storage wiring
└── Web UI (console/web/)
    ├── Agent Management
    ├── Sessions / Chat
    ├── Scheduler
    └── Traces
```

## Runtime Model

The Console is session-first:

1. create or choose an agent config
2. create a session for that agent
3. send input to the session over SSE
4. let the scheduler decide whether to submit, enqueue, or steer the root runtime

Important runtime services live under `console/server/services/runtime/`:

- `build_agent(...)` constructs stable runtime agents from registry records
- `AgentRuntimeCache` reuses runtime agents across requests
- `SessionContextService` creates and forks sessions
- `SessionRuntimeService` routes session input through the scheduler
- `SessionViewService` builds session list/detail projections for the API
- `SchedulerTreeViewService` assembles scheduler tree responses for the UI

## Quick Start

### Start the API Server (Host Mode)

```bash
pip install agiwo-console
cat > .env <<'EOF'
OPENAI_API_KEY=...
EOF
agiwo-console serve --env-file .env
```

If you are running from the source repository instead, start from `console/.env.example.full`.

The server starts at `http://localhost:8422`.

Useful routes:

- `GET /api/health`
- `GET /api/overview`
- `GET /api/agents`
- `POST /api/agents/{agent_id}/sessions`
- `POST /api/sessions/{session_id}/input`
- `GET /api/scheduler/states`
- `GET /api/traces`

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

The Docker path starts the FastAPI backend, Web UI, Agent runtime, and Bash execution in one managed container. The public entrypoint remains `http://localhost:8422`.

If the agent should see host directories, declare them explicitly:

```bash
agiwo-console container up \
  --data-dir "$HOME/agiwo-data" \
  --env-file .env \
  --mount "$HOME/projects:projects"
```

Mounted host directories appear inside the container as `/mnt/host/<alias>`. Undeclared host paths are not visible to the agent runtime.

### Start the Web UI

```bash
cd console/web
npm install
echo 'NEXT_PUBLIC_API_URL=http://localhost:8422' > .env.local
npm run dev
```

The UI starts at `http://localhost:3000`.

## Current Positioning

- recommended use: internal or self-hosted operator workflows
- built-in channel integrations today: Feishu only
- production readiness: not yet positioned as a production end-user product

## Configuration

Console-specific settings use the `AGIWO_CONSOLE_*` prefix. SDK settings (`AGIWO_*`) and provider credentials (`OPENAI_API_KEY`, etc.) are also read by the Console.

## Features

- create and replace agent configs through the registry API or web UI
- inspect available providers, builtin tools, and globally discovered skills
- create standalone sessions per agent and stream new user input over SSE
- browse runs, steps, session summaries, and execution traces
- inspect scheduler states, trees, pending events, and aggregate stats
- cancel, steer, resume, or submit persistent scheduler roots
- optionally enable the Feishu channel runtime

## Extending the Console

### Adding a Channel

1. create a new package in `console/server/channels/your_channel/`
2. reuse `SessionContextService` and `SessionRuntimeService` instead of building a parallel execution path
3. initialize and register the channel service in `console/server/app.py`

### Adding an API Route

1. define request/response models in `console/server/models/`
2. create a router in `console/server/routers/your_router.py`
3. include it in `console/server/app.py`
