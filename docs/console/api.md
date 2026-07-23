# Console API Reference

Base URL: `http://localhost:8422`

User-facing chat goes **Session → `SessionGateway` → root Run** (ADR 0048). There is no Objectives API, assignment templates, or Objective SSE.

## Health

### `GET /api/health`

```json
{"status": "ok", "service": "agiwo-console"}
```

## Overview

### `GET /api/overview`

Dashboard aggregates: sessions, agents, traces, tokens, scheduler status counts.

## Agents

### `GET /api/agents`

List saved agent configurations.

### `GET /api/agents/capabilities`

Supported model providers and capability hints (`requires_base_url`, `requires_api_key_env_name` for compatible providers).

### `GET /api/agents/tools/available`

Functional tools assignable to agents (built-in + `agent:<id>`). Query: `exclude` — agent ID to omit from agent-as-tool references.

### `GET /api/agents/skills/available`

Globally discovered skills (`name`, `description`).

### `POST /api/agents`

Create an agent configuration.

**Request:**
```json
{
  "name": "researcher",
  "description": "Research specialist",
  "system_prompt": "You are thorough and cite sources.",
  "model_provider": "openai",
  "model_name": "gpt-5.4",
  "allowed_tools": ["bash", "web_search"],
  "allowed_skills": ["brainstorming"],
  "options": {
    "max_steps_per_run": 60,
    "run_timeout": 900
  },
  "model_params": {
    "temperature": 0.7,
    "max_output_tokens": 4096
  }
}
```

### `GET /api/agents/{agent_id}`

Get one agent configuration.

### `PUT /api/agents/{agent_id}`

Replace an agent configuration (same body shape as create).

### `DELETE /api/agents/{agent_id}`

Delete an agent configuration (204).

### `GET /api/agents/{agent_id}/sessions`

List sessions whose base agent is `agent_id`.

### `POST /api/agents/{agent_id}/sessions`

Create a standalone session for an agent (201).

**Response:**
```json
{
  "session_id": "session-123",
  "source_session_id": null
}
```

## Sessions

Session ID doubles as the persistent scheduler root state ID.

### `GET /api/sessions`

List sessions. Query: `limit`, `offset`, `include_archived` (default `false`).

### `GET /api/sessions/{session_id}`

Session detail (base agent binding, summary fields).

### `POST /api/sessions/{session_id}/input`

Send user input via **`SessionGateway`**: append Session history, then start or inject a root Run. Optional header: `Idempotency-Key` (generated if omitted).

**Request:**
```json
{
  "message": "What changed between the two implementations?"
}
```

**Response (SSE):**

Success — single `session_turn` event, then stream ends:

```text
event: session_turn
data: {"kind":"session","session_id":"...","run_id":"...","status":"completed","response":"..."}
```

Error:

```text
event: session_error
data: {"message":"..."}
```

Feishu and Console web use the same gateway path.

### `POST /api/sessions/{session_id}/cancel`

Cancel the scheduler root bound to the session.

**Request:**
```json
{
  "reason": "Cancelled by operator"
}
```

### `POST /api/sessions/{session_id}/fork`

Fork into a new session. `context_summary` is stored on the new session and consumed once on the first user input (injected as context, not as the user message).

**Request:**
```json
{
  "context_summary": "Keep the current research context, but branch into pricing analysis."
}
```

**Response:**
```json
{
  "session_id": "new-session-id",
  "source_session_id": "original-session-id"
}
```

### `POST /api/sessions/{session_id}/archive`

Archive a session: cancel any active root run, wait up to 30s for drain, then set `archived_at`. Returns 409 if the root is still active after the timeout.

### `POST /api/sessions/{session_id}/restore`

Clear `archived_at` (does not restart runs).

### `GET /api/sessions/{session_id}/summary`

Aggregated summary metrics for one session.

### `GET /api/sessions/{session_id}/steps`

Session steps. Query: `start_seq`, `end_seq`, `run_id`, `agent_id`, `limit` (max 5000), `order` (`asc`|`desc`).

## Runs

### `GET /api/runs`

List runs. Query: `user_id`, `session_id`, `limit`, `offset`.

### `GET /api/runs/{run_id}`

Get one run by ID.

## Scheduler

Debug / operator surface. Ordinary user chat uses `/api/sessions/{id}/input`, not scheduler routes.

### `GET /api/scheduler/states`

List scheduler states. Query: `status`, `limit`, `offset`.

### `GET /api/scheduler/states/{state_id}`

Get one scheduler state.

### `GET /api/scheduler/states/{state_id}/children`

Direct child states.

### `GET /api/scheduler/states/{state_id}/tree`

Scheduler tree rooted at `state_id` (max 500 nodes).

### `GET /api/scheduler/states/{state_id}/pending-events`

Pending mailbox events for a state.

### `GET /api/scheduler/stats`

Aggregate counts: `pending`, `running`, `waiting`, `idle`, `queued`, `completed`, `failed`, `total`.

### `POST /api/scheduler/states/create`

Create and submit a persistent root from an agent config.

**Request:**
```json
{
  "agent_config_id": "agent-abc",
  "initial_task": "Research topic X",
  "session_id": null
}
```

`agent_config_id` is required. `session_id` defaults to a new UUID.

### `POST /api/scheduler/states/{state_id}/cancel`

Cancel a state and its descendants.

**Request:**
```json
{
  "reason": "Cancelled by operator"
}
```

### `POST /api/scheduler/states/{state_id}/resume`

Enqueue input for a persistent root (`Scheduler.enqueue_input`: next cycle, live Loop, or USER_HINT).

**Request:**
```json
{
  "message": "Continue with the next step"
}
```

## Traces

### `GET /api/traces`

List traces. Query: `agent_id`, `session_id`, `user_id`, `status`, `limit`, `offset`.

### `GET /api/traces/{trace_id}`

Trace detail including full span tree.

## Runtime Config

Process-local overrides; restart reverts to environment config.

### `GET /api/config/runtime`

Current runtime config snapshot.

### `PUT /api/config/runtime`

Replace editable runtime overrides for the current process.

## Feishu

### `GET /api/channels/feishu/status`

Feishu long-connection status. Returns `{"enabled": false}` when the channel is not wired.
