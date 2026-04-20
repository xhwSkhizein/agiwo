# Console API Reference

Base URL: `http://localhost:8422`

## Health

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{"status": "ok", "service": "agiwo-console"}
```

## Overview

### `GET /api/overview`

Dashboard aggregates for sessions, agents, traces, tokens, and scheduler status.

## Agents

### `GET /api/agents`

List all configured agents.

### `GET /api/agents/capabilities`

List supported model providers plus capability hints such as whether `base_url` or `api_key_env_name` is required.

### `GET /api/agents/tools/available`

List available functional tools that can be assigned to agents, including built-in tools and `agent:<id>` references.

### `GET /api/agents/skills/available`

List globally discovered skills.

### `POST /api/agents`

Create a new agent configuration.

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
    "max_steps": 60,
    "run_timeout": 900
  },
  "model_params": {
    "temperature": 0.7,
    "max_output_tokens": 4096
  }
}
```

### `GET /api/agents/{agent_id}`

Get a specific agent configuration.

### `PUT /api/agents/{agent_id}`

Replace an existing agent configuration.

### `DELETE /api/agents/{agent_id}`

Delete an agent configuration.

### `GET /api/agents/{agent_id}/sessions`

List sessions whose base agent is `agent_id`.

### `POST /api/agents/{agent_id}/sessions`

Create a standalone session for an agent.

**Response:**
```json
{
  "session_id": "session-123",
  "source_session_id": null
}
```

## Sessions

### `GET /api/sessions`

List sessions with lightweight summary fields.

### `GET /api/sessions/{session_id}`

Get session detail, including current base agent binding and latest summary fields.

### `POST /api/sessions/{session_id}/input`

Send new user input into a session and receive SSE events.

**Request:**
```json
{
  "message": "What changed between the two implementations?"
}
```

**Response:**

SSE messages use the event type as the SSE `event` field, and `AgentStreamItem.to_dict()` as the JSON payload:

```text
event: run_started
data: {"type":"run_started", ...}

event: step_delta
data: {"type":"step_delta","delta":{"content":"The first approach ..."}}

event: run_completed
data: {"type":"run_completed","response":"...","termination_reason":"completed"}
```

If the session is already attached to a running root and no direct stream is available, the endpoint emits a `scheduler_ack` event instead.

### `POST /api/sessions/{session_id}/cancel`

Cancel the active scheduler root bound to the session.

### `POST /api/sessions/{session_id}/fork`

Fork a session into a new session ID.

**Request:**
```json
{
  "context_summary": "Keep the current research context, but branch into pricing analysis."
}
```

### `DELETE /api/sessions/{session_id}`

Delete a stored session.

### `GET /api/sessions/{session_id}/summary`

Fetch the aggregated summary view for one session.

### `GET /api/sessions/{session_id}/steps`

Fetch session steps. Supports `start_seq`, `end_seq`, `run_id`, `agent_id`, `limit`, and `order`.

## Scheduler

### `GET /api/scheduler/states`

List scheduler states. Supports `status`, `limit`, and `offset`.

**Response:**
```json
{
  "items": [
    {
      "id": "agent-abc123",
      "root_state_id": "agent-abc123",
      "status": "running",
      "task": "Research topic X",
      "parent_id": null,
      "agent_config_id": "config-1",
      "is_persistent": true,
      "depth": 0,
      "wake_count": 1,
      "created_at": "2026-03-17T10:00:00Z",
      "updated_at": "2026-03-17T10:05:00Z"
    }
  ],
  "limit": 50,
  "offset": 0,
  "has_more": false,
  "total": null
}
```

### `GET /api/scheduler/states/{state_id}`

Get details for a specific scheduler state.

### `GET /api/scheduler/states/{state_id}/children`

List direct child states.

### `GET /api/scheduler/states/{state_id}/tree`

Get the scheduler tree rooted at `state_id`.

### `GET /api/scheduler/states/{state_id}/pending-events`

List pending mailbox/events for a state.

### `GET /api/scheduler/stats`

Get aggregate counts for `pending/running/waiting/idle/queued/completed/failed`.

### `POST /api/scheduler/states/create`

Create and submit a new persistent root from an existing agent config.

### `POST /api/scheduler/states/{state_id}/cancel`

Cancel a running scheduler agent.

### `POST /api/scheduler/states/{state_id}/steer`

Send steering input to a scheduler state.

**Request:**
```json
{
  "message": "Focus on cost analysis instead",
  "urgent": false
}
```

### `POST /api/scheduler/states/{state_id}/resume`

Resume a persistent root with a new message.

## Runs

### `GET /api/runs`

List runs with optional filtering.

Supported query parameters:

- `user_id`
- `session_id`
- `limit`
- `offset`

### `GET /api/runs/{run_id}`

Get a single run by ID.

## Traces

### `GET /api/traces`

List execution traces.

Supported query parameters:

- `agent_id`
- `session_id`
- `user_id`
- `status`
- `limit`
- `offset`

### `GET /api/traces/{trace_id}`

Get detailed trace information including the full span tree.

## Runtime Config

### `GET /api/config/runtime`

Inspect the process-local runtime config snapshot used by the Console.

### `PUT /api/config/runtime`

Replace editable runtime config overrides for the current process.

## Feishu

### `GET /api/channels/feishu/status`

Inspect Feishu long-connection status when the channel is enabled.
