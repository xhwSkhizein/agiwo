# Console API Reference

Base URL: `http://localhost:8422`

## Health

### `GET /api/health`

Health check endpoint.

**Response:**
```json
{"status": "ok"}
```

## Agents

### `GET /api/agents`

List all configured agents.

**Response:**
```json
[
  {
    "id": "agent-abc123",
    "name": "assistant",
    "description": "A helpful assistant",
    "model_provider": "openai",
    "model_name": "gpt-4o",
    "created_at": "2026-03-17T10:00:00Z"
  }
]
```

### `POST /api/agents`

Create a new agent configuration.

**Request:**
```json
{
  "name": "researcher",
  "description": "Research specialist",
  "system_prompt": "You are thorough and cite sources.",
  "model_provider": "openai",
  "model_name": "gpt-4o",
  "temperature": 0.7
}
```

### `GET /api/agents/{agent_id}`

Get a specific agent configuration.

### `PUT /api/agents/{agent_id}`

Update an agent configuration (full replace).

### `DELETE /api/agents/{agent_id}`

Delete an agent configuration.

## Chat

### `POST /api/chat/{agent_id}`

Send a message to an agent and receive a streaming response via SSE.

**Request:**
```json
{
  "message": "What is the capital of France?",
  "session_id": "optional-session-id"
}
```

**Response (SSE stream):**
```
data: {"delta": {"content": "The"}, "type": "content"}
data: {"delta": {"content": " capital"}, "type": "content"}
data: {"delta": {"content": " is"}, "type": "content"}
data: {"delta": {"content": " Paris."}, "type": "content"}
data: {"type": "done"}
```

## Scheduler

### `GET /api/scheduler/states`

List all scheduler agent states.

**Response:**
```json
[
  {
    "id": "agent-abc123",
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
]
```

### `GET /api/scheduler/states/{state_id}`

Get details for a specific scheduler state.

### `GET /api/scheduler/states/{state_id}/children`

List direct child states.

### `GET /api/scheduler/states/{state_id}/pending-events`

List pending mailbox/events for a state.

### `GET /api/scheduler/stats`

Get aggregate counts for `pending/running/waiting/idle/queued/completed/failed`.

### `POST /api/scheduler/states/{state_id}/cancel`

Cancel a running scheduler agent.

### `POST /api/scheduler/states/{state_id}/steer`

Send steering input to a running agent.

**Request:**
```json
{
  "message": "Focus on cost analysis instead"
}
```

## Traces

### `GET /api/traces`

List execution traces.

**Query parameters:**
- `limit` — Max results (default: 50)
- `agent_id` — Filter by agent

**Response:**
```json
[
  {
    "trace_id": "trace-123",
    "agent_id": "agent-abc123",
    "run_id": "run-456",
    "status": "completed",
    "started_at": "2026-03-17T10:00:00Z",
    "duration_ms": 5420,
    "total_tokens": 1523,
    "steps": 3
  }
]
```

### `GET /api/traces/{trace_id}`

Get detailed trace information including all steps, tool calls, and LLM interactions.

## Sessions

### `GET /api/sessions`

List active sessions.

### `GET /api/sessions/{session_id}`

Get session details and history.
