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

### `GET /api/agents/tools/available`

List all available built-in tools that can be assigned to agents.

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

### `POST /api/chat/{agent_id}/cancel`

Cancel a running SSE chat for a specific agent.

## Chat Sessions

Chat-level session management (tied to a specific agent).

### `GET /api/chat/{agent_id}/sessions`

List sessions for a specific agent.

### `POST /api/chat/{agent_id}/sessions/create`

Create a new session for an agent.

### `POST /api/chat/{agent_id}/sessions/switch`

Switch the current session for an agent.

### `POST /api/chat/{agent_id}/sessions/{session_id}/fork`

Fork an existing session to create a new branch.

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

### `POST /api/scheduler/states/create`

Submit a new task to the scheduler.

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

### `POST /api/scheduler/states/{state_id}/resume`

Resume a persistent (parked) agent.

## Runs

### `GET /api/runs`

List runs with optional filtering.

**Query parameters:**
- `user_id` — Filter by user ID
- `session_id` — Filter by session ID
- `limit` — Max results (default: 20, max: 200)
- `offset` — Pagination offset (default: 0)

### `GET /api/runs/{run_id}`

Get a single run by ID.

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

List sessions by aggregating runs (with pagination).

**Query parameters:**
- `limit` — Max results (default: 20, max: 200)
- `offset` — Pagination offset (default: 0)

### `GET /api/sessions/{session_id}/summary`

Get full aggregated metrics for a specific session.

### `GET /api/sessions/{session_id}/steps`

Get all steps for a session.

**Query parameters:**
- `agent_id` — Filter by agent ID
- `limit` — Max results (default: 1000, max: 5000)
