# Runtime Tool Boundaries

This document defines the operational boundary between scheduler child-agent
orchestration and shell background process management.

## Scheduler Child Agents

Use `spawn_child_agent` or `fork_child_agent` when work should run as a
scheduler-managed child agent.

The returned child agent ID may be used with:

- `sleep_and_wait(wake_type="waitset", wait_for=[child_id])`
- `query_spawned_agent`
- `cancel_agent`

`sleep_and_wait(wait_for=...)` accepts only direct child agent IDs in the same
scheduler session.

## Bash Background Jobs

Use `bash(background=true)` when a shell command should keep running outside the
current tool call.

The returned `job_id` is a bash process job ID, not a scheduler child agent ID.

The returned `job_id` may be used with:

- `bash_process(action="status", job_id=job_id)`
- `bash_process(action="logs", job_id=job_id)`
- `bash_process(action="paths", job_id=job_id)`
- `bash_process(action="input", job_id=job_id)`
- `bash_process(action="stop", job_id=job_id)`

Do not pass a bash `job_id` to `sleep_and_wait(wait_for=...)`.

## Browser CLI Example

If a browser command is started in the background:

```json
{
  "name": "bash",
  "arguments": {
    "background": true,
    "command": "uv run browser-cli read https://example.com --snapshot"
  }
}
```

the next check should use:

```json
{
  "name": "bash_process",
  "arguments": {
    "action": "status",
    "job_id": "<job_id>"
  }
}
```

and then:

```json
{
  "name": "bash_process",
  "arguments": {
    "action": "logs",
    "job_id": "<job_id>",
    "tail": 200
  }
}
```

Use `sleep_and_wait(wake_type="timer", delay_seconds=...)` only when the agent
intentionally wants to pause and wake itself later. A timer wait does not monitor
bash process completion.

## Browser CLI Upstream Boundary

A `browser-cli status` value such as `degraded` with missing trusted workspace
binding is browser-cli runtime state. Agiwo should report that status and
preserve logs, but should not invent browser-cli-specific recovery semantics
inside scheduler wait handling.
