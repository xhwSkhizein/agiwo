# Runtime Tool Surface Hardening Design

## Context

Feishu channel testing exposed two recurring runtime-tool failures:

- `bash` calls failed with `stdin requires pty=true` when the model supplied
  `stdin: ""` together with `pty: false`.
- `spawn_agent` calls failed because the single tool exposed two different
  modes through a `fork` boolean. The model combined `fork=true` with custom
  parameters, then later used tool names as `allowed_skills`.

The Feishu channel and scheduler dispatch path worked correctly. The failures
came from a public tool surface that made invalid parameter combinations easy
for the model to produce.

## Goals

- Make child-agent creation tools harder to misuse by splitting regular spawn
  and fork semantics into separate tools.
- Let regular child-agent spawning explicitly restrict available tools through
  `allowed_tools`.
- Keep `allowed_skills` scoped to skills only.
- Treat empty bash stdin as absent stdin so harmless model-generated
  `stdin: ""` does not require PTY mode.
- Avoid compatibility shims for historical `spawn_agent` calls; old test data
  will be cleared.

## Non-Goals

- Do not migrate existing scheduler state or historical run data.
- Do not preserve `spawn_agent` as an alias.
- Do not split the internal scheduler command model unless required by tests.
- Do not relax bash safety checks for non-empty stdin.

## Public Tool API

### `spawn_child_agent`

Creates a regular child agent for an independent subtask.

Parameters:

- `task: string` required.
- `instruction: string | null` optional child guidance.
- `allowed_tools: list[str] | null` optional explicit functional tool list.
- `allowed_skills: list[str] | null` optional explicit skill list.

Semantics:

- Creates `SpawnChildRequest(..., fork=False)`.
- `allowed_tools` is validated as a child subset of the parent agent's
  `allowed_tools` when the parent has a restriction.
- `allowed_skills` is validated through the global skill manager and remains
  skill-only.
- Child agents cannot spawn further child agents.

### `fork_child_agent`

Creates a forked child agent that inherits parent conversation context.

Parameters:

- `task: string` required.

Semantics:

- Creates `SpawnChildRequest(..., fork=True)`.
- Does not expose `instruction`, `allowed_tools`, or `allowed_skills`.
- Inherits parent runtime context according to the existing fork path.
- Child agents cannot spawn further child agents.

### Removed Tool

`spawn_agent` is removed from the scheduler runtime tool registry. If an old
prompt or stale conversation calls it, the normal unknown-tool path should
surface the error.

## Internal Implementation

The internal scheduler primitive remains `SpawnChildRequest` with `fork: bool`
for this change. That keeps the implementation focused on the public tool
surface while reusing the existing scheduler flow:

- `agiwo.scheduler.commands.SpawnChildRequest`
- `agiwo.scheduler.tool_control.SchedulerToolControl.spawn_child`
- `agiwo.scheduler.runner.SchedulerRunner._ensure_child_agent`

`SpawnChildRequest.allowed_tools` already exists and should be wired from the
new `spawn_child_agent` tool.

The runtime tool module should replace `SpawnAgentTool` with two concrete tool
classes:

- `SpawnChildAgentTool`
- `ForkChildAgentTool`

Both can share a small private helper for common spawn result formatting and
error conversion.

## Bash Stdin Normalization

Normalize bash stdin during parameter parsing:

- Missing `stdin` remains `None`.
- `stdin == ""` becomes `None`.
- Non-string stdin remains a parse error.
- Non-empty stdin with `pty=false` still fails with `stdin requires pty=true`.

This keeps the safety boundary intact while removing a common false-positive
failure caused by model-generated empty strings.

## Error Handling

- Invalid `allowed_tools` still returns a failed tool result with the existing
  child-subset validation message.
- Invalid `allowed_skills` still returns a failed tool result with the existing
  skill validation message.
- Calls to removed `spawn_agent` should not be intercepted or rewritten.
- `fork_child_agent` should not need fork-specific parameter conflict errors
  because its schema does not expose conflicting parameters.

## Tests

Add or update tests for:

- Bash parser normalizes `stdin: ""` to `None`.
- Bash still rejects non-empty stdin when `pty=false`.
- Scheduler runtime tools expose `spawn_child_agent` and `fork_child_agent`,
  and no longer expose `spawn_agent`.
- `spawn_child_agent` passes `allowed_tools` into `SpawnChildRequest` and child
  overrides.
- `fork_child_agent` creates a forked child without custom tool or skill
  parameters.
- Existing child-agent spawn, wait, query, cancel, and list behavior remains
  intact under the new tool names.

## Documentation Updates

Update references in:

- `AGENTS.md`
- multi-agent docs and website docs
- scheduler/runtime-tool tests and expected prompt strings

The docs should describe `spawn_child_agent` and `fork_child_agent` as separate
tools instead of documenting a single `spawn_agent(fork=...)` interface.

