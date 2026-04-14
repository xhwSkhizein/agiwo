# Scheduler Semantics Alignment Design

- Date: 2026-04-14
- Status: Proposed
- Scope: SDK scheduler, scheduler state storage, Console runtime/API models

## Context

Release review found four issues that are individually small but share the same root problem: the scheduler currently mixes lifecycle state, run outcome, and display summary.

Observed problems:

1. `Scheduler.wait_for()` reconstructs results from `AgentState.status` and loses the original `TerminationReason`.
2. Scheduler stream subscription opens too late and can miss the entire run for fast completions.
3. Console default-agent construction treats `allowed_tools=[]` as falsy and silently re-enables default tools.
4. Console runtime agent cache does not invalidate on `allowed_skills` changes and can continue serving stale permissions.

For the first release, we prefer semantic correctness over compatibility with the current flawed behavior. Old scheduler state can be deleted. We do not need backward-compatible persistence or response shims.

## Goals

- Make scheduler-owned result semantics explicit and durable.
- Align `Scheduler.run()` / `wait_for()` behavior with streamed terminal events.
- Remove stream subscriber race conditions for root runs.
- Align Console default-agent permission semantics with SDK `allowed_tools` rules.
- Ensure Console runtime agent cache refreshes when tool or skill permissions change.
- Expose the scheduler's last run outcome directly in Console APIs.

## Non-Goals

- Preserve compatibility with old scheduler state rows or API payload shapes.
- Add scheduler run history. This design stores only the most recent run result.
- Refactor scheduler into a separate run-result subsystem.
- Change core Agent run semantics.

## Options Considered

### Option A: Minimal Patches

Patch `wait_for()` and stream setup locally without changing scheduler state shape.

Pros:

- Smallest code diff.
- Lowest short-term implementation cost.

Cons:

- Does not solve persisted root-state queries or restart-time correctness.
- Keeps result semantics implicit and fragile.
- Leaves Console querying code with no durable source of truth.

### Option B: Persist `last_run_result` on `AgentState`

Add a small scheduler-owned record that captures the most recent agent-cycle outcome and use it as the canonical source for non-streamed result semantics.

Pros:

- Separates scheduler lifecycle from agent-cycle outcome cleanly.
- Works for persistent roots, resume flows, restart-time queries, and Console APIs.
- Keeps the change local to existing scheduler state and storage paths.

Cons:

- Requires state model, storage, serialization, and API changes.
- Intentionally breaks compatibility with old persisted scheduler data.

### Option C: Separate Scheduler Run Result Store

Create a distinct persisted store for scheduler-owned run results.

Pros:

- Cleanest long-term modeling if we later need histories or analytics.

Cons:

- Too heavy for a release-blocking semantics fix.
- Introduces a new subsystem without enough payoff right now.

## Decision

Choose Option B.

The scheduler will continue to own lifecycle state through `AgentState.status`, but it will also persist the most recent run outcome in a dedicated `last_run_result` record. `status` will answer "what is the scheduler doing now?" and `last_run_result` will answer "how did the most recent agent cycle finish?"

## 1. Data Model

Add a new immutable scheduler model:

```python
@dataclass(frozen=True, slots=True)
class SchedulerRunResult:
    run_id: str | None
    termination_reason: TerminationReason
    summary: str | None = None
    error: str | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
```

Add to `AgentState`:

```python
last_run_result: SchedulerRunResult | None = None
```

`result_summary` remains on `AgentState`, but it becomes a scheduler/UI summary field rather than the canonical machine-readable outcome.

Modeling rules:

- `status` is scheduler lifecycle only.
- `last_run_result` is the canonical durable outcome of the most recent completed agent cycle.
- Only one result is stored. New cycles replace old results.

## 2. State Transition Semantics

### New root submission

- New root state starts with `last_run_result=None`.

### Persistent root enqueue

- When a persistent root moves to `QUEUED`, preserve the previous `last_run_result`.
- Rationale: before the next cycle actually starts, callers should still see the previous completed outcome.

### Cycle start

- Any transition to `RUNNING` for a new cycle clears `last_run_result`.
- This includes root queued-input dispatch, child pending dispatch, timeout/event wake, and normal root submit.

### Successful completion

- Write `last_run_result` with:
  - `run_id`
  - `termination_reason=TerminationReason.COMPLETED`
  - `summary=response`
  - `error=None`
- Persistent roots end in `IDLE`.
- Non-persistent roots and children end in `COMPLETED`.

### Failed or cancelled completion

- Write `last_run_result` for:
  - `CANCELLED`
  - `TIMEOUT`
  - `ERROR`
  - `ERROR_WITH_CONTEXT`
- `summary` may carry best-effort output text.
- `error` carries the primary human-readable failure message.

### Sleeping / waiting

- `TerminationReason.SLEEPING` is not a final external run outcome and must not be persisted as `last_run_result`.
- Periodic or waiting intermediate rounds must not overwrite the last durable terminal result.

### Scheduler-originated termination

Scheduler-side failures must also write `last_run_result`, not only `result_summary`. This includes:

- wake rejected by guard
- shutdown before completion
- parent cancelled
- scheduler dispatch failure paths that conclude the cycle

If a direct `TerminationReason` exists, use it. Otherwise map to `ERROR` and preserve details in `error`.

## 3. Read Semantics

### `Scheduler.wait_for()`

`wait_for()` must stop inferring run outcomes from `AgentState.status`.

New behavior:

- Wait until the state has a terminal scheduler status or a root-persistent idle status with a non-`None` `last_run_result`.
- Return a `RunOutput` derived from `last_run_result`.
- If timeout is hit before a result is written, return `TerminationReason.TIMEOUT`.

This makes non-streamed waiting consistent with streamed terminal events.

### Console runtime and APIs

Expose `last_run_result` directly in:

- scheduler state responses
- scheduler state list items
- session detail responses
- any SSE fallback acknowledgment path that currently depends on `result_summary`

The Console should show both:

- current scheduler `status`
- most recent `last_run_result`

These answer different questions and should not be merged.

## 4. Stream Subscription Race Fix

The scheduler stream channel must exist before `submit()`, `enqueue_input()`, or `steer()` begins producing events.

Change:

- `route_with_stream()` opens the channel before invoking `operation()`.
- If `operation()` fails, close the channel before re-raising.
- Keep the single-subscriber invariant.

This removes the fast-run race where the root run can start and finish before the subscriber exists.

## 5. Console Permission Semantics

`build_default_agent_record()` must distinguish:

- `allowed_tools is None`: use default builtin tools
- `allowed_tools == []`: no functional tools

It must not use truthiness to decide fallback behavior.

This aligns Console default-agent behavior with SDK `allowed_tools` semantics documented in `AGENTS.md`.

## 6. Runtime Cache Invalidation

`AgentRuntimeCache` snapshots must include all permission-shaping fields, including `allowed_skills`.

Minimum invalidation set:

- `allowed_tools`
- `allowed_skills`
- `system_prompt`
- `model_provider`
- `model_name`
- `options`
- `model_params`

If a cached session agent becomes stale:

- If scheduler rebind is allowed, replace it immediately.
- If the state is active and rebind is deferred, keep serving the current runtime agent until the state becomes rebindable, then use the refreshed config on the next acquisition path.

The goal is correctness, not hot-reload sophistication.

## 7. Storage and Serialization

Update scheduler persistence for memory and sqlite storage to encode/decode `last_run_result`.

Constraints:

- No migration layer.
- Old rows failing to decode is acceptable for this release.
- Console view models and response serializers must expose the new structure explicitly.

## 8. Testing

Add tests for the following:

### Scheduler

- `wait_for()` returns `COMPLETED`, `CANCELLED`, `TIMEOUT`, and `ERROR` from `last_run_result` rather than guessing from `status`
- persistent root preserves previous `last_run_result` while `QUEUED`
- transitioning to `RUNNING` clears prior `last_run_result`
- scheduler-originated failures write `last_run_result`
- fast root runs do not drop `run_started` / `run_completed` after stream setup changes

### Console

- scheduler and session detail APIs expose `last_run_result`
- SSE fallback uses `last_run_result` semantics
- default-agent `allowed_tools=[]` stays empty
- runtime cache invalidates on `allowed_skills` changes

### Storage

- memory store round-trips `last_run_result`
- sqlite store round-trips `last_run_result`

## 9. Risks and Mitigations

### Risk: accidental mixing of `status` and `last_run_result`

Mitigation:

- keep `wait_for()` and response serializers reading from `last_run_result`
- leave `status` untouched as scheduler lifecycle

### Risk: periodic/waiting flows overwrite durable terminal results incorrectly

Mitigation:

- treat `SLEEPING` as non-terminal for `last_run_result`
- only write terminal outcomes for externally meaningful cycle ends

### Risk: API consumers keep reading `result_summary`

Mitigation:

- expose `last_run_result` explicitly in Console responses
- keep `result_summary` for display only, not machine semantics

## 10. Acceptance Criteria

- `Scheduler.run()` / `wait_for()` return the same terminal semantics a stream consumer would observe.
- Fast runs no longer lose their root stream events due to late subscription.
- `allowed_tools=None` and `allowed_tools=[]` have distinct Console behavior.
- Runtime agent cache refreshes when skill allowlists change.
- Console scheduler and session APIs expose `last_run_result`.
- Tests cover the new semantics across scheduler, Console, and scheduler storage.
