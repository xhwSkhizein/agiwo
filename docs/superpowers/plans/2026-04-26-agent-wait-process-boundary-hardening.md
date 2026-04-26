# Agent Wait/Process Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent agents from confusing scheduler child-agent waits with bash background process jobs, and make browser automation failures diagnosable without patching browser-cli behavior inside Agiwo.

**Architecture:** Keep the boundary explicit: `sleep_and_wait(wait_for=...)` only accepts direct scheduler child agent IDs, while `bash(background=true)` jobs are managed only through `bash_process`. Add fail-fast validation at the scheduler tool-control boundary, clearer fallback formatting for persisted/legacy invalid waitsets, stronger bash result guidance, and regression tests that reproduce the browser-cli trajectory.

**Tech Stack:** Python 3.10+, pytest, Agiwo scheduler runtime tools, builtin bash/bash_process tools, SQLite-backed run logs for operational verification.

---

## Problem Summary

The latest browser-cli trajectory failed because the agent launched:

```json
{"name": "bash", "arguments": {"background": true, "command": "uv run browser-cli read https://example.com --snapshot"}}
```

The bash tool returned:

```text
started background job e2cf1f96-5ce8-4a6b-899a-ce7029e2083d
```

The agent then called:

```json
{"name": "sleep_and_wait", "arguments": {"wake_type": "waitset", "wait_for": ["e2cf1f96-5ce8-4a6b-899a-ce7029e2083d"]}}
```

That ID was a bash process job, not a scheduler child agent. The scheduler accepted it and only later timed out with `Agent state not found`, which made the failure look like a browser-cli daemon problem.

This plan fixes the Agiwo side. The browser-cli daemon `degraded` status caused by missing trusted workspace binding is a browser-cli design/runtime recovery issue and should be fixed upstream, not papered over inside Agiwo.

## File Structure

- Modify `agiwo/scheduler/tool_control.py`
  - Owns tool-facing scheduler mutations.
  - Add explicit validation for `SleepRequest.wait_for` when `wake_type=WAITSET`.
  - Reject unknown, wrong-session, or non-child targets immediately with actionable guidance.

- Modify `agiwo/scheduler/runner.py`
  - Owns final child result collection after wake/timeout.
  - Replace misleading `"Agent state not found"` fallback with a boundary-specific message for any persisted invalid wait target.

- Modify `agiwo/scheduler/runtime_tools.py`
  - Owns public tool schema and descriptions for scheduler runtime tools.
  - Clarify that `sleep_and_wait.wait_for` accepts only child agent IDs from `spawn_child_agent`/`fork_child_agent`, never bash background job IDs.

- Modify `agiwo/tool/builtin/bash_tool/tool.py`
  - Owns bash tool user-facing description and background job return.
  - Make background job guidance stronger at the point the model receives the `job_id`.

- Modify `agiwo/tool/builtin/bash_tool/result_formatter.py`
  - Owns bash `ToolResult` content/payload formatting.
  - Include optional guidance lines in successful background job content without changing foreground command output.

- Modify `tests/scheduler/test_tools.py`
  - Add fail-fast tests for invalid `sleep_and_wait.wait_for`.
  - Add tests that direct-child IDs still work.

- Modify `tests/scheduler/test_scheduler.py`
  - Add a regression test for legacy/persisted invalid waitsets to ensure timeout messages no longer say `Agent state not found`.

- Modify `tests/tool/test_bash_tool.py`
  - Add assertion that background bash output tells the agent to use `bash_process`, not `sleep_and_wait`.

- Create `docs/runtime-tool-boundaries.md`
  - Document the runtime-tool boundary: child-agent orchestration vs shell process management.
  - Include browser-cli as the motivating example.

## Task 1: Fail Fast On Invalid Waitset Targets

**Files:**
- Modify: `agiwo/scheduler/tool_control.py`
- Test: `tests/scheduler/test_tools.py`

- [ ] **Step 1: Write failing tests for invalid explicit wait targets**

Add these tests inside `class TestSleepAndWaitTool` in `tests/scheduler/test_tools.py`, immediately after `test_sleep_waitset_explicit_wait_for`:

```python
    @pytest.mark.asyncio
    async def test_sleep_waitset_rejects_unknown_explicit_wait_for(
        self, store, control, context
    ):
        await _register_parent(store)

        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute(
            {
                "wake_type": "waitset",
                "wait_for": ["e2cf1f96-5ce8-4a6b-899a-ce7029e2083d"],
                "tool_call_id": "tc-invalid",
            },
            context,
        )

        assert result.is_success is False
        assert "wait_for only accepts direct child agent IDs" in result.content
        assert "bash(background=true)" in result.content
        assert "bash_process" in result.content

        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.RUNNING
        assert state.wake_condition is None

    @pytest.mark.asyncio
    async def test_sleep_waitset_rejects_non_child_explicit_wait_for(
        self, store, control, context
    ):
        await _register_parent(store)
        await store.save_state(
            AgentState(
                id="other-root-child",
                session_id="sess-1",
                status=AgentStateStatus.COMPLETED,
                task="not a direct child",
                parent_id="other-root",
                depth=1,
            )
        )

        sleep_tool = SleepAndWaitTool(control)
        result = await sleep_tool.execute(
            {
                "wake_type": "waitset",
                "wait_for": ["other-root-child"],
                "tool_call_id": "tc-invalid-parent",
            },
            context,
        )

        assert result.is_success is False
        assert "not direct children of agent 'orch'" in result.content

        state = await store.get_state("orch")
        assert state is not None
        assert state.status == AgentStateStatus.RUNNING
        assert state.wake_condition is None
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/scheduler/test_tools.py::TestSleepAndWaitTool::test_sleep_waitset_rejects_unknown_explicit_wait_for tests/scheduler/test_tools.py::TestSleepAndWaitTool::test_sleep_waitset_rejects_non_child_explicit_wait_for -v
```

Expected: both tests fail because `_resolve_waitset_targets()` currently returns explicit `wait_for` values without validation.

- [ ] **Step 3: Implement explicit wait target validation**

In `agiwo/scheduler/tool_control.py`, replace `_resolve_waitset_targets()` with:

```python
    async def _resolve_waitset_targets(self, request: SleepRequest) -> list[str]:
        if request.wait_for is not None:
            return await self._validate_explicit_waitset_targets(request)
        # Fetch all children with pagination to avoid missing children beyond page_size
        all_children: list[AgentState] = []
        offset = 0
        while True:
            page = await self._store.list_states(
                parent_id=request.agent_id,
                session_id=request.session_id,
                limit=self._state_list_page_size,
                offset=offset,
            )
            if not page:
                break
            all_children.extend(page)
            # If we got fewer than page_size, we've reached the end
            if len(page) < self._state_list_page_size:
                break
            offset += len(page)
        return [child.id for child in all_children]

    async def _validate_explicit_waitset_targets(
        self, request: SleepRequest
    ) -> list[str]:
        assert request.wait_for is not None
        wait_for = [target.strip() for target in request.wait_for if target.strip()]
        if not wait_for:
            return []

        missing: list[str] = []
        wrong_session: list[str] = []
        wrong_parent: list[str] = []
        for target_id in wait_for:
            target_state = await self._store.get_state(target_id)
            if target_state is None:
                missing.append(target_id)
                continue
            if target_state.session_id != request.session_id:
                wrong_session.append(target_id)
                continue
            if target_state.parent_id != request.agent_id:
                wrong_parent.append(target_id)

        if missing or wrong_session or wrong_parent:
            details: list[str] = []
            if missing:
                details.append(f"unknown targets: {', '.join(missing)}")
            if wrong_session:
                details.append(f"targets from another session: {', '.join(wrong_session)}")
            if wrong_parent:
                details.append(
                    f"targets not direct children of agent '{request.agent_id}': "
                    f"{', '.join(wrong_parent)}"
                )
            detail_text = "; ".join(details)
            raise ValueError(
                "wait_for only accepts direct child agent IDs created by "
                "spawn_child_agent or fork_child_agent. "
                f"{detail_text}. "
                "If this ID came from bash(background=true), inspect it with "
                "bash_process instead of sleep_and_wait."
            )

        return wait_for
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
uv run pytest tests/scheduler/test_tools.py::TestSleepAndWaitTool::test_sleep_waitset_explicit_wait_for tests/scheduler/test_tools.py::TestSleepAndWaitTool::test_sleep_waitset_rejects_unknown_explicit_wait_for tests/scheduler/test_tools.py::TestSleepAndWaitTool::test_sleep_waitset_rejects_non_child_explicit_wait_for -v
```

Expected: all three tests pass.

- [ ] **Step 5: Commit**

```bash
git add agiwo/scheduler/tool_control.py tests/scheduler/test_tools.py
git commit -m "fix: reject invalid scheduler wait targets"
```

## Task 2: Replace Misleading Legacy Timeout Message

**Files:**
- Modify: `agiwo/scheduler/runner.py`
- Test: `tests/scheduler/test_scheduler.py`

- [ ] **Step 1: Write a regression test for persisted invalid waitsets**

Add this test in `tests/scheduler/test_scheduler.py` near existing wait/timeout tests:

```python
    async def test_waitset_timeout_reports_invalid_wait_target(self):
        scheduler = Scheduler(
            config=SchedulerConfig(check_interval=0.01),
            store=InMemoryAgentStateStorage(),
        )
        state = AgentState(
            id="root-invalid-wait",
            session_id="sess-invalid-wait",
            status=AgentStateStatus.WAITING,
            task="root task",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=("e2cf1f96-5ce8-4a6b-899a-ce7029e2083d",),
                wait_mode=WaitMode.ALL,
                timeout_at=datetime.now(timezone.utc),
            ),
            is_persistent=True,
        )
        await scheduler._store.save_state(state)

        await scheduler._tick_once()

        updated = await scheduler._store.get_state("root-invalid-wait")
        assert updated is not None
        assert updated.pending_input is not None
        assert "Invalid wait target" in updated.pending_input
        assert "Agent state not found" not in updated.pending_input
        assert "bash_process" in updated.pending_input
```

If the local scheduler test suite uses a helper instead of direct `_tick_once()`, adapt only the call site; keep the assertions exactly focused on the pending input text.

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run pytest tests/scheduler/test_scheduler.py::TestScheduler::test_waitset_timeout_reports_invalid_wait_target -v
```

Expected: the test fails because the current timeout summary contains `Agent state not found`.

- [ ] **Step 3: Update child result collection fallback**

In `agiwo/scheduler/runner.py`, change:

```python
            if child is None:
                failed[child_id] = "Agent state not found"
```

to:

```python
            if child is None:
                failed[child_id] = (
                    "Invalid wait target: no direct child agent state exists for "
                    "this ID. sleep_and_wait(wait_for=...) only accepts child "
                    "agent IDs from spawn_child_agent/fork_child_agent. If this "
                    "ID came from bash(background=true), use bash_process."
                )
```

- [ ] **Step 4: Run targeted test**

Run:

```bash
uv run pytest tests/scheduler/test_scheduler.py::TestScheduler::test_waitset_timeout_reports_invalid_wait_target -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add agiwo/scheduler/runner.py tests/scheduler/test_scheduler.py
git commit -m "fix: clarify invalid wait target timeout message"
```

## Task 3: Make Tool Schemas Harder To Misuse

**Files:**
- Modify: `agiwo/scheduler/runtime_tools.py`
- Modify: `agiwo/tool/builtin/bash_tool/tool.py`
- Modify: `agiwo/tool/builtin/bash_tool/result_formatter.py`
- Test: `tests/tool/test_bash_tool.py`

- [ ] **Step 1: Write failing bash guidance test**

Add this test to `class TestBashToolBasic` in `tests/tool/test_bash_tool.py`:

```python
    async def test_background_result_points_to_bash_process_not_sleep_wait(
        self, bash_tool, mock_context
    ):
        result = await bash_tool.execute(
            {
                "command": "sleep 30",
                "background": True,
                "tool_call_id": "tc_background_guidance",
            },
            mock_context,
        )

        assert result.output["ok"] is True
        assert result.output["background"] is True
        assert result.output["job_id"]
        assert "Use bash_process" in result.content
        assert "Do not pass this job_id to sleep_and_wait" in result.content
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
uv run pytest tests/tool/test_bash_tool.py::TestBashToolBasic::test_background_result_points_to_bash_process_not_sleep_wait -v
```

Expected: FAIL because successful `bash(background=true)` content only says `started background job ...`.

- [ ] **Step 3: Strengthen `sleep_and_wait` schema wording**

In `agiwo/scheduler/runtime_tools.py`, update `SleepAndWaitTool.description` to:

```python
    description = (
        "Put the current agent to sleep and wait for a scheduler condition. "
        "Use 'waitset' only to wait for spawned child agents from "
        "spawn_child_agent or fork_child_agent. Do not use waitset for bash "
        "background jobs; use bash_process for jobs returned by bash(background=true). "
        "Use 'timer' to sleep for a fixed duration. "
        "Use 'periodic' to periodically wake up and check."
    )
```

Update the `wait_for` property description to:

```python
                    "description": (
                        "Optional list of direct child agent IDs to wait for. "
                        "IDs must come from spawn_child_agent/fork_child_agent. "
                        "Do not pass bash background job IDs here; use "
                        "bash_process for those jobs. If omitted, waits for all "
                        "direct child agents."
                    ),
```

- [ ] **Step 4: Strengthen bash tool description**

In `agiwo/tool/builtin/bash_tool/tool.py`, update the background-related description lines to:

```python
            "Set `background=true` to start a background shell job and return "
            "a bash job_id immediately. Use the separate `bash_process` tool "
            "to inspect, read logs, stop, or feed background jobs. Do not pass "
            "bash job_id values to sleep_and_wait; sleep_and_wait waits for "
            "scheduler child agents only. "
```

- [ ] **Step 5: Add optional guidance support to bash result formatter**

In `agiwo/tool/builtin/bash_tool/result_formatter.py`, update `ok()` content assembly from:

```python
        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}"
```

to:

```python
        content = f"exit_code: {payload['exit_code']}\nstdout: {payload['stdout']}"
        guidance = payload.get("guidance")
        if guidance:
            content += f"\nguidance: {guidance}"
```

- [ ] **Step 6: Attach background guidance when returning a job ID**

In `agiwo/tool/builtin/bash_tool/tool.py`, update the `self._formatter.ok(...)` call in the `if background:` branch to include:

```python
                guidance=(
                    "Use bash_process with this job_id to check status, read logs, "
                    "send input, or stop the job. Do not pass this job_id to "
                    "sleep_and_wait; sleep_and_wait waits for scheduler child agents."
                ),
```

The full branch should still return `job_id`, `state="running"`, `background=True`, and `mode`.

- [ ] **Step 7: Run targeted tests**

Run:

```bash
uv run pytest tests/tool/test_bash_tool.py::TestBashToolBasic::test_description_points_to_bash_process_tool tests/tool/test_bash_tool.py::TestBashToolBasic::test_background_result_points_to_bash_process_not_sleep_wait -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add agiwo/scheduler/runtime_tools.py agiwo/tool/builtin/bash_tool/tool.py agiwo/tool/builtin/bash_tool/result_formatter.py tests/tool/test_bash_tool.py
git commit -m "fix: clarify bash job and scheduler wait boundaries"
```

## Task 4: Document Runtime Tool Boundaries

**Files:**
- Create: `docs/runtime-tool-boundaries.md`

- [ ] **Step 1: Create the boundary document**

Create `docs/runtime-tool-boundaries.md` with:

```markdown
# Runtime Tool Boundaries

This document defines the operational boundary between scheduler child-agent orchestration and shell background process management.

## Scheduler Child Agents

Use `spawn_child_agent` or `fork_child_agent` when work should run as a scheduler-managed child agent.

The returned child agent ID may be used with:

- `sleep_and_wait(wake_type="waitset", wait_for=[child_id])`
- `query_spawned_agent`
- `cancel_agent`

`sleep_and_wait(wait_for=...)` accepts only direct child agent IDs in the same scheduler session.

## Bash Background Jobs

Use `bash(background=true)` when a shell command should keep running outside the current tool call.

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
{"name": "bash", "arguments": {"background": true, "command": "uv run browser-cli read https://example.com --snapshot"}}
```

the next check should use:

```json
{"name": "bash_process", "arguments": {"action": "status", "job_id": "<job_id>"}}
```

and then:

```json
{"name": "bash_process", "arguments": {"action": "logs", "job_id": "<job_id>", "tail": 200}}
```

Use `sleep_and_wait(wake_type="timer", delay_seconds=...)` only when the agent intentionally wants to pause and wake itself later. A timer wait does not monitor bash process completion.

## Browser CLI Upstream Boundary

A `browser-cli status` value such as `degraded` with missing trusted workspace binding is browser-cli runtime state. Agiwo should report that status and preserve logs, but should not invent browser-cli-specific recovery semantics inside scheduler wait handling.
```

- [ ] **Step 2: Commit**

```bash
git add docs/runtime-tool-boundaries.md
git commit -m "docs: define runtime tool boundaries"
```

## Task 5: Browser CLI Upstream Improvements

**Files:**
- No Agiwo code changes.
- Track as browser-cli upstream issues or patches in the browser-cli repository.

- [ ] **Step 1: File an upstream issue for daemon status clarity**

Use this issue body:

```markdown
## Problem

When Browser CLI reports:

```text
Status: degraded
Summary: Browser CLI no longer has a trusted extension workspace binding.
Guidance:
- Rebuild workspace binding to restore Browser CLI-owned tab tracking.
- Reconnect the extension if workspace state does not recover.
```

the status is understandable to a human, but hard for an agent to recover from reliably because the CLI does not expose a single non-interactive recovery command with machine-readable outcome.

## Requested improvement

Add a non-interactive recovery command, for example:

```bash
browser-cli recover-workspace --json
```

It should:

- rebuild workspace binding when possible
- reconnect or prompt extension reconnect through a clear status code when required
- return JSON with `status`, `action_taken`, `remaining_issue`, and `next_action`
- exit nonzero only when recovery cannot complete

## Why this matters

Agent runtimes can call `browser-cli status --json`, inspect `status=degraded`, run one recovery command, and continue without relying on free-form guidance text.
```

- [ ] **Step 2: File an upstream issue for stable JSON status**

Use this issue body:

```markdown
## Problem

`browser-cli status` is useful, but agent integrations need stable JSON fields to distinguish daemon health, extension connection, browser startup, workspace binding, active tab state, and command activity.

## Requested improvement

Add or stabilize:

```bash
browser-cli status --json
```

with fields:

```json
{
  "status": "healthy|degraded|stopped|failed",
  "daemon": {"running": true, "socket_reachable": true},
  "backend": {"active_driver": "extension|playwright|none", "extension_connected": true},
  "browser": {"started": true, "workspace_window": "present|absent", "active_tab": null},
  "stability": {"active_command": null, "queued_runs": 0},
  "recovery": {"recommended_action": "rebuild-workspace-binding|reconnect-extension|none"}
}
```

## Why this matters

Agents should branch on structured fields instead of parsing terminal prose.
```

- [ ] **Step 3: File an upstream issue for command lifecycle semantics**

Use this issue body:

```markdown
## Problem

Long browser commands can be launched through an external process manager such as Agiwo `bash(background=true)`, but Browser CLI itself does not expose a clear daemon-side command lifecycle ID that can be polled independently of shell process state.

## Requested improvement

For long-running commands, expose a command/run ID and status API:

```bash
browser-cli read <url> --snapshot --async --json
browser-cli run-status <run_id> --json
browser-cli run-logs <run_id> --tail 200
browser-cli run-cancel <run_id>
```

## Why this matters

Shell process status can tell whether the CLI process exited, but a daemon-side run ID can report browser-level progress, retries, degraded state, and recoverable errors.
```

## Task 6: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run scheduler tests touched by this work**

Run:

```bash
uv run pytest tests/scheduler/test_tools.py::TestSleepAndWaitTool tests/scheduler/test_scheduler.py::TestScheduler::test_waitset_timeout_reports_invalid_wait_target -v
```

Expected: PASS.

- [ ] **Step 2: Run bash tool tests touched by this work**

Run:

```bash
uv run pytest tests/tool/test_bash_tool.py::TestBashToolBasic tests/tool/test_bash_process_tool.py -v
```

Expected: PASS.

- [ ] **Step 3: Run lint**

Run:

```bash
uv run python scripts/lint.py ci
```

Expected: PASS.

- [ ] **Step 4: Run console tests because this changes user-visible scheduler behavior**

Run:

```bash
uv run python scripts/check.py console-tests
```

Expected: PASS.

## Rollout Notes

- Existing persisted invalid waitsets may still wake once, but the message should no longer say `Agent state not found`.
- New invalid waitsets should fail synchronously inside `sleep_and_wait` and should not move the agent into `WAITING`.
- This work intentionally does not auto-run `bash_process` from scheduler code. That would blur the boundary between scheduler agent orchestration and shell process management.
- This work intentionally does not add browser-cli-specific recovery to Agiwo. Browser CLI should own daemon recovery commands and structured degraded-state reporting.

## Self-Review

- Spec coverage: Covers the prior browser-cli install/deploy trajectory, the current daemon-started-but-`Agent State not found` failure, and the distinction between Agiwo boundary bugs and browser-cli degraded-state design.
- Placeholder scan: No implementation task relies on `TBD`, vague error handling, or unspecified tests.
- Type consistency: Uses existing `SleepRequest`, `WakeCondition`, `AgentState`, `AgentStateStatus`, `WakeType`, `WaitMode`, `SleepAndWaitTool`, `BashTool`, and `bash_process` names from the current codebase.
