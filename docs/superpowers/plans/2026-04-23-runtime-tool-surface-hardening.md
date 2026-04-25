# Runtime Tool Surface Hardening Implementation Plan

> **Status:** Completed and archived.
> Created: 2026-04-23
> Verified against current code: 2026-04-24
> Archival note: `stdin: ""` normalization, `SpawnChildAgentTool`, `allowed_tools`, and the `spawn_child_agent` / `fork_child_agent` split are already implemented in the current codebase. Use this plan as historical context, not as pending work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ambiguous `spawn_agent` runtime tool with explicit `spawn_child_agent` and `fork_child_agent`, expose `allowed_tools` for regular child spawning, and normalize empty bash stdin so `stdin: ""` no longer falsely requires PTY mode.

**Architecture:** Keep the internal scheduler primitive on `SpawnChildRequest` plus its existing `fork` flag and harden the public tool surface instead of rewriting the scheduler core. Split runtime-tool entrypoints into two concrete tools, wire `allowed_tools` through the existing child override path, and fix bash stdin normalization in the parser so the execution layer only sees meaningful stdin values.

**Tech Stack:** Python 3.10+, pytest, scheduler runtime tools, bash builtin tool, Markdown docs

---

## File Structure

- Modify: `agiwo/tool/builtin/bash_tool/parameter_parser.py`
  Normalize empty-string stdin before execution.
- Modify: `agiwo/tool/builtin/bash_tool/tool.py`
  Keep runtime checks aligned with the new stdin normalization behavior.
- Modify: `agiwo/scheduler/runtime_tools.py`
  Remove `SpawnAgentTool`, add `SpawnChildAgentTool` and `ForkChildAgentTool`, and expose `allowed_tools` on the regular spawn tool.
- Modify: `agiwo/scheduler/runner.py`
  Rename the excluded runtime tool for non-fork children and update fork-related comments.
- Modify: `agiwo/scheduler/formatting.py`
  Update fork notices to refer to `fork_child_agent` instead of `spawn_agent`.
- Modify: `tests/tool/test_bash_tool.py`
  Cover empty-stdin normalization and keep the non-empty stdin rejection test.
- Modify: `tests/scheduler/test_tools.py`
  Replace `SpawnAgentTool` coverage with `SpawnChildAgentTool` and `ForkChildAgentTool`, and add `allowed_tools` assertions.
- Modify: `tests/scheduler/test_scheduler.py`
  Update runtime-tool presence assertions and fork/non-fork inheritance checks for the new tool names.
- Modify: `tests/scheduler/test_models.py`
  Update fork notice expectations.
- Modify: `AGENTS.md`
  Replace public runtime-tool references and child inheritance wording.
- Modify: `docs/guides/multi-agent.md`
  Document the new tool names and semantics.
- Modify: `website/src/content/docs/docs/guides/multi-agent.mdx`
  Mirror the doc-site version of the multi-agent guide update.
- Modify: `docs/concepts/scheduler.md`
  Update the runtime tool table and child-agent orchestration wording.
- Modify: `docs/concepts/agent.md`
  Update the system-tool description for scheduler-owned runtime tools.
- Modify: `website/src/content/docs/docs/concepts/tool.mdx`
  Update scheduler runtime tool examples.

## Task 1: Verify Bash Empty Stdin Handling

**Files:**
- Modify: `tests/tool/test_bash_tool.py`
- Modify: `agiwo/tool/builtin/bash_tool/parameter_parser.py`
- Modify: `agiwo/tool/builtin/bash_tool/tool.py`

- [x] **Step 1: Verify the empty-stdin regression test already exists**

```python
async def test_empty_stdin_does_not_require_pty(self, bash_tool, mock_context):
    result = await bash_tool.execute(
        {"command": "echo hi", "stdin": "", "tool_call_id": "tc_005d"},
        mock_context,
    )

    assert result.output["ok"] is True
    call = bash_tool.config.sandbox.execute_calls[-1]
    assert call["stdin"] is None
```

- [x] **Step 2: Run the focused bash test file to verify the current behavior**

Run: `uv run pytest tests/tool/test_bash_tool.py -v`

Expected: PASS, including `test_empty_stdin_does_not_require_pty`, because `stdin=""` is already normalized to `None`.

- [x] **Step 3: Confirm the parser already normalizes empty-string stdin**

```python
def parse_stdin(self, parameters: dict[str, Any]) -> str | None | ParseError:
    """Return parsed stdin value, None, or a ParseError."""
    stdin_value = parameters.get("stdin")
    if stdin_value is None:
        return None
    if not isinstance(stdin_value, str):
        return ParseError("stdin must be a string")
    if stdin_value == "":
        return None
    return stdin_value
```

- [x] **Step 4: Confirm the execution guard remains strict for non-empty stdin**

```python
if background and stdin is not None:
    return self._formatter.error(
        parameters,
        "stdin is only supported for foreground PTY execution",
        tool_call_id=tool_call_id,
    )
if stdin is not None and not use_pty:
    return self._formatter.error(
        parameters, "stdin requires pty=true", tool_call_id=tool_call_id
    )
```

The current implementation already uses parser normalization rather than relaxing the execution guard. Keep `agiwo/tool/builtin/bash_tool/tool.py` unchanged unless unrelated cleanup is required.

- [x] **Step 5: Re-run the bash tests after verification**

Run: `uv run pytest tests/tool/test_bash_tool.py -v`

Expected: PASS, including the existing `test_stdin_requires_pty` case for non-empty stdin and the empty-stdin normalization case.

- [ ] **Step 6: Commit the bash hardening slice**

```bash
git add tests/tool/test_bash_tool.py agiwo/tool/builtin/bash_tool/parameter_parser.py agiwo/tool/builtin/bash_tool/tool.py
git commit -m "fix: normalize empty bash stdin"
```

## Task 2: Split Runtime Tool Entry Points

**Files:**
- Modify: `tests/scheduler/test_tools.py`
- Modify: `agiwo/scheduler/runtime_tools.py`

- [ ] **Step 1: Replace the old spawn-tool imports and add new test classes**

```python
from agiwo.scheduler.runtime_tools import (
    CancelAgentTool,
    ForkChildAgentTool,
    ListAgentsTool,
    QuerySpawnedAgentTool,
    ReviewTrajectoryTool,
    SleepAndWaitTool,
    SpawnChildAgentTool,
)


class TestSpawnChildAgentTool:
    """Coverage for the regular child-spawn runtime tool."""


class TestForkChildAgentTool:
    """Coverage for the fork child-spawn runtime tool."""
```

- [ ] **Step 2: Add failing tests for `allowed_tools` passthrough and fork-tool schema**

```python
@pytest.mark.asyncio
async def test_spawn_child_passes_allowed_tools(self, store, control, context):
    await _register_parent(store)
    tool = SpawnChildAgentTool(control)

    tool_result = await tool.execute(
        {
            "task": "Research topic A",
            "allowed_tools": [],
            "child_id": "restricted-child",
            "tool_call_id": "tc-1",
        },
        context,
    )

    assert tool_result.is_success
    state = await store.get_state("restricted-child")
    assert state is not None
    assert state.config_overrides["allowed_tools"] == []


@pytest.mark.asyncio
async def test_fork_child_sets_fork_without_custom_overrides(
    self, store, control, context
):
    await _register_parent(store)
    tool = ForkChildAgentTool(control)

    tool_result = await tool.execute(
        {"task": "Continue analysis", "child_id": "fork-child", "tool_call_id": "tc-1"},
        context,
    )

    assert tool_result.is_success
    state = await store.get_state("fork-child")
    assert state is not None
    assert state.config_overrides.get("fork") is True
    assert state.config_overrides.get("instruction") is None
    assert state.config_overrides.get("allowed_tools") is None
    assert state.config_overrides.get("allowed_skills") is None
```

- [x] **Step 3: Run the scheduler tool tests to verify the split-tool behavior**

Run: `uv run pytest tests/scheduler/test_tools.py -v`

Expected: PASS because `SpawnChildAgentTool` and `ForkChildAgentTool` already exist and the old `SpawnAgentTool` has already been removed.

- [x] **Step 4: Reference the current shared-base implementation**

```python
class _BaseSpawnChildTool(BaseTool):
    _fork: bool = False
    _success_verb = "Spawned"

    def __init__(self, port: SchedulerToolControl) -> None:
        self._port = port
        super().__init__()

    async def gate(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
    ) -> ToolGateDecision:
        del parameters
        if context.depth > 0:
            return ToolGateDecision.deny("Child agents cannot spawn further agents.")
        return ToolGateDecision.allow()

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        ...


class SpawnChildAgentTool(_BaseSpawnChildTool):
    name = "spawn_child_agent"
    description = """Spawn a child agent to handle an independent sub-task asynchronously.
Use this when you want a fresh child agent with optional instruction, tool restrictions,
or skill restrictions. The child agent cannot spawn further child agents."""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief, complete task for the child agent.",
                },
                "instruction": {
                    "type": "string",
                    "description": "Optional child-specific execution guidance.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit functional tool list for the child.",
                },
                "allowed_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional explicit skill list for the child.",
                },
            },
            "required": ["task"],
        }

    def _get_instruction(self, parameters: dict[str, Any]) -> str | None:
        instruction = parameters.get("instruction")
        return instruction if isinstance(instruction, str) else None

    def _get_allowed_skills(self, parameters: dict[str, Any]) -> list[str] | None:
        return _parse_optional_string_list(
            parameters.get("allowed_skills"),
            field_name="allowed_skills",
        )

    def _get_allowed_tools(self, parameters: dict[str, Any]) -> list[str] | None:
        return _parse_optional_string_list(
            parameters.get("allowed_tools"),
            field_name="allowed_tools",
        )


class ForkChildAgentTool(_BaseSpawnChildTool):
    name = "fork_child_agent"
    description = """Fork the current agent into a child agent that inherits the parent
conversation context. Use this when the child should continue from the parent's context
instead of starting fresh. The child agent cannot spawn further child agents."""
    _fork = True
    _success_verb = "Forked"

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Brief task for the forked child to continue from the parent's context.",
                },
            },
            "required": ["task"],
        }
```

`_BaseSpawnChildTool` owns `__init__(port)`, `gate()`, and the shared `execute()` flow so the depth-check gate and base spawn logic stay aligned. `ForkChildAgentTool` only switches `_fork = True` and `_success_verb = "Forked"` while keeping the base plumbing intact.

- [ ] **Step 5: Update the tests to use the new tool classes and names**

```python
tool = SpawnChildAgentTool(control)
assert "Spawned child agent" in tool_result.content

tool = ForkChildAgentTool(control)
assert state.config_overrides.get("fork") is True
```

Delete or rewrite tests that currently exercise `fork=True` on the old `SpawnAgentTool`. The new contract is two distinct tools, so no test should call a single tool with a `fork` flag anymore.

- [ ] **Step 6: Re-run the scheduler tool test file**

Run: `uv run pytest tests/scheduler/test_tools.py -v`

Expected: PASS, including the new `allowed_tools` persistence coverage and the fork-child coverage under the new tool name.

- [ ] **Step 7: Commit the runtime-tool split**

```bash
git add tests/scheduler/test_tools.py agiwo/scheduler/runtime_tools.py
git commit -m "refactor: split child spawn runtime tools"
```

## Task 3: Wire New Tool Names Through Scheduler Runtime Behavior

**Files:**
- Modify: `tests/scheduler/test_scheduler.py`
- Modify: `tests/scheduler/test_models.py`
- Modify: `agiwo/scheduler/runner.py`
- Modify: `agiwo/scheduler/formatting.py`

- [ ] **Step 1: Add failing assertions for runtime-tool visibility**

```python
assert "spawn_child_agent" in tool_names
assert "fork_child_agent" in tool_names
assert "spawn_agent" not in tool_names
```

Update both root-agent and child-agent visibility tests in `tests/scheduler/test_scheduler.py`, including the fork-child inheritance assertions that currently mention `spawn_agent`.

- [ ] **Step 2: Run the scheduler runtime test files to verify name-based failures**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/scheduler/test_models.py -v`

Expected: FAIL because the scheduler still excludes `spawn_agent` by name, the fork notice still says `Do NOT use spawn_agent`, and runtime tool expectations still reference the removed name.

- [ ] **Step 3: Rename the excluded non-fork tool and fork notice text**

```python
_CHILD_EXCLUDED_SYSTEM_TOOLS: frozenset[str] = frozenset({"spawn_child_agent"})
```

```python
_FORK_NOTICE = _system_notice(
    "You are a forked child agent. Your conversation history has been "
    "inherited from the parent agent. Do NOT use fork_child_agent or "
    "spawn_child_agent — child agents cannot spawn further child agents. "
    "Complete the following task directly."
)
```

Preserve the existing runtime behavior:

- non-fork children exclude the regular spawn tool
- fork children still inherit all system tools for prompt/KV alignment
- gate checks still prevent child agents from actually spawning children

- [ ] **Step 4: Update scheduler runtime tests to the new names**

```python
parent_has_spawn = "spawn_child_agent" in parent_tool_names
child_has_spawn = "spawn_child_agent" in child_tool_names
assert "fork_child_agent" in child_tool_names
assert "spawn_agent" not in child_tool_names
```

Also update `tests/scheduler/test_models.py` string assertions so they match the new fork notice text.

- [ ] **Step 5: Re-run the scheduler runtime tests**

Run: `uv run pytest tests/scheduler/test_scheduler.py tests/scheduler/test_models.py -v`

Expected: PASS, with root and fork-child tool visibility aligned to the new public runtime-tool surface.

- [ ] **Step 6: Commit the scheduler runtime naming updates**

```bash
git add tests/scheduler/test_scheduler.py tests/scheduler/test_models.py agiwo/scheduler/runner.py agiwo/scheduler/formatting.py
git commit -m "refactor: align scheduler runtime with child tool split"
```

## Task 4: Update Specs, Guides, and Repo Contract Docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/guides/multi-agent.md`
- Modify: `website/src/content/docs/docs/guides/multi-agent.mdx`
- Modify: `docs/concepts/scheduler.md`
- Modify: `docs/concepts/agent.md`
- Modify: `website/src/content/docs/docs/concepts/tool.mdx`

- [ ] **Step 1: Add failing doc grep checks locally**

Run:

```bash
rg -n "\bspawn_agent\b" AGENTS.md docs/guides/multi-agent.md website/src/content/docs/docs/guides/multi-agent.mdx docs/concepts/scheduler.md docs/concepts/agent.md website/src/content/docs/docs/concepts/tool.mdx
```

Expected: matches are found in all of these files because the docs still describe the removed tool name.

- [ ] **Step 2: Update `AGENTS.md` to describe the new runtime tool names**

```md
- Scheduler runtime tools（`spawn_child_agent`、`fork_child_agent`、`sleep_and_wait` 等）通过 `runtime_agent.inject_system_tools` 注入，不混入 `tools`（extra_tools），不受 `allowed_tools` 约束。
- 子 Agent 的 system_tools 由 `SchedulerRunner` 从父 Agent 的 `system_tools` 派生；非 fork 模式排除 `spawn_child_agent`，fork 模式继承全部（gate 检查仍阻止实际继续派生 child）。
```

- [ ] **Step 3: Update the multi-agent and concepts docs**

Use replacements of this shape:

```md
- `spawn_child_agent`: create a fresh child agent for an independent sub-task
- `fork_child_agent`: fork the current agent into a child that inherits parent context
```

```md
Scheduler tools such as `spawn_child_agent`, `fork_child_agent`, and `sleep_and_wait`
are runtime-owned system tools. They are not registered manually on the agent.
```

Make the same semantic update in both repository docs and website docs so the published site does not drift from repo documentation.

- [ ] **Step 4: Re-run the doc grep check and the repo lint gate**

Run:

```bash
rg -n "\bspawn_agent\b" AGENTS.md docs/guides/multi-agent.md website/src/content/docs/docs/guides/multi-agent.mdx docs/concepts/scheduler.md docs/concepts/agent.md website/src/content/docs/docs/concepts/tool.mdx
uv run python scripts/lint.py ci
```

Expected:

- the `rg` command returns no matches in the targeted files
- the lint gate passes

- [ ] **Step 5: Commit the documentation sweep**

```bash
git add AGENTS.md docs/guides/multi-agent.md website/src/content/docs/docs/guides/multi-agent.mdx docs/concepts/scheduler.md docs/concepts/agent.md website/src/content/docs/docs/concepts/tool.mdx
git commit -m "docs: rename child runtime tools"
```

## Task 5: Full Regression Pass

**Files:**
- Modify: none
- Test: `tests/tool/test_bash_tool.py`
- Test: `tests/scheduler/test_tools.py`
- Test: `tests/scheduler/test_scheduler.py`
- Test: `tests/scheduler/test_models.py`

- [ ] **Step 1: Run the focused regression suite**

Run:

```bash
uv run pytest \
  tests/tool/test_bash_tool.py \
  tests/scheduler/test_tools.py \
  tests/scheduler/test_scheduler.py \
  tests/scheduler/test_models.py \
  -v
```

Expected: PASS across bash normalization, runtime tool behavior, and scheduler visibility assertions.

- [ ] **Step 2: Run the repo CI-aligned lint gate one more time**

Run: `uv run python scripts/lint.py ci`

Expected: PASS.

- [ ] **Step 3: Inspect the working tree**

Run: `git status --short`

Expected: only intended modified files remain. Do not touch unrelated pre-existing changes outside this plan.

- [ ] **Step 4: Create the final integration commit**

```bash
git add agiwo/tool/builtin/bash_tool/parameter_parser.py \
  agiwo/tool/builtin/bash_tool/tool.py \
  agiwo/scheduler/runtime_tools.py \
  agiwo/scheduler/runner.py \
  agiwo/scheduler/formatting.py \
  tests/tool/test_bash_tool.py \
  tests/scheduler/test_tools.py \
  tests/scheduler/test_scheduler.py \
  tests/scheduler/test_models.py \
  AGENTS.md \
  docs/guides/multi-agent.md \
  website/src/content/docs/docs/guides/multi-agent.mdx \
  docs/concepts/scheduler.md \
  docs/concepts/agent.md \
  website/src/content/docs/docs/concepts/tool.mdx
git commit -m "refactor: harden runtime tool surface"
```

- [ ] **Step 5: Summarize the rollout outcome**

Use this structure in the handoff note:

```md
- Replaced `spawn_agent` with `spawn_child_agent` and `fork_child_agent`.
- Exposed `allowed_tools` on regular child spawning.
- Normalized empty bash stdin so `stdin: ""` no longer falsely requires PTY.
- Updated scheduler runtime tests, docs, and AGENTS guidance.
```

## Self-Review

- Spec coverage: this plan covers the three approved design changes: bash stdin normalization, child-runtime tool split, and doc/contract updates for the new tool names.
- Placeholder scan: no `TBD`, `TODO`, or deferred “handle later” text remains in task steps.
- Type consistency: the plan consistently uses `spawn_child_agent`, `fork_child_agent`, `allowed_tools`, and `SpawnChildRequest` with the existing `fork` flag across code, tests, and docs.
