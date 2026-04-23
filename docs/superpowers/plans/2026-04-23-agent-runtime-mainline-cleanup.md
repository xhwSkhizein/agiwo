# Agent Runtime Mainline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `agiwo/agent/run_loop.py` the only single-run execution mainline and reduce `agiwo/agent/runtime/` to runtime support objects.

**Architecture:** Move the step commit execution pipeline into `RunLoopOrchestrator`, pass that commit path explicitly to helper modules that need it, and delete runtime compatibility shells. Keep `RunStateWriter` as the committed-state writer and `SessionRuntime` as projection owner; do not change public `agiwo.agent` behavior.

**Tech Stack:** Python 3.10+, pytest, ruff, import-linter, existing `RunLog` / `StepView` / `HookRegistry` abstractions.

---

## File Structure

- Modify `agiwo/agent/run_loop.py`: owner of `_commit_step(...)`, callers pass it into compaction, termination summary, and tool-batch helpers; remove `RunEngine`.
- Modify `agiwo/agent/compaction.py`: accept an explicit commit callback instead of importing `runtime.step_committer`.
- Modify `agiwo/agent/termination/summarizer.py`: accept an explicit commit callback instead of importing `runtime.step_committer`.
- Modify `agiwo/agent/run_tool_batch.py`: accept an explicit commit callback instead of importing `runtime.step_committer`.
- Modify `agiwo/agent/runtime/__init__.py`: export only `RunContext`, `RunRuntime`, `SessionRuntime`, and `RunStateWriter`.
- Delete `agiwo/agent/runtime/run_engine.py`: empty compatibility shell.
- Delete `agiwo/agent/runtime/step_committer.py`: old commit pipeline shell.
- Delete `agiwo/agent/runtime/hook_dispatcher.py`: thin hook wrapper.
- Modify `tests/agent/test_run_engine.py`: add orchestrator commit-pipeline coverage.
- Modify `tests/agent/test_compact.py`: pass an explicit test commit callback into `compact_if_needed`.
- Modify `tests/agent/test_termination.py`: pass an explicit test commit callback into `maybe_generate_termination_summary`.
- Modify `tests/agent/test_module_layout.py`: assert removed runtime shells are gone and support modules remain.
- Modify `AGENTS.md`: document the sharpened `run_loop.py` / `runtime/` boundary.

## Task 1: Lock In Runtime Layout Expectations

**Files:**
- Modify: `tests/agent/test_module_layout.py`

- [ ] **Step 1: Replace the runtime layout test with the new boundary**

Edit `tests/agent/test_module_layout.py` so `test_runtime_modules_expose_context_session_and_step_helpers` becomes:

```python
def test_runtime_modules_expose_only_runtime_support_boundaries() -> None:
    runtime_module = importlib.import_module("agiwo.agent.runtime")
    context_module = importlib.import_module("agiwo.agent.runtime.context")
    session_module = importlib.import_module("agiwo.agent.runtime.session")
    state_writer_module = importlib.import_module("agiwo.agent.runtime.state_writer")
    state_ops_module = importlib.import_module("agiwo.agent.runtime.state_ops")

    assert hasattr(context_module, "RunContext")
    assert hasattr(context_module, "RunRuntime")
    assert hasattr(session_module, "SessionRuntime")
    assert hasattr(state_writer_module, "RunStateWriter")
    assert hasattr(state_ops_module, "track_step_state")
    assert hasattr(state_ops_module, "set_termination_reason")
    assert runtime_module.__all__ == [
        "RunContext",
        "RunRuntime",
        "RunStateWriter",
        "SessionRuntime",
    ]


def test_runtime_execution_shells_have_been_removed() -> None:
    removed_modules = (
        "agiwo.agent.runtime.run_engine",
        "agiwo.agent.runtime.step_committer",
        "agiwo.agent.runtime.hook_dispatcher",
    )

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None
```

- [ ] **Step 2: Run the layout test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_module_layout.py::test_runtime_modules_expose_only_runtime_support_boundaries tests/agent/test_module_layout.py::test_runtime_execution_shells_have_been_removed -v
```

Expected: FAIL because `runtime.__all__` still contains execution shells and the removed modules still exist.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/agent/test_module_layout.py
git commit -m "test: lock agent runtime support boundary"
```

## Task 2: Move Step Commit Pipeline Into RunLoopOrchestrator

**Files:**
- Modify: `agiwo/agent/run_loop.py`
- Modify: `tests/agent/test_run_engine.py`

- [ ] **Step 1: Add orchestrator commit-pipeline coverage**

In `tests/agent/test_run_engine.py`, update the imports:

```python
from agiwo.agent.hooks import HookPhase, HookRegistry, observe, transform
```

Add this test after `_NeverCalledModel`:

```python
@pytest.mark.asyncio
async def test_orchestrator_commit_step_writes_projects_and_dispatches_hook() -> None:
    storage = InMemoryRunLogStorage()
    session_runtime = SessionRuntime(
        session_id="commit-session",
        run_log_storage=storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]
    seen_steps = []

    async def capture_step(payload: dict) -> None:
        seen_steps.append(payload["step"].role.value)

    hooks = HookRegistry(
        [
            observe(
                HookPhase.AFTER_STEP_COMMIT,
                "capture_step",
                capture_step,
            )
        ]
    )
    context = RunContext(
        identity=RunIdentity(
            run_id="run-1",
            agent_id="agent-1",
            agent_name="agent",
        ),
        session_runtime=session_runtime,
    )
    context.hooks = hooks
    runtime = RunRuntime(
        session_runtime=session_runtime,
        config=AgentOptions(),
        hooks=hooks,
        model=_FixedResponseModel(),
        tools_map={},
        abort_signal=None,
        root_path=".",
        compact_start_seq=0,
        max_input_tokens_per_call=0,
        max_context_window=None,
        compact_prompt=None,
    )
    orchestrator = RunLoopOrchestrator(context, runtime)
    step = StepView.user(context, sequence=1, user_input="hello")

    committed = await orchestrator._commit_step(step)

    assert committed is step
    entries = await session_runtime.list_run_log_entries(run_id="run-1")
    assert [entry.kind.value for entry in entries] == ["user_step_committed"]
    assert [item.type for item in published] == ["step_completed"]
    assert seen_steps == ["user"]
    assert context.ledger.messages[-1] == {"role": "user", "content": "hello"}
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
uv run pytest tests/agent/test_run_engine.py::test_orchestrator_commit_step_writes_projects_and_dispatches_hook -v
```

Expected: FAIL with `AttributeError: 'RunLoopOrchestrator' object has no attribute '_commit_step'`.

- [ ] **Step 3: Add `_commit_step` to `RunLoopOrchestrator`**

In `agiwo/agent/run_loop.py`, remove this import:

```python
from agiwo.agent.runtime.step_committer import commit_step
```

Add this method below `_project_entries`:

```python
    async def _commit_step(
        self,
        step: StepView,
        *,
        llm: LLMCallContext | None = None,
        append_message: bool = True,
        track_state: bool = True,
    ) -> StepView:
        del llm
        entries = await self.writer.commit_step(
            step,
            append_message=append_message,
            track_state=track_state,
        )
        await self._project_entries(entries)
        await self.context.hooks.on_step(step, self.context)
        return step
```

- [ ] **Step 4: Route existing run-loop step commits through `_commit_step`**

In `execute_run`, replace:

```python
            await commit_step(
                self.context,
                bootstrap.user_step,
                append_message=False,
                track_state=False,
            )
```

with:

```python
            await self._commit_step(
                bootstrap.user_step,
                append_message=False,
                track_state=False,
            )
```

In `_run_assistant_turn`, replace:

```python
        await commit_step(self.context, step, llm=llm_context)
```

with:

```python
        await self._commit_step(step, llm=llm_context)
```

- [ ] **Step 5: Run the focused orchestrator tests**

Run:

```bash
uv run pytest tests/agent/test_run_engine.py::test_orchestrator_commit_step_writes_projects_and_dispatches_hook tests/agent/test_run_engine.py::test_agent_run_writes_basic_run_log_entries -v
```

Expected: PASS.

- [ ] **Step 6: Commit the orchestrator commit path**

```bash
git add agiwo/agent/run_loop.py tests/agent/test_run_engine.py
git commit -m "refactor: move step commit pipeline into run loop"
```

## Task 3: Pass the Commit Pipeline Into Runtime Helpers

**Files:**
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/compaction.py`
- Modify: `agiwo/agent/termination/summarizer.py`
- Modify: `agiwo/agent/run_tool_batch.py`
- Modify: `tests/agent/test_compact.py`
- Modify: `tests/agent/test_termination.py`

- [ ] **Step 1: Add a callback type to helper modules**

In each of `agiwo/agent/compaction.py`, `agiwo/agent/termination/summarizer.py`, and `agiwo/agent/run_tool_batch.py`, add this import:

```python
from collections.abc import Awaitable, Callable
```

Add this local type alias after imports in each file:

```python
StepCommitter = Callable[..., Awaitable[StepView]]
```

In `run_tool_batch.py`, keep the existing `ToolTerminationWriter` alias.

- [ ] **Step 2: Remove runtime commit imports from helper modules**

Delete this import from `compaction.py`, `termination/summarizer.py`, and `run_tool_batch.py`:

```python
from agiwo.agent.runtime.step_committer import commit_step
```

- [ ] **Step 3: Require the commit callback in compaction**

Change `compact_if_needed` in `agiwo/agent/compaction.py` from:

```python
async def compact_if_needed(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    max_context_window: int | None,
    compact_prompt: str | None = None,
    compact_start_seq: int,
    root_path: str | None = None,
) -> CompactResult:
```

to:

```python
async def compact_if_needed(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    max_context_window: int | None,
    commit_step: StepCommitter,
    compact_prompt: str | None = None,
    compact_start_seq: int,
    root_path: str | None = None,
) -> CompactResult:
```

Update the `_compact(...)` call to include `commit_step=commit_step`.

Change `_compact` from:

```python
async def _compact(
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    *,
    compact_prompt: str | None,
    compact_start_seq: int,
    root_path: str,
) -> CompactMetadata:
```

to:

```python
async def _compact(
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
    *,
    compact_prompt: str | None,
    compact_start_seq: int,
    root_path: str,
    commit_step: StepCommitter,
) -> CompactMetadata:
```

Inside `_compact`, replace:

```python
    await commit_step(state, compact_user_step, append_message=True)
```

with:

```python
    await commit_step(compact_user_step, append_message=True)
```

Replace:

```python
    await commit_step(state, step, llm=llm_context, append_message=False)
```

with:

```python
    await commit_step(step, llm=llm_context, append_message=False)
```

- [ ] **Step 4: Require the commit callback in termination summary**

Change `maybe_generate_termination_summary` in `agiwo/agent/termination/summarizer.py` from:

```python
async def maybe_generate_termination_summary(
    *,
    state: RunContext,
    options: AgentOptions,
    model: Model,
    abort_signal: AbortSignal | None,
) -> None:
```

to:

```python
async def maybe_generate_termination_summary(
    *,
    state: RunContext,
    options: AgentOptions,
    model: Model,
    abort_signal: AbortSignal | None,
    commit_step: StepCommitter,
) -> None:
```

Replace:

```python
    await commit_step(state, summary_user_step, append_message=True)
```

with:

```python
    await commit_step(summary_user_step, append_message=True)
```

Replace:

```python
        await commit_step(state, step, llm=llm_context, append_message=False)
```

with:

```python
        await commit_step(step, llm=llm_context, append_message=False)
```

- [ ] **Step 5: Require the commit callback in tool-batch execution**

Change `execute_tool_batch_cycle` in `agiwo/agent/run_tool_batch.py` from:

```python
async def execute_tool_batch_cycle(
    *,
    context: RunContext,
    runtime: RunRuntime,
    tool_calls: list[dict[str, Any]],
    set_termination_reason: ToolTerminationWriter,
) -> bool:
```

to:

```python
async def execute_tool_batch_cycle(
    *,
    context: RunContext,
    runtime: RunRuntime,
    tool_calls: list[dict[str, Any]],
    set_termination_reason: ToolTerminationWriter,
    commit_step: StepCommitter,
) -> bool:
```

Replace:

```python
        await commit_step(context, tool_step)
```

with:

```python
        await commit_step(tool_step)
```

- [ ] **Step 6: Pass `self._commit_step` from `RunLoopOrchestrator`**

In `agiwo/agent/run_loop.py`, update the `compact_if_needed(...)` call:

```python
            commit_step=self._commit_step,
```

Add it with the other keyword arguments in `_run_compaction_cycle`.

Update `maybe_generate_termination_summary(...)` in `_finalize_run`:

```python
            commit_step=self._commit_step,
```

Update `execute_tool_batch_cycle(...)` in `_execute_tool_calls`:

```python
            commit_step=self._commit_step,
```

- [ ] **Step 7: Add test commit helpers for direct helper tests**

In `tests/agent/test_compact.py`, add:

```python
async def _commit_step_for_test(
    state: RunContext,
    step: StepView,
    *,
    llm=None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepView:
    del llm
    writer = RunStateWriter(state)
    entries = await writer.commit_step(
        step,
        append_message=append_message,
        track_state=track_state,
    )
    await state.session_runtime.project_run_log_entries(
        entries,
        run_id=state.run_id,
        agent_id=state.agent_id,
        parent_run_id=state.parent_run_id,
        depth=state.depth,
    )
    await state.hooks.on_step(step, state)
    return step
```

Ensure `RunStateWriter` and `StepView` are imported:

```python
from agiwo.agent.models.step import StepView
from agiwo.agent.runtime.state_writer import RunStateWriter
```

Update the direct `compact_if_needed(...)` call:

```python
        commit_step=lambda step, **kwargs: _commit_step_for_test(
            state,
            step,
            **kwargs,
        ),
```

- [ ] **Step 8: Add a test commit helper for termination summary**

In `tests/agent/test_termination.py`, add this helper:

```python
async def _commit_step_for_test(
    state: RunContext,
    step: StepView,
    *,
    llm=None,
    append_message: bool = True,
    track_state: bool = True,
) -> StepView:
    del llm
    writer = RunStateWriter(state)
    entries = await writer.commit_step(
        step,
        append_message=append_message,
        track_state=track_state,
    )
    await state.session_runtime.project_run_log_entries(
        entries,
        run_id=state.run_id,
        agent_id=state.agent_id,
        parent_run_id=state.parent_run_id,
        depth=state.depth,
    )
    await state.hooks.on_step(step, state)
    return step
```

Ensure `RunStateWriter` and `StepView` are imported.

Update the direct `maybe_generate_termination_summary(...)` call:

```python
        commit_step=lambda step, **kwargs: _commit_step_for_test(
            state,
            step,
            **kwargs,
        ),
```

- [ ] **Step 9: Run focused helper tests**

Run:

```bash
uv run pytest tests/agent/test_compact.py::test_compact_if_needed_uses_the_same_step_commit_pipeline_as_run_loop tests/agent/test_termination.py::test_termination_summary_writes_canonical_llm_call_facts tests/agent/test_run_engine.py::test_agent_run_writes_basic_run_log_entries -v
```

Expected: PASS.

- [ ] **Step 10: Commit helper callback migration**

```bash
git add agiwo/agent/run_loop.py agiwo/agent/compaction.py agiwo/agent/termination/summarizer.py agiwo/agent/run_tool_batch.py tests/agent/test_compact.py tests/agent/test_termination.py
git commit -m "refactor: pass step commit pipeline to run helpers"
```

## Task 4: Delete Runtime Shell Modules And Tighten Runtime Facade

**Files:**
- Delete: `agiwo/agent/runtime/run_engine.py`
- Delete: `agiwo/agent/runtime/step_committer.py`
- Delete: `agiwo/agent/runtime/hook_dispatcher.py`
- Modify: `agiwo/agent/runtime/__init__.py`
- Modify: `agiwo/agent/run_loop.py`

- [ ] **Step 1: Remove `RunEngine` from `run_loop.py`**

At the end of `agiwo/agent/run_loop.py`, replace:

```python
RunEngine = RunLoopOrchestrator


__all__ = ["execute_run", "RunEngine", "RunLoopOrchestrator"]
```

with:

```python
__all__ = ["execute_run", "RunLoopOrchestrator"]
```

- [ ] **Step 2: Replace `runtime/__init__.py` with support-only exports**

Replace the contents of `agiwo/agent/runtime/__init__.py` with:

```python
"""Runtime support objects for agent execution."""

from agiwo.agent.runtime.context import RunContext, RunRuntime
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.runtime.state_writer import RunStateWriter

__all__ = [
    "RunContext",
    "RunRuntime",
    "RunStateWriter",
    "SessionRuntime",
]
```

- [ ] **Step 3: Delete the shell modules**

Run:

```bash
mv agiwo/agent/runtime/run_engine.py trash/agent_runtime_run_engine.py
mv agiwo/agent/runtime/step_committer.py trash/agent_runtime_step_committer.py
mv agiwo/agent/runtime/hook_dispatcher.py trash/agent_runtime_hook_dispatcher.py
```

Expected: files are removed from their runtime paths and retained under `trash/`.

- [ ] **Step 4: Confirm no production imports remain**

Run:

```bash
rg -n "runtime\\.run_engine|runtime\\.step_committer|runtime\\.hook_dispatcher|RunEngine|HookDispatcher|commit_step" agiwo tests console -S
```

Expected: no references to `runtime.run_engine`, `runtime.step_committer`, `runtime.hook_dispatcher`, `RunEngine`, or `HookDispatcher`. Remaining `commit_step` references are allowed only as callback parameter names or `_commit_step` methods.

- [ ] **Step 5: Run layout tests**

Run:

```bash
uv run pytest tests/agent/test_module_layout.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit runtime shell removal**

```bash
git add agiwo/agent/run_loop.py agiwo/agent/runtime/__init__.py trash/agent_runtime_run_engine.py trash/agent_runtime_step_committer.py trash/agent_runtime_hook_dispatcher.py
git add -u agiwo/agent/runtime
git commit -m "refactor: remove agent runtime execution shells"
```

## Task 5: Update Repository Boundary Documentation

**Files:**
- Modify: `AGENTS.md`

- [ ] **Step 1: Update the `agiwo/agent/` repository layout row**

In `AGENTS.md`, update the `agiwo/agent/` row so it includes this boundary sentence:

```markdown
`run_loop.py` is the only single-run execution owner; `agiwo.agent.runtime` is a support package for `RunContext` / `RunRuntime`, `SessionRuntime`, and `RunStateWriter`, and must not expose execution-owner aliases, commit-pipeline facades, or compatibility shells.
```

- [ ] **Step 2: Update the Agent core component section**

Add this bullet in the Agent section near the `SessionRuntime` / `RunStateWriter` bullets:

```markdown
- 单次 run 的主线只在 `agiwo.agent.run_loop.RunLoopOrchestrator`；runtime 包只承载 context、session 与 committed-state writer，不保留 `RunEngine` 这类别名入口，也不保留 runtime-level `commit_step` 包装层。
```

- [ ] **Step 3: Verify the documentation mentions the removed shells only as forbidden patterns**

Run:

```bash
rg -n "RunEngine|runtime-level `commit_step`|commit-pipeline facades|runtime/run_engine" AGENTS.md docs/superpowers/specs/2026-04-23-agent-runtime-mainline-cleanup-design.md
```

Expected: matches are in design/spec guidance or AGENTS boundary text, not in runtime implementation references.

- [ ] **Step 4: Commit documentation updates**

```bash
git add AGENTS.md
git commit -m "docs: clarify agent runtime mainline boundary"
```

## Task 6: Run Full Verification

**Files:**
- No source edits expected.

- [ ] **Step 1: Run agent tests**

Run:

```bash
uv run pytest tests/agent -v
```

Expected: PASS.

- [ ] **Step 2: Run CI lint gate**

Run:

```bash
uv run python scripts/lint.py ci
```

Expected: PASS with all import-linter contracts kept and ruff passing.

- [ ] **Step 3: Run final reference scan**

Run:

```bash
rg -n "runtime\\.run_engine|runtime\\.step_committer|runtime\\.hook_dispatcher|from agiwo\\.agent\\.runtime import .*commit_step|RunEngine|HookDispatcher" agiwo tests console -S
```

Expected: no matches.

- [ ] **Step 4: Commit verification-only fixes if any were needed**

If Step 1 or Step 2 required small fixes, inspect `git status --short`, add only the files changed for this cleanup, and commit them:

```bash
git status --short
git add path/to/fixed_file.py path/to/fixed_test.py
git commit -m "fix: complete agent runtime mainline cleanup"
```

If no fixes were needed, do not create an empty commit.
