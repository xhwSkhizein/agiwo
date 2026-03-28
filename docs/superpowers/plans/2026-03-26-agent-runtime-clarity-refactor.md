# Agent Runtime Clarity Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `agiwo/agent` so step recording, mutable run state, child-definition resolution, public runtime types, and nested-agent tool context each have one clear owner.

**Architecture:** Introduce small runtime seams around step commit, run-state mutation, agent definition resolution, and stream/message adapters. Keep the public `agiwo.agent` surface stable, but move internal modules onto narrower dependencies so compaction, child execution, and downstream serialization cannot silently drift apart.

**Tech Stack:** Python 3.11+, asyncio, dataclasses, pytest, ruff, existing Agiwo agent/storage abstractions.

---

## File Structure

### Create
- `agiwo/agent/step_pipeline.py`
  - Canonical owner for committing a `StepRecord` into runtime state, storage, trace, hooks, and stream publication.
- `agiwo/agent/run_identity.py`
  - Immutable run identity and metadata carrier used by `RunContext`.
- `agiwo/agent/run_ledger.py`
  - Mutable execution ledger for messages, counters, termination, and compaction metadata.
- `agiwo/agent/run_mutations.py`
  - Narrow mutation helpers so runtime modules stop writing arbitrary `RunContext` fields directly.
- `agiwo/agent/definition.py`
  - Definition-scoped assembly and shared child-definition resolution.
- `agiwo/agent/records.py`
  - Runtime record types such as `Run`, `RunOutput`, `StepRecord`, `StepMetrics`, `RunMetrics`.
- `agiwo/agent/stream_events.py`
  - Public stream event payloads and event serialization helpers.
- `agiwo/agent/message_adapters.py`
  - `step_to_message()` and related adapters between runtime records and LLM message dicts.
- `agiwo/agent/runtime_tool_context.py`
  - Runtime-only `ToolContext` subtype for nested agent execution.

### Modify
- `agiwo/agent/run_loop.py`
  - Stop owning step-commit logic and run-state mutation details directly.
- `agiwo/agent/compaction.py`
  - Route compaction request/response steps through the same step pipeline as normal runs.
- `agiwo/agent/run_state.py`
  - Compose `SessionRuntime`, `RunIdentity`, and `RunLedger` instead of acting as a mutable grab bag.
- `agiwo/agent/state_tracking.py`
  - Update only the ledger portion of runtime state.
- `agiwo/agent/tool_executor.py`
  - Build plain `ToolContext` for plain tools and runtime tool context only where needed.
- `agiwo/agent/agent.py`
  - Delegate definition assembly and child spec resolution to `definition.py`.
- `agiwo/agent/child.py`
  - Depend on the explicit runtime tool context instead of raw `ToolContext` internals.
- `agiwo/agent/types.py`
  - Become a public facade/re-export module only.
- `agiwo/agent/__init__.py`
  - Keep public exports stable after the internal split.
- `agiwo/agent/storage/serialization.py`
  - Import from the new record/event modules instead of the monolithic `types.py`.
- `agiwo/agent/trace_writer.py`
  - Import `StepRecord`/`Run`/metrics from `records.py`.
- `console/server/response_serialization.py`
  - Continue consuming stable public types after the internal split.
- `tests/agent/test_compact.py`
  - Add the missing compaction pipeline regression coverage.
- `tests/agent/test_state_tracking.py`
  - Add ledger/mutation coverage.
- `tests/agent/test_definition_contracts.py`
  - Lock the shared child-definition contract.
- `tests/agent/test_agent_tool.py`
  - Lock the nested-agent runtime context boundary.
- `tests/agent/test_storage_serialization.py`
  - Lock the public record/event serialization contract.
- `console/tests/test_response_serialization.py`
  - Verify Console payloads keep working across the internal type split.

### Keep As-Is But Read While Implementing
- `AGENTS.md`
  - The target architecture is already documented here; use it as the acceptance contract.
- `tests/agent/test_run_contracts.py`
  - Existing run-stream lifecycle coverage is the reference behavior for root execution.
- `docs/guides/multi-agent.md`
  - Public composition semantics must stay stable while the internals move.

### Non-Goals
- Do not touch `docs/superpowers/plans/2026-03-26-console-remote-workspace-refactor.md`.
- Do not change the scheduler architecture or Console remote workspace semantics in this refactor.
- Do not add compatibility shims for deleted internal imports outside the stable public `agiwo.agent` facade.

### Task 1: Canonicalize Step Commit And Compaction Recording

**Files:**
- Create: `agiwo/agent/step_pipeline.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/compaction.py`
- Test: `tests/agent/test_compact.py`

- [ ] **Step 1: Write the failing compaction regression test**

```python
import pytest

from agiwo.agent.compaction import _compact
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.run_state import RunContext, SessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage
from agiwo.agent.storage.session import InMemorySessionStorage
from agiwo.llm.base import Model, StreamChunk


class _CompactModel(Model):
    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content='{"summary": "compressed"}')
        yield StreamChunk(finish_reason="stop")


@pytest.mark.asyncio
async def test_compact_uses_the_same_step_commit_pipeline_as_normal_runs(tmp_path):
    step_storage = InMemoryRunStepStorage()
    session_storage = InMemorySessionStorage()
    session_runtime = SessionRuntime(
        session_id="sess-1",
        run_step_storage=step_storage,
        session_storage=session_storage,
    )
    published = []

    async def publish(item):
        published.append(item)

    session_runtime.publish = publish  # type: ignore[method-assign]

    seen_steps = []

    async def on_step(step):
        seen_steps.append(step.name or step.role.value)

    state = RunContext(
        session_runtime=session_runtime,
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent",
        hooks=AgentHooks(on_step=on_step),
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
    )

    metadata = await _compact(
        state,
        _CompactModel(id="compact-model", name="compact-model", provider="openai"),
        session_storage,
        abort_signal=None,
        compact_prompt=None,
        compact_start_seq=1,
        root_path=str(tmp_path),
    )

    steps = await step_storage.get_steps("sess-1", run_id="run-1")

    assert [step.name for step in steps[-2:]] == ["compact_request", "compact"]
    assert seen_steps[-2:] == ["compact_request", "compact"]
    assert [item.type for item in published][-2:] == ["step_completed", "step_completed"]
    assert metadata.get_summary() == "compressed"
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `uv run pytest tests/agent/test_compact.py -k same_step_commit_pipeline -v`
Expected: FAIL because `_compact()` only calls `track_step_state(...)`, so the compaction steps never hit step storage, trace/hooks, or stream publication.

- [ ] **Step 3: Add a single step pipeline owner and route both run loop and compaction through it**

```python
# agiwo/agent/step_pipeline.py
from collections.abc import Sequence
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.run_state import RunContext
from agiwo.agent.state_tracking import track_step_state
from agiwo.agent.types import LLMCallContext, StepCompletedEvent, StepRecord


async def commit_step(
    state: RunContext,
    step: StepRecord,
    *,
    llm: LLMCallContext | None = None,
    append_message: bool = True,
) -> StepRecord:
    track_step_state(state, step, append_message=append_message)
    await state.session_runtime.run_step_storage.save_step(step)
    if state.session_runtime.trace_runtime is not None:
        await state.session_runtime.trace_runtime.on_step(step, llm)
    if state.hooks.on_step is not None:
        await state.hooks.on_step(step)
    await state.session_runtime.publish(
        StepCompletedEvent.from_context(state, step=step),
    )
    return step


def replace_messages(
    state: RunContext,
    messages: Sequence[dict[str, Any]],
    *,
    compact_metadata: CompactMetadata | None = None,
) -> None:
    state.messages = list(messages)
    if compact_metadata is not None:
        state.last_compact_metadata = compact_metadata
```

```python
# agiwo/agent/run_loop.py
from agiwo.agent.step_pipeline import commit_step

async def _run_assistant_turn(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
) -> tuple[StepRecord, LLMCallContext]:
    _apply_steering_messages(state.messages, state.steering_queue)
    if state.hooks.on_before_llm_call:
        modified = await state.hooks.on_before_llm_call(state.messages)
        if modified is not None:
            state.messages = list(modified)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
    )
    await commit_step(state, step, llm=llm_context)
    return step, llm_context
```

```python
# agiwo/agent/compaction.py
from agiwo.agent.step_pipeline import commit_step, replace_messages

await commit_step(state, compact_user_step, append_message=True)

step, llm_context = await stream_assistant_step(
    model,
    state,
    abort_signal,
    messages=list(state.messages),
    tools=None,
)
step.name = "compact"
await commit_step(state, step, llm=llm_context, append_message=False)

replace_messages(
    state,
    compacted_messages,
    compact_metadata=metadata,
)
```

- [ ] **Step 4: Re-run the focused compaction tests**

Run: `uv run pytest tests/agent/test_compact.py -v`
Expected: PASS, including the new regression that proves compaction now persists steps and emits the same hook/stream side effects as a normal assistant turn.

- [ ] **Step 5: Run changed-file lint and commit**

Run: `uv run python scripts/lint.py files agiwo/agent/step_pipeline.py agiwo/agent/run_loop.py agiwo/agent/compaction.py tests/agent/test_compact.py`
Expected: PASS

```bash
git add agiwo/agent/step_pipeline.py agiwo/agent/run_loop.py agiwo/agent/compaction.py tests/agent/test_compact.py
git commit -m "refactor: unify agent step commit pipeline"
```

### Task 2: Separate Run Identity From Mutable Execution Ledger

**Files:**
- Create: `agiwo/agent/run_identity.py`
- Create: `agiwo/agent/run_ledger.py`
- Create: `agiwo/agent/run_mutations.py`
- Modify: `agiwo/agent/run_state.py`
- Modify: `agiwo/agent/state_tracking.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/compaction.py`
- Test: `tests/agent/test_state_tracking.py`

- [ ] **Step 1: Add failing tests that lock the ledger boundary**

```python
from datetime import datetime

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.run_mutations import (
    record_compaction_metadata,
    replace_messages,
    set_termination_reason,
)
from agiwo.agent.types import TerminationReason


def test_run_mutations_only_touch_mutable_ledger_state() -> None:
    state = _make_state()

    replace_messages(state, [{"role": "assistant", "content": "summary"}])
    record_compaction_metadata(
        state,
        CompactMetadata(
            session_id="session-1",
            agent_id="agent-1",
            start_seq=1,
            end_seq=2,
            before_token_estimate=100,
            after_token_estimate=10,
            message_count=2,
            transcript_path="/tmp/t.jsonl",
            analysis={"summary": "summary"},
            created_at=datetime(2026, 3, 26, 12, 0, 0),
        ),
    )
    set_termination_reason(state, TerminationReason.CANCELLED)

    assert state.messages == [{"role": "assistant", "content": "summary"}]
    assert state.last_compact_metadata is not None
    assert state.termination_reason == TerminationReason.CANCELLED
    assert state.run_id == "run-1"
    assert state.session_id == "session-1"
```

- [ ] **Step 2: Run the state-tracking tests and verify they fail**

Run: `uv run pytest tests/agent/test_state_tracking.py -v`
Expected: FAIL because `run_mutations.py` and the split identity/ledger model do not exist yet.

- [ ] **Step 3: Introduce `RunIdentity`, `RunLedger`, and explicit mutation helpers**

```python
# agiwo/agent/run_identity.py
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    agent_id: str
    agent_name: str
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
# agiwo/agent/run_ledger.py
import time
from dataclasses import dataclass, field

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.types import TerminationReason


@dataclass
class RunLedger:
    messages: list[dict] = field(default_factory=list)
    tool_schemas: list[dict] | None = None
    start_time: float = field(default_factory=time.time)
    termination_reason: TerminationReason | None = None
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0
    steps_count: int = 0
    tool_calls_count: int = 0
    assistant_steps_count: int = 0
    response_content: str | None = None
    last_compact_metadata: CompactMetadata | None = None
```

```python
# agiwo/agent/run_mutations.py
from collections.abc import Sequence
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.run_state import RunContext
from agiwo.agent.types import TerminationReason


def replace_messages(state: RunContext, messages: Sequence[dict[str, Any]]) -> None:
    state.ledger.messages = list(messages)


def record_compaction_metadata(
    state: RunContext,
    metadata: CompactMetadata,
) -> None:
    state.ledger.last_compact_metadata = metadata


def set_termination_reason(
    state: RunContext,
    reason: TerminationReason,
) -> None:
    state.ledger.termination_reason = reason
```

```python
# agiwo/agent/run_state.py
from dataclasses import dataclass, field
from typing import Any

from agiwo.agent.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.run_identity import RunIdentity
from agiwo.agent.run_ledger import RunLedger

@dataclass
class RunContext:
    session_runtime: SessionRuntime
    identity: RunIdentity
    ledger: RunLedger = field(default_factory=RunLedger)
    config: AgentOptions = field(default_factory=AgentOptions)
    hooks: AgentHooks = field(default_factory=AgentHooks)

    def __init__(
        self,
        *,
        session_runtime: SessionRuntime,
        run_id: str,
        agent_id: str,
        agent_name: str,
        user_id: str | None = None,
        depth: int = 0,
        parent_run_id: str | None = None,
        timeout_at: float | None = None,
        metadata: dict[str, Any] | None = None,
        config: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
        messages: list[dict] | None = None,
    ) -> None:
        self.session_runtime = session_runtime
        self.identity = RunIdentity(
            run_id=run_id,
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            depth=depth,
            parent_run_id=parent_run_id,
            timeout_at=timeout_at,
            metadata=dict(metadata or {}),
        )
        self.ledger = RunLedger(messages=list(messages or []))
        self.config = config or AgentOptions()
        self.hooks = hooks or AgentHooks()
```

- [ ] **Step 4: Move runtime modules off direct field writes**

```python
# agiwo/agent/state_tracking.py
def track_step_state(
    state: RunContext,
    step: StepRecord,
    *,
    append_message: bool = True,
) -> None:
    ledger = state.ledger
    ledger.steps_count += 1
    metrics = step.metrics
    if metrics is not None:
        if metrics.token_cost is not None:
            ledger.token_cost += metrics.token_cost
        if metrics.total_tokens is not None:
            ledger.total_tokens += metrics.total_tokens
        if metrics.input_tokens is not None:
            ledger.input_tokens += metrics.input_tokens
        if metrics.output_tokens is not None:
            ledger.output_tokens += metrics.output_tokens
        if metrics.cache_read_tokens is not None:
            ledger.cache_read_tokens += metrics.cache_read_tokens
        if metrics.cache_creation_tokens is not None:
            ledger.cache_creation_tokens += metrics.cache_creation_tokens
    if step.is_assistant_step():
        ledger.assistant_steps_count += 1
        if step.content is not None:
            ledger.response_content = step.content
        if step.tool_calls:
            ledger.tool_calls_count += len(step.tool_calls)
    if append_message:
        ledger.messages.append(step_to_message(step))
```

```python
# agiwo/agent/run_loop.py
from agiwo.agent.run_mutations import replace_messages, set_termination_reason

async def _run_assistant_turn(
    *,
    state: RunContext,
    model: Model,
    abort_signal: AbortSignal | None,
) -> tuple[StepRecord, LLMCallContext]:
    _apply_steering_messages(state.messages, state.steering_queue)
    if state.hooks.on_before_llm_call:
        modified = await state.hooks.on_before_llm_call(state.messages)
        if modified is not None:
            replace_messages(state, modified)

    step, llm_context = await stream_assistant_step(
        model,
        state,
        abort_signal,
    )
    await commit_step(state, step, llm=llm_context)
    return step, llm_context


def _handle_cancelled_run(state: RunContext) -> None:
    set_termination_reason(state, TerminationReason.CANCELLED)
```

```python
# agiwo/agent/compaction.py
from agiwo.agent.run_mutations import record_compaction_metadata, replace_messages

replace_messages(state, compacted_messages)
record_compaction_metadata(state, metadata)
```

- [ ] **Step 5: Run tests, changed-file lint, and commit**

Run: `uv run pytest tests/agent/test_state_tracking.py tests/agent/test_compact.py -v`
Expected: PASS

Run: `uv run python scripts/lint.py files agiwo/agent/run_identity.py agiwo/agent/run_ledger.py agiwo/agent/run_mutations.py agiwo/agent/run_state.py agiwo/agent/state_tracking.py agiwo/agent/run_loop.py agiwo/agent/compaction.py tests/agent/test_state_tracking.py`
Expected: PASS

```bash
git add agiwo/agent/run_identity.py agiwo/agent/run_ledger.py agiwo/agent/run_mutations.py agiwo/agent/run_state.py agiwo/agent/state_tracking.py agiwo/agent/run_loop.py agiwo/agent/compaction.py tests/agent/test_state_tracking.py tests/agent/test_compact.py
git commit -m "refactor: split run identity from mutable ledger"
```

### Task 3: Extract Agent Definition Assembly And Shared Child Resolution

**Files:**
- Create: `agiwo/agent/definition.py`
- Modify: `agiwo/agent/agent.py`
- Test: `tests/agent/test_definition_contracts.py`

- [ ] **Step 1: Add a failing contract test for shared child resolution**

```python
import pytest

import agiwo.agent.agent as agent_module


@pytest.mark.asyncio
async def test_run_child_and_create_child_agent_share_the_same_child_resolution() -> None:
    agent = _build_agent()

    resolved = agent_module.resolve_child_definition(
        agent,
        instruction="Handle only child work",
        system_prompt_override=None,
        exclude_tool_names={"dummy_tool"},
        extra_tools=None,
    )
    clone = await agent.create_child_agent(
        child_id="child-agent",
        instruction="Handle only child work",
        exclude_tool_names={"dummy_tool"},
    )

    assert clone.config.system_prompt == resolved.config.system_prompt
    assert {tool.get_name() for tool in clone.tools} == {
        tool.get_name() for tool in resolved.tools
    }
    assert clone.config.options.enable_termination_summary is True
```

- [ ] **Step 2: Run the definition contract tests and verify they fail**

Run: `uv run pytest tests/agent/test_definition_contracts.py -v`
Expected: FAIL because `resolve_child_definition(...)` does not exist and `agent.py` still duplicates child resolution logic.

- [ ] **Step 3: Move definition-scoped assembly and child resolution out of `Agent`**

```python
# agiwo/agent/definition.py
from collections.abc import Sequence
from dataclasses import dataclass, replace as dataclass_replace

from agiwo.agent.config import AgentConfig
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.memory_hooks import DefaultMemoryHook
from agiwo.config.settings import settings
from agiwo.skill import SkillDiscoveryConfig, SkillManager, normalize_skill_dirs
from agiwo.tool.base import BaseTool
from agiwo.tool.registry import DEFAULT_TOOLS, ensure_builtin_tools_loaded
from agiwo.tool.utils import ensure_bash_tool_pair
from agiwo.workspace import build_agent_workspace


@dataclass(frozen=True)
class ResolvedAgentDefinition:
    config: AgentConfig
    hooks: AgentHooks
    tools: Sequence[BaseTool]
    skill_manager: SkillManager | None
    workspace: object


@dataclass(frozen=True)
class ResolvedChildDefinition:
    config: AgentConfig
    tools: Sequence[BaseTool]
    disabled_sdk_tool_names: set[str]


def resolve_agent_definition(
    *,
    config: AgentConfig,
    model,
    hooks: AgentHooks | None,
    tools: list[BaseTool] | None,
    agent_id: str,
    disabled_sdk_tool_names: set[str] | None,
) -> ResolvedAgentDefinition:
    del model
    resolved_hooks = dataclass_replace(hooks or AgentHooks())
    if resolved_hooks.on_memory_retrieve is None:
        memory_hook = DefaultMemoryHook(
            root_path=config.options.get_effective_root_path(),
        )
        resolved_hooks.on_memory_retrieve = memory_hook.retrieve_memories
    skill_manager = None
    if config.options.enable_skill:
        skill_manager = SkillManager(
            SkillDiscoveryConfig(
                configured_dirs=normalize_skill_dirs(config.options.skills_dirs),
                env_dirs=list(settings.skills_dirs or []),
                root_path=config.options.get_effective_root_path(),
            )
        )
    ensure_builtin_tools_loaded()
    disabled_names = set(disabled_sdk_tool_names or set())
    provided_tools = list(tools or [])
    base_tool_names = {tool.get_name() for tool in provided_tools}
    default_tools: list[BaseTool] = []
    for name, tool_cls in DEFAULT_TOOLS.items():
        if name in disabled_names or name in base_tool_names:
            continue
        default_tools.append(tool_cls())
    resolved_tools = ensure_bash_tool_pair([*provided_tools, *default_tools])
    if skill_manager is not None and "skill" not in disabled_names:
        if all(tool.get_name() != "skill" for tool in resolved_tools):
            resolved_tools.append(skill_manager.get_skill_tool())
    return ResolvedAgentDefinition(
        config=config,
        hooks=resolved_hooks,
        tools=tuple(resolved_tools),
        skill_manager=skill_manager,
        workspace=build_agent_workspace(
            root_path=config.options.get_effective_root_path(),
            agent_name=config.name,
            agent_id=agent_id,
        ),
    )


def resolve_child_definition(
    agent: "Agent",
    *,
    instruction: str | None,
    system_prompt_override: str | None,
    exclude_tool_names: set[str] | None,
    extra_tools: list[BaseTool] | None,
) -> ResolvedChildDefinition:
    tool_names_to_skip = agent._normalize_disabled_sdk_tool_names(exclude_tool_names)
    child_tools = [
        tool for tool in agent.tools if tool.get_name() not in tool_names_to_skip
    ]
    if extra_tools:
        child_tools.extend(extra_tools)
    child_options = agent.config.options.model_copy(deep=True)
    child_options.enable_termination_summary = True
    child_config = AgentConfig(
        name=agent.name,
        description=agent.description,
        system_prompt=agent._compose_child_system_prompt(
            system_prompt_override=system_prompt_override,
            instruction=instruction,
        ),
        options=child_options,
    )
    disabled_sdk_tool_names = agent._normalize_disabled_sdk_tool_names(
        {
            name
            for name in set(exclude_tool_names or set())
            if name in agent._exact_tool_disable_set()
        }
    )
    return ResolvedChildDefinition(
        config=child_config,
        tools=tuple(child_tools),
        disabled_sdk_tool_names=disabled_sdk_tool_names,
    )
```

```python
# agiwo/agent/agent.py
from agiwo.agent.definition import (
    ResolvedAgentDefinition,
    resolve_agent_definition,
    resolve_child_definition,
)

self._definition = resolve_agent_definition(
    config=self._config,
    model=self._model,
    hooks=hooks,
    tools=tools,
    agent_id=self._id,
    disabled_sdk_tool_names=disabled_sdk_tool_names,
)
self._hooks = self._definition.hooks
self._skill_manager = self._definition.skill_manager
self._tools = self._definition.tools
self._workspace = self._definition.workspace

resolved_child = resolve_child_definition(
    self,
    instruction=instruction,
    system_prompt_override=system_prompt_override,
    exclude_tool_names=exclude_tool_names,
    extra_tools=extra_tools,
)
```

- [ ] **Step 4: Make both child code paths consume the same resolved child definition**

```python
# agiwo/agent/agent.py
async def run_child(
    self,
    user_input: UserInput,
    *,
    child_id: str,
    session_runtime,
    parent_run_id: str,
    parent_depth: int,
    parent_user_id: str | None,
    parent_timeout_at: float | None,
    parent_metadata: dict[str, Any],
    instruction: str | None = None,
    system_prompt_override: str | None = None,
    exclude_tool_names: set[str] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    metadata_updates: dict | None = None,
    abort_signal: AbortSignal | None = None,
) -> RunOutput:
    resolved_child = resolve_child_definition(
        self,
        instruction=instruction,
        system_prompt_override=system_prompt_override,
        exclude_tool_names=exclude_tool_names,
        extra_tools=None,
    )
    return await execute_run(
        user_input,
        context=context,
        model=self._model,
        system_prompt=await self._build_system_prompt(
            resolved_child.config.system_prompt,
        ),
        tools=list(resolved_child.tools),
        hooks=self._build_hooks(self._hooks),
        options=resolved_child.config.options.model_copy(deep=True),
        abort_signal=child_abort_signal,
        root_path=resolved_child.config.options.get_effective_root_path(),
    )


async def create_child_agent(
    self,
    *,
    child_id: str,
    instruction: str | None = None,
    system_prompt_override: str | None = None,
    exclude_tool_names: set[str] | None = None,
    extra_tools: list[BaseTool] | None = None,
) -> "Agent":
    resolved_child = resolve_child_definition(
        self,
        instruction=instruction,
        system_prompt_override=system_prompt_override,
        exclude_tool_names=exclude_tool_names,
        extra_tools=extra_tools,
    )
    return self.__class__(
        resolved_child.config,
        id=child_id,
        model=self.model,
        tools=list(resolved_child.tools),
        hooks=self._build_hooks(self._hooks),
        disabled_sdk_tool_names=resolved_child.disabled_sdk_tool_names,
    )
```

- [ ] **Step 5: Run definition tests, changed-file lint, and commit**

Run: `uv run pytest tests/agent/test_definition_contracts.py tests/agent/test_child_contracts.py -v`
Expected: PASS

Run: `uv run python scripts/lint.py files agiwo/agent/definition.py agiwo/agent/agent.py tests/agent/test_definition_contracts.py`
Expected: PASS

```bash
git add agiwo/agent/definition.py agiwo/agent/agent.py tests/agent/test_definition_contracts.py
git commit -m "refactor: extract agent definition resolution"
```

### Task 4: Split Runtime Records From Event And Message Adapters

**Files:**
- Create: `agiwo/agent/records.py`
- Create: `agiwo/agent/stream_events.py`
- Create: `agiwo/agent/message_adapters.py`
- Modify: `agiwo/agent/types.py`
- Modify: `agiwo/agent/__init__.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `console/server/response_serialization.py`
- Test: `tests/agent/test_storage_serialization.py`
- Test: `console/tests/test_response_serialization.py`

- [ ] **Step 1: Add a failing public-surface regression test**

```python
from agiwo.agent import StepCompletedEvent, StepRecord, step_to_message
from agiwo.agent.message_adapters import step_to_message as internal_step_to_message
from agiwo.agent.records import StepRecord as InternalStepRecord
from agiwo.agent.stream_events import StepCompletedEvent as InternalStepCompletedEvent


def test_public_agent_exports_remain_stable_after_internal_type_split() -> None:
    assert StepRecord is InternalStepRecord
    assert StepCompletedEvent is InternalStepCompletedEvent
    assert step_to_message is internal_step_to_message
```

- [ ] **Step 2: Run serialization-focused tests and verify they fail**

Run: `uv run pytest tests/agent/test_storage_serialization.py console/tests/test_response_serialization.py -v`
Expected: FAIL because the internal modules do not exist yet.

- [ ] **Step 3: Move records, stream events, and message adapters into focused modules**

```python
# agiwo/agent/message_adapters.py
from typing import Any

from agiwo.agent.records import StepRecord


def step_to_message(step: StepRecord) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": step.role.value}
    if step.content is not None:
        msg["content"] = step.content
    if step.reasoning_content is not None:
        msg["reasoning_content"] = step.reasoning_content
    if step.tool_calls is not None:
        msg["tool_calls"] = step.tool_calls
    if step.tool_call_id is not None:
        msg["tool_call_id"] = step.tool_call_id
    if step.name is not None:
        msg["name"] = step.name
    return msg


def steps_to_messages(steps: list[StepRecord]) -> list[dict[str, Any]]:
    return [step_to_message(step) for step in steps]
```

```python
# agiwo/agent/records.py
from enum import Enum
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from agiwo.agent.input import MessageContent, UserInput


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class StepMetrics:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    token_cost: float = 0.0
    usage_source: str | None = None
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "token_cost": self.token_cost,
            "usage_source": self.usage_source,
            "model_name": self.model_name,
        }


@dataclass
class StepRecord:
    session_id: str
    run_id: str
    sequence: int
    role: MessageRole
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str | None = None
    content: MessageContent | None = None
    content_for_user: str | None = None
    reasoning_content: str | None = None
    user_input: UserInput | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetrics | None = None
    created_at: datetime = field(default_factory=datetime.now)
    parent_run_id: str | None = None
    depth: int = 0
```

```python
# agiwo/agent/stream_events.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias

from agiwo.agent.records import RunMetrics, StepDelta, StepRecord, TerminationReason
from agiwo.utils.serialization import serialize_optional_datetime, to_json


@dataclass(kw_only=True)
class AgentStreamItemBase:
    session_id: str
    run_id: str
    agent_id: str
    parent_run_id: str | None
    depth: int
    timestamp: datetime = field(default_factory=datetime.now)

    def _base_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,  # type: ignore[attr-defined]
            "session_id": self.session_id,
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "parent_run_id": self.parent_run_id,
            "depth": self.depth,
            "timestamp": serialize_optional_datetime(self.timestamp),
        }

    def to_dict(self) -> dict[str, Any]:
        return self._base_dict()

    def to_sse(self) -> str:
        return f"data: {to_json(self)}\\n\\n"


@dataclass(kw_only=True)
class RunStartedEvent(AgentStreamItemBase):
    type: Literal["run_started"] = "run_started"


@dataclass(kw_only=True)
class StepDeltaEvent(AgentStreamItemBase):
    step_id: str
    delta: StepDelta
    type: Literal["step_delta"] = "step_delta"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["step_id"] = self.step_id
        payload["delta"] = self.delta.to_dict()
        return payload


@dataclass(kw_only=True)
class StepCompletedEvent(AgentStreamItemBase):
    step: StepRecord
    type: Literal["step_completed"] = "step_completed"

    def to_dict(self) -> dict[str, Any]:
        payload = self._base_dict()
        payload["step"] = self.step.to_dict()
        return payload


@dataclass(kw_only=True)
class RunCompletedEvent(AgentStreamItemBase):
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    type: Literal["run_completed"] = "run_completed"


@dataclass(kw_only=True)
class RunFailedEvent(AgentStreamItemBase):
    error: str
    type: Literal["run_failed"] = "run_failed"


AgentStreamItem: TypeAlias = (
    StepCompletedEvent
    | RunStartedEvent
    | StepDeltaEvent
    | RunCompletedEvent
    | RunFailedEvent
)
```

```python
# agiwo/agent/types.py
from agiwo.agent.message_adapters import step_to_message, steps_to_messages
from agiwo.agent.records import (
    LLMCallContext,
    MessageRole,
    Run,
    RunMetrics,
    RunOutput,
    RunStatus,
    StepDelta,
    StepMetrics,
    StepRecord,
    TerminationReason,
)
from agiwo.agent.stream_events import (
    AgentStreamItem,
    AgentStreamItemBase,
    RunCompletedEvent,
    RunFailedEvent,
    RunStartedEvent,
    StepCompletedEvent,
    StepDeltaEvent,
)
```

- [ ] **Step 4: Update internal consumers to import the focused modules directly**

```python
# agiwo/agent/storage/serialization.py
from agiwo.agent.records import Run, StepRecord
```

```python
# agiwo/agent/trace_writer.py
from agiwo.agent.records import LLMCallContext, Run, StepRecord
```

```python
# console/server/response_serialization.py
from agiwo.agent import StepCompletedEvent, StepRecord
```

- [ ] **Step 5: Run serialization tests, changed-file lint, and commit**

Run: `uv run pytest tests/agent/test_storage_serialization.py console/tests/test_response_serialization.py -v`
Expected: PASS

Run: `uv run python scripts/lint.py files agiwo/agent/records.py agiwo/agent/stream_events.py agiwo/agent/message_adapters.py agiwo/agent/types.py agiwo/agent/__init__.py agiwo/agent/storage/serialization.py agiwo/agent/trace_writer.py console/server/response_serialization.py tests/agent/test_storage_serialization.py console/tests/test_response_serialization.py`
Expected: PASS

```bash
git add agiwo/agent/records.py agiwo/agent/stream_events.py agiwo/agent/message_adapters.py agiwo/agent/types.py agiwo/agent/__init__.py agiwo/agent/storage/serialization.py agiwo/agent/trace_writer.py console/server/response_serialization.py tests/agent/test_storage_serialization.py console/tests/test_response_serialization.py
git commit -m "refactor: split agent records from adapters"
```

### Task 5: Tighten The Nested-Agent Tool Boundary

**Files:**
- Create: `agiwo/agent/runtime_tool_context.py`
- Modify: `agiwo/tool/context.py`
- Modify: `agiwo/agent/tool_executor.py`
- Modify: `agiwo/agent/child.py`
- Test: `tests/agent/test_agent_tool.py`

- [ ] **Step 1: Add failing tests for the explicit runtime tool context**

```python
import pytest

from agiwo.agent.runtime_tool_context import AgentToolContext
from agiwo.tool.context import ToolContext


@pytest.mark.asyncio
async def test_agent_tool_rejects_plain_tool_context_without_runtime_bridge():
    tool = AgentTool(FakeAgent(id="agent-child", name="child"))

    result = await tool.execute({"task": "hello"}, ToolContext(session_id="sess-1"))

    assert result.is_success is False
    assert result.error == "AgentTool requires agent runtime context"


def test_agent_tool_context_can_be_built_from_run_context() -> None:
    runtime_context = AgentToolContext.from_run_context(_make_context(), timeout_at=1.0)

    assert runtime_context.session_id == "sess-1"
    assert runtime_context.parent_run_id == "run-1"
    assert runtime_context.depth == 0
```

- [ ] **Step 2: Run the agent-tool tests and verify they fail**

Run: `uv run pytest tests/agent/test_agent_tool.py -v`
Expected: FAIL because `AgentToolContext` does not exist and `ToolContext` still carries runtime-only fields.

- [ ] **Step 3: Remove runtime leakage from plain `ToolContext` and add a runtime-only subtype**

```python
# agiwo/tool/context.py
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    agent_id: str | None = None
    agent_name: str | None = None
    user_id: str | None = None
    timeout_at: float | None = None
    depth: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    gate_checked: bool = False
```

```python
# agiwo/agent/runtime_tool_context.py
from dataclasses import dataclass

from agiwo.agent.run_state import RunContext, SessionRuntime
from agiwo.tool.context import ToolContext


@dataclass(frozen=True)
class AgentToolContext(ToolContext):
    parent_run_id: str
    session_runtime: SessionRuntime

    @classmethod
    def from_run_context(
        cls,
        ctx: RunContext,
        *,
        timeout_at: float | None,
        gate_checked: bool = True,
    ) -> "AgentToolContext":
        return cls(
            session_id=ctx.session_id,
            agent_id=ctx.agent_id,
            agent_name=ctx.agent_name,
            user_id=ctx.user_id,
            timeout_at=timeout_at,
            depth=ctx.depth,
            parent_run_id=ctx.run_id,
            session_runtime=ctx.session_runtime,
            metadata=dict(ctx.metadata),
            gate_checked=gate_checked,
        )
```

- [ ] **Step 4: Teach the tool executor and `AgentTool` to use the explicit runtime subtype**

```python
# agiwo/agent/tool_executor.py
from agiwo.agent.child import AgentTool
from agiwo.agent.runtime_tool_context import AgentToolContext

def _build_tool_context(ctx: RunContext, tool: BaseTool) -> ToolContext:
    timeout_seconds = tool.timeout_seconds or _DEFAULT_TIMEOUT_SECONDS
    timeout_at = ctx.timeout_at
    if timeout_seconds:
        timeout_at = time.time() + timeout_seconds
    if isinstance(tool, AgentTool):
        return AgentToolContext.from_run_context(
            ctx,
            timeout_at=timeout_at,
            gate_checked=True,
        )
    return ToolContext(
        session_id=ctx.session_id,
        agent_id=ctx.agent_id,
        agent_name=ctx.agent_name,
        user_id=ctx.user_id,
        timeout_at=timeout_at,
        depth=ctx.depth,
        metadata=dict(ctx.metadata),
        gate_checked=True,
    )
```

```python
# agiwo/agent/child.py
from agiwo.agent.runtime_tool_context import AgentToolContext

if not isinstance(context, AgentToolContext):
    return ToolResult.failed(
        tool_name=self.get_name(),
        error="AgentTool requires agent runtime context",
        tool_call_id=str(parameters.get("tool_call_id", "")),
        input_args=parameters,
        start_time=start_time,
    )
```

- [ ] **Step 5: Run tests, changed-file lint, and commit**

Run: `uv run pytest tests/agent/test_agent_tool.py tests/agent/test_child_contracts.py -v`
Expected: PASS

Run: `uv run python scripts/lint.py files agiwo/agent/runtime_tool_context.py agiwo/tool/context.py agiwo/agent/tool_executor.py agiwo/agent/child.py tests/agent/test_agent_tool.py`
Expected: PASS

```bash
git add agiwo/agent/runtime_tool_context.py agiwo/tool/context.py agiwo/agent/tool_executor.py agiwo/agent/child.py tests/agent/test_agent_tool.py
git commit -m "refactor: make agent tool runtime context explicit"
```

## Final Verification

- [ ] **Step 1: Run the full SDK + Console lint stack required by the repo**

Run: `uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/`
Expected: PASS

Run: `uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/`
Expected: PASS

Run: `uv run python scripts/lint.py imports`
Expected: PASS

Run: `uv run python scripts/repo_guard.py`
Expected: PASS, with no new `agiwo/agent`-related warnings beyond any pre-existing unrelated budget warnings.

- [ ] **Step 2: Run the focused SDK and downstream Console tests**

Run: `uv run pytest tests/agent -v`
Expected: PASS

Run: `cd console && uv run pytest tests/test_response_serialization.py -v`
Expected: PASS

- [ ] **Step 3: Update docs if the extracted module boundaries changed the public architecture wording**

```markdown
# AGENTS.md
- `agiwo/agent/` internal execution state is split into definition resolution, run state, step pipeline, and stream/message adapter modules.
- `agiwo/agent/types.py` is a public facade; internal modules should depend on focused submodules instead.
```

- [ ] **Step 4: Create the integration commit**

```bash
git add AGENTS.md agiwo/agent console/server tests console/tests
git commit -m "refactor: clarify agent runtime boundaries"
```
