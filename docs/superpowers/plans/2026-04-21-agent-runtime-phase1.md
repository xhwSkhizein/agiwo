# Agent Runtime Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current `agiwo.agent` runtime internals with a `RunLog`-based execution core while keeping `Agent.start()/run()/run_stream()/run_child()` usable and the repository green.

**Architecture:** Phase 1 replaces `Run`, `StepRecord`, `AgentHooks`, and `RunStepStorage` as canonical runtime models inside `agiwo.agent`. A new `RunLoopOrchestrator` writes strongly typed `RunLog` entries through `RunLogStorage`, and trace/stream/read views are rebuilt from those entries. Scheduler and console do not get their full architecture migration in this phase, but they must be updated enough to read the new runtime views without depending on deleted source models.

**Tech Stack:** Python 3.10+, dataclasses, existing `Model` and `BaseTool` abstractions, SQLite/in-memory storage, existing trace and SSE protocols

---

## Scope Check

The design spec covers three implementation waves:

1. Phase 1: replace `agiwo.agent` runtime internals
2. Phase 2: migrate scheduler runtime semantics
3. Phase 3: migrate console runtime/query surfaces completely

This plan only covers Phase 1. It includes the minimum console/query touch points needed to keep tests and public views working after `Run`, `StepRecord`, and `RunStepStorage` disappear.

## File Structure

### Create

- `agiwo/agent/models/log.py`
- `agiwo/agent/runtime/run_engine.py`
- `agiwo/agent/runtime/hook_dispatcher.py`
- `agiwo/agent/runtime/state_writer.py`
- `tests/agent/test_run_log_models.py`
- `tests/agent/test_hook_dispatcher.py`
- `tests/agent/test_run_log_storage.py`
- `tests/agent/test_run_engine.py`

### Modify

- `agiwo/agent/__init__.py`
- `agiwo/agent/agent.py`
- `agiwo/agent/compaction.py`
- `agiwo/agent/definition.py`
- `agiwo/agent/hooks.py`
- `agiwo/agent/models/__init__.py`
- `agiwo/agent/models/config.py`
- `agiwo/agent/models/run.py`
- `agiwo/agent/models/step.py`
- `agiwo/agent/models/stream.py`
- `agiwo/agent/prompt.py`
- `agiwo/agent/review/executor.py`
- `agiwo/agent/run_loop.py`
- `agiwo/agent/runtime/__init__.py`
- `agiwo/agent/runtime/context.py`
- `agiwo/agent/runtime/session.py`
- `agiwo/agent/storage/__init__.py`
- `agiwo/agent/storage/base.py`
- `agiwo/agent/storage/factory.py`
- `agiwo/agent/storage/serialization.py`
- `agiwo/agent/storage/sqlite.py`
- `agiwo/agent/termination/limits.py`
- `agiwo/agent/termination/summarizer.py`
- `agiwo/agent/trace_writer.py`
- `console/server/response_serialization.py`
- `console/server/routers/sessions.py`
- `console/server/services/metrics.py`
- `console/server/services/runtime/session_view_service.py`
- `tests/agent/test_compact.py`
- `tests/agent/test_definition_contracts.py`
- `tests/agent/test_memory_hooks.py`
- `tests/agent/test_step_back.py`
- `tests/agent/test_run_contracts.py`
- `tests/agent/test_run_loop_contracts.py`
- `tests/agent/test_state_tracking.py`
- `tests/agent/test_storage_factory.py`
- `tests/agent/test_storage_serialization.py`
- `tests/agent/test_termination.py`
- `tests/agent/test_user_input_serialization.py`
- `console/tests/test_session_summary.py`
- `console/tests/test_sessions_api.py`

### Delete

- `Run` class from `agiwo/agent/models/run.py`
- `StepRecord` class from `agiwo/agent/models/step.py`
- `AgentHooks` dataclass and old callback typedefs from `agiwo/agent/hooks.py`
- `RunStepStorage` interface and `InMemoryRunStepStorage` naming from `agiwo/agent/storage/base.py`

## Task 1: Introduce `RunLog` Models And Public Exports

**Files:**
- Create: `agiwo/agent/models/log.py`
- Modify: `agiwo/agent/models/run.py`
- Modify: `agiwo/agent/models/step.py`
- Modify: `agiwo/agent/models/__init__.py`
- Modify: `agiwo/agent/__init__.py`
- Test: `tests/agent/test_run_log_models.py`

- [ ] **Step 1: Write the failing model/export test**

```python
# tests/agent/test_run_log_models.py
from agiwo.agent import (
    AssistantStepCommitted,
    HookFailed,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
)


def test_run_log_entries_are_public_and_typed() -> None:
    entry = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )

    assert entry.kind is RunLogEntryKind.RUN_STARTED
    assert HookFailed.__name__ == "HookFailed"
    assert AssistantStepCommitted.__name__ == "AssistantStepCommitted"
    assert TerminationDecided.__name__ == "TerminationDecided"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/agent/test_run_log_models.py -v`
Expected: FAIL with `ImportError` because `RunLogEntryKind` and the new entry classes do not exist yet.

- [ ] **Step 3: Add `RunLog` entry families and slim old run/step models**

```python
# agiwo/agent/models/log.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agiwo.config.termination import TerminationReason


class RunLogEntryKind(str, Enum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    RUN_FAILED = "run_failed"
    CONTEXT_ASSEMBLED = "context_assembled"
    MESSAGES_REBUILT = "messages_rebuilt"
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    USER_STEP_COMMITTED = "user_step_committed"
    ASSISTANT_STEP_COMMITTED = "assistant_step_committed"
    TOOL_STEP_COMMITTED = "tool_step_committed"
    COMPACTION_APPLIED = "compaction_applied"
    STEP_BACK_APPLIED = "step_back_applied"
    TERMINATION_DECIDED = "termination_decided"
    HOOK_FAILED = "hook_failed"


@dataclass(frozen=True)
class RunLogEntry:
    sequence: int
    session_id: str
    run_id: str
    agent_id: str
    kind: RunLogEntryKind
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class RunStarted(RunLogEntry):
    user_input: Any = None

    def __init__(self, *, sequence: int, session_id: str, run_id: str, agent_id: str, user_input: Any) -> None:
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "run_id", run_id)
        object.__setattr__(self, "agent_id", agent_id)
        object.__setattr__(self, "kind", RunLogEntryKind.RUN_STARTED)
        object.__setattr__(self, "created_at", datetime.now(timezone.utc))
        object.__setattr__(self, "user_input", user_input)


@dataclass(frozen=True)
class AssistantStepCommitted(RunLogEntry):
    content: Any
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class TerminationDecided(RunLogEntry):
    termination_reason: TerminationReason
    phase: str
    source: str


@dataclass(frozen=True)
class HookFailed(RunLogEntry):
    phase: str
    hook_name: str
    error: str
```

```python
# agiwo/agent/models/run.py
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


@dataclass
class RunView:
    run_id: str
    session_id: str
    agent_id: str
    status: str
    response: str | None = None
    termination_reason: TerminationReason | None = None
    metrics: RunMetrics | None = None
    last_user_input: Any = None


@dataclass
class RunOutput:
    session_id: str | None = None
    run_id: str | None = None
    response: str | None = None
    metrics: RunMetrics | None = None
    termination_reason: TerminationReason | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

```python
# agiwo/agent/models/step.py
class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class StepMetrics:
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_read_tokens: int | None = None
    cache_creation_tokens: int | None = None
    token_cost: float | None = None
    usage_source: str | None = None
    model_name: str | None = None
    provider: str | None = None
    first_token_latency_ms: float | None = None


@dataclass
class StepView:
    sequence: int
    session_id: str
    run_id: str
    agent_id: str
    role: MessageRole
    content: Any = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    metrics: StepMetrics | None = None
```

- [ ] **Step 4: Export the new public types and rerun the model test**

```python
# agiwo/agent/models/__init__.py
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    HookFailed,
    RunLogEntry,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
)
```

```python
# agiwo/agent/__init__.py
from agiwo.agent.models.log import (
    AssistantStepCommitted,
    HookFailed,
    RunLogEntry,
    RunLogEntryKind,
    RunStarted,
    TerminationDecided,
)
```

Run: `uv run pytest tests/agent/test_run_log_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/models/log.py agiwo/agent/models/run.py agiwo/agent/models/step.py agiwo/agent/models/__init__.py agiwo/agent/__init__.py tests/agent/test_run_log_models.py
git commit -m "refactor: add run log domain models"
```

## Task 2: Replace `AgentHooks` With Phase-Based Hook Handlers

**Files:**
- Modify: `agiwo/agent/hooks.py`
- Modify: `agiwo/agent/definition.py`
- Modify: `agiwo/agent/agent.py`
- Test: `tests/agent/test_hook_dispatcher.py`
- Test: `tests/agent/test_memory_hooks.py`

- [ ] **Step 1: Write the failing hook contract test**

```python
# tests/agent/test_hook_dispatcher.py
from agiwo.agent.hooks import HookCapability, HookPhase, HookRegistration


def test_hook_registration_exposes_phase_and_capability() -> None:
    registration = HookRegistration(
        phase=HookPhase.BEFORE_LLM,
        capability=HookCapability.TRANSFORM,
        handler_name="rewrite_messages",
    )

    assert registration.phase is HookPhase.BEFORE_LLM
    assert registration.capability is HookCapability.TRANSFORM
    assert registration.critical is False
```

- [ ] **Step 2: Run the new hook test**

Run: `uv run pytest tests/agent/test_hook_dispatcher.py -v`
Expected: FAIL with `ImportError` because the new phase-based hook types do not exist.

- [ ] **Step 3: Replace the old dataclass hook surface**

```python
# agiwo/agent/hooks.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Protocol


class HookPhase(str, Enum):
    PREPARE = "prepare"
    ASSEMBLE_CONTEXT = "assemble_context"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL_BATCH = "before_tool_batch"
    AFTER_TOOL_BATCH = "after_tool_batch"
    BEFORE_COMPACTION = "before_compaction"
    AFTER_COMPACTION = "after_compaction"
    BEFORE_REVIEW = "before_review"
    AFTER_STEP_BACK = "after_step_back"
    BEFORE_TERMINATION = "before_termination"
    AFTER_TERMINATION = "after_termination"
    AFTER_STEP_COMMIT = "after_step_commit"
    RUN_FINALIZED = "run_finalized"
    MEMORY_PERSIST = "memory_persist"
    COMPACTION_FAILED = "compaction_failed"


class HookCapability(str, Enum):
    OBSERVE_ONLY = "observe_only"
    TRANSFORM = "transform"
    DECISION_SUPPORT = "decision_support"


class PhaseHook(Protocol):
    async def __call__(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError


@dataclass(frozen=True)
class HookRegistration:
    phase: HookPhase
    capability: HookCapability
    handler_name: str
    handler: PhaseHook | None = None
    order: int = 100
    critical: bool = False


@dataclass
class HookRegistry:
    registrations: list[HookRegistration] = field(default_factory=list)

    def for_phase(self, phase: HookPhase) -> list[HookRegistration]:
        return sorted(
            [item for item in self.registrations if item.phase == phase],
            key=lambda item: item.order,
        )
```

- [ ] **Step 4: Thread the new hook type through `Agent` and preserve default memory retrieval**

```python
# agiwo/agent/definition.py
def build_hook_registry(
    root_path: str,
    hooks: HookRegistry | list[HookRegistration] | None,
) -> HookRegistry:
    registry = hooks if isinstance(hooks, HookRegistry) else HookRegistry(hooks or [])
    if not any(item.handler_name == "default_memory_retrieve" for item in registry.registrations):
        async def default_memory_retrieve_handler(payload: dict[str, Any]) -> dict[str, Any]:
            default_hook = DefaultMemoryHook(root_path=root_path)
            memories = await default_hook.retrieve_memories(payload["user_input"], payload["context"])
            updated = dict(payload)
            updated["memories"] = memories
            return updated

        registry.registrations.append(
            HookRegistration(
                phase=HookPhase.ASSEMBLE_CONTEXT,
                capability=HookCapability.TRANSFORM,
                handler_name="default_memory_retrieve",
                handler=default_memory_retrieve_handler,
            )
        )
    return registry
```

```python
# agiwo/agent/agent.py
def __init__(
    self,
    config: AgentConfig,
    *,
    model: Model,
    tools: list[BaseTool] | None = None,
    hooks: HookRegistry | list[HookRegistration] | None = None,
    id: str | None = None,
) -> None:
    self._config = copy.deepcopy(config)
    self._id = id or _generate_default_id(self._config.name)
    self._model = model
    self._hooks = build_hook_registry(
        self._config.options.get_effective_root_path(),
        hooks,
    )
    self._extra_tools = tuple(tools or [])
```

Run: `uv run pytest tests/agent/test_hook_dispatcher.py tests/agent/test_memory_hooks.py -v`
Expected: PASS with updated assertions that inspect `HookRegistry` instead of `AgentHooks`.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/hooks.py agiwo/agent/definition.py agiwo/agent/agent.py tests/agent/test_hook_dispatcher.py tests/agent/test_memory_hooks.py
git commit -m "refactor: replace agent hooks with phase registry"
```

## Task 3: Replace `RunStepStorage` With `RunLogStorage`

**Files:**
- Modify: `agiwo/agent/models/config.py`
- Modify: `agiwo/agent/storage/base.py`
- Modify: `agiwo/agent/storage/factory.py`
- Modify: `agiwo/agent/runtime/session.py`
- Modify: `agiwo/agent/storage/__init__.py`
- Test: `tests/agent/test_run_log_storage.py`
- Test: `tests/agent/test_storage_factory.py`

- [ ] **Step 1: Write the failing storage test**

```python
# tests/agent/test_run_log_storage.py
from agiwo.agent.models.log import RunStarted
from agiwo.agent.storage.base import InMemoryRunLogStorage


async def test_in_memory_run_log_storage_appends_and_replays() -> None:
    storage = InMemoryRunLogStorage()
    entry = RunStarted(
        sequence=1,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-1",
        user_input="hello",
    )

    await storage.append_entries([entry])

    replay = await storage.list_entries(session_id="sess-1")
    assert replay == [entry]
```

- [ ] **Step 2: Run the storage test to verify it fails**

Run: `uv run pytest tests/agent/test_run_log_storage.py -v`
Expected: FAIL because `InMemoryRunLogStorage` and `append_entries()` do not exist.

- [ ] **Step 3: Replace the storage interface and config names**

```python
# agiwo/agent/models/config.py
@dataclass
class RunLogStorageConfig:
    storage_type: Literal["memory", "sqlite", "mongodb"] = "memory"
    config: dict[str, Any] = field(default_factory=dict)


class AgentStorageOptions(BaseModel):
    run_log_storage: RunLogStorageConfig = Field(default_factory=RunLogStorageConfig)
    trace_storage: TraceStorageConfig = Field(default_factory=TraceStorageConfig)
```

```python
# agiwo/agent/storage/base.py
class RunLogStorage(ABC):
    @abstractmethod
    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        raise NotImplementedError

    @abstractmethod
    async def list_entries(
        self,
        *,
        session_id: str,
        run_id: str | None = None,
        agent_id: str | None = None,
        after_sequence: int | None = None,
        limit: int = 1000,
    ) -> list[RunLogEntry]:
        raise NotImplementedError

    @abstractmethod
    async def allocate_sequence(self, session_id: str) -> int:
        raise NotImplementedError


class InMemoryRunLogStorage(RunLogStorage):
    def __init__(self) -> None:
        self._entries: dict[str, list[RunLogEntry]] = {}
        self._sequence_counters: dict[str, int] = {}
```

- [ ] **Step 4: Update factory/session wiring and rerun storage tests**

```python
# agiwo/agent/storage/factory.py
def create_run_log_storage(config: RunLogStorageConfig) -> RunLogStorage:
    if config.storage_type == "memory":
        return InMemoryRunLogStorage()
    if config.storage_type == "sqlite":
        return SQLiteRunLogStorage(db_path=_resolve_db_path(config.config["db_path"]))
    raise ValueError(f"Unsupported run_log storage type: {config.storage_type}")
```

```python
# agiwo/agent/runtime/session.py
class SessionRuntime:
    def __init__(
        self,
        *,
        session_id: str,
        run_log_storage: RunLogStorage,
        trace_runtime: AgentTraceCollector | None = None,
        abort_signal: AbortSignal | None = None,
        steering_queue: asyncio.Queue[object] | None = None,
    ) -> None:
        self.run_log_storage = run_log_storage
```

Run: `uv run pytest tests/agent/test_run_log_storage.py tests/agent/test_storage_factory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/models/config.py agiwo/agent/storage/base.py agiwo/agent/storage/factory.py agiwo/agent/storage/__init__.py agiwo/agent/runtime/session.py tests/agent/test_run_log_storage.py tests/agent/test_storage_factory.py
git commit -m "refactor: replace run step storage with run log storage"
```

## Task 4: Introduce `RunLoopOrchestrator`, `HookRegistry`, And `RunStateWriter`

**Files:**
- Create: `agiwo/agent/runtime/run_engine.py`
- Create: `agiwo/agent/runtime/hook_dispatcher.py`
- Create: `agiwo/agent/runtime/state_writer.py`
- Modify: `agiwo/agent/run_loop.py`
- Modify: `agiwo/agent/runtime/context.py`
- Modify: `agiwo/agent/runtime/__init__.py`
- Test: `tests/agent/test_run_engine.py`
- Test: `tests/agent/test_run_loop_contracts.py`

- [ ] **Step 1: Write the failing engine smoke test**

```python
# tests/agent/test_run_engine.py
from agiwo.agent.models.log import RunLogEntryKind


async def test_run_engine_writes_started_llm_and_finished_entries(fake_runtime, fake_model) -> None:
    result = await fake_runtime.run_engine.execute("hello")

    assert result.response == "ok"
    entries = await fake_runtime.session_runtime.run_log_storage.list_entries(session_id="sess-1")
    assert [entry.kind for entry in entries][:3] == [
        RunLogEntryKind.RUN_STARTED,
        RunLogEntryKind.LLM_CALL_COMPLETED,
        RunLogEntryKind.RUN_FINISHED,
    ]
```

- [ ] **Step 2: Run the engine test to verify it fails**

Run: `uv run pytest tests/agent/test_run_engine.py -v`
Expected: FAIL because `RunLoopOrchestrator` does not exist and `run_loop.py` still owns execution.

- [ ] **Step 3: Add the new runtime components**

```python
# agiwo/agent/runtime/hook_dispatcher.py
class HookRegistry:
    def __init__(self, registry: HookRegistry, writer: "RunStateWriter") -> None:
        self._registry = registry
        self._writer = writer

    async def dispatch(self, phase: HookPhase, payload: dict[str, Any]) -> dict[str, Any]:
        current = dict(payload)
        for registration in self._registry.for_phase(phase):
            try:
                if registration.handler is None:
                    continue
                result = await registration.handler(current)
            except Exception as error:
                await self._writer.record_hook_failure(phase=phase, registration=registration, error=error)
                if registration.critical:
                    raise
                continue
            if registration.capability is HookCapability.TRANSFORM and result is not None:
                current = result
        return current
```

```python
# agiwo/agent/runtime/state_writer.py
class RunStateWriter:
    def __init__(self, context: RunContext) -> None:
        self._context = context

    async def append_entry(self, entry: RunLogEntry) -> None:
        await self._context.session_runtime.run_log_storage.append_entries([entry])

    async def rebuild_messages(self, messages: list[dict[str, Any]], *, reason: str) -> None:
        self._context.ledger.messages = list(messages)
        await self.append_entry(
            MessagesRebuilt(
                sequence=await self._context.session_runtime.allocate_sequence(),
                session_id=self._context.session_id,
                run_id=self._context.run_id,
                agent_id=self._context.agent_id,
                reason=reason,
                messages=list(messages),
            )
        )
```

```python
# agiwo/agent/runtime/run_engine.py
class RunLoopOrchestrator:
    def __init__(self, context: RunContext, runtime: RunRuntime) -> None:
        self._context = context
        self._runtime = runtime
        self._writer = RunStateWriter(context)
        self._hooks = runtime.hooks

    async def execute(self, user_input: UserInput, *, system_prompt: str, pending_tool_calls: list[dict] | None = None) -> RunOutput:
        await self._writer.append_entry(
            RunStarted(
                sequence=await self._context.session_runtime.allocate_sequence(),
                session_id=self._context.session_id,
                run_id=self._context.run_id,
                agent_id=self._context.agent_id,
                user_input=user_input,
            )
        )
        return await self._run_loop(user_input=user_input, system_prompt=system_prompt, pending_tool_calls=pending_tool_calls)
```

```python
# agiwo/agent/runtime/context.py
@dataclass
class RunRuntime:
    session_runtime: SessionRuntime
    config: AgentOptions
    hooks: HookRegistry
    model: Model
    tools_map: dict[str, BaseTool]
    abort_signal: AbortSignal | None
    root_path: str

    @classmethod
    def from_inputs(
        cls,
        *,
        session_runtime: SessionRuntime,
        config: AgentOptions,
        hooks: HookRegistry,
        model: Model,
        tools: tuple[BaseTool, ...],
        abort_signal: AbortSignal | None,
        root_path: str | None,
    ) -> "RunRuntime":
        return cls(
            session_runtime=session_runtime,
            config=config,
            hooks=hooks,
            model=model,
            tools_map={tool.name: tool for tool in tools},
            abort_signal=abort_signal,
            root_path=root_path or config.get_effective_root_path(),
        )
```

- [ ] **Step 4: Make `run_loop.py` a thin facade and rerun engine/contract tests**

```python
# agiwo/agent/run_loop.py
async def execute_run(
    user_input: UserInput,
    *,
    context: RunContext,
    system_prompt: str,
    model: Model,
    tools: tuple[BaseTool, ...],
    options: AgentOptions | None = None,
    hooks: HookRegistry | list[HookRegistration] | None = None,
    pending_tool_calls: list[dict] | None = None,
    abort_signal: AbortSignal | None = None,
    root_path: str | None = None,
) -> RunOutput:
    runtime = RunRuntime.from_inputs(
        session_runtime=context.session_runtime,
        config=options or AgentOptions(),
        hooks=build_hook_registry(root_path or settings.root_path, hooks),
        model=model,
        tools=tools,
        abort_signal=abort_signal,
        root_path=root_path,
    )
    return await RunLoopOrchestrator(context, runtime).execute(
        user_input,
        system_prompt=system_prompt,
        pending_tool_calls=pending_tool_calls,
    )
```

Run: `uv run pytest tests/agent/test_run_engine.py tests/agent/test_run_loop_contracts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/runtime/run_engine.py agiwo/agent/runtime/hook_dispatcher.py agiwo/agent/runtime/state_writer.py agiwo/agent/run_loop.py agiwo/agent/runtime/context.py agiwo/agent/runtime/__init__.py tests/agent/test_run_engine.py tests/agent/test_run_loop_contracts.py
git commit -m "refactor: route agent execution through run engine"
```

## Task 5: Move `termination`, `compaction`, `step-back`, Trace, And Stream Onto `RunLog`

**Files:**
- Modify: `agiwo/agent/compaction.py`
- Modify: `agiwo/agent/review/executor.py`
- Modify: `agiwo/agent/termination/limits.py`
- Modify: `agiwo/agent/termination/summarizer.py`
- Modify: `agiwo/agent/trace_writer.py`
- Modify: `agiwo/agent/models/stream.py`
- Test: `tests/agent/test_compact.py`
- Test: `tests/agent/test_step_back.py`
- Test: `tests/agent/test_termination.py`
- Test: `tests/agent/test_run_contracts.py`

- [ ] **Step 1: Write failing policy/trace expectations**

```python
# tests/agent/test_step_back.py
from agiwo.agent.models.log import RunLogEntryKind


async def test_step_back_is_recorded_as_run_log_entry(storage, runtime) -> None:
    await runtime.trigger_step_back()

    entries = await storage.list_entries(session_id="sess-1")
    assert RunLogEntryKind.STEP_BACK_APPLIED in [entry.kind for entry in entries]
```

```python
# tests/agent/test_termination.py
async def test_termination_entry_records_phase_and_reason(runtime, storage) -> None:
    await runtime.run_until_terminated()
    entry = await storage.get_latest_termination(session_id="sess-1", run_id="run-1")
    assert entry.phase == "after_llm"
    assert entry.termination_reason is not None
```

- [ ] **Step 2: Run the focused tests**

Run: `uv run pytest tests/agent/test_compact.py tests/agent/test_step_back.py tests/agent/test_termination.py tests/agent/test_run_contracts.py -v`
Expected: FAIL because these paths still update step/run storage and trace/stream side effects directly.

- [ ] **Step 3: Make policies and view builders consume/write `RunLog`**

```python
# agiwo/agent/compaction.py
await writer.append_entry(
    CompactionApplied(
        sequence=await context.session_runtime.allocate_sequence(),
        session_id=context.session_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
        start_sequence=compact_start_seq,
        end_sequence=metadata.end_seq,
        transcript_path=metadata.transcript_path,
        summary=metadata.get_summary(),
    )
)
```

```python
# agiwo/agent/review/executor.py
await writer.append_entry(
    StepBackApplied(
        sequence=await context.session_runtime.allocate_sequence(),
        session_id=context.session_id,
        run_id=context.run_id,
        agent_id=context.agent_id,
        affected_step_ids=[item.step_id for item in affected_steps],
        feedback=feedback,
        replacement=condensed_content,
    )
)
```

```python
# agiwo/agent/trace_writer.py
class TraceBuilder:
    async def build_from_entries(self, entries: list[RunLogEntry]) -> Trace:
        trace = Trace(trace_id=str(uuid4()))
        for entry in entries:
            if entry.kind is RunLogEntryKind.LLM_CALL_COMPLETED:
                trace.add_span(
                    Span(
                        trace_id=trace.trace_id,
                        kind=SpanKind.LLM_CALL,
                        name="llm",
                        run_id=entry.run_id,
                        attributes={"agent_id": entry.agent_id},
                    )
                )
            if entry.kind is RunLogEntryKind.TOOL_STEP_COMMITTED:
                trace.add_span(
                    Span(
                        trace_id=trace.trace_id,
                        kind=SpanKind.TOOL_CALL,
                        name=getattr(entry, "name", "tool"),
                        run_id=entry.run_id,
                        attributes={"agent_id": entry.agent_id},
                    )
                )
        return trace
```

```python
# agiwo/agent/models/stream.py
def stream_items_from_entries(entries: list[RunLogEntry]) -> list[AgentStreamItem]:
    items: list[AgentStreamItem] = []
    for entry in entries:
        if entry.kind is RunLogEntryKind.USER_STEP_COMMITTED:
            items.append(
                StepCompletedEvent(type="step_completed", step=_step_from_entry(entry))
            )
        elif entry.kind is RunLogEntryKind.ASSISTANT_STEP_COMMITTED:
            items.append(
                StepCompletedEvent(type="step_completed", step=_step_from_entry(entry))
            )
        elif entry.kind is RunLogEntryKind.TERMINATION_DECIDED:
            items.append(
                RunCompletedEvent(
                    type="run_completed",
                    run_id=entry.run_id,
                    session_id=entry.session_id,
                    termination_reason=entry.termination_reason,
                )
            )
    return items
```

- [ ] **Step 4: Rerun the focused policy tests**

Run: `uv run pytest tests/agent/test_compact.py tests/agent/test_step_back.py tests/agent/test_termination.py tests/agent/test_run_contracts.py -v`
Expected: PASS with assertions updated to read `RunLog`-backed views and stream items.

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/compaction.py agiwo/agent/review/executor.py agiwo/agent/termination/limits.py agiwo/agent/termination/summarizer.py agiwo/agent/trace_writer.py agiwo/agent/models/stream.py tests/agent/test_compact.py tests/agent/test_step_back.py tests/agent/test_termination.py tests/agent/test_run_contracts.py
git commit -m "refactor: record runtime policies in run log"
```

## Task 6: Migrate SQLite Serialization And Minimal Read Views

**Files:**
- Modify: `agiwo/agent/storage/sqlite.py`
- Modify: `agiwo/agent/storage/serialization.py`
- Modify: `console/server/services/metrics.py`
- Modify: `console/server/services/runtime/session_view_service.py`
- Modify: `console/server/response_serialization.py`
- Modify: `console/server/routers/sessions.py`
- Test: `tests/agent/test_storage_serialization.py`
- Test: `tests/agent/test_user_input_serialization.py`
- Test: `console/tests/test_session_summary.py`
- Test: `console/tests/test_sessions_api.py`

- [ ] **Step 1: Write the failing serialization/view tests**

```python
# tests/agent/test_storage_serialization.py
from agiwo.agent.models.log import RunStarted, UserStepCommitted


async def test_sqlite_round_trips_run_log_entries(storage) -> None:
    started = RunStarted(sequence=1, session_id="sess-1", run_id="run-1", agent_id="agent-1", user_input="hello")
    step = UserStepCommitted(sequence=2, session_id="sess-1", run_id="run-1", agent_id="agent-1", content="hello")

    await storage.append_entries([started, step])
    loaded = await storage.list_entries(session_id="sess-1")

    assert [item.kind for item in loaded] == [started.kind, step.kind]
```

```python
# console/tests/test_session_summary.py
async def test_session_summary_reads_latest_run_view(run_log_storage) -> None:
    summary = await service.get_session_detail("sess-1")
    assert summary.summary.last_response == "world"
```

- [ ] **Step 2: Run the serialization and console tests**

Run: `uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_user_input_serialization.py console/tests/test_session_summary.py console/tests/test_sessions_api.py -v`
Expected: FAIL because SQLite and console still rely on `Run` and `StepRecord`.

- [ ] **Step 3: Implement SQLite `RunLogStorage` and view queries**

```python
# agiwo/agent/storage/sqlite.py
class SQLiteRunLogStorage(RunLogStorage):
    async def append_entries(self, entries: list[RunLogEntry]) -> None:
        rows = [serialize_run_log_entry(entry) for entry in entries]
        await self._insert_many("run_log_entries", rows)

    async def get_latest_run_view(self, session_id: str) -> RunView | None:
        rows = await self._fetchall(
            "SELECT * FROM run_log_entries WHERE session_id = ? ORDER BY sequence DESC LIMIT 200",
            (session_id,),
        )
        return build_run_view_from_entries(deserialize_run_log_entries(rows))
```

```python
# agiwo/agent/storage/serialization.py
def serialize_run_log_entry(entry: RunLogEntry) -> dict[str, Any]:
    return {
        "sequence": entry.sequence,
        "session_id": entry.session_id,
        "run_id": entry.run_id,
        "agent_id": entry.agent_id,
        "kind": entry.kind.value,
        "payload": json.dumps(asdict(entry), default=str, ensure_ascii=False),
    }


def deserialize_run_log_entries(rows: list[dict[str, Any]]) -> list[RunLogEntry]:
    entries: list[RunLogEntry] = []
    for row in rows:
        payload = json.loads(row["payload"])
        entries.append(run_log_entry_from_payload(row["kind"], payload))
    return entries


def build_run_view_from_entries(entries: list[RunLogEntry]) -> RunView | None:
    if not entries:
        return None
    latest = entries[-1]
    termination = next(
        (entry for entry in reversed(entries) if entry.kind is RunLogEntryKind.TERMINATION_DECIDED),
        None,
    )
    last_assistant = next(
        (entry for entry in reversed(entries) if entry.kind is RunLogEntryKind.ASSISTANT_STEP_COMMITTED),
        None,
    )
    first_started = next(entry for entry in entries if entry.kind is RunLogEntryKind.RUN_STARTED)
    return RunView(
        run_id=latest.run_id,
        session_id=latest.session_id,
        agent_id=latest.agent_id,
        status="completed" if termination is not None else "running",
        response=getattr(last_assistant, "content", None),
        termination_reason=getattr(termination, "termination_reason", None),
        last_user_input=getattr(first_started, "user_input", None),
    )
```

- [ ] **Step 4: Point console read paths at `RunLogStorage` views and rerun tests**

```python
# console/server/services/runtime/session_view_service.py
latest_runs = await self._run_storage.batch_get_latest_run_views(session_ids)
step_counts = await self._run_storage.batch_get_committed_step_counts(session_ids)
```

```python
# console/server/response_serialization.py
def run_response_from_sdk(run: RunView) -> RunResponse:
    return RunResponse(
        id=run.run_id,
        agent_id=run.agent_id,
        session_id=run.session_id,
        user_input=run.last_user_input,
        status=run.status,
        response_content=run.response,
        metrics=RunMetricsResponse(
            duration_ms=run.metrics.duration_ms if run.metrics else None,
            input_tokens=run.metrics.input_tokens if run.metrics else None,
            output_tokens=run.metrics.output_tokens if run.metrics else None,
            total_tokens=run.metrics.total_tokens if run.metrics else None,
            cache_read_tokens=run.metrics.cache_read_tokens if run.metrics else None,
            cache_creation_tokens=run.metrics.cache_creation_tokens if run.metrics else None,
            token_cost=run.metrics.token_cost if run.metrics else None,
            steps_count=run.metrics.steps_count if run.metrics else 0,
            tool_calls_count=run.metrics.tool_calls_count if run.metrics else 0,
        ) if run.metrics else None,
    )


def step_response_from_sdk(step: StepView) -> StepResponse:
    return StepResponse(
        id=f"{step.run_id}:{step.sequence}",
        session_id=step.session_id,
        run_id=step.run_id,
        sequence=step.sequence,
        role=step.role.value,
        agent_id=step.agent_id,
        content=step.content,
        tool_calls=step.tool_calls,
        tool_call_id=step.tool_call_id,
        name=step.name,
        metrics=step_metrics_response_from_sdk(step.metrics) if step.metrics else None,
    )
```

Run: `uv run pytest tests/agent/test_storage_serialization.py tests/agent/test_user_input_serialization.py console/tests/test_session_summary.py console/tests/test_sessions_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agiwo/agent/storage/sqlite.py agiwo/agent/storage/serialization.py console/server/services/metrics.py console/server/services/runtime/session_view_service.py console/server/response_serialization.py console/server/routers/sessions.py tests/agent/test_storage_serialization.py tests/agent/test_user_input_serialization.py console/tests/test_session_summary.py console/tests/test_sessions_api.py
git commit -m "refactor: expose run log views to sqlite and console"
```

## Task 7: Remove Legacy Symbols, Update Remaining Tests, And Refresh Docs

**Files:**
- Modify: `agiwo/agent/agent.py`
- Modify: `agiwo/agent/models/__init__.py`
- Modify: `agiwo/agent/__init__.py`
- Modify: `AGENTS.md`
- Modify: remaining `tests/agent/*.py` imports/assertions that still mention `Run`, `StepRecord`, `AgentHooks`, or `RunStepStorage`

- [ ] **Step 1: Write the failing cleanup assertion**

```python
# tests/agent/test_definition_contracts.py
import inspect

from agiwo.agent import Agent


def test_agent_constructor_no_longer_exposes_agenthooks() -> None:
    assert "AgentHooks" not in str(inspect.signature(Agent.__init__))
```

- [ ] **Step 2: Run the cleanup-focused tests**

Run: `uv run pytest tests/agent/test_definition_contracts.py tests/agent/test_state_tracking.py tests/agent/test_run_contracts.py -v`
Expected: FAIL because public exports and tests still reference legacy symbols.

- [ ] **Step 3: Delete legacy exports and update docs/import sites**

```python
# agiwo/agent/__init__.py
__all__ = [
    "Agent",
    "AgentExecutionHandle",
    "AgentConfig",
    "AgentOptions",
    "AgentStorageOptions",
    "RunLogEntry",
    "RunLogEntryKind",
    "RunOutput",
    "RunMetrics",
    "TerminationReason",
    "HookRegistry",
    "HookRegistration",
    "HookPhase",
    "HookCapability",
]
```

```markdown
# AGENTS.md
- hook contract 收口在 `agiwo.agent.hooks`，以 phase-based hook registry 替代旧 `AgentHooks`
- agent storage 以 `RunLogStorage` 为 canonical runtime persistence，不再由 `Run` / `StepRecord` 作为 source of truth
```

- [ ] **Step 4: Run the full validation set for Phase 1**

Run: `uv run python scripts/lint.py ci`
Expected: PASS

Run: `uv run pytest tests/agent/ -v`
Expected: PASS

Run: `uv run python scripts/check.py console-tests`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add AGENTS.md agiwo/agent/__init__.py agiwo/agent/models/__init__.py agiwo/agent/agent.py tests/agent/ console/tests/
git commit -m "refactor: remove legacy agent runtime symbols"
```

## Self-Review

### Spec Coverage

The plan covers every Phase 1 requirement from the design spec:

1. `RunLog` as the only source of truth: Tasks 1, 3, 4, 5
2. deleting `Run`, `StepRecord`, and `AgentHooks`: Tasks 1, 2, 7
3. first-class `termination`, `compaction`, and `step-back`: Task 5
4. trace/stream rebuilt from `RunLog`: Task 5
5. public `Agent` execution entrypoints preserved: Task 4
6. minimal console/query adaptation without full scheduler migration: Task 6

There are no uncovered Phase 1 requirements.

### Placeholder Scan

Checked for:

1. common placeholder markers
2. deferred implementation language
3. undefined "add tests" style placeholders

None are present.

### Type Consistency

The plan uses these names consistently across tasks:

1. `RunLogEntry` / `RunLogEntryKind`
2. `RunLogStorage`
3. `InMemoryRunLogStorage`
4. `SQLiteRunLogStorage`
5. `RunLoopOrchestrator`
6. `HookRegistry`
7. `HookRegistration`
8. `RunStateWriter`
9. `TraceBuilder`
10. `StepView` and `RunView`

No later task refers back to the deleted `Run`, `StepRecord`, `AgentHooks`, or `RunStepStorage` as implementation targets.
