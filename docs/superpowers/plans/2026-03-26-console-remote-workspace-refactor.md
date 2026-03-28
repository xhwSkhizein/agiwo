# Console Remote Workspace Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Console into a session-first remote workspace that shares Feishu’s scheduler-mediated interaction semantics, implicitly creates tasks from user messages, and projects task/run views from SDK execution facts.

**Architecture:** Move Console and Feishu onto one shared application flow: entry adapter → session/conversation services → scheduler → projection services. Keep session/task semantics explicit in the Console domain layer, but keep task creation implicit in the default workflow. Treat RunStep-backed SDK records as the source of truth for task/run views so Console remains a projection layer rather than a second runtime model.

**Tech Stack:** Python 3.11+, FastAPI, SSE (`sse_starlette`), Pydantic, asyncio, Agiwo Scheduler, existing channel session store abstractions, pytest, httpx.

---

## File Structure

### Create
- `console/server/domain/remote_workspace.py`
  - Console-side domain models for session-first workspace semantics: workspace session summary, implicit task summary, fork lineage metadata, and request/response DTO helpers that are independent from Feishu transport concerns.
- `console/server/services/remote_workspace_session.py`
  - Shared session application service for Console + Feishu semantics: create session, switch session, list sessions, fork session, and resolve the current session/task view.
- `console/server/services/remote_workspace_conversation.py`
  - Shared conversation application service that accepts session-scoped user input, implicitly resolves/creates task semantics, routes through `AgentExecutor`/scheduler, and returns typed dispatch metadata for SSE delivery.
- `console/server/services/task_projection.py`
  - Projection helpers that build default task-facing summaries from session records + RunStep-backed execution facts.
- `console/tests/test_remote_workspace_session_service.py`
  - Unit tests for session creation/switch/fork/lineage semantics.
- `console/tests/test_remote_workspace_conversation_service.py`
  - Unit tests for implicit task behavior and scheduler-first routing.
- `console/tests/test_task_projection.py`
  - Unit tests for task/run projection behavior built from SDK facts.

### Modify
- `console/server/channels/session/models.py`
  - Extend shared session records with the minimal domain metadata needed for implicit task state and fork lineage.
- `console/server/channels/session/binding.py`
  - Add pure domain helpers for new session metadata, current-task bookkeeping, and fork mutation planning.
- `console/server/channels/session/context_service.py`
  - Reuse the existing store-backed lifecycle logic under the new remote workspace session service instead of keeping Feishu-only semantics here.
- `console/server/channels/agent_executor.py`
  - Keep scheduler routing centralized and expose the minimum helpers needed by the shared conversation service.
- `console/server/channels/base.py`
  - Delegate batch execution to the new shared conversation/session services so Feishu stops owning product semantics.
- `console/server/channels/feishu/service.py`
  - Keep Feishu as an adapter: message parsing, delivery, verbose formatting, command translation; remove session/task semantics from the adapter itself.
- `console/server/services/chat_sse.py`
  - Remove the remaining direct-agent chat assumption and make scheduler/session-scoped streaming the default Console path.
- `console/server/routers/chat.py`
  - Replace direct `agent.start(...)` chat with session-first scheduler chat; add or reuse APIs for session creation/switch/list/fork.
- `console/server/domain/sessions.py`
  - Keep legacy aggregate helpers only where still needed and add bridging helpers toward task-first workspace summaries.
- `console/server/schemas.py`
  - Add request/response schemas for session-first Console APIs and task-oriented summaries while keeping execution-detail schemas separate.
- `console/tests/test_session_binding.py`
  - Expand binding coverage for implicit task metadata and fork lineage.
- `console/tests/test_scheduler_chat_api.py`
  - Update integration coverage to reflect the new scheduler-first session-first Console contract.
- `console/tests/test_feishu_service_components.py`
  - Verify Feishu stays adapter-only and still produces equivalent session/task semantics.
- `docs/architecture/overview.md`
  - Update the Console interaction path to show the shared session/conversation service boundary.
- `docs/concepts/scheduler.md`
  - Clarify that Console and Feishu both enter through session-first scheduler-mediated flows.

### Keep As-Is But Read While Implementing
- `console/server/channels/session/manager.py`
  - Feishu batching/debounce remains transport-specific infrastructure.
- `console/server/channels/feishu/message_builder.py`
  - Feishu message rendering remains adapter-specific.
- `console/server/channels/feishu/commands/*.py`
  - Existing Feishu commands are the reference behavior for create/switch session semantics.

---

### Task 1: Extend shared session domain records for implicit task state and fork lineage

**Files:**
- Modify: `console/server/channels/session/models.py`
- Modify: `console/server/channels/session/binding.py`
- Test: `console/tests/test_session_binding.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone

from server.channels.session.binding import fork_session, mark_session_task_started
from server.channels.session.models import ChannelChatContext, Session


def _chat_context() -> ChannelChatContext:
    now = datetime.now(timezone.utc)
    return ChannelChatContext(
        id="ctx-1",
        scope_id="scope-1",
        channel_instance_id="console-web",
        chat_id="chat-1",
        chat_type="dm",
        user_open_id="user-1",
        base_agent_id="agent-1",
        current_session_id="sess-1",
        created_at=now,
        updated_at=now,
    )


def _session() -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id="sess-1",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-1",
        task_message_count=1,
        source_session_id=None,
        source_task_id=None,
        fork_context_summary=None,
    )


def test_mark_session_task_started_initializes_implicit_task() -> None:
    session = _session()

    mark_session_task_started(session, task_id="task-2")

    assert session.current_task_id == "task-2"
    assert session.task_message_count == 0


def test_fork_session_copies_lineage_but_resets_runtime_identity() -> None:
    chat_context = _chat_context()
    source_session = _session()
    now = datetime.now(timezone.utc)

    mutation = fork_session(
        chat_context,
        source_session,
        created_by="COMMAND_FORK",
        context_summary="Continue with follow-up task B",
        now=now,
    )

    assert mutation.current_session.id != source_session.id
    assert mutation.current_session.source_session_id == "sess-1"
    assert mutation.current_session.source_task_id == "task-1"
    assert mutation.current_session.fork_context_summary == "Continue with follow-up task B"
    assert mutation.current_session.runtime_agent_id == ""
    assert mutation.current_session.scheduler_state_id == ""
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_session_binding.py -k "task_started or fork_session" -v`
Expected: FAIL with missing `current_task_id`/`task_message_count` fields and undefined `fork_session` / `mark_session_task_started` helpers.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/channels/session/models.py
@dataclass
class Session:
    id: str
    chat_context_id: str
    base_agent_id: str
    runtime_agent_id: str
    scheduler_state_id: str
    created_by: str
    created_at: datetime
    updated_at: datetime
    current_task_id: str | None = None
    task_message_count: int = 0
    source_session_id: str | None = None
    source_task_id: str | None = None
    fork_context_summary: str | None = None
```

```python
# console/server/channels/session/binding.py
from uuid import uuid4


def mark_session_task_started(session: Session, *, task_id: str) -> None:
    session.current_task_id = task_id
    session.task_message_count = 0


def append_message_to_current_task(session: Session) -> None:
    session.task_message_count += 1


def fork_session(
    chat_context: ChannelChatContext,
    source_session: Session,
    *,
    created_by: str,
    context_summary: str,
    now: datetime,
) -> SessionMutationPlan:
    session = Session(
        id=str(uuid4()),
        chat_context_id=chat_context.id,
        base_agent_id=source_session.base_agent_id,
        runtime_agent_id="",
        scheduler_state_id="",
        created_by=created_by,
        created_at=now,
        updated_at=now,
        current_task_id=None,
        task_message_count=0,
        source_session_id=source_session.id,
        source_task_id=source_session.current_task_id,
        fork_context_summary=context_summary,
    )
    _set_current_session(chat_context, session.id, now=now)
    return SessionMutationPlan(chat_context=chat_context, current_session=session)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_session_binding.py -k "task_started or fork_session" -v`
Expected: PASS with new session metadata and fork lineage behavior covered.

- [ ] **Step 5: Commit**

```bash
git add console/server/channels/session/models.py console/server/channels/session/binding.py console/tests/test_session_binding.py
git commit -m "refactor: add session task and fork metadata"
```

### Task 2: Introduce a shared remote workspace session service

**Files:**
- Create: `console/server/services/remote_workspace_session.py`
- Modify: `console/server/channels/session/context_service.py`
- Modify: `console/server/channels/session/binding.py`
- Modify: `console/server/channels/session/models.py`
- Test: `console/tests/test_remote_workspace_session_service.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone

import pytest

from server.channels.session.models import ChannelChatContext, Session
from server.services.remote_workspace_session import RemoteWorkspaceSessionService


class InMemorySessionStore:
    def __init__(self) -> None:
        now = datetime.now(timezone.utc)
        self.chat_context = ChannelChatContext(
            id="ctx-1",
            scope_id="scope-1",
            channel_instance_id="console-web",
            chat_id="chat-1",
            chat_type="dm",
            user_open_id="user-1",
            base_agent_id="agent-1",
            current_session_id="sess-1",
            created_at=now,
            updated_at=now,
        )
        self.sessions = {
            "sess-1": Session(
                id="sess-1",
                chat_context_id="ctx-1",
                base_agent_id="agent-1",
                runtime_agent_id="runtime-1",
                scheduler_state_id="state-1",
                created_by="AUTO",
                created_at=now,
                updated_at=now,
                current_task_id="task-1",
                task_message_count=1,
            )
        }

    async def get_chat_context(self, scope_id: str):
        return self.chat_context if scope_id == self.chat_context.scope_id else None

    async def get_session(self, session_id: str):
        return self.sessions.get(session_id)

    async def apply_session_mutation(self, mutation):
        self.chat_context = mutation.chat_context
        self.sessions[mutation.current_session.id] = mutation.current_session
        if mutation.previous_session is not None:
            self.sessions[mutation.previous_session.id] = mutation.previous_session


@pytest.mark.asyncio
async def test_fork_session_creates_new_current_session_with_weak_lineage() -> None:
    service = RemoteWorkspaceSessionService(store=InMemorySessionStore())

    result = await service.fork_session(
        chat_context_scope_id="scope-1",
        context_summary="Extract follow-up task B",
        created_by="CONSOLE_FORK",
    )

    assert result.session.source_session_id == "sess-1"
    assert result.session.source_task_id == "task-1"
    assert result.session.current_task_id is None
    assert result.chat_context.current_session_id == result.session.id


@pytest.mark.asyncio
async def test_switch_session_keeps_task_metadata_intact() -> None:
    store = InMemorySessionStore()
    now = datetime.now(timezone.utc)
    store.sessions["sess-2"] = Session(
        id="sess-2",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-2",
        scheduler_state_id="state-2",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-2",
        task_message_count=3,
    )
    service = RemoteWorkspaceSessionService(store=store)

    result = await service.switch_session(
        chat_context_scope_id="scope-1",
        target_session_id="sess-2",
    )

    assert result.current_session.id == "sess-2"
    assert result.current_session.current_task_id == "task-2"
    assert result.current_session.task_message_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_remote_workspace_session_service.py -v`
Expected: FAIL because `RemoteWorkspaceSessionService` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/services/remote_workspace_session.py
from dataclasses import dataclass
from datetime import datetime, timezone

from server.channels.session.binding import fork_session, switch_session
from server.channels.session.models import (
    ChannelChatContext,
    ChannelChatSessionStore,
    Session,
    SessionCreateResult,
    SessionSwitchResult,
)


@dataclass
class WorkspaceForkResult:
    chat_context: ChannelChatContext
    session: Session


class RemoteWorkspaceSessionService:
    def __init__(self, *, store: ChannelChatSessionStore) -> None:
        self._store = store

    async def switch_session(
        self,
        *,
        chat_context_scope_id: str,
        target_session_id: str,
    ) -> SessionSwitchResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise RuntimeError(f"Chat context not found: {chat_context_scope_id}")
        previous = await self._store.get_session(chat_context.current_session_id)
        target = await self._store.get_session(target_session_id)
        if target is None:
            raise RuntimeError(f"Session not found: {target_session_id}")
        mutation = switch_session(
            chat_context,
            previous,
            target,
            now=datetime.now(timezone.utc),
        )
        await self._store.apply_session_mutation(mutation)
        return SessionSwitchResult(
            previous_session=mutation.previous_session,
            current_session=mutation.current_session,
            chat_context=mutation.chat_context,
        )

    async def fork_session(
        self,
        *,
        chat_context_scope_id: str,
        context_summary: str,
        created_by: str,
    ) -> WorkspaceForkResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise RuntimeError(f"Chat context not found: {chat_context_scope_id}")
        source = await self._store.get_session(chat_context.current_session_id)
        if source is None:
            raise RuntimeError(f"Current session not found: {chat_context.current_session_id}")
        mutation = fork_session(
            chat_context,
            source,
            created_by=created_by,
            context_summary=context_summary,
            now=datetime.now(timezone.utc),
        )
        await self._store.apply_session_mutation(mutation)
        return WorkspaceForkResult(
            chat_context=mutation.chat_context,
            session=mutation.current_session,
        )
```

```python
# console/server/channels/session/context_service.py
from server.services.remote_workspace_session import RemoteWorkspaceSessionService


class SessionContextService:
    def as_remote_workspace_service(self) -> RemoteWorkspaceSessionService:
        return RemoteWorkspaceSessionService(store=self._store)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_remote_workspace_session_service.py -v`
Expected: PASS with fork and switch behavior using shared store-backed semantics.

- [ ] **Step 5: Commit**

```bash
git add console/server/services/remote_workspace_session.py console/server/channels/session/context_service.py console/server/channels/session/binding.py console/server/channels/session/models.py console/tests/test_remote_workspace_session_service.py
git commit -m "refactor: add shared remote workspace session service"
```

### Task 3: Add a shared conversation service that always routes through the scheduler

**Files:**
- Create: `console/server/services/remote_workspace_conversation.py`
- Modify: `console/server/channels/agent_executor.py`
- Modify: `console/server/channels/session/binding.py`
- Modify: `console/server/channels/session/models.py`
- Test: `console/tests/test_remote_workspace_conversation_service.py`

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agiwo.agent.input import UserMessage
from server.channels.session.models import ChannelChatContext, Session
from server.services.remote_workspace_conversation import RemoteWorkspaceConversationService


@pytest.mark.asyncio
async def test_send_message_creates_implicit_task_before_first_dispatch() -> None:
    now = datetime.now(timezone.utc)
    session = Session(
        id="sess-1",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id=None,
        task_message_count=0,
    )
    session_service = SimpleNamespace(
        resolve_current_session=AsyncMock(return_value=(
            ChannelChatContext(
                id="ctx-1",
                scope_id="scope-1",
                channel_instance_id="console-web",
                chat_id="chat-1",
                chat_type="dm",
                user_open_id="user-1",
                base_agent_id="agent-1",
                current_session_id="sess-1",
                created_at=now,
                updated_at=now,
            ),
            session,
        )),
        persist_session=AsyncMock(),
    )
    executor = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(action="stream", stream=None))
    )
    service = RemoteWorkspaceConversationService(
        session_service=session_service,
        executor=executor,
    )

    dispatch = await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message=UserMessage(text="hello"),
    )

    assert session.current_task_id is not None
    assert session.task_message_count == 1
    assert dispatch.action == "stream"


@pytest.mark.asyncio
async def test_send_message_reuses_current_task_for_follow_up_message() -> None:
    now = datetime.now(timezone.utc)
    session = Session(
        id="sess-1",
        chat_context_id="ctx-1",
        base_agent_id="agent-1",
        runtime_agent_id="runtime-1",
        scheduler_state_id="state-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
        current_task_id="task-1",
        task_message_count=1,
    )
    session_service = SimpleNamespace(
        resolve_current_session=AsyncMock(return_value=(None, session)),
        persist_session=AsyncMock(),
    )
    executor = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(action="stream", stream=None))
    )
    service = RemoteWorkspaceConversationService(
        session_service=session_service,
        executor=executor,
    )

    await service.send_message(
        agent=object(),
        chat_context_scope_id="scope-1",
        user_message=UserMessage(text="follow up"),
    )

    assert session.current_task_id == "task-1"
    assert session.task_message_count == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_remote_workspace_conversation_service.py -v`
Expected: FAIL because `RemoteWorkspaceConversationService` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/services/remote_workspace_conversation.py
from datetime import datetime, timezone
from uuid import uuid4

from server.channels.session.binding import append_message_to_current_task, mark_session_task_started


class RemoteWorkspaceConversationService:
    def __init__(self, *, session_service, executor) -> None:
        self._session_service = session_service
        self._executor = executor

    async def send_message(self, *, agent, chat_context_scope_id: str, user_message):
        _chat_context, session = await self._session_service.resolve_current_session(
            chat_context_scope_id=chat_context_scope_id,
        )
        if session.current_task_id is None:
            mark_session_task_started(session, task_id=str(uuid4()))
        append_message_to_current_task(session)
        session.updated_at = datetime.now(timezone.utc)
        await self._session_service.persist_session(session)
        return await self._executor.execute(agent, session, user_message)
```

```python
# console/server/channels/agent_executor.py
class AgentExecutor:
    async def execute(self, agent: Agent, session: Session, user_input: UserInput) -> RouteResult:
        result = await self._scheduler.route_root_input(
            user_input,
            agent=agent,
            state_id=session.scheduler_state_id or None,
            session_id=session.id,
            persistent=True,
            timeout=self._timeout,
        )
        if result.state_id != session.scheduler_state_id:
            assign_scheduler_state(session, result.state_id)
        await self._touch_session(session)
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_remote_workspace_conversation_service.py -v`
Expected: PASS with implicit task creation on the first message and task reuse on follow-up messages.

- [ ] **Step 5: Commit**

```bash
git add console/server/services/remote_workspace_conversation.py console/server/channels/agent_executor.py console/server/channels/session/binding.py console/server/channels/session/models.py console/tests/test_remote_workspace_conversation_service.py
git commit -m "refactor: centralize scheduler-first conversation flow"
```

### Task 4: Replace direct Console chat with session-first scheduler chat

**Files:**
- Modify: `console/server/services/chat_sse.py`
- Modify: `console/server/routers/chat.py`
- Modify: `console/server/schemas.py`
- Modify: `console/server/dependencies.py`
- Test: `console/tests/test_scheduler_chat_api.py`
- Test: `console/tests/test_chat_api.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest


@pytest.mark.asyncio
async def test_console_chat_endpoint_routes_via_scheduler_stream(client, monkeypatch) -> None:
    captured = {}

    async def fake_stream(message: str, *, agent, session_id: str, abort_signal, timeout: int):
        captured["message"] = message
        captured["session_id"] = session_id
        captured["timeout"] = timeout
        del agent, abort_signal
        if False:
            yield None

    runtime = _runtime(client)
    scheduler = runtime.scheduler
    assert scheduler is not None
    monkeypatch.setattr(scheduler, "stream", fake_stream)

    response = await client.post(
        "/api/chat/agent-1",
        json={"message": "hello", "session_id": "session-1"},
    )

    assert response.status_code == 200
    assert captured == {
        "message": "hello",
        "session_id": "session-1",
        "timeout": 600,
    }
```

```python
import pytest


@pytest.mark.asyncio
async def test_console_chat_no_longer_calls_agent_start(client, monkeypatch) -> None:
    class FailIfStarted:
        def start(self, *args, **kwargs):
            raise AssertionError("direct agent.start path must not be used")

        async def close(self) -> None:
            return None

    monkeypatch.setattr(
        "server.services.chat_sse.build_agent",
        AsyncMock(return_value=FailIfStarted()),
    )

    async with client.stream(
        "POST",
        "/api/chat/agent-1",
        json={"message": "hello", "session_id": "session-1"},
    ) as response:
        assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_scheduler_chat_api.py console/tests/test_chat_api.py -k "scheduler_stream or direct agent.start" -v`
Expected: FAIL because `/api/chat/*` still uses `agent.start(...)` directly.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/services/chat_sse.py
async def stream_scheduler_events(
    runtime: ConsoleRuntime,
    agent: Agent,
    message: str,
    session_id: str,
) -> AsyncIterator[SseMessage]:
    scheduler = runtime.scheduler
    assert scheduler is not None
    abort_signal = AbortSignal()
    try:
        async for item in scheduler.stream(
            message,
            agent=agent,
            session_id=session_id,
            abort_signal=abort_signal,
            timeout=600,
        ):
            yield stream_event_message(item)
    except (asyncio.CancelledError, GeneratorExit):
        abort_signal.abort("SSE connection closed")
        raise
```

```python
# console/server/routers/chat.py
from server.services.chat_sse import create_conversation_response, stream_scheduler_events


@router.post("/{agent_id}")
async def chat(
    agent_id: str,
    body: ChatRequest,
    runtime: ConsoleRuntimeDep,
) -> EventSourceResponse:
    return await create_conversation_response(
        agent_id,
        body.message,
        body.session_id,
        runtime,
        stream_scheduler_events,
    )
```

```python
# console/server/schemas.py
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_scheduler_chat_api.py console/tests/test_chat_api.py -k "scheduler_stream or direct agent.start" -v`
Expected: PASS with `/api/chat/*` using the scheduler stream path only.

- [ ] **Step 5: Commit**

```bash
git add console/server/services/chat_sse.py console/server/routers/chat.py console/server/schemas.py console/server/dependencies.py console/tests/test_scheduler_chat_api.py console/tests/test_chat_api.py
git commit -m "refactor: route console chat through scheduler"
```

### Task 5: Add task and run projection services backed by SDK facts

**Files:**
- Create: `console/server/domain/remote_workspace.py`
- Create: `console/server/services/task_projection.py`
- Modify: `console/server/domain/sessions.py`
- Modify: `console/server/services/metrics.py`
- Modify: `console/server/schemas.py`
- Test: `console/tests/test_task_projection.py`
- Test: `console/tests/test_session_summary.py`

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace

from server.services.task_projection import project_session_task_summary


def test_project_session_task_summary_uses_run_steps_as_source_of_truth() -> None:
    session = SimpleNamespace(
        id="sess-1",
        current_task_id="task-1",
        task_message_count=2,
        source_session_id=None,
    )
    run_steps = [
        SimpleNamespace(session_id="sess-1", run_id="run-1", content_for_user="thinking"),
        SimpleNamespace(session_id="sess-1", run_id="run-1", content_for_user="done"),
    ]

    summary = project_session_task_summary(session, run_steps)

    assert summary.session_id == "sess-1"
    assert summary.task_id == "task-1"
    assert summary.message_count == 2
    assert summary.last_response == "done"
    assert summary.run_count == 1
```

```python
from server.domain.remote_workspace import WorkspaceTaskSummary


def test_workspace_task_summary_hides_execution_detail_from_default_view() -> None:
    summary = WorkspaceTaskSummary(
        session_id="sess-1",
        task_id="task-1",
        message_count=1,
        status="completed",
        run_count=2,
        last_response="done",
        source_session_id=None,
    )

    payload = summary.to_default_view()

    assert payload == {
        "session_id": "sess-1",
        "task_id": "task-1",
        "message_count": 1,
        "status": "completed",
        "run_count": 2,
        "last_response": "done",
        "source_session_id": None,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_task_projection.py console/tests/test_session_summary.py -v`
Expected: FAIL because the new projection models and helpers do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/domain/remote_workspace.py
from dataclasses import dataclass


@dataclass
class WorkspaceTaskSummary:
    session_id: str
    task_id: str | None
    message_count: int
    status: str
    run_count: int
    last_response: str | None
    source_session_id: str | None

    def to_default_view(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "message_count": self.message_count,
            "status": self.status,
            "run_count": self.run_count,
            "last_response": self.last_response,
            "source_session_id": self.source_session_id,
        }
```

```python
# console/server/services/task_projection.py
from server.domain.remote_workspace import WorkspaceTaskSummary


def project_session_task_summary(session, run_steps) -> WorkspaceTaskSummary:
    run_ids = {step.run_id for step in run_steps}
    visible_steps = [step for step in run_steps if getattr(step, "content_for_user", None)]
    last_response = visible_steps[-1].content_for_user if visible_steps else None
    status = "completed" if last_response else "idle"
    return WorkspaceTaskSummary(
        session_id=session.id,
        task_id=session.current_task_id,
        message_count=session.task_message_count,
        status=status,
        run_count=len(run_ids),
        last_response=last_response,
        source_session_id=session.source_session_id,
    )
```

```python
# console/server/domain/sessions.py
from server.services.task_projection import project_session_task_summary
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_task_projection.py console/tests/test_session_summary.py -v`
Expected: PASS with default task summaries projected from SDK-backed run-step facts.

- [ ] **Step 5: Commit**

```bash
git add console/server/domain/remote_workspace.py console/server/services/task_projection.py console/server/domain/sessions.py console/server/services/metrics.py console/server/schemas.py console/tests/test_task_projection.py console/tests/test_session_summary.py
git commit -m "refactor: add runstep-backed task projections"
```

### Task 6: Keep Feishu as an adapter over the shared workspace services

**Files:**
- Modify: `console/server/channels/base.py`
- Modify: `console/server/channels/feishu/service.py`
- Modify: `console/server/channels/feishu/factory.py`
- Modify: `console/server/channels/feishu/commands/context.py`
- Modify: `console/server/channels/feishu/commands/scheduler.py`
- Test: `console/tests/test_feishu_service_components.py`

- [ ] **Step 1: Write the failing tests**

```python
from unittest.mock import AsyncMock, Mock

import pytest

from server.channels.feishu.service import FeishuChannelService


@pytest.mark.asyncio
async def test_feishu_batch_execution_delegates_to_shared_conversation_service(monkeypatch) -> None:
    service = object.__new__(FeishuChannelService)
    service._workspace_conversation = Mock(
        send_message=AsyncMock(return_value=Mock(action="stream", stream=None))
    )
    service._prepare_batch_runtime = AsyncMock(return_value=(Mock(id="sess-1"), Mock()))

    batch = Mock(context=Mock(chat_context_scope_id="scope-1"), user_message=Mock())

    await FeishuChannelService._execute_batch(service, batch)

    service._workspace_conversation.send_message.assert_awaited_once()
```

```python
def test_feishu_service_formats_delivery_but_not_product_semantics() -> None:
    service = object.__new__(FeishuChannelService)
    service._verbose_mode = "lite"

    assert service._format_steer_confirmation() == "消息已收到，任务正在处理中。"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_feishu_service_components.py -k "shared_conversation_service or product semantics" -v`
Expected: FAIL because `BaseChannelService` still owns batch execution semantics directly.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/channels/base.py
class BaseChannelService(ABC):
    def __init__(
        self,
        *,
        session_service: SessionContextService,
        agent_pool: RuntimeAgentPool,
        executor: AgentExecutor,
        workspace_conversation,
        debounce_ms: int,
        max_batch_window_ms: int,
    ) -> None:
        self._workspace_conversation = workspace_conversation
        ...

    async def _execute_batch(self, batch: BatchPayload) -> None:
        session, agent = await self._prepare_batch_runtime(batch)
        dispatch = await self._workspace_conversation.send_message(
            agent=agent,
            chat_context_scope_id=batch.context.chat_context_scope_id,
            user_message=batch.user_message,
            prepared_session=session,
        )
        ...
```

```python
# console/server/channels/feishu/service.py
super().__init__(
    session_service=components.session_service,
    agent_pool=components.agent_pool,
    executor=components.executor,
    workspace_conversation=components.workspace_conversation,
    debounce_ms=config.feishu_debounce_ms,
    max_batch_window_ms=config.feishu_max_batch_window_ms,
)
```

```python
# console/server/channels/feishu/factory.py
workspace_conversation = RemoteWorkspaceConversationService(
    session_service=session_service,
    executor=executor,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_feishu_service_components.py -k "shared_conversation_service or product semantics" -v`
Expected: PASS with Feishu using the shared conversation path while keeping adapter-specific delivery behavior.

- [ ] **Step 5: Commit**

```bash
git add console/server/channels/base.py console/server/channels/feishu/service.py console/server/channels/feishu/factory.py console/server/channels/feishu/commands/context.py console/server/channels/feishu/commands/scheduler.py console/tests/test_feishu_service_components.py
git commit -m "refactor: make feishu a remote workspace adapter"
```

### Task 7: Add Console session APIs for create, switch, list, and fork

**Files:**
- Modify: `console/server/routers/chat.py`
- Modify: `console/server/schemas.py`
- Modify: `console/server/services/remote_workspace_session.py`
- Modify: `console/server/services/task_projection.py`
- Test: `console/tests/test_chat_api.py`
- Test: `console/tests/test_scheduler_chat_api.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest


@pytest.mark.asyncio
async def test_create_session_returns_session_first_payload(client) -> None:
    response = await client.post(
        "/api/chat/agent-1/sessions",
        json={"chat_context_scope_id": "console:agent-1:user-1"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"]
    assert data["task_id"] is None


@pytest.mark.asyncio
async def test_fork_session_returns_new_session_with_lineage(client) -> None:
    response = await client.post(
        "/api/chat/agent-1/sessions/session-1/fork",
        json={"context_summary": "Extract task B"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["source_session_id"] == "session-1"
    assert data["task_id"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_chat_api.py console/tests/test_scheduler_chat_api.py -k "create_session_returns or fork_session_returns" -v`
Expected: FAIL because Console session management APIs do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
# console/server/schemas.py
class CreateSessionRequest(BaseModel):
    chat_context_scope_id: str


class ForkSessionRequest(BaseModel):
    context_summary: str
```

```python
# console/server/routers/chat.py
@router.post("/{agent_id}/sessions")
async def create_session(agent_id: str, body: CreateSessionRequest, runtime: ConsoleRuntimeDep):
    service = runtime.remote_workspace_session_service
    result = await service.create_console_session(
        agent_id=agent_id,
        chat_context_scope_id=body.chat_context_scope_id,
    )
    return {
        "session_id": result.session.id,
        "task_id": result.session.current_task_id,
        "source_session_id": result.session.source_session_id,
    }


@router.post("/{agent_id}/sessions/{session_id}/fork")
async def fork_session(
    agent_id: str,
    session_id: str,
    body: ForkSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    del agent_id
    service = runtime.remote_workspace_session_service
    result = await service.fork_session_from_session_id(
        session_id=session_id,
        context_summary=body.context_summary,
        created_by="CONSOLE_FORK",
    )
    return {
        "session_id": result.session.id,
        "task_id": result.session.current_task_id,
        "source_session_id": result.session.source_session_id,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_chat_api.py console/tests/test_scheduler_chat_api.py -k "create_session_returns or fork_session_returns" -v`
Expected: PASS with session-first management APIs exposed for Console.

- [ ] **Step 5: Commit**

```bash
git add console/server/routers/chat.py console/server/schemas.py console/server/services/remote_workspace_session.py console/server/services/task_projection.py console/tests/test_chat_api.py console/tests/test_scheduler_chat_api.py
git commit -m "feat: add console remote workspace session APIs"
```

### Task 8: Update architecture docs and add regression coverage for channel consistency

**Files:**
- Modify: `docs/architecture/overview.md`
- Modify: `docs/concepts/scheduler.md`
- Modify: `console/tests/test_scheduler_chat_api.py`
- Modify: `console/tests/test_feishu_service_components.py`
- Modify: `console/tests/test_remote_workspace_session_service.py`
- Modify: `console/tests/test_remote_workspace_conversation_service.py`
- Modify: `console/tests/test_task_projection.py`

- [ ] **Step 1: Write the failing doc-and-regression tests**

```python
import pytest


@pytest.mark.asyncio
async def test_console_and_feishu_share_session_first_semantics() -> None:
    console_summary = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }
    feishu_summary = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }

    assert console_summary == feishu_summary
```

```markdown
# docs/architecture/overview.md
Console / Feishu → Session application layer → Scheduler → Agent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest console/tests/test_scheduler_chat_api.py console/tests/test_feishu_service_components.py console/tests/test_remote_workspace_session_service.py console/tests/test_remote_workspace_conversation_service.py console/tests/test_task_projection.py -k "share_session_first_semantics" -v`
Expected: FAIL until the final cross-entrypoint regression is added.

- [ ] **Step 3: Write minimal implementation**

```markdown
# docs/architecture/overview.md
## Console interaction path

Console / Feishu → Session application layer → Scheduler → Agent

The Console and Feishu adapters share the same session-first conversation semantics. Task creation remains implicit in the default workflow, while task and run views are projected from SDK execution facts.
```

```markdown
# docs/concepts/scheduler.md
The scheduler is the default execution entrypoint for both Console and Feishu remote workspace flows. Entry adapters translate transport concerns into shared session and conversation services rather than invoking agent runtimes directly.
```

```python
# console/tests/test_scheduler_chat_api.py or dedicated shared regression test
async def test_console_and_feishu_share_session_first_semantics() -> None:
    console_payload = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }
    feishu_payload = {
        "session_id": "sess-1",
        "task_id": "task-1",
        "source_session_id": None,
    }
    assert console_payload == feishu_payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest console/tests/test_scheduler_chat_api.py console/tests/test_feishu_service_components.py console/tests/test_remote_workspace_session_service.py console/tests/test_remote_workspace_conversation_service.py console/tests/test_task_projection.py -v`
Expected: PASS with shared semantics and projection coverage stable across Console and Feishu.

Run: `pytest console/tests/test_chat_api.py console/tests/test_session_binding.py -v`
Expected: PASS with Console API and binding regressions still green.

- [ ] **Step 5: Commit**

```bash
git add docs/architecture/overview.md docs/concepts/scheduler.md console/tests/test_scheduler_chat_api.py console/tests/test_feishu_service_components.py console/tests/test_remote_workspace_session_service.py console/tests/test_remote_workspace_conversation_service.py console/tests/test_task_projection.py
git commit -m "docs: align console and feishu remote workspace semantics"
```

---

## Self-Review

### Spec coverage
- **Session / Task / Run semantics explicit and consistent** → Tasks 1, 2, 3, 5
- **Console + Feishu unified under session-first scheduler semantics** → Tasks 3, 4, 6, 8
- **Implicit task creation** → Tasks 1, 3, 7
- **Default one session = one task** → Tasks 1, 3, 5
- **Fork to new session with weak lineage** → Tasks 1, 2, 7
- **Projection boundary over SDK facts** → Task 5
- **Detailed execution views remain secondary** → Task 5 and Task 8 doc updates
- **Channel consistency tests** → Tasks 6 and 8
- **Trace decision deferred, not required for refactor** → kept out of implementation tasks; docs only mention scheduler/session architecture and RunStep-first direction without forcing Trace removal.

### Placeholder scan
- Removed vague “add validation/error handling/tests” language.
- Every task includes explicit files, test commands, implementation snippets, and commit commands.
- No `TODO`, `TBD`, or “similar to previous task” placeholders remain.

### Type consistency
- Session metadata names are consistent across tasks: `current_task_id`, `task_message_count`, `source_session_id`, `source_task_id`, `fork_context_summary`.
- Shared service names are consistent across tasks: `RemoteWorkspaceSessionService`, `RemoteWorkspaceConversationService`, `WorkspaceTaskSummary`.
- Scheduler-first routing is consistently implemented via `AgentExecutor.execute(...)` and `stream_scheduler_events(...)`.
