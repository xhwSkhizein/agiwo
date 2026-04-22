# Agent Runtime Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the console-side migration onto `RunLog`-backed views so run/session/step queries and canonical-agent materialization no longer depend on pre-refactor assumptions or ambiguous runtime naming.

**Architecture:** Phase 3 keeps `RunLogStorage` as the single persisted source of truth and adds a console-facing query facade that assembles run, step, and session read models from `RunView` / `StepView` / session stats. In parallel, Console runtime construction is renamed from generic `build_*` wording to explicit canonical-agent materialization so router/cache/resume paths all communicate the same responsibility.

**Tech Stack:** Python 3.10+, FastAPI, `RunLogStorage`, `RunView`, `StepView`, Console runtime services, pytest, ruff

---

## Scope Check

This plan covers only **Phase 3: Console Migration** from the runtime refactor spec:

1. migrate console queries and SSE serialization to `RunLog`-backed views
2. remove direct assumptions about persisted `Run` / `StepRecord` models
3. align session summary and timeline views with a console query facade
4. clarify console-side canonical-agent materialization naming

It does **not** reopen scheduler state-machine work or change external API semantics for `route_root_input()`.

## File Structure

### Create

- `console/server/services/runtime/run_query_service.py`
- `console/tests/test_run_query_service.py`

### Modify

- `console/server/dependencies.py`
- `console/server/routers/sessions.py`
- `console/server/routers/scheduler.py`
- `console/server/services/runtime/agent_factory.py`
- `console/server/services/runtime/agent_runtime_cache.py`
- `console/server/services/runtime/session_view_service.py`
- `console/server/services/runtime/__init__.py`
- `console/tests/test_agent_runtime_components.py`
- `console/tests/test_config_env.py`
- `console/tests/test_sessions_api.py`
- `docs/architecture/scheduler-console-runtime-refactor.md`

### Responsibilities

- `console/server/services/runtime/run_query_service.py`
  - Own `RunLog`-backed read operations for console: runs, session steps, and session stats.
- `console/server/services/runtime/session_view_service.py`
  - Stop reading storage ad hoc; assemble summaries through the query facade.
- `console/server/routers/sessions.py`
  - Stop reaching into `runtime.run_log_storage` directly; use the query facade.
- `console/server/services/runtime/agent_factory.py`
  - Rename ambiguous `build_agent(...)` wording to explicit canonical-agent materialization.
- `console/server/services/runtime/agent_runtime_cache.py`
  - Consume the renamed materialization function and keep runtime-agent refresh semantics unchanged.
- `console/server/routers/scheduler.py`
  - Use the same canonical-agent materialization entrypoint for persistent-root creation.

## Task 1: Add A Console Run Query Facade

**Files:**
- Create: `console/server/services/runtime/run_query_service.py`
- Modify: `console/server/dependencies.py`
- Modify: `console/server/routers/sessions.py`
- Modify: `console/server/services/runtime/session_view_service.py`
- Test: `console/tests/test_run_query_service.py`
- Test: `console/tests/test_sessions_api.py`

- [ ] **Step 1: Write failing query-facade tests**

```python
import pytest

from agiwo.agent import MessageRole, RunFinished, RunMetrics, RunStarted, UserStepCommitted
from agiwo.agent.storage.base import InMemoryRunLogStorage
from server.services.runtime.run_query_service import RunQueryService


@pytest.mark.asyncio
async def test_run_query_service_lists_runs_newest_first() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            RunStarted(sequence=1, session_id="sess-1", run_id="run-1", agent_id="a", user_input="one"),
            RunFinished(sequence=2, session_id="sess-1", run_id="run-1", agent_id="a", response="first"),
            RunStarted(sequence=3, session_id="sess-1", run_id="run-2", agent_id="a", user_input="two"),
            RunFinished(sequence=4, session_id="sess-1", run_id="run-2", agent_id="a", response="second"),
        ]
    )
    service = RunQueryService(run_storage=storage)

    page = await service.list_runs(session_id="sess-1", limit=20, offset=0)

    assert [run.run_id for run in page.items] == ["run-2", "run-1"]
    assert page.has_more is False


@pytest.mark.asyncio
async def test_run_query_service_lists_session_steps_and_total() -> None:
    storage = InMemoryRunLogStorage()
    await storage.append_entries(
        [
            UserStepCommitted(
                sequence=1,
                session_id="sess-1",
                run_id="run-1",
                agent_id="agent-1",
                step_id="step-1",
                role=MessageRole.USER,
                content="hello",
                user_input="hello",
            )
        ]
    )
    service = RunQueryService(run_storage=storage)

    page = await service.list_session_steps("sess-1", limit=20, order="asc")

    assert [step.id for step in page.items] == ["step-1"]
    assert page.total == 1
```

- [ ] **Step 2: Run the targeted tests to confirm the gap**

Run: `cd console && uv run pytest tests/test_run_query_service.py tests/test_sessions_api.py -v`
Expected: FAIL because `RunQueryService` does not exist and `/api/runs` / `/api/sessions/{id}/steps` still read storage directly in the router.

- [ ] **Step 3: Implement the query facade**

```python
from dataclasses import dataclass

from agiwo.agent import RunLogStorage, RunView, StepView

from server.models.session import PageSlice


@dataclass(slots=True)
class RunQueryService:
    run_storage: RunLogStorage

    async def list_runs(
        self,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int,
        offset: int,
    ) -> PageSlice[RunView]:
        runs = await self.run_storage.list_run_views(
            user_id=user_id,
            session_id=session_id,
            limit=limit + 1,
            offset=offset,
        )
        has_more = len(runs) > limit
        return PageSlice(
            items=runs[:limit],
            limit=limit,
            offset=offset,
            has_more=has_more,
            total=None,
        )

    async def get_run(self, run_id: str) -> RunView | None:
        return await self.run_storage.get_run_view(run_id)

    async def list_session_steps(
        self,
        session_id: str,
        *,
        start_seq: int | None = None,
        end_seq: int | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
        limit: int,
        order: str,
    ) -> PageSlice[StepView]:
        raw_steps = await self.run_storage.list_step_views(
            session_id=session_id,
            start_seq=start_seq,
            end_seq=end_seq,
            run_id=run_id,
            agent_id=agent_id,
            limit=5001 if order == "desc" else limit + 1,
        )
        if order == "desc":
            raw_steps = list(reversed(raw_steps))
        has_more = len(raw_steps) > limit
        total = None
        if start_seq is None and end_seq is None and run_id is None and agent_id is None:
            total = await self.run_storage.get_committed_step_count(session_id)
        return PageSlice(
            items=raw_steps[:limit],
            limit=limit,
            offset=0,
            has_more=has_more,
            total=total,
        )
```

- [ ] **Step 4: Route session/run endpoints through the query facade**

```python
def get_run_query_service(runtime: ConsoleRuntime) -> RunQueryService:
    return RunQueryService(run_storage=runtime.run_log_storage)
```

```python
page = await get_run_query_service(runtime).list_runs(
    user_id=user_id,
    session_id=session_id,
    limit=limit,
    offset=offset,
)
```

```python
page = await get_run_query_service(runtime).list_session_steps(
    session_id,
    start_seq=start_seq,
    end_seq=end_seq,
    run_id=run_id,
    agent_id=agent_id,
    limit=limit,
    order=order,
)
```

- [ ] **Step 5: Make `SessionViewService` consume the same query facade**

```python
class SessionViewService:
    def __init__(
        self,
        *,
        run_queries: RunQueryService,
        session_store: ChannelChatSessionStore | None,
        scheduler: Scheduler | None,
    ) -> None:
        self._run_queries = run_queries
```

```python
stats = await self._run_queries.get_session_run_stats(session_id)
```

- [ ] **Step 6: Re-run the targeted console query tests**

Run: `cd console && uv run pytest tests/test_run_query_service.py tests/test_sessions_api.py -v`
Expected: PASS, with routers and session summaries reading through the console query facade instead of storage directly.

## Task 2: Rename Console Canonical-Agent Materialization Clearly

**Files:**
- Modify: `console/server/services/runtime/agent_factory.py`
- Modify: `console/server/services/runtime/agent_runtime_cache.py`
- Modify: `console/server/routers/scheduler.py`
- Modify: `console/server/services/runtime/__init__.py`
- Test: `console/tests/test_agent_runtime_components.py`
- Test: `console/tests/test_config_env.py`

- [ ] **Step 1: Write the failing naming-path tests**

```python
async def test_runtime_cache_uses_materialize_agent_entrypoint(monkeypatch):
    called = {}

    async def fake_materialize_agent(*args, **kwargs):
        called["ok"] = True
        return FakeAgent("sess-1")

    monkeypatch.setattr(
        "server.services.runtime.agent_runtime_cache.materialize_agent",
        fake_materialize_agent,
    )

    ...

    assert called["ok"] is True
```

- [ ] **Step 2: Run targeted tests to confirm the old name is still wired**

Run: `cd console && uv run pytest tests/test_agent_runtime_components.py tests/test_config_env.py -v`
Expected: FAIL because cache/router/tests still import `build_agent`.

- [ ] **Step 3: Rename `build_agent(...)` to `materialize_agent(...)`**

```python
async def materialize_agent(
    config: AgentConfigRecord,
    console_config: ConsoleConfig,
    registry: AgentRegistry,
    *,
    id: str | None = None,
    _building: set[str] | None = None,
) -> Agent:
    ...
```

```python
child_agent = await materialize_agent(
    child_config,
    console_config,
    registry,
    _building=_building.copy(),
)
```

- [ ] **Step 4: Update runtime cache, scheduler router, and resume paths to the new name**

```python
from server.services.runtime.agent_factory import materialize_agent
```

```python
agent = await materialize_agent(
    base_config,
    self._console_config,
    self._agent_registry,
    id=session.id,
)
```

- [ ] **Step 5: Re-run the targeted materialization tests**

Run: `cd console && uv run pytest tests/test_agent_runtime_components.py tests/test_config_env.py -v`
Expected: PASS, with all console runtime creation paths using the explicit canonical-agent materialization name.

## Task 3: Validate Console Runtime Migration End-To-End

**Files:**
- Modify: `docs/architecture/scheduler-console-runtime-refactor.md`
- Test: `console/tests/test_sessions_api.py`
- Test: `console/tests/test_scheduler_api.py`
- Test: `console/tests/test_scheduler_chat_api.py`

- [ ] **Step 1: Add the missing integration assertions**

```python
async def test_list_runs_uses_run_log_backed_view_ordering(client):
    response = await client.get("/api/runs", params={"session_id": "session-a"})
    payload = response.json()

    assert [item["id"] for item in payload["items"]] == ["run-a2", "run-a1"]
```

```python
async def test_session_steps_endpoint_returns_total_from_query_service(client):
    response = await client.get("/api/sessions/session-a/steps")
    payload = response.json()

    assert payload["total"] == 3
```

- [ ] **Step 2: Document the Phase 3 landing state**

```md
Phase 3 completed state:

- Console run/session/step queries now flow through a dedicated `RunQueryService`
- session summary/detail and timeline endpoints consume the same `RunLog`-backed facade
- console canonical-agent construction uses explicit `materialize_agent(...)` naming
```

- [ ] **Step 3: Run the affected console backend suite**

Run: `cd console && uv run pytest tests/test_sessions_api.py tests/test_scheduler_api.py tests/test_scheduler_chat_api.py -v`
Expected: PASS

- [ ] **Step 4: Run the console/backend lint gate**

Run: `uv run python scripts/lint.py ci`
Expected: PASS

## Coverage Check

This plan covers every remaining Phase 3 requirement from the design spec:

1. console queries migrate to `RunLog`-backed views: Task 1
2. direct assumptions about removed persisted models disappear behind a console query facade: Task 1
3. session summary and timeline views align on one query surface: Task 1 and Task 3
4. console-side canonical-agent materialization naming becomes explicit and unified: Task 2

There are no uncovered Phase 3 requirements in the current spec.
