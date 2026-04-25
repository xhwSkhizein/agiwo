from datetime import datetime, timezone

import pytest

from agiwo.agent import ContentPart, ContentType, UserMessage
from agiwo.agent.models.log import RunFinished, RunStarted
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.agent.models.run import RunMetrics
from agiwo.observability.trace import Span, SpanKind, SpanStatus, Trace
from server.config import ConsoleConfig
from server.models.session import Session
from server.services.runtime.run_query_service import RunQueryService
from server.services.runtime.trace_query_service import TraceQueryService
from server.services.runtime.session_view_service import SessionViewService
from server.services.storage_wiring import create_trace_storage


class FakeSessionStore:
    def __init__(self, sessions: list[Session]):
        self._sessions = {s.id: s for s in sessions}

    async def get_session(self, session_id: str):
        return self._sessions.get(session_id)

    async def get_chat_context(self, scope_id: str):
        return None

    async def list_sessions(self):
        return list(self._sessions.values())

    async def list_sessions_by_base_agent(self, base_agent_id: str):
        return [s for s in self._sessions.values() if s.base_agent_id == base_agent_id]


def _make_session(session_id: str, agent_id: str = "agent-a") -> Session:
    now = datetime.now(timezone.utc)
    return Session(
        id=session_id,
        chat_context_scope_id=None,
        base_agent_id=agent_id,
        created_by="test",
        created_at=now,
        updated_at=now,
    )


def _make_trace_query_service() -> TraceQueryService:
    config = ConsoleConfig(
        storage={
            "run_log_type": "memory",
            "trace_type": "memory",
            "metadata_type": "memory",
        }
    )
    return TraceQueryService(trace_storage=create_trace_storage(config))


async def _append_run_view_entries(
    storage: InMemoryRunLogStorage,
    *,
    session_id: str,
    run_id: str,
    agent_id: str,
    user_input,
    response: str | None,
    metrics: RunMetrics | None = None,
) -> None:
    await storage.append_entries(
        [
            RunStarted(
                sequence=1 if run_id.endswith("1") else 101,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                user_input=user_input,
            ),
            RunFinished(
                sequence=100 if run_id.endswith("1") else 200,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                response=response,
                metrics=metrics.to_dict() if metrics is not None else None,
            ),
        ]
    )


@pytest.mark.asyncio
async def test_list_sessions_returns_summary_with_latest_run() -> None:
    session = _make_session("sess-1")
    store = FakeSessionStore([session])
    storage = InMemoryRunLogStorage()

    await _append_run_view_entries(
        storage,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-a",
        user_input="first input",
        response="old-response",
    )
    structured_input = UserMessage.serialize(
        [ContentPart(type=ContentType.TEXT, text="latest")]
    )
    await _append_run_view_entries(
        storage,
        session_id="sess-1",
        run_id="run-2",
        agent_id="agent-a",
        user_input=structured_input,
        response="new-response",
    )

    service = SessionViewService(
        run_queries=RunQueryService(run_storage=storage),
        trace_queries=_make_trace_query_service(),
        session_store=store,
        scheduler=None,
    )
    page = await service.list_sessions(limit=10, offset=0)

    assert len(page.items) == 1
    summary = page.items[0]
    assert summary.session_id == "sess-1"
    assert summary.base_agent_id == "agent-a"
    assert summary.last_response == "new-response"
    assert summary.run_count == 2
    assert summary.step_count == 0


@pytest.mark.asyncio
async def test_get_session_detail_populates_metrics() -> None:
    session = _make_session("sess-1")
    store = FakeSessionStore([session])
    storage = InMemoryRunLogStorage()

    metrics = RunMetrics(
        steps_count=1,
        tool_calls_count=0,
        duration_ms=1.0,
        input_tokens=10,
        output_tokens=5,
        total_tokens=15,
        cache_read_tokens=0,
        cache_creation_tokens=0,
        token_cost=0.01,
    )
    await _append_run_view_entries(
        storage,
        session_id="sess-1",
        run_id="run-1",
        agent_id="agent-a",
        user_input="hello",
        response="world",
        metrics=metrics,
    )
    trace_queries = _make_trace_query_service()
    await trace_queries.trace_storage.save_trace(
        Trace(
            trace_id="trace-1",
            agent_id="agent-a",
            session_id="sess-1",
            status=SpanStatus.OK,
            total_tokens=15,
        )
    )

    service = SessionViewService(
        run_queries=RunQueryService(run_storage=storage),
        trace_queries=trace_queries,
        session_store=store,
        scheduler=None,
    )
    detail = await service.get_session_detail("sess-1")

    assert detail is not None
    assert detail.summary.session_id == "sess-1"
    assert detail.summary.metrics.run_count == 1
    assert detail.summary.metrics.input_tokens == 10
    assert detail.summary.metrics.token_cost == pytest.approx(0.01)
    assert detail.session is not None
    assert detail.session.id == "sess-1"
    assert detail.observability is not None
    assert len(detail.observability.recent_traces) == 1
    assert detail.observability.recent_traces[0].trace_id == "trace-1"


def test_select_mainline_trace_skips_trace_without_root_agent_span() -> None:
    malformed = Trace(
        trace_id="malformed",
        agent_id="agent-a",
        session_id="sess-1",
        status=SpanStatus.OK,
    )
    mainline = Trace(
        trace_id="mainline",
        agent_id="agent-a",
        session_id="sess-1",
        status=SpanStatus.OK,
    )
    mainline.spans = [
        Span(
            trace_id="mainline",
            span_id="root",
            parent_span_id=None,
            kind=SpanKind.AGENT,
            name="agent-a",
            depth=0,
            run_id="run-1",
            attributes={"nested": False},
        )
    ]

    selected = SessionViewService._select_mainline_trace([malformed, mainline])

    assert selected is mainline


@pytest.mark.asyncio
async def test_get_session_detail_returns_none_for_missing() -> None:
    store = FakeSessionStore([])
    storage = InMemoryRunLogStorage()

    service = SessionViewService(
        run_queries=RunQueryService(run_storage=storage),
        trace_queries=_make_trace_query_service(),
        session_store=store,
        scheduler=None,
    )
    detail = await service.get_session_detail("nonexistent")

    assert detail is None
