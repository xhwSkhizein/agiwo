from types import SimpleNamespace

import pytest

from agiwo.agent import RunStatus
from server.models.metrics import RunMetricsSummary
from server.services.metrics import summarize_runs_paginated, summarize_traces_paginated


class FakeRunStorage:
    def __init__(self, runs):
        self._runs = runs

    async def list_runs(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ):
        runs = self._runs
        if user_id is not None:
            runs = [run for run in runs if run.user_id == user_id]
        if session_id is not None:
            runs = [run for run in runs if run.session_id == session_id]
        return runs[offset : offset + limit]


class FakeTraceStorage:
    def __init__(self, traces):
        self._traces = traces

    async def query_traces(self, query):
        limit = query.get("limit", 50)
        offset = query.get("offset", 0)
        traces = self._traces
        if query.get("agent_id") is not None:
            traces = [trace for trace in traces if trace.agent_id == query["agent_id"]]
        return traces[offset : offset + limit]


def _make_run(
    *,
    session_id: str,
    agent_id: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls_count: int = 0,
    token_cost: float = 0.0,
    status: RunStatus = RunStatus.COMPLETED,
):
    return SimpleNamespace(
        session_id=session_id,
        agent_id=agent_id,
        user_id=None,
        status=status,
        metrics=SimpleNamespace(
            steps_count=1,
            tool_calls_count=tool_calls_count,
            duration_ms=10.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            token_cost=token_cost,
        ),
    )


def _make_trace(
    *,
    trace_id: str,
    agent_id: str,
    total_tokens: int,
    total_llm_calls: int,
    total_tool_calls: int,
    total_token_cost: float,
):
    return SimpleNamespace(
        trace_id=trace_id,
        agent_id=agent_id,
        total_tokens=total_tokens,
        total_llm_calls=total_llm_calls,
        total_tool_calls=total_tool_calls,
        total_token_cost=total_token_cost,
    )


@pytest.mark.asyncio
async def test_summarize_runs_paginated_aggregates_all_pages() -> None:
    storage = FakeRunStorage(
        [
            _make_run(
                session_id="sess-1",
                agent_id="agent-a",
                input_tokens=10,
                output_tokens=5,
                tool_calls_count=1,
                token_cost=0.1,
            ),
            _make_run(
                session_id="sess-1",
                agent_id="agent-a",
                input_tokens=7,
                output_tokens=3,
                token_cost=0.2,
            ),
            _make_run(
                session_id="sess-1",
                agent_id="agent-b",
                input_tokens=2,
                output_tokens=1,
                token_cost=0.05,
                status=RunStatus.RUNNING,
            ),
        ]
    )

    summary = await summarize_runs_paginated(
        storage,
        session_id="sess-1",
        page_size=2,
    )

    assert summary.run_count == 3
    assert summary.completed_run_count == 2
    assert summary.step_count == 3
    assert summary.tool_calls_count == 1
    assert summary.duration_ms == 30.0
    assert summary.input_tokens == 19
    assert summary.output_tokens == 9
    assert summary.total_tokens == 28
    assert summary.cache_read_tokens == 0
    assert summary.cache_creation_tokens == 0
    assert summary.token_cost == pytest.approx(0.35)


@pytest.mark.asyncio
async def test_summarize_runs_paginated_returns_empty_summary_for_missing_session() -> (
    None
):
    storage = FakeRunStorage([])

    summary = await summarize_runs_paginated(storage, session_id="missing", page_size=2)

    assert summary == RunMetricsSummary()


@pytest.mark.asyncio
async def test_summarize_traces_paginated_aggregates_all_pages() -> None:
    storage = FakeTraceStorage(
        [
            _make_trace(
                trace_id="trace-1",
                agent_id="agent-a",
                total_tokens=11,
                total_llm_calls=1,
                total_tool_calls=0,
                total_token_cost=0.1,
            ),
            _make_trace(
                trace_id="trace-2",
                agent_id="agent-a",
                total_tokens=7,
                total_llm_calls=2,
                total_tool_calls=1,
                total_token_cost=0.2,
            ),
            _make_trace(
                trace_id="trace-3",
                agent_id="agent-b",
                total_tokens=5,
                total_llm_calls=1,
                total_tool_calls=3,
                total_token_cost=0.05,
            ),
        ]
    )

    summary = await summarize_traces_paginated(
        storage,
        agent_id="agent-a",
        page_size=1,
    )

    assert summary.trace_count == 2
    assert summary.total_tokens == 18
    assert summary.total_llm_calls == 3
    assert summary.total_tool_calls == 1
    assert summary.total_token_cost == pytest.approx(0.3)
