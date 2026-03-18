from types import SimpleNamespace

import pytest

from agiwo.agent import ContentPart, ContentType, serialize_user_input
from server.domain.run_metrics import RunMetricsSummary
from server.domain.sessions import (
    session_aggregate_to_chat_summary,
    session_aggregate_to_summary_data,
)
from server.services.metrics import collect_session_aggregates


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


def _make_run(
    *,
    run_id: str,
    session_id: str,
    agent_id: str,
    user_input,
    response_content: str | None,
    created_at: int,
    updated_at: int,
):
    return SimpleNamespace(
        id=run_id,
        session_id=session_id,
        agent_id=agent_id,
        user_id=None,
        user_input=user_input,
        response_content=response_content,
        created_at=created_at,
        updated_at=updated_at,
        status="completed",
        metrics=SimpleNamespace(
            steps_count=1,
            tool_calls_count=0,
            duration_ms=1.0,
            input_tokens=1,
            output_tokens=1,
            total_tokens=2,
            cache_read_tokens=0,
            cache_creation_tokens=0,
            token_cost=0.1,
        ),
    )


@pytest.mark.asyncio
async def test_collect_session_aggregates_uses_latest_run_and_paginates() -> None:
    storage = FakeRunStorage(
        [
            _make_run(
                run_id="run-1",
                session_id="sess-1",
                agent_id="agent-a",
                user_input="first",
                response_content="old-response",
                created_at=1,
                updated_at=1,
            ),
            _make_run(
                run_id="run-2",
                session_id="sess-1",
                agent_id="agent-a",
                user_input=serialize_user_input(
                    [ContentPart(type=ContentType.TEXT, text="latest")]
                ),
                response_content="new-response",
                created_at=2,
                updated_at=3,
            ),
            _make_run(
                run_id="run-3",
                session_id="sess-2",
                agent_id="agent-b",
                user_input="other",
                response_content="other-response",
                created_at=1,
                updated_at=2,
            ),
        ]
    )

    sessions = await collect_session_aggregates(storage, agent_id="agent-a")

    assert len(sessions) == 1
    summary = session_aggregate_to_summary_data(sessions[0])
    chat_summary = session_aggregate_to_chat_summary(sessions[0])

    assert summary.session_id == "sess-1"
    assert summary.last_response == "new-response"
    assert isinstance(summary.last_user_input, list)
    assert len(summary.last_user_input) == 1
    assert summary.last_user_input[0].type == ContentType.TEXT
    assert summary.last_user_input[0].text == "latest"
    assert summary.run_count == 2
    assert isinstance(summary.metrics, RunMetricsSummary)
    assert summary.metrics.run_count == 2
    assert (
        chat_summary["last_input"]
        == '{"__type": "content_parts", "parts": [{"type": "text", "text": "latest"}]}'
    )
