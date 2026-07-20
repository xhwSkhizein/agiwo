"""Replayable Objective SSE tests (P5-02)."""

import pytest

from agiwo.objective import (
    BudgetLimits,
    CreateObjectiveRequest,
    ObjectiveService,
)
from agiwo.agent.models.input import UserMessage
from agiwo.objective.store.memory import InMemoryObjectiveStore

from server.services.objective_event_stream import stream_objective_events


@pytest.mark.asyncio
async def test_stream_replays_facts_after_cursor() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    result = await service.create_objective(
        CreateObjectiveRequest(
            session_id="sess-sse",
            user_message=UserMessage.from_value("hello"),
            budget=BudgetLimits(
                handoffs=5,
                verification_attempts=3,
                llm_cost_usd=2.0,
                active_seconds=600,
            ),
            idempotency_key="sse-1",
        )
    )
    facts = await store.list_facts(objective_id=result.objective_id)
    assert len(facts) >= 1
    first = facts[0].sequence

    messages = []
    async for message in stream_objective_events(
        store,
        result.objective_id,
        after_sequence=0,
        wait_timeout=0.05,
        max_pending=500,
    ):
        messages.append(message)
        if len(messages) >= len(facts):
            break

    assert messages
    assert messages[0]["event"] == "objective_event"
    assert messages[0]["id"] == str(first)
    assert '"type": "objective_event"' in messages[0]["data"] or (
        '"type":"objective_event"' in messages[0]["data"]
    )

    after_first = []
    async for message in stream_objective_events(
        store,
        result.objective_id,
        after_sequence=first,
        wait_timeout=0.05,
        max_pending=500,
    ):
        after_first.append(message)
        if len(after_first) >= max(0, len(facts) - 1):
            break

    for message in after_first:
        assert message["id"] != str(first)


@pytest.mark.asyncio
async def test_stream_stops_when_client_lags() -> None:
    store = InMemoryObjectiveStore()
    service = ObjectiveService(store)
    result = await service.create_objective(
        CreateObjectiveRequest(
            session_id="sess-lag",
            user_message=UserMessage.from_value("lag"),
            budget=BudgetLimits(
                handoffs=5,
                verification_attempts=3,
                llm_cost_usd=2.0,
                active_seconds=600,
            ),
            idempotency_key="lag-1",
        )
    )
    # Create leaves multiple facts; max_pending=1 forces overflow stop.
    out = []
    async for message in stream_objective_events(
        store,
        result.objective_id,
        after_sequence=0,
        wait_timeout=0.01,
        max_pending=1,
    ):
        out.append(message)
    assert len(out) == 1
