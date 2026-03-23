"""Tests for AgentStateStorage implementations."""

from datetime import datetime, timezone

import pytest

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    SchedulerEventType,
    WakeCondition,
    WakeType,
)
from agiwo.scheduler.store import InMemoryAgentStateStorage, SQLiteAgentStateStorage


@pytest.fixture
def store():
    return InMemoryAgentStateStorage()


def _make_state(
    id: str = "agent-1",
    session_id: str = "sess-1",
    parent_id: str | None = "parent",
    status: AgentStateStatus = AgentStateStatus.PENDING,
    task: str = "do something",
    **kwargs,
) -> AgentState:
    return AgentState(
        id=id,
        session_id=session_id,
        status=status,
        task=task,
        parent_id=parent_id,
        **kwargs,
    )


class TestInMemorySaveAndGet:
    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        state = _make_state()
        await store.save_state(state)
        retrieved = await store.get_state("agent-1")
        assert retrieved is not None
        assert retrieved.id == "agent-1"
        assert retrieved.status == AgentStateStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        assert await store.get_state("nope") is None

    @pytest.mark.asyncio
    async def test_get_returns_snapshot_objects(self, store):
        state = _make_state()
        await store.save_state(state)

        first = await store.get_state("agent-1")
        second = await store.get_state("agent-1")

        assert first is not None
        assert second is not None
        assert first is not state
        assert second is not state
        assert second is not first


class TestInMemorySaveStateUpdate:
    @pytest.mark.asyncio
    async def test_save_updates_status_and_wake_condition(self, store):
        state = _make_state()
        await store.save_state(state)
        wake = WakeCondition(type=WakeType.WAITSET, wait_for=["c1"])
        await store.save_state(state.with_waiting(wake_condition=wake))
        retrieved = await store.get_state("agent-1")
        assert retrieved is not None
        assert retrieved.status == AgentStateStatus.WAITING
        assert retrieved.wake_condition is not None
        assert retrieved.wake_condition.wait_for == ["c1"]

    @pytest.mark.asyncio
    async def test_save_updates_pending_input_and_summary(self, store):
        state = _make_state()
        await store.save_state(state)
        await store.save_state(
            state.with_updates(
                pending_input="next",
                result_summary="done",
                explain="idle",
                wake_count=2,
            )
        )
        retrieved = await store.get_state("agent-1")
        assert retrieved is not None
        assert retrieved.pending_input == "next"
        assert retrieved.result_summary == "done"
        assert retrieved.explain == "idle"
        assert retrieved.wake_count == 2


class TestInMemoryListStates:
    @pytest.mark.asyncio
    async def test_filters_by_parent_and_session(self, store):
        await store.save_state(_make_state(id="a", parent_id="root", session_id="s1"))
        await store.save_state(_make_state(id="b", parent_id="root", session_id="s2"))
        await store.save_state(_make_state(id="c", parent_id="other", session_id="s1"))
        result = await store.list_states(parent_id="root", session_id="s1", limit=1000)
        assert [state.id for state in result] == ["a"]

    @pytest.mark.asyncio
    async def test_filters_by_statuses(self, store):
        await store.save_state(_make_state(id="a", status=AgentStateStatus.PENDING))
        await store.save_state(_make_state(id="b", status=AgentStateStatus.RUNNING))
        await store.save_state(_make_state(id="c", status=AgentStateStatus.WAITING))
        result = await store.list_states(
            statuses=(AgentStateStatus.PENDING, AgentStateStatus.WAITING),
            limit=1000,
        )
        assert {state.id for state in result} == {"a", "c"}

    @pytest.mark.asyncio
    async def test_filters_by_signal_propagated(self, store):
        await store.save_state(_make_state(id="a", signal_propagated=True))
        await store.save_state(_make_state(id="b", signal_propagated=False))
        result = await store.list_states(signal_propagated=True, limit=1000)
        assert [state.id for state in result] == ["a"]


def _make_event(
    id: str = "evt-1",
    target_agent_id: str = "parent",
    session_id: str = "sess-1",
    event_type: SchedulerEventType = SchedulerEventType.CHILD_COMPLETED,
    payload: dict | None = None,
    source_agent_id: str | None = "child",
    created_at: datetime | None = None,
) -> PendingEvent:
    return PendingEvent(
        id=id,
        target_agent_id=target_agent_id,
        session_id=session_id,
        event_type=event_type,
        payload=payload or {"result": "done"},
        source_agent_id=source_agent_id,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestInMemoryEvents:
    @pytest.mark.asyncio
    async def test_save_and_list_events(self, store):
        await store.save_event(_make_event())
        events = await store.list_events(target_agent_id="parent", session_id="sess-1")
        assert len(events) == 1
        assert events[0].id == "evt-1"

    @pytest.mark.asyncio
    async def test_delete_events(self, store):
        await store.save_event(_make_event(id="e1"))
        await store.save_event(_make_event(id="e2"))
        await store.delete_events(["e1"])
        events = await store.list_events(target_agent_id="parent", session_id="sess-1")
        assert [event.id for event in events] == ["e2"]


class TestSQLiteAgentStateStorage:
    @pytest.mark.asyncio
    async def test_save_and_get_round_trip(self, tmp_path):
        store = SQLiteAgentStateStorage(str(tmp_path / "scheduler.db"))
        base_state = _make_state(
            id="sqlite-agent",
            status=AgentStateStatus.PENDING,
        )
        state = base_state.with_updates(
            status=AgentStateStatus.WAITING,
            explain="waiting for child",
            pending_input="next input",
            wake_condition=WakeCondition(
                type=WakeType.WAITSET,
                wait_for=["child-1"],
                completed_ids=["child-1"],
            ),
        )

        await store.save_state(state)
        retrieved = await store.get_state("sqlite-agent")

        assert retrieved is not None
        assert retrieved.id == "sqlite-agent"
        assert retrieved.status == AgentStateStatus.WAITING
        assert retrieved.pending_input == "next input"
        assert retrieved.explain == "waiting for child"
        assert retrieved.wake_condition is not None
        assert retrieved.wake_condition.completed_ids == ["child-1"]

        await store.close()

    @pytest.mark.asyncio
    async def test_get_returns_snapshot_objects(self, tmp_path):
        store = SQLiteAgentStateStorage(str(tmp_path / "scheduler.db"))
        state = _make_state(id="sqlite-snapshot")
        await store.save_state(state)

        first = await store.get_state("sqlite-snapshot")
        second = await store.get_state("sqlite-snapshot")

        assert first is not None
        assert second is not None
        assert first is not state
        assert second is not state
        assert second is not first

        await store.close()
