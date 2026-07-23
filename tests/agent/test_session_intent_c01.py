"""C-01: SessionIntent model and I-S1 store tests."""

import tempfile
from pathlib import Path

import pytest

from agiwo.agent import (
    InMemorySessionIntentStore,
    IntentEntry,
    RunLogStorageConfig,
    SQLiteSessionIntentStore,
    create_session_intent_store,
)
from agiwo.agent.models.plan import Milestone, RunPlan


def _user_entry(text: str, *, at: int) -> IntentEntry:
    return IntentEntry(kind="user_input", text=text, at=at)


def _report_entry(text: str, *, at: int, run_id: str) -> IntentEntry:
    return IntentEntry(kind="run_report", text=text, at=at, run_id=run_id)


@pytest.mark.asyncio
async def test_in_memory_append_preserves_order() -> None:
    store = InMemorySessionIntentStore()
    session_id = "session-order"

    await store.append_entry(session_id, _user_entry("first", at=1))
    await store.append_entry(session_id, _user_entry("second", at=2))
    await store.append_entry(
        session_id,
        _report_entry("summary", at=3, run_id="run-1"),
    )

    intent = await store.get(session_id)
    assert intent is not None
    assert [entry.text for entry in intent.entries] == [
        "first",
        "second",
        "summary",
    ]
    assert [entry.kind for entry in intent.entries] == [
        "user_input",
        "user_input",
        "run_report",
    ]


@pytest.mark.asyncio
async def test_in_memory_round_trip_with_last_run_plan() -> None:
    store = InMemorySessionIntentStore()
    session_id = "session-plan"
    plan = RunPlan(
        milestones=[
            Milestone(id="m1", description="ship intent", status="completed"),
        ],
        revision=2,
    )

    await store.append_entry(
        session_id,
        _user_entry("plan this", at=10),
        last_run_plan=plan,
    )

    loaded = await store.get(session_id)
    assert loaded is not None
    assert loaded.last_run_plan is not None
    assert loaded.last_run_plan.revision == 2
    assert loaded.last_run_plan.milestones[0].description == "ship intent"
    assert loaded.updated_at >= 10


@pytest.mark.asyncio
async def test_in_memory_isolates_sessions() -> None:
    store = InMemorySessionIntentStore()

    await store.append_entry("session-a", _user_entry("a-only", at=1))
    await store.append_entry("session-b", _user_entry("b-only", at=2))

    intent_a = await store.get("session-a")
    intent_b = await store.get("session-b")

    assert intent_a is not None
    assert intent_b is not None
    assert [entry.text for entry in intent_a.entries] == ["a-only"]
    assert [entry.text for entry in intent_b.entries] == ["b-only"]


@pytest.mark.asyncio
async def test_sqlite_round_trip_and_isolation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "intent.db")
        store = SQLiteSessionIntentStore(db_path=db_path)
        try:
            await store.append_entry("sess-1", _user_entry("hello", at=5))
            await store.append_entry("sess-2", _user_entry("world", at=6))

            one = await store.get("sess-1")
            two = await store.get("sess-2")

            assert one is not None
            assert two is not None
            assert one.entries[0].text == "hello"
            assert two.entries[0].text == "world"
        finally:
            await store.close()


def test_factory_matches_run_log_backend() -> None:
    memory_store = create_session_intent_store(
        RunLogStorageConfig(storage_type="memory")
    )
    assert isinstance(memory_store, InMemorySessionIntentStore)

    with tempfile.TemporaryDirectory() as tmpdir:
        sqlite_store = create_session_intent_store(
            RunLogStorageConfig(
                storage_type="sqlite",
                config={"db_path": str(Path(tmpdir) / "shared.db")},
            )
        )
        assert isinstance(sqlite_store, SQLiteSessionIntentStore)
