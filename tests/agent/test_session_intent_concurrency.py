"""Concurrency and truncation tests for SessionIntent stores (Task 21)."""

import asyncio
import tempfile
from pathlib import Path

import aiosqlite
import pytest

from agiwo.agent.intent.base import InMemorySessionIntentStore, SessionIntentStore
from agiwo.agent.intent.models import MAX_INTENT_ENTRIES, IntentEntry
from agiwo.agent.intent.sqlite import SQLiteSessionIntentStore


def _entry(n: int) -> IntentEntry:
    return IntentEntry(kind="user_input", text=f"msg-{n}", at=n)


async def _append_many(store: SessionIntentStore, session_id: str, count: int) -> None:
    await asyncio.gather(
        *[store.append_entry(session_id, _entry(i)) for i in range(count)]
    )


@pytest.mark.asyncio
async def test_in_memory_concurrent_appends_lose_nothing() -> None:
    store = InMemorySessionIntentStore()
    session_id = "session-concurrent-mem"

    # Amplify interleaving pressure with a yield inside the unlocked path.
    original = store._append_entry_unlocked

    async def _slow_append(session_id, entry, *, last_run_plan=None):  # noqa: ANN001
        await asyncio.sleep(0)
        return await original(session_id, entry, last_run_plan=last_run_plan)

    store._append_entry_unlocked = _slow_append  # type: ignore[method-assign]
    await _append_many(store, session_id, 50)
    intent = await store.get(session_id)
    assert intent is not None
    assert len(intent.entries) == 50
    assert {e.text for e in intent.entries} == {f"msg-{i}" for i in range(50)}


@pytest.mark.asyncio
async def test_sqlite_concurrent_appends_lose_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        store = SQLiteSessionIntentStore(db_path=str(Path(tmpdir) / "intent.db"))
        try:
            session_id = "session-concurrent-sqlite"
            original = store._append_entry_unlocked

            async def _slow_append(session_id, entry, *, last_run_plan=None):  # noqa: ANN001
                await asyncio.sleep(0)
                return await original(session_id, entry, last_run_plan=last_run_plan)

            store._append_entry_unlocked = _slow_append  # type: ignore[method-assign]
            await _append_many(store, session_id, 50)
            intent = await store.get(session_id)
            assert intent is not None
            assert len(intent.entries) == 50
            assert {e.text for e in intent.entries} == {f"msg-{i}" for i in range(50)}
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_get_returns_newest_n_after_overflow() -> None:
    store = InMemorySessionIntentStore()
    session_id = "session-truncate"
    total = MAX_INTENT_ENTRIES * 2 + 1
    for i in range(total):
        await store.append_entry(session_id, _entry(i))
    intent = await store.get(session_id)
    assert intent is not None
    assert len(intent.entries) == MAX_INTENT_ENTRIES
    assert intent.entries[0].text == f"msg-{total - MAX_INTENT_ENTRIES}"
    assert intent.entries[-1].text == f"msg-{total - 1}"


@pytest.mark.asyncio
async def test_sqlite_rejects_legacy_single_table_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "legacy.db")

        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE session_intent (
                    session_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at INTEGER NOT NULL
                )
                """
            )
            await conn.commit()

        store = SQLiteSessionIntentStore(db_path=db_path)
        with pytest.raises(RuntimeError, match="incompatible"):
            await store.connect()
