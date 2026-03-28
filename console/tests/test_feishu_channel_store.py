from datetime import datetime, timedelta, timezone

import pytest

from server.channels.feishu.store import create_feishu_channel_store
from server.channels.feishu.store.sqlite import SqliteFeishuChannelStore
from server.channels.session.binding import open_initial_session
from server.channels.session.models import ChannelChatContext, Session


def _make_chat_context(
    *,
    context_id: str,
    scope_id: str,
    user_open_id: str = "user-1",
    updated_at: datetime | None = None,
    created_at: datetime | None = None,
) -> ChannelChatContext:
    now = updated_at or datetime.now(timezone.utc)
    return ChannelChatContext(
        id=context_id,
        scope_id=scope_id,
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id=user_open_id,
        base_agent_id="base-agent",
        current_session_id="session-current",
        created_at=created_at or now,
        updated_at=now,
    )


def _make_session(
    *,
    session_id: str,
    chat_context_id: str,
    updated_at: datetime,
    created_at: datetime | None = None,
) -> Session:
    return Session(
        id=session_id,
        chat_context_id=chat_context_id,
        base_agent_id="base-agent",
        runtime_agent_id=f"runtime-{session_id}",
        scheduler_state_id=f"state-{session_id}",
        created_by="AUTO",
        created_at=created_at or (updated_at - timedelta(minutes=1)),
        updated_at=updated_at,
    )


def _create_store(kind: str, tmp_path) -> object:
    return create_feishu_channel_store(
        db_path=str(tmp_path / "feishu-store.sqlite3"),
        use_persistent_store=kind == "sqlite",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_feishu_channel_store_round_trips_context_and_sessions(
    kind: str,
    tmp_path,
) -> None:
    store = _create_store(kind, tmp_path)
    await store.connect()
    try:
        now = datetime.now(timezone.utc)
        chat_context = _make_chat_context(
            context_id="ctx-1",
            scope_id="scope-1",
            updated_at=now,
        )
        later = now + timedelta(minutes=5)
        newer_session = _make_session(
            session_id="sess-2",
            chat_context_id=chat_context.id,
            updated_at=later,
        )
        older_session = _make_session(
            session_id="sess-1",
            chat_context_id=chat_context.id,
            updated_at=now,
        )

        assert await store.claim_event("feishu-main", "event-1") is True
        assert await store.claim_event("feishu-main", "event-1") is False

        await store.upsert_chat_context(chat_context)
        await store.upsert_session(older_session)
        await store.upsert_session(newer_session)

        loaded_context = await store.get_chat_context(chat_context.scope_id)
        loaded_by_id = await store.get_chat_context_by_id(chat_context.id)
        loaded_session = await store.get_session("sess-1")
        loaded_with_context = await store.get_session_with_context("sess-2")
        user_sessions = await store.list_sessions_by_user(chat_context.user_open_id)
        context_sessions = await store.list_sessions_by_chat_context(chat_context.id)

        assert loaded_context == chat_context
        assert loaded_by_id == chat_context
        assert loaded_session == older_session
        assert loaded_with_context is not None
        assert loaded_with_context.session == newer_session
        assert loaded_with_context.chat_context == chat_context
        assert [item.session.id for item in user_sessions] == ["sess-2", "sess-1"]
        assert [session.id for session in context_sessions] == ["sess-2", "sess-1"]
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_feishu_channel_store_applies_session_mutation_atomically(
    kind: str,
    tmp_path,
) -> None:
    store = _create_store(kind, tmp_path)
    await store.connect()
    try:
        now = datetime.now(timezone.utc)
        mutation = open_initial_session(
            chat_context_scope_id="scope-atomic",
            channel_instance_id="feishu-main",
            chat_id="chat-atomic",
            chat_type="p2p",
            user_open_id="user-atomic",
            base_agent_id="base-agent",
            created_by="AUTO",
            now=now,
        )

        await store.apply_session_mutation(mutation)

        assert (
            await store.get_chat_context(mutation.chat_context.scope_id)
            == mutation.chat_context
        )
        assert (
            await store.get_session(mutation.current_session.id)
            == mutation.current_session
        )
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_feishu_channel_store_keeps_indexes_in_sync_on_upsert(
    kind: str,
    tmp_path,
) -> None:
    store = _create_store(kind, tmp_path)
    await store.connect()
    try:
        now = datetime.now(timezone.utc)
        context_v1 = _make_chat_context(
            context_id="ctx-1",
            scope_id="scope-1",
            updated_at=now,
        )
        context_v2 = _make_chat_context(
            context_id="ctx-2",
            scope_id="scope-1",
            updated_at=now + timedelta(minutes=1),
            created_at=context_v1.created_at,
        )
        other_context = _make_chat_context(
            context_id="ctx-3",
            scope_id="scope-2",
            updated_at=now + timedelta(minutes=2),
        )
        await store.upsert_chat_context(context_v1)
        await store.upsert_chat_context(context_v2)
        await store.upsert_chat_context(other_context)

        assert await store.get_chat_context_by_id("ctx-1") is None
        assert await store.get_chat_context_by_id("ctx-2") == context_v2

        session_v1 = _make_session(
            session_id="sess-1",
            chat_context_id=context_v2.id,
            updated_at=now,
        )
        session_v2 = _make_session(
            session_id="sess-1",
            chat_context_id=other_context.id,
            updated_at=now + timedelta(minutes=3),
            created_at=session_v1.created_at,
        )
        await store.upsert_session(session_v1)
        await store.upsert_session(session_v2)

        moved_session = await store.get_session("sess-1")
        old_context_sessions = await store.list_sessions_by_chat_context(context_v2.id)
        new_context_sessions = await store.list_sessions_by_chat_context(
            other_context.id
        )

        assert moved_session == session_v2
        assert old_context_sessions == []
        assert [session.id for session in new_context_sessions] == ["sess-1"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_sqlite_feishu_channel_store_persists_across_reconnects(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "feishu-store.sqlite3")
    now = datetime.now(timezone.utc)
    chat_context = _make_chat_context(
        context_id="ctx-1",
        scope_id="scope-1",
        updated_at=now,
    )
    session = _make_session(
        session_id="sess-1",
        chat_context_id=chat_context.id,
        updated_at=now + timedelta(minutes=1),
    )

    first_store = SqliteFeishuChannelStore(db_path=db_path)
    await first_store.connect()
    try:
        await first_store.upsert_chat_context(chat_context)
        await first_store.upsert_session(session)
        assert await first_store.claim_event("feishu-main", "event-1") is True
    finally:
        await first_store.close()

    second_store = SqliteFeishuChannelStore(db_path=db_path)
    await second_store.connect()
    try:
        assert await second_store.claim_event("feishu-main", "event-1") is False
        assert (
            await second_store.get_chat_context(chat_context.scope_id) == chat_context
        )
        # feishu_session is dropped on reconnect (test-phase schema reset),
        # so the session written by first_store is gone.
        assert await second_store.get_session_with_context(session.id) is None
    finally:
        await second_store.close()


@pytest.mark.asyncio
async def test_sqlite_feishu_channel_store_preserves_created_at_on_upsert(
    tmp_path,
) -> None:
    db_path = str(tmp_path / "feishu-store.sqlite3")
    store = SqliteFeishuChannelStore(db_path=db_path)
    await store.connect()
    try:
        original_created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        original_updated_at = original_created_at + timedelta(minutes=1)
        chat_context_v1 = _make_chat_context(
            context_id="ctx-1",
            scope_id="scope-1",
            created_at=original_created_at,
            updated_at=original_updated_at,
        )
        session_v1 = _make_session(
            session_id="sess-1",
            chat_context_id=chat_context_v1.id,
            created_at=original_created_at,
            updated_at=original_updated_at,
        )

        await store.upsert_chat_context(chat_context_v1)
        await store.upsert_session(session_v1)

        replacement_created_at = original_created_at + timedelta(days=1)
        replacement_updated_at = original_updated_at + timedelta(days=1)
        chat_context_v2 = _make_chat_context(
            context_id="ctx-1",
            scope_id="scope-1",
            created_at=replacement_created_at,
            updated_at=replacement_updated_at,
        )
        session_v2 = _make_session(
            session_id="sess-1",
            chat_context_id=chat_context_v2.id,
            created_at=replacement_created_at,
            updated_at=replacement_updated_at,
        )

        await store.upsert_chat_context(chat_context_v2)
        await store.upsert_session(session_v2)

        loaded_context = await store.get_chat_context(chat_context_v2.scope_id)
        loaded_session = await store.get_session(session_v2.id)

        assert loaded_context is not None
        assert loaded_context.created_at == original_created_at
        assert loaded_context.updated_at == replacement_updated_at
        assert loaded_session is not None
        assert loaded_session.created_at == original_created_at
        assert loaded_session.updated_at == replacement_updated_at
    finally:
        await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["memory", "sqlite"])
async def test_feishu_channel_store_round_trips_task_and_fork_fields(
    kind: str,
    tmp_path,
) -> None:
    store = _create_store(kind, tmp_path)
    await store.connect()
    try:
        now = datetime.now(timezone.utc)
        chat_context = _make_chat_context(
            context_id="ctx-1",
            scope_id="scope-1",
            updated_at=now,
        )
        session = Session(
            id="sess-1",
            chat_context_id="ctx-1",
            base_agent_id="base-agent",
            runtime_agent_id="runtime-1",
            scheduler_state_id="state-1",
            created_by="AUTO",
            created_at=now,
            updated_at=now,
            current_task_id="task-42",
            task_message_count=3,
            source_session_id="sess-0",
            source_task_id="task-0",
            fork_context_summary="forked for subtask",
        )

        await store.upsert_chat_context(chat_context)
        await store.upsert_session(session)

        loaded = await store.get_session("sess-1")
        assert loaded is not None
        assert loaded.current_task_id == "task-42"
        assert loaded.task_message_count == 3
        assert loaded.source_session_id == "sess-0"
        assert loaded.source_task_id == "task-0"
        assert loaded.fork_context_summary == "forked for subtask"

        with_context = await store.get_session_with_context("sess-1")
        assert with_context is not None
        assert with_context.session.current_task_id == "task-42"
        assert with_context.session.source_session_id == "sess-0"
    finally:
        await store.close()
