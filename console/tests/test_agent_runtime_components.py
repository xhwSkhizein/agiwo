from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from agiwo.agent import RunCompletedEvent
from agiwo.scheduler.commands import RouteResult

from server.channels.batch_manager import ChannelBatchManager
from server.config import ConsoleConfig
from server.models.session import (
    BatchContext,
    BatchPayload,
    ChannelChatContext,
    Session,
    SessionWithContext,
)
from server.services.agent_registry import AgentConfigRecord
from server.services.runtime import (
    AgentRuntimeCache,
    CachedAgent,
    SessionContextService,
    SessionRuntimeService,
)


class FakeChannelChatSessionStore:
    def __init__(self) -> None:
        self.chat_contexts: dict[str, ChannelChatContext] = {}
        self.sessions: dict[str, Session] = {}
        self.upserted_chat_contexts: list[ChannelChatContext] = []
        self.upserted_sessions: list[Session] = []

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        return self.chat_contexts.get(scope_id)

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        self.chat_contexts[chat_context.scope_id] = chat_context
        self.upserted_chat_contexts.append(chat_context)

    async def get_session(self, session_id: str) -> Session | None:
        return self.sessions.get(session_id)

    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None:
        session = self.sessions.get(session_id)
        if session is None or session.chat_context_scope_id is None:
            return None
        chat_context = self.chat_contexts.get(session.chat_context_scope_id)
        if chat_context is None:
            return None
        return SessionWithContext(session=session, chat_context=chat_context)

    async def upsert_session(self, session: Session) -> None:
        self.sessions[session.id] = session
        self.upserted_sessions.append(session)

    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]:
        items: list[SessionWithContext] = []
        for session in self.sessions.values():
            if session.chat_context_scope_id is None:
                continue
            chat_context = self.chat_contexts.get(session.chat_context_scope_id)
            if chat_context is None or chat_context.user_open_id != user_open_id:
                continue
            items.append(SessionWithContext(session=session, chat_context=chat_context))
        return items

    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]:
        return [
            session
            for session in self.sessions.values()
            if session.chat_context_scope_id == chat_context_scope_id
        ]

    async def list_sessions_by_base_agent(self, base_agent_id: str) -> list[Session]:
        return [
            session
            for session in self.sessions.values()
            if session.base_agent_id == base_agent_id
        ]

    async def list_sessions(self) -> list[Session]:
        return list(self.sessions.values())


class FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _TestChannelService:
    def __init__(self, *, session_service, agent_pool, executor) -> None:
        self.reply_calls: list[tuple[BatchContext, str]] = []
        self._session_service = session_service
        self._agent_pool = agent_pool
        self._executor = executor
        self._batch_mgr = ChannelBatchManager(
            on_batch_ready=lambda *a: None,
            debounce_ms=1,
            max_batch_window_ms=1,
        )

    async def close_base(self) -> None:
        await self._batch_mgr.close()
        await self._agent_pool.close()

    async def _deliver_reply(self, context: BatchContext, text: str) -> None:
        self.reply_calls.append((context, text))

    async def _execute_batch(self, batch: BatchPayload) -> None:
        resolution = await self._session_service.get_or_create_current_session(
            batch.context
        )
        session = resolution.session
        agent = await self._agent_pool.get_or_create_runtime_agent(session)
        dispatch = await self._executor.execute(agent, session, batch.user_message)

        if dispatch.stream is None:
            if await self._can_deliver_session(batch.context, session):
                await self._deliver_reply(batch.context, "消息已收到，正在继续处理。")
            return

        final_text = None
        async for item in dispatch.stream:
            if (
                isinstance(item, RunCompletedEvent)
                and item.depth == 0
                and item.response
            ):
                final_text = item.response

        if final_text is None:
            return
        if await self._can_deliver_session(batch.context, session):
            await self._deliver_reply(batch.context, final_text)

    async def _can_deliver_session(
        self, context: BatchContext, session: Session
    ) -> bool:
        (
            _,
            current_session,
        ) = await self._session_service.get_chat_context_and_current_session(
            context.chat_context_scope_id
        )
        return current_session is not None and current_session.id == session.id


@pytest.mark.asyncio
async def test_session_service_rebinds_missing_base_agent_to_default() -> None:
    store = FakeChannelChatSessionStore()
    now = datetime.now(timezone.utc)
    chat_context = ChannelChatContext(
        scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="old-context-base",
        current_session_id="sess-1",
        created_at=now,
        updated_at=now,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="missing-base",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )
    await store.upsert_chat_context(chat_context)
    await store.upsert_session(session)

    default_config = AgentConfigRecord(
        id="agent-default",
        name="default",
        model_provider="openai",
        model_name="gpt-test",
    )
    registry = SimpleNamespace(
        get_agent=AsyncMock(return_value=None),
        get_agent_by_name=AsyncMock(return_value=default_config),
    )
    service = SessionContextService(
        store=store,
        agent_registry=registry,
        default_agent_name="default",
    )

    result = await service.get_or_create_current_session(
        BatchContext(
            chat_context_scope_id="scope-1",
            channel_instance_id="feishu-main",
            chat_id="chat-1",
            chat_type="p2p",
            trigger_user_id="user-1",
            trigger_message_id="msg-1",
            base_agent_id="incoming-base",
        )
    )

    assert result.session.base_agent_id == "agent-default"
    assert result.chat_context.base_agent_id == "agent-default"


@pytest.mark.asyncio
async def test_session_service_create_standalone_session_has_no_chat_context() -> None:
    service = SessionContextService(
        store=FakeChannelChatSessionStore(),
        agent_registry=SimpleNamespace(),
        default_agent_name="default",
    )

    session = await service.create_standalone_session(
        base_agent_id="agent-1",
        created_by="CONSOLE_CREATE",
    )

    assert session.base_agent_id == "agent-1"
    assert session.chat_context_scope_id is None


@pytest.mark.asyncio
async def test_agent_runtime_cache_uses_session_id_as_runtime_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeChannelChatSessionStore()
    base_config = AgentConfigRecord(
        id="base-agent",
        name="base",
        model_provider="openai",
        model_name="gpt-test",
    )
    registry = SimpleNamespace(get_agent=AsyncMock(return_value=base_config))
    scheduler = SimpleNamespace(rebind_agent=AsyncMock(return_value=True))
    built_agent = FakeAgent("sess-1")

    async def fake_build_agent(*args, **kwargs):
        return built_agent

    monkeypatch.setattr(
        "server.services.runtime.agent_runtime_cache.build_agent",
        fake_build_agent,
    )

    pool = AgentRuntimeCache(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=ConsoleConfig(),
        session_store=store,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    agent = await pool.get_or_create_runtime_agent(session)

    assert agent is built_agent
    assert store.sessions["sess-1"] is session
    assert pool.runtime_agents["sess-1"] is built_agent


@pytest.mark.asyncio
async def test_agent_runtime_cache_defers_refresh_while_state_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeChannelChatSessionStore()
    base_config = AgentConfigRecord(
        id="base-agent",
        name="base",
        model_provider="openai",
        model_name="gpt-test",
        system_prompt="updated prompt",
    )
    registry = SimpleNamespace(get_agent=AsyncMock(return_value=base_config))
    scheduler = SimpleNamespace(rebind_agent=AsyncMock(return_value=False))
    replacement_agent = FakeAgent("sess-1")

    async def fake_build_agent(*args, **kwargs):
        return replacement_agent

    monkeypatch.setattr(
        "server.services.runtime.agent_runtime_cache.build_agent",
        fake_build_agent,
    )

    pool = AgentRuntimeCache(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=ConsoleConfig(),
        session_store=store,
    )
    existing_agent = FakeAgent("sess-1")
    pool._cache["sess-1"] = CachedAgent(
        agent=existing_agent,
        config_snapshot=("stale", "", "", "", "", None, None, (), ()),
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    agent = await pool.get_or_create_runtime_agent(session)

    assert agent is existing_agent
    assert replacement_agent.closed is True
    scheduler.rebind_agent.assert_awaited_once_with("sess-1", replacement_agent)


@pytest.mark.asyncio
async def test_agent_runtime_cache_refreshes_when_allowed_skills_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = FakeChannelChatSessionStore()
    first_config = AgentConfigRecord.model_construct(
        id="base-agent",
        name="base",
        model_provider="openai",
        model_name="gpt-test",
        allowed_skills=["alpha"],
    )
    second_config = first_config.model_copy(update={"allowed_skills": ["beta"]})
    registry = SimpleNamespace(
        get_agent=AsyncMock(side_effect=[first_config, second_config])
    )
    scheduler = SimpleNamespace(rebind_agent=AsyncMock(return_value=True))
    first_agent = FakeAgent("sess-1-a")
    second_agent = FakeAgent("sess-1-b")

    monkeypatch.setattr(
        "server.services.runtime.agent_runtime_cache.build_agent",
        AsyncMock(side_effect=[first_agent, second_agent]),
    )

    pool = AgentRuntimeCache(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=ConsoleConfig(),
        session_store=store,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    initial = await pool.get_or_create_runtime_agent(session)
    refreshed = await pool.get_or_create_runtime_agent(session)

    assert initial is first_agent
    assert refreshed is second_agent
    assert first_agent.closed is True
    scheduler.rebind_agent.assert_awaited_once_with("sess-1", second_agent)


@pytest.mark.asyncio
async def test_session_runtime_service_routes_with_session_id_as_root_state() -> None:
    store = FakeChannelChatSessionStore()
    route_result = SimpleNamespace(action="steered", state_id="sess-1", stream=None)
    scheduler = SimpleNamespace(
        route_root_input=AsyncMock(return_value=route_result),
        wait_for=AsyncMock(),
    )
    runtime_service = SessionRuntimeService(
        scheduler=scheduler,
        session_store=store,
        timeout=60,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    dispatch = await runtime_service.execute(FakeAgent("sess-1"), session, "hello")

    assert dispatch.action == "steered"
    scheduler.route_root_input.assert_awaited_once_with(
        "hello",
        agent=ANY,
        state_id="sess-1",
        session_id="sess-1",
        persistent=True,
        timeout=60,
        stream_mode="until_settled",
    )
    assert store.sessions["sess-1"] is session


@pytest.mark.asyncio
async def test_session_runtime_service_cancel_if_active_uses_session_id() -> None:
    active_state = SimpleNamespace(is_active=lambda: True)
    inactive_state = SimpleNamespace(is_active=lambda: False)
    scheduler = SimpleNamespace(
        get_state=AsyncMock(side_effect=[active_state, inactive_state]),
        cancel=AsyncMock(),
    )
    runtime_service = SessionRuntimeService(
        scheduler=scheduler,
        session_store=FakeChannelChatSessionStore(),
    )
    active_session = Session(
        id="sess-active",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    inactive_session = Session(
        id="sess-inactive",
        chat_context_scope_id=None,
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    await runtime_service.cancel_if_active(active_session, "stop")
    await runtime_service.cancel_if_active(inactive_session, "stop")

    scheduler.cancel.assert_awaited_once_with("sess-active", "stop")


@pytest.mark.asyncio
async def test_base_channel_service_skips_delivery_for_stale_session() -> None:
    now = datetime.now(timezone.utc)
    current_session = Session(
        id="sess-current",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )
    stale_session = Session(
        id="sess-stale",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )

    async def _stream():
        yield RunCompletedEvent(
            session_id=stale_session.id,
            run_id="run-1",
            agent_id=stale_session.id,
            parent_run_id=None,
            depth=0,
            response="stale output",
        )

    executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=RouteResult(
                action="submitted",
                state_id=stale_session.id,
                stream=_stream(),
            )
        ),
    )
    session_service = SimpleNamespace(
        get_or_create_current_session=AsyncMock(
            return_value=SimpleNamespace(session=stale_session)
        ),
        get_chat_context_and_current_session=AsyncMock(
            return_value=(SimpleNamespace(id="ctx-1"), current_session)
        ),
    )
    agent_pool = SimpleNamespace(
        get_or_create_runtime_agent=AsyncMock(return_value=FakeAgent(stale_session.id)),
        close=AsyncMock(),
    )
    service = _TestChannelService(
        session_service=session_service,
        agent_pool=agent_pool,
        executor=executor,
    )
    batch_context = BatchContext(
        chat_context_scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        trigger_user_id="user-1",
        trigger_message_id="msg-1",
        base_agent_id="agent-1",
    )

    try:
        await service._execute_batch(
            BatchPayload(
                context=batch_context,
                messages=[],
                user_message="hello",
            )
        )
        assert service.reply_calls == []
    finally:
        await service.close_base()
