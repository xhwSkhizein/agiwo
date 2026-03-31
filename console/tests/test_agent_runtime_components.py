from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import ANY, AsyncMock

import pytest

from agiwo.scheduler.commands import RouteResult
from agiwo.agent import RunCompletedEvent
from server.channels.agent_executor import AgentExecutor
from server.channels.utils import (
    extract_stream_text,
    safe_close_all,
    split_text_into_chunks,
)
from server.channels.runtime_agent_pool import RuntimeAgentPool, _CachedAgent
from server.channels.session import SessionContextService, SessionManager
from server.channels.session.models import (
    BatchContext,
    BatchPayload,
    ChannelChatContext,
    Session,
    SessionWithContext,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentConfigRecord


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
        if session is None:
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


class FakeAgent:
    def __init__(self, agent_id: str) -> None:
        self.id = agent_id
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _TestChannelService:
    """Lightweight test stub replicating the channel message pipeline."""

    def __init__(self, *, session_service, agent_pool, executor) -> None:
        self.reply_calls: list[tuple[BatchContext, str]] = []
        self.message_calls: list[tuple[BatchContext, str]] = []
        self._session_service = session_service
        self._agent_pool = agent_pool
        self._executor = executor
        self._session_mgr = SessionManager(
            on_batch_ready=lambda *a: None,
            debounce_ms=1,
            max_batch_window_ms=1,
        )

    async def close_base(self) -> None:
        await safe_close_all(self._session_mgr, self._agent_pool)

    async def _deliver_reply(self, context: BatchContext, text: str) -> None:
        self.reply_calls.append((context, text))

    async def _deliver_message(self, context: BatchContext, text: str) -> None:
        self.message_calls.append((context, text))

    async def _execute_batch(self, batch: BatchPayload) -> None:
        session, agent = await self._prepare_batch_runtime(batch)
        dispatch = await self._executor.execute(agent, session, batch.user_message)

        had_output = await self._consume_dispatch_stream(
            batch, session, dispatch.stream
        )
        if had_output:
            return

        if not await self._can_deliver_target(batch.context, session):
            return

        if dispatch.stream is None:
            await self._deliver_reply(batch.context, "消息已收到，正在继续处理。")
            return

        state = await self._executor.get_state(session.scheduler_state_id)
        if state is not None and state.result_summary:
            await self._deliver_stream_text(
                batch.context, state.result_summary, had_output=False
            )
            return
        await self._deliver_reply(batch.context, "执行完成，但未产出可展示内容。")

    async def _prepare_batch_runtime(self, batch):
        resolution = await self._session_service.get_or_create_current_session(
            batch.context
        )
        if resolution.retired_runtime_agent_id is not None:
            await self._agent_pool.close_runtime_agent(
                resolution.retired_runtime_agent_id
            )
        session = resolution.session
        agent = await self._agent_pool.get_or_create_runtime_agent(session)
        return session, agent

    async def _consume_dispatch_stream(self, batch, session, stream):
        if stream is None:
            return False
        had_output = False
        async for item in stream:
            if not await self._can_deliver_target(batch.context, session):
                continue
            text = extract_stream_text(item)
            if text is None:
                continue
            had_output = await self._deliver_stream_text(
                batch.context, text, had_output=had_output
            )
        return had_output

    async def _deliver_stream_text(self, context, text, *, had_output):
        chunks = split_text_into_chunks(text)
        for index, chunk in enumerate(chunks):
            if not had_output and index == 0:
                await self._deliver_reply(context, chunk)
                had_output = True
                continue
            await self._deliver_message(context, chunk)
            had_output = True
        return had_output

    async def _can_deliver_target(self, context, session):
        return await self._can_deliver_session(
            context, session_id=session.id, state_id=session.scheduler_state_id
        )

    async def _can_deliver_session(self, context, *, session_id, state_id):
        (
            _,
            current_session,
        ) = await self._session_service.get_chat_context_and_current_session(
            context.chat_context_scope_id
        )
        if current_session is None:
            return False
        return (
            current_session.id == session_id
            and current_session.scheduler_state_id == state_id
        )


class _SteeredIntoRunningExecutor:
    def __init__(self) -> None:
        pass

    async def execute(self, agent, session, user_input):
        del agent, session, user_input
        return RouteResult(action="steered", state_id="runtime-1")

    async def get_state(self, state_id: str | None):
        del state_id
        return None


@pytest.mark.asyncio
async def test_session_service_returns_retired_agent_id_on_rebind() -> None:
    store = FakeChannelChatSessionStore()
    chat_context = ChannelChatContext(
        scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="old-context-base",
        current_session_id="sess-1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="missing-base",
        runtime_agent_id="runtime-old",
        scheduler_state_id="runtime-old",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
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
    assert result.retired_runtime_agent_id == "runtime-old"
    assert result.chat_context.base_agent_id == "agent-default"


@pytest.mark.asyncio
async def test_session_service_create_new_session_honors_explicit_base_agent() -> None:
    store = FakeChannelChatSessionStore()
    now = datetime.now(timezone.utc)
    existing_context = ChannelChatContext(
        scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="old-base",
        current_session_id="sess-old",
        created_at=now,
        updated_at=now,
    )
    await store.upsert_chat_context(existing_context)

    service = SessionContextService(
        store=store,
        agent_registry=SimpleNamespace(),
        default_agent_name="default",
    )

    created = await service.create_new_session(
        chat_context_scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type="p2p",
        user_open_id="user-1",
        base_agent_id="new-base",
        created_by="COMMAND_NEW",
    )

    assert created.chat_context.base_agent_id == "new-base"
    assert created.session.base_agent_id == "new-base"
    assert created.chat_context.current_session_id == created.session.id
    assert store.upserted_sessions[-1] is created.session


@pytest.mark.asyncio
async def test_runtime_agent_pool_assigns_runtime_identity_and_persists_session(
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
    built_agent = FakeAgent("generated-agent")

    async def fake_build_agent(*args, **kwargs):
        return built_agent

    monkeypatch.setattr(
        "server.channels.runtime_agent_pool.build_agent",
        fake_build_agent,
    )

    pool = RuntimeAgentPool(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=ConsoleConfig(),
        store=store,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="",
        scheduler_state_id="",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    agent = await pool.get_or_create_runtime_agent(session)

    assert agent is built_agent
    assert session.runtime_agent_id == "generated-agent"
    assert session.scheduler_state_id == "generated-agent"
    assert store.sessions["sess-1"] is session
    assert pool.runtime_agents["generated-agent"] is built_agent


@pytest.mark.asyncio
async def test_runtime_agent_pool_defers_refresh_while_state_running(
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
    replacement_agent = FakeAgent("runtime-1")

    async def fake_build_agent(*args, **kwargs):
        return replacement_agent

    monkeypatch.setattr(
        "server.channels.runtime_agent_pool.build_agent",
        fake_build_agent,
    )

    pool = RuntimeAgentPool(
        scheduler=scheduler,
        agent_registry=registry,
        console_config=ConsoleConfig(),
        store=store,
    )
    existing_agent = FakeAgent("runtime-1")
    pool._cache["runtime-1"] = _CachedAgent(
        agent=existing_agent, config_fingerprint="stale-fingerprint"
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="runtime-1",
        scheduler_state_id="runtime-1",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    agent = await pool.get_or_create_runtime_agent(session)

    assert agent is existing_agent
    assert existing_agent.closed is False
    assert replacement_agent.closed is True
    scheduler.rebind_agent.assert_awaited_once()


@pytest.mark.asyncio
async def test_agent_executor_steers_running_state_and_returns_steered_dispatch() -> (
    None
):
    store = FakeChannelChatSessionStore()
    route_result = SimpleNamespace(action="steered", state_id="runtime-1", stream=None)
    scheduler = SimpleNamespace(
        route_root_input=AsyncMock(return_value=route_result),
        wait_for=AsyncMock(),
    )
    executor = AgentExecutor(
        scheduler=scheduler,
        store=store,
        timeout=60,
    )
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="runtime-1",
        scheduler_state_id="runtime-1",
        created_by="AUTO",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    dispatch = await executor.execute(FakeAgent("runtime-1"), session, "hello")

    assert dispatch.action == "steered"
    assert dispatch.stream is None
    scheduler.route_root_input.assert_awaited_once_with(
        "hello",
        agent=ANY,
        state_id="runtime-1",
        session_id="sess-1",
        persistent=True,
        timeout=60,
    )
    assert store.sessions["sess-1"] is session


@pytest.mark.asyncio
async def test_base_channel_service_acks_steered_into_running() -> None:
    now = datetime.now(timezone.utc)
    session = Session(
        id="sess-1",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="runtime-1",
        scheduler_state_id="runtime-1",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )
    executor = _SteeredIntoRunningExecutor()
    session_service = SimpleNamespace(
        get_or_create_current_session=AsyncMock(
            return_value=SimpleNamespace(
                session=session,
                retired_runtime_agent_id=None,
            )
        ),
        get_chat_context_and_current_session=AsyncMock(
            return_value=(SimpleNamespace(id="ctx-1"), session)
        ),
    )
    agent_pool = SimpleNamespace(
        get_or_create_runtime_agent=AsyncMock(return_value=FakeAgent("runtime-1")),
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
        assert [text for _, text in service.reply_calls] == [
            "消息已收到，正在继续处理。"
        ]
        assert service.message_calls == []
    finally:
        await service.close_base()


@pytest.mark.asyncio
async def test_base_channel_service_skips_delivery_for_stale_session() -> None:
    now = datetime.now(timezone.utc)
    current_session = Session(
        id="sess-current",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="runtime-current",
        scheduler_state_id="runtime-current",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )
    stale_session = Session(
        id="sess-stale",
        chat_context_scope_id="scope-1",
        base_agent_id="base-agent",
        runtime_agent_id="runtime-stale",
        scheduler_state_id="runtime-stale",
        created_by="AUTO",
        created_at=now,
        updated_at=now,
    )

    async def _stream():
        yield RunCompletedEvent(
            session_id=stale_session.id,
            run_id="run-1",
            agent_id="agent-1",
            parent_run_id=None,
            depth=0,
            response="stale output",
        )

    executor = SimpleNamespace(
        execute=AsyncMock(
            return_value=RouteResult(
                action="submitted",
                state_id="runtime-stale",
                stream=_stream(),
            )
        ),
        get_state=AsyncMock(
            return_value=SimpleNamespace(is_active=lambda: False, result_summary=None)
        ),
        wait_for=AsyncMock(),
    )
    session_service = SimpleNamespace(
        get_or_create_current_session=AsyncMock(
            return_value=SimpleNamespace(
                session=stale_session,
                retired_runtime_agent_id=None,
            )
        ),
        get_chat_context_and_current_session=AsyncMock(
            return_value=(SimpleNamespace(id="ctx-1"), current_session)
        ),
    )
    agent_pool = SimpleNamespace(
        get_or_create_runtime_agent=AsyncMock(return_value=FakeAgent("runtime-stale")),
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
        assert service.message_calls == []
    finally:
        await service.close_base()
