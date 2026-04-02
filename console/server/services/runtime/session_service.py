"""Session and channel chat-context coordination."""

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from agiwo.utils.logging import get_logger

from server.channels.exceptions import (
    BaseAgentNotFoundError,
    ChatContextNotFoundError,
    SessionNotFoundError,
    SessionNotInChatContextError,
)
from server.models.session import (
    BatchContext,
    ChannelChatContext,
    ChannelChatSessionStore,
    Session,
    SessionCreateResult,
    SessionSwitchResult,
    UserSessionItem,
)
from server.services.agent_registry import AgentConfigRecord, AgentRegistry

logger = get_logger(__name__)


@dataclass(slots=True)
class SessionContextResolution:
    chat_context: ChannelChatContext
    session: Session


class SessionContextService:
    """Channel-facing session service.

    Feishu and other channels keep a server-side current session pointer.
    Console web should not use ``switch_session`` and instead work with explicit
    ``session_id`` values in the URL and request payloads.
    """

    def __init__(
        self,
        *,
        store: ChannelChatSessionStore,
        agent_registry: AgentRegistry,
        default_agent_name: str,
    ) -> None:
        self._store = store
        self._agent_registry = agent_registry
        self._default_agent_name = default_agent_name
        self._default_agent_resolved = False
        self._default_agent_cache: AgentConfigRecord | None = None

    async def resolve_default_agent_config(self) -> AgentConfigRecord | None:
        if not self._default_agent_name:
            return None
        if self._default_agent_resolved:
            return self._default_agent_cache
        config = await self._agent_registry.get_agent_by_name(self._default_agent_name)
        self._default_agent_cache = config
        self._default_agent_resolved = True
        return config

    async def get_chat_context(
        self,
        chat_context_scope_id: str,
    ) -> ChannelChatContext | None:
        return await self._store.get_chat_context(chat_context_scope_id)

    async def get_chat_context_and_current_session(
        self,
        chat_context_scope_id: str,
    ) -> tuple[ChannelChatContext | None, Session | None]:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            return None, None
        session = await self._store.get_session(chat_context.current_session_id)
        return chat_context, session

    async def get_or_create_current_session(
        self,
        context: BatchContext,
    ) -> SessionContextResolution:
        chat_context = await self._store.get_chat_context(context.chat_context_scope_id)
        if chat_context is None:
            created = await self._create_chat_context_with_session(
                chat_context_scope_id=context.chat_context_scope_id,
                channel_instance_id=context.channel_instance_id,
                chat_id=context.chat_id,
                chat_type=context.chat_type,
                user_open_id=context.trigger_user_id,
                base_agent_id=context.base_agent_id,
                created_by="AUTO",
            )
            return SessionContextResolution(
                chat_context=created.chat_context,
                session=created.session,
            )

        if chat_context.base_agent_id != context.base_agent_id:
            chat_context.base_agent_id = context.base_agent_id
            chat_context.updated_at = datetime.now(timezone.utc)
            await self._store.upsert_chat_context(chat_context)

        session = await self._store.get_session(chat_context.current_session_id)
        if session is None:
            created = await self._create_session_for_chat_context(
                chat_context,
                base_agent_id=context.base_agent_id,
                created_by="AUTO_RECOVER",
            )
            return SessionContextResolution(
                chat_context=created.chat_context,
                session=created.session,
            )

        return await self._ensure_session_base_agent(session, chat_context)

    async def create_new_session(
        self,
        *,
        chat_context_scope_id: str,
        channel_instance_id: str,
        chat_id: str,
        chat_type: str,
        user_open_id: str,
        base_agent_id: str,
        created_by: str,
    ) -> SessionCreateResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            return await self._create_chat_context_with_session(
                chat_context_scope_id=chat_context_scope_id,
                channel_instance_id=channel_instance_id,
                chat_id=chat_id,
                chat_type=chat_type,
                user_open_id=user_open_id,
                base_agent_id=base_agent_id,
                created_by=created_by,
            )

        return await self._create_session_for_chat_context(
            chat_context,
            base_agent_id=base_agent_id,
            created_by=created_by,
        )

    async def create_standalone_session(
        self,
        *,
        base_agent_id: str,
        created_by: str,
    ) -> Session:
        now = datetime.now(timezone.utc)
        session = Session(
            id=str(uuid4()),
            chat_context_scope_id=None,
            base_agent_id=base_agent_id,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        await self._store.upsert_session(session)
        return session

    async def switch_session(
        self,
        *,
        chat_context_scope_id: str,
        target_session_id: str,
    ) -> SessionSwitchResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise ChatContextNotFoundError(chat_context_scope_id)

        target = await self._store.get_session(target_session_id)
        if target is None:
            raise SessionNotFoundError(target_session_id)
        if target.chat_context_scope_id != chat_context.scope_id:
            raise SessionNotInChatContextError(target_session_id, chat_context.scope_id)

        previous = await self._store.get_session(chat_context.current_session_id)
        now = datetime.now(timezone.utc)
        chat_context.current_session_id = target.id
        chat_context.updated_at = now
        if previous is not None:
            previous.updated_at = now
        target.updated_at = now
        await self._store.upsert_chat_context(chat_context)
        if previous is not None:
            await self._store.upsert_session(previous)
        await self._store.upsert_session(target)
        return SessionSwitchResult(
            previous_session=previous,
            current_session=target,
            chat_context=chat_context,
        )

    async def list_user_sessions(
        self,
        *,
        user_open_id: str,
        current_chat_context_scope_id: str | None,
    ) -> list[UserSessionItem]:
        current_chat_context = None
        if current_chat_context_scope_id:
            current_chat_context = await self._store.get_chat_context(
                current_chat_context_scope_id
            )
        records = await self._store.list_sessions_by_user(user_open_id)
        items: list[UserSessionItem] = []
        for record in records:
            chat_context = record.chat_context
            session = record.session
            items.append(
                UserSessionItem(
                    session=session,
                    chat_context=chat_context,
                    is_current=session.id == chat_context.current_session_id,
                    in_current_context=(
                        current_chat_context is not None
                        and chat_context.scope_id == current_chat_context.scope_id
                    ),
                )
            )
        items.sort(key=lambda item: item.session.updated_at, reverse=True)
        return items

    async def fork_session(
        self,
        *,
        chat_context_scope_id: str,
        context_summary: str,
        created_by: str,
    ) -> SessionCreateResult:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            raise ChatContextNotFoundError(chat_context_scope_id)
        source = await self._store.get_session(chat_context.current_session_id)
        if source is None:
            raise SessionNotFoundError(chat_context.current_session_id)

        session = await self._build_forked_session(
            source=source,
            created_by=created_by,
            context_summary=context_summary,
            chat_context_scope_id=chat_context.scope_id,
        )
        chat_context.current_session_id = session.id
        chat_context.updated_at = session.updated_at
        await self._store.upsert_chat_context(chat_context)
        await self._store.upsert_session(session)
        return SessionCreateResult(chat_context=chat_context, session=session)

    async def fork_session_by_id(
        self,
        *,
        session_id: str,
        context_summary: str,
        created_by: str,
        update_chat_context: bool = True,
    ) -> SessionCreateResult:
        source = await self._store.get_session(session_id)
        if source is None:
            raise SessionNotFoundError(session_id)

        chat_context = None
        if source.chat_context_scope_id is not None:
            chat_context = await self._store.get_chat_context(
                source.chat_context_scope_id
            )
            if chat_context is None and update_chat_context:
                raise ChatContextNotFoundError(source.chat_context_scope_id)

        session = await self._build_forked_session(
            source=source,
            created_by=created_by,
            context_summary=context_summary,
            chat_context_scope_id=source.chat_context_scope_id
            if update_chat_context
            else None,
        )
        if update_chat_context and chat_context is not None:
            chat_context.current_session_id = session.id
            chat_context.updated_at = session.updated_at
            await self._store.upsert_chat_context(chat_context)
        await self._store.upsert_session(session)
        return SessionCreateResult(chat_context=chat_context, session=session)

    async def list_sessions(
        self,
        *,
        chat_context_scope_id: str,
    ) -> list[Session]:
        chat_context = await self._store.get_chat_context(chat_context_scope_id)
        if chat_context is None:
            return []
        return await self._store.list_sessions_by_chat_context(chat_context.scope_id)

    async def _create_chat_context_with_session(
        self,
        *,
        chat_context_scope_id: str,
        channel_instance_id: str,
        chat_id: str,
        chat_type: str,
        user_open_id: str,
        base_agent_id: str,
        created_by: str,
    ) -> SessionCreateResult:
        now = datetime.now(timezone.utc)
        session_id = str(uuid4())
        chat_context = ChannelChatContext(
            scope_id=chat_context_scope_id,
            channel_instance_id=channel_instance_id,
            chat_id=chat_id,
            chat_type=chat_type,
            user_open_id=user_open_id,
            base_agent_id=base_agent_id,
            current_session_id=session_id,
            created_at=now,
            updated_at=now,
        )
        session = Session(
            id=session_id,
            chat_context_scope_id=chat_context_scope_id,
            base_agent_id=base_agent_id,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        await self._store.upsert_chat_context(chat_context)
        await self._store.upsert_session(session)
        return SessionCreateResult(chat_context=chat_context, session=session)

    async def _create_session_for_chat_context(
        self,
        chat_context: ChannelChatContext,
        *,
        base_agent_id: str,
        created_by: str,
    ) -> SessionCreateResult:
        now = datetime.now(timezone.utc)
        session = Session(
            id=str(uuid4()),
            chat_context_scope_id=chat_context.scope_id,
            base_agent_id=base_agent_id,
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        chat_context.current_session_id = session.id
        chat_context.base_agent_id = base_agent_id
        chat_context.updated_at = now
        await self._store.upsert_chat_context(chat_context)
        await self._store.upsert_session(session)
        return SessionCreateResult(chat_context=chat_context, session=session)

    async def _ensure_session_base_agent(
        self,
        session: Session,
        chat_context: ChannelChatContext,
    ) -> SessionContextResolution:
        base_config = await self._agent_registry.get_agent(session.base_agent_id)
        if base_config is not None:
            return SessionContextResolution(chat_context=chat_context, session=session)

        default_config = await self.resolve_default_agent_config()
        if default_config is None:
            raise BaseAgentNotFoundError(session.base_agent_id)

        old_base_agent_id = session.base_agent_id
        now = datetime.now(timezone.utc)
        session.base_agent_id = default_config.id
        session.updated_at = now
        chat_context.base_agent_id = default_config.id
        chat_context.updated_at = now
        await self._store.upsert_chat_context(chat_context)
        await self._store.upsert_session(session)
        logger.warning(
            "channel_session_rebind_base_agent",
            session_id=session.id,
            old_base_agent_id=old_base_agent_id,
            new_base_agent_id=session.base_agent_id,
        )
        return SessionContextResolution(chat_context=chat_context, session=session)

    async def _build_forked_session(
        self,
        *,
        source: Session,
        created_by: str,
        context_summary: str,
        chat_context_scope_id: str | None,
    ) -> Session:
        now = datetime.now(timezone.utc)
        return Session(
            id=str(uuid4()),
            chat_context_scope_id=chat_context_scope_id,
            base_agent_id=source.base_agent_id,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            source_session_id=source.id,
            fork_context_summary=context_summary,
        )
