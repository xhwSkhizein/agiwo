"""SQLite-backed Feishu channel metadata store."""

from datetime import datetime, timezone
import os

import aiosqlite

from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection
from server.channels.models import ChannelChatContext, Session, SessionWithContext
from server.channels.session_binding import SessionMutationPlan

_SESSION_WITH_CONTEXT_SELECT = """
SELECT
    s.id AS s_id,
    s.chat_context_id AS s_chat_context_id,
    s.base_agent_id AS s_base_agent_id,
    s.runtime_agent_id AS s_runtime_agent_id,
    s.scheduler_state_id AS s_scheduler_state_id,
    s.created_by AS s_created_by,
    s.created_at AS s_created_at,
    s.updated_at AS s_updated_at,
    c.id AS c_id,
    c.scope_id AS c_scope_id,
    c.channel_instance_id AS c_channel_instance_id,
    c.chat_id AS c_chat_id,
    c.chat_type AS c_chat_type,
    c.user_open_id AS c_user_open_id,
    c.base_agent_id AS c_base_agent_id,
    c.current_session_id AS c_current_session_id,
    c.created_at AS c_created_at,
    c.updated_at AS c_updated_at
FROM feishu_session s
JOIN feishu_channel_chat_context c
  ON s.chat_context_id = c.id
"""


def _row_to_chat_context(row: aiosqlite.Row) -> ChannelChatContext:
    return ChannelChatContext(
        id=row["id"],
        scope_id=row["scope_id"],
        channel_instance_id=row["channel_instance_id"],
        chat_id=row["chat_id"],
        chat_type=row["chat_type"],
        user_open_id=row["user_open_id"],
        base_agent_id=row["base_agent_id"],
        current_session_id=row["current_session_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_session(row: aiosqlite.Row) -> Session:
    return Session(
        id=row["id"],
        chat_context_id=row["chat_context_id"],
        base_agent_id=row["base_agent_id"],
        runtime_agent_id=row["runtime_agent_id"],
        scheduler_state_id=row["scheduler_state_id"],
        created_by=row["created_by"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _joined_row_to_session_with_context(row: aiosqlite.Row) -> SessionWithContext:
    return SessionWithContext(
        session=Session(
            id=row["s_id"],
            chat_context_id=row["s_chat_context_id"],
            base_agent_id=row["s_base_agent_id"],
            runtime_agent_id=row["s_runtime_agent_id"],
            scheduler_state_id=row["s_scheduler_state_id"],
            created_by=row["s_created_by"],
            created_at=datetime.fromisoformat(row["s_created_at"]),
            updated_at=datetime.fromisoformat(row["s_updated_at"]),
        ),
        chat_context=ChannelChatContext(
            id=row["c_id"],
            scope_id=row["c_scope_id"],
            channel_instance_id=row["c_channel_instance_id"],
            chat_id=row["c_chat_id"],
            chat_type=row["c_chat_type"],
            user_open_id=row["c_user_open_id"],
            base_agent_id=row["c_base_agent_id"],
            current_session_id=row["c_current_session_id"],
            created_at=datetime.fromisoformat(row["c_created_at"]),
            updated_at=datetime.fromisoformat(row["c_updated_at"]),
        ),
    )


class SqliteFeishuChannelStore:
    def __init__(self, db_path: str) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        if self._conn is not None:
            return
        self._conn = await get_shared_connection(self._db_path)
        await self._create_tables()

    async def close(self) -> None:
        if self._conn is None:
            return
        await release_shared_connection(self._db_path)
        self._conn = None

    async def claim_event(self, channel_instance_id: str, event_id: str) -> bool:
        conn = await self._require_conn()
        cursor = await conn.execute(
            """
            INSERT OR IGNORE INTO feishu_event_dedup (
                channel_instance_id,
                event_id,
                created_at
            ) VALUES (?, ?, ?)
            """,
            (
                channel_instance_id,
                event_id,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await conn.commit()
        return cursor.rowcount == 1

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_channel_chat_context
            WHERE scope_id = ?
            """,
            (scope_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_chat_context(row)

    async def get_chat_context_by_id(
        self,
        chat_context_id: str,
    ) -> ChannelChatContext | None:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_channel_chat_context
            WHERE id = ?
            """,
            (chat_context_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_chat_context(row)

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        conn = await self._require_conn()
        await self._upsert_chat_context_record(conn, chat_context)
        await conn.commit()

    async def get_session(self, session_id: str) -> Session | None:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session
            WHERE id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_session(row)

    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None:
        conn = await self._require_conn()
        async with conn.execute(
            f"""
            {_SESSION_WITH_CONTEXT_SELECT}
            WHERE s.id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return _joined_row_to_session_with_context(row)

    async def upsert_session(self, session: Session) -> None:
        conn = await self._require_conn()
        await self._upsert_session_record(conn, session)
        await conn.commit()

    async def apply_session_mutation(self, mutation: SessionMutationPlan) -> None:
        conn = await self._require_conn()
        await conn.execute("BEGIN")
        try:
            await self._upsert_chat_context_record(conn, mutation.chat_context)
            await self._upsert_session_record(conn, mutation.current_session)
        except Exception:
            await conn.rollback()
            raise
        await conn.commit()

    async def list_sessions_by_user(self, user_open_id: str) -> list[SessionWithContext]:
        conn = await self._require_conn()
        async with conn.execute(
            f"""
            {_SESSION_WITH_CONTEXT_SELECT}
            WHERE c.user_open_id = ?
            ORDER BY s.updated_at DESC
            """,
            (user_open_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_joined_row_to_session_with_context(row) for row in rows]

    async def list_sessions_by_chat_context(self, chat_context_id: str) -> list[Session]:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session
            WHERE chat_context_id = ?
            ORDER BY updated_at DESC
            """,
            (chat_context_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_session(row) for row in rows]

    async def _create_tables(self) -> None:
        conn = await self._require_conn()
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feishu_event_dedup (
                channel_instance_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (channel_instance_id, event_id)
            )
            """
        )
        await conn.execute("DROP TABLE IF EXISTS feishu_session_runtime")
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feishu_channel_chat_context (
                id TEXT PRIMARY KEY,
                scope_id TEXT NOT NULL UNIQUE,
                channel_instance_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                user_open_id TEXT NOT NULL,
                base_agent_id TEXT NOT NULL,
                current_session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feishu_session (
                id TEXT PRIMARY KEY,
                chat_context_id TEXT NOT NULL,
                base_agent_id TEXT NOT NULL,
                runtime_agent_id TEXT NOT NULL,
                scheduler_state_id TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_chat_context_user_open_id "
            "ON feishu_channel_chat_context(user_open_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_session_chat_context_id "
            "ON feishu_session(chat_context_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_session_updated_at "
            "ON feishu_session(updated_at)"
        )
        await conn.commit()

    async def _upsert_chat_context_record(
        self,
        conn: aiosqlite.Connection,
        chat_context: ChannelChatContext,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO feishu_channel_chat_context (
                id,
                scope_id,
                channel_instance_id,
                chat_id,
                chat_type,
                user_open_id,
                base_agent_id,
                current_session_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                id = excluded.id,
                channel_instance_id = excluded.channel_instance_id,
                chat_id = excluded.chat_id,
                chat_type = excluded.chat_type,
                user_open_id = excluded.user_open_id,
                base_agent_id = excluded.base_agent_id,
                current_session_id = excluded.current_session_id,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                chat_context.id,
                chat_context.scope_id,
                chat_context.channel_instance_id,
                chat_context.chat_id,
                chat_context.chat_type,
                chat_context.user_open_id,
                chat_context.base_agent_id,
                chat_context.current_session_id,
                chat_context.created_at.isoformat(),
                chat_context.updated_at.isoformat(),
            ),
        )

    async def _upsert_session_record(
        self,
        conn: aiosqlite.Connection,
        session: Session,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO feishu_session (
                id,
                chat_context_id,
                base_agent_id,
                runtime_agent_id,
                scheduler_state_id,
                created_by,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                chat_context_id = excluded.chat_context_id,
                base_agent_id = excluded.base_agent_id,
                runtime_agent_id = excluded.runtime_agent_id,
                scheduler_state_id = excluded.scheduler_state_id,
                created_by = excluded.created_by,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                session.id,
                session.chat_context_id,
                session.base_agent_id,
                session.runtime_agent_id,
                session.scheduler_state_id,
                session.created_by,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
            ),
        )

    async def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.connect()
        assert self._conn is not None
        return self._conn


__all__ = ["SqliteFeishuChannelStore"]
