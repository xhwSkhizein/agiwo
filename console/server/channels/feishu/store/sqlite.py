"""SQLite-backed Feishu/channel metadata store."""

from datetime import datetime, timezone
import os

import aiosqlite

from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection
from server.models.session import (
    ChannelChatContext,
    Session,
    SessionWithContext,
)

_SESSION_WITH_CONTEXT_SELECT = """
SELECT
    s.id AS s_id,
    s.chat_context_scope_id AS s_chat_context_scope_id,
    s.base_agent_id AS s_base_agent_id,
    s.created_by AS s_created_by,
    s.created_at AS s_created_at,
    s.updated_at AS s_updated_at,
    s.source_session_id AS s_source_session_id,
    s.fork_context_summary AS s_fork_context_summary,
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
  ON s.chat_context_scope_id = c.scope_id
"""


def _row_to_chat_context(row: aiosqlite.Row) -> ChannelChatContext:
    return ChannelChatContext(
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
        chat_context_scope_id=row["chat_context_scope_id"],
        base_agent_id=row["base_agent_id"],
        created_by=row["created_by"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        source_session_id=row["source_session_id"],
        fork_context_summary=row["fork_context_summary"],
    )


def _joined_row_to_session_with_context(row: aiosqlite.Row) -> SessionWithContext:
    return SessionWithContext(
        session=Session(
            id=row["s_id"],
            chat_context_scope_id=row["s_chat_context_scope_id"],
            base_agent_id=row["s_base_agent_id"],
            created_by=row["s_created_by"],
            created_at=datetime.fromisoformat(row["s_created_at"]),
            updated_at=datetime.fromisoformat(row["s_updated_at"]),
            source_session_id=row["s_source_session_id"],
            fork_context_summary=row["s_fork_context_summary"],
        ),
        chat_context=ChannelChatContext(
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

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        conn = await self._require_conn()
        await conn.execute(
            """
            INSERT INTO feishu_channel_chat_context (
                scope_id,
                channel_instance_id,
                chat_id,
                chat_type,
                user_open_id,
                base_agent_id,
                current_session_id,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope_id) DO UPDATE SET
                channel_instance_id = excluded.channel_instance_id,
                chat_id = excluded.chat_id,
                chat_type = excluded.chat_type,
                user_open_id = excluded.user_open_id,
                base_agent_id = excluded.base_agent_id,
                current_session_id = excluded.current_session_id,
                updated_at = excluded.updated_at
            """,
            (
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
        await conn.execute(
            """
            INSERT INTO feishu_session (
                id,
                chat_context_scope_id,
                base_agent_id,
                created_by,
                created_at,
                updated_at,
                source_session_id,
                fork_context_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                chat_context_scope_id = excluded.chat_context_scope_id,
                base_agent_id = excluded.base_agent_id,
                created_by = excluded.created_by,
                updated_at = excluded.updated_at,
                source_session_id = excluded.source_session_id,
                fork_context_summary = excluded.fork_context_summary
            """,
            (
                session.id,
                session.chat_context_scope_id,
                session.base_agent_id,
                session.created_by,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                session.source_session_id,
                session.fork_context_summary,
            ),
        )
        await conn.commit()

    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]:
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

    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session
            WHERE chat_context_scope_id = ?
            ORDER BY updated_at DESC
            """,
            (chat_context_scope_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_session(row) for row in rows]

    async def list_sessions_by_base_agent(self, base_agent_id: str) -> list[Session]:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session
            WHERE base_agent_id = ?
            ORDER BY updated_at DESC
            """,
            (base_agent_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_session(row) for row in rows]

    async def list_sessions(self) -> list[Session]:
        conn = await self._require_conn()
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session
            ORDER BY updated_at DESC
            """
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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feishu_channel_chat_context (
                scope_id TEXT PRIMARY KEY,
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
                chat_context_scope_id TEXT,
                base_agent_id TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_session_id TEXT,
                fork_context_summary TEXT
            )
            """
        )
        await self._validate_schema(
            conn,
            "feishu_channel_chat_context",
            {
                "scope_id",
                "channel_instance_id",
                "chat_id",
                "chat_type",
                "user_open_id",
                "base_agent_id",
                "current_session_id",
                "created_at",
                "updated_at",
            },
        )
        await self._validate_schema(
            conn,
            "feishu_session",
            {
                "id",
                "chat_context_scope_id",
                "base_agent_id",
                "created_by",
                "created_at",
                "updated_at",
                "source_session_id",
                "fork_context_summary",
            },
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_chat_context_user_open_id "
            "ON feishu_channel_chat_context(user_open_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_session_chat_context_scope_id "
            "ON feishu_session(chat_context_scope_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_session_base_agent_id "
            "ON feishu_session(base_agent_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_feishu_session_updated_at "
            "ON feishu_session(updated_at)"
        )
        await conn.commit()

    async def _validate_schema(
        self,
        conn: aiosqlite.Connection,
        table_name: str,
        expected_columns: set[str],
    ) -> None:
        async with conn.execute(f"PRAGMA table_info({table_name})") as cursor:
            columns = {row[1] for row in await cursor.fetchall()}
        if columns != expected_columns:
            raise RuntimeError(
                f"{table_name} schema is incompatible with current console version; "
                f"please clear old session metadata before restarting"
            )

    async def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.connect()
        assert self._conn is not None
        return self._conn


__all__ = ["SqliteFeishuChannelStore"]
