from datetime import datetime, timezone
import os

import aiosqlite
from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection

from server.channels.models import SessionRuntime


class FeishuChannelStore:
    def __init__(self, db_path: str, use_persistent_store: bool) -> None:
        self._db_path = os.path.expanduser(db_path)
        self._use_persistent_store = use_persistent_store
        self._conn: aiosqlite.Connection | None = None

        self._event_dedup: set[str] = set()
        self._session_runtime_map: dict[str, SessionRuntime] = {}

    async def connect(self) -> None:
        if not self._use_persistent_store:
            return
        if self._conn is not None:
            return

        self._conn = await get_shared_connection(self._db_path)
        await self._create_tables()

    async def close(self) -> None:
        if self._conn is not None:
            await release_shared_connection(self._db_path)
            self._conn = None

    async def claim_event(self, channel_instance_id: str, event_id: str) -> bool:
        dedup_key = f"{channel_instance_id}:{event_id}"
        if not self._use_persistent_store:
            if dedup_key in self._event_dedup:
                return False
            self._event_dedup.add(dedup_key)
            return True

        conn = await self._get_conn()
        assert conn is not None
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

    async def get_session_runtime(self, session_key: str) -> SessionRuntime | None:
        if not self._use_persistent_store:
            return self._session_runtime_map.get(session_key)

        conn = await self._get_conn()
        assert conn is not None
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session_runtime
            WHERE session_key = ?
            """,
            (session_key,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            return None

        return SessionRuntime(
            session_key=row["session_key"],
            agiwo_session_id=row["agiwo_session_id"],
            runtime_agent_id=row["runtime_agent_id"],
            scheduler_state_id=row["scheduler_state_id"],
            base_agent_id=row["base_agent_id"],
            chat_id=row["chat_id"],
            chat_type=row["chat_type"],
            trigger_user_id=row["trigger_user_id"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    async def list_session_runtimes_by_user(
        self, trigger_user_id: str
    ) -> list[SessionRuntime]:
        if not self._use_persistent_store:
            return [
                r
                for r in self._session_runtime_map.values()
                if r.trigger_user_id == trigger_user_id
            ]

        conn = await self._get_conn()
        assert conn is not None
        async with conn.execute(
            """
            SELECT *
            FROM feishu_session_runtime
            WHERE trigger_user_id = ?
            ORDER BY updated_at DESC
            """,
            (trigger_user_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            SessionRuntime(
                session_key=row["session_key"],
                agiwo_session_id=row["agiwo_session_id"],
                runtime_agent_id=row["runtime_agent_id"],
                scheduler_state_id=row["scheduler_state_id"],
                base_agent_id=row["base_agent_id"],
                chat_id=row["chat_id"],
                chat_type=row["chat_type"],
                trigger_user_id=row["trigger_user_id"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def upsert_session_runtime(self, runtime: SessionRuntime) -> None:
        if not self._use_persistent_store:
            self._session_runtime_map[runtime.session_key] = runtime
            return

        conn = await self._get_conn()
        assert conn is not None
        await conn.execute(
            """
            INSERT INTO feishu_session_runtime (
                session_key,
                agiwo_session_id,
                runtime_agent_id,
                scheduler_state_id,
                base_agent_id,
                chat_id,
                chat_type,
                trigger_user_id,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_key) DO UPDATE SET
                agiwo_session_id = excluded.agiwo_session_id,
                runtime_agent_id = excluded.runtime_agent_id,
                scheduler_state_id = excluded.scheduler_state_id,
                base_agent_id = excluded.base_agent_id,
                chat_id = excluded.chat_id,
                chat_type = excluded.chat_type,
                trigger_user_id = excluded.trigger_user_id,
                updated_at = excluded.updated_at
            """,
            (
                runtime.session_key,
                runtime.agiwo_session_id,
                runtime.runtime_agent_id,
                runtime.scheduler_state_id,
                runtime.base_agent_id,
                runtime.chat_id,
                runtime.chat_type,
                runtime.trigger_user_id,
                runtime.updated_at.isoformat(),
            ),
        )
        await conn.commit()

    async def _create_tables(self) -> None:
        conn = await self._get_conn()
        assert conn is not None

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
            CREATE TABLE IF NOT EXISTS feishu_session_runtime (
                session_key TEXT PRIMARY KEY,
                agiwo_session_id TEXT NOT NULL,
                runtime_agent_id TEXT NOT NULL,
                scheduler_state_id TEXT NOT NULL,
                base_agent_id TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_type TEXT NOT NULL,
                trigger_user_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        await conn.commit()

    async def _get_conn(self) -> aiosqlite.Connection | None:
        if not self._use_persistent_store:
            return None
        if self._conn is None:
            await self.connect()
        return self._conn
