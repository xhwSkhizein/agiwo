"""Feishu message deduplication store.

This is a Feishu-specific store for preventing duplicate message processing.
It is separate from the generic SessionStore.
"""

import os
from collections import OrderedDict
from datetime import datetime, timezone

import aiosqlite

from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection

_EVENT_DEDUP_MAX_SIZE = 10_000


class InMemoryFeishuDedupStore:
    """In-memory deduplication store for Feishu messages.

    Suitable for development and testing. Data is lost on restart.
    """

    def __init__(self) -> None:
        self._event_dedup: OrderedDict[str, None] = OrderedDict()

    async def connect(self) -> None:
        """No-op for in-memory store."""
        return None

    async def close(self) -> None:
        """No-op for in-memory store."""
        return None

    async def claim_event(self, channel_instance_id: str, event_id: str) -> bool:
        """Attempt to claim an event for processing.

        Returns True if the event was newly claimed (should process),
        False if it was already claimed (should skip).
        """
        dedup_key = f"{channel_instance_id}:{event_id}"
        if dedup_key in self._event_dedup:
            return False
        self._event_dedup[dedup_key] = None
        while len(self._event_dedup) > _EVENT_DEDUP_MAX_SIZE:
            self._event_dedup.popitem(last=False)
        return True


class SqliteFeishuDedupStore:
    """SQLite-backed deduplication store for Feishu messages.

    Provides persistent deduplication across restarts.
    """

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
        """Attempt to claim an event for processing.

        Returns True if the event was newly claimed (should process),
        False if it was already claimed (should skip).
        """
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
        await conn.commit()

    async def _require_conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            await self.connect()
        assert self._conn is not None
        return self._conn


def create_feishu_dedup_store(
    *,
    db_path: str,
    use_persistent_store: bool,
) -> InMemoryFeishuDedupStore | SqliteFeishuDedupStore:
    """Create a Feishu deduplication store.

    Args:
        db_path: Path to SQLite database (used when use_persistent_store=True)
        use_persistent_store: If True, use SQLite; otherwise use in-memory store
    """
    if use_persistent_store:
        return SqliteFeishuDedupStore(db_path=db_path)
    return InMemoryFeishuDedupStore()
