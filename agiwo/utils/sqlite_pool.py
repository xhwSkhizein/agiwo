"""
SQLite connection pool — shared connections for all storage implementations.

Provides a singleton connection pool that manages SQLite connections by db_path.
All storage implementations should use this pool instead of creating their own connections.

Connections use autocommit (isolation_level=None). Multi-statement atomicity must
use explicit BEGIN/COMMIT under the per-db write lock so RunLog /
Scheduler stores cannot nest transactions on the shared connection.
"""

import asyncio
from pathlib import Path

import aiosqlite

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteConnectionPool:
    """Connection pool for SQLite databases.

    Each unique db_path gets one shared connection and one write lock.
    Thread-safe via asyncio.Lock.
    Use the module-level ``get_sqlite_pool()`` factory to obtain the
    global singleton instance.
    """

    def __init__(self) -> None:
        self._connections: dict[str, aiosqlite.Connection] = {}
        self._ref_counts: dict[str, int] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._lock: asyncio.Lock | None = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    def _normalize_path(db_path: str) -> str:
        return str(Path(db_path).expanduser().resolve())

    def write_lock(self, db_path: str) -> asyncio.Lock:
        """Return the per-db lock that must guard all writes / explicit transactions."""
        normalized_path = self._normalize_path(db_path)
        lock = self._write_locks.get(normalized_path)
        if lock is None:
            lock = asyncio.Lock()
            self._write_locks[normalized_path] = lock
        return lock

    async def get_connection(self, db_path: str) -> aiosqlite.Connection:
        """
        Get a shared connection for the given db_path.

        Creates a new connection if one doesn't exist.
        Increments reference count for tracking.
        """
        lock = self._get_lock()
        async with lock:
            # Normalize path
            normalized_path = self._normalize_path(db_path)

            if normalized_path not in self._connections:
                # Ensure parent directory exists
                Path(normalized_path).parent.mkdir(parents=True, exist_ok=True)

                # Use daemon worker thread so leaked connections never block process exit.
                connector = aiosqlite.connect(
                    normalized_path,
                    isolation_level=None,  # autocommit; explicit BEGIN for atomic writes
                )
                connector._thread.daemon = True
                conn = await connector
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

                self._connections[normalized_path] = conn
                self._ref_counts[normalized_path] = 0
                self._write_locks.setdefault(normalized_path, asyncio.Lock())

                logger.info("sqlite_pool_connection_created", db_path=normalized_path)

            self._ref_counts[normalized_path] += 1
            return self._connections[normalized_path]

    async def release_connection(self, db_path: str) -> None:
        """
        Release a connection reference.

        Connection is closed when reference count reaches 0.
        """
        lock = self._get_lock()
        async with lock:
            normalized_path = self._normalize_path(db_path)

            if normalized_path not in self._connections:
                return

            self._ref_counts[normalized_path] -= 1

            if self._ref_counts[normalized_path] <= 0:
                conn = self._connections.pop(normalized_path)
                self._ref_counts.pop(normalized_path)
                self._write_locks.pop(normalized_path, None)
                await conn.close()
                logger.info("sqlite_pool_connection_closed", db_path=normalized_path)

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        lock = self._get_lock()
        async with lock:
            for path, conn in list(self._connections.items()):
                await conn.close()
                logger.info("sqlite_pool_connection_closed", db_path=path)
            self._connections.clear()
            self._ref_counts.clear()
            self._write_locks.clear()

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)


# Global pool instance
_pool: SQLiteConnectionPool | None = None


def get_sqlite_pool() -> SQLiteConnectionPool:
    """Get the global SQLite connection pool."""
    global _pool
    if _pool is None:
        _pool = SQLiteConnectionPool()
    return _pool


async def get_shared_connection(db_path: str) -> aiosqlite.Connection:
    """Convenience function to get a shared connection."""
    return await get_sqlite_pool().get_connection(db_path)


def get_shared_write_lock(db_path: str) -> asyncio.Lock:
    """Per-db lock for writes and explicit transactions on the shared connection."""
    return get_sqlite_pool().write_lock(db_path)


async def release_shared_connection(db_path: str) -> None:
    """Convenience function to release a shared connection."""
    await get_sqlite_pool().release_connection(db_path)


async def close_all_connections() -> None:
    """Close all connections in the global pool."""
    await get_sqlite_pool().close_all()


def reset_sqlite_pool() -> None:
    """Reset the global pool instance (useful for testing)."""
    global _pool
    _pool = None


__all__ = [
    "SQLiteConnectionPool",
    "get_sqlite_pool",
    "get_shared_connection",
    "get_shared_write_lock",
    "release_shared_connection",
    "close_all_connections",
    "reset_sqlite_pool",
]
