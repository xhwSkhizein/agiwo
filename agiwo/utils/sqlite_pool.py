"""
SQLite connection pool — shared connections for all storage implementations.

Provides a singleton connection pool that manages SQLite connections by db_path.
All storage implementations should use this pool instead of creating their own connections.
"""

import asyncio
from pathlib import Path

import aiosqlite

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SQLiteConnectionPool:
    """
    Singleton connection pool for SQLite databases.
    
    Each unique db_path gets one shared connection.
    Thread-safe via asyncio.Lock.
    """

    _instance: "SQLiteConnectionPool | None" = None
    _lock: asyncio.Lock | None = None

    def __new__(cls) -> "SQLiteConnectionPool":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connections: dict[str, aiosqlite.Connection] = {}
            cls._instance._ref_counts: dict[str, int] = {}
        return cls._instance

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        """Get or create the asyncio lock (must be called within event loop)."""
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    async def get_connection(self, db_path: str) -> aiosqlite.Connection:
        """
        Get a shared connection for the given db_path.
        
        Creates a new connection if one doesn't exist.
        Increments reference count for tracking.
        """
        lock = self._get_lock()
        async with lock:
            # Normalize path
            normalized_path = str(Path(db_path).expanduser().resolve())

            if normalized_path not in self._connections:
                # Ensure parent directory exists
                Path(normalized_path).parent.mkdir(parents=True, exist_ok=True)

                # Use daemon worker thread so leaked connections never block process exit.
                conn = aiosqlite.connect(normalized_path)
                conn._thread.daemon = True
                # Create new connection with WAL mode for better concurrency
                conn = await conn
                conn.row_factory = aiosqlite.Row
                await conn.execute("PRAGMA journal_mode=WAL")
                await conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

                self._connections[normalized_path] = conn
                self._ref_counts[normalized_path] = 0

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
            normalized_path = str(Path(db_path).expanduser().resolve())

            if normalized_path not in self._connections:
                return

            self._ref_counts[normalized_path] -= 1

            if self._ref_counts[normalized_path] <= 0:
                conn = self._connections.pop(normalized_path)
                self._ref_counts.pop(normalized_path)
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


async def release_shared_connection(db_path: str) -> None:
    """Convenience function to release a shared connection."""
    await get_sqlite_pool().release_connection(db_path)


async def close_all_connections() -> None:
    """Close all connections in the global pool."""
    await get_sqlite_pool().close_all()


__all__ = [
    "SQLiteConnectionPool",
    "get_sqlite_pool",
    "get_shared_connection",
    "release_shared_connection",
    "close_all_connections",
]
