"""Shared SQLite runtime helpers for storage implementations."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

import aiosqlite

from agiwo.utils.logging import FilteringBoundLogger, get_logger
from agiwo.utils.sqlite_pool import (
    get_shared_connection,
    get_shared_write_lock,
    release_shared_connection,
)

SQLiteInitializer = Callable[[aiosqlite.Connection], Awaitable[None]]

_tx_logger = get_logger(__name__)


def dumps_json_object(payload: dict[str, Any]) -> str:
    """Serialize a dict for SQLite JSON columns (schemas differ per store)."""
    return json.dumps(payload, ensure_ascii=False, default=str)


def loads_json_object(raw: str) -> dict[str, Any]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("expected JSON object payload")
    return data


class SQLiteConnectionRuntime:
    """Own the shared SQLite connection lifecycle for one store instance."""

    def __init__(
        self,
        db_path: str,
        *,
        logger: FilteringBoundLogger,
        connect_event: str,
        disconnect_event: str | None = None,
    ) -> None:
        self.db_path = db_path
        self._logger = logger
        self._connect_event = connect_event
        self._disconnect_event = disconnect_event
        self._connection: aiosqlite.Connection | None = None
        self._initialized = False

    @property
    def connection(self) -> aiosqlite.Connection | None:
        return self._connection

    @property
    def initialized(self) -> bool:
        return self._initialized

    async def ensure_connection(
        self,
        initializer: SQLiteInitializer,
    ) -> aiosqlite.Connection:
        if self._initialized and self._connection is not None:
            return self._connection

        if self._connection is None:
            self._connection = await get_shared_connection(self.db_path)

        await initializer(self._connection)
        self._initialized = True
        self._logger.info(self._connect_event, db_path=self.db_path)
        return self._connection

    async def disconnect(self) -> None:
        if self._connection is None:
            return

        await release_shared_connection(self.db_path)
        self._connection = None
        self._initialized = False
        if self._disconnect_event is not None:
            self._logger.info(self._disconnect_event, db_path=self.db_path)


async def execute_statements(
    connection: aiosqlite.Connection,
    statements: Sequence[str],
) -> None:
    """Execute a sequence of schema statements in order."""
    for statement in statements:
        await connection.execute(statement)


async def get_table_columns(
    connection: aiosqlite.Connection,
    table_name: str,
) -> set[str]:
    """Return the column names currently present on a SQLite table."""
    async with connection.execute(f"PRAGMA table_info({table_name})") as cursor:
        rows = await cursor.fetchall()
    return {row[1] for row in rows}


def _is_nested_transaction_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "within a transaction" in message or "cannot start a transaction" in message


async def safe_rollback(connection: aiosqlite.Connection) -> None:
    """Rollback only when SQLite still has an open transaction."""
    if connection.in_transaction:
        await connection.rollback()


async def begin_immediate(connection: aiosqlite.Connection) -> None:
    """Start an exclusive transaction; clear any stale open transaction first.

    aiosqlite's ``in_transaction`` can lag the real SQLite state on a shared
    autocommit connection, so recover from nested-BEGIN errors explicitly.
    """
    if connection.in_transaction:
        _tx_logger.warning("sqlite_stale_transaction_rollback")
        await safe_rollback(connection)
    try:
        await connection.execute("BEGIN IMMEDIATE")
    except Exception as exc:
        if not _is_nested_transaction_error(exc):
            raise
        _tx_logger.warning(
            "sqlite_nested_transaction_recovered",
            error=str(exc),
        )
        await safe_rollback(connection)
        await connection.execute("BEGIN IMMEDIATE")


@asynccontextmanager
async def exclusive_sqlite_access(
    db_path: str,
) -> AsyncIterator[None]:
    """Serialize writers that share one pooled connection for ``db_path``."""
    async with get_shared_write_lock(db_path):
        yield


@asynccontextmanager
async def immediate_transaction(
    connection: aiosqlite.Connection,
    db_path: str,
) -> AsyncIterator[aiosqlite.Connection]:
    """Exclusive BEGIN IMMEDIATE … COMMIT/ROLLBACK on a shared connection."""
    async with exclusive_sqlite_access(db_path):
        await begin_immediate(connection)
        try:
            yield connection
            if connection.in_transaction:
                await connection.commit()
        except BaseException:
            await safe_rollback(connection)
            raise


__all__ = [
    "SQLiteConnectionRuntime",
    "begin_immediate",
    "dumps_json_object",
    "exclusive_sqlite_access",
    "execute_statements",
    "get_table_columns",
    "immediate_transaction",
    "loads_json_object",
    "safe_rollback",
]
