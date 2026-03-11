"""Shared SQLite runtime helpers for storage implementations."""

from collections.abc import Awaitable, Callable, Mapping, Sequence

import aiosqlite

from agiwo.utils.logging import FilteringBoundLogger
from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection

SQLiteInitializer = Callable[[aiosqlite.Connection], Awaitable[None]]


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


async def ensure_column(
    connection: aiosqlite.Connection,
    table_name: str,
    column_name: str,
    column_def: str,
    *,
    existing: set[str] | None = None,
) -> set[str]:
    """Add a column when it is missing and return the updated column set."""
    existing_columns = (
        await get_table_columns(connection, table_name)
        if existing is None
        else existing
    )
    if column_name in existing_columns:
        return existing_columns

    await connection.execute(
        f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_def}"
    )
    existing_columns.add(column_name)
    return existing_columns


async def ensure_columns(
    connection: aiosqlite.Connection,
    table_name: str,
    columns: Mapping[str, str],
    *,
    existing: set[str] | None = None,
) -> set[str]:
    """Ensure a set of columns exists on a SQLite table."""
    existing_columns = (
        await get_table_columns(connection, table_name)
        if existing is None
        else existing
    )
    for column_name, column_def in columns.items():
        existing_columns = await ensure_column(
            connection,
            table_name,
            column_name,
            column_def,
            existing=existing_columns,
        )
    return existing_columns


__all__ = [
    "SQLiteConnectionRuntime",
    "ensure_column",
    "ensure_columns",
    "execute_statements",
    "get_table_columns",
]
