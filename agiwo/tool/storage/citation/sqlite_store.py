"""SQLite implementation of CitationSourceRepository."""

import json
from datetime import datetime
from typing import Any

import aiosqlite

from agiwo.utils.storage_support.sqlite_runtime import (
    SQLiteConnectionRuntime,
    execute_statements,
)
from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)
from agiwo.tool.storage.citation.utils import (
    reorder_simplified_sources,
    sort_simplified_sources,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)
_INDEX_COLUMN = "citation_index"


class SQLiteCitationStore:
    """SQLite implementation of Citation Store."""

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = db_path
        self._connection: aiosqlite.Connection | None = None
        self._runtime = SQLiteConnectionRuntime(
            db_path=db_path,
            logger=logger,
            connect_event="sqlite_citation_store_connected",
        )

    async def connect(self) -> None:
        """Initialize database connection using shared pool."""
        self._connection = await self._runtime.ensure_connection(
            self._initialize_schema
        )

    async def disconnect(self) -> None:
        """Release database connection back to pool."""
        if self._connection:
            await self._runtime.disconnect()
            self._connection = None

    async def _initialize_schema(self, connection: aiosqlite.Connection) -> None:
        """Create database tables and indexes."""
        await execute_statements(
            connection,
            [
                """
                CREATE TABLE IF NOT EXISTS citation_sources (
                    citation_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    snippet TEXT,
                    date_published TEXT,
                    source TEXT,
                    full_content TEXT,
                    processed_content TEXT,
                    original_content TEXT,
                    related_citation_id TEXT,
                    related_index INTEGER,
                    query TEXT,
                    parameters TEXT,
                    citation_index INTEGER,
                    created_at TEXT NOT NULL
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_citation_session_id ON citation_sources(session_id)",
                "CREATE INDEX IF NOT EXISTS idx_citation_session_index "
                f"ON citation_sources(session_id, {_INDEX_COLUMN})",
                "CREATE INDEX IF NOT EXISTS idx_citation_created_at ON citation_sources(created_at)",
            ],
        )
        await connection.commit()

    async def _ensure_connection(self) -> aiosqlite.Connection:
        """Ensure database connection is established and return it."""
        if self._connection is None:
            await self.connect()
        assert self._connection is not None
        return self._connection

    def _serialize_citation(self, source: CitationSourceRaw) -> dict:
        """Serialize CitationSourceRaw to dict for database storage."""
        data = source.model_dump(mode="json", exclude_none=True)

        # Recursively convert dict/list values to JSON strings for SQLite
        for key, value in list(data.items()):
            if isinstance(value, (dict, list)):
                data[key] = json.dumps(value)

        # Rename index column
        if "index" in data:
            data[_INDEX_COLUMN] = data.pop("index")

        return data

    def _deserialize_citation(self, row: aiosqlite.Row) -> CitationSourceRaw:
        """Deserialize database row to CitationSourceRaw."""
        data = dict(row)

        # Parse JSON fields
        if data.get("original_content"):
            data["original_content"] = json.loads(data["original_content"])
        if data.get("parameters"):
            data["parameters"] = json.loads(data["parameters"])
        if _INDEX_COLUMN in data:
            data["index"] = data.pop(_INDEX_COLUMN)

        return CitationSourceRaw.model_validate(data)

    async def store_citation_sources(
        self,
        session_id: str,
        sources: list[CitationSourceRaw],
    ) -> list[str]:
        """Store citation sources and return citation_id list."""
        conn = await self._ensure_connection()

        citation_ids = []
        for source in sources:
            source.session_id = session_id

            try:
                data = self._serialize_citation(source)

                columns = ", ".join(data.keys())
                placeholders = ", ".join(["?" for _ in data])
                values = list(data.values())

                query = f"""
                    INSERT OR REPLACE INTO citation_sources ({columns})
                    VALUES ({placeholders})
                """

                await conn.execute(query, values)
                citation_ids.append(source.citation_id)
            except Exception as e:
                logger.exception(
                    "store_citation_source_failed",
                    error=str(e),
                    citation_id=source.citation_id,
                )
                raise

        await conn.commit()

        logger.info(
            "citation_sources_stored",
            session_id=session_id,
            count=len(citation_ids),
        )
        return citation_ids

    async def get_citation_source(
        self,
        citation_id: str,
        session_id: str,
    ) -> CitationSourceRaw | None:
        """Get citation source by citation_id."""
        conn = await self._ensure_connection()

        try:
            async with conn.execute(
                "SELECT * FROM citation_sources WHERE citation_id = ? AND session_id = ?",
                (citation_id, session_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_citation(row)
                return None
        except Exception as e:
            logger.error(
                "get_citation_source_failed",
                error=str(e),
                citation_id=citation_id,
            )
            raise

    async def get_simplified_sources(
        self,
        session_id: str,
        citation_ids: list[str],
    ) -> list[CitationSourceSimplified]:
        """Get simplified citation sources."""
        conn = await self._ensure_connection()

        try:
            if not citation_ids:
                return []

            placeholders = ", ".join(["?" for _ in citation_ids])
            query = f"""
                SELECT citation_id, source_type, url, {_INDEX_COLUMN} AS "index", title, snippet,
                       date_published, source, created_at
                FROM citation_sources
                WHERE citation_id IN ({placeholders}) AND session_id = ?
            """

            params = list(citation_ids) + [session_id]

            simplified: list[CitationSourceSimplified] = []
            async with conn.execute(query, params) as cursor:
                async for row in cursor:
                    simplified.append(
                        CitationSourceSimplified.model_validate(dict(row))
                    )
            return reorder_simplified_sources(simplified, citation_ids)
        except Exception as e:
            logger.error(
                "get_simplified_sources_failed",
                error=str(e),
                session_id=session_id,
            )
            raise

    async def get_session_citations(
        self,
        session_id: str,
    ) -> list[CitationSourceSimplified]:
        """Get all citations for a session."""
        conn = await self._ensure_connection()

        try:
            query = """
                SELECT citation_id, source_type, url, citation_index AS "index", title, snippet,
                       date_published, source, created_at
                FROM citation_sources
                WHERE session_id = ?
                ORDER BY created_at ASC
            """

            simplified: list[CitationSourceSimplified] = []
            async with conn.execute(query, (session_id,)) as cursor:
                async for row in cursor:
                    simplified.append(
                        CitationSourceSimplified.model_validate(dict(row))
                    )
            return sort_simplified_sources(simplified)
        except Exception as e:
            logger.error(
                "get_session_citations_failed",
                error=str(e),
                session_id=session_id,
            )
            raise

    async def update_citation_source(
        self,
        citation_id: str,
        session_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update citation source."""
        conn = await self._ensure_connection()

        try:
            update_data = {}
            for key, value in updates.items():
                if key == "index":
                    key = _INDEX_COLUMN
                if isinstance(value, (dict, list)):
                    update_data[key] = json.dumps(value)
                elif isinstance(value, datetime):
                    update_data[key] = value.isoformat()
                else:
                    update_data[key] = value

            if not update_data:
                return False

            set_clause = ", ".join([f"{k} = ?" for k in update_data.keys()])
            values = list(update_data.values()) + [citation_id, session_id]

            query = f"""
                UPDATE citation_sources
                SET {set_clause}
                WHERE citation_id = ? AND session_id = ?
            """

            cursor = await conn.execute(query, values)
            await conn.commit()

            return cursor.rowcount > 0
        except Exception as e:
            logger.error(
                "update_citation_source_failed",
                error=str(e),
                citation_id=citation_id,
            )
            raise

    async def get_source_by_index(
        self,
        session_id: str,
        index: int,
    ) -> CitationSourceRaw | None:
        """Get citation source by index."""
        conn = await self._ensure_connection()

        try:
            async with conn.execute(
                f"SELECT * FROM citation_sources WHERE session_id = ? AND {_INDEX_COLUMN} = ?",
                (session_id, index),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return self._deserialize_citation(row)
                return None
        except Exception as e:
            logger.error(
                "get_source_by_index_failed",
                error=str(e),
                session_id=session_id,
                index=index,
            )
            raise

    async def cleanup_session(self, session_id: str) -> None:
        """Cleanup citations for a session."""
        conn = await self._ensure_connection()

        try:
            cursor = await conn.execute(
                "DELETE FROM citation_sources WHERE session_id = ?",
                (session_id,),
            )
            await conn.commit()

            logger.info(
                "citation_session_cleaned",
                session_id=session_id,
                deleted_count=cursor.rowcount,
            )
        except Exception as e:
            logger.error(
                "cleanup_session_failed",
                error=str(e),
                session_id=session_id,
            )
            raise


__all__ = ["SQLiteCitationStore"]
