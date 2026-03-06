"""SQLite implementation of CitationSourceRepository."""

import json
import os
from datetime import datetime
from typing import Any

import aiosqlite

from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)
from agiwo.utils.logging import get_logger
from agiwo.utils.sqlite_pool import get_shared_connection, release_shared_connection

logger = get_logger(__name__)
_INDEX_COLUMN = "citation_index"


class SQLiteCitationStore:
    """SQLite implementation of Citation Store."""

    def __init__(self, db_path: str = "agiwo.db") -> None:
        self.db_path = os.path.expanduser(db_path)
        self._connection: aiosqlite.Connection | None = None
        self._initialized = False

    async def connect(self) -> None:
        """Initialize database connection using shared pool."""
        if self._initialized:
            return

        self._connection = await get_shared_connection(self.db_path)
        await self._create_tables()
        self._initialized = True

        logger.info("sqlite_citation_store_connected", db_path=self.db_path)

    async def disconnect(self) -> None:
        """Release database connection back to pool."""
        if self._connection:
            await release_shared_connection(self.db_path)
            self._connection = None
            self._initialized = False

    async def _create_tables(self) -> None:
        """Create database tables and indexes."""
        if self._connection is None:
            raise RuntimeError("Database connection not established")

        await self._connection.execute(
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
        """
        )

        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation_session_id ON citation_sources(session_id)"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation_session_index "
            f"ON citation_sources(session_id, {_INDEX_COLUMN})"
        )
        await self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_citation_created_at ON citation_sources(created_at)"
        )

        await self._connection.commit()

    async def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if not self._initialized:
            await self.connect()

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
        await self._ensure_connection()

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

                if self._connection is None:
                    raise RuntimeError("Database connection not established")
                await self._connection.execute(query, values)
                citation_ids.append(source.citation_id)
            except Exception as e:
                logger.exception(
                    "store_citation_source_failed",
                    error=str(e),
                    citation_id=source.citation_id,
                )
                raise

        if self._connection is None:
            raise RuntimeError("Database connection not established")
        await self._connection.commit()

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
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
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
        await self._ensure_connection()

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

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            simplified = []
            async with self._connection.execute(query, params) as cursor:
                async for row in cursor:
                    simplified.append(
                        CitationSourceSimplified(
                            citation_id=row["citation_id"],
                            source_type=row["source_type"],
                            url=row["url"],
                            index=row["index"],
                            title=row["title"],
                            snippet=row["snippet"],
                            date_published=row["date_published"],
                            source=row["source"],
                            created_at=row["created_at"],
                        )
                    )
            return simplified
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
        await self._ensure_connection()

        try:
            query = """
                SELECT citation_id, source_type, url, citation_index AS "index", title, snippet,
                       date_published, source, created_at
                FROM citation_sources
                WHERE session_id = ?
                ORDER BY created_at ASC
            """

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            simplified = []
            async with self._connection.execute(query, (session_id,)) as cursor:
                async for row in cursor:
                    simplified.append(
                        CitationSourceSimplified(
                            citation_id=row["citation_id"],
                            source_type=row["source_type"],
                            url=row["url"],
                            index=row["index"],
                            title=row["title"],
                            snippet=row["snippet"],
                            date_published=row["date_published"],
                            source=row["source"],
                            created_at=row["created_at"],
                        )
                    )
            return simplified
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
        await self._ensure_connection()

        try:
            # Convert complex fields to JSON strings
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

            if self._connection is None:
                raise RuntimeError("Database connection not established")
            cursor = await self._connection.execute(query, values)
            await self._connection.commit()

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
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            async with self._connection.execute(
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
        await self._ensure_connection()

        try:
            if self._connection is None:
                raise RuntimeError("Database connection not established")
            cursor = await self._connection.execute(
                "DELETE FROM citation_sources WHERE session_id = ?",
                (session_id,),
            )
            await self._connection.commit()

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
