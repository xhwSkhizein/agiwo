"""MongoDB implementation of CitationSourceRepository."""

from typing import TYPE_CHECKING, Any

from motor.motor_asyncio import AsyncIOMotorClient

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class MongoCitationStore:
    """MongoDB implementation of Citation Store."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agio",
        collection_name: str = "citation_sources",
    ):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name or "citation_sources"
        self.client: AsyncIOMotorClient[Any] | None = None
        self.db: AsyncIOMotorDatabase[Any] | None = None
        self.citations_collection: AsyncIOMotorCollection[Any] | None = None

    async def _ensure_connection(self):
        """Ensure database connection is established."""
        if self.client is None:
            from motor.motor_asyncio import (
                AsyncIOMotorClient,
                AsyncIOMotorCollection,
                AsyncIOMotorDatabase,
            )

            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]
            self.citations_collection = self.db[self.collection_name]

            if self.citations_collection is None:
                raise RuntimeError("Failed to get citations collection")
            await self.citations_collection.create_index("citation_id", unique=True)
            await self.citations_collection.create_index("session_id")
            await self.citations_collection.create_index(
                [("session_id", 1), ("index", 1)]
            )
            await self.citations_collection.create_index("created_at")

            logger.info(
                "mongodb_citation_store_connected",
                uri=self.uri,
                db_name=self.db_name,
            )
        else:
            # Check if client is still open
            try:
                # This might not be perfect for motor, but it's a hint
                # For motor, we usually trust the connection pool
                pass
            except Exception:
                self.client = None
                await self._ensure_connection()

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
                if self.citations_collection is None:
                    raise RuntimeError("Database collection not initialized")
                source_data = source.model_dump(mode="json", exclude_none=True)
                await self.citations_collection.update_one(
                    {"citation_id": source.citation_id},
                    {"$set": source_data},
                    upsert=True,
                )
                citation_ids.append(source.citation_id)
            except Exception as e:
                logger.error(
                    "store_citation_source_failed",
                    error=str(e),
                    citation_id=source.citation_id,
                )
                raise

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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            doc = await self.citations_collection.find_one(
                {"citation_id": citation_id, "session_id": session_id}
            )
            if doc:
                return CitationSourceRaw.model_validate(doc)
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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            cursor = self.citations_collection.find(
                {"citation_id": {"$in": citation_ids}, "session_id": session_id}
            )

            simplified = []
            async for doc in cursor:
                source = CitationSourceRaw.model_validate(doc)
                simplified.append(
                    CitationSourceSimplified(
                        citation_id=source.citation_id,
                        source_type=source.source_type,
                        url=source.url,
                        index=source.index,
                        title=source.title,
                        snippet=source.snippet,
                        date_published=source.date_published,
                        source=source.source,
                        created_at=source.created_at,
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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            cursor = self.citations_collection.find({"session_id": session_id})

            simplified = []
            async for doc in cursor:
                source = CitationSourceRaw.model_validate(doc)
                simplified.append(
                    CitationSourceSimplified(
                        citation_id=source.citation_id,
                        source_type=source.source_type,
                        url=source.url,
                        index=source.index,
                        title=source.title,
                        snippet=source.snippet,
                        date_published=source.date_published,
                        source=source.source,
                        created_at=source.created_at,
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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            result = await self.citations_collection.update_one(
                {"citation_id": citation_id, "session_id": session_id},
                {"$set": updates},
            )
            return result.modified_count > 0
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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            doc = await self.citations_collection.find_one(
                {"session_id": session_id, "index": index}
            )
            if doc:
                return CitationSourceRaw.model_validate(doc)
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
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            result = await self.citations_collection.delete_many(
                {"session_id": session_id}
            )
            logger.info(
                "citation_session_cleaned",
                session_id=session_id,
                deleted_count=result.deleted_count,
            )
        except Exception as e:
            logger.error(
                "cleanup_session_failed",
                error=str(e),
                session_id=session_id,
            )
            raise

    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.citations_collection = None
            logger.info("mongodb_citation_store_disconnected")


__all__ = ["MongoCitationStore"]
