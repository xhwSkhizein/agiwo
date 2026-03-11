"""MongoDB implementation of CitationSourceRepository."""

from typing import Any

from agiwo.utils.storage_support.mongo_runtime import (
    MongoCollectionRuntime,
    MongoIndexSpec,
)
from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)
from agiwo.tool.storage.citation.utils import (
    reorder_simplified_sources,
    sort_simplified_sources,
    to_simplified_source,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class MongoCitationStore:
    """MongoDB implementation of Citation Store."""

    def __init__(
        self,
        uri: str = "mongodb://localhost:27017",
        db_name: str = "agiwo",
        collection_name: str = "citation_sources",
    ):
        self.uri = uri
        self.db_name = db_name
        self.collection_name = collection_name or "citation_sources"
        self.client: Any = None
        self.db: Any = None
        self.citations_collection: Any = None
        self._runtime = MongoCollectionRuntime(
            uri=uri,
            db_name=db_name,
            logger=logger,
            connect_event="mongodb_citation_store_connected",
            disconnect_event="mongodb_citation_store_disconnected",
        )

    async def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if self.client is not None:
            return

        self.citations_collection = await self._runtime.ensure_collection(
            self.collection_name,
            indexes=[
                MongoIndexSpec("citation_id", unique=True),
                MongoIndexSpec("session_id"),
                MongoIndexSpec([("session_id", 1), ("index", 1)]),
                MongoIndexSpec("created_at"),
            ],
        )
        self.client = self._runtime.client
        self.db = self._runtime.db

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
                logger.exception(
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
            if not citation_ids:
                return []
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            cursor = self.citations_collection.find(
                {"citation_id": {"$in": citation_ids}, "session_id": session_id}
            ).sort("created_at", 1)

            simplified: list[CitationSourceSimplified] = []
            async for doc in cursor:
                source = CitationSourceRaw.model_validate(doc)
                simplified.append(to_simplified_source(source))
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
        await self._ensure_connection()

        try:
            if self.citations_collection is None:
                raise RuntimeError("Database collection not initialized")
            cursor = self.citations_collection.find({"session_id": session_id}).sort(
                "created_at", 1
            )

            simplified: list[CitationSourceSimplified] = []
            async for doc in cursor:
                source = CitationSourceRaw.model_validate(doc)
                simplified.append(to_simplified_source(source))
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
        """Release MongoDB connection back to the shared pool."""
        if self.client is not None:
            await self._runtime.disconnect()
            self.client = None
            self.db = None
            self.citations_collection = None


__all__ = ["MongoCitationStore"]
