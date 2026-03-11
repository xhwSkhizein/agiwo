"""
In-memory citation store implementation.

Used for testing and scenarios that don't require persistence.
"""

from typing import Any

from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)
from agiwo.tool.storage.citation.utils import (
    reorder_simplified_sources,
    sort_simplified_sources,
    to_simplified_source,
)


class InMemoryCitationStore:
    """In-memory citation store implementation."""

    def __init__(self) -> None:
        # session_id -> citation_id -> CitationSourceRaw
        self._sources: dict[str, dict[str, CitationSourceRaw]] = {}
        # session_id -> index -> citation_id
        self._index_map: dict[str, dict[int, str]] = {}

    async def store_citation_sources(
        self,
        session_id: str,
        sources: list[CitationSourceRaw],
    ) -> list[str]:
        """Store citation sources and return list of citation_ids."""
        if session_id not in self._sources:
            self._sources[session_id] = {}
            self._index_map[session_id] = {}

        citation_ids = []
        for source in sources:
            # Ensure session_id consistency
            source.session_id = session_id

            # Store source
            self._sources[session_id][source.citation_id] = source
            citation_ids.append(source.citation_id)

            # Build index mapping if index is provided
            if source.index is not None:
                self._index_map[session_id][source.index] = source.citation_id

        return citation_ids

    async def get_citation_source(
        self,
        citation_id: str,
        session_id: str,
    ) -> CitationSourceRaw | None:
        """Get citation source by citation_id."""
        if session_id not in self._sources:
            return None
        return self._sources[session_id].get(citation_id)

    async def get_simplified_sources(
        self,
        session_id: str,
        citation_ids: list[str],
    ) -> list[CitationSourceSimplified]:
        """Get simplified citation sources."""
        if session_id not in self._sources:
            return []

        simplified = []
        for citation_id in citation_ids:
            source = self._sources[session_id].get(citation_id)
            if source:
                simplified.append(to_simplified_source(source))
        return reorder_simplified_sources(simplified, citation_ids)

    async def get_session_citations(
        self,
        session_id: str,
    ) -> list[CitationSourceSimplified]:
        """Get all citations for a session."""
        if session_id not in self._sources:
            return []

        sources = self._sources[session_id].values()
        return sort_simplified_sources(
            to_simplified_source(source) for source in sources
        )

    async def update_citation_source(
        self,
        citation_id: str,
        session_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update citation source fields."""
        if session_id not in self._sources:
            return False

        source = self._sources[session_id].get(citation_id)
        if not source:
            return False

        # Update fields
        for key, value in updates.items():
            if hasattr(source, key):
                setattr(source, key, value)

        return True

    async def get_source_by_index(
        self,
        session_id: str,
        index: int,
    ) -> CitationSourceRaw | None:
        """Get citation source by index."""
        if session_id not in self._index_map:
            return None

        citation_id = self._index_map[session_id].get(index)
        if not citation_id:
            return None

        return self._sources[session_id].get(citation_id)

    async def cleanup_session(self, session_id: str) -> None:
        """Clean up citation sources for a specific session."""
        if session_id in self._sources:
            del self._sources[session_id]
        if session_id in self._index_map:
            del self._index_map[session_id]

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_sessions": len(self._sources),
            "total_sources": sum(len(sources) for sources in self._sources.values()),
            "sessions": {
                session_id: len(sources)
                for session_id, sources in self._sources.items()
            },
        }
