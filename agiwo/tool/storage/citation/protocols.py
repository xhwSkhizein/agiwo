"""
Citation storage protocols.

Defines abstract interfaces for citation source storage.
"""

from typing import Any, Protocol

from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)


class CitationSourceRepository(Protocol):
    """Citation source repository abstract interface.

    Responsibilities:
    - Store citation sources and generate citation_id
    - Retrieve citation sources by citation_id
    - Provide simplified results (hide detailed information)
    - Support updating citation sources (for supplementing full content during fetch)
    """

    async def store_citation_sources(
        self,
        session_id: str,
        sources: list[CitationSourceRaw],
    ) -> list[str]:
        """Store citation sources and return citation_id list.

        Args:
            session_id: Session ID
            sources: Citation source list

        Returns:
            List of citation IDs
        """
        ...

    async def get_citation_source(
        self,
        citation_id: str,
        session_id: str,
    ) -> CitationSourceRaw | None:
        """Get raw citation source by citation_id.

        Args:
            citation_id: Citation ID
            session_id: Session ID (for verification)

        Returns:
            Raw citation source or None
        """
        ...

    async def get_simplified_sources(
        self,
        session_id: str,
        citation_ids: list[str],
    ) -> list[CitationSourceSimplified]:
        """Get simplified citation sources (hide detailed information).

        Args:
            session_id: Session ID
            citation_ids: Citation ID list

        Returns:
            List of simplified citation sources
        """
        ...

    async def get_session_citations(
        self,
        session_id: str,
    ) -> list[CitationSourceSimplified]:
        """Get all citations for a session.

        Args:
            session_id: Session ID

        Returns:
            List of simplified citation sources
        """
        ...

    async def update_citation_source(
        self,
        citation_id: str,
        session_id: str,
        updates: dict[str, Any],
    ) -> bool:
        """Update citation source (for supplementing full content during fetch).

        Args:
            citation_id: Citation ID
            session_id: Session ID (for verification)
            updates: Dictionary of fields to update

        Returns:
            Whether update was successful
        """
        ...

    async def get_source_by_index(
        self,
        session_id: str,
        index: int,
    ) -> CitationSourceRaw | None:
        """Get citation source by index (for web_fetch(index=N)).

        Args:
            session_id: Session ID
            index: Numeric index

        Returns:
            Raw citation source or None
        """
        ...

    async def cleanup_session(self, session_id: str) -> None:
        """
        Clean up citation sources for specific session.

        Args:
            session_id: Session ID to clean up
        """
        ...
