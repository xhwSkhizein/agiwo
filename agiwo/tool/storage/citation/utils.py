"""Citation utilities."""

import secrets
from datetime import datetime
from typing import Iterable

from agiwo.tool.storage.citation.models import (
    CitationSourceRaw,
    CitationSourceSimplified,
)


def generate_citation_id(prefix: str = "cite") -> str:
    """Generate a unique citation ID."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_suffix = secrets.token_hex(4)
    return f"{prefix}-{timestamp}-{random_suffix}"


def to_simplified_source(source: CitationSourceRaw) -> CitationSourceSimplified:
    return CitationSourceSimplified(
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


def sort_simplified_sources(
    sources: Iterable[CitationSourceSimplified],
) -> list[CitationSourceSimplified]:
    return sorted(
        sources,
        key=lambda source: (
            source.created_at or datetime.min,
            source.index if source.index is not None else float("inf"),
            source.citation_id,
        ),
    )


def reorder_simplified_sources(
    sources: Iterable[CitationSourceSimplified],
    citation_ids: list[str],
) -> list[CitationSourceSimplified]:
    by_id = {source.citation_id: source for source in sources}
    return [by_id[citation_id] for citation_id in citation_ids if citation_id in by_id]


__all__ = [
    "generate_citation_id",
    "reorder_simplified_sources",
    "sort_simplified_sources",
    "to_simplified_source",
]
