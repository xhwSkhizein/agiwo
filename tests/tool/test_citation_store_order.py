from datetime import datetime, timedelta

import pytest

from agiwo.tool.storage.citation.memory_store import InMemoryCitationStore
from agiwo.tool.storage.citation.models import CitationSourceRaw, CitationSourceType


def _source(
    citation_id: str,
    *,
    index: int,
    created_at: datetime,
) -> CitationSourceRaw:
    return CitationSourceRaw(
        citation_id=citation_id,
        session_id="session-1",
        source_type=CitationSourceType.SEARCH,
        url=f"https://example.com/{citation_id}",
        title=citation_id,
        index=index,
        created_at=created_at,
    )


@pytest.mark.asyncio
async def test_in_memory_get_simplified_sources_preserves_requested_order() -> None:
    store = InMemoryCitationStore()
    now = datetime(2026, 3, 8, 12, 0, 0)
    sources = [
        _source("cite-1", index=0, created_at=now),
        _source("cite-2", index=1, created_at=now + timedelta(seconds=1)),
    ]

    await store.store_citation_sources("session-1", sources)
    simplified = await store.get_simplified_sources(
        "session-1",
        ["cite-2", "cite-1"],
    )

    assert [item.citation_id for item in simplified] == ["cite-2", "cite-1"]


@pytest.mark.asyncio
async def test_in_memory_get_session_citations_is_sorted_by_creation_time() -> None:
    store = InMemoryCitationStore()
    now = datetime(2026, 3, 8, 12, 0, 0)
    sources = [
        _source("cite-late", index=1, created_at=now + timedelta(seconds=10)),
        _source("cite-early", index=0, created_at=now),
    ]

    await store.store_citation_sources("session-1", sources)
    simplified = await store.get_session_citations("session-1")

    assert [item.citation_id for item in simplified] == ["cite-early", "cite-late"]
