from datetime import datetime

import pytest

from agiwo.tool.storage.citation.models import CitationSourceRaw, CitationSourceType
from agiwo.tool.storage.citation.sqlite_store import SQLiteCitationStore


def _citation(citation_id: str, index: int) -> CitationSourceRaw:
    return CitationSourceRaw(
        citation_id=citation_id,
        session_id="",
        source_type=CitationSourceType.SEARCH,
        url=f"https://example.com/{citation_id}",
        title=citation_id,
        index=index,
        created_at=datetime(2026, 3, 11, 10, 0, index),
    )


@pytest.mark.asyncio
async def test_sqlite_citation_store_round_trip(tmp_path) -> None:
    store = SQLiteCitationStore(str(tmp_path / "citations.db"))
    sources = [_citation("cite-1", 0), _citation("cite-2", 1)]

    citation_ids = await store.store_citation_sources("session-1", sources)
    simplified = await store.get_simplified_sources("session-1", ["cite-2", "cite-1"])
    session_citations = await store.get_session_citations("session-1")

    assert citation_ids == ["cite-1", "cite-2"]
    assert [item.citation_id for item in simplified] == ["cite-2", "cite-1"]
    assert [item.citation_id for item in session_citations] == ["cite-1", "cite-2"]

    await store.disconnect()
