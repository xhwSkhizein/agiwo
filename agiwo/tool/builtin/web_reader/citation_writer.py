from datetime import datetime
from typing import Any

from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceType,
    generate_citation_id,
)
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebReaderCitationWriter:
    def __init__(self, *, citation_source_store) -> None:
        self._citation_source_store = citation_source_store

    async def store(
        self,
        *,
        session_id: str,
        url: str,
        title: str | None,
        processed_content: str,
        original_content: dict[str, Any],
        search_query: str | None,
        summarize: bool,
        existing_source: CitationSourceRaw | None,
    ) -> str | None:
        try:
            if existing_source is not None:
                citation_id = existing_source.citation_id
                await self._citation_source_store.update_citation_source(
                    citation_id=citation_id,
                    session_id=session_id,
                    updates={
                        "full_content": processed_content,
                        "processed_content": processed_content,
                        "original_content": original_content,
                        "parameters": {
                            "search_query": search_query,
                            "summarize": summarize,
                        },
                    },
                )
                return citation_id

            citation_id = generate_citation_id(prefix="reader")
            await self._citation_source_store.store_citation_sources(
                session_id=session_id,
                sources=[
                    CitationSourceRaw(
                        citation_id=citation_id,
                        session_id=session_id,
                        source_type=CitationSourceType.DIRECT_URL,
                        url=url,
                        title=title,
                        full_content=processed_content,
                        processed_content=processed_content,
                        original_content=original_content,
                        parameters={
                            "search_query": search_query,
                            "summarize": summarize,
                        },
                        created_at=datetime.now(),
                    )
                ],
            )
            return citation_id
        except Exception as exc:  # noqa: BLE001
            logger.error("citation_store_failed", error=str(exc), url=url)
            return None
