from dataclasses import dataclass
from typing import Protocol

from agiwo.tool.base import ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.storage.citation import CitationSourceRaw
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ResolvedWebReaderRequest:
    session_id: str
    url: str
    summarize: bool
    search_query: str | None
    existing_source: CitationSourceRaw | None


class CitationSourceLookupStore(Protocol):
    async def get_source_by_index(
        self,
        session_id: str,
        index: int,
    ) -> CitationSourceRaw | None: ...


async def resolve_web_reader_request(
    *,
    parameters: dict[str, object],
    context: ToolContext,
    citation_source_store: CitationSourceLookupStore,
    tool_name: str,
    start_time: float,
) -> ResolvedWebReaderRequest | ToolResult:
    summarize = bool(parameters.get("summarize", False))
    search_query_value = parameters.get("search_query")
    if summarize and search_query_value:
        return ToolResult.failed(
            tool_name=tool_name,
            error="Error: summarize and search_query are mutually exclusive",
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            start_time=start_time,
        )

    session_id = str(parameters.get("session_id") or context.session_id or "default")
    existing_source: CitationSourceRaw | None = None
    url = parameters.get("url")
    index = parameters.get("index")
    if index is not None:
        if not isinstance(index, int):
            return ToolResult.failed(
                tool_name=tool_name,
                error="Error: index must be an integer",
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )
        existing_source = await citation_source_store.get_source_by_index(
            session_id, index
        )
        if existing_source is None:
            return ToolResult.failed(
                tool_name=tool_name,
                error=f"Search result with index {index} not found",
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )
        url = existing_source.url
        logger.info("web_reader_found_source_by_index", index=index, url=url)

    if not isinstance(url, str) or not url.strip():
        return ToolResult.failed(
            tool_name=tool_name,
            error="Error: url must be a non-empty string",
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            start_time=start_time,
        )

    search_query = (
        search_query_value.strip()
        if isinstance(search_query_value, str) and search_query_value.strip()
        else None
    )
    return ResolvedWebReaderRequest(
        session_id=session_id,
        url=url.strip(),
        summarize=summarize,
        search_query=search_query,
        existing_source=existing_source,
    )
