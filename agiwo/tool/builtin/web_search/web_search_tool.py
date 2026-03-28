"""Web search tool backed by Serper."""

import time
from datetime import datetime, timezone
from typing import Any

from agiwo.config.settings import settings
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.builtin.http_client import AsyncHttpClient
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceSimplified,
    CitationSourceType,
    CitationStoreConfig,
    create_citation_store,
    generate_citation_id,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_search")
class WebSearchTool(BaseTool):
    """Search the web and store lightweight citations for follow-up reads."""

    cacheable = True

    def __init__(
        self,
        *,
        citation_store_config: CitationStoreConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.timeout_seconds = settings.web_search_api_timeout
        self.max_results = settings.web_search_api_max_results
        self.recency_days = settings.web_search_api_recency_days
        key = settings.web_search_serper_api_key
        self.serper_api_key = key.get_secret_value() if key is not None else None
        self._citation_source_store = create_citation_store(citation_store_config)
        self._http_client = AsyncHttpClient(
            base_url=settings.web_search_api_base_url,
            timeout=self.timeout_seconds,
            max_retries=settings.web_search_api_max_retries,
        )

    def get_name(self) -> str:
        return "web_search"

    def get_description(self) -> str:
        return """This tool performs web searches and returns:
- Numbered search results (0, 1, 2...) with titles and snippets
- Publication dates (when available)
- Source information

Use this tool when you need to:
- Find current information not in your training data
- Research topics or gather facts
- Discover relevant web resources
- Get multiple perspectives on a topic

To fetch full content from a search result, use web_reader with the result index: web_reader(index=0) => Fetch content from the first search result"""

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Single web search query.",
                },
            },
            "required": ["query"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    def _contains_chinese(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _build_recency_filter(self) -> str | None:
        days = self.recency_days
        if days <= 0:
            return None
        if days <= 1:
            return "qdr:d"
        if days <= 7:
            return "qdr:w"
        if days <= 31:
            return "qdr:m"
        return "qdr:y"

    def _build_payload(self, query: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"q": query, "num": self.max_results}
        if self._contains_chinese(query):
            payload.update({"location": "China", "gl": "cn", "hl": "zh-cn"})
        else:
            payload.update({"location": "United States", "gl": "us", "hl": "en"})

        tbs = self._build_recency_filter()
        if tbs is not None:
            payload["tbs"] = tbs
        return payload

    async def _google_search_with_serper(
        self, query: str
    ) -> list[dict[str, str]] | str:
        if not self.serper_api_key:
            return (
                "Error: SERPER_API_KEY is not set. "
                "Please set the SERPER_API_KEY environment variable."
            )

        try:
            response = await self._http_client.post_json(
                endpoint="/search",
                data=self._build_payload(query),
                headers={"X-API-KEY": self.serper_api_key},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("search_api_failed", query=query, error=str(exc))
            return f"Error performing web search: {exc}"

        organic = response.get("organic")
        if not isinstance(organic, list) or not organic:
            return f"No results found for query: '{query}'. Try a less specific query."

        search_results: list[dict[str, str]] = []
        for page in organic:
            if len(search_results) >= self.max_results:
                break
            if not isinstance(page, dict):
                continue
            snippet = str(page.get("snippet") or "").replace(
                "Your browser can't play this video.", ""
            )
            search_results.append(
                {
                    "url": str(page.get("link") or ""),
                    "title": str(page.get("title") or "No title"),
                    "snippet": snippet,
                    "date_published": str(page.get("date") or ""),
                    "source": str(
                        page.get("source")
                        or page.get("hostName")
                        or page.get("host_name")
                        or ""
                    ),
                }
            )

        logger.info(
            "search_api_success",
            query=query,
            result_count=len(search_results),
        )
        return search_results

    def _format_simplified_results(
        self,
        query: str,
        simplified_results: list[CitationSourceSimplified],
        citation_ids: list[str],
    ) -> str:
        content_parts = [
            f"A web search for '{query}' found {len(simplified_results)} results:",
            "\n## Web Results\n",
        ]

        for i, result in enumerate(simplified_results):
            citation_id = (
                citation_ids[i] if i < len(citation_ids) else result.citation_id
            )
            citation_mark = f"[cite:{citation_id}] " if citation_id else ""
            index_str = f"{result.index}. " if result.index is not None else ""

            parts = [f"{index_str}{citation_mark}{result.title or 'No title'}"]
            if result.date_published:
                parts.append(f"\nDate published: {result.date_published}")
            if result.source:
                parts.append(f"\nSource: {result.source}")
            if result.snippet:
                parts.append(f"\n{result.snippet}")
            content_parts.append("".join(parts))

        content_parts.append(
            "\n\n**Note**: To fetch full content from any result, use: web_reader(index=N)"
        )
        return "\n\n".join(content_parts)

    def _format_fallback_results(
        self,
        query: str,
        raw_results: list[dict[str, str]],
    ) -> str:
        content_parts = [
            f"A web search for '{query}' found {len(raw_results)} results:",
            "\n## Web Results\n",
        ]

        for idx, item in enumerate(raw_results):
            parts = [f"{idx}. [{item['title']}]({item['url']})"]
            if item.get("date_published"):
                parts.append(f"\nDate published: {item['date_published']}")
            if item.get("source"):
                parts.append(f"\nSource: {item['source']}")
            if item.get("snippet"):
                parts.append(f"\n{item['snippet']}")
            content_parts.append("".join(parts))

        content_parts.append(
            "\n\n**Note**: To fetch full content from any result, use: web_reader(index=N)"
        )
        return "\n\n".join(content_parts)

    def _convert_to_citation_sources(
        self,
        raw_results: list[dict[str, str]],
        query: str,
        session_id: str,
        start_index: int = 0,
    ) -> list[CitationSourceRaw]:
        citation_sources = []
        for offset, item in enumerate(raw_results):
            citation_sources.append(
                CitationSourceRaw(
                    citation_id=generate_citation_id(prefix="search"),
                    session_id=session_id,
                    source_type=CitationSourceType.SEARCH,
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    date_published=item.get("date_published"),
                    source=item.get("source"),
                    query=query,
                    index=start_index + offset,
                    created_at=datetime.now(timezone.utc),
                )
            )
        return citation_sources

    async def _process_and_format_results(
        self,
        query: str,
        raw_results: list[dict[str, str]],
        session_id: str,
    ) -> tuple[str, list[str]]:
        try:
            existing_sources = await self._citation_source_store.get_session_citations(
                session_id
            )
            start_index = (
                max(
                    (
                        source.index
                        for source in existing_sources
                        if source.index is not None
                    ),
                    default=-1,
                )
                + 1
            )

            citation_sources = self._convert_to_citation_sources(
                raw_results, query, session_id, start_index
            )
            citation_ids = await self._citation_source_store.store_citation_sources(
                session_id=session_id,
                sources=citation_sources,
            )
            simplified_results = (
                await self._citation_source_store.get_simplified_sources(
                    session_id=session_id,
                    citation_ids=citation_ids,
                )
            )

            logger.info(
                "citation_sources_stored_and_simplified",
                query=query,
                result_count=len(simplified_results),
                citation_ids=citation_ids,
            )
            return (
                self._format_simplified_results(
                    query=query,
                    simplified_results=simplified_results,
                    citation_ids=citation_ids,
                ),
                citation_ids,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "citation_source_store_failed",
                query=query,
                error=str(exc),
            )
            return self._format_fallback_results(query, raw_results), []

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()

        try:
            if abort_signal and abort_signal.is_aborted():
                return ToolResult.aborted(
                    tool_name=self.name,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    start_time=start_time,
                )

            query = parameters.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult.failed(
                    tool_name=self.name,
                    error="Error: query must be a non-empty string",
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    start_time=start_time,
                )
            query = query.strip()
            session_id = str(
                parameters.get("session_id") or context.session_id or "default"
            )

            raw_results = await self._google_search_with_serper(query)
            if isinstance(raw_results, str):
                return ToolResult.failed(
                    tool_name=self.name,
                    error=raw_results,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    start_time=start_time,
                )

            if abort_signal and abort_signal.is_aborted():
                return ToolResult.aborted(
                    tool_name=self.name,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    start_time=start_time,
                )

            result_text, citation_ids = await self._process_and_format_results(
                query=query,
                raw_results=raw_results,
                session_id=session_id,
            )

            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                content=result_text,
                output={
                    "query": query,
                    "result_count": len(raw_results),
                    "citation_ids": citation_ids,
                },
                start_time=start_time,
            )
        except Exception as exc:  # noqa: BLE001
            error_message = str(exc) if str(exc) else "Unknown error occurred"
            return ToolResult.failed(
                tool_name=self.name,
                error=f"Error performing web search: {error_message}",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
