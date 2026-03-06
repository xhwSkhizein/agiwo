"""
WebSearchTool - Web search tool.

Uses Serper API for Google search.
"""

import http.client
import json
import time
from datetime import datetime
from typing import Any

from agiwo.tool.base import ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceSimplified,
    CitationSourceType,
    CitationSourceRepository,
    generate_citation_id,
)
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.config import WebSearchApiConfig
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_search")
class WebSearchTool(BaseTool):
    """Web search tool using Serper API for Google search.

    Optimization features:
    - Search results stored to MongoDB (with TTL auto-cleanup support)
    - Use numeric indices (0, 1, 2...) to minimize token usage
    - Hide URLs from LLM, reducing 60-70% token consumption
    - Support fetching full content via index in web_fetch
    """

    # Enable caching - search results are stable within a session
    cacheable = True

    def __init__(
        self,
        *,
        config: WebSearchApiConfig | None = None,
        citation_source_store: "CitationSourceRepository | None" = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._config = config or WebSearchApiConfig()
        self.timeout_seconds = self._config.timeout_seconds
        self.max_results = self._config.max_results
        self.serper_api_key = self._config.serper_api_key.get_secret_value()
        self._citation_source_store = citation_source_store

    def get_name(self) -> str:
        """Return tool name."""
        return "web_search"

    def get_description(self) -> str:
        """Return tool description."""
        return """This tool performs Google searches and returns:
- Numbered search results (0, 1, 2...) with titles and snippets
- Publication dates (when available)
- Source information

Use this tool when you need to:
- Find current information not in your training data
- Research topics or gather facts
- Discover relevant web resources
- Get multiple perspectives on a topic

To fetch full content from a search result, use web_fetch tool with the result index:
> web_fetch(index=0) => Fetch content from the first search result"""

    def get_parameters(self) -> dict[str, Any]:
        """Return tool parameters JSON Schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "oneOf": [
                        {"type": "string", "description": "single search query"},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "multiple search queries list",
                        },
                    ],
                    "description": "search query (single or multiple)",
                },
            },
            "required": ["query"],
        }

    def is_concurrency_safe(self) -> bool:
        """Whether the tool supports concurrency."""
        return True

    def needs_permissions(self, parameters: dict[str, Any]) -> bool:
        """Check if the tool needs permission check."""
        # Web search is a low-risk operation
        return False

    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _google_search_with_serp(self, query: str) -> list[dict] | str:
        """Execute Google search using Serper API."""
        if not self.serper_api_key:
            return (
                "Error: SERPER_API_KEY is not set. "
                "Please set the SERPER_API_KEY environment variable."
            )

        try:
            conn = http.client.HTTPSConnection("google.serper.dev")

            # Set region and language based on query language
            # if self._contains_chinese(query):
            #     payload = json.dumps(
            #         {
            #             "q": query,
            #             "location": "China",
            #             "gl": "cn",
            #             "hl": "zh-cn",
            #             "tbs": "qdr:y",
            #         }
            #     )
            # else:
            payload = json.dumps(
                {
                    "q": query,
                    "location": "United States",
                    "gl": "us",
                    "hl": "en",
                    "tbs": "qdr:y",
                }
            )

            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }

            # Retry mechanism (up to 5 times)
            res = None
            for i in range(5):
                try:
                    conn.request("POST", "/search", payload, headers)
                    res = conn.getresponse()
                    break
                except Exception as e:
                    logger.warning(f"Search attempt {i + 1} failed: {e!s}")
                    if i == 4:
                        return "Error: Google search timeout. Please try again later."
                    continue

            if res is None:
                return "Error: Failed to connect to search service."

            data = res.read()
            results = json.loads(data.decode("utf-8"))

            # Check if search results exist
            if "organic" not in results:
                return (
                    f"No results found for query: '{query}'. Try a less specific query."
                )

            # Extract search results (return raw data)
            search_results = []
            for page in results.get("organic", []):
                if len(search_results) >= self.max_results:
                    break

                # Extract result data
                title = page.get("title", "No title")
                link = page.get("link", "")
                snippet = page.get("snippet", "")
                # Clean up useless content
                snippet = snippet.replace("Your browser can't play this video.", "")
                date_published = page.get("date", "")
                source = page.get("source", "")

                search_results.append(
                    {
                        "url": link,
                        "title": title,
                        "snippet": snippet,
                        "date_published": date_published,
                        "source": source,
                    }
                )

            return search_results

        except Exception as e:
            logger.error(f"Error during search: {e!s}")
            raise

    def _format_simplified_results(
        self,
        query: str,
        simplified_results: list[CitationSourceSimplified],
        citation_ids: list[str],
    ) -> str:
        """Format simplified results (without URLs, using numeric indices and citation marks)."""
        content_parts = [
            f"A Google search for '{query}' found {len(simplified_results)} results:",
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
            "\n\n**Note**: To fetch full content from any result, use: web_fetch(index=N)"
        )

        return "\n\n".join(content_parts)

    def _format_fallback_results(
        self,
        query: str,
        raw_results: list[dict],
    ) -> str:
        """Fallback formatting (with URLs, used when storage fails)."""
        content_parts = [
            f"A Google search for '{query}' found {len(raw_results)} results:",
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

        return "\n\n".join(content_parts)

    def _convert_to_citation_sources(
        self,
        raw_results: list[dict],
        query: str,
        session_id: str,
        start_index: int = 0,
    ) -> list[CitationSourceRaw]:
        """Convert raw results to CitationSourceRaw models."""
        citation_sources = []
        for offset, item in enumerate(raw_results):
            citation_id = generate_citation_id(prefix="search")
            citation_sources.append(
                CitationSourceRaw(
                    citation_id=citation_id,
                    session_id=session_id,
                    source_type=CitationSourceType.SEARCH,
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    date_published=item.get("date_published"),
                    source=item.get("source"),
                    query=query,
                    index=start_index + offset,
                    created_at=datetime.now(),
                )
            )
        return citation_sources

    async def _process_and_format_results(
        self,
        query: str,
        raw_results: list[dict],
        session_id: str,
    ) -> tuple[str, list[str]]:
        """Process and format search results (storage + formatting, with fallback).

        Returns:
            Tuple of (formatted result text, citation_id list)
        """
        # Try to store and return simplified format
        if self._citation_source_store:
            try:
                # Get maximum index for current session
                existing_sources = (
                    await self._citation_source_store.get_session_citations(session_id)
                )
                start_index = (
                    max(
                        (s.index for s in existing_sources if s.index is not None),
                        default=-1,
                    )
                    + 1
                    if existing_sources
                    else 0
                )

                # Convert to CitationSourceRaw models
                citation_sources = self._convert_to_citation_sources(
                    raw_results, query, session_id, start_index
                )

                # Store and get citation_ids
                citation_ids = await self._citation_source_store.store_citation_sources(
                    session_id=session_id,
                    sources=citation_sources,
                )

                # Get simplified results
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

                result_text = self._format_simplified_results(
                    query=query,
                    simplified_results=simplified_results,
                    citation_ids=citation_ids,
                )
                return result_text, citation_ids

            except Exception as e:
                logger.error(
                    "citation_source_store_failed",
                    query=query,
                    error=str(e),
                )
                # Fallback: return raw format
                return self._format_fallback_results(query, raw_results), []
        else:
            # No store instance: return raw format
            return self._format_fallback_results(query, raw_results), []

    async def execute(
        self,
        parameters: dict[str, Any],
        context: "ExecutionContext",
        abort_signal: "AbortSignal | None" = None,
    ) -> ToolResult:
        """Execute web search."""
        start_time = time.time()

        try:
            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # Get parameters
            session_id = parameters.get("session_id", "default")
            query = parameters.get("query")

            if not query:
                return self._create_error_result(
                    parameters, "Error: No query provided", start_time
                )

            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # Process single query
            if isinstance(query, str):
                raw_results = self._google_search_with_serp(query)

                if isinstance(raw_results, str):  # Error message
                    return self._create_error_result(
                        parameters, raw_results, start_time
                    )

                # Check abort signal
                if abort_signal and abort_signal.is_aborted():
                    return self._create_abort_result(parameters, start_time)

                # Process and format results
                result_text, citation_ids = await self._process_and_format_results(
                    query=query,
                    raw_results=raw_results,
                    session_id=session_id,
                )

                return ToolResult(
                    tool_name=self.name,
                    tool_call_id=parameters.get("tool_call_id", ""),
                    input_args=parameters,
                    content=result_text,
                    output={
                        "query": query,
                        "result_count": len(raw_results),
                        "citation_ids": citation_ids,
                    },
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                    is_success=True,
                )

            # Process multiple queries (not yet supported)
            return self._create_error_result(
                parameters,
                "Error: Multiple queries not yet supported in refactored version",
                start_time,
            )

        except Exception as exc:
            error_message = str(exc) if str(exc) else "Unknown error occurred"
            return self._create_error_result(
                parameters, f"Error performing web search: {error_message}", start_time
            )
