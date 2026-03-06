"""
WebFetchTool - Advanced Web Content Extraction Tool

Complete rewrite with improved content extraction, quality validation, and smart filtering.
Supports multiple extraction strategies and content type detection.
"""

import time
from datetime import datetime
from typing import Any

from agiwo.llm.base import Model
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceRepository,
    CitationSourceType,
    generate_citation_id,
)
from agiwo.tool.base import BaseTool,ToolResult
from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.web_reader.curl_cffi import SimpleAsyncClient
from agiwo.tool.builtin.html_extract import HtmlContent
from agiwo.tool.builtin.web_reader.playwright.core import (
    PlaywrightCrawler
)
from agiwo.tool.builtin.utils import truncate_middle
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_reader")
class WebReaderTool(BaseTool):
    """Web content fetching tool.

    Core features:
    1. Use curl_cffi for fast content fetching, fallback to Playwright on failure
    2. Use trafilatura for main content extraction
    3. Support LLM summarization and query extraction (optional)
    4. Support Citation system (optional)
    """

    # Enable caching - web page content is stable within a session
    cacheable = True

    def __init__(
        self,
        *,
        config: WebReaderApiConfig | None = None,
        citation_source_store: "CitationSourceRepository | None" = None,
        llm_model: "Model | None" = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._config = config or WebReaderApiConfig()
        self._citation_source_store = citation_source_store
        self._llm_model = llm_model

        # Core configuration
        self.timeout_seconds = self._config.timeout_seconds
        self.max_length = self._config.max_content_length

        # HTTP client and crawler
        self._curl_cffi_client = SimpleAsyncClient()
        self._playwright_crawl = PlaywrightCrawler(config=self._config)

    def get_name(self) -> str:
        return "web_reader"

    def get_description(self) -> str:
        return """Fetch web content into text only format.
**Usage modes:**
1. **By search index**: `index=0` (recommended, uses web_search results)
2. **By direct URL**: `url="https://..."` (backward compatible)

**Content processing options:**
- `search_query`: if provided, return content relevant to your specific query
- `summarize`: if provided, generate concise summary of main content
- Note: These options are mutually exclusive
"""

    def get_parameters(self) -> dict[str, Any]:
        """Return tool parameters JSON Schema"""
        return {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Search result index from web_search (0, 1, 2...)",
                    "minimum": 0,
                },
                "url": {
                    "oneOf": [
                        {"type": "string", "description": "Single URL"},
                        {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Multiple URLs",
                        },
                    ],
                    "description": "Direct URL (backward compatible)",
                },
                "search_query": {
                    "type": "string",
                    "description": (
                        "Extract content relevant to this query "
                        "(mutually exclusive with summarize)"
                    ),
                },
                "summarize": {
                    "type": "boolean",
                    "description": (
                        "Generate concise summary "
                        "(mutually exclusive with search_query, default: false)"
                    ),
                    "default": False,
                },
            },
            "oneOf": [
                {"required": ["index"]},
                {"required": ["url"]},
            ],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    def needs_permissions(self, parameters: dict[str, Any]) -> bool:
        return False

    async def execute(
        self,
        parameters: dict[str, Any],
        context: "ExecutionContext",
        abort_signal: "AbortSignal | None" = None,
    ) -> ToolResult:
        """Execute web content fetching."""
        start_time = time.time()

        try:
            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # Get parameters
            session_id = parameters.get("session_id", "default")
            index = parameters.get("index")
            url = parameters.get("url")
            search_query = parameters.get("search_query")
            summarize = parameters.get("summarize", False)

            # Get URL by index (if store is available)
            existing_source: CitationSourceRaw | None = None
            if index is not None:
                if not self._citation_source_store:
                    return self._create_error_result(
                        parameters,
                        "Citation store not available, cannot use index",
                        start_time,
                    )

                existing_source = await self._citation_source_store.get_source_by_index(
                    session_id, index
                )
                if not existing_source:
                    return self._create_error_result(
                        parameters,
                        f"Search result with index {index} not found",
                        start_time,
                    )
                url = existing_source.url
                logger.info(f"Found source by index {index}: {url}")

            if not url:
                return self._create_error_result(
                    parameters, "Error: url parameter is required", start_time
                )

            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # Fetch content: try curl_cffi first, fallback to Playwright on failure
            content: HtmlContent | None = await self._curl_cffi_client.fetch(url)
            if content:
                logger.info(f"curl_cffi fetch succeeded: {url}")
            else:
                # Fallback to playwright
                logger.info(f"curl_cffi failed, using Playwright: {url}")
                started = False
                fallback_error: str | None = None
                try:
                    await self._playwright_crawl.start()
                    started = True
                    content = await self._playwright_crawl.crawl_url(url)
                except Exception as exc:  # noqa: BLE001
                    fallback_error = str(exc)
                    content = None
                finally:
                    if started:
                        await self._playwright_crawl.stop()

            if not content:
                if fallback_error:
                    return self._create_error_result(
                        parameters, fallback_error, start_time
                    )
                return self._create_error_result(
                    parameters, "Failed to fetch content", start_time
                )

            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # Content processing: summarization or query extraction
            if self._llm_model:
                if summarize or len(content.text or "") > self.max_length:
                    content = await self._summarize_by_llm(content, abort_signal)
                elif search_query:
                    content = await self._extract_by_query(
                        content, search_query, abort_signal
                    )

            # Truncate content
            processed_content = content.raw_text or content.text or ""
            if len(processed_content) > self.max_length:
                processed_content = truncate_middle(processed_content, self.max_length)

            # Prepare metadata
            original_content_dict = content.model_dump(exclude_none=True)
            original_content_dict.pop("raw_html", None)
            original_content_dict.pop("text", None)

            # Store/update citation (if store is available)
            citation_id = None
            if self._citation_source_store:
                try:
                    if existing_source:
                        # Update existing record
                        citation_id = existing_source.citation_id
                        await self._citation_source_store.update_citation_source(
                            citation_id=citation_id,
                            session_id=session_id,
                            updates={
                                "full_content": processed_content,
                                "processed_content": processed_content,
                                "original_content": original_content_dict,
                                "parameters": {
                                    "search_query": search_query,
                                    "summarize": summarize,
                                },
                            },
                        )
                        logger.info(f"Updated citation {citation_id}")
                    else:
                        # Create new record
                        citation_id = generate_citation_id(prefix="fetch")
                        new_source = CitationSourceRaw(
                            citation_id=citation_id,
                            session_id=session_id,
                            source_type=CitationSourceType.DIRECT_URL,
                            url=url,
                            title=content.title,
                            full_content=processed_content,
                            processed_content=processed_content,
                            original_content=original_content_dict,
                            parameters={
                                "search_query": search_query,
                                "summarize": summarize,
                            },
                            created_at=datetime.now(),
                        )
                        await self._citation_source_store.store_citation_sources(
                            session_id=session_id,
                            sources=[new_source],
                        )
                        logger.info(f"Created citation {citation_id}")
                except Exception as e:
                    logger.error(f"Citation store failed: {e}")

            # Format result
            result_content = processed_content
            if citation_id:
                result_content = f"[cite:{citation_id}]\n\n{processed_content}"

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=result_content,
                output={
                    "url": url,
                    "title": content.title,
                    "content_length": len(processed_content),
                    "citation_id": citation_id,
                    "truncated": len(processed_content) >= self.max_length,
                },
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                is_success=True,
            )

        except Exception as e:
            return self._create_error_result(parameters, f"Error: {e!s}", start_time)

    async def _summarize_by_llm(
        self,
        content: HtmlContent,
        abort_signal: "AbortSignal | None" = None,
    ) -> HtmlContent:
        """Summarize content using LLM.

        Args:
            content: HTML content
            abort_signal: Abort signal

        Returns:
            Processed content
        """
        if not self._llm_model:
            return content

        system_prompt = """You are a professional content summarization assistant.
Your task is to:
1. Extract the core information and key insights from the content user provided
2. Maintain factual accuracy and preserve important details
3. Keep the summary comprehensive but concise (aim for 25-35% of original length)
"""

        user_message = f"""summarize the following content:

{content.text}
"""

        try:
            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return content

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Call Model streaming interface and aggregate results
            content_parts = []
            async for chunk in self._llm_model.arun_stream(
                messages=messages, tools=None
            ):
                # Check abort signal
                if abort_signal and abort_signal.is_aborted():
                    logger.info("LLM summarization aborted")
                    return content

                if chunk.content:
                    content_parts.append(chunk.content)

            if content_parts:
                content.raw_text = "".join(content_parts)
                return content
            else:
                return content
        except Exception as e:
            logger.error(f"Error summarizing content: {e}")
            return content

    async def _extract_by_query(
        self,
        content: HtmlContent,
        query: str,
        abort_signal: "AbortSignal | None" = None,
    ) -> HtmlContent:
        """Extract content by query using LLM.

        Args:
            content: HTML content
            query: Query string
            abort_signal: Abort signal

        Returns:
            Processed content
        """
        if not self._llm_model:
            return content

        system_prompt = """You are a professional content extraction assistant.
Your task is to:
1. Extract content specifically relevant to the user's query from the content user provided
2. Maintain full context and accuracy of extracted information
3. Preserve technical details, data, and important specifics
4. If no relevant content exists, clearly state so
5. Focus on substantive information, not just keyword matches
"""

        user_message = f"""the content is:

{content.text}

user's query is:
{query}
"""

        try:
            # Check abort signal
            if abort_signal and abort_signal.is_aborted():
                return content

            # Build messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ]

            # Call Model streaming interface and aggregate results
            content_parts = []
            async for chunk in self._llm_model.arun_stream(
                messages=messages, tools=None
            ):
                # Check abort signal
                if abort_signal and abort_signal.is_aborted():
                    logger.info("LLM extraction aborted")
                    return content

                if chunk.content:
                    content_parts.append(chunk.content)

            if content_parts:
                content.raw_text = "".join(content_parts)
                return content
            else:
                return content
        except Exception as e:
            logger.error(
                f"Error extracting content by query: {e}", extra={"query": query}
            )
            return content
