"""Web content reader with lightweight fetch and browser fallback."""

import time
from datetime import datetime
from typing import Any

from agiwo.agent.execution_context import ExecutionContext
from agiwo.llm import ModelConfig, create_model
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.html_extract import HtmlContent
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.utils import truncate_middle
from agiwo.tool.builtin.web_reader.curl_cffi import SimpleAsyncClient
from agiwo.tool.builtin.web_reader.playwright.core import PlaywrightCrawler
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceType,
    CitationStoreConfig,
    create_citation_store,
    generate_citation_id,
)
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_reader")
class WebReaderTool(BaseTool):
    """Fetch and optionally post-process web content for the agent."""

    cacheable = True

    def __init__(
        self,
        *,
        config: WebReaderApiConfig | None = None,
        citation_store_config: CitationStoreConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._config = config or WebReaderApiConfig()
        self.timeout_seconds = self._config.timeout_seconds
        self.max_length = self._config.max_content_length
        self._citation_source_store = create_citation_store(citation_store_config)
        self._model_config = ModelConfig(
            provider=self._config.model_provider,  # type: ignore[arg-type]
            model_name=self._config.model_name,
            base_url=self._config.model_base_url,
            api_key_env_name=self._config.api_key_env_name,
            temperature=self._config.model_temperature,
            top_p=self._config.model_top_p,
            max_output_tokens=self._config.model_max_tokens,
        )
        self._llm_model = None
        self._curl_cffi_client = SimpleAsyncClient(timeout=self._config.timeout_seconds)
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
        return {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Search result index from web_search (0, 1, 2...)",
                    "minimum": 0,
                },
                "url": {
                    "type": "string",
                    "description": "Direct URL.",
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

    async def _fetch_content(self, url: str) -> tuple[HtmlContent | None, str | None]:
        content = await self._curl_cffi_client.fetch(url)
        if content is not None:
            logger.info("web_reader_fetch_succeeded", strategy="curl_cffi", url=url)
            return content, None

        logger.info("web_reader_fetch_fallback_playwright", url=url)
        started = False
        try:
            await self._playwright_crawl.start()
            started = True
            return await self._playwright_crawl.crawl_url(url), None
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)
        finally:
            if started:
                await self._playwright_crawl.stop()

    async def _run_model_prompt(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback_text: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        if abort_signal and abort_signal.is_aborted():
            return fallback_text

        try:
            model = self._get_llm_model()
        except Exception as exc:  # noqa: BLE001
            logger.error("web_reader_model_init_failed", error=str(exc))
            return fallback_text

        content_parts: list[str] = []
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            async for chunk in model.arun_stream(messages=messages, tools=None):
                if abort_signal and abort_signal.is_aborted():
                    logger.info("web_reader_model_processing_aborted")
                    return fallback_text
                if chunk.content:
                    content_parts.append(chunk.content)
        except Exception as exc:  # noqa: BLE001
            logger.error("web_reader_model_processing_failed", error=str(exc))
            return fallback_text

        result = "".join(content_parts).strip()
        if result:
            return result
        return fallback_text

    def _get_llm_model(self):
        if self._llm_model is None:
            self._llm_model = create_model(self._model_config)
        return self._llm_model

    async def _summarize_content(
        self,
        text: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        system_prompt = """You are a professional content summarization assistant.
Your task is to:
1. Extract the core information and key insights from the content user provided
2. Maintain factual accuracy and preserve important details
3. Keep the summary comprehensive but concise (aim for 25-35% of original length)
"""
        user_prompt = f"""summarize the following content:

{text}
"""
        return await self._run_model_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_text=text,
            abort_signal=abort_signal,
        )

    async def _extract_by_query(
        self,
        text: str,
        query: str,
        abort_signal: AbortSignal | None = None,
    ) -> str:
        system_prompt = """You are a professional content extraction assistant.
Your task is to:
1. Extract content specifically relevant to the user's query from the content user provided
2. Maintain full context and accuracy of extracted information
3. Preserve technical details, data, and important specifics
4. If no relevant content exists, clearly state so
5. Focus on substantive information, not just keyword matches
"""
        user_prompt = f"""the content is:

{text}

user's query is:
{query}
"""
        return await self._run_model_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback_text=text,
            abort_signal=abort_signal,
        )

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        start_time = time.time()

        try:
            if abort_signal and abort_signal.is_aborted():
                return ToolResult.aborted(
                    tool_name=self.name,
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                )

            resolved = await self._resolve_request(
                parameters, context, start_time=start_time
            )
            if isinstance(resolved, ToolResult):
                return resolved

            session_id, url, summarize, search_query, existing_source = resolved

            content, fallback_error = await self._fetch_content(url)
            if content is None:
                return ToolResult.failed(
                    tool_name=self.name,
                    error=fallback_error or "Failed to fetch content",
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                )

            if abort_signal and abort_signal.is_aborted():
                return ToolResult.aborted(
                    tool_name=self.name,
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                )

            processed_content = await self._process_content(
                content.raw_text or content.text or "",
                summarize=summarize,
                search_query=search_query,
                abort_signal=abort_signal,
            )

            if len(processed_content) > self.max_length:
                processed_content = truncate_middle(processed_content, self.max_length)

            original_content_dict = content.model_dump(exclude_none=True)
            original_content_dict.pop("raw_html", None)
            original_content_dict.pop("text", None)

            citation_id = await self._store_citation_source(
                session_id=session_id,
                url=url,
                title=content.title,
                processed_content=processed_content,
                original_content=original_content_dict,
                search_query=search_query,
                summarize=summarize,
                existing_source=existing_source,
            )

            result_content = processed_content
            if citation_id:
                result_content = f"[cite:{citation_id}]\n\n{processed_content}"

            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=str(parameters.get("tool_call_id", "")),
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
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failed(
                tool_name=self.name,
                error=f"Error: {exc!s}",
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )

    async def _resolve_request(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        *,
        start_time: float,
    ) -> tuple[str, str, bool, str | None, CitationSourceRaw | None] | ToolResult:
        summarize = bool(parameters.get("summarize", False))
        search_query_value = parameters.get("search_query")
        if summarize and search_query_value:
            return ToolResult.failed(
                tool_name=self.name,
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
                    tool_name=self.name,
                    error="Error: index must be an integer",
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                )
            existing_source = await self._citation_source_store.get_source_by_index(
                session_id, index
            )
            if existing_source is None:
                return ToolResult.failed(
                    tool_name=self.name,
                    error=f"Search result with index {index} not found",
                    tool_call_id=str(parameters.get("tool_call_id", "")),
                    input_args=parameters,
                    start_time=start_time,
                )
            url = existing_source.url
            logger.info("web_reader_found_source_by_index", index=index, url=url)

        if not isinstance(url, str) or not url.strip():
            return ToolResult.failed(
                tool_name=self.name,
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
        return session_id, url.strip(), summarize, search_query, existing_source

    async def _process_content(
        self,
        content: str,
        *,
        summarize: bool,
        search_query: str | None,
        abort_signal: AbortSignal | None,
    ) -> str:
        if summarize or len(content) > self.max_length:
            return await self._summarize_content(
                content,
                abort_signal=abort_signal,
            )
        if search_query:
            return await self._extract_by_query(
                content,
                search_query,
                abort_signal=abort_signal,
            )
        return content

    async def _store_citation_source(
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
