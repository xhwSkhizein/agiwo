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
            max_tokens=self._config.model_max_tokens,
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

    def needs_permissions(self, parameters: dict[str, Any]) -> bool:
        return False

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
                return self._create_abort_result(parameters, start_time)

            summarize = bool(parameters.get("summarize", False))
            search_query = parameters.get("search_query")
            if summarize and search_query:
                return self._create_error_result(
                    parameters,
                    "Error: summarize and search_query are mutually exclusive",
                    start_time,
                )

            session_id = str(parameters.get("session_id") or context.session_id or "default")
            index = parameters.get("index")
            url = parameters.get("url")

            existing_source: CitationSourceRaw | None = None
            if index is not None:
                if not isinstance(index, int):
                    return self._create_error_result(
                        parameters,
                        "Error: index must be an integer",
                        start_time,
                    )
                existing_source = await self._citation_source_store.get_source_by_index(
                    session_id, index
                )
                if existing_source is None:
                    return self._create_error_result(
                        parameters,
                        f"Search result with index {index} not found",
                        start_time,
                    )
                url = existing_source.url
                logger.info("web_reader_found_source_by_index", index=index, url=url)

            if not isinstance(url, str) or not url.strip():
                return self._create_error_result(
                    parameters,
                    "Error: url must be a non-empty string",
                    start_time,
                )
            url = url.strip()

            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            content, fallback_error = await self._fetch_content(url)
            if content is None:
                if fallback_error:
                    return self._create_error_result(
                        parameters, fallback_error, start_time
                    )
                return self._create_error_result(
                    parameters, "Failed to fetch content", start_time
                )

            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            processed_content = content.raw_text or content.text or ""
            if summarize or len(processed_content) > self.max_length:
                processed_content = await self._summarize_content(
                    processed_content,
                    abort_signal=abort_signal,
                )
            elif isinstance(search_query, str) and search_query.strip():
                processed_content = await self._extract_by_query(
                    processed_content,
                    search_query.strip(),
                    abort_signal=abort_signal,
                )

            if len(processed_content) > self.max_length:
                processed_content = truncate_middle(processed_content, self.max_length)

            original_content_dict = content.model_dump(exclude_none=True)
            original_content_dict.pop("raw_html", None)
            original_content_dict.pop("text", None)

            citation_id: str | None = None
            try:
                if existing_source:
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
                else:
                    citation_id = generate_citation_id(prefix="reader")
                    await self._citation_source_store.store_citation_sources(
                        session_id=session_id,
                        sources=[
                            CitationSourceRaw(
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
                        ],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("citation_store_failed", error=str(exc), url=url)
                citation_id = None

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
        except Exception as exc:  # noqa: BLE001
            return self._create_error_result(parameters, f"Error: {exc!s}", start_time)
