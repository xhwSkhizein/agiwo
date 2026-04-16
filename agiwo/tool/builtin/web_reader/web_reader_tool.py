"""Web content reader with lightweight fetch and browser fallback."""

import time
from typing import Any

from agiwo.llm import ModelSpec
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.utils import truncate_middle
from agiwo.tool.builtin.web_reader.browser_cli_adapter import BrowserCliAdapter
from agiwo.tool.builtin.web_reader.citation_writer import WebReaderCitationWriter
from agiwo.tool.builtin.web_reader.content_fetcher import WebContentFetcher
from agiwo.tool.builtin.web_reader.content_processor import WebContentProcessor
from agiwo.tool.builtin.web_reader.curl_cffi import SimpleAsyncClient
from agiwo.tool.builtin.web_reader.request_resolver import resolve_web_reader_request
from agiwo.tool.storage.citation import (
    CitationStoreConfig,
    create_citation_store,
)
from agiwo.utils.abort_signal import AbortSignal


@default_enable
@builtin_tool("web_reader")
class WebReaderTool(BaseTool):
    """Fetch and optionally post-process web content for the agent."""

    name = "web_reader"
    description = """Fetch web content as plain text. Modes: 1. index=0 uses web search results (recommended) 2. url=https://... for direct URL.
Options are mutually exclusive: search_query returns relevant content, summarize generates concise summary."""
    cacheable = True
    is_stateless = True

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
        self._model_config = ModelSpec(
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
        self._browser_cli_adapter = BrowserCliAdapter(config=self._config)
        self._content_fetcher = WebContentFetcher(
            client=self._curl_cffi_client,
            browser_cli_adapter=self._browser_cli_adapter,
        )
        self._content_processor = WebContentProcessor(
            model_config=self._model_config,
            max_length=self.max_length,
        )
        self._citation_writer = WebReaderCitationWriter(
            citation_source_store=self._citation_source_store
        )

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

            resolved = await resolve_web_reader_request(
                parameters=parameters,
                context=context,
                citation_source_store=self._citation_source_store,
                tool_name=self.name,
                start_time=start_time,
            )
            if isinstance(resolved, ToolResult):
                return resolved

            session_id = resolved.session_id
            url = resolved.url
            summarize = resolved.summarize
            search_query = resolved.search_query
            existing_source = resolved.existing_source

            content, fallback_error = await self._content_fetcher.fetch(url)
            if content is None:
                return ToolResult.failed(
                    tool_name=self.name,
                    error=fallback_error or "Failed to fetch content",
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

            processed_content = await self._content_processor.process(
                content.raw_text or content.text or "",
                summarize=summarize,
                search_query=search_query,
                abort_signal=abort_signal,
                llm_model=self._llm_model,
            )
            self._llm_model = self._content_processor.llm_model

            if len(processed_content) > self.max_length:
                processed_content = truncate_middle(processed_content, self.max_length)

            original_content_dict = content.model_dump(exclude_none=True)
            original_content_dict.pop("raw_html", None)
            original_content_dict.pop("text", None)

            citation_id = await self._citation_writer.store(
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
                tool_call_id=context.tool_call_id,
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
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
