"""
WebReaderApiTool - Web content reader using unified API.

使用统一的 Web Reader API 替代 curl_cffi 和 Playwright。
"""

import time
from datetime import datetime
from typing import Any

from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceRepository,
    CitationSourceType,
    generate_citation_id,
)
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.http_client import AsyncHttpClient
from agiwo.tool.builtin.config import WebReaderApiConfig
from agiwo.tool.builtin.html_extract import (
    HtmlContent,
    extract_content_from_html,
)
from agiwo.tool.builtin.utils import truncate_middle
from agiwo.tool.builtin.llm_client import LlmApiClient
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_reader")
class WebReaderApiTool(BaseTool):
    """Web content reader tool using unified API.

    核心特性:
    1. 使用统一的 Web Reader API 获取内容，简化依赖
    2. 使用 trafilatura 提取主要内容
    3. 支持 LLM 总结和查询提取（可选）
    4. 支持 Citation 系统（可选）
    """

    # 启用缓存 - 网页内容在 session 内是稳定的
    cacheable = True

    def __init__(
        self,
        *,
        config: WebReaderApiConfig | None = None,
        citation_source_store: "CitationSourceRepository | None" = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._config = config or WebReaderApiConfig()
        self._citation_source_store = citation_source_store

        # 核心配置
        self.timeout_seconds = self._config.timeout_seconds
        self.max_length = self._config.max_content_length

        # HTTP 客户端和 LLM 客户端（依赖注入）
        self._http_client = AsyncHttpClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
            max_retries=self._config.max_retries,
        )
        self._llm_client = LlmApiClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
            max_retries=self._config.max_retries,
        )

    def get_name(self) -> str:
        return "web_reader"
    def get_short_description(self) -> str:
        return """Fetch web content into text only format, use when you need to read a weburl."""
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
                    "type": "string",
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

    async def _fetch_content_from_api(self, url: str) -> HtmlContent | None:
        """从 API 获取网页内容。

        Args:
            url: 网页 URL

        Returns:
            提取的 HTML 内容，失败返回 None
        """
        try:
            response = await self._http_client.post_json(
                endpoint="/api/ai/web-reader",
                data={"url": url},
            )

            # 提取 HTML
            html = response.get("data", {}).get("html", "")
            if not html:
                logger.warning("web_reader_api_empty_html", url=url)
                return None

            # 使用 extract_content_from_html 提取结构化内容
            content = extract_content_from_html(html=html, original_url=url)

            if content:
                logger.info(
                    "web_reader_api_success",
                    url=url,
                    title=content.title,
                    content_length=len(content.text or ""),
                )
            else:
                logger.warning("web_reader_api_extraction_failed", url=url)

            return content

        except Exception as e:
            logger.error("web_reader_api_failed", url=url, error=str(e))
            return None

    async def execute(
        self,
        parameters: dict[str, Any],
        context: "ExecutionContext",
        abort_signal: "AbortSignal | None" = None,
    ) -> ToolResult:
        """执行 web 内容获取。"""
        start_time = time.time()

        try:
            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 获取参数
            session_id = parameters.get("session_id", "default")
            index = parameters.get("index")
            url = parameters.get("url")
            search_query = parameters.get("search_query")
            summarize = parameters.get("summarize", False)

            # 通过 index 获取 URL（如果有 store）
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

            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 从 API 获取内容
            content = await self._fetch_content_from_api(url)

            if not content:
                return self._create_error_result(
                    parameters, "Failed to fetch content", start_time
                )

            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 内容处理: 总结或查询提取
            processed_text = content.raw_text or content.text or ""

            if summarize or len(processed_text) > self.max_length:
                processed_text = await self._llm_client.summarize(
                    text=processed_text,
                    abort_signal=abort_signal,
                )
            elif search_query:
                processed_text = await self._llm_client.extract_by_query(
                    text=processed_text,
                    query=search_query,
                    abort_signal=abort_signal,
                )

            # 截断内容
            if len(processed_text) > self.max_length:
                processed_text = truncate_middle(processed_text, self.max_length)

            # 准备元数据
            original_content_dict = content.model_dump(exclude_none=True)
            original_content_dict.pop("raw_html", None)
            original_content_dict.pop("text", None)

            # 存储/更新 citation（如果有 store）
            citation_id = None
            if self._citation_source_store:
                try:
                    if existing_source:
                        # 更新现有记录
                        citation_id = existing_source.citation_id
                        await self._citation_source_store.update_citation_source(
                            citation_id=citation_id,
                            session_id=session_id,
                            updates={
                                "full_content": processed_text,
                                "processed_content": processed_text,
                                "original_content": original_content_dict,
                                "parameters": {
                                    "search_query": search_query,
                                    "summarize": summarize,
                                },
                            },
                        )
                        logger.info(f"Updated citation {citation_id}")
                    else:
                        # 创建新记录
                        citation_id = generate_citation_id(prefix="reader")
                        new_source = CitationSourceRaw(
                            citation_id=citation_id,
                            session_id=session_id,
                            source_type=CitationSourceType.DIRECT_URL,
                            url=url,
                            title=content.title,
                            full_content=processed_text,
                            processed_content=processed_text,
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

            # 格式化结果
            result_content = processed_text
            if citation_id:
                result_content = f"[cite:{citation_id}]\n\n{processed_text}"

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=result_content,
                output={
                    "url": url,
                    "title": content.title,
                    "content_length": len(processed_text),
                    "citation_id": citation_id,
                    "truncated": len(processed_text) >= self.max_length,
                },
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                is_success=True,
            )

        except Exception as e:
            return self._create_error_result(parameters, f"Error: {e!s}", start_time)
