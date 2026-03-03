"""
WebSearchApiTool - Web search using unified API.

使用统一的 Web Search API 替代 Serper API。
"""

import time
from datetime import datetime
from typing import Any

from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.storage.citation import (
    CitationSourceRaw,
    CitationSourceRepository,
    CitationSourceSimplified,
    CitationSourceType,
    generate_citation_id,
)
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.http_client import AsyncHttpClient
from agiwo.tool.builtin.config import WebSearchApiConfig
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("web_search")
class WebSearchApiTool(BaseTool):
    """Web search tool using unified API.

    优化特性:
    - 使用统一的 API 接口，简化依赖
    - 搜索结果存储到 MongoDB (支持 TTL 自动清理)
    - 使用数字索引 (0, 1, 2...) 减少 token 消耗
    - 隐藏 URL，减少 60-70% token 消耗
    - 支持通过索引在 web_reader_api 中获取完整内容
    """

    # 启用缓存 - 搜索结果在 session 内是稳定的
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
        self._citation_source_store = citation_source_store

        # HTTP 客户端（依赖注入，遵循依赖倒置原则）
        self._http_client = AsyncHttpClient(
            base_url=self._config.base_url,
            timeout=self._config.timeout_seconds,
            max_retries=self._config.max_retries,
        )

    def get_name(self) -> str:
        """返回工具名称。"""
        return "web_search"
    def get_short_description(self) -> str:
        return """Search information tool, use when you need realtime infomations."""
    def get_description(self) -> str:
        """返回工具描述。"""
        return """This tool performs web searches and returns:
- Numbered search results (0, 1, 2...) with titles and snippets
- Publication dates (when available)
- Source information

Use this tool when you need to:
- Find current information not in your training data
- Research topics or gather facts
- Discover relevant web resources
- Get multiple perspectives on a topic

To fetch full content from a search result, use web_reader tool with the result index:
> web_reader(index=0) => Fetch content from the first search result"""

    def get_parameters(self) -> dict[str, Any]:
        """返回工具参数 JSON Schema。"""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "search query",
                },
            },
            "required": ["query"],
        }

    def is_concurrency_safe(self) -> bool:
        """工具是否支持并发。"""
        return True

    def needs_permissions(self, parameters: dict[str, Any]) -> bool:
        """检查工具是否需要权限。"""
        # Web 搜索是低风险操作
        return False

    async def _call_search_api(self, query: str) -> list[dict]:
        """调用搜索 API。

        Args:
            query: 搜索查询

        Returns:
            搜索结果列表

        Raises:
            Exception: API 调用失败
        """
        try:
            response = await self._http_client.post_json(
                endpoint="/api/ai/web-search",
                data={
                    "query": query,
                    "num": self.max_results,
                    "recency_days": self._config.recency_days,
                },
            )

            # 提取结果
            results = response.get("results", [])
            logger.info(
                "search_api_success",
                query=query,
                result_count=len(results),
            )
            return results

        except Exception as e:
            logger.error(
                "search_api_failed",
                query=query,
                error=str(e),
            )
            raise

    def _convert_api_results_to_citations(
        self,
        api_results: list[dict],
        query: str,
        session_id: str,
        start_index: int = 0,
    ) -> list[CitationSourceRaw]:
        """将 API 结果转换为 CitationSourceRaw 对象。

        遵循单一职责原则，专门负责结果转换。

        Args:
            api_results: API 返回的原始结果
            query: 搜索查询
            session_id: 会话 ID
            start_index: 起始索引

        Returns:
            CitationSourceRaw 对象列表
        """
        citation_sources = []
        for offset, item in enumerate(api_results):
            citation_id = generate_citation_id(prefix="search")
            citation_sources.append(
                CitationSourceRaw(
                    citation_id=citation_id,
                    session_id=session_id,
                    source_type=CitationSourceType.SEARCH,
                    url=item.get("url", ""),
                    title=item.get("name", "No title"),
                    snippet=item.get("snippet", ""),
                    date_published=item.get("date", ""),
                    source=item.get("host_name", ""),
                    query=query,
                    index=start_index + offset,
                    created_at=datetime.now(),
                )
            )
        return citation_sources

    def _format_simplified_results(
        self,
        query: str,
        simplified_results: list[CitationSourceSimplified],
        citation_ids: list[str],
    ) -> str:
        """格式化简化结果（不包含 URL，使用数字索引和引用标记）。

        复用格式化逻辑，遵循开闭原则。

        Args:
            query: 搜索查询
            simplified_results: 简化的结果列表
            citation_ids: 引用 ID 列表

        Returns:
            格式化的结果文本
        """
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
        api_results: list[dict],
    ) -> str:
        """回退格式化（包含 URL，存储失败时使用）。

        Args:
            query: 搜索查询
            api_results: API 原始结果

        Returns:
            格式化的结果文本
        """
        content_parts = [
            f"A web search for '{query}' found {len(api_results)} results:",
            "\n## Web Results\n",
        ]

        for idx, item in enumerate(api_results):
            title = item.get("name", "No title")
            url = item.get("url", "")
            parts = [f"{idx}. [{title}]({url})"]
            if item.get("date"):
                parts.append(f"\nDate published: {item['date']}")
            if item.get("host_name"):
                parts.append(f"\nSource: {item['host_name']}")
            if item.get("snippet"):
                parts.append(f"\n{item['snippet']}")

            content_parts.append("".join(parts))

        return "\n\n".join(content_parts)

    async def _process_and_format_results(
        self,
        query: str,
        api_results: list[dict],
        session_id: str,
    ) -> tuple[str, list[str]]:
        """处理并格式化搜索结果（存储 + 格式化，带回退）。

        Args:
            query: 搜索查询
            api_results: API 原始结果
            session_id: 会话 ID

        Returns:
            (格式化结果文本, citation_id 列表) 元组
        """
        # 尝试存储并返回简化格式
        if self._citation_source_store:
            try:
                # 获取当前 session 的最大索引
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

                # 转换为 CitationSourceRaw 对象
                citation_sources = self._convert_api_results_to_citations(
                    api_results, query, session_id, start_index
                )

                # 存储并获取 citation_ids
                citation_ids = await self._citation_source_store.store_citation_sources(
                    session_id=session_id,
                    sources=citation_sources,
                )

                # 获取简化结果
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
                # 回退: 返回原始格式
                return self._format_fallback_results(query, api_results), []
        else:
            # 没有 store 实例: 返回原始格式
            return self._format_fallback_results(query, api_results), []

    async def execute(
        self,
        parameters: dict[str, Any],
        context: "ExecutionContext",
        abort_signal: "AbortSignal | None" = None,
    ) -> ToolResult:
        """执行 web 搜索。"""
        start_time = time.time()

        try:
            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 获取参数
            session_id = parameters.get("session_id", "default")
            query = parameters.get("query")

            if not query:
                return self._create_error_result(
                    parameters, "Error: No query provided", start_time
                )

            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 调用搜索 API
            api_results = await self._call_search_api(query)

            if not api_results:
                return self._create_error_result(
                    parameters,
                    f"No results found for query: '{query}'. Try a different query.",
                    start_time,
                )

            # 检查中止信号
            if abort_signal and abort_signal.is_aborted():
                return self._create_abort_result(parameters, start_time)

            # 处理并格式化结果
            result_text, citation_ids = await self._process_and_format_results(
                query=query,
                api_results=api_results,
                session_id=session_id,
            )

            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=result_text,
                output={
                    "query": query,
                    "result_count": len(api_results),
                    "citation_ids": citation_ids,
                },
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                is_success=True,
            )

        except Exception as exc:
            error_message = str(exc) if str(exc) else "Unknown error occurred"
            return self._create_error_result(
                parameters, f"Error performing web search: {error_message}", start_time
            )
