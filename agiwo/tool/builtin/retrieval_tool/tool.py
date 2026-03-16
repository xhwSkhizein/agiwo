"""
MemoryRetrievalTool - Find relevant notes and memory from MEMORY directory.
"""

import time
from typing import Any

from agiwo.config.settings import settings
from agiwo.memory import WorkspaceMemoryService
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.context import ToolContext
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("memory_retrieval")
class MemoryRetrievalTool(BaseTool):
    """Tool for searching memories in MEMORY directory using hybrid search."""

    def __init__(
        self,
        *,
        top_k: int | None = None,
        embedding_provider: str | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self._top_k = top_k if top_k is not None else settings.memory_top_k
        self._memory_service = WorkspaceMemoryService(
            embedding_provider=embedding_provider,
        )

    def get_name(self) -> str:
        return "memory_retrieval"

    def get_description(self) -> str:
        return "Search your MEMORY directory for relevant past notes, decisions, and knowledge. Use this before answering questions about prior work, preferences, or historical context."

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant memories",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 5)",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        """Execute memory search with detailed debug logging."""
        start_time = time.time()

        if abort_signal and abort_signal.is_aborted():
            return ToolResult.aborted(
                tool_name=self.name,
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )

        query = parameters.get("query", "").strip()
        top_k = parameters.get("top_k", self._top_k)

        logger.info(
            "memory_retrieval_start",
            query=query,
            top_k=top_k,
            agent_name=context.agent_name,
        )

        if not query:
            return ToolResult.failed(
                tool_name=self.name,
                error="No query provided",
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )

        search_start = time.time()
        workspace, results = await self._memory_service.search(
            agent_name=context.agent_name,
            agent_id=context.agent_id,
            query=query,
            top_k=top_k,
        )
        if workspace is None:
            logger.warning("memory_retrieval_no_workspace", context=context)
            return ToolResult.failed(
                tool_name=self.name,
                error="Could not resolve workspace directory",
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )

        workspace_dir = workspace.workspace

        if abort_signal and abort_signal.is_aborted():
            return ToolResult.aborted(
                tool_name=self.name,
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                start_time=start_time,
            )
        search_duration_ms = (time.time() - search_start) * 1000

        logger.info(
            "memory_retrieval_search_complete",
            query=query,
            result_count=len(results),
            duration_ms=search_duration_ms,
            workspace=str(workspace_dir),
        )

        if not results:
            content = f"No memories found for query: {query}"
            logger.info(
                "memory_retrieval_no_results", query=query, workspace=str(workspace_dir)
            )
            return ToolResult.success(
                tool_name=self.name,
                tool_call_id=str(parameters.get("tool_call_id", "")),
                input_args=parameters,
                content=content,
                output={"results": [], "debug": {"workspace": str(workspace_dir)}},
                start_time=start_time,
            )

        content = self._format_results(query, results)
        output = {
            "results": [
                {
                    "path": r.path,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                    "score": r.score,
                    "text": r.text,
                }
                for r in results
            ]
        }

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=str(parameters.get("tool_call_id", "")),
            input_args=parameters,
            content=content,
            output=output,
            start_time=start_time,
        )

    def _format_results(self, query: str, results: list) -> str:
        """Format search results for LLM consumption."""
        lines = [f'## Memory Search Results for: "{query}"\n']

        for i, r in enumerate(results, 1):
            lines.append(f"### [{i}] {r.path} (lines {r.start_line}-{r.end_line})")
            lines.append(f"Score: {r.score:.2f}")
            lines.append("---")
            lines.append(r.text)
            lines.append("")

        return "\n".join(lines)
