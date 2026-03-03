"""
MemoryRetrievalTool - Find relevant notes and memory from MEMORY directory.
"""

import time
from pathlib import Path
from typing import Any

from agiwo.config.settings import settings
from agiwo.agent.execution_context import ExecutionContext
from agiwo.tool.base import BaseTool, ToolResult
from agiwo.tool.builtin.config import MemoryConfig
from agiwo.tool.builtin.registry import builtin_tool, default_enable
from agiwo.tool.builtin.retrieval_tool.store import MemoryIndexStore
from agiwo.utils.abort_signal import AbortSignal
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@default_enable
@builtin_tool("memory_retrieval")
class MemoryRetrievalTool(BaseTool):
    """Tool for searching memories in MEMORY directory using hybrid search."""

    def __init__(self, *, config: MemoryConfig | None = None, **kwargs: Any):
        super().__init__()
        self._config = config or MemoryConfig()
        self._stores: dict[str, MemoryIndexStore] = {}

    def get_name(self) -> str:
        return "memory_retrieval"

    def get_description(self) -> str:
        return (
            "Search your MEMORY directory for relevant past notes, decisions, "
            "and knowledge. Use this before answering questions about prior work, "
            "preferences, or historical context."
        )

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

    def needs_permissions(self, parameters: dict[str, Any]) -> bool:
        return False

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        """Execute memory search with detailed debug logging."""
        start_time = time.time()

        if abort_signal and abort_signal.is_aborted():
            return self._create_abort_result(parameters, start_time)

        query = parameters.get("query", "").strip()
        top_k = parameters.get("top_k", self._config.top_k)

        logger.info("memory_retrieval_start", query=query, top_k=top_k, agent_name=getattr(context, "agent_name", None))

        if not query:
            return self._create_error_result(
                parameters, "No query provided", start_time
            )

        workspace_dir = self._resolve_workspace(context)
        if not workspace_dir:
            logger.warning("memory_retrieval_no_workspace", context=context)
            return self._create_error_result(
                parameters, "Could not resolve workspace directory", start_time
            )

        logger.debug("memory_retrieval_workspace", workspace=str(workspace_dir))

        store = await self._get_or_create_store(workspace_dir)

        # Sync files and log stats
        sync_start = time.time()
        await store.sync_files()
        logger.debug("memory_retrieval_sync_complete", duration_ms=(time.time() - sync_start) * 1000)

        if abort_signal and abort_signal.is_aborted():
            return self._create_abort_result(parameters, start_time)

        # Search with detailed logging
        search_start = time.time()
        results = await store.search(query, top_k)
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
            logger.info("memory_retrieval_no_results", query=query, workspace=str(workspace_dir))
            return ToolResult(
                tool_name=self.name,
                tool_call_id=parameters.get("tool_call_id", ""),
                input_args=parameters,
                content=content,
                output={"results": [], "debug": {"workspace": str(workspace_dir)}},
                start_time=start_time,
                end_time=time.time(),
                duration=time.time() - start_time,
                is_success=True,
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

        return ToolResult(
            tool_name=self.name,
            tool_call_id=parameters.get("tool_call_id", ""),
            input_args=parameters,
            content=content,
            output=output,
            start_time=start_time,
            end_time=time.time(),
            duration=time.time() - start_time,
            is_success=True,
        )

    def _resolve_workspace(self, context: ExecutionContext) -> Path | None:
        """Resolve workspace directory from context."""
        config_root = settings.get_root_path()
        agent_name = getattr(context, "agent_name", None)

        if not agent_name:
            agent_id = getattr(context, "agent_id", None)
            if agent_id:
                agent_name = agent_id

        if not agent_name:
            logger.warning("no_agent_name_in_context")
            return None

        workspace = config_root / agent_name
        return workspace

    async def _get_or_create_store(self, workspace_dir: Path) -> MemoryIndexStore:
        """Get or create a MemoryIndexStore for the workspace."""
        key = str(workspace_dir)
        if key not in self._stores:
            self._stores[key] = MemoryIndexStore(workspace_dir, self._config)
        return self._stores[key]

    def _format_results(self, query: str, results: list) -> str:
        """Format search results for LLM consumption."""
        lines = [f"## Memory Search Results for: \"{query}\"\n"]

        for i, r in enumerate(results, 1):
            lines.append(f"### [{i}] {r.path} (lines {r.start_line}-{r.end_line})")
            lines.append(f"Score: {r.score:.2f}")
            lines.append("---")
            lines.append(r.text)
            lines.append("")

        return "\n".join(lines)