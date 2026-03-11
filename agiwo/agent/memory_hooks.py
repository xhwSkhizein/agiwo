"""
Default memory hooks for automatic MEMORY injection.

Provides automatic memory retrieval hook using HybridSearcher.
"""

from pathlib import Path

from agiwo.config.settings import settings
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.memory_types import MemoryRecord
from agiwo.tool.builtin.retrieval_tool.store import MemoryIndexStore
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class DefaultMemoryHook:
    """Default memory hook implementation using MemoryIndexStore."""

    _stores: dict[str, MemoryIndexStore] = {}

    def __init__(
        self,
        *,
        embedding_provider: str | None = None,
        top_k: int | None = None,
    ):
        self._embedding_provider = embedding_provider
        self._top_k = top_k if top_k is not None else settings.memory_top_k

    async def retrieve_memories(
        self, user_input: UserInput, context: ExecutionContext
    ) -> list[MemoryRecord]:
        """
        Retrieve relevant memories from MEMORY directory.

        This hook is called automatically by Agent if no custom
        on_memory_retrieve hook is provided.
        """
        query = extract_text(user_input)
        if not query or len(query.strip()) < 3:
            return []

        agent_id = getattr(context, "agent_id", None)
        workspace_dir = self._resolve_workspace(context)
        if not workspace_dir:
            logger.debug("memory_retrieve_no_workspace", agent_id=agent_id)
            return []

        store = await self._get_or_create_store(workspace_dir)

        # Sync files and search
        try:
            await store.sync_files()
            results = await store.search(query, top_k=self._top_k)
        except Exception as e:  # noqa: BLE001 - memory retrieval boundary
            logger.warning("memory_retrieve_error", error=str(e), query=query[:50])
            return []

        if not results:
            return []

        # Convert SearchResult to MemoryRecord
        records: list[MemoryRecord] = []
        for r in results:
            content = f"[{r.path}:{r.start_line}-{r.end_line}] {r.text}"
            records.append(
                MemoryRecord(
                    content=content,
                    relevance_score=r.score,
                    source=r.path,
                    metadata={
                        "chunk_id": r.chunk_id,
                        "start_line": r.start_line,
                        "end_line": r.end_line,
                        "vector_score": r.vector_score,
                        "bm25_score": r.bm25_score,
                    },
                )
            )

        logger.debug(
            "memory_retrieved",
            query=query[:50],
            count=len(records),
            agent_id=agent_id,
        )
        return records

    def _resolve_workspace(self, context: ExecutionContext) -> Path | None:
        """Resolve workspace directory from context using agent_name for shared memory."""
        config_root = settings.get_root_path()
        agent_name = context.agent_name

        if not agent_name:
            return None

        workspace = config_root / agent_name
        return workspace

    async def _get_or_create_store(self, workspace_dir: Path) -> MemoryIndexStore:
        """Get or create a MemoryIndexStore for the workspace."""
        key = str(workspace_dir)
        if key not in self._stores:
            store_kwargs = {}
            if self._embedding_provider:
                store_kwargs["embedding_provider"] = self._embedding_provider
            self._stores[key] = MemoryIndexStore(workspace_dir, **store_kwargs)
        return self._stores[key]


def create_default_memory_hooks(
    *,
    embedding_provider: str | None = None,
    top_k: int | None = None,
) -> "AgentHooks":
    """
    Create AgentHooks with default memory retrieve hook.

    Usage:
        hooks = create_default_memory_hooks()
        agent = Agent(..., hooks=hooks)

    Or merge with existing hooks:
        hooks = create_default_memory_hooks()
        hooks.on_after_run = my_custom_hook
        agent = Agent(..., hooks=hooks)
    """
    memory_hook = DefaultMemoryHook(
        embedding_provider=embedding_provider,
        top_k=top_k,
    )

    return AgentHooks(
        on_memory_retrieve=memory_hook.retrieve_memories,
    )


__all__ = [
    "DefaultMemoryHook",
    "create_default_memory_hooks",
]
