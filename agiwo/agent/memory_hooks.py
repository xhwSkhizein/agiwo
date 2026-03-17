"""Default memory hooks for automatic MEMORY injection."""

from agiwo.config.settings import settings
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.memory_types import MemoryRecord
from agiwo.agent.runtime import AgentContext
from agiwo.memory import WorkspaceMemoryService
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class DefaultMemoryHook:
    """Default memory hook implementation using WorkspaceMemoryService."""

    def __init__(
        self,
        *,
        embedding_provider: str | None = None,
        top_k: int | None = None,
        root_path: str | None = None,
    ) -> None:
        self._top_k = top_k if top_k is not None else settings.memory_top_k
        self._memory_service = WorkspaceMemoryService(
            root_path=root_path,
            embedding_provider=embedding_provider,
        )

    def _resolve_workspace(self, context: AgentContext):
        workspace = self._memory_service.resolve_workspace(
            agent_name=getattr(context, "agent_name", None),
            agent_id=getattr(context, "agent_id", None),
        )
        if workspace is None:
            return None
        return workspace.workspace

    async def retrieve_memories(
        self, user_input: UserInput, context: AgentContext
    ) -> list[MemoryRecord]:
        """
        Retrieve relevant memories from MEMORY directory.

        This hook is called automatically by Agent if no custom
        on_memory_retrieve hook is provided.
        """
        query = extract_text(user_input)
        if not query or len(query.strip()) < 3:
            return []

        try:
            workspace, results = await self._memory_service.search(
                agent_name=context.agent_name,
                agent_id=context.agent_id,
                query=query,
                top_k=self._top_k,
            )
        except Exception as error:  # noqa: BLE001 - memory retrieval boundary
            logger.warning("memory_retrieve_error", error=str(error), query=query[:50])
            return []

        if workspace is None:
            logger.debug("memory_retrieve_no_workspace", agent_id=context.agent_id)
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
            agent_id=context.agent_id,
        )
        return records


def create_default_memory_hooks(
    *,
    embedding_provider: str | None = None,
    top_k: int | None = None,
    root_path: str | None = None,
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
        root_path=root_path,
    )

    return AgentHooks(
        on_memory_retrieve=memory_hook.retrieve_memories,
    )


__all__ = [
    "DefaultMemoryHook",
    "create_default_memory_hooks",
]
