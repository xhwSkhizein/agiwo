"""Agent hook adapter for WorkspaceMemoryService-backed retrieval."""

from collections.abc import Mapping

from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.memory import MemoryRecord
from agiwo.agent.runtime.context import RunContext
from agiwo.config.settings import get_settings
from agiwo.memory import WorkspaceMemoryService
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


def _text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 0.9

    a_words = set(a.split())
    b_words = set(b.split())
    if not a_words or not b_words:
        return 0.0

    intersection = a_words & b_words
    union = a_words | b_words
    return len(intersection) / len(union)


def filter_relevant_memories(
    messages: list[Mapping[str, object]],
    memories: list[MemoryRecord],
) -> list[MemoryRecord]:
    if not memories:
        return []

    min_relevance_score = 0.5
    similarity_threshold = 0.8

    existing_texts: list[str] = [
        content
        for message in messages[:-1]
        if isinstance(content := message.get("content"), str)
    ]

    def _is_similar_to_history(content: str) -> bool:
        content_lower = content.lower()
        for text in existing_texts:
            if _text_similarity(content_lower, text.lower()) > similarity_threshold:
                return True
        return False

    filtered: list[MemoryRecord] = []
    seen_contents: set[str] = set()

    for memory in sorted(
        [m for m in memories if m.relevance_score is not None],
        key=lambda m: m.relevance_score or 0,
        reverse=True,
    ):
        if memory.relevance_score < min_relevance_score:
            continue

        content_normalized = memory.content.strip()
        if content_normalized in seen_contents:
            continue
        seen_contents.add(content_normalized)

        if _is_similar_to_history(content_normalized):
            continue

        filtered.append(memory)

    return filtered


class DefaultMemoryHook:
    """Default memory hook implementation using WorkspaceMemoryService."""

    def __init__(
        self,
        *,
        embedding_provider: str | None = None,
        top_k: int | None = None,
        root_path: str | None = None,
    ) -> None:
        self._top_k = top_k if top_k is not None else get_settings().memory_top_k
        self._memory_service = WorkspaceMemoryService(
            root_path=root_path,
            embedding_provider=embedding_provider,
        )

    def _resolve_workspace(self, context: RunContext):
        workspace = self._memory_service.resolve_workspace(
            agent_name=getattr(context, "agent_name", None),
            agent_id=getattr(context, "agent_id", None),
        )
        if workspace is None:
            return None
        return workspace.workspace

    async def retrieve_memories(
        self, user_input: UserInput, context: RunContext
    ) -> list[MemoryRecord]:
        query = UserMessage.from_value(user_input).extract_text()
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


__all__ = ["DefaultMemoryHook", "filter_relevant_memories"]
