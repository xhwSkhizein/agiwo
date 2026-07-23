from typing import TYPE_CHECKING

from agiwo.memory.chunker import MemoryChunk, MemoryChunker
from agiwo.memory.index_store import MemoryIndexStore
from agiwo.memory.searcher import HybridSearcher, SearchResult
from agiwo.memory.service import WorkspaceMemoryService

if TYPE_CHECKING:
    from agiwo.memory.defaults import DefaultMemoryHook, filter_relevant_memories

__all__ = [
    "DefaultMemoryHook",
    "HybridSearcher",
    "MemoryChunk",
    "MemoryChunker",
    "MemoryIndexStore",
    "SearchResult",
    "WorkspaceMemoryService",
    "filter_relevant_memories",
]


def __getattr__(name: str) -> object:
    # Lazy: defaults imports agent models; eager export would cycle when tools
    # load ``from agiwo.memory import WorkspaceMemoryService``.
    if name in {"DefaultMemoryHook", "filter_relevant_memories"}:
        from agiwo.memory.defaults import (  # noqa: PLC0415
            DefaultMemoryHook,
            filter_relevant_memories,
        )

        exports = {
            "DefaultMemoryHook": DefaultMemoryHook,
            "filter_relevant_memories": filter_relevant_memories,
        }
        value = exports[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
