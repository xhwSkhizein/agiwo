from pathlib import Path

from agiwo.config.settings import get_settings
from agiwo.memory.index_store import MemoryIndexStore
from agiwo.memory.searcher import SearchResult
from agiwo.utils.logging import get_logger
from agiwo.workspace import (
    AgentWorkspace,
    WorkspaceBootstrapper,
    build_agent_workspace,
)

logger = get_logger(__name__)


class WorkspaceMemoryService:
    """Shared MEMORY search service used by hooks and builtin tools."""

    def __init__(
        self,
        *,
        root_path: str | Path | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        self._root_path = root_path or get_settings().get_root_path()
        self._embedding_provider = embedding_provider
        self._stores: dict[str, MemoryIndexStore] = {}
        self._bootstrapper = WorkspaceBootstrapper()

    def resolve_workspace(
        self,
        *,
        agent_name: str | None,
        agent_id: str | None = None,
    ) -> AgentWorkspace | None:
        resolved_name = agent_name or agent_id
        if not resolved_name:
            return None
        resolved_id = agent_id or resolved_name
        return build_agent_workspace(
            root_path=self._root_path,
            agent_name=resolved_name,
            agent_id=resolved_id,
        )

    async def search(
        self,
        *,
        agent_name: str | None,
        agent_id: str | None,
        query: str,
        top_k: int,
    ) -> tuple[AgentWorkspace | None, list[SearchResult]]:
        workspace = self.resolve_workspace(
            agent_name=agent_name,
            agent_id=agent_id,
        )
        if workspace is None:
            return None, []

        await self._bootstrapper.ensure_memory_ready(workspace)
        store = await self._get_or_create_store(workspace.workspace)
        await store.sync_files()
        return workspace, await store.search(query, top_k)

    async def _get_or_create_store(self, workspace_dir: Path) -> MemoryIndexStore:
        key = str(workspace_dir)
        if key not in self._stores:
            store_kwargs = {}
            if self._embedding_provider:
                store_kwargs["embedding_provider"] = self._embedding_provider
            self._stores[key] = MemoryIndexStore(workspace_dir, **store_kwargs)
        return self._stores[key]
