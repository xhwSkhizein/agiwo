"""Agent registry store abstractions and backend factory."""

from typing import Protocol

from server.config import ConsoleConfig
from server.services.agent_registry.models import AgentConfigRecord
from server.services.agent_registry.store.memory import InMemoryAgentRegistryStore
from server.services.agent_registry.store.sqlite import SqliteAgentRegistryStore


class AgentRegistryStore(Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConfigRecord]: ...
    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None: ...
    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None: ...
    async def upsert_agent(self, record: AgentConfigRecord) -> None: ...
    async def delete_agent(self, agent_id: str) -> bool: ...


def create_agent_registry_store(config: ConsoleConfig) -> AgentRegistryStore:
    if config.metadata_storage_type == "sqlite":
        return SqliteAgentRegistryStore(db_path=config.sqlite_db_path)
    return InMemoryAgentRegistryStore()


__all__ = [
    "AgentRegistryStore",
    "create_agent_registry_store",
]
