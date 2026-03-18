"""In-memory backend for agent registry."""

from server.services.agent_registry.models import AgentConfigRecord


def _record_sort_key(record: AgentConfigRecord) -> tuple[object, object, str]:
    return (record.updated_at, record.created_at, record.id)


class InMemoryAgentRegistryStore:
    def __init__(self) -> None:
        self._records: dict[str, AgentConfigRecord] = {}

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConfigRecord]:
        records = sorted(
            self._records.values(),
            key=_record_sort_key,
            reverse=True,
        )
        return records[offset : offset + limit]

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        return self._records.get(agent_id)

    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None:
        records = sorted(
            self._records.values(),
            key=_record_sort_key,
            reverse=True,
        )
        for record in records:
            if record.name == agent_name:
                return record
        return None

    async def upsert_agent(self, record: AgentConfigRecord) -> None:
        self._records[record.id] = record

    async def delete_agent(self, agent_id: str) -> bool:
        return self._records.pop(agent_id, None) is not None


__all__ = ["InMemoryAgentRegistryStore"]
