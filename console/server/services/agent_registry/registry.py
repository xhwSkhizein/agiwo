"""Agent registry domain service — CRUD operations and domain validation."""

from datetime import datetime

from server.config import ConsoleConfig
from server.domain.agent_configs import AgentConfigInput
from server.domain.tool_references import serialize_tool_references
from server.services.agent_registry.models import AgentConfigRecord
from server.services.agent_registry.store import (
    AgentRegistryStore,
    create_agent_registry_store,
)


def _validate_agent_config_record(record: AgentConfigRecord) -> AgentConfigRecord:
    normalized_input = AgentConfigInput(
        name=record.name,
        description=record.description,
        model_provider=record.model_provider,
        model_name=record.model_name,
        system_prompt=record.system_prompt,
        tools=record.tools,
        options=record.options,
        model_params=record.model_params,
    )
    return AgentConfigRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "tools": serialize_tool_references(normalized_input.tools),
        }
    )


class AgentRegistry:
    """CRUD operations and domain validation for agent configurations."""

    def __init__(self, config: ConsoleConfig) -> None:
        self._config = config
        self._store: AgentRegistryStore | None = None
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        self._store = create_agent_registry_store(self._config)
        await self._store.connect()
        self._initialized = True

    async def close(self) -> None:
        if self._store is not None:
            await self._store.close()
            self._store = None
        self._initialized = False

    async def list_agents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AgentConfigRecord]:
        return await self._require_store().list_agents(limit=limit, offset=offset)

    async def get_agent(self, agent_id: str) -> AgentConfigRecord | None:
        return await self._require_store().get_agent(agent_id)

    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None:
        return await self._require_store().get_agent_by_name(agent_name)

    async def create_agent(self, record: AgentConfigRecord) -> AgentConfigRecord:
        normalized = _validate_agent_config_record(record)
        await self._require_store().upsert_agent(normalized)
        return normalized

    async def replace_agent(
        self,
        agent_id: str,
        record: AgentConfigRecord,
    ) -> AgentConfigRecord | None:
        existing = await self.get_agent(agent_id)
        if existing is None:
            return None

        replacement = AgentConfigRecord.model_validate(
            {
                **record.model_dump(mode="python"),
                "id": existing.id,
                "created_at": existing.created_at,
                "updated_at": datetime.now(),
            }
        )
        normalized = _validate_agent_config_record(replacement)
        await self._require_store().upsert_agent(normalized)
        return normalized

    async def delete_agent(self, agent_id: str) -> bool:
        return await self._require_store().delete_agent(agent_id)

    def _require_store(self) -> AgentRegistryStore:
        assert self._store is not None, "AgentRegistry.initialize() must be called before use"
        return self._store


__all__ = ["AgentRegistry"]
