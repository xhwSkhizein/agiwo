"""Agent registry domain service — CRUD operations and domain validation."""

from datetime import datetime

from server.config import ConsoleConfig
from server.models import AgentConfigPayload, AgentOptionsInput, ModelParamsInput
from server.services.agent_registry.models import AgentConfigRecord
from server.services.agent_registry.store import (
    AgentRegistryStore,
    create_agent_registry_store,
)


def _validate_agent_config_record(record: AgentConfigRecord) -> AgentConfigRecord:
    payload = AgentConfigPayload(
        name=record.name,
        description=record.description,
        model_provider=record.model_provider,
        model_name=record.model_name,
        system_prompt=record.system_prompt,
        tools=record.tools,
        options=AgentOptionsInput.model_validate(record.options or {}),
        model_params=ModelParamsInput.model_validate(record.model_params or {}),
    )
    return AgentConfigRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "tools": payload.tools,
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
        """Get agent by id. Returns default agent from .env if not in DB."""
        record = await self._require_store().get_agent(agent_id)
        if record is not None:
            return record

        # DB 中没有，检查是否是默认 Agent
        if agent_id == self._config.default_agent.id:
            return self._build_default_agent_record()

        return None

    def _build_default_agent_record(self) -> AgentConfigRecord:
        """Build default agent from .env config (not persisted to DB)."""
        template = self._config.default_agent
        return AgentConfigRecord(
            id=template.id,
            name=template.name,
            description=template.description,
            model_provider=template.model_provider,
            model_name=template.model_name,
            system_prompt=template.system_prompt,
            tools=list(template.tools),
            options=AgentOptionsInput.model_validate({}).model_dump(exclude_none=True),
            model_params=ModelParamsInput.model_validate(
                template.model_params or {}
            ).model_dump(exclude_none=True),
        )

    async def get_agent_by_name(self, agent_name: str) -> AgentConfigRecord | None:
        """Get agent by name. Returns default agent from .env if not in DB."""
        record = await self._require_store().get_agent_by_name(agent_name)
        if record is not None:
            return record

        # DB 中没有，检查是否是默认 Agent 的名称
        if agent_name == self._config.default_agent.name:
            return self._build_default_agent_record()

        return None

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
        assert self._store is not None, (
            "AgentRegistry.initialize() must be called before use"
        )
        return self._store


__all__ = ["AgentRegistry"]
