"""Agent registry domain service — CRUD operations and domain validation."""

from datetime import datetime

from agiwo.llm.config_policy import validate_provider_model_params
from agiwo.skill.manager import get_global_skill_manager
from pydantic import BaseModel, Field, field_validator, model_validator

from server.config import ConsoleConfig
from server.models.agent_config import AgentOptionsInput, ModelParamsInput
from server.services.agent_registry.models import AgentConfigRecord
from server.services.agent_registry.store import (
    AgentRegistryStore,
    create_agent_registry_store,
)


class _ValidatedAgentConfig(BaseModel):
    model_provider: str
    model_name: str
    allowed_tools: list[str] | None = None
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _validate_allowed_tools(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise TypeError("allowed_tools must be a list")
        return list(value)

    @model_validator(mode="after")
    def _validate_model_connection(self) -> "_ValidatedAgentConfig":
        validate_provider_model_params(self.model_provider, self.model_params)
        return self


def _validate_agent_config_record(record: AgentConfigRecord) -> AgentConfigRecord:
    validated = _ValidatedAgentConfig(
        model_provider=record.model_provider,
        model_name=record.model_name,
        allowed_tools=record.allowed_tools,
        options=AgentOptionsInput.model_validate(record.options or {}),
        model_params=ModelParamsInput.model_validate(record.model_params or {}),
    )
    return AgentConfigRecord.model_validate(
        {
            **record.model_dump(mode="python"),
            "allowed_tools": validated.allowed_tools,
            "options": validated.options.model_dump(exclude_none=True),
            "model_params": validated.model_params.model_dump(exclude_none=True),
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
        store = self._require_store()
        default_id = self._config.default_agent.id
        persisted_default = await store.get_agent(default_id)
        if persisted_default is not None:
            return await store.list_agents(limit=limit, offset=offset)

        if limit <= 0:
            return []

        if offset == 0:
            records = await store.list_agents(limit=max(limit - 1, 0), offset=0)
            return [self._build_default_agent_record(), *records]

        return await store.list_agents(limit=limit, offset=offset - 1)

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
        allowed_skills = get_global_skill_manager().expand_allowed_skills(
            template.allowed_skills
        )
        return AgentConfigRecord(
            id=template.id,
            name=template.name,
            description=template.description,
            model_provider=template.model_provider,
            model_name=template.model_name,
            system_prompt=template.system_prompt,
            allowed_tools=template.allowed_tools,
            allowed_skills=allowed_skills,
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
