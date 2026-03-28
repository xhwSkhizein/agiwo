"""Console domain inputs for agent configuration validation."""

from pydantic import BaseModel, Field, field_validator, model_validator

from agiwo.agent import AgentOptions, RunStepStorageConfig, TraceStorageConfig
from agiwo.config.settings import ModelProvider, settings
from agiwo.llm.config_policy import (
    sanitize_model_params_data,
    validate_provider_model_params,
)
from agiwo.skill.config import normalize_skill_dirs
from server.domain.tool_references import ToolReference, parse_tool_references


def sanitize_agent_options_data(
    data: object,
    *,
    preserve_non_dict: bool = False,
) -> object:
    if not isinstance(data, dict):
        return data if preserve_non_dict else {}
    sanitized = dict(data)
    skills_dirs = normalize_skill_dirs(sanitized.get("skills_dirs"))
    if skills_dirs is None:
        sanitized.pop("skills_dirs", None)
    else:
        sanitized["skills_dirs"] = skills_dirs
    return sanitized


class AgentOptionsInput(AgentOptions):
    max_steps: int = Field(default=50, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        return sanitize_agent_options_data(data, preserve_non_dict=True)

    def to_agent_options(
        self,
        *,
        run_step_storage: RunStepStorageConfig,
        trace_storage: TraceStorageConfig,
    ) -> AgentOptions:
        data = self.model_dump(exclude_none=True)
        data["run_step_storage"] = run_step_storage
        data["trace_storage"] = trace_storage
        return AgentOptions(**data)


class ModelParamsInput(BaseModel):
    model_config = {"extra": "ignore"}

    base_url: str | None = None
    api_key_env_name: str | None = None
    max_output_tokens: int = Field(default=4096, ge=1)
    max_context_window: int = Field(default=200000, ge=1)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    frequency_penalty: float = Field(default=0.0, ge=-2, le=2)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2)
    cache_hit_price: float = Field(default=0.0, ge=0)
    input_price: float = Field(default=0.0, ge=0)
    output_price: float = Field(default=0.0, ge=0)
    aws_region: str | None = None
    aws_profile: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_plain_api_key(cls, data: object) -> object:
        return sanitize_model_params_data(
            data,
            preserve_non_dict=True,
            drop_null_keys=False,
        )


class AgentConfigInput(BaseModel):
    """Full agent configuration validated at the console domain boundary."""

    name: str
    description: str = ""
    model_provider: ModelProvider
    model_name: str
    system_prompt: str = ""
    tools: list[ToolReference] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @field_validator("tools", mode="before")
    @classmethod
    def _normalize_tools(cls, value: object) -> list[ToolReference]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("tools must be a list")
        return parse_tool_references(value)

    @model_validator(mode="after")
    def _validate_model_connection(self) -> "AgentConfigInput":
        validate_provider_model_params(self.model_provider, self.model_params)
        return self


__all__ = [
    "AgentConfigInput",
    "AgentOptionsInput",
    "ModelParamsInput",
    "sanitize_agent_options_data",
]
