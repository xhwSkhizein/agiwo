"""Console domain inputs for agent configuration validation."""

from pydantic import BaseModel, Field, field_validator, model_validator

from agiwo.agent.options import AgentOptionsInput
from agiwo.llm.config_policy import validate_provider_model_params
from agiwo.llm.factory import ModelParamsInput
from agiwo.config.settings import ModelProvider
from server.domain.tool_references import ToolReference, parse_tool_references


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


__all__ = ["AgentConfigInput"]
