"""Agent configuration request/response models."""

from pydantic import BaseModel, Field, field_validator, model_validator

from agiwo.config.settings import ModelProvider
from agiwo.llm.config_policy import validate_provider_model_params
from server.models.agent_options import AgentOptionsInput, ModelParamsInput
from server.models.tool_reference import parse_tool_references


class AgentConfigPayload(BaseModel):
    name: str
    description: str = ""
    model_provider: ModelProvider
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @field_validator("tools", mode="before")
    @classmethod
    def _validate_tools(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("tools must be a list")
        return parse_tool_references(value)

    @model_validator(mode="after")
    def _validate_model_connection(self) -> "AgentConfigPayload":
        validate_provider_model_params(self.model_provider, self.model_params)
        return self


class AgentConfigResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: AgentOptionsInput = Field(default_factory=AgentOptionsInput)
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)
    created_at: str
    updated_at: str
