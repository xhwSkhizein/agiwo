"""Shared runtime-config models for Console services and API routes."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from agiwo.config.settings import ModelProvider
from agiwo.skill.allowlist import normalize_allowed_skills
from server.models.agent_config import ModelParamsInput


class DefaultAgentConfigPayload(BaseModel):
    id: str = "default-console-agent"
    name: str
    description: str = ""
    model_provider: ModelProvider
    model_name: str
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    allowed_skills: list[str] | None = None
    model_params: ModelParamsInput = Field(default_factory=ModelParamsInput)

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _validate_allowed_tools(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("allowed_tools must be a list")
        return list(value)

    @field_validator("allowed_skills", mode="before")
    @classmethod
    def _validate_allowed_skills(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("allowed_skills must be a list")
        normalized = normalize_allowed_skills(value)
        return list(normalized) if normalized is not None else None


class RuntimeConfigEditablePayload(BaseModel):
    skills_dirs: list[str] = Field(default_factory=list)
    default_agent: DefaultAgentConfigPayload

    @field_validator("skills_dirs", mode="before")
    @classmethod
    def _normalize_skills_dirs(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("skills_dirs must be a list")
        normalized: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("skills_dirs entries must be strings")
            stripped = item.strip()
            if stripped:
                normalized.append(stripped)
        return normalized


class RuntimeConfigResponse(BaseModel):
    editable: RuntimeConfigEditablePayload
    effective: dict[str, Any] = Field(default_factory=dict)
    readonly: dict[str, Any] = Field(default_factory=dict)
    runtime_only: bool = True
    restart_required: list[str] = Field(default_factory=list)


__all__ = [
    "DefaultAgentConfigPayload",
    "RuntimeConfigEditablePayload",
    "RuntimeConfigResponse",
]
