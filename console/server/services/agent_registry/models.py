"""Shared models for agent registry persistence."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from agiwo.llm.config_policy import sanitize_model_params_data
from server.models import sanitize_agent_options_data


class AgentConfigRecord(BaseModel):
    """Persisted agent configuration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    tools: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)
    model_params: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="before")
    @classmethod
    def _normalize_record_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        normalized["options"] = sanitize_agent_options_data(normalized.get("options"))
        normalized["model_params"] = sanitize_model_params_data(
            normalized.get("model_params"),
            reject_plain_api_key=False,
        )
        return normalized


__all__ = ["AgentConfigRecord"]
