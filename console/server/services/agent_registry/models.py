"""Shared models for agent registry persistence."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from agiwo.llm.config_policy import sanitize_model_params_data
from agiwo.skill.allowlist import normalize_allowed_skills
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.manager import get_global_tool_manager
from agiwo.tool.reference import AgentToolReference
from server.models.agent_config import sanitize_agent_options_data


class AgentConfigRecord(BaseModel):
    """Persisted agent configuration."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    model_provider: str
    model_name: str
    system_prompt: str = ""
    allowed_tools: list[str] | None = (
        None  # Allowed builtin tool names (None = all defaults)
    )
    allowed_skills: list[str] | None = None
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
        normalized["allowed_skills"] = (
            get_global_skill_manager().validate_explicit_allowed_skills(
                list(normalize_allowed_skills(normalized.get("allowed_skills")) or ())
                if normalized.get("allowed_skills") is not None
                else None
            )
        )
        if normalized.get("allowed_tools") is not None:
            # Validate agent: references before parsing
            for name in normalized["allowed_tools"]:
                if isinstance(name, str) and name.startswith("agent:"):
                    agent_ref = AgentToolReference.parse(name)
                    if agent_ref is None or not agent_ref.agent_id:
                        raise ValueError("Empty agent id in agent: reference")
            refs = get_global_tool_manager().parse_allowed_tools(
                list(normalized["allowed_tools"])
            )
            normalized["allowed_tools"] = [str(ref) for ref in refs] if refs else []
        return normalized


__all__ = ["AgentConfigRecord"]
