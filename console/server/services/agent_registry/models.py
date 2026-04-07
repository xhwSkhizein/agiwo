"""Shared models for agent registry persistence."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from agiwo.llm.config_policy import sanitize_model_params_data
from agiwo.skill.allowlist import normalize_allowed_skills
from agiwo.skill.manager import get_global_skill_manager
from agiwo.tool.manager import get_global_tool_manager
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
        # Validate allowed_tools if provided (allow agent: prefix for agent tool references)
        if normalized.get("allowed_tools") is not None:
            tool_manager = get_global_tool_manager()
            allowed_tools = list(normalized.get("allowed_tools"))
            # Separate builtin tool names from agent tool references
            builtin_tools = [t for t in allowed_tools if not t.startswith("agent:")]
            agent_refs = [t for t in allowed_tools if t.startswith("agent:")]
            # Validate agent references have non-empty agent ID
            for ref in agent_refs:
                agent_id = ref[len("agent:") :].strip()
                if not agent_id:
                    raise ValueError(f"Invalid tool reference: {ref!r}")
            # Only validate builtin tool names
            if builtin_tools:
                validated_builtin = tool_manager.normalize_allowed_tools(builtin_tools)
                normalized["allowed_tools"] = list(validated_builtin) + agent_refs
            else:
                normalized["allowed_tools"] = agent_refs
        return normalized


__all__ = ["AgentConfigRecord"]
