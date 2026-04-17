"""Shared helpers for building Console default agent records."""

from server.config import DefaultAgentConfig
from server.models.agent_config import AgentOptionsInput, ModelParamsInput

from agiwo.skill.manager import get_global_skill_manager

from server.services.agent_registry.models import AgentConfigRecord


def build_default_agent_record(template: DefaultAgentConfig) -> AgentConfigRecord:
    """Build a normalized default agent record from Console config."""
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
        allowed_tools=(
            list(template.allowed_tools) if template.allowed_tools is not None else None
        ),
        allowed_skills=allowed_skills,
        options=AgentOptionsInput.model_validate({}).model_dump(exclude_none=True),
        model_params=ModelParamsInput.model_validate(
            template.model_params or {}
        ).model_dump(exclude_none=True),
    )


__all__ = ["build_default_agent_record"]
