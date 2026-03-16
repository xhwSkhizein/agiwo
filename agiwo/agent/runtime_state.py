from dataclasses import dataclass

from agiwo.agent.hooks import AgentHooks
from agiwo.agent.prompt import AgentPromptRuntime
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.observability.base import BaseTraceStorage
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool


@dataclass
class AgentRuntimeState:
    """Owned runtime components assembled for one Agent instance."""

    hooks: AgentHooks
    skill_manager: SkillManager | None
    sdk_tools: list[BaseTool]
    prompt_runtime: AgentPromptRuntime
    run_step_storage: RunStepStorage
    trace_storage: BaseTraceStorage | None
    session_storage: SessionStorage


__all__ = ["AgentRuntimeState"]
