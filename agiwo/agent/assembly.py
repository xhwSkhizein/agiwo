import copy

from agiwo.agent.config import AgentConfig
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.memory_hooks import DefaultMemoryHook
from agiwo.agent.options import AgentOptions
from agiwo.agent.prompt import AgentPromptRuntime
from agiwo.agent.runtime_state import AgentRuntimeState
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.agent.storage.factory import StorageFactory
from agiwo.config.settings import settings
from agiwo.observability.factory import create_trace_storage
from agiwo.skill.config import SkillDiscoveryConfig, normalize_skill_dirs
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import ensure_builtin_tools_loaded
from agiwo.tool.builtin.registry import DEFAULT_TOOLS
from agiwo.utils.logging import get_logger
from agiwo.workspace import build_agent_workspace

logger = get_logger(__name__)


def create_skill_manager(options: AgentOptions, agent_name: str) -> SkillManager:
    """Create SkillManager from options configuration."""
    discovery_config = SkillDiscoveryConfig(
        configured_dirs=normalize_skill_dirs(options.skills_dirs),
        env_dirs=settings.get_env_skills_dirs(),
        root_path=options.get_effective_root_path(),
    )
    manager = SkillManager(config=discovery_config)
    logger.info(
        "skill_manager_created",
        agent_name=agent_name,
        skills_dirs=[str(d) for d in manager.get_resolved_skills_dirs()],
    )
    return manager


def build_effective_hooks(
    *,
    config: AgentConfig,
    agent_id: str,
    hooks: AgentHooks | None,
) -> AgentHooks:
    resolved_hooks = copy.deepcopy(hooks) if hooks is not None else AgentHooks()
    if resolved_hooks.on_memory_retrieve is None:
        memory_hook = DefaultMemoryHook(
            embedding_provider="auto",
            top_k=5,
            root_path=config.options.get_effective_root_path(),
        )
        resolved_hooks.on_memory_retrieve = memory_hook.retrieve_memories
        logger.debug("default_memory_hook_injected", agent_id=agent_id)
    return resolved_hooks


def build_sdk_tools(
    *,
    provided_tools: list[RuntimeToolLike],
    skill_manager: SkillManager | None,
    agent_id: str,
) -> list[BaseTool]:
    ensure_builtin_tools_loaded()
    sdk_tools: list[BaseTool] = []
    provided_tool_names = {tool.get_name() for tool in provided_tools}
    default_tools = [
        cls()
        for tool_name, cls in DEFAULT_TOOLS.items()
        if tool_name not in provided_tool_names
    ]
    sdk_tools.extend(default_tools)
    if skill_manager is not None:
        skill_tool = skill_manager.get_skill_tool()
        sdk_tools.append(skill_tool)
        logger.debug("skill_tool_added", agent_id=agent_id, tool_name=skill_tool.get_name())
    return sdk_tools


def build_prompt_runtime(
    *,
    base_prompt: str,
    options: AgentOptions,
    agent_name: str,
    agent_id: str,
    tools: list[RuntimeToolLike],
    skill_manager: SkillManager | None,
) -> AgentPromptRuntime:
    workspace = build_agent_workspace(
        root_path=options.get_effective_root_path(),
        agent_name=agent_name,
        agent_id=agent_id,
    )
    return AgentPromptRuntime(
        base_prompt=base_prompt,
        workspace=workspace,
        tools=list(tools),
        skill_manager=skill_manager,
    )


def build_agent_runtime_state(
    *,
    config: AgentConfig,
    agent_id: str,
    provided_tools: list[RuntimeToolLike],
    hooks: AgentHooks | None,
) -> AgentRuntimeState:
    resolved_hooks = build_effective_hooks(
        config=config,
        agent_id=agent_id,
        hooks=hooks,
    )
    skill_manager = None
    if config.options.enable_skill:
        skill_manager = create_skill_manager(config.options, config.name)
    sdk_tools = build_sdk_tools(
        provided_tools=provided_tools,
        skill_manager=skill_manager,
        agent_id=agent_id,
    )
    prompt_runtime = build_prompt_runtime(
        base_prompt=config.system_prompt,
        options=config.options,
        agent_name=config.name,
        agent_id=agent_id,
        tools=[*provided_tools, *sdk_tools],
        skill_manager=skill_manager,
    )
    return AgentRuntimeState(
        hooks=resolved_hooks,
        skill_manager=skill_manager,
        sdk_tools=sdk_tools,
        prompt_runtime=prompt_runtime,
        run_step_storage=StorageFactory.create_run_step_storage(
            config.options.run_step_storage
        ),
        trace_storage=create_trace_storage(config.options.trace_storage),
        session_storage=StorageFactory.create_session_storage(
            config.options.run_step_storage
        ),
    )


__all__ = [
    "build_agent_runtime_state",
    "build_effective_hooks",
    "build_prompt_runtime",
    "build_sdk_tools",
    "create_skill_manager",
]
