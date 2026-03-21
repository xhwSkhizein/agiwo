import copy
from dataclasses import dataclass

from agiwo.agent.config import AgentConfig
from agiwo.agent.execution import ChildAgentSpec
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.memory_hooks import DefaultMemoryHook
from agiwo.agent.options import AgentOptions
from agiwo.agent.prompt import AgentPromptRuntime
from agiwo.agent.runtime_tools import RuntimeToolLike
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.skill.config import SkillDiscoveryConfig, normalize_skill_dirs
from agiwo.skill.manager import SkillManager
from agiwo.tool.base import BaseTool
from agiwo.tool.builtin import ensure_builtin_tools_loaded
from agiwo.tool.builtin.registry import DEFAULT_TOOLS
from agiwo.utils.logging import get_logger
from agiwo.workspace import build_agent_workspace

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ResolvedExecutionDefinition:
    """Immutable execution snapshot consumed by the orchestrator/engine."""

    agent_id: str
    agent_name: str
    description: str
    model: Model
    hooks: AgentHooks
    options: AgentOptions
    tools: tuple[RuntimeToolLike, ...]
    system_prompt: str


@dataclass(frozen=True, slots=True)
class AgentCloneSpec:
    """Pure inputs for cloning a scheduler child Agent template."""

    agent_id: str
    config: AgentConfig
    hooks: AgentHooks
    tools: tuple[RuntimeToolLike, ...]


def _create_skill_manager(options: AgentOptions, agent_name: str) -> SkillManager:
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


def _build_effective_hooks(
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


def _build_sdk_tools(
    *,
    provided_tools: list[RuntimeToolLike],
    skill_manager: SkillManager | None,
    agent_id: str,
) -> list[BaseTool]:
    ensure_builtin_tools_loaded()
    provided_tool_names = {tool.get_name() for tool in provided_tools}
    default_tools = [
        cls()
        for tool_name, cls in DEFAULT_TOOLS.items()
        if tool_name not in provided_tool_names
    ]
    sdk_tools: list[BaseTool] = list(default_tools)
    if skill_manager is not None:
        skill_tool = skill_manager.get_skill_tool()
        sdk_tools.append(skill_tool)
        logger.debug(
            "skill_tool_added", agent_id=agent_id, tool_name=skill_tool.get_name()
        )
    return sdk_tools


def _build_prompt_runtime(
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


class AgentDefinitionRuntime:
    """Single owner for definition-scoped agent runtime state."""

    def __init__(
        self,
        *,
        config: AgentConfig,
        agent_id: str,
        provided_tools: list[RuntimeToolLike],
        hooks: AgentHooks | None,
    ) -> None:
        self._config = config
        self._agent_id = agent_id
        self._provided_tools = list(provided_tools)
        self._runtime_tools: list[RuntimeToolLike] = []
        self._hooks = _build_effective_hooks(
            config=self._config,
            agent_id=self._agent_id,
            hooks=hooks,
        )
        self._skill_manager = (
            _create_skill_manager(self._config.options, self._config.name)
            if self._config.options.enable_skill
            else None
        )
        self._sdk_tools = _build_sdk_tools(
            provided_tools=self._provided_tools,
            skill_manager=self._skill_manager,
            agent_id=self._agent_id,
        )
        self._prompt_runtime = self._create_prompt_runtime(
            base_prompt=self._config.system_prompt,
            agent_id=self._agent_id,
            agent_name=self._config.name,
            tools=self._effective_tools(),
            options=self._config.options,
        )

    @property
    def hooks(self) -> AgentHooks:
        return self._hooks

    @hooks.setter
    def hooks(self, hooks: AgentHooks | None) -> None:
        self._hooks = _build_effective_hooks(
            config=self._config,
            agent_id=self._agent_id,
            hooks=hooks,
        )

    @property
    def tools(self) -> tuple[RuntimeToolLike, ...]:
        return tuple(self._effective_tools())

    def install_runtime_tools(self, tools: list[RuntimeToolLike]) -> None:
        existing_names = {tool.get_name() for tool in self._effective_tools()}
        changed = False
        for tool in tools:
            tool_name = tool.get_name()
            if tool_name in existing_names:
                continue
            self._runtime_tools.append(tool)
            existing_names.add(tool_name)
            changed = True
        if changed:
            self._refresh_prompt_runtime()

    async def get_effective_system_prompt(self) -> str:
        return await self._prompt_runtime.get_system_prompt()

    async def snapshot_root_definition(
        self,
        *,
        model: Model,
    ) -> ResolvedExecutionDefinition:
        options = self._config.options.model_copy(deep=True)
        return ResolvedExecutionDefinition(
            agent_id=self._agent_id,
            agent_name=self._config.name,
            description=self._config.description,
            model=model,
            hooks=copy.deepcopy(self._hooks),
            options=options,
            tools=self.tools,
            system_prompt=await self.get_effective_system_prompt(),
        )

    async def snapshot_child_definition(
        self,
        *,
        model: Model,
        spec: ChildAgentSpec,
    ) -> ResolvedExecutionDefinition:
        options, tools, base_prompt, hooks = self._materialize_child(spec)
        prompt_runtime = self._create_prompt_runtime(
            base_prompt=base_prompt,
            agent_id=spec.agent_id,
            agent_name=spec.agent_name,
            tools=list(tools),
            options=options,
        )
        return ResolvedExecutionDefinition(
            agent_id=spec.agent_id,
            agent_name=spec.agent_name,
            description=spec.description,
            model=model,
            hooks=hooks,
            options=options,
            tools=tools,
            system_prompt=await prompt_runtime.get_system_prompt(),
        )

    def build_scheduler_child_clone(
        self,
        *,
        child_id: str,
        instruction: str | None = None,
        system_prompt_override: str | None = None,
        exclude_tool_names: set[str] | None = None,
    ) -> AgentCloneSpec:
        spec = ChildAgentSpec(
            agent_id=child_id,
            agent_name=self._config.name,
            description=self._config.description,
            instruction=instruction,
            system_prompt_override=system_prompt_override,
            exclude_tool_names=frozenset(exclude_tool_names or ()),
        )
        options, tools, base_prompt, hooks = self._materialize_child(spec)
        child_config = copy.deepcopy(self._config)
        child_config.system_prompt = base_prompt
        child_config.options = options
        return AgentCloneSpec(
            agent_id=child_id,
            config=child_config,
            hooks=hooks,
            tools=tools,
        )

    def _materialize_child(
        self,
        spec: ChildAgentSpec,
    ) -> tuple[AgentOptions, tuple[RuntimeToolLike, ...], str, AgentHooks]:
        options = self._config.options.model_copy(deep=True)
        options.enable_termination_summary = True
        tools = tuple(
            tool
            for tool in self._effective_tools()
            if tool.get_name() not in spec.exclude_tool_names
        )
        base_prompt = self._append_instruction(
            spec.system_prompt_override or self._config.system_prompt,
            spec.instruction,
        )
        return options, tools, base_prompt, copy.deepcopy(self._hooks)

    @staticmethod
    def _append_instruction(base_prompt: str, instruction: str | None) -> str:
        if not instruction:
            return base_prompt
        return (
            f"{base_prompt}\n\n<task-instruction>\n{instruction}\n</task-instruction>"
        )

    def _refresh_prompt_runtime(self) -> None:
        self._prompt_runtime = self._create_prompt_runtime(
            base_prompt=self._config.system_prompt,
            agent_id=self._agent_id,
            agent_name=self._config.name,
            tools=self._effective_tools(),
            options=self._config.options,
        )

    def _effective_tools(self) -> list[RuntimeToolLike]:
        return [
            *self._provided_tools,
            *self._sdk_tools,
            *self._runtime_tools,
        ]

    def _create_prompt_runtime(
        self,
        *,
        base_prompt: str,
        agent_id: str,
        agent_name: str,
        tools: list[RuntimeToolLike],
        options: AgentOptions,
    ) -> AgentPromptRuntime:
        return _build_prompt_runtime(
            base_prompt=base_prompt,
            options=options,
            agent_name=agent_name,
            agent_id=agent_id,
            tools=tools,
            skill_manager=self._skill_manager,
        )


__all__ = [
    "AgentCloneSpec",
    "AgentDefinitionRuntime",
    "ResolvedExecutionDefinition",
]
