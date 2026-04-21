"""Tool Manager - Unified entry point for Agent Tools.

This module provides the main ToolManager class that coordinates tool
discovery, metadata access, and tool creation for agents.
"""

from typing import Type

from agiwo.tool.base import BaseTool
from agiwo.tool.builtin.registry import (
    BUILTIN_TOOLS,
    DEFAULT_TOOLS,
    ensure_builtin_tools_loaded,
)
from agiwo.tool.builtin.bash_tool import ensure_bash_tool_pair
from agiwo.tool.reference import (
    AgentToolReference,
    BuiltinToolReference,
    ToolReference,
    parse_tool_reference,
    InvalidToolReferenceError,
)
from agiwo.tool.storage.citation import CitationStoreConfig
import threading

from agiwo.skill.allowlist import skills_enabled
from agiwo.skill.manager import get_global_skill_manager
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

# Re-export for convenience
AGENT_TOOL_PREFIX = AgentToolReference.PREFIX

_GLOBAL_TOOL_MANAGER: "ToolManager | None" = None
_GLOBAL_TOOL_MANAGER_LOCK = threading.Lock()


def get_global_tool_manager(
    citation_store_config: CitationStoreConfig | None = None,
) -> "ToolManager":
    """Get or create the global ToolManager singleton.

    Args:
        citation_store_config: Optional config for citation-enabled tools.
            Used when creating the manager for the first time, or to upgrade
            an existing manager that was initialized without citation config.
    """
    global _GLOBAL_TOOL_MANAGER
    with _GLOBAL_TOOL_MANAGER_LOCK:
        if _GLOBAL_TOOL_MANAGER is None:
            _GLOBAL_TOOL_MANAGER = ToolManager(
                citation_store_config=citation_store_config
            )
        elif (
            citation_store_config is not None
            and _GLOBAL_TOOL_MANAGER._citation_store_config is None
        ):
            _GLOBAL_TOOL_MANAGER.set_citation_store_config(citation_store_config)
    return _GLOBAL_TOOL_MANAGER


class ToolManager:
    """
    Unified manager for Agent Tools system.

    Responsibilities:
    - Coordinate tool discovery and metadata caching
    - Provide tool instances filtered by allowlist
    - Support both builtin tools and custom tools
    """

    def __init__(
        self, citation_store_config: CitationStoreConfig | None = None
    ) -> None:
        """Initialize tool manager and ensure builtin tools are loaded.

        Args:
            citation_store_config: Optional config for citation-enabled tools
        """
        ensure_builtin_tools_loaded()
        self._citation_store_config = citation_store_config
        self._tool_cache: dict[str, BaseTool] = {}  # Cache for stateless tools
        self._initialized = True
        logger.debug("tool_manager_initialized", builtin_count=len(BUILTIN_TOOLS))

    def set_citation_store_config(self, config: CitationStoreConfig) -> None:
        """Set citation store config and reinitialize dependent tools.

        This method updates the citation configuration and clears the tool cache
        for citation-enabled tools (web_search, web_reader) so they will be
        recreated with the new config on next access.

        Args:
            config: The citation store configuration to apply
        """
        self._citation_store_config = config
        _CITATION_TOOLS = {"web_search", "web_reader"}
        for name in _CITATION_TOOLS:
            self._tool_cache.pop(name, None)
        logger.debug("citation_config_updated", cleared_tools=list(_CITATION_TOOLS))

    def list_available_tools(self) -> list[dict[str, str]]:
        """List all available builtin tools with their metadata.

        Returns:
            List of tool info dicts with 'name', 'description' keys
        """
        return [
            {
                "name": name,
                "description": tool_cls().description,
            }
            for name, tool_cls in BUILTIN_TOOLS.items()
        ]

    def list_available_tool_names(self) -> list[str]:
        """Return list of all available builtin tool names."""
        return list(BUILTIN_TOOLS.keys())

    def list_default_tool_names(self) -> list[str]:
        """Return list of default tool names (enabled when tools=None)."""
        return list(DEFAULT_TOOLS.keys())

    def get_tool_class(self, name: str) -> Type[BaseTool] | None:
        """Get the tool class by name.

        Args:
            name: Tool name

        Returns:
            Tool class or None if not found
        """
        return BUILTIN_TOOLS.get(name)

    def is_tool_available(self, name: str) -> bool:
        """Check if a tool is available.

        Args:
            name: Tool name to check

        Returns:
            True if tool exists in builtin registry
        """
        return name in BUILTIN_TOOLS

    def is_default_tool(self, name: str) -> bool:
        """Check if a tool is enabled by default.

        Args:
            name: Tool name to check

        Returns:
            True if tool is in DEFAULT_TOOLS
        """
        return name in DEFAULT_TOOLS

    def parse_allowed_tools(
        self,
        allowed_tools: list[str] | None,
    ) -> list[ToolReference] | None:
        """Parse allowed_tools into structured references.

        - None -> None (all default tools allowed)
        - [] -> [] (no tools allowed)
        - ["bash", "agent:x"] -> [BuiltinToolReference, AgentToolReference]

        Args:
            allowed_tools: List of tool reference strings.

        Returns:
            List of ToolReference objects, or None.

        Raises:
            InvalidToolReferenceError: If a reference is malformed.
        """
        if allowed_tools is None:
            return None

        result: list[ToolReference] = []
        for name in allowed_tools:
            try:
                ref = parse_tool_reference(name)
                result.append(ref)
            except InvalidToolReferenceError:
                # Unknown tools are kept as custom references
                # They will be matched against extra_tools by name
                result.append(BuiltinToolReference(name=name))

        return result

    def get_tools(
        self,
        allowed_tools: list[str] | None = None,
        extra_tools: list[BaseTool] | None = None,
        allowed_skills: list[str] | None = None,
        system_tools: list[BaseTool] | None = None,
    ) -> tuple[BaseTool, ...]:
        """Get tool instances based on allowlist.

        Tools are split into two categories:

        **Functional tools** (controlled by ``allowed_tools``):
            - Builtin tools (bash, web_search, …)
            - Extra / custom tools (AgentTool, user-supplied BaseTool)

        **System tools** (NOT controlled by ``allowed_tools``):
            - SkillTool (controlled by ``allowed_skills``)
            - Scheduler runtime tools (passed via ``system_tools``)

        Args:
            allowed_tools: Allowlist for functional tools.  ``None`` = all
                defaults; ``[]`` = none.
            extra_tools: Additional custom tool instances (subject to
                ``allowed_tools`` filtering).
            allowed_skills: Skill allowlist.  Drives SkillTool creation.
            system_tools: System-level tool instances injected unconditionally.

        Returns:
            Tuple of resolved tool instances
        """
        # 1. Functional tools (controlled by allowed_tools)
        builtin_tools = self._build_builtin_tools(
            self._resolve_tool_names(allowed_tools)
        )
        filtered_extras = self._filter_extra_tools(extra_tools, allowed_tools)

        # 2. System tools (not controlled by allowed_tools)
        skill_tool = self._build_skill_tool(allowed_skills)

        # 3. Merge and finalize
        merged = self._merge_tools(
            builtin_tools, filtered_extras, skill_tool, system_tools
        )
        return self._finalize_tools(merged)

    def _resolve_tool_names(
        self, allowed_tools: list[str] | list[ToolReference] | None
    ) -> set[str]:
        """Resolve which builtin tool names to include.

        Args:
            allowed_tools: List of allowed tool names or references,
                or None for defaults.

        Returns:
            Set of tool names to include.
        """
        if allowed_tools is None:
            return set(DEFAULT_TOOLS.keys())

        result: set[str] = set()
        for item in allowed_tools:
            if isinstance(item, ToolReference):
                if isinstance(item, BuiltinToolReference):
                    result.add(item.name)
                # AgentToolReference is not a builtin, skip
            else:
                # String reference
                if not item.startswith(AgentToolReference.PREFIX):
                    result.add(item)
        return result

    def _build_builtin_tools(self, names: set[str]) -> list[BaseTool]:
        """Build builtin tool instances (with caching for stateless tools).

        Args:
            names: Set of tool names to build

        Returns:
            List of built tool instances
        """
        tools: list[BaseTool] = []

        for name in names:
            tool_cls = BUILTIN_TOOLS.get(name)
            if tool_cls is None:
                continue  # Skip unknown tools

            tool = self._get_or_create_tool(name, tool_cls)
            tools.append(tool)

        return tools

    def _get_or_create_tool(
        self,
        name: str,
        tool_cls: type[BaseTool],
    ) -> BaseTool:
        """Get cached stateless tool or create new instance.

        Args:
            name: Tool name
            tool_cls: Tool class

        Returns:
            Tool instance
        """
        if tool_cls.is_stateless:
            if name not in self._tool_cache:
                self._tool_cache[name] = self._create_tool_instance(name, tool_cls)
            return self._tool_cache[name]

        return self._create_tool_instance(name, tool_cls)

    def _create_tool_instance(
        self,
        name: str,
        tool_cls: type[BaseTool],
    ) -> BaseTool:
        """Create a fresh tool instance with optional citation config.

        Args:
            name: Tool name
            tool_cls: Tool class

        Returns:
            New tool instance
        """
        _CITATION_TOOLS = {"web_search", "web_reader"}

        kwargs: dict[str, object] = {}
        if name in _CITATION_TOOLS and self._citation_store_config is not None:
            kwargs["citation_store_config"] = self._citation_store_config

        return tool_cls(**kwargs)

    def _build_skill_tool(
        self,
        allowed_skills: list[str] | None,
    ) -> BaseTool | None:
        """Create skill tool if skills are enabled.

        Semantics:
            - None → all skills allowed → create SkillTool(allowed_skills=None)
            - [] → skills disabled → return None
            - ["x"] → specific skills → create SkillTool(allowed_skills=["x"])
        """
        if not skills_enabled(allowed_skills):
            return None
        skill_manager = get_global_skill_manager()
        return skill_manager.create_skill_tool(allowed_skills)

    def _filter_extra_tools(
        self,
        extra_tools: list[BaseTool] | None,
        allowed_tools: list[str] | list[ToolReference] | None,
    ) -> list[BaseTool]:
        """Filter extra tools by the allowed_tools allowlist.

        - ``allowed_tools is None`` → all extras pass through
        - ``allowed_tools == []`` → nothing passes
        - Otherwise keep only extras whose ``name`` appears in the resolved
          allowed set (``agent:x`` entries are matched by the bare name ``x``).
        """
        if not extra_tools:
            return []
        if allowed_tools is None:
            return list(extra_tools)

        allowed_names: set[str] = set()
        for item in allowed_tools:
            if isinstance(item, AgentToolReference):
                allowed_names.add(item.agent_id)
            elif isinstance(item, BuiltinToolReference):
                allowed_names.add(item.name)
            elif isinstance(item, str):
                if item.startswith(AgentToolReference.PREFIX):
                    allowed_names.add(item[len(AgentToolReference.PREFIX) :])
                else:
                    allowed_names.add(item)
            else:
                # Fallback: treat as string
                allowed_names.add(str(item))

        return [t for t in extra_tools if t.name in allowed_names]

    def _merge_tools(
        self,
        builtin_tools: list[BaseTool],
        extra_tools: list[BaseTool],
        skill_tool: BaseTool | None,
        system_tools: list[BaseTool] | None,
    ) -> list[BaseTool]:
        """Merge functional and system tools.

        Args:
            builtin_tools: Builtin tool instances (functional)
            extra_tools: Filtered custom tool instances (functional)
            skill_tool: SkillTool instance or None (system)
            system_tools: Scheduler / runtime tool instances (system)

        Returns:
            Merged list of all tools
        """
        result = list(builtin_tools)
        result.extend(extra_tools)

        if skill_tool:
            result.append(skill_tool)
        if system_tools:
            result.extend(system_tools)

        return result

    def _finalize_tools(self, tools: list[BaseTool]) -> tuple[BaseTool, ...]:
        """Apply final consistency checks and return immutable tuple.

        Args:
            tools: List of tool instances

        Returns:
            Finalized tuple of tool instances
        """
        resolved = ensure_bash_tool_pair(tools)
        return tuple(resolved)

    def render_tools_section(self, allowed_tools: list[str] | None = None) -> str:
        """Render a markdown section describing available tools.

        Args:
            allowed_tools: Optional filter for tools to describe

        Returns:
            Markdown formatted tools description
        """
        if allowed_tools is None:
            tools = self.list_available_tools()
        else:
            all_tools = {t["name"]: t for t in self.list_available_tools()}
            tools = [all_tools[name] for name in allowed_tools if name in all_tools]

        if not tools:
            return ""

        lines = ["## Available Tools", ""]
        for tool in tools:
            lines.append(f"- **{tool['name']}**: {tool['description']}")
        lines.append("")

        return "\n".join(lines)
