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
from agiwo.tool.storage.citation import CitationStoreConfig
from agiwo.skill.allowlist import skills_enabled
from agiwo.skill.manager import get_global_skill_manager
from agiwo.utils.logging import get_logger

AGENT_TOOL_PREFIX = "agent:"

logger = get_logger(__name__)

_GLOBAL_TOOL_MANAGER: "ToolManager | None" = None


def get_global_tool_manager(
    citation_store_config: CitationStoreConfig | None = None,
) -> "ToolManager":
    """Get or create the global ToolManager singleton.

    Args:
        citation_store_config: Optional config for citation-enabled tools.
            Only used when creating the manager for the first time.
    """
    global _GLOBAL_TOOL_MANAGER
    if _GLOBAL_TOOL_MANAGER is None:
        _GLOBAL_TOOL_MANAGER = ToolManager(citation_store_config=citation_store_config)
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

    def normalize_allowed_tools(
        self,
        allowed_tools: list[str] | None,
    ) -> list[str] | None:
        """Normalize allowed_tools list.

        - None means all default tools are allowed
        - Empty list means no builtin tools allowed
        - ``agent:<id>`` prefixed items are accepted as valid tool references
        - Validates that all non-prefixed tool names exist in the builtin registry

        Raises:
            ValueError: If unknown builtin tool names or malformed agent refs.
        """
        if allowed_tools is None:
            return None

        normalized: list[str] = []
        available = set(self.list_available_tool_names())
        unknown: list[str] = []

        for name in allowed_tools:
            if name.startswith(AGENT_TOOL_PREFIX):
                agent_id = name[len(AGENT_TOOL_PREFIX) :].strip()
                if not agent_id:
                    raise ValueError(f"Invalid tool reference: {name!r}")
                normalized.append(name)
            elif name in available:
                normalized.append(name)
            else:
                unknown.append(name)

        if unknown:
            raise ValueError(f"Unknown tool names: {', '.join(unknown)}")

        return normalized

    def get_tools(
        self,
        allowed_tools: list[str] | None = None,
        extra_tools: list[BaseTool] | None = None,
        allowed_skills: list[str] | None = None,
    ) -> tuple[BaseTool, ...]:
        """Get tool instances based on allowlist.

        Stateless tools are cached and reused. Non-stateless tools are created fresh.
        If allowed_skills is provided and non-empty, a SkillTool will be added.

        Args:
            allowed_tools: List of allowed tool names, or None for defaults
            extra_tools: Additional custom tools to include
            allowed_skills: List of allowed skill names. If non-empty, adds skill tool.

        Returns:
            Tuple of resolved tool instances
        """
        # 1. Determine which builtin tools to include
        selected_tool_names = self._resolve_tool_names(allowed_tools)

        # 2. Build builtin tools
        tools = self._build_builtin_tools(selected_tool_names)

        # 3. Build skill tool if needed
        skill_tool = self._build_skill_tool(allowed_skills)

        # 4. Merge all tools and finalize
        merged = self._merge_tools(tools, extra_tools, skill_tool)
        return self._finalize_tools(merged)

    def _resolve_tool_names(self, allowed_tools: list[str] | None) -> set[str]:
        """Resolve which builtin tool names to include.

        Args:
            allowed_tools: List of allowed tool names, or None for defaults

        Returns:
            Set of tool names to include
        """
        if allowed_tools is None:
            return set(DEFAULT_TOOLS.keys())
        return set(allowed_tools)

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

    def _merge_tools(
        self,
        builtin_tools: list[BaseTool],
        extra_tools: list[BaseTool] | None,
        skill_tool: BaseTool | None,
    ) -> list[BaseTool]:
        """Merge builtin, extra, and skill tools.

        Args:
            builtin_tools: List of builtin tool instances
            extra_tools: List of custom/extra tool instances
            skill_tool: Skill tool instance or None

        Returns:
            Merged list of all tools
        """
        result = list(builtin_tools)

        if extra_tools:
            result.extend(extra_tools)

        if skill_tool:
            result.append(skill_tool)

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
