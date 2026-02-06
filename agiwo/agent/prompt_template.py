"""Unified Jinja2 template renderer for agiwo configuration system.

This module provides a single, simple interface for rendering templates
across all components: config loading, agent execution, and agent prompts.
"""

from jinja2 import Template, Undefined
from jinja2.sandbox import SandboxedEnvironment

from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class SilentUndefined(Undefined):
    """Custom Undefined that returns empty string instead of raising errors.

    This allows templates to gracefully handle missing variables.
    """

    def __str__(self) -> str:
        return ""

    def __getattr__(self, name: str) -> "SilentUndefined":
        return self

    def __getitem__(self, key: str) -> "SilentUndefined":
        return self


class TemplateRenderer:
    """Unified Jinja2 template renderer.

    Usage:
        from agiwo.config.template import renderer
        result = renderer.render("Hello {{ name }}", name="World")
    """

    def __init__(self) -> None:
        """Initialize Jinja2 environment with sandbox for security."""
        self.env = SandboxedEnvironment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=SilentUndefined,
        )
        self._template_cache: dict[str, Template] = {}

    def render(self, template_str: str, **context) -> str:
        """Render template with given context.

        Args:
            template_str: Template string (Jinja2 syntax)
            **context: Context variables (e.g., env=os.environ, input="text")

        Returns:
            Rendered string

        Raises:
            ValueError: If template rendering fails

        Example:
            >>> renderer.render("Hello {{ name }}", name="World")
            'Hello World'
            >>> renderer.render("{{ value | default('N/A') }}", value=None)
            'N/A'
        """
        if not template_str:
            return ""

        # Short-circuit: if no Jinja2 markers, return original string
        if "{{" not in template_str and "{%" not in template_str:
            return template_str

        # Simple caching strategy - cache compiled templates
        cache_key = template_str
        if cache_key not in self._template_cache:
            try:
                self._template_cache[cache_key] = self.env.from_string(template_str)
            except Exception as e:
                raise ValueError(
                    f"Template compilation error: {e}\n"
                    f"Template: {template_str[:5000]}..."
                ) from e

        try:
            return self._template_cache[cache_key].render(**context)
        except Exception as e:
            # Provide clear error message
            raise ValueError(
                f"Template render error: {e}\n"
                f"Template: {template_str[:5000]}...\n"
                f"Available context keys: {list(context.keys())}"
            ) from e

    def clear_cache(self) -> None:
        """Clear template cache (useful for testing)."""
        self._template_cache.clear()


# Global singleton instance - import and use this everywhere
renderer = TemplateRenderer()
