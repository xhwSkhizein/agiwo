"""
Tool configuration classes for builtin tools.

All configuration classes support:
1. Constructor parameters (highest priority)
2. Environment variables (fallback)
3. Default values (lowest priority)
"""

from dataclasses import dataclass, field

from agiwo.config.settings import settings


@dataclass
class WebReaderApiConfig:
    """Configuration for WebReaderTool and its browser fallback."""

    timeout_seconds: int = field(
        default_factory=lambda: settings.web_reader_api_timeout
    )
    max_content_length: int = field(
        default_factory=lambda: settings.web_reader_api_max_content_length
    )
    model_provider: str = field(
        default_factory=lambda: settings.tool_default_model_provider
    )
    model_name: str = field(default_factory=lambda: settings.get_tool_model_name())
    model_base_url: str | None = field(
        default_factory=lambda: settings.tool_default_model_base_url
    )
    api_key_env_name: str | None = field(
        default_factory=lambda: settings.get_tool_model_api_key_env_name()
    )
    model_temperature: float = field(
        default_factory=lambda: settings.tool_default_model_temperature
    )
    model_top_p: float = field(
        default_factory=lambda: settings.tool_default_model_top_p
    )
    model_max_tokens: int = field(
        default_factory=lambda: settings.tool_default_model_max_tokens
    )
    max_retries: int = 3
    headless: bool = False
    wait_strategy: str = "domcontentloaded"
    max_browsers: int = 1
    browser_idle_ttl_seconds: int = 300
    browser_max_uses: int = 20
    save_login_state: bool = True
    user_data_dir: str = field(
        default_factory=lambda: str(settings.get_root_path() / "browser_data")
    )
    browser_launch_timeout: int = 30
    auto_close_browser: bool = True


__all__ = [
    "WebReaderApiConfig",
]
