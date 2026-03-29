"""
Console server configuration.

Console-specific settings use AGIWO_CONSOLE_ prefix.
Storage/LLM settings are inherited from SDK's AgiwoSettings (AGIWO_ prefix).
"""

from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agiwo.config.settings import settings as sdk_settings
from agiwo.llm.config_policy import sanitize_model_params_data
from agiwo.config.settings import ModelProvider


class ConsoleConfig(BaseSettings):
    """
    Console-specific configuration.

    Storage paths are inherited from SDK's AgiwoSettings.
    Console only defines its own unique settings (server, CORS, Feishu, etc.).
    """

    model_config = SettingsConfigDict(
        env_prefix="AGIWO_CONSOLE_",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 8422

    # Storage backend for run/step persistence
    # Note: when using sqlite, path comes from SDK settings (AGIWO_SQLITE_DB_PATH)
    run_step_storage_type: Literal["memory", "sqlite"] = "sqlite"

    # Storage backend for traces.
    trace_storage_type: Literal["memory", "sqlite"] = "sqlite"

    # Storage backend for console metadata (agent registry, channel runtime, scheduler state).
    metadata_storage_type: Literal["memory", "sqlite"] = "sqlite"

    # Default Agent Configuration
    default_agent_id: str = "default-console-agent"
    default_agent_name: str = "Console Agent"
    default_agent_description: str = "Default agent for Console channels"
    default_agent_model_provider: ModelProvider = "openai-compatible"
    default_agent_model_name: str = "codex-5.3"
    default_agent_model_params: dict[str, Any] = Field(default_factory=dict)
    default_agent_system_prompt: str = ""
    default_agent_tools: list[str] = Field(
        default_factory=lambda: [
            "bash",
            "bash_process",
            "web_search",
            "web_reader",
            "memory_retrieval",
        ]
    )

    # Feishu channel
    feishu_enabled: bool = False
    feishu_channel_instance_id: str = "feishu-main"
    feishu_api_base_url: str = "https://open.feishu.cn"
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    feishu_sdk_log_level: Literal["debug", "info", "warn", "error"] = "info"
    feishu_bot_open_id: str = ""
    feishu_default_agent_name: str = ""
    feishu_whitelist_open_ids: list[str] = []
    feishu_debounce_ms: int = 3000
    feishu_max_batch_window_ms: int = 15000
    feishu_scheduler_wait_timeout: int = 900
    feishu_ack_reaction_emoji: str = "Typing"
    feishu_ack_fallback_text: str = "收到，正在处理。"

    # --- Properties that delegate to SDK settings ---

    @property
    def sqlite_db_path(self) -> str:
        """SQLite database path from SDK settings."""
        resolved = sdk_settings.resolve_path(sdk_settings.sqlite_db_path)
        return str(resolved) if resolved else "agiwo.db"

    @property
    def sqlite_trace_collection(self) -> str:
        """Trace collection name from SDK settings."""
        return sdk_settings.trace_collection_name or "agiwo_traces"

    @field_validator("trace_storage_type", mode="before")
    @classmethod
    def _normalize_trace_storage_type(cls, value: object) -> str:
        if value == "none":
            return "memory"
        return value  # type: ignore[return-value]

    @field_validator("default_agent_model_params", mode="before")
    @classmethod
    def _normalize_default_agent_model_params(cls, value: object) -> dict[str, Any]:
        sanitized = sanitize_model_params_data(value)
        if not isinstance(sanitized, dict):
            return {}
        return sanitized
