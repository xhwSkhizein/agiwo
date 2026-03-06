"""
Console server configuration.

Console-specific settings use AGIWO_CONSOLE_ prefix.
Storage/LLM settings are inherited from SDK's AgiwoSettings (AGIWO_ prefix).
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agiwo.config.settings import settings as sdk_settings


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
    run_step_storage_type: Literal["memory", "sqlite", "mongodb"] = "sqlite"

    # Storage backend for traces.
    # "none" is treated the same as "memory" for lightweight runtime-only tracing.
    trace_storage_type: Literal["none", "memory", "sqlite", "mongodb"] = "sqlite"

    # Storage backend for console metadata (agent registry, channel runtime).
    # Scheduler state storage currently supports memory/sqlite only:
    # metadata_storage_type=mongodb falls back to in-memory scheduler state.
    metadata_storage_type: Literal["memory", "sqlite", "mongodb"] = "sqlite"

    # MongoDB settings (used when any *_storage_type=mongodb)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "agiwo"
    mongodb_trace_collection: str = "traces"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]

    # Default Agent Configuration
    default_agent_id: str = "Walaha000"
    default_agent_name: str = "Walaha"
    default_agent_description: str = ""
    default_agent_model_provider: str = "generic"
    default_agent_model_name: str = "codex-5.3"
    default_agent_model_params: dict = Field(default_factory=dict)
    default_agent_system_prompt: str = ""

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

    @property
    def effective_trace_storage_type(self) -> Literal["memory", "sqlite", "mongodb"]:
        """Normalized trace storage type."""
        if self.trace_storage_type == "none":
            return "memory"
        return self.trace_storage_type

    @property
    def scheduler_state_storage_type(self) -> Literal["memory", "sqlite"]:
        """Scheduler state storage type derived from metadata backend support."""
        if self.metadata_storage_type == "sqlite":
            return "sqlite"
        return "memory"
