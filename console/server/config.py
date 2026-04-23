"""Console server configuration."""

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agiwo.config.settings import ModelProvider
from agiwo.config.settings import settings as sdk_settings
from agiwo.llm.config_policy import (
    sanitize_model_params_data,
    validate_provider_model_params,
)
from agiwo.skill.allowlist import normalize_allowed_skills


class DefaultAgentConfig(BaseModel):
    """Template for the default agent created on first boot."""

    id: str = "default-console-agent"
    name: str = "Console Agent"
    description: str = "Default agent for Console channels"
    model_provider: ModelProvider = "openai-compatible"
    model_name: str = "codex-5.3"
    model_params: dict[str, Any] = Field(default_factory=dict)
    system_prompt: str = ""
    allowed_tools: list[str] | None = None
    allowed_skills: list[str] | None = None

    @field_validator("model_params", mode="before")
    @classmethod
    def _normalize_model_params(cls, value: object) -> dict[str, Any]:
        sanitized = sanitize_model_params_data(value)
        if not isinstance(sanitized, dict):
            return {}
        return sanitized

    @field_validator("allowed_tools", mode="before")
    @classmethod
    def _parse_allowed_tools(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return parsed
                return [value]
            except json.JSONDecodeError:
                return [value]
        if not isinstance(value, list):
            raise ValueError("allowed_tools must be a list")
        return value

    @field_validator("allowed_skills", mode="before")
    @classmethod
    def _normalize_allowed_skills(cls, value: object) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("allowed_skills must be a list")
        normalized = normalize_allowed_skills(value)
        return list(normalized) if normalized is not None else None


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8422


class StorageConfig(BaseModel):
    run_log_type: Literal["memory", "sqlite"] = "sqlite"
    trace_type: Literal["memory", "sqlite"] = "sqlite"
    metadata_type: Literal["memory", "sqlite"] = "sqlite"

    @field_validator("trace_type", mode="before")
    @classmethod
    def _normalize_trace_type(cls, value: object) -> str:
        if value == "none":
            return "memory"
        return value  # type: ignore[return-value]


class FeishuConfig(BaseModel):
    enabled: bool = False
    channel_instance_id: str = "feishu-main"
    api_base_url: str = "https://open.feishu.cn"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""
    sdk_log_level: Literal["debug", "info", "warn", "error"] = "info"
    bot_open_id: str = ""
    default_agent_name: str = ""
    whitelist_open_ids: list[str] = Field(default_factory=list)
    debounce_ms: int = 3000
    max_batch_window_ms: int = 15000
    scheduler_wait_timeout: int = 900
    ack_reaction_emoji: str = "OnIt"
    ack_fallback_text: str = "收到，正在处理。"


class ChannelConfig(BaseModel):
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


_LEGACY_FLAT_FIELD_HINTS: dict[str, str] = {
    "host": "server.host",
    "port": "server.port",
    "run_log_storage_type": "storage.run_log_type",
    "trace_storage_type": "storage.trace_type",
    "metadata_storage_type": "storage.metadata_type",
    "feishu_enabled": "channels.feishu.enabled",
    "feishu_channel_instance_id": "channels.feishu.channel_instance_id",
    "feishu_api_base_url": "channels.feishu.api_base_url",
    "feishu_app_id": "channels.feishu.app_id",
    "feishu_app_secret": "channels.feishu.app_secret",
    "feishu_verification_token": "channels.feishu.verification_token",
    "feishu_encrypt_key": "channels.feishu.encrypt_key",
    "feishu_sdk_log_level": "channels.feishu.sdk_log_level",
    "feishu_bot_open_id": "channels.feishu.bot_open_id",
    "feishu_default_agent_name": "channels.feishu.default_agent_name",
    "feishu_whitelist_open_ids": "channels.feishu.whitelist_open_ids",
    "feishu_debounce_ms": "channels.feishu.debounce_ms",
    "feishu_max_batch_window_ms": "channels.feishu.max_batch_window_ms",
    "feishu_scheduler_wait_timeout": "channels.feishu.scheduler_wait_timeout",
    "feishu_ack_reaction_emoji": "channels.feishu.ack_reaction_emoji",
    "feishu_ack_fallback_text": "channels.feishu.ack_fallback_text",
}

_LEGACY_DEFAULT_AGENT_GROUP_ENV_HINTS: dict[str, tuple[str, str]] = {
    "default_agent_model": (
        "AGIWO_CONSOLE_DEFAULT_AGENT_MODEL__*",
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER / "
        "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_NAME",
    ),
    "default_agent_system": (
        "AGIWO_CONSOLE_DEFAULT_AGENT_SYSTEM__PROMPT",
        "AGIWO_CONSOLE_DEFAULT_AGENT__SYSTEM_PROMPT",
    ),
}


def _reject_legacy_flat_fields(
    normalized: dict[str, Any],
) -> None:
    for legacy_key, replacement in _LEGACY_FLAT_FIELD_HINTS.items():
        if normalized.get(legacy_key) is None:
            continue
        raise ValueError(
            f"Unsupported legacy config field '{legacy_key}'. Use '{replacement}' instead."
        )


def _reject_default_agent_legacy_fields(
    normalized: dict[str, Any],
) -> None:
    for legacy_key, env_hints in _LEGACY_DEFAULT_AGENT_GROUP_ENV_HINTS.items():
        if normalized.get(legacy_key) is None:
            continue
        legacy_env, replacement = env_hints
        raise ValueError(
            "Unsupported legacy default-agent env keys "
            f"({legacy_env}). Use {replacement} instead."
        )


def _validate_default_agent_override(
    normalized: dict[str, Any],
) -> None:
    default_agent = normalized.get("default_agent")
    if not isinstance(default_agent, dict) or not default_agent:
        return

    configured_keys = {key for key, value in default_agent.items() if value is not None}
    if not configured_keys:
        return

    required_model_keys = {"model_provider", "model_name"}
    provided_model_keys = configured_keys & required_model_keys
    if provided_model_keys != required_model_keys:
        raise ValueError(
            "When overriding AGIWO_CONSOLE_DEFAULT_AGENT__*, you must also set "
            "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_PROVIDER and "
            "AGIWO_CONSOLE_DEFAULT_AGENT__MODEL_NAME."
        )

    model_params = sanitize_model_params_data(default_agent.get("model_params") or {})
    validate_provider_model_params(
        str(default_agent["model_provider"]),
        model_params if isinstance(model_params, dict) else {},
    )


class ConsoleConfig(BaseSettings):
    """Console-specific configuration root."""

    model_config = SettingsConfigDict(
        env_prefix="AGIWO_CONSOLE_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    default_agent: DefaultAgentConfig = Field(default_factory=DefaultAgentConfig)
    channels: ChannelConfig = Field(default_factory=ChannelConfig)

    # Legacy flat settings are retained only so the boundary can reject them
    # explicitly instead of silently ignoring them.
    host: str | None = Field(default=None, exclude=True)
    port: int | None = Field(default=None, exclude=True)
    run_log_storage_type: Literal["memory", "sqlite"] | None = Field(
        default=None,
        exclude=True,
    )
    trace_storage_type: Literal["memory", "sqlite", "none"] | None = Field(
        default=None,
        exclude=True,
    )
    metadata_storage_type: Literal["memory", "sqlite"] | None = Field(
        default=None,
        exclude=True,
    )
    feishu_enabled: bool | None = Field(default=None, exclude=True)
    feishu_channel_instance_id: str | None = Field(default=None, exclude=True)
    feishu_api_base_url: str | None = Field(default=None, exclude=True)
    feishu_app_id: str | None = Field(default=None, exclude=True)
    feishu_app_secret: str | None = Field(default=None, exclude=True)
    feishu_verification_token: str | None = Field(default=None, exclude=True)
    feishu_encrypt_key: str | None = Field(default=None, exclude=True)
    feishu_sdk_log_level: Literal["debug", "info", "warn", "error"] | None = Field(
        default=None,
        exclude=True,
    )
    feishu_bot_open_id: str | None = Field(default=None, exclude=True)
    feishu_default_agent_name: str | None = Field(default=None, exclude=True)
    feishu_whitelist_open_ids: list[str] | None = Field(default=None, exclude=True)
    feishu_debounce_ms: int | None = Field(default=None, exclude=True)
    feishu_max_batch_window_ms: int | None = Field(default=None, exclude=True)
    feishu_scheduler_wait_timeout: int | None = Field(default=None, exclude=True)
    feishu_ack_reaction_emoji: str | None = Field(default=None, exclude=True)
    feishu_ack_fallback_text: str | None = Field(default=None, exclude=True)
    default_agent_model: dict[str, Any] | None = Field(default=None, exclude=True)
    default_agent_system: dict[str, Any] | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _reject_legacy_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        _reject_legacy_flat_fields(normalized)
        _reject_default_agent_legacy_fields(normalized)
        _validate_default_agent_override(normalized)
        return normalized

    @property
    def sqlite_db_path(self) -> str:
        """SQLite database path from SDK settings."""
        resolved = sdk_settings.resolve_path(sdk_settings.sqlite_db_path)
        return str(resolved) if resolved else "agiwo.db"

    @property
    def sqlite_trace_collection(self) -> str:
        """Trace collection name from SDK settings."""
        return sdk_settings.trace_collection_name or "agiwo_traces"


DefaultAgentTemplate = DefaultAgentConfig
