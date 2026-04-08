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
    def _normalize_allowed_tools(cls, value: object) -> list[str] | None:
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
    run_step_type: Literal["memory", "sqlite"] = "sqlite"
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
    ack_reaction_emoji: str = "Typing"
    ack_fallback_text: str = "收到，正在处理。"


class ChannelConfig(BaseModel):
    feishu: FeishuConfig = Field(default_factory=FeishuConfig)


def _apply_server_legacy_fields(
    normalized: dict[str, Any],
) -> None:
    server = dict(normalized.get("server") or {})
    if "host" in normalized:
        server.setdefault("host", normalized["host"])
    if "port" in normalized:
        server.setdefault("port", normalized["port"])
    if server:
        normalized["server"] = server


def _apply_storage_legacy_fields(
    normalized: dict[str, Any],
) -> None:
    storage = dict(normalized.get("storage") or {})
    if "run_step_storage_type" in normalized:
        storage.setdefault("run_step_type", normalized["run_step_storage_type"])
    if "trace_storage_type" in normalized:
        storage.setdefault("trace_type", normalized["trace_storage_type"])
    if "metadata_storage_type" in normalized:
        storage.setdefault("metadata_type", normalized["metadata_storage_type"])
    if storage:
        normalized["storage"] = storage


_LEGACY_FEISHU_FIELDS: dict[str, str] = {
    "feishu_enabled": "enabled",
    "feishu_channel_instance_id": "channel_instance_id",
    "feishu_api_base_url": "api_base_url",
    "feishu_app_id": "app_id",
    "feishu_app_secret": "app_secret",
    "feishu_verification_token": "verification_token",
    "feishu_encrypt_key": "encrypt_key",
    "feishu_sdk_log_level": "sdk_log_level",
    "feishu_bot_open_id": "bot_open_id",
    "feishu_default_agent_name": "default_agent_name",
    "feishu_whitelist_open_ids": "whitelist_open_ids",
    "feishu_debounce_ms": "debounce_ms",
    "feishu_max_batch_window_ms": "max_batch_window_ms",
    "feishu_scheduler_wait_timeout": "scheduler_wait_timeout",
    "feishu_ack_reaction_emoji": "ack_reaction_emoji",
    "feishu_ack_fallback_text": "ack_fallback_text",
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


def _apply_feishu_legacy_fields(
    normalized: dict[str, Any],
) -> None:
    channels = dict(normalized.get("channels") or {})
    feishu = dict(channels.get("feishu") or {})
    for legacy_key, nested_key in _LEGACY_FEISHU_FIELDS.items():
        if legacy_key in normalized:
            feishu.setdefault(nested_key, normalized[legacy_key])
    if feishu:
        channels["feishu"] = feishu
        normalized["channels"] = channels


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

    # Legacy flat settings preserved as aliases for environment variables and kwargs.
    host: str | None = Field(default=None, exclude=True)
    port: int | None = Field(default=None, exclude=True)
    run_step_storage_type: Literal["memory", "sqlite"] | None = Field(
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
    def _lift_legacy_flat_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        _apply_server_legacy_fields(normalized)
        _apply_storage_legacy_fields(normalized)
        _apply_feishu_legacy_fields(normalized)
        _reject_default_agent_legacy_fields(normalized)
        _validate_default_agent_override(normalized)
        return normalized

    @model_validator(mode="after")
    def _populate_legacy_accessors(self) -> "ConsoleConfig":
        self.host = self.server.host
        self.port = self.server.port
        self.run_step_storage_type = self.storage.run_step_type
        self.trace_storage_type = self.storage.trace_type
        self.metadata_storage_type = self.storage.metadata_type
        self.feishu_enabled = self.channels.feishu.enabled
        self.feishu_channel_instance_id = self.channels.feishu.channel_instance_id
        self.feishu_api_base_url = self.channels.feishu.api_base_url
        self.feishu_app_id = self.channels.feishu.app_id
        self.feishu_app_secret = self.channels.feishu.app_secret
        self.feishu_verification_token = self.channels.feishu.verification_token
        self.feishu_encrypt_key = self.channels.feishu.encrypt_key
        self.feishu_sdk_log_level = self.channels.feishu.sdk_log_level
        self.feishu_bot_open_id = self.channels.feishu.bot_open_id
        self.feishu_default_agent_name = self.channels.feishu.default_agent_name
        self.feishu_whitelist_open_ids = self.channels.feishu.whitelist_open_ids
        self.feishu_debounce_ms = self.channels.feishu.debounce_ms
        self.feishu_max_batch_window_ms = self.channels.feishu.max_batch_window_ms
        self.feishu_scheduler_wait_timeout = self.channels.feishu.scheduler_wait_timeout
        self.feishu_ack_reaction_emoji = self.channels.feishu.ack_reaction_emoji
        self.feishu_ack_fallback_text = self.channels.feishu.ack_fallback_text
        return self

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
