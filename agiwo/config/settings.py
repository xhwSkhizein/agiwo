"""
Global settings from environment variables.
"""

import os
from pathlib import Path
from typing import Literal, get_args

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ModelProvider = Literal[
    "openai",
    "openai-compatible",
    "deepseek",
    "anthropic",
    "anthropic-compatible",
    "nvidia",
    "bedrock-anthropic",
]

ALL_MODEL_PROVIDERS: tuple[str, ...] = get_args(ModelProvider)
COMPATIBLE_MODEL_PROVIDERS = frozenset({"openai-compatible", "anthropic-compatible"})
_PROVIDER_DEFAULT_MODEL_NAME_ATTRS: dict[str, str] = {
    "openai": "openai_model_name",
    "deepseek": "deepseek_model_name",
    "anthropic": "anthropic_model_name",
    "nvidia": "nvidia_model_name",
}


class AgiwoSettings(BaseSettings):
    """
    Global configuration loaded from environment variables.

    Environment variables should be prefixed with AGIWO_
    Example: AGIWO_DEBUG=true, AGIWO_MONGO_URI=mongodb://localhost:27017
    """

    model_config = SettingsConfigDict(
        env_prefix="AGIWO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # === Core ===
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

    # === Paths / Storage ===
    # Root path - all other paths are relative to this unless absolute
    root_path: str = ".agiwo"

    # Storage settings (relative to root_path if not absolute)
    mongo_uri: str | None = None
    mongo_db_name: str | None = "agiwo"
    sqlite_db_path: str | None = "agiwo.db"

    # Vector DB settings (relative to root_path if not absolute)
    vector_db_path: str | None = "vector_db"

    # Repository and Storage settings
    default_session_store: str | None = None
    default_trace_storage: str | None = None
    trace_collection_name: str | None = "agiwo_traces"
    trace_buffer_size: int = 200

    # === Provider Defaults ===
    # OpenAI
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
    )
    openai_base_url: str | None = Field(
        default="https://api.openai.com/v1",
        validation_alias=AliasChoices("OPENAI_BASE_URL"),
    )
    openai_model_name: str | None = "gpt-4o-mini"

    # Deepseek
    deepseek_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str | None = Field(
        default="https://api.deepseek.com",
        validation_alias=AliasChoices("DEEPSEEK_BASE_URL"),
    )
    deepseek_model_name: str | None = "deepseek-chat"

    # NVIDIA
    nvidia_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("NVIDIA_BUILD_API_KEY"),
    )
    nvidia_base_url: str | None = Field(
        default="https://integrate.api.nvidia.com/v1",
        validation_alias=AliasChoices("NVIDIA_BUILD_BASE_URL"),
    )
    nvidia_model_name: str | None = "moonshotai/kimi-k2.5"  # z-ai/glm4.7

    # Anthropic
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY"),
    )
    anthropic_base_url: str | None = Field(
        default="https://api.anthropic.com/v1",
        validation_alias=AliasChoices("ANTHROPIC_BASE_URL"),
    )
    anthropic_model_name: str | None = "claude-3-5-sonnet-20240620"

    # AWS Bedrock
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS_REGION"),
    )
    aws_profile: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AWS_PROFILE"),
    )

    # === Tool Model Defaults ===
    tool_default_model_provider: ModelProvider = Field(
        default="deepseek",
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_PROVIDER"),
    )
    tool_default_model_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_NAME"),
    )
    tool_default_model_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_BASE_URL"),
    )
    tool_default_model_api_key_env_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_API_KEY_ENV_NAME"),
    )
    tool_default_model_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_TEMPERATURE"),
    )
    tool_default_model_top_p: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_TOP_P"),
    )
    tool_default_model_max_tokens: int = Field(
        default=2048,
        ge=1,
        validation_alias=AliasChoices("AGIWO_TOOL_DEFAULT_MODEL_MAX_TOKENS"),
    )

    # === Observability / OTLP ===
    otlp_enabled: bool = False
    otlp_endpoint: str | None = None  # e.g., "http://localhost:4317" for gRPC
    otlp_protocol: Literal["grpc", "http"] = "grpc"
    otlp_headers: dict[str, str] = Field(default_factory=dict)
    otlp_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0
    )  # 1.0 = 100% sampling

    # === Embedding ===
    embedding_provider: str = "auto"  # openai | openai-like | local | auto | disabled
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_EMBEDDING_API_KEY"),
    )
    embedding_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_EMBEDDING_BASE_URL"),
    )
    local_embedding_model_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_LOCAL_EMBEDDING_MODEL_PATH"),
    )

    # === Memory Retrieval ===
    memory_chunk_tokens: int = 400
    memory_chunk_overlap: int = 80
    memory_top_k: int = 5
    memory_vector_weight: float = 0.7
    memory_bm25_weight: float = 0.3
    memory_temporal_decay: bool = False
    memory_temporal_decay_half_life: float = 30.0
    memory_mmr_enabled: bool = False
    memory_mmr_lambda: float = 0.5

    # === Builtin Tools ===
    web_search_api_base_url: str = Field(
        default="https://google.serper.dev",
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_SEARCH_API_BASE_URL"),
    )

    web_search_serper_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("SERPER_API_KEY"),
    )

    web_search_api_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_SEARCH_API_TIMEOUT"),
    )
    web_search_api_max_results: int = Field(
        default=10,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_SEARCH_API_MAX_RESULTS"),
    )
    web_search_api_recency_days: int = Field(
        default=7,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_SEARCH_API_RECENCY_DAYS"),
    )
    web_search_api_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_SEARCH_API_MAX_RETRIES"),
    )
    web_reader_api_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_API_TIMEOUT"),
    )
    web_reader_api_max_content_length: int = Field(
        default=4096,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_API_MAX_CONTENT_LENGTH"),
    )

    # === Skills ===
    # Skills configuration (paths relative to root_path if not absolute)
    skills_dirs: list[str] = Field(
        default_factory=lambda: ["examples/skills", "skills"],
        description="Skill directories to scan (relative to root_path if not absolute)",
    )
    is_skills_enabled: bool = True

    # === Scheduler ===
    event_debounce_min_count: int = Field(
        default=1,
        description="Minimum number of pending events to trigger a debounce wake",
    )
    event_debounce_max_wait_seconds: float = Field(
        default=30.0,
        description="Maximum seconds to wait before triggering a debounce wake regardless of count",
    )

    # === Compact ===
    compact_threshold_ratio: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Trigger compact when context tokens reach this ratio of model max_context_window",
    )
    compact_model: str | None = Field(
        default=None,
        description="Model to use for compact summarization. None means use agent's model",
    )
    compact_interruptible: bool = Field(
        default=False,
        description="Whether compact operation can be interrupted by AbortSignal",
    )
    compact_retry_count: int = Field(
        default=3,
        ge=0,
        description="Number of retries for compact LLM call on failure",
    )
    compact_assistant_response: str = Field(
        default="Understood. I have the context from the summary. Continuing.",
        description="Assistant response text after compact summary",
    )

    @model_validator(mode="after")
    def _validate_tool_model_defaults(self) -> "AgiwoSettings":
        if self.tool_default_model_name is not None:
            return self
        if (
            self.get_default_model_name_for_provider(self.tool_default_model_provider)
            is not None
        ):
            return self
        raise ValueError(
            "AGIWO_TOOL_DEFAULT_MODEL_NAME is required when "
            f"AGIWO_TOOL_DEFAULT_MODEL_PROVIDER='{self.tool_default_model_provider}'"
        )

    def resolve_path(self, path: str | None) -> Path | None:
        """Resolve a path relative to root_path if it's not absolute.

        Args:
            path: The path to resolve. If None, returns None.
                  If absolute, returns as-is. If relative, joins with root_path.

        Returns:
            Resolved Path object or None if input was None.
        """
        if path is None:
            return None
        p = Path(path)
        if p.is_absolute():
            return p
        return Path(self.root_path) / p

    def get_root_path(self) -> Path:
        """Get the root path as a Path object.

        Returns:
            Path object for root_path (expanded user and absolute).
        """
        return Path(self.root_path).expanduser().resolve()

    @staticmethod
    def _secret_to_str(value: SecretStr | None) -> str | None:
        if value is None:
            return None
        return value.get_secret_value()

    def get_openai_api_key(self) -> str | None:
        return self._secret_to_str(self.openai_api_key)

    def get_deepseek_api_key(self) -> str | None:
        return self._secret_to_str(self.deepseek_api_key)

    def get_nvidia_api_key(self) -> str | None:
        return self._secret_to_str(self.nvidia_api_key)

    def get_anthropic_api_key(self) -> str | None:
        return self._secret_to_str(self.anthropic_api_key)

    def get_default_model_name_for_provider(self, provider: str) -> str | None:
        attr_name = _PROVIDER_DEFAULT_MODEL_NAME_ATTRS.get(provider)
        if attr_name is None:
            return None
        return getattr(self, attr_name)

    def get_tool_model_provider(self) -> str:
        return self.tool_default_model_provider

    def get_tool_model_name(self) -> str:
        if self.tool_default_model_name:
            return self.tool_default_model_name

        provider = self.get_tool_model_provider()
        default_name = self.get_default_model_name_for_provider(provider)
        if default_name:
            return default_name
        raise ValueError(
            f"No default tool model_name configured for provider '{provider}'"
        )

    def get_tool_model_base_url(self) -> str | None:
        return self.tool_default_model_base_url

    def get_tool_model_api_key_env_name(self) -> str | None:
        if (
            isinstance(self.tool_default_model_api_key_env_name, str)
            and self.tool_default_model_api_key_env_name.strip()
        ):
            return self.tool_default_model_api_key_env_name.strip()
        return None

    def get_tool_model_temperature(self) -> float:
        return self.tool_default_model_temperature

    def get_tool_model_top_p(self) -> float:
        return self.tool_default_model_top_p

    def get_tool_model_max_tokens(self) -> int:
        return self.tool_default_model_max_tokens

    def get_embedding_api_key(self) -> str | None:
        if self.embedding_api_key:
            return self.embedding_api_key
        return self.get_openai_api_key()

    def get_embedding_base_url(self) -> str | None:
        if self.embedding_base_url:
            return self.embedding_base_url
        value = os.getenv("OPENAI_BASE_URL")
        if value:
            return value
        return None

    def get_env_skills_dirs(self) -> list[str]:
        """Get skill dirs only when explicitly set by environment."""
        if "skills_dirs" not in self.model_fields_set:
            return []
        return [item.strip() for item in self.skills_dirs if str(item).strip()]


def load_settings(*, include_env_file: bool = False) -> AgiwoSettings:
    """Load settings from current process environment.

    Args:
        include_env_file: Whether to include `.env` file loading.
            False loads from process environment only.
    """
    if include_env_file:
        return AgiwoSettings()
    return AgiwoSettings(_env_file=None)


# Global settings instance (singleton)
settings = load_settings(include_env_file=True)


__all__ = ["AgiwoSettings", "load_settings", "settings"]
