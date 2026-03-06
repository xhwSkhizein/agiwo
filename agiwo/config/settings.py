"""
Global settings from environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Core settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_json: bool = False

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

    # Model Provider Settings
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

    # Tool model defaults
    tool_default_model_provider: Literal[
        "openai",
        "openai-compatible",
        "deepseek",
        "anthropic",
        "anthropic-compatible",
        "nvidia",
    ] = Field(
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
        validation_alias=AliasChoices(
            "AGIWO_TOOL_DEFAULT_MODEL_API_KEY_ENV_NAME"
        ),
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
    web_reader_model_provider: Literal[
        "openai",
        "openai-compatible",
        "deepseek",
        "anthropic",
        "anthropic-compatible",
        "nvidia",
        "bedrock-anthropic",
    ] | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_PROVIDER"),
    )
    web_reader_model_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_NAME"),
    )
    web_reader_model_base_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_BASE_URL"),
    )
    web_reader_model_api_key_env_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AGIWO_TOOL_WEB_READER_MODEL_API_KEY_ENV_NAME"
        ),
    )
    web_reader_model_temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_TEMPERATURE"),
    )
    web_reader_model_top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_TOP_P"),
    )
    web_reader_model_max_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("AGIWO_TOOL_WEB_READER_MODEL_MAX_TOKENS"),
    )

    # Observability - OTLP Export
    otlp_enabled: bool = False
    otlp_endpoint: str | None = None  # e.g., "http://localhost:4317" for gRPC
    otlp_protocol: Literal["grpc", "http"] = "grpc"
    otlp_headers: dict[str, str] = Field(default_factory=dict)
    otlp_sampling_rate: float = Field(
        default=1.0, ge=0.0, le=1.0
    )  # 1.0 = 100% sampling

    # Embedding settings
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

    # Memory retrieval settings
    memory_chunk_tokens: int = 400
    memory_chunk_overlap: int = 80
    memory_top_k: int = 5
    memory_vector_weight: float = 0.7
    memory_bm25_weight: float = 0.3
    memory_temporal_decay: bool = False
    memory_temporal_decay_half_life: float = 30.0
    memory_mmr_enabled: bool = False
    memory_mmr_lambda: float = 0.5

    # Builtin tools settings
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
    vlm_api_base_url: str = Field(
        default="https://oaiaiai.space.z.ai",
        validation_alias=AliasChoices("AGIWO_TOOL_VLM_API_BASE_URL"),
    )
    vlm_api_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("AGIWO_TOOL_VLM_API_TIMEOUT"),
    )
    vlm_image_fetch_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("AGIWO_TOOL_VLM_IMAGE_FETCH_TIMEOUT"),
    )
    vlm_api_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("AGIWO_TOOL_VLM_API_MAX_RETRIES"),
    )
    image_gen_api_base_url: str = Field(
        default="https://oaiaiai.space.z.ai",
        validation_alias=AliasChoices("AGIWO_TOOL_IMAGE_GEN_API_BASE_URL"),
    )
    image_gen_api_timeout: int = Field(
        default=120,
        validation_alias=AliasChoices("AGIWO_TOOL_IMAGE_GEN_API_TIMEOUT"),
    )
    image_gen_output_dir: str = Field(
        default="./generated_images",
        validation_alias=AliasChoices("AGIWO_TOOL_IMAGE_GEN_OUTPUT_DIR"),
    )
    image_gen_default_size: str = Field(
        default="1024x1024",
        validation_alias=AliasChoices("AGIWO_TOOL_IMAGE_GEN_DEFAULT_SIZE"),
    )
    image_gen_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("AGIWO_TOOL_IMAGE_GEN_MAX_RETRIES"),
    )
    tts_api_base_url: str = Field(
        default="https://oaiaiai.space.z.ai",
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_API_BASE_URL"),
    )
    tts_api_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_API_TIMEOUT"),
    )
    tts_output_dir: str = Field(
        default="./generated_audio",
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_OUTPUT_DIR"),
    )
    tts_default_voice: str = Field(
        default="tongtong",
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_DEFAULT_VOICE"),
    )
    tts_default_speed: float = Field(
        default=1.0,
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_DEFAULT_SPEED"),
    )
    tts_default_volume: float = Field(
        default=1.0,
        validation_alias=AliasChoices("AGIWO_TOOL_TTS_DEFAULT_VOLUME"),
    )
    asr_api_base_url: str = Field(
        default="https://oaiaiai.space.z.ai",
        validation_alias=AliasChoices("AGIWO_TOOL_ASR_API_BASE_URL"),
    )
    asr_api_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("AGIWO_TOOL_ASR_API_TIMEOUT"),
    )
    asr_audio_fetch_timeout: int = Field(
        default=30,
        validation_alias=AliasChoices("AGIWO_TOOL_ASR_AUDIO_FETCH_TIMEOUT"),
    )
    video_gen_api_base_url: str = Field(
        default="https://oaiaiai.space.z.ai",
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_API_BASE_URL"),
    )
    video_gen_api_timeout: int = Field(
        default=60,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_API_TIMEOUT"),
    )
    video_gen_output_dir: str = Field(
        default="./generated_videos",
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_OUTPUT_DIR"),
    )
    video_gen_default_quality: str = Field(
        default="speed",
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DEFAULT_QUALITY"),
    )
    video_gen_default_with_audio: bool = Field(
        default=False,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DEFAULT_WITH_AUDIO"),
    )
    video_gen_default_size: str = Field(
        default="1920x1080",
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DEFAULT_SIZE"),
    )
    video_gen_default_fps: int = Field(
        default=30,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DEFAULT_FPS"),
    )
    video_gen_default_duration: int = Field(
        default=5,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DEFAULT_DURATION"),
    )
    video_gen_poll_interval: int = Field(
        default=5,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_POLL_INTERVAL"),
    )
    video_gen_max_poll_count: int = Field(
        default=120,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_MAX_POLL_COUNT"),
    )
    video_gen_download_video: bool = Field(
        default=True,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DOWNLOAD_VIDEO"),
    )
    video_gen_download_timeout: int = Field(
        default=120,
        validation_alias=AliasChoices("AGIWO_TOOL_VIDEO_GEN_DOWNLOAD_TIMEOUT"),
    )

    # Skills configuration (paths relative to root_path if not absolute)
    skills_dirs: list[str] = Field(
        default_factory=lambda: ["examples/skills", "skills"],
        description="Skill directories to scan (relative to root_path if not absolute)",
    )
    is_skills_enabled: bool = True

    # Scheduler pending-event debounce settings
    event_debounce_min_count: int = Field(
        default=1,
        description="Minimum number of pending events to trigger a debounce wake",
    )
    event_debounce_max_wait_seconds: float = Field(
        default=30.0,
        description="Maximum seconds to wait before triggering a debounce wake regardless of count",
    )

    # Context Compact settings
    compact_threshold_ratio: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Trigger compact when context tokens reach this ratio of max_context_window_tokens",
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
        if provider == "openai":
            return self.openai_model_name
        if provider == "deepseek":
            return self.deepseek_model_name
        if provider == "anthropic":
            return self.anthropic_model_name
        if provider == "nvidia":
            return self.nvidia_model_name
        return None

    def get_tool_model_provider(self, tool_name: str) -> str:
        override = getattr(self, f"{tool_name}_model_provider", None)
        if override:
            return override
        return self.tool_default_model_provider

    def get_tool_model_name(self, tool_name: str) -> str:
        override = getattr(self, f"{tool_name}_model_name", None)
        if override:
            return override
        if self.tool_default_model_name:
            return self.tool_default_model_name

        provider = self.get_tool_model_provider(tool_name)
        default_name = self.get_default_model_name_for_provider(provider)
        if default_name:
            return default_name
        raise ValueError(
            f"No default model_name configured for tool '{tool_name}' with provider '{provider}'"
        )

    def get_tool_model_base_url(self, tool_name: str) -> str | None:
        override = getattr(self, f"{tool_name}_model_base_url", None)
        if override:
            return override
        return self.tool_default_model_base_url

    def get_tool_model_api_key_env_name(self, tool_name: str) -> str | None:
        override = getattr(self, f"{tool_name}_model_api_key_env_name", None)
        if isinstance(override, str) and override.strip():
            return override.strip()
        if (
            isinstance(self.tool_default_model_api_key_env_name, str)
            and self.tool_default_model_api_key_env_name.strip()
        ):
            return self.tool_default_model_api_key_env_name.strip()
        return None

    def get_tool_model_temperature(self, tool_name: str) -> float:
        override = getattr(self, f"{tool_name}_model_temperature", None)
        if override is not None:
            return override
        return self.tool_default_model_temperature

    def get_tool_model_top_p(self, tool_name: str) -> float:
        override = getattr(self, f"{tool_name}_model_top_p", None)
        if override is not None:
            return override
        return self.tool_default_model_top_p

    def get_tool_model_max_tokens(self, tool_name: str) -> int:
        override = getattr(self, f"{tool_name}_model_max_tokens", None)
        if override is not None:
            return override
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
