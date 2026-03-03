"""
Tool configuration classes for builtin tools.

All configuration classes support:
1. Constructor parameters (highest priority)
2. Environment variables (fallback)
3. Default values (lowest priority)
"""

import os
from dataclasses import dataclass, field
from typing import Any


def _get_env_float(key: str, default: str) -> float:
    """Get float from environment variable."""
    return float(os.getenv(key, default))


def _get_env_int(key: str, default: str) -> int:
    """Get int from environment variable."""
    return int(os.getenv(key, default))


def _get_env_bool(key: str, default: str) -> bool:
    """Get bool from environment variable."""
    return os.getenv(key, default).lower() == "true"


def _get_env_str(key: str, default: str) -> str:
    """Get string from environment variable."""
    return os.getenv(key, default)


@dataclass
class MemoryConfig:
    """Configuration for Memory System (indexing + retrieval)."""

    # Embedding provider (uses agiwo.embedding module)
    # Supported: "openai" | "openai-like" | "local" | "auto" | "disabled"
    embedding_provider: str = field(
        default_factory=lambda: _get_env_str("AGIWO_EMBEDDING_PROVIDER", "auto")
    )
    embedding_model: str = field(
        default_factory=lambda: _get_env_str(
            "AGIWO_EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    embedding_api_base: str = field(
        default_factory=lambda: _get_env_str("AGIWO_EMBEDDING_BASE_URL", "")
    )  # For openai-like providers
    embedding_api_key: str = field(
        default_factory=lambda: _get_env_str("AGIWO_EMBEDDING_API_KEY", "")
    )  # Falls back to OPENAI_API_KEY
    embedding_dims: int = field(
        default_factory=lambda: _get_env_int("AGIWO_EMBEDDING_DIMENSIONS", "1536")
    )

    # Chunking
    chunk_tokens: int = field(
        default_factory=lambda: _get_env_int("AGIWO_MEMORY_CHUNK_TOKENS", "400")
    )
    chunk_overlap_tokens: int = field(
        default_factory=lambda: _get_env_int("AGIWO_MEMORY_CHUNK_OVERLAP", "80")
    )

    # Search
    top_k: int = field(
        default_factory=lambda: _get_env_int("AGIWO_MEMORY_TOP_K", "5")
    )
    vector_weight: float = field(
        default_factory=lambda: _get_env_float("AGIWO_MEMORY_VECTOR_WEIGHT", "0.7")
    )
    bm25_weight: float = field(
        default_factory=lambda: _get_env_float("AGIWO_MEMORY_BM25_WEIGHT", "0.3")
    )

    # Temporal decay
    temporal_decay_enabled: bool = field(
        default_factory=lambda: _get_env_bool("AGIWO_MEMORY_TEMPORAL_DECAY", "false")
    )
    temporal_decay_half_life_days: float = field(
        default_factory=lambda: _get_env_float(
            "AGIWO_MEMORY_TEMPORAL_DECAY_HALF_LIFE", "30.0"
        )
    )

    # MMR re-ranking
    mmr_enabled: bool = field(
        default_factory=lambda: _get_env_bool("AGIWO_MEMORY_MMR_ENABLED", "false")
    )
    mmr_lambda: float = field(
        default_factory=lambda: _get_env_float("AGIWO_MEMORY_MMR_LAMBDA", "0.5")
    )


@dataclass
class WebSearchApiConfig:
    """Configuration for WebSearchApiTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_WEB_SEARCH_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_SEARCH_API_TIMEOUT", "30")
    )
    max_results: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_SEARCH_API_MAX_RESULTS", "10")
    )
    recency_days: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_SEARCH_API_RECENCY_DAYS", "7")
    )
    max_retries: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_SEARCH_API_MAX_RETRIES", "3")
    )


@dataclass
class WebReaderApiConfig:
    """Configuration for WebReaderApiTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_WEB_READER_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_READER_API_TIMEOUT", "30")
    )
    max_content_length: int = field(
        default_factory=lambda: _get_env_int(
            "AGIO_WEB_READER_API_MAX_CONTENT_LENGTH", "4096"
        )
    )
    max_retries: int = field(
        default_factory=lambda: _get_env_int("AGIO_WEB_READER_API_MAX_RETRIES", "3")
    )


def create_config_from_dict(config_class: type, data: dict) -> Any:
    """Create config object from dictionary, supporting partial fields."""
    import dataclasses
    import inspect

    # Get dataclass field names (more accurate)
    if dataclasses.is_dataclass(config_class):
        field_names = {f.name for f in dataclasses.fields(config_class)}
        valid_params = {k: v for k, v in data.items() if k in field_names}
    else:
        # Fallback to inspect method
        sig = inspect.signature(config_class.__init__)
        valid_params = {k: v for k, v in data.items() if k in sig.parameters}

    return config_class(**valid_params)


def filter_config_kwargs(config_class: type, kwargs: dict) -> dict:
    """Filter kwargs to only include valid config fields."""
    import dataclasses

    if dataclasses.is_dataclass(config_class):
        field_names = {f.name for f in dataclasses.fields(config_class)}
        return {k: v for k, v in kwargs.items() if k in field_names}
    return kwargs


@dataclass
class VlmConfig:
    """Configuration for VlmTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_VLM_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_VLM_API_TIMEOUT", "60")
    )
    image_fetch_timeout: int = field(
        default_factory=lambda: _get_env_int("AGIO_VLM_IMAGE_FETCH_TIMEOUT", "30")
    )
    max_retries: int = field(
        default_factory=lambda: _get_env_int("AGIO_VLM_API_MAX_RETRIES", "3")
    )


@dataclass
class ImageGenConfig:
    """Configuration for ImageGenTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_IMAGE_GEN_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_IMAGE_GEN_API_TIMEOUT", "120")
    )
    output_dir: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_IMAGE_GEN_OUTPUT_DIR", "./generated_images"
        )
    )
    default_size: str = field(
        default_factory=lambda: _get_env_str("AGIO_IMAGE_GEN_DEFAULT_SIZE", "1024x1024")
    )
    max_retries: int = field(
        default_factory=lambda: _get_env_int("AGIO_IMAGE_GEN_MAX_RETRIES", "3")
    )


@dataclass
class TtsConfig:
    """Configuration for TtsTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_TTS_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_TTS_API_TIMEOUT", "60")
    )
    output_dir: str = field(
        default_factory=lambda: _get_env_str("AGIO_TTS_OUTPUT_DIR", "./generated_audio")
    )
    default_voice: str = field(
        default_factory=lambda: _get_env_str("AGIO_TTS_DEFAULT_VOICE", "tongtong")
    )
    default_speed: float = field(
        default_factory=lambda: _get_env_float("AGIO_TTS_DEFAULT_SPEED", "1.0")
    )
    default_volume: float = field(
        default_factory=lambda: _get_env_float("AGIO_TTS_DEFAULT_VOLUME", "1.0")
    )


@dataclass
class AsrConfig:
    """Configuration for AsrTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_ASR_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_ASR_API_TIMEOUT", "60")
    )
    audio_fetch_timeout: int = field(
        default_factory=lambda: _get_env_int("AGIO_ASR_AUDIO_FETCH_TIMEOUT", "30")
    )


@dataclass
class VideoGenConfig:
    """Configuration for VideoGenTool."""

    base_url: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_VIDEO_GEN_API_BASE_URL", "https://oaiaiai.space.z.ai"
        )
    )
    timeout_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_API_TIMEOUT", "60")
    )
    output_dir: str = field(
        default_factory=lambda: _get_env_str(
            "AGIO_VIDEO_GEN_OUTPUT_DIR", "./generated_videos"
        )
    )
    default_quality: str = field(
        default_factory=lambda: _get_env_str("AGIO_VIDEO_GEN_DEFAULT_QUALITY", "speed")
    )
    default_with_audio: bool = field(
        default_factory=lambda: _get_env_bool(
            "AGIO_VIDEO_GEN_DEFAULT_WITH_AUDIO", "false"
        )
    )
    default_size: str = field(
        default_factory=lambda: _get_env_str("AGIO_VIDEO_GEN_DEFAULT_SIZE", "1920x1080")
    )
    default_fps: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_DEFAULT_FPS", "30")
    )
    default_duration: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_DEFAULT_DURATION", "5")
    )
    poll_interval_seconds: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_POLL_INTERVAL", "5")
    )
    max_poll_count: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_MAX_POLL_COUNT", "120")
    )
    download_video: bool = field(
        default_factory=lambda: _get_env_bool("AGIO_VIDEO_GEN_DOWNLOAD_VIDEO", "true")
    )
    download_timeout: int = field(
        default_factory=lambda: _get_env_int("AGIO_VIDEO_GEN_DOWNLOAD_TIMEOUT", "120")
    )


__all__ = [
    "MemoryConfig",
    "WebSearchApiConfig",
    "WebReaderApiConfig",
    "VlmConfig",
    "ImageGenConfig",
    "TtsConfig",
    "AsrConfig",
    "VideoGenConfig",
    "create_config_from_dict",
    "filter_config_kwargs",
]
