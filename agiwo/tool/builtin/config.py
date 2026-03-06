"""
Tool configuration classes for builtin tools.

All configuration classes support:
1. Constructor parameters (highest priority)
2. Environment variables (fallback)
3. Default values (lowest priority)
"""

from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from agiwo.config.settings import settings


@dataclass
class MemoryConfig:
    """Configuration for Memory System (indexing + retrieval)."""

    # Embedding provider (uses agiwo.embedding module)
    # Supported: "openai" | "openai-like" | "local" | "auto" | "disabled"
    embedding_provider: str = field(default_factory=lambda: settings.embedding_provider)
    embedding_model: str = field(default_factory=lambda: settings.embedding_model)
    embedding_api_base: str = field(
        default_factory=lambda: settings.embedding_base_url or ""
    )  # For openai-like providers
    embedding_api_key: str = field(
        default_factory=lambda: settings.get_embedding_api_key() or ""
    )  # Falls back to OPENAI_API_KEY
    embedding_dims: int = field(default_factory=lambda: settings.embedding_dimensions)

    # Chunking
    chunk_tokens: int = field(default_factory=lambda: settings.memory_chunk_tokens)
    chunk_overlap_tokens: int = field(
        default_factory=lambda: settings.memory_chunk_overlap
    )

    # Search
    top_k: int = field(default_factory=lambda: settings.memory_top_k)
    vector_weight: float = field(default_factory=lambda: settings.memory_vector_weight)
    bm25_weight: float = field(default_factory=lambda: settings.memory_bm25_weight)

    # Temporal decay
    temporal_decay_enabled: bool = field(
        default_factory=lambda: settings.memory_temporal_decay
    )
    temporal_decay_half_life_days: float = field(
        default_factory=lambda: settings.memory_temporal_decay_half_life
    )

    # MMR re-ranking
    mmr_enabled: bool = field(default_factory=lambda: settings.memory_mmr_enabled)
    mmr_lambda: float = field(default_factory=lambda: settings.memory_mmr_lambda)


@dataclass
class WebSearchApiConfig:
    """Configuration for WebSearchTool."""

    base_url: str = field(default_factory=lambda: settings.web_search_api_base_url)

    serper_api_key: SecretStr | None = field(
        default_factory=lambda: settings.web_search_serper_api_key
    )

    timeout_seconds: int = field(
        default_factory=lambda: settings.web_search_api_timeout
    )
    max_results: int = field(
        default_factory=lambda: settings.web_search_api_max_results
    )
    recency_days: int = field(
        default_factory=lambda: settings.web_search_api_recency_days
    )
    max_retries: int = field(
        default_factory=lambda: settings.web_search_api_max_retries
    )


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
        default_factory=lambda: settings.get_tool_model_provider("web_reader")
    )
    model_name: str = field(
        default_factory=lambda: settings.get_tool_model_name("web_reader")
    )
    model_base_url: str | None = field(
        default_factory=lambda: settings.get_tool_model_base_url("web_reader")
    )
    api_key_env_name: str | None = field(
        default_factory=lambda: settings.get_tool_model_api_key_env_name("web_reader")
    )
    model_temperature: float = field(
        default_factory=lambda: settings.get_tool_model_temperature("web_reader")
    )
    model_top_p: float = field(
        default_factory=lambda: settings.get_tool_model_top_p("web_reader")
    )
    model_max_tokens: int = field(
        default_factory=lambda: settings.get_tool_model_max_tokens("web_reader")
    )
    headless: bool = False
    wait_strategy: str = "domcontentloaded"
    max_browsers: int = 1
    browser_idle_ttl_seconds: int = 300
    browser_max_uses: int = 20
    save_login_state: bool = True
    user_data_dir: str = "browser_data"
    browser_launch_timeout: int = 30
    auto_close_browser: bool = True


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


__all__ = [
    "MemoryConfig",
    "WebSearchApiConfig",
    "WebReaderApiConfig",
    "create_config_from_dict",
    "filter_config_kwargs",
]
