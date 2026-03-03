"""
Global settings from environment variables.
"""

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgiwoSettings(BaseSettings):
    """
    Global configuration loaded from environment variables.

    Environment variables should be prefixed with agiwo_
    Example: agiwo_DEBUG=true, agiwo_MONGO_URI=mongodb://localhost:27017
    """

    model_config = SettingsConfigDict(
        env_prefix="agiwo_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # Core settings
    debug: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

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
    openai_api_key: SecretStr | None = SecretStr(os.getenv("OPENAI_API_KEY"))
    openai_base_url: str | None = "https://api.openai.com/v1"
    openai_model_name: str | None = "gpt-4o-mini"

    # Deepseek
    deepseek_api_key: SecretStr | None = SecretStr(os.getenv("DEEPSEEK_API_KEY"))
    deepseek_base_url: str | None = "https://api.deepseek.com"
    deepseek_model_name: str | None = "deepseek-chat"

    # NVIDIA
    nvidia_api_key: SecretStr | None = SecretStr(os.getenv("NVIDIA_BUILD_API_KEY"))
    nvidia_base_url: str | None = "https://integrate.api.nvidia.com/v1"
    nvidia_model_name: str | None = "moonshotai/kimi-k2.5"  # z-ai/glm4.7

    # Anthropic
    anthropic_api_key: SecretStr | None = SecretStr(os.getenv("ANTHROPIC_API_KEY"))
    anthropic_base_url: str | None = "https://api.anthropic.com/v1"
    anthropic_model_name: str | None = "claude-3-5-sonnet-20240620"

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
    embedding_api_key: str | None = None  # Falls back to OPENAI_API_KEY
    embedding_base_url: str | None = None  # For openai-like providers
    local_embedding_model_path: str | None = None  # Path to GGUF model

    # Skills configuration (paths relative to root_path if not absolute)
    skills_dirs: list[str] = Field(
        default_factory=lambda: ["examples/skills", "skills"],
        description="Skill directories to scan (relative to root_path if not absolute)",
    )
    is_skills_enabled: bool = True

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


# Global settings instance (singleton)
settings = AgiwoSettings()


__all__ = ["AgiwoSettings", "settings"]
