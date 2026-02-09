"""
Global settings from environment variables.
"""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

import os


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

    # Storage settings
    mongo_uri: str | None = None
    mongo_db_name: str | None = "agiwo"
    sqlite_db_path: str | None = "~/.agiwo/agiwo.db"

    # Vector DB settings
    vector_db_path: str | None = "~/.agiwo/vector_db"

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

    # Skills configuration
    skills_dirs: list[str] = Field(
        default_factory=lambda: ["examples/skills", "~/.agiwo/skills"],
        description="Skill directories to scan",
    )
    is_skills_enabled: bool = True


# Global settings instance (singleton)
settings = AgiwoSettings()


__all__ = ["AgiwoSettings", "settings"]
