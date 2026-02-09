"""
Console server configuration.

Loaded from environment variables with AGIWO_CONSOLE_ prefix.
"""

from typing import Literal

from pydantic_settings import BaseSettings


class ConsoleConfig(BaseSettings):
    model_config = {"env_prefix": "AGIWO_CONSOLE_"}

    # Server
    host: str = "0.0.0.0"
    port: int = 8422

    # Storage backend: "sqlite" | "mongodb"
    storage_type: Literal["sqlite", "mongodb"] = "sqlite"

    # SQLite settings
    sqlite_db_path: str = "agiwo.db"
    sqlite_trace_collection: str = "agiwo_traces"

    # MongoDB settings
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "agiwo"
    mongodb_trace_collection: str = "traces"

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:3001"]
