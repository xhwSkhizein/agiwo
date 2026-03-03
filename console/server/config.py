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
    feishu_ack_reaction_emoji: str = "FIREWORKS"  # ref: https://open.feishu.cn/document/server-docs/im-v1/message-reaction/emojis-introduce
    feishu_ack_fallback_text: str = "收到，正在处理。"
