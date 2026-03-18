"""Scheduler agent state storage package."""

from agiwo.scheduler.models import AgentStateStorageConfig
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage
from agiwo.scheduler.store.mongo import MongoAgentStateStorage
from agiwo.scheduler.store.sqlite import SQLiteAgentStateStorage


def create_agent_state_storage(config: AgentStateStorageConfig) -> AgentStateStorage:
    """Factory: create AgentStateStorage from configuration."""
    storage_type = config.storage_type
    cfg = config.config

    if storage_type == "memory":
        return InMemoryAgentStateStorage()
    if storage_type == "sqlite":
        db_path = cfg.get("db_path", "scheduler.db")
        return SQLiteAgentStateStorage(db_path=db_path)
    if storage_type == "mongodb":
        uri = cfg.get("uri", "mongodb://localhost:27017")
        db_name = cfg.get("db_name", "agiwo")
        return MongoAgentStateStorage(uri=uri, db_name=db_name)
    raise ValueError(f"Unknown agent_state_storage_type: {storage_type}")


__all__ = [
    "AgentStateStorage",
    "InMemoryAgentStateStorage",
    "MongoAgentStateStorage",
    "SQLiteAgentStateStorage",
    "create_agent_state_storage",
]
