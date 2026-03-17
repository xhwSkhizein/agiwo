"""Scheduler agent state storage package."""

from agiwo.scheduler.models import AgentStateStorageConfig
from agiwo.scheduler.store.base import AgentStateStorage
from agiwo.scheduler.store.memory import InMemoryAgentStateStorage
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
    raise ValueError(f"Unknown agent_state_storage_type: {storage_type}")


__all__ = [
    "AgentStateStorage",
    "InMemoryAgentStateStorage",
    "SQLiteAgentStateStorage",
    "create_agent_state_storage",
]
