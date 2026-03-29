"""Agent configuration: AgentConfig + AgentOptions (merged)."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agiwo.config.settings import settings


@dataclass
class RunStepStorageConfig:
    """Configuration for RunStepStorage.

    storage_type: "memory" | "sqlite" | "mongodb"
        - memory: In-memory storage (default, no persistence)
        - sqlite: SQLite database storage
        - mongodb: MongoDB storage
    config: storage-specific configuration
        - sqlite: {"db_path": str}
        - mongodb: {"mongo_uri": str, "db_name": str}
    """

    storage_type: Literal["memory", "sqlite", "mongodb"] = "memory"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStorageConfig:
    """Configuration for TraceStorage.

    storage_type: "memory" | "sqlite" | "mongodb" | None
        - None: Tracing disabled (default)
        - memory: In-memory storage (with ring buffer)
        - sqlite: SQLite database storage
        - mongodb: MongoDB storage
    config: storage-specific configuration
        - sqlite: {"db_path": str, "collection_name": str, "buffer_size": int}
        - mongodb: {"mongo_uri": str, "db_name": str, "collection_name": str, "buffer_size": int}
    """

    storage_type: Literal["memory", "sqlite", "mongodb"] | None = None
    config: dict[str, Any] = field(default_factory=dict)


class AgentStorageOptions(BaseModel):
    """Infrastructure / persistence configuration, separated from runtime behaviour."""

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    run_step_storage: RunStepStorageConfig = Field(default_factory=RunStepStorageConfig)
    trace_storage: TraceStorageConfig = Field(default_factory=TraceStorageConfig)


class AgentOptions(BaseModel):
    """Configuration for Agent execution behavior.

    All fields are pure configuration - no side effects in construction.
    Storage configuration lives in ``storage``.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def _migrate_storage_fields(cls, data: Any) -> Any:
        """Accept legacy top-level storage kwargs and nest them into ``storage``."""
        if not isinstance(data, dict):
            return data
        storage = data.get("storage") or {}
        if isinstance(storage, AgentStorageOptions):
            storage = {
                "run_step_storage": storage.run_step_storage,
                "trace_storage": storage.trace_storage,
            }
        for key in ("run_step_storage", "trace_storage"):
            if key in data and key not in storage:
                storage[key] = data.pop(key)
        if storage:
            data["storage"] = storage
        return data

    config_root: str = ""
    max_steps: int = 50
    run_timeout: int = 600
    max_input_tokens_per_call: int | None = None
    max_run_cost: float | None = None
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""
    enable_skill: bool = False
    skills_dirs: str | list[str] | None = None
    relevant_memory_max_token: int = 2048
    stream_cleanup_timeout: float = 300.0
    compact_prompt: str = ""
    storage: AgentStorageOptions = Field(default_factory=AgentStorageOptions)

    # Deprecated shims -- remove after one release cycle
    @property
    def run_step_storage(self) -> RunStepStorageConfig:
        return self.storage.run_step_storage

    @property
    def trace_storage(self) -> TraceStorageConfig:
        return self.storage.trace_storage

    def get_effective_root_path(self) -> str:
        if self.config_root:
            return self.config_root
        return settings.root_path


@dataclass
class AgentConfig:
    """Canonical public Agent configuration without live runtime objects."""

    name: str
    description: str = ""
    system_prompt: str = ""
    options: AgentOptions = field(default_factory=AgentOptions)
    disabled_sdk_tool_names: set[str] = field(default_factory=set)
