"""Agent configuration: AgentConfig + AgentOptions (merged)."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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


class AgentOptions(BaseModel):
    """
    Configuration for Agent execution behavior.

    All fields are pure configuration - no side effects in construction.

    Storage Configuration:
        run_step_storage: Configuration for run/step persistence.
            Defaults to in-memory storage. Use sqlite or mongodb for persistence.
        trace_storage: Configuration for trace/observability persistence.
            Defaults to None (tracing disabled). Enable for observability features.
    """

    model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)

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
    run_step_storage: RunStepStorageConfig = Field(default_factory=RunStepStorageConfig)
    trace_storage: TraceStorageConfig = Field(default_factory=TraceStorageConfig)

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
