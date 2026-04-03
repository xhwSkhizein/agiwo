"""Agent configuration: AgentConfig + AgentOptions (merged)."""

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agiwo.config.settings import settings
from agiwo.skill.allowlist import (
    normalize_allowed_skills,
    validate_expanded_allowed_skills,
)


@dataclass
class RunStepStorageConfig:
    """Configuration for RunStepStorage.

    storage_type: backend selector (e.g. ``memory``, ``sqlite``).
        - memory: in-process, no persistence
        - sqlite: SQLite persistence
    config: backend-specific options (e.g. ``db_path`` for sqlite).
    """

    storage_type: Literal["memory", "sqlite", "mongodb"] = "memory"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceStorageConfig:
    """Configuration for TraceStorage.

    storage_type: ``None`` disables tracing (default). Otherwise a backend
        selector such as ``memory`` or ``sqlite``.
        - memory: in-process ring buffer
        - sqlite: SQLite persistence
    config: backend-specific options (paths, collection names, buffer size, etc.).
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

    config_root: str = ""
    max_steps: int = 50
    run_timeout: int = 600
    max_input_tokens_per_call: int | None = None
    max_run_cost: float | None = None
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""
    relevant_memory_max_token: int = 2048
    stream_cleanup_timeout: float = 300.0
    compact_prompt: str = ""
    storage: AgentStorageOptions = Field(default_factory=AgentStorageOptions)

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
    allowed_skills: list[str] | None = None

    def __post_init__(self) -> None:
        normalized = normalize_allowed_skills(self.allowed_skills)
        validate_expanded_allowed_skills(normalized)
        self.allowed_skills = list(normalized) if normalized is not None else None
