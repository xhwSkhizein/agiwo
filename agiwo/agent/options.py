"""
Agent configuration options.
"""

from dataclasses import dataclass, field
from pathlib import Path
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

    def get_configured_skills_dirs(self) -> list[Path]:
        normalized = normalize_skills_dirs(self.skills_dirs)
        if normalized:
            return resolve_skills_dirs(normalized, self.get_effective_root_path())

        default_dir = resolve_relative_path("skills", self.get_effective_root_path())
        if default_dir.exists():
            return [default_dir]
        return []


def sanitize_agent_options_data(
    data: Any,
    *,
    preserve_non_dict: bool = False,
) -> Any:
    if not isinstance(data, dict):
        return data if preserve_non_dict else {}
    sanitized = dict(data)
    skills_dirs = normalize_skills_dirs(sanitized.get("skills_dirs"))
    if skills_dirs is None:
        sanitized.pop("skills_dirs", None)
    else:
        sanitized["skills_dirs"] = skills_dirs
    return sanitized


class AgentOptionsInput(AgentOptions):
    max_steps: int = Field(default=50, ge=1)
    run_timeout: int = Field(default=600, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_skill: bool = Field(default_factory=lambda: settings.is_skills_enabled)
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int = Field(default=2048, ge=1)
    stream_cleanup_timeout: float = Field(default=300.0, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return sanitize_agent_options_data(data, preserve_non_dict=True)

    def to_agent_options(
        self,
        *,
        run_step_storage: RunStepStorageConfig,
        trace_storage: TraceStorageConfig,
    ) -> AgentOptions:
        data = self.model_dump(exclude_none=True)
        data["run_step_storage"] = run_step_storage
        data["trace_storage"] = trace_storage
        return AgentOptions(**data)


class AgentOptionsPatch(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config_root: str | None = None
    max_steps: int | None = Field(default=None, ge=1)
    run_timeout: int | None = Field(default=None, ge=1)
    max_input_tokens_per_call: int | None = Field(default=None, ge=1)
    max_run_cost: float | None = Field(default=None, ge=0)
    enable_termination_summary: bool | None = None
    termination_summary_prompt: str | None = None
    enable_skill: bool | None = None
    skills_dirs: list[str] | None = None
    relevant_memory_max_token: int | None = Field(default=None, ge=1)
    stream_cleanup_timeout: float | None = Field(default=None, gt=0)
    compact_prompt: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        return sanitize_agent_options_data(data, preserve_non_dict=True)


def normalize_skills_dirs(value: str | list[str] | None) -> list[str] | None:
    """Normalize skill directory config to a cleaned list of strings."""
    if value is None:
        return None

    entries = [value] if isinstance(value, str) else value
    normalized: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        stripped = entry.strip()
        if stripped:
            normalized.append(stripped)

    return normalized or None


def resolve_relative_path(path: str, root_path: str) -> Path:
    """Resolve a path relative to root_path when it is not absolute."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(root_path).expanduser().resolve()
    return (root / candidate).resolve()


def resolve_skills_dirs(skills_dirs: list[str], root_path: str) -> list[Path]:
    """Resolve skill directories against root_path and deduplicate them."""
    resolved: list[Path] = []
    seen: set[str] = set()
    for entry in skills_dirs:
        path = resolve_relative_path(entry, root_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved
