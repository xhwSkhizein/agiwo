"""
Agent configuration options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
        - mongodb: {"uri": str, "db_name": str}
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


@dataclass
class AgentOptions:
    """
    Configuration for Agent execution behavior.

    All fields are pure configuration - no side effects in construction.

    Storage Configuration:
        run_step_storage: Configuration for run/step persistence.
            Defaults to in-memory storage. Use sqlite or mongodb for persistence.
        trace_storage: Configuration for trace/observability persistence.
            Defaults to None (tracing disabled). Enable for observability features.
    """
    # Root path for agent data - if empty, uses AgiwoSettings.root_path
    config_root: str = ""

    # Execution limits
    max_steps: int = 50
    run_timeout: int = 600  # seconds
    max_context_window_tokens: int = 200000
    max_tokens_per_run: int = 200000
    max_run_token_cost: float | None = None

    # Termination summary
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""

    # Skills
    enable_skill: bool = False
    skills_dirs: str | list[str] | None = None  # None means use env/default skill dirs

    # Memory
    relevant_memory_max_token: int = 2048  # max tokens for retrieved memories

    # Stream
    stream_cleanup_timeout: float = 300.0  # seconds

    # Compact
    compact_prompt: str = ""  # Custom compact prompt (overrides default)

    # Storage (new - replaces manual storage injection)
    run_step_storage: RunStepStorageConfig = field(default_factory=RunStepStorageConfig)
    trace_storage: TraceStorageConfig = field(default_factory=TraceStorageConfig)

    def get_effective_root_path(self) -> str:
        """Get the effective root path.
        
        Returns config_root if set, otherwise falls back to AgiwoSettings.root_path.
        """
        if self.config_root:
            return self.config_root
        return settings.root_path

    def get_configured_skills_dirs(self) -> list[Path]:
        """Get configured skill directories resolved against the effective root path."""
        normalized = normalize_skills_dirs(self.skills_dirs)
        if normalized:
            return resolve_skills_dirs(normalized, self.get_effective_root_path())

        default_dir = resolve_relative_path("skills", self.get_effective_root_path())
        if default_dir.exists():
            return [default_dir]
        return []


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
