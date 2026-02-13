"""
Agent configuration options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


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

    # Execution limits
    max_steps: int = 10
    run_timeout: int = 600  # seconds
    max_output_tokens: int = 8196

    # Termination summary
    enable_termination_summary: bool = True
    termination_summary_prompt: str = ""

    # Skills
    enable_skill: bool = False
    skills_dir: str | None = "~/.agent/skills"

    # Memory
    relevant_memory_max_token: int = 2048  # max tokens for retrieved memories

    # Stream
    stream_cleanup_timeout: float = 300.0  # seconds

    # Storage (new - replaces manual storage injection)
    run_step_storage: RunStepStorageConfig = field(default_factory=RunStepStorageConfig)
    trace_storage: TraceStorageConfig = field(default_factory=TraceStorageConfig)
