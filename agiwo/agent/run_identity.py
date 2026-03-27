from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    agent_id: str
    agent_name: str
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
