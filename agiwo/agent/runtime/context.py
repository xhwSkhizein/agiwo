"""Run-scoped identity, ledger, and IO dependencies."""

import copy
import time
from dataclasses import dataclass, field
from typing import Any

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.models.run import RunLedger
from agiwo.agent.runtime.session import SessionRuntime


@dataclass
class RunContext:
    """All identity, mutable state, and IO deps for a single agent run.

    Identity fields live directly on this class.
    Mutable accounting state lives in ``ledger``.
    """

    session_runtime: SessionRuntime
    ledger: RunLedger
    config: AgentOptions
    hooks: AgentHooks

    run_id: str = ""
    agent_id: str = ""
    agent_name: str = ""
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        *,
        session_runtime: SessionRuntime,
        run_id: str,
        agent_id: str,
        agent_name: str,
        user_id: str | None = None,
        depth: int = 0,
        parent_run_id: str | None = None,
        timeout_at: float | None = None,
        metadata: dict[str, Any] | None = None,
        config: AgentOptions | None = None,
        hooks: AgentHooks | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session_runtime = session_runtime
        self.run_id = run_id
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.user_id = user_id
        self.depth = depth
        self.parent_run_id = parent_run_id
        self.timeout_at = timeout_at
        self.metadata = dict(metadata or {})
        self.ledger = RunLedger(messages=list(messages or []))
        self.config = config or AgentOptions()
        self.hooks = hooks or AgentHooks()

    @property
    def session_id(self) -> str:
        return self.session_runtime.session_id

    @property
    def trace_id(self) -> str | None:
        tr = self.session_runtime.trace_runtime
        return tr.trace_id if tr is not None else None

    def snapshot_messages(self) -> list[dict[str, Any]]:
        """Return a deep copy of the current messages for sending to the LLM."""
        return copy.deepcopy(self.ledger.messages)

    copy_messages = snapshot_messages  # backward-compat alias

    def copy_tool_schemas(self) -> list[dict[str, Any]] | None:
        return copy.deepcopy(self.ledger.tool_schemas)

    @property
    def elapsed(self) -> float:
        return time.time() - self.ledger.start_time

    @property
    def is_terminal(self) -> bool:
        return self.ledger.termination_reason is not None


__all__ = ["RunContext"]
