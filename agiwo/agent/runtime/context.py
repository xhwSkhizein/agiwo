"""Run-scoped identity, ledger, and IO dependencies."""

import copy
import time
from dataclasses import dataclass, replace
from typing import Any

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import HookRegistry
from agiwo.agent.models.run import RunIdentity, RunLedger
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool
from agiwo.utils.abort_signal import AbortSignal


@dataclass
class RunRuntime:
    """Ephemeral runtime state for a single run."""

    session_runtime: SessionRuntime
    config: AgentOptions
    hooks: HookRegistry
    model: Model
    tools_map: dict[str, BaseTool]
    abort_signal: AbortSignal | None
    root_path: str
    compact_start_seq: int
    max_input_tokens_per_call: int
    max_context_window: int | None
    compact_prompt: str | None


class RunContext:
    """Facade for run identity, mutable ledger state, and IO dependencies."""

    __slots__ = ("_identity", "ledger", "_session_runtime", "config", "hooks")

    def __init__(
        self,
        *,
        identity: RunIdentity,
        ledger: RunLedger | None = None,
        session_runtime: SessionRuntime,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        self._identity = identity
        self.ledger = ledger or RunLedger(messages=list(messages or []))
        self._session_runtime = session_runtime
        # Overwritten by execute_run with the effective options/hooks for
        # the run. Defaults let early readers (prompt assembly, trace setup)
        # operate safely if they fire before execute_run has injected them.
        self.config = AgentOptions()
        self.hooks = HookRegistry()

    @property
    def run_id(self) -> str:
        return self._identity.run_id

    @property
    def agent_id(self) -> str:
        return self._identity.agent_id

    @property
    def agent_name(self) -> str:
        return self._identity.agent_name

    @property
    def user_id(self) -> str | None:
        return self._identity.user_id

    @property
    def depth(self) -> int:
        return self._identity.depth

    @property
    def parent_run_id(self) -> str | None:
        return self._identity.parent_run_id

    @property
    def timeout_at(self) -> float | None:
        return self._identity.timeout_at

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._identity.metadata)

    def update_metadata(self, updates: dict[str, Any]) -> None:
        if not updates:
            return
        merged = dict(self._identity.metadata)
        merged.update(updates)
        self._identity = replace(self._identity, metadata=merged)

    @property
    def session_runtime(self) -> SessionRuntime:
        return self._session_runtime

    @property
    def session_id(self) -> str:
        return self._session_runtime.session_id

    @property
    def trace_id(self) -> str | None:
        tr = self._session_runtime.trace_runtime
        return tr.trace_id if tr is not None else None

    def snapshot_messages(self) -> list[dict[str, Any]]:
        """Return a deep copy of the current messages for sending to the LLM."""
        return copy.deepcopy(self.ledger.messages)

    def copy_tool_schemas(self) -> list[dict[str, Any]] | None:
        return copy.deepcopy(self.ledger.tool_schemas)

    @property
    def elapsed(self) -> float:
        return time.time() - self.ledger.start_time

    @property
    def is_terminal(self) -> bool:
        return self.ledger.termination_reason is not None


__all__ = ["RunContext", "RunRuntime"]
