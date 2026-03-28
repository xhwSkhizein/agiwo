"""Run-scoped identity, ledger, and IO dependencies."""

import asyncio
import copy
import time
from dataclasses import dataclass
from typing import Any

from agiwo.agent.models.compact import CompactMetadata
from agiwo.agent.models.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.models.run import RunIdentity, RunLedger, TerminationReason
from agiwo.agent.runtime.session import SessionRuntime


@dataclass(init=False)
class RunContext:
    """All identity, mutable state, and IO deps for a single agent run."""

    session_runtime: SessionRuntime
    identity: RunIdentity
    ledger: RunLedger
    config: AgentOptions
    hooks: AgentHooks

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
        self.identity = RunIdentity(
            run_id=run_id,
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            depth=depth,
            parent_run_id=parent_run_id,
            timeout_at=timeout_at,
            metadata=dict(metadata or {}),
        )
        self.ledger = RunLedger(messages=list(messages or []))
        self.config = config or AgentOptions()
        self.hooks = hooks or AgentHooks()

    @property
    def run_id(self) -> str:
        return self.identity.run_id

    @property
    def agent_id(self) -> str:
        return self.identity.agent_id

    @property
    def agent_name(self) -> str:
        return self.identity.agent_name

    @property
    def user_id(self) -> str | None:
        return self.identity.user_id

    @property
    def depth(self) -> int:
        return self.identity.depth

    @property
    def parent_run_id(self) -> str | None:
        return self.identity.parent_run_id

    @property
    def timeout_at(self) -> float | None:
        return self.identity.timeout_at

    @property
    def metadata(self) -> dict[str, Any]:
        return self.identity.metadata

    @property
    def session_id(self) -> str:
        return self.session_runtime.session_id

    @property
    def steering_queue(self) -> asyncio.Queue[object]:
        return self.session_runtime.steering_queue

    @property
    def trace_id(self) -> str | None:
        trace_runtime = self.session_runtime.trace_runtime
        return trace_runtime.trace_id if trace_runtime is not None else None

    @property
    def messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.copy_messages())

    def copy_messages(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.ledger.messages)

    @property
    def tool_schemas(self) -> tuple[dict[str, Any], ...] | None:
        schemas = self.copy_tool_schemas()
        return tuple(schemas) if schemas is not None else None

    def copy_tool_schemas(self) -> list[dict[str, Any]] | None:
        return copy.deepcopy(self.ledger.tool_schemas)

    @property
    def start_time(self) -> float:
        return self.ledger.start_time

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self.ledger.termination_reason

    @property
    def total_tokens(self) -> int:
        return self.ledger.total_tokens

    @property
    def input_tokens(self) -> int:
        return self.ledger.input_tokens

    @property
    def output_tokens(self) -> int:
        return self.ledger.output_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self.ledger.cache_read_tokens

    @property
    def cache_creation_tokens(self) -> int:
        return self.ledger.cache_creation_tokens

    @property
    def token_cost(self) -> float:
        return self.ledger.token_cost

    @property
    def steps_count(self) -> int:
        return self.ledger.steps_count

    @property
    def tool_calls_count(self) -> int:
        return self.ledger.tool_calls_count

    @property
    def assistant_steps_count(self) -> int:
        return self.ledger.assistant_steps_count

    @property
    def response_content(self) -> str | None:
        return self.ledger.response_content

    @property
    def last_compact_metadata(self) -> CompactMetadata | None:
        return self.ledger.last_compact_metadata

    @property
    def elapsed(self) -> float:
        return time.time() - self.ledger.start_time

    @property
    def is_terminal(self) -> bool:
        return self.ledger.termination_reason is not None


__all__ = ["RunContext"]
