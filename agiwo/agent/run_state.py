"""RunContext — session runtime plus split identity and mutable ledger."""

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from agiwo.agent.compact_types import CompactMetadata
from agiwo.agent.config import AgentOptions
from agiwo.agent.hooks import AgentHooks
from agiwo.agent.input import UserInput
from agiwo.agent.input_codec import extract_text
from agiwo.agent.run_identity import RunIdentity
from agiwo.agent.run_ledger import RunLedger
from agiwo.agent.storage.base import RunStepStorage
from agiwo.agent.storage.session import SessionStorage
from agiwo.agent.trace_writer import AgentTraceCollector
from agiwo.agent.types import AgentStreamItem, TerminationReason
from agiwo.utils.abort_signal import AbortSignal


class SessionRuntime:
    """Session-scoped runtime state shared by root and child runs."""

    _SENTINEL = object()

    def __init__(
        self,
        *,
        session_id: str,
        run_step_storage: RunStepStorage,
        session_storage: SessionStorage,
        trace_runtime: AgentTraceCollector | None = None,
        abort_signal: AbortSignal | None = None,
        steering_queue: asyncio.Queue[object] | None = None,
    ) -> None:
        self.session_id = session_id
        self.run_step_storage = run_step_storage
        self.session_storage = session_storage
        self.trace_runtime = trace_runtime
        self.abort_signal = abort_signal or AbortSignal()
        self.steering_queue = steering_queue or asyncio.Queue()
        self._subscribers: set[asyncio.Queue[AgentStreamItem | object]] = set()
        self._closed = False

    async def allocate_sequence(self) -> int:
        return await self.run_step_storage.allocate_sequence(self.session_id)

    def subscribe(self) -> AsyncIterator[AgentStreamItem]:
        queue: asyncio.Queue[AgentStreamItem | object] = asyncio.Queue()
        if self._closed:
            queue.put_nowait(self._SENTINEL)
        self._subscribers.add(queue)

        async def _iterator() -> AsyncIterator[AgentStreamItem]:
            try:
                while True:
                    item = await queue.get()
                    if item is self._SENTINEL:
                        break
                    yield item
            finally:
                self._subscribers.discard(queue)

        return _iterator()

    async def enqueue_steer(self, user_input: UserInput) -> bool:
        if self._closed:
            return False
        text = extract_text(user_input)
        if not text or not text.strip():
            return False
        await self.steering_queue.put(user_input)
        return True

    async def publish(self, item: AgentStreamItem) -> None:
        if self._closed:
            return
        for subscriber in list(self._subscribers):
            await subscriber.put(item)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.trace_runtime is not None:
            await self.trace_runtime.stop()
        for subscriber in list(self._subscribers):
            await subscriber.put(self._SENTINEL)


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
    def messages(self) -> list[dict[str, Any]]:
        return self.ledger.messages

    @messages.setter
    def messages(self, value: list[dict[str, Any]]) -> None:
        self.ledger.messages = value

    @property
    def tool_schemas(self) -> list[dict[str, Any]] | None:
        return self.ledger.tool_schemas

    @tool_schemas.setter
    def tool_schemas(self, value: list[dict[str, Any]] | None) -> None:
        self.ledger.tool_schemas = value

    @property
    def start_time(self) -> float:
        return self.ledger.start_time

    @start_time.setter
    def start_time(self, value: float) -> None:
        self.ledger.start_time = value

    @property
    def termination_reason(self) -> TerminationReason | None:
        return self.ledger.termination_reason

    @termination_reason.setter
    def termination_reason(self, value: TerminationReason | None) -> None:
        self.ledger.termination_reason = value

    @property
    def total_tokens(self) -> int:
        return self.ledger.total_tokens

    @total_tokens.setter
    def total_tokens(self, value: int) -> None:
        self.ledger.total_tokens = value

    @property
    def input_tokens(self) -> int:
        return self.ledger.input_tokens

    @input_tokens.setter
    def input_tokens(self, value: int) -> None:
        self.ledger.input_tokens = value

    @property
    def output_tokens(self) -> int:
        return self.ledger.output_tokens

    @output_tokens.setter
    def output_tokens(self, value: int) -> None:
        self.ledger.output_tokens = value

    @property
    def cache_read_tokens(self) -> int:
        return self.ledger.cache_read_tokens

    @cache_read_tokens.setter
    def cache_read_tokens(self, value: int) -> None:
        self.ledger.cache_read_tokens = value

    @property
    def cache_creation_tokens(self) -> int:
        return self.ledger.cache_creation_tokens

    @cache_creation_tokens.setter
    def cache_creation_tokens(self, value: int) -> None:
        self.ledger.cache_creation_tokens = value

    @property
    def token_cost(self) -> float:
        return self.ledger.token_cost

    @token_cost.setter
    def token_cost(self, value: float) -> None:
        self.ledger.token_cost = value

    @property
    def steps_count(self) -> int:
        return self.ledger.steps_count

    @steps_count.setter
    def steps_count(self, value: int) -> None:
        self.ledger.steps_count = value

    @property
    def tool_calls_count(self) -> int:
        return self.ledger.tool_calls_count

    @tool_calls_count.setter
    def tool_calls_count(self, value: int) -> None:
        self.ledger.tool_calls_count = value

    @property
    def assistant_steps_count(self) -> int:
        return self.ledger.assistant_steps_count

    @assistant_steps_count.setter
    def assistant_steps_count(self, value: int) -> None:
        self.ledger.assistant_steps_count = value

    @property
    def response_content(self) -> str | None:
        return self.ledger.response_content

    @response_content.setter
    def response_content(self, value: str | None) -> None:
        self.ledger.response_content = value

    @property
    def last_compact_metadata(self) -> CompactMetadata | None:
        return self.ledger.last_compact_metadata

    @last_compact_metadata.setter
    def last_compact_metadata(self, value: CompactMetadata | None) -> None:
        self.ledger.last_compact_metadata = value

    @property
    def elapsed(self) -> float:
        return time.time() - self.ledger.start_time

    @property
    def is_terminal(self) -> bool:
        return self.ledger.termination_reason is not None
