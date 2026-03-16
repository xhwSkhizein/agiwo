import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agiwo.agent.stream_channel import StreamChannel
from agiwo.agent.storage.base import RunStepStorage


@dataclass(slots=True)
class AgentRunContext:
    session_id: str
    run_id: str
    agent_id: str
    agent_name: str
    channel: StreamChannel
    run_step_storage: RunStepStorage
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    trace_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    steering_queue: asyncio.Queue[Any] | None = None

    @classmethod
    def create_root(
        cls,
        *,
        session_id: str,
        agent_id: str,
        agent_name: str,
        run_step_storage: RunStepStorage,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        steering_queue: asyncio.Queue[Any] | None = None,
    ) -> "AgentRunContext":
        return cls(
            session_id=session_id,
            run_id=str(uuid4()),
            agent_id=agent_id,
            agent_name=agent_name,
            channel=StreamChannel(),
            run_step_storage=run_step_storage,
            user_id=user_id,
            metadata=dict(metadata or {}),
            steering_queue=steering_queue,
        )

    async def next_sequence(self) -> int:
        return await self.run_step_storage.allocate_sequence(self.session_id)

    def new_child(
        self,
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> "AgentRunContext":
        return AgentRunContext(
            session_id=self.session_id,
            run_id=run_id or str(uuid4()),
            agent_id=agent_id or self.agent_id,
            agent_name=agent_name or self.agent_name,
            channel=self.channel,
            run_step_storage=self.run_step_storage,
            user_id=self.user_id,
            depth=self.depth + 1,
            parent_run_id=self.run_id,
            trace_id=self.trace_id,
            timeout_at=self.timeout_at,
            metadata=dict(self.metadata),
        )


__all__ = ["AgentRunContext"]
