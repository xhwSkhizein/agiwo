import asyncio
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from agiwo.agent.lifecycle.session import AgentSessionRuntime


@dataclass(slots=True)
class AgentRunContext:
    session_runtime: AgentSessionRuntime
    run_id: str
    agent_id: str
    agent_name: str
    user_id: str | None = None
    depth: int = 0
    parent_run_id: str | None = None
    timeout_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create_root(
        cls,
        *,
        session_runtime: AgentSessionRuntime,
        agent_id: str,
        agent_name: str,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AgentRunContext":
        return cls(
            session_runtime=session_runtime,
            run_id=str(uuid4()),
            agent_id=agent_id,
            agent_name=agent_name,
            user_id=user_id,
            metadata=dict(metadata or {}),
        )

    @property
    def session_id(self) -> str:
        return self.session_runtime.session_id

    @property
    def steering_queue(self) -> asyncio.Queue[object]:
        return self.session_runtime.steering_queue

    @property
    def trace_id(self) -> str | None:
        return self.session_runtime.trace_id

    async def next_sequence(self) -> int:
        return await self.session_runtime.allocate_sequence()

    def new_child(
        self,
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        run_id: str | None = None,
    ) -> "AgentRunContext":
        return AgentRunContext(
            session_runtime=self.session_runtime,
            run_id=run_id or str(uuid4()),
            agent_id=agent_id or self.agent_id,
            agent_name=agent_name or self.agent_name,
            user_id=self.user_id,
            depth=self.depth + 1,
            parent_run_id=self.run_id,
            timeout_at=self.timeout_at,
            metadata=dict(self.metadata),
        )


__all__ = ["AgentRunContext"]
