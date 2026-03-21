from typing import Any

from agiwo.agent.engine.context import AgentRunContext
from agiwo.agent.lifecycle.session import AgentSessionRuntime
from agiwo.agent.storage.base import InMemoryRunStepStorage, RunStepStorage
from agiwo.agent.storage.session import InMemorySessionStorage, SessionStorage


def build_agent_context(
    *,
    session_id: str = "test-session",
    run_id: str = "test-run",
    agent_id: str = "test-agent",
    agent_name: str = "test-agent",
    run_step_storage: RunStepStorage | None = None,
    session_storage: SessionStorage | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentRunContext:
    session_runtime = AgentSessionRuntime(
        session_id=session_id,
        run_step_storage=run_step_storage or InMemoryRunStepStorage(),
        session_storage=session_storage or InMemorySessionStorage(),
    )
    return AgentRunContext(
        session_runtime=session_runtime,
        run_id=run_id,
        agent_id=agent_id,
        agent_name=agent_name,
        user_id=user_id,
        metadata=dict(metadata or {}),
    )


__all__ = ["AgentRunContext", "build_agent_context"]
