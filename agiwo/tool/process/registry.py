from typing import Protocol, runtime_checkable


@runtime_checkable
class AgentProcessRegistry(Protocol):
    """Capability for inspecting agent-owned background processes."""

    async def list_agent_processes(
        self,
        agent_id: str,
        *,
        state: str = "running",
    ) -> list[dict[str, object]]:
        """Return structured process summaries owned by one agent."""
