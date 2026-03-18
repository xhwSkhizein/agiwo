"""Agent state storage abstract interface."""

from abc import ABC, abstractmethod
from collections.abc import Collection

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
)


class AgentStateStorage(ABC):
    """Abstract base for generic scheduler state/event persistence."""

    @abstractmethod
    async def save_state(self, state: AgentState) -> None: ...

    @abstractmethod
    async def get_state(self, state_id: str) -> AgentState | None: ...

    @abstractmethod
    async def list_states(
        self,
        *,
        statuses: Collection[AgentStateStatus] | None = None,
        parent_id: str | None = None,
        session_id: str | None = None,
        signal_propagated: bool | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]: ...

    @abstractmethod
    async def save_event(self, event: PendingEvent) -> None: ...

    @abstractmethod
    async def list_events(
        self,
        *,
        target_agent_id: str | None = None,
        session_id: str | None = None,
    ) -> list[PendingEvent]: ...

    @abstractmethod
    async def delete_events(self, event_ids: list[str]) -> None: ...

    async def close(self) -> None:
        pass
