"""Agent state storage abstract interface."""

from abc import ABC, abstractmethod
from datetime import datetime

from agiwo.scheduler.models import (
    AgentState,
    AgentStateStatus,
    PendingEvent,
    WakeCondition,
)


class AgentStateStorage(ABC):
    """Abstract base for agent state persistence."""

    @abstractmethod
    async def save_state(self, state: AgentState) -> None:
        ...

    @abstractmethod
    async def get_state(self, state_id: str) -> AgentState | None:
        ...

    @abstractmethod
    async def update_status(
        self,
        state_id: str,
        status: AgentStateStatus,
        *,
        wake_condition: WakeCondition | None = ...,
        result_summary: str | None = ...,
        explain: str | None = ...,
        last_activity_at: datetime | None = ...,
        recent_steps: list[dict] | None = ...,
    ) -> None:
        ...

    @abstractmethod
    async def get_states_by_parent(self, parent_id: str) -> list[AgentState]:
        ...

    @abstractmethod
    async def find_pending(self) -> list[AgentState]:
        ...

    @abstractmethod
    async def find_wakeable(self, now: datetime) -> list[AgentState]:
        ...

    @abstractmethod
    async def find_unpropagated_completed(self) -> list[AgentState]:
        ...

    @abstractmethod
    async def mark_child_completed(self, parent_id: str, child_id: str) -> None:
        ...

    @abstractmethod
    async def mark_propagated(self, state_id: str) -> None:
        ...

    @abstractmethod
    async def find_timed_out(self, now: datetime) -> list[AgentState]:
        ...

    @abstractmethod
    async def increment_wake_count(self, state_id: str) -> None:
        ...

    @abstractmethod
    async def list_all(
        self,
        *,
        status: AgentStateStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentState]:
        ...

    @abstractmethod
    async def find_running(self) -> list[AgentState]:
        ...

    @abstractmethod
    async def save_event(self, event: PendingEvent) -> None:
        ...

    @abstractmethod
    async def get_pending_events(
        self, target_agent_id: str, session_id: str
    ) -> list[PendingEvent]:
        ...

    @abstractmethod
    async def delete_events(self, event_ids: list[str]) -> None:
        ...

    @abstractmethod
    async def find_agents_with_debounced_events(
        self,
        min_count: int,
        max_wait_seconds: float,
        now: datetime,
    ) -> list[tuple[str, str]]:
        ...

    @abstractmethod
    async def append_recent_step(self, state_id: str, step: dict) -> None:
        ...

    @abstractmethod
    async def delete_events_by_agent(self, target_agent_id: str) -> None:
        ...

    @abstractmethod
    async def has_recent_health_warning(
        self,
        target_agent_id: str,
        source_agent_id: str,
        within_seconds: float,
        now: datetime,
    ) -> bool:
        ...

    async def close(self) -> None:
        pass
