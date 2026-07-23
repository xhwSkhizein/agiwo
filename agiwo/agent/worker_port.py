"""Worker delegation port — agent-side contract for Scheduler-backed Workers."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agiwo.agent.agent import Agent


class WorkerStateStatus(str, Enum):
    """Lifecycle status for a delegated Worker."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    IDLE = "idle"
    QUEUED = "queued"
    COMPLETED = "completed"
    FAILED = "failed"


ACTIVE_WORKER_STATUSES = frozenset(
    {
        WorkerStateStatus.PENDING,
        WorkerStateStatus.RUNNING,
        WorkerStateStatus.WAITING,
        WorkerStateStatus.IDLE,
        WorkerStateStatus.QUEUED,
    }
)


@dataclass(frozen=True, slots=True)
class WorkerSpawnRequest:
    parent_agent_id: str
    session_id: str
    task: str
    instruction: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerStateView:
    worker_id: str
    status: WorkerStateStatus
    result_summary: str | None = None


@runtime_checkable
class WorkerSchedulerPort(Protocol):
    """Narrow Scheduler surface for MainAgent Worker spawn/wait/cancel."""

    async def register_parent(
        self,
        *,
        state_id: str,
        session_id: str,
        agent: "Agent",
    ) -> None: ...

    async def start(self) -> None: ...

    async def spawn_worker(self, request: WorkerSpawnRequest) -> WorkerStateView: ...

    def nudge(self) -> None: ...

    async def list_children(
        self,
        *,
        parent_id: str,
        session_id: str,
        limit: int,
    ) -> list[WorkerStateView]: ...

    async def cancel_worker(self, worker_id: str, reason: str) -> None: ...

    async def wait_for_worker(self, worker_id: str) -> None: ...

    async def get_worker_state(self, worker_id: str) -> WorkerStateView | None: ...

    async def get_worker_report(self, worker_id: str) -> str: ...

    async def sync_parent_idle(self, parent_id: str) -> None: ...


__all__ = [
    "ACTIVE_WORKER_STATUSES",
    "WorkerSchedulerPort",
    "WorkerSpawnRequest",
    "WorkerStateStatus",
    "WorkerStateView",
]
