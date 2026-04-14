"""Scheduler tree read models for console operator workflows."""

from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone

from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentState, AgentStateStatus, SchedulerRunResult

from server.models.session import ChannelChatSessionStore


class SchedulerTreeError(RuntimeError):
    """Base error for scheduler tree assembly failures."""


class SchedulerTreeNotFoundError(SchedulerTreeError):
    """Raised when the requested root state does not exist."""


class SchedulerTreeValidationError(SchedulerTreeError):
    """Raised when the requested state cannot produce a valid tree."""


class SchedulerTreeTooLargeError(SchedulerTreeValidationError):
    """Raised when the requested tree exceeds the configured node limit."""


@dataclass(slots=True)
class SchedulerTreeStatsRecord:
    total: int = 0
    running: int = 0
    waiting: int = 0
    queued: int = 0
    idle: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0


@dataclass(slots=True)
class SchedulerTreeNodeRecord:
    state_id: str
    root_state_id: str
    parent_state_id: str | None
    child_ids: list[str]
    session_id: str | None
    agent_id: str
    task_id: str | None
    status: str
    depth: int
    created_at: datetime | None
    updated_at: datetime | None
    completed_at: datetime | None
    wake_condition: object | None
    pending_event_count: int
    last_error: str | None
    result_summary: str | None
    last_run_result: SchedulerRunResult | None


@dataclass(slots=True)
class SchedulerTreeRecord:
    root_state_id: str
    root_session_id: str | None
    nodes: list[SchedulerTreeNodeRecord]
    stats: SchedulerTreeStatsRecord
    generated_at: datetime


def _state_sort_key(state: AgentState) -> tuple[datetime, str]:
    return (
        state.created_at
        or state.updated_at
        or datetime.min.replace(tzinfo=timezone.utc),
        state.id,
    )


def _is_cancelled_state(state: AgentState) -> bool:
    if state.status != AgentStateStatus.FAILED:
        return False
    if (
        state.last_run_result is not None
        and state.last_run_result.termination_reason.value == "cancelled"
    ):
        return True
    summary = (
        state.last_run_result.summary
        if state.last_run_result is not None and state.last_run_result.summary
        else state.result_summary or ""
    ).lower()
    return "cancelled" in summary or "canceled" in summary


def _last_error(state: AgentState) -> str | None:
    if state.explain:
        return state.explain
    if state.last_run_result is not None and state.last_run_result.error:
        return state.last_run_result.error
    if state.status == AgentStateStatus.FAILED:
        if state.result_summary:
            return state.result_summary
        if state.last_run_result is not None:
            return state.last_run_result.summary
    return None


def _completed_at(state: AgentState) -> datetime | None:
    if (
        state.last_run_result is not None
        and state.last_run_result.completed_at is not None
    ):
        return state.last_run_result.completed_at
    if state.status in (AgentStateStatus.COMPLETED, AgentStateStatus.FAILED):
        return state.updated_at
    return None


class SchedulerTreeViewService:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        session_store: ChannelChatSessionStore | None = None,
        max_nodes: int = 500,
    ) -> None:
        self._scheduler = scheduler
        self._session_store = session_store
        self._max_nodes = max_nodes

    async def resolve_root_state_id(self, state_id: str) -> str | None:
        state_cache: dict[str, AgentState | None] = {}
        root_cache: dict[str, str | None] = {}
        return await self._resolve_root_state_id(
            state_id,
            state_cache=state_cache,
            root_cache=root_cache,
        )

    async def resolve_root_state_ids(
        self,
        state_ids: list[str],
    ) -> dict[str, str | None]:
        state_cache: dict[str, AgentState | None] = {}
        root_cache: dict[str, str | None] = {}
        resolved: dict[str, str | None] = {}
        for state_id in state_ids:
            resolved[state_id] = await self._resolve_root_state_id(
                state_id,
                state_cache=state_cache,
                root_cache=root_cache,
            )
        return resolved

    async def get_tree(self, root_state_id: str) -> SchedulerTreeRecord:
        root = await self._scheduler.get_state(root_state_id)
        if root is None:
            raise SchedulerTreeNotFoundError(f"Agent state '{root_state_id}' not found")
        if root.parent_id is not None:
            raise SchedulerTreeValidationError(
                f"Scheduler tree requires a root state id; '{root_state_id}' is a child state"
            )

        collected: list[AgentState] = []
        children_by_parent: dict[str, list[AgentState]] = {}
        queue: deque[AgentState] = deque([root])
        while queue:
            state = queue.popleft()
            collected.append(state)
            if len(collected) > self._max_nodes:
                raise SchedulerTreeTooLargeError(
                    f"Scheduler tree too large for root '{root_state_id}' "
                    f"(limit {self._max_nodes} nodes)"
                )
            children = await self._scheduler.list_states(
                parent_id=state.id,
                limit=self._max_nodes + 1,
            )
            children.sort(key=_state_sort_key)
            children_by_parent[state.id] = children
            queue.extend(children)

        pending_events = await self._scheduler.list_events(session_id=root.session_id)
        pending_counts = Counter(event.target_agent_id for event in pending_events)
        task_ids = await self._load_task_ids(collected)

        nodes = [
            SchedulerTreeNodeRecord(
                state_id=state.id,
                root_state_id=root.id,
                parent_state_id=state.parent_id,
                child_ids=[child.id for child in children_by_parent.get(state.id, [])],
                session_id=state.session_id,
                agent_id=state.id,
                task_id=task_ids.get(state.session_id),
                status=state.status.value,
                depth=state.depth,
                created_at=state.created_at,
                updated_at=state.updated_at,
                completed_at=_completed_at(state),
                wake_condition=state.wake_condition,
                pending_event_count=pending_counts.get(state.id, 0),
                last_error=_last_error(state),
                result_summary=state.result_summary,
                last_run_result=state.last_run_result,
            )
            for state in collected
        ]

        return SchedulerTreeRecord(
            root_state_id=root.id,
            root_session_id=root.session_id,
            nodes=nodes,
            stats=self._build_stats(collected),
            generated_at=datetime.now(timezone.utc),
        )

    async def _resolve_root_state_id(
        self,
        state_id: str,
        *,
        state_cache: dict[str, AgentState | None],
        root_cache: dict[str, str | None],
    ) -> str | None:
        if state_id in root_cache:
            return root_cache[state_id]
        state = await self._get_state(state_id, state_cache)
        if state is None:
            root_cache[state_id] = None
            return None

        path: list[str] = []
        cursor = state
        seen: set[str] = set()
        while True:
            if cursor.id in root_cache:
                root_id = root_cache[cursor.id]
                break
            if cursor.id in seen:
                raise SchedulerTreeValidationError(
                    f"Cycle detected while resolving root state for '{state_id}'"
                )
            seen.add(cursor.id)
            path.append(cursor.id)
            if cursor.parent_id is None:
                root_id = cursor.id
                break
            parent = await self._get_state(cursor.parent_id, state_cache)
            if parent is None:
                raise SchedulerTreeValidationError(
                    f"Parent state '{cursor.parent_id}' not found while resolving "
                    f"root state for '{state_id}'"
                )
            cursor = parent

        for item_id in path:
            root_cache[item_id] = root_id
        return root_id

    async def _get_state(
        self,
        state_id: str,
        state_cache: dict[str, AgentState | None],
    ) -> AgentState | None:
        if state_id not in state_cache:
            state_cache[state_id] = await self._scheduler.get_state(state_id)
        return state_cache[state_id]

    async def _load_task_ids(
        self,
        states: list[AgentState],
    ) -> dict[str, str | None]:
        del states
        return {}

    def _build_stats(self, states: list[AgentState]) -> SchedulerTreeStatsRecord:
        stats = SchedulerTreeStatsRecord(total=len(states))
        for state in states:
            if state.status == AgentStateStatus.RUNNING:
                stats.running += 1
            elif state.status == AgentStateStatus.WAITING:
                stats.waiting += 1
            elif state.status == AgentStateStatus.QUEUED:
                stats.queued += 1
            elif state.status == AgentStateStatus.IDLE:
                stats.idle += 1
            elif state.status == AgentStateStatus.COMPLETED:
                stats.completed += 1
            elif state.status == AgentStateStatus.FAILED:
                if _is_cancelled_state(state):
                    stats.cancelled += 1
                else:
                    stats.failed += 1
        return stats


__all__ = [
    "SchedulerTreeError",
    "SchedulerTreeNodeRecord",
    "SchedulerTreeNotFoundError",
    "SchedulerTreeRecord",
    "SchedulerTreeStatsRecord",
    "SchedulerTreeTooLargeError",
    "SchedulerTreeValidationError",
    "SchedulerTreeViewService",
]
