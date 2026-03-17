"""Tree-wide scheduler operations such as cancel and shutdown."""

from agiwo.scheduler.coordinator import SchedulerCoordinator
from agiwo.scheduler.models import AgentState, AgentStateStatus
from agiwo.scheduler.state_ops import SchedulerStateOps
from agiwo.scheduler.store import AgentStateStorage
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

_SHUTDOWN_SUMMARY_TASK = (
    "System shutdown requested. Please produce a final summary "
    "report of all work done so far."
)


class SchedulerTreeOps:
    def __init__(
        self,
        *,
        store: AgentStateStorage,
        coordinator: SchedulerCoordinator,
        state_ops: SchedulerStateOps,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._state_ops = state_ops

    async def cancel_subtree(self, state_id: str, reason: str) -> None:
        handle = self._coordinator.get_execution_handle(state_id)
        if handle is not None:
            handle.cancel(reason)

        signal = self._coordinator.get_abort_signal(state_id)
        if signal is not None and not signal.is_aborted():
            signal.abort(reason)

        for child in await self._active_children(state_id):
            await self.cancel_subtree(child.id, reason)

        state = await self._store.get_state(state_id)
        if state is not None:
            await self._state_ops.mark_failed(state, reason)

    async def shutdown_subtree(self, state_id: str) -> None:
        for child in await self._active_children(state_id):
            await self.shutdown_subtree(child.id)

        state = await self._store.get_state(state_id)
        if state is None:
            return
        if state.is_root and state.status in (
            AgentStateStatus.WAITING,
            AgentStateStatus.IDLE,
        ):
            await self._state_ops.mark_queued(
                state,
                pending_input=_SHUTDOWN_SUMMARY_TASK,
            )
        elif state.status == AgentStateStatus.WAITING:
            await self._state_ops.mark_failed(state, "Shutdown before completion")
        elif state.status == AgentStateStatus.PENDING:
            await self._state_ops.mark_failed(state, "Shutdown before execution")

    async def _active_children(self, state_id: str) -> list[AgentState]:
        children = await self._store.list_states(parent_id=state_id, limit=1000)
        return [child for child in children if child.is_active()]


__all__ = ["SchedulerTreeOps"]
