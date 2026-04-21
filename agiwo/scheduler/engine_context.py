"""Shared scheduler runtime surface for helper modules.

Helper modules (``_tick``, ``_stream``, ``_tree_ops``) operate on the subset
of scheduler state they need via this struct.  This removes the previous
``TYPE_CHECKING`` cycle with ``engine.py`` (the helpers no longer reference
``Scheduler``) and turns the ad-hoc ``sched._store / sched._rt / ...``
cross-module access into a small, declared dependency surface.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agiwo.scheduler.guard import TaskGuard
from agiwo.scheduler.models import AgentState, SchedulerConfig
from agiwo.scheduler.runner import SchedulerRunner
from agiwo.scheduler.runtime_state import RuntimeState
from agiwo.scheduler.store.base import AgentStateStorage


@dataclass(slots=True)
class EngineContext:
    """Dependency surface for scheduler helper modules."""

    config: SchedulerConfig
    store: AgentStateStorage
    rt: RuntimeState
    guard: TaskGuard
    runner: SchedulerRunner
    save_state: Callable[[AgentState], Awaitable[None]]
    track_active_task: Callable[[asyncio.Task[object]], None]

    @property
    def state_list_page_size(self) -> int:
        return self.config.state_list_page_size


__all__ = ["EngineContext"]
