"""Worker spawn/wait/report service for session MainAgent (ADR 0049 Wave D)."""

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agiwo.agent.worker_port import (
    ACTIVE_WORKER_STATUSES,
    WorkerSchedulerPort,
    WorkerSpawnRequest,
)
from agiwo.utils.logging import get_logger

if TYPE_CHECKING:
    from agiwo.agent.main_agent import MainAgent

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerHandle:
    """One-shot Scheduler-delegated worker (distinct ``agent_id``)."""

    worker_id: str
    main_run_id: str
    sync: bool


class WorkerService:
    """MainAgent-owned Worker registry backed by a WorkerSchedulerPort."""

    def __init__(self, main_agent: "MainAgent", scheduler: WorkerSchedulerPort) -> None:
        self._main = main_agent
        self._scheduler = scheduler
        self._started = False
        self._workers: dict[str, WorkerHandle] = {}
        self._monitor_tasks: dict[str, asyncio.Task[None]] = {}
        self._list_page_size = 256

    @property
    def active_worker_ids(self) -> frozenset[str]:
        return frozenset(self._workers.keys())

    async def ensure_started(self) -> None:
        if self._started:
            return
        await self._scheduler.start()
        await self._scheduler.register_parent(
            state_id=self._main.agent_id,
            session_id=self._main.session_id,
            agent=self._main.agent,
        )
        self._started = True

    async def spawn_worker(
        self,
        *,
        task: str,
        main_run_id: str,
        sync: bool,
        instruction: str | None = None,
    ) -> tuple[WorkerHandle, str | None]:
        """Spawn a depth-1 Worker.

        Returns ``(handle, report)``; ``report`` is set only for sync spawns.
        """
        await self.ensure_started()

        state = await self._scheduler.spawn_worker(
            WorkerSpawnRequest(
                parent_agent_id=self._main.agent_id,
                session_id=self._main.session_id,
                task=task,
                instruction=instruction,
            )
        )
        handle = WorkerHandle(
            worker_id=state.worker_id,
            main_run_id=main_run_id,
            sync=sync,
        )
        self._workers[state.worker_id] = handle
        self._scheduler.nudge()

        if sync:
            report = await self._scheduler.get_worker_report(state.worker_id)
            self._workers.pop(state.worker_id, None)
            return handle, report

        monitor = asyncio.create_task(
            self._monitor_async_worker(state.worker_id, main_run_id)
        )
        self._monitor_tasks[state.worker_id] = monitor
        return handle, None

    async def cancel_all_workers(self, reason: str) -> None:
        children = await self._scheduler.list_children(
            parent_id=self._main.agent_id,
            session_id=self._main.session_id,
            limit=self._list_page_size,
        )
        for child in children:
            if child.status in ACTIVE_WORKER_STATUSES:
                await self._scheduler.cancel_worker(child.worker_id, reason)
        for task in list(self._monitor_tasks.values()):
            task.cancel()
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks.values(), return_exceptions=True)
        self._monitor_tasks.clear()
        self._workers.clear()

    async def _monitor_async_worker(self, worker_id: str, main_run_id: str) -> None:
        try:
            report = await self._scheduler.get_worker_report(worker_id)
            await self._deliver_report_to_main(
                main_run_id=main_run_id,
                report=report,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - worker monitor boundary
            logger.warning(
                "worker_monitor_failed",
                worker_id=worker_id,
                main_run_id=main_run_id,
                error=str(error),
            )
        finally:
            self._workers.pop(worker_id, None)
            self._monitor_tasks.pop(worker_id, None)

    async def _deliver_report_to_main(
        self,
        *,
        main_run_id: str,
        report: str,
    ) -> None:
        await self._main.deliver_worker_report(main_run_id, report)

    async def sync_parent_idle(self) -> None:
        await self._scheduler.sync_parent_idle(self._main.agent_id)


__all__ = ["WorkerHandle", "WorkerService"]
