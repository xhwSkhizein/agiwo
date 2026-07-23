"""P3-04: Scheduler recoverable pause/resume (not cancel)."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, RunStatus
from agiwo.agent.models.config import AgentConfig
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler import Scheduler


class _GateModel(Model):
    def __init__(self) -> None:
        super().__init__(id="gate", name="gate", temperature=0.0)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        self.calls += 1
        self.started.set()
        await self.release.wait()
        yield StreamChunk(content=f"turn-{self.calls}")
        yield StreamChunk(finish_reason="stop")


@pytest.mark.asyncio
async def test_request_recoverable_pause_then_resume() -> None:
    model = _GateModel()
    agent = Agent(
        AgentConfig(name="t", description="t"),
        model=model,
        id="root-pause",
    )
    sched = Scheduler()
    await sched.start()
    try:
        state_id = await sched._submit(
            agent,
            "work",
            session_id="sess-pause",
            persistent=True,
        )
        await asyncio.wait_for(model.started.wait(), timeout=5)
        handle = sched._rt.execution_handles[state_id]
        pause_task = asyncio.create_task(
            sched.request_recoverable_pause([handle.run_id], "user_pause")
        )
        await asyncio.sleep(0.05)
        model.release.set()
        await asyncio.wait_for(pause_task, timeout=10)
        status = await sched.get_run_status(handle.run_id)
        assert status is RunStatus.PAUSED

        await sched.prepare_resume([handle.run_id])
        model.release = asyncio.Event()
        model.started = asyncio.Event()

        async def _release_soon() -> None:
            await asyncio.wait_for(model.started.wait(), timeout=10)
            model.release.set()

        releaser = asyncio.create_task(_release_soon())
        await sched.release_resume_barrier()
        await asyncio.wait_for(releaser, timeout=15)
        await asyncio.wait_for(sched.wait_for("root-pause", timeout=15), timeout=20)
        final = await sched.get_run_status(handle.run_id)
        assert final in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}
    finally:
        await sched.stop()
        await agent.close()
