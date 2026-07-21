"""P3-04: Scheduler recoverable pause/resume (not cancel)."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, RunTreeRole, RunExecutionRequest, RunStatus
from agiwo.agent.budget_gate import PermissiveLlmBudgetGate
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import UserMessage
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler import Scheduler, SchedulerExecutionRequest


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
    agent.llm_budget_gate = PermissiveLlmBudgetGate()
    sched = Scheduler()
    await sched.start()
    try:
        req = SchedulerExecutionRequest(
            state_id="root-pause",
            session_id="sess-pause",
            user_input=UserMessage.from_system("work"),
            execution=RunExecutionRequest(
                run_id="run_sched_pause",
                run_tree_role=RunTreeRole.NONE,
            ),
            persistent=True,
        )
        await sched.dispatch_execution(agent, req)
        await asyncio.wait_for(model.started.wait(), timeout=5)
        pause_task = asyncio.create_task(
            sched.request_recoverable_pause(["run_sched_pause"], "user_pause")
        )
        await asyncio.sleep(0.05)
        model.release.set()
        await asyncio.wait_for(pause_task, timeout=10)
        status = await sched.get_run_status("run_sched_pause")
        assert status is RunStatus.PAUSED

        await sched.prepare_resume(["run_sched_pause"])
        # Second turn after resume: reset gate for next LLM call.
        model.release = asyncio.Event()
        model.started = asyncio.Event()

        async def _release_soon() -> None:
            await asyncio.wait_for(model.started.wait(), timeout=10)
            model.release.set()

        releaser = asyncio.create_task(_release_soon())
        await sched.release_resume_barrier()
        await asyncio.wait_for(releaser, timeout=15)
        await asyncio.wait_for(sched.wait_for("root-pause", timeout=15), timeout=20)
        final = await sched.get_run_status("run_sched_pause")
        assert final in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.INTERRUPTED}
    finally:
        await sched.stop()
        await agent.close()
