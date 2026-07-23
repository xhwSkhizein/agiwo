"""Contract tests for Scheduler's public Worker delegation surface (Task 13)."""

import pytest

from agiwo.agent import Agent, AgentConfig, AgentOptions
from agiwo.agent.worker_port import WorkerSchedulerPort
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler.commands import SpawnChildRequest
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentStateStatus, SchedulerConfig
from agiwo.scheduler.worker_bridge import SchedulerWorkerPort, scheduler_worker_port


class _StubModel(Model):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", temperature=0.0)

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content="ok")
        yield StreamChunk(finish_reason="stop")


def _parent_agent(*, agent_id: str = "parent-worker") -> Agent:
    return Agent(
        AgentConfig(
            name="parent",
            description="worker parent",
            system_prompt="Test",
            options=AgentOptions(max_steps_per_run=3),
        ),
        model=_StubModel(),
        id=agent_id,
    )


@pytest.mark.asyncio
async def test_spawn_worker_returns_depth_one_child() -> None:
    async with Scheduler(SchedulerConfig(check_interval=0.05)) as scheduler:
        parent = _parent_agent()
        await scheduler.register_worker_parent(
            state_id=parent.id,
            session_id="session-worker-api",
            agent=parent,
        )
        child = await scheduler.spawn_worker(
            SpawnChildRequest(
                parent_agent_id=parent.id,
                session_id="session-worker-api",
                task="do work",
            )
        )
        assert child.parent_id == parent.id
        assert child.depth == 1
        assert child.task == "do work"
        assert child.status is AgentStateStatus.PENDING


@pytest.mark.asyncio
async def test_mark_parent_idle_is_idempotent() -> None:
    async with Scheduler(SchedulerConfig(check_interval=0.05)) as scheduler:
        parent = _parent_agent(agent_id="parent-idle")
        await scheduler.register_worker_parent(
            state_id=parent.id,
            session_id="session-idle",
            agent=parent,
        )
        # Already IDLE after registration.
        await scheduler.mark_parent_idle(parent.id)
        state = await scheduler.get_state(parent.id)
        assert state is not None
        assert state.status is AgentStateStatus.IDLE

        # Missing parent is a no-op.
        await scheduler.mark_parent_idle("missing-parent")

        # Non-IDLE → IDLE.
        await scheduler._save_state(state.with_updates(status=AgentStateStatus.RUNNING))
        await scheduler.mark_parent_idle(parent.id)
        state = await scheduler.get_state(parent.id)
        assert state is not None
        assert state.status is AgentStateStatus.IDLE


@pytest.mark.asyncio
async def test_scheduler_worker_port_satisfies_protocol() -> None:
    async with Scheduler(SchedulerConfig(check_interval=0.05)) as scheduler:
        port = scheduler_worker_port(scheduler)
        assert isinstance(port, WorkerSchedulerPort)
        assert isinstance(SchedulerWorkerPort(scheduler), WorkerSchedulerPort)
