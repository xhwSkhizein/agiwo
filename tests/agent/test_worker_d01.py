"""Wave D-01: Worker identity and L1-a isolation tests."""

import pytest

from agiwo.agent.worker_tools import SpawnWorkerTool
from agiwo.scheduler.commands import SpawnChildRequest
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentStateStatus, SchedulerConfig
from tests.agent.worker_test_helpers import (
    QuickModel,
    build_main_agent_with_scheduler,
    register_parent_state,
    worker_response,
)
from tests.utils.agent_context import build_tool_context


@pytest.mark.asyncio
async def test_worker_identity_isolated_from_main_projection() -> None:
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    model = QuickModel([worker_response("worker-report-text")])
    main = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    handle, report = await main._worker_service.spawn_worker(  # type: ignore[union-attr]
        task="isolated task",
        main_run_id="run-main-1",
        sync=True,
    )

    assert report == "worker-report-text"
    assert handle.worker_id != main.agent_id

    worker_state = await scheduler.get_state(handle.worker_id)
    assert worker_state is not None
    assert worker_state.parent_id == main.agent_id
    assert worker_state.depth == 1
    assert worker_state.status in {
        AgentStateStatus.COMPLETED,
        AgentStateStatus.IDLE,
    }

    main_steps = await main.run_log_storage.list_step_views(
        session_id=main.session_id,
        agent_id=main.agent_id,
    )
    assert all(step.agent_id == main.agent_id for step in main_steps)
    assert handle.worker_id not in {step.agent_id for step in main_steps}

    await scheduler.stop()


@pytest.mark.asyncio
async def test_spawn_worker_denies_depth_one_workers() -> None:
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    main = build_main_agent_with_scheduler(scheduler)
    service = main._worker_service
    assert service is not None
    tool = SpawnWorkerTool(service)

    child_context = build_tool_context(
        session_id=main.session_id,
        run_id="run-worker",
        agent_id="worker-child",
        depth=1,
    )
    gate = await tool.gate({"task": "nested"}, child_context)
    assert gate.action == "deny"
    assert "Workers cannot spawn Workers" in gate.reason

    await scheduler.stop()


@pytest.mark.asyncio
async def test_spawn_worker_requires_registered_parent_state() -> None:
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    main = build_main_agent_with_scheduler(scheduler)
    service = main._worker_service
    assert service is not None

    with pytest.raises(ValueError, match="Parent agent state"):
        await scheduler._tool_control.spawn_child(
            SpawnChildRequest(
                parent_agent_id=main.agent_id,
                session_id=main.session_id,
                task="orphan",
            )
        )

    await scheduler.stop()
