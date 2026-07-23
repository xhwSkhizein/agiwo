"""Wave D-03: cancel Workers and Agent-as-Tool coexistence."""

import asyncio
import inspect

import agiwo.agent.nested.agent_tool as agent_tool
import pytest

from agiwo.agent import MainAgentState
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import (
    ACTIVE_AGENT_STATUSES,
    AgentStateStatus,
    SchedulerConfig,
)
from tests.agent.worker_test_helpers import (
    QuickModel,
    build_main_agent_with_scheduler,
    register_parent_state,
    worker_response,
)


@pytest.mark.asyncio
async def test_main_cancel_cancels_in_flight_workers() -> None:
    start_event = asyncio.Event()
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    model = QuickModel([worker_response("blocked")], start_event=start_event)
    main = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    await main.accept("block worker")
    main_run_id = main._handle.run_id if main._handle else ""
    handle, _ = await main._worker_service.spawn_worker(  # type: ignore[union-attr]
        task="long worker",
        main_run_id=main_run_id,
        sync=False,
    )

    cancel_task = asyncio.create_task(main.cancel("user cancel"))
    await asyncio.sleep(0.05)
    start_event.set()
    await cancel_task
    assert main.state is MainAgentState.IDLE

    worker_state = await scheduler.get_state(handle.worker_id)
    if worker_state is not None:
        assert (
            worker_state.status not in ACTIVE_AGENT_STATUSES
            or worker_state.status
            in {
                AgentStateStatus.COMPLETED,
                AgentStateStatus.FAILED,
            }
        )

    await scheduler.stop()


def test_agent_tool_module_does_not_use_worker_name() -> None:
    source = inspect.getsource(agent_tool)
    lowered = source.lower()
    assert "spawn_worker" not in lowered
    assert "worker protocol" not in lowered
