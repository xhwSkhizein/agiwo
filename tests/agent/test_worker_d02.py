"""Wave D-02: sync/async Worker report and same-Run resume tests."""

import asyncio

import pytest

from agiwo.agent import MainAgentState, MessageRole
from agiwo.agent.models.log import RunFinished, RunStarted
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import SchedulerConfig
from tests.agent.worker_test_helpers import (
    QuickModel,
    build_main_agent_with_scheduler,
    register_parent_state,
    worker_response,
)


@pytest.mark.asyncio
async def test_sync_worker_returns_report_without_queue_item() -> None:
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    model = QuickModel([worker_response("sync-report")])
    main = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    handle, report = await main._worker_service.spawn_worker(  # type: ignore[union-attr]
        task="sync task",
        main_run_id="run-sync",
        sync=True,
    )

    assert report == "sync-report"
    assert main.peek_pending() is None
    assert handle.worker_id

    await scheduler.stop()


@pytest.mark.asyncio
async def test_async_worker_report_enqueued_while_main_running() -> None:
    start_event = asyncio.Event()
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    model = QuickModel(
        [
            worker_response("main-ok"),
            worker_response("late-report"),
        ],
        start_event=start_event,
    )
    main = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    await main.accept("hold main")
    main_run_id = main._handle.run_id if main._handle else ""
    assert main_run_id

    start_event.set()
    await asyncio.sleep(0.05)

    await main._worker_service.spawn_worker(  # type: ignore[union-attr]
        task="async task",
        main_run_id=main_run_id,
        sync=False,
    )
    await asyncio.sleep(0.4)

    if main.state is MainAgentState.RUNNING:
        await main.wait_current_run()

    pending = main.peek_pending()
    steps = await main.run_log_storage.list_step_views(
        session_id=main.session_id,
        agent_id=main.agent_id,
        run_id=main_run_id,
    )
    texts = [step.get_display_text() or "" for step in steps]
    assert pending is None or pending.kind.value == "worker_report"
    assert any("late-report" in text for text in texts) or any(
        "late-report" in (pending.text or "") for _ in [pending] if pending
    )

    await scheduler.stop()


@pytest.mark.asyncio
async def test_async_after_idle_resumes_same_run_id() -> None:
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    model = QuickModel(
        [
            worker_response("main-ok"),
            worker_response("after-idle-report"),
            worker_response("main-continues"),
        ]
    )
    main = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    first_handle = await main.accept("start")
    assert first_handle is not None
    first_run_id = first_handle.run_id
    await main.wait_current_run()
    if main._completion_task is not None:
        await main._completion_task
    assert main.state is MainAgentState.IDLE

    await main._worker_service.spawn_worker(  # type: ignore[union-attr]
        task="late worker",
        main_run_id=first_run_id,
        sync=False,
    )
    await asyncio.sleep(0.5)
    if main.state is MainAgentState.RUNNING:
        await main.wait_current_run()

    entries = await main.run_log_storage.list_entries(
        session_id=main.session_id,
        run_id=first_run_id,
        agent_id=main.agent_id,
    )
    started = [entry for entry in entries if isinstance(entry, RunStarted)]
    finished = [entry for entry in entries if isinstance(entry, RunFinished)]
    assert started
    assert all(entry.run_id == first_run_id for entry in started + finished)

    steps = await main.run_log_storage.list_step_views(
        session_id=main.session_id,
        agent_id=main.agent_id,
        run_id=first_run_id,
    )
    texts = [
        step.get_display_text() or "" for step in steps if step.role is MessageRole.USER
    ]
    assert any("after-idle-report" in text for text in texts)

    await scheduler.stop()
