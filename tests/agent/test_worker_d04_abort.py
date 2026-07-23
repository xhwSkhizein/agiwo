"""Wave D-04: sync SpawnWorkerTool respects abort_signal."""

import asyncio
from dataclasses import dataclass, field

import pytest

from agiwo.agent import AgentConfig, AgentOptions, AgentSpec, MainAgent
from agiwo.agent.hooks import HookRegistry
from agiwo.agent.worker import WorkerService
from agiwo.agent.worker_port import (
    WorkerSpawnRequest,
    WorkerStateStatus,
    WorkerStateView,
)
from agiwo.agent.worker_tools import SpawnWorkerTool
from agiwo.llm.base import Model, StreamChunk
from agiwo.utils.abort_signal import AbortSignal
from tests.utils.agent_context import build_tool_context


class _StubModel(Model):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", temperature=0.0)

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        yield StreamChunk(content="ok")
        yield StreamChunk(finish_reason="stop")


@dataclass
class _HangingSchedulerPort:
    """Worker port whose get_worker_report hangs until cancelled."""

    cancel_calls: list[tuple[str, str]] = field(default_factory=list)
    _report_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def register_parent(self, *, state_id, session_id, agent) -> None:  # noqa: ANN001
        del state_id, session_id, agent

    async def start(self) -> None:
        return None

    async def spawn_worker(self, request: WorkerSpawnRequest) -> WorkerStateView:
        return WorkerStateView(
            worker_id=f"worker-for-{request.parent_agent_id}",
            status=WorkerStateStatus.RUNNING,
        )

    def nudge(self) -> None:
        return None

    async def list_children(self, *, parent_id, session_id, limit):  # noqa: ANN001
        del parent_id, session_id, limit
        return []

    async def cancel_worker(self, worker_id: str, reason: str) -> None:
        self.cancel_calls.append((worker_id, reason))
        self._report_event.set()

    async def wait_for_worker(self, worker_id: str) -> None:
        del worker_id
        await self._report_event.wait()

    async def get_worker_state(self, worker_id: str) -> WorkerStateView | None:
        return WorkerStateView(worker_id=worker_id, status=WorkerStateStatus.RUNNING)

    async def get_worker_report(self, worker_id: str) -> str:
        del worker_id
        await self._report_event.wait()
        return "should-not-return"

    async def sync_parent_idle(self, parent_id: str) -> None:
        del parent_id


def _build_main(port: _HangingSchedulerPort) -> MainAgent:
    return MainAgent(
        session_id="session-abort",
        agent_id="main-abort",
        spec=AgentSpec(
            config=AgentConfig(
                name="main",
                description="abort test",
                system_prompt="Test",
                options=AgentOptions(max_steps_per_run=3),
            )
        ),
        model=_StubModel(),
        hooks=HookRegistry(),
        worker_scheduler=port,
    )


@pytest.mark.asyncio
async def test_sync_spawn_worker_tool_aborts_and_cancels_worker() -> None:
    port = _HangingSchedulerPort()
    main = _build_main(port)
    assert main._worker_service is not None
    tool = SpawnWorkerTool(main._worker_service)
    abort = AbortSignal()
    context = build_tool_context(
        agent_id=main.agent_id,
        run_id="run-abort",
        depth=0,
    )

    async def _abort_soon() -> None:
        await asyncio.sleep(0.05)
        abort.abort("test abort")

    abort_task = asyncio.create_task(_abort_soon())
    result = await asyncio.wait_for(
        tool.execute(
            {"task": "hang forever", "sync": True},
            context,
            abort_signal=abort,
        ),
        timeout=2.0,
    )
    await abort_task

    assert result.is_success is False
    assert result.error == "Aborted"
    assert port.cancel_calls
    assert port.cancel_calls[0][0].startswith("worker-for-")
    assert "abort" in port.cancel_calls[0][1].lower()


@pytest.mark.asyncio
async def test_sync_spawn_without_abort_still_returns_report() -> None:
    @dataclass
    class _ImmediatePort(_HangingSchedulerPort):
        async def get_worker_report(self, worker_id: str) -> str:
            del worker_id
            return "immediate-report"

    port = _ImmediatePort()
    main = _build_main(port)
    assert isinstance(main._worker_service, WorkerService)
    handle, report = await main._worker_service.spawn_worker(
        task="quick",
        main_run_id="run-ok",
        sync=True,
    )
    assert report == "immediate-report"
    assert handle.worker_id
    assert not port.cancel_calls
