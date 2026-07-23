"""Shared helpers for Worker protocol tests."""

import asyncio
import tempfile
import uuid
from pathlib import Path

from agiwo.agent import AgentConfig, AgentOptions, AgentSpec, MainAgent
from agiwo.agent.hooks import HookPhase, HookRegistry, transform
from agiwo.agent.models.config import AgentStorageOptions, RunLogStorageConfig
from agiwo.llm.base import Model, StreamChunk
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import AgentStateStatus
from agiwo.scheduler.worker_bridge import scheduler_worker_port

_SHARED_DB_DIR = Path(tempfile.gettempdir()) / "agiwo_worker_tests"
_SHARED_DB_DIR.mkdir(parents=True, exist_ok=True)


def _next_shared_db_path() -> str:
    return str(_SHARED_DB_DIR / f"worker-test-{uuid.uuid4().hex}.db")


def worker_response(text: str = "worker-report-text") -> list[StreamChunk]:
    return [StreamChunk(content=text), StreamChunk(finish_reason="stop")]


class QuickModel(Model):
    """Returns one completion per call."""

    def __init__(
        self,
        responses: list[list[StreamChunk]] | None = None,
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="quick-model", name="quick-model", temperature=0.0)
        self._responses = responses or [
            [StreamChunk(content="ok"), StreamChunk(finish_reason="stop")]
        ]
        self._index = 0
        self._start_event = start_event

    async def arun_stream(self, messages, tools=None):
        del messages, tools
        if self._start_event is not None:
            await self._start_event.wait()
        chunks = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        for chunk in chunks:
            yield chunk


def build_main_agent_with_scheduler(
    scheduler: Scheduler,
    *,
    session_id: str = "session-worker",
    agent_id: str = "main-agent",
    model: Model | None = None,
) -> MainAgent:
    db_path = _next_shared_db_path()

    async def _noop_memory(payload: dict[str, object]) -> dict[str, object]:
        payload = dict(payload)
        payload["memories"] = []
        return payload

    hooks = HookRegistry(
        [
            transform(
                HookPhase.ASSEMBLE_CONTEXT,
                "noop_memory_retrieve",
                _noop_memory,
            )
        ]
    )
    spec = AgentSpec(
        config=AgentConfig(
            name="main-agent",
            description="worker test agent",
            system_prompt="Test prompt",
            options=AgentOptions(
                max_steps_per_run=8,
                storage=AgentStorageOptions(
                    run_log_storage=RunLogStorageConfig(
                        storage_type="sqlite",
                        config={"db_path": db_path},
                    ),
                ),
            ),
        )
    )
    return MainAgent(
        session_id=session_id,
        agent_id=agent_id,
        spec=spec,
        model=model or QuickModel(),
        hooks=hooks,
        worker_scheduler=scheduler_worker_port(scheduler),
    )


async def register_parent_state(
    scheduler: Scheduler,
    main: MainAgent,
    *,
    status: str = "idle",
) -> None:
    await scheduler.register_worker_parent(
        state_id=main.agent_id,
        session_id=main.session_id,
        agent=main.agent,
    )
    state = await scheduler.get_state(main.agent_id)
    assert state is not None
    mapped = AgentStateStatus.RUNNING if status == "running" else AgentStateStatus.IDLE
    if state.status != mapped:
        await scheduler._save_state(state.with_updates(status=mapped))
