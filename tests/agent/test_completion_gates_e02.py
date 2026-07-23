"""Wave E-02: Loop integration for mechanical completion gates."""

import json
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    MainAgent,
    RunExecutionRequest,
)
from agiwo.agent.completion_gates import CompletionGates
from agiwo.agent.models.execution import RunTreeRole
from agiwo.llm.base import Model, StreamChunk
from tests.agent.worker_test_helpers import build_main_agent_with_scheduler
from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import SchedulerConfig
from tests.agent.worker_test_helpers import register_parent_state


class _ScriptedModel(Model):
    def __init__(self, responses: list[str | dict]) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = iter(responses)
        self.calls: list[list[dict]] = []

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools
        self.calls.append(messages)
        response = next(self._responses)
        if isinstance(response, dict):
            yield StreamChunk(tool_calls=[response])
        else:
            yield StreamChunk(content=response)
        yield StreamChunk(finish_reason="stop")


async def _run_root(
    model: Model,
    *,
    active_worker_ids=None,
    max_steps_per_run: int = 50,
):
    agent = Agent(
        AgentConfig(
            name="gate-integration",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=max_steps_per_run,
            ),
        ),
        id="gate-integration",
        model=model,
    )
    handle = agent.start_prevalidated(
        "complete the work",
        session_id="gate-integration-session",
        execution_request=RunExecutionRequest(
            run_id="gate-root-run",
            run_tree_role=RunTreeRole.ROOT,
        ),
        active_worker_ids=active_worker_ids,
    )
    return await handle.wait()


@pytest.mark.asyncio
async def test_semantic_gates_seam_defaults_to_mechanical_only() -> None:
    gates = CompletionGates()
    assert gates.semantic_enabled is False


@pytest.mark.asyncio
async def test_clean_state_allows_root_stop() -> None:
    model = _ScriptedModel(["ordinary report"])
    result = await _run_root(model)
    assert result.response == "ordinary report"
    assert result.finalization is not None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_open_milestones_block_then_unblock_root_stop() -> None:
    update_plan_call = {
        "index": 0,
        "id": "plan-1",
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": json.dumps(
                {
                    "changes": [
                        {
                            "id": "inspect",
                            "description": "Inspect the result",
                            "status": "pending",
                        }
                    ]
                }
            ),
        },
    }
    model = _ScriptedModel(
        [
            update_plan_call,
            "premature report",
            {
                **update_plan_call,
                "id": "plan-2",
                "function": {
                    "name": "update_plan",
                    "arguments": json.dumps(
                        {"changes": [{"id": "inspect", "status": "completed"}]}
                    ),
                },
            },
            "completed report",
        ]
    )

    result = await _run_root(model)

    assert result.response == "completed report"
    assert len(model.calls) == 4
    guard_call = model.calls[2]
    guard_text = guard_call[-1].get("content", "")
    assert isinstance(guard_text, str)
    assert "unfinished milestones" in guard_text
    assert "inspect" in guard_text


@pytest.mark.asyncio
async def test_unfinished_workers_block_then_unblock_root_stop() -> None:
    active_workers: set[str] = {"worker-1"}

    class _WorkerClearingModel(_ScriptedModel):
        async def arun_stream(self, messages, tools=None):
            # Simulate the Worker finishing once the gate feedback lands in
            # the live Run's message flow.
            if any(
                "Unfinished Workers remain" in str(message.get("content", ""))
                for message in messages
            ):
                active_workers.clear()
            async for chunk in super().arun_stream(messages, tools):
                yield chunk

    model = _WorkerClearingModel(
        [
            "premature report",
            "completed after workers clear",
        ]
    )
    result = await _run_root(
        model,
        active_worker_ids=lambda: frozenset(active_workers),
    )

    assert result.response == "completed after workers clear"
    assert len(model.calls) == 2
    guard_text = model.calls[1][-1].get("content", "")
    assert isinstance(guard_text, str)
    assert "Unfinished Workers remain" in guard_text
    assert "worker-1" in guard_text


@pytest.mark.asyncio
async def test_main_agent_routes_gate_feedback_into_live_run() -> None:
    update_plan_call = {
        "index": 0,
        "id": "plan-1",
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": json.dumps(
                {
                    "changes": [
                        {
                            "id": "inspect",
                            "description": "Inspect the result",
                            "status": "pending",
                        }
                    ]
                }
            ),
        },
    }
    model = _ScriptedModel(
        [
            update_plan_call,
            "premature report",
            {
                **update_plan_call,
                "id": "plan-2",
                "function": {
                    "name": "update_plan",
                    "arguments": json.dumps(
                        {"changes": [{"id": "inspect", "status": "completed"}]}
                    ),
                },
            },
            "completed report",
        ]
    )
    scheduler = Scheduler(config=SchedulerConfig(check_interval=0.05))
    main: MainAgent = build_main_agent_with_scheduler(scheduler, model=model)
    await register_parent_state(scheduler, main, status="idle")

    await main.accept("complete the work")
    await main.wait_current_run()

    # Gate feedback is written directly into the live Run's loop queue, so
    # the model call after the blocked completion attempt must contain it.
    assert len(model.calls) == 4
    guard_messages = model.calls[2]
    assert any(
        "unfinished milestones" in str(message.get("content", ""))
        for message in guard_messages
    )

    await scheduler.stop()
