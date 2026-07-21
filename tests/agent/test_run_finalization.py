"""Root-run plan guard and completion snapshot tests (ADR 0048)."""

import json
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    RunExecutionRequest,
)
from agiwo.agent.models.execution import RunTreeRole
from agiwo.llm.base import Model, StreamChunk


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
    max_steps_per_run: int = 50,
):
    agent = Agent(
        AgentConfig(
            name="root-finalization",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=max_steps_per_run,
            ),
        ),
        id="root-finalization",
        model=model,
    )
    handle = agent.start_prevalidated(
        "complete the work",
        session_id="root-finalization-session",
        execution_request=RunExecutionRequest(
            run_id="root-run",
            run_tree_role=RunTreeRole.ROOT,
        ),
    )
    return await handle.wait()


@pytest.mark.asyncio
async def test_simple_root_completion_snapshot() -> None:
    model = _ScriptedModel(["ordinary report"])

    result = await _run_root(model)

    assert result.response == "ordinary report"
    assert result.finalization is not None
    assert result.finalization.decision == {"reason": "run_completed"}
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_plan_milestones_still_guard_unfinished_root() -> None:
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

    assert result.finalization is not None
    assert result.finalization.decision == {"reason": "run_completed"}
    assert len(model.calls) == 4
    guard_call = model.calls[2]
    assert guard_call[-1]["origin"] == "assignment_plan_guard"


@pytest.mark.asyncio
async def test_session_none_run_also_keeps_completion_snapshot() -> None:
    model = _ScriptedModel(["ordinary report"])
    agent = Agent(
        AgentConfig(
            name="ordinary-run",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=model,
    )
    handle = agent.start_prevalidated(
        "hello",
        session_id="ordinary-session",
        execution_request=RunExecutionRequest(
            run_id="ordinary-run",
            run_tree_role=RunTreeRole.NONE,
        ),
    )
    result = await handle.wait()

    assert result.response == "ordinary report"
    assert result.finalization is not None
    assert result.finalization.decision == {"reason": "run_completed"}
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_root_max_steps_forces_fault_snapshot() -> None:
    model = _ScriptedModel([])

    result = await _run_root(model, max_steps_per_run=0)

    assert result.finalization is not None
    assert result.finalization.decision == {
        "reason": "max_steps_per_run_mechanical_handoff",
    }
