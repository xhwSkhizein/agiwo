"""Assignment-root plan guard and finalization tests."""

import json
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    RunExecutionRequest,
)
from agiwo.agent.budget_gate import PermissiveLlmBudgetGate
from agiwo.agent.models.execution import RunTreeRole
from agiwo.llm.base import Model, StreamChunk


def _finalization_json(*, target: str = "verifier") -> str:
    return json.dumps(
        {
            "decision": {"target": target},
            "new_contributions": [],
            "contribution_annotations": [],
            "objective_update": None,
            "artifact_refs": [],
            "carry_forward": [],
        }
    )


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


async def _run_root(model: Model, *, max_steps_per_run: int = 50):
    agent = Agent(
        AgentConfig(
            name="assignment-finalization",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=max_steps_per_run,
            ),
        ),
        id="assignment-finalization",
        model=model,
    )
    agent.llm_budget_gate = PermissiveLlmBudgetGate()
    handle = agent._start_runtime(
        "complete the assignment",
        session_id="assignment-finalization-session",
        execution_request=RunExecutionRequest(
            run_id="assignment-run",
            objective_id="objective-1",
            run_tree_role=RunTreeRole.ROOT,
        ),
    )
    return await handle.wait()


@pytest.mark.asyncio
async def test_root_empty_plan_finalizes_with_valid_json() -> None:
    model = _ScriptedModel(["ordinary report", _finalization_json()])

    result = await _run_root(model)

    assert result.response == "ordinary report"
    assert result.finalization is not None
    assert result.finalization.report == "ordinary report"
    assert result.finalization.decision["target"] == "verifier"
    assert len(model.calls) == 2
    assert model.calls[-1][-1]["origin"] == "run_finalization"


@pytest.mark.asyncio
async def test_root_pending_plan_gets_reminder_before_finalization() -> None:
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
            _finalization_json(),
        ]
    )

    result = await _run_root(model)

    assert result.finalization is not None
    assert len(model.calls) == 5
    guard_call = model.calls[2]
    assert guard_call[-1]["origin"] == "assignment_plan_guard"
    assert model.calls[4][-1]["origin"] == "run_finalization"


@pytest.mark.asyncio
async def test_non_root_run_keeps_ordinary_completion_behavior() -> None:
    model = _ScriptedModel(["ordinary report"])
    agent = Agent(
        AgentConfig(
            name="ordinary-run",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=model,
    )

    result = await agent.run("hello", session_id="ordinary-session")

    assert result.response == "ordinary report"
    assert result.finalization is None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_invalid_finalization_json_is_corrected_once() -> None:
    model = _ScriptedModel(["ordinary report", "not JSON", _finalization_json()])

    result = await _run_root(model)

    assert result.finalization is not None
    assert result.finalization.mechanical_handoff is False
    assert len(model.calls) == 3
    assert model.calls[-1][-1]["origin"] == "run_finalization"


@pytest.mark.asyncio
async def test_two_invalid_finalizations_force_mechanical_handoff() -> None:
    model = _ScriptedModel(["ordinary report", "not JSON", "also not JSON"])

    result = await _run_root(model)

    assert result.finalization is not None
    assert result.finalization.mechanical_handoff is True
    assert result.finalization.decision["target"] == "agent"
    assert result.finalization.report == "ordinary report"


@pytest.mark.asyncio
async def test_root_max_steps_forces_mechanical_agent_handoff() -> None:
    model = _ScriptedModel([])

    result = await _run_root(model, max_steps_per_run=0)

    assert result.finalization is not None
    assert result.finalization.mechanical_handoff is True
    assert result.finalization.decision == {
        "target": "agent",
        "reason": "max_steps_per_run_mechanical_handoff",
    }
