"""Root-run plan guard and mechanical next-action tests (ADR 0047)."""

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
    verification_required: bool = False,
    objective_run_role: str | None = "work",
):
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
    handle = agent.start_prevalidated(
        "complete the assignment",
        session_id="assignment-finalization-session",
        execution_request=RunExecutionRequest(
            run_id="assignment-run",
            objective_id="objective-1",
            run_tree_role=RunTreeRole.ROOT,
            verification_required=verification_required,
            objective_run_role=objective_run_role,
        ),
    )
    return await handle.wait()


@pytest.mark.asyncio
async def test_simple_root_mechanical_delivery() -> None:
    model = _ScriptedModel(["ordinary report"])

    result = await _run_root(model, verification_required=False)

    assert result.response == "ordinary report"
    assert result.finalization is not None
    assert result.finalization.decision["target"] == "user"
    assert result.finalization.decision["expects_reply"] is False
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_verification_required_mechanical_verifier_handoff() -> None:
    model = _ScriptedModel(["ordinary report"])

    result = await _run_root(model, verification_required=True)

    assert result.finalization is not None
    assert result.finalization.decision["target"] == "verifier"
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_verification_role_mechanical_delivery() -> None:
    model = _ScriptedModel(["verified ok"])

    result = await _run_root(
        model,
        verification_required=True,
        objective_run_role="verification",
    )

    assert result.finalization is not None
    assert result.finalization.decision["target"] == "user"
    assert result.finalization.decision["expects_reply"] is False
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_plan_milestones_arm_verifier_handoff() -> None:
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

    result = await _run_root(model, verification_required=False)

    assert result.finalization is not None
    assert result.finalization.decision["target"] == "verifier"
    assert len(model.calls) == 4
    guard_call = model.calls[2]
    assert guard_call[-1]["origin"] == "assignment_plan_guard"


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
    agent.llm_budget_gate = PermissiveLlmBudgetGate()
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
    assert result.finalization is None
    assert len(model.calls) == 1


@pytest.mark.asyncio
async def test_root_max_steps_forces_mechanical_agent_handoff() -> None:
    model = _ScriptedModel([])

    result = await _run_root(model, max_steps_per_run=0)

    assert result.finalization is not None
    assert result.finalization.decision == {
        "target": "agent",
        "reason": "max_steps_per_run_mechanical_handoff",
    }
