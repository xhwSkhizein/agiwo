"""Shared ObjectiveService wiring for Console API integration tests."""

from collections.abc import AsyncIterator

from agiwo.agent import Agent, AgentConfig
from agiwo.agent.models.config import AgentOptions
from agiwo.llm.base import Model, StreamChunk
from agiwo.objective import ObjectiveService, default_assignment_templates
from agiwo.scheduler.engine import Scheduler


class _StubModel(Model):
    def __init__(self) -> None:
        super().__init__(id="stub", name="stub", temperature=0.0)
        self._calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        self._calls += 1
        if self._calls == 1:
            yield StreamChunk(content="stub")
        else:
            yield StreamChunk(
                content=(
                    '{"decision":{"target":"user","expects_reply":false},'
                    '"new_contributions":[],"contribution_annotations":[],'
                    '"objective_update":null,"artifact_refs":[],"carry_forward":[]}'
                )
            )
        yield StreamChunk(finish_reason="stop")


def build_test_objective_service(
    objective_store,
    scheduler: Scheduler,
) -> tuple[Agent, ObjectiveService]:
    agent = Agent(
        AgentConfig(
            name="console-test",
            description="console test agent",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=5,
            ),
        ),
        model=_StubModel(),
        id="console-test-agent",
    )

    async def default_agent_provider(_session_id: str) -> Agent:
        return agent

    service = ObjectiveService(
        objective_store,
        scheduler=scheduler,
        default_agent_provider=default_agent_provider,
        templates=default_assignment_templates(),
    )
    return agent, service


__all__ = ["build_test_objective_service"]
