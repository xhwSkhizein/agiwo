"""P2-06: work deliver and work→verification mainline."""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import Agent, AgentOptions, UserMessage
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType
from agiwo.llm.base import Model, StreamChunk
from agiwo.objective import ObjectiveService
from agiwo.objective.models import (
    BudgetLimits,
    CreateObjectiveRequest,
    ObjectiveStatus,
    RunRole,
    new_id,
)
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore
from agiwo.scheduler import Scheduler


def _user(text: str) -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def _finalization_json(
    *,
    target: str,
    expects_reply: bool | None = None,
    report: str | None = None,
) -> str:
    del report
    decision: dict = {"target": target}
    if expects_reply is not None:
        decision["expects_reply"] = expects_reply
    return json.dumps(
        {
            "decision": decision,
            "new_contributions": [],
            "contribution_annotations": [],
            "objective_update": None,
            "artifact_refs": [],
            "carry_forward": [],
        }
    )


class ScriptedFinalizationModel(Model):
    """Queue of model responses: ordinary report, then finalization JSON, ..."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = list(responses)
        self.calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools, messages
        self.calls += 1
        if not self._responses:
            text = _finalization_json(report="fallback", target="agent")
        else:
            text = self._responses.pop(0)
        yield StreamChunk(content=text)
        yield StreamChunk(finish_reason="stop")


def _scripted_run_turns(*finals: str) -> list[str]:
    """Each root Run: one ordinary report turn + one finalization JSON turn."""
    responses: list[str] = []
    for final in finals:
        responses.append("working")
        responses.append(final)
    return responses


async def _wait_completed(service: ObjectiveService, objective_id: str) -> None:
    for _ in range(80):
        view = await service.get_view(objective_id)
        assert view is not None
        if view.status == ObjectiveStatus.COMPLETED and view.delivery_report:
            return
        await asyncio.sleep(0.1)
    view = await service.get_view(objective_id)
    raise AssertionError(
        f"objective not completed: status={view.status if view else None} "
        f"root_runs={[(r.role, r.status) for r in view.root_runs] if view else []}"
    )


@pytest.mark.asyncio
async def test_simple_work_delivered_mainline() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()
    model = ScriptedFinalizationModel(
        _scripted_run_turns(
            _finalization_json(
                report="delivered",
                target="user",
                expects_reply=False,
            ),
        )
    )
    agent = Agent(
        AgentConfig(
            name="mainline",
            description="mainline",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=10,
            ),
        ),
        model=model,
        id="session-mainline",
    )

    async def provider(session_id: str) -> Agent:
        del session_id
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    await service.start_dispatcher()
    try:
        created = await service.create_objective(
            CreateObjectiveRequest(
                session_id="session-mainline",
                user_message=_user("please complete the report"),
                budget=BudgetLimits(
                    handoffs=10,
                    verification_attempts=5,
                    llm_cost_usd=10.0,
                    active_seconds=3600,
                ),
                idempotency_key=new_id(),
            )
        )
        assert created.payload.get("run_id")
        await _wait_completed(service, created.objective_id)
        view = await service.get_view(created.objective_id)
        assert view is not None
        assert view.status == ObjectiveStatus.COMPLETED
        assert view.delivery_report
        roles = [r.role for r in view.root_runs]
        assert roles == [RunRole.WORK]
        assert all(r.outcome is not None for r in view.root_runs)
    finally:
        await service.stop_dispatcher()
        await scheduler.stop()
        await agent.close()


@pytest.mark.asyncio
async def test_simple_work_delivered_sqlite(
    tmp_path,
) -> None:
    db_path = tmp_path / "objective-mainline.db"
    store = SQLiteObjectiveStore(str(db_path))
    await store.connect()
    scheduler = Scheduler()
    await scheduler.start()
    model = ScriptedFinalizationModel(
        _scripted_run_turns(
            _finalization_json(
                report="delivered",
                target="user",
                expects_reply=False,
            ),
        )
    )
    agent = Agent(
        AgentConfig(
            name="mainline-sqlite",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=model,
        id="session-mainline-sqlite",
    )

    async def provider(session_id: str) -> Agent:
        del session_id
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    await service.start_dispatcher()
    try:
        created = await service.create_objective(
            CreateObjectiveRequest(
                session_id="session-mainline-sqlite",
                user_message=_user("sqlite mainline"),
                budget=BudgetLimits(
                    handoffs=10,
                    verification_attempts=5,
                    llm_cost_usd=10.0,
                    active_seconds=3600,
                ),
                idempotency_key=new_id(),
            )
        )
        await _wait_completed(service, created.objective_id)
        view = await service.get_view(created.objective_id)
        assert view is not None
        assert view.status == ObjectiveStatus.COMPLETED
    finally:
        await service.stop_dispatcher()
        await scheduler.stop()
        await agent.close()
        await store.close()


@pytest.mark.asyncio
async def test_work_verification_delivered_when_required() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()
    model = ScriptedFinalizationModel(
        _scripted_run_turns(
            _finalization_json(report="work done", target="verifier"),
            _finalization_json(
                report="verified and delivered",
                target="user",
                expects_reply=False,
            ),
        )
    )
    agent = Agent(
        AgentConfig(
            name="mainline-verify",
            description="mainline-verify",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=10,
            ),
        ),
        model=model,
        id="session-mainline-verify",
    )

    async def provider(session_id: str) -> Agent:
        del session_id
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    await service.start_dispatcher()
    try:
        created = await service.create_objective(
            CreateObjectiveRequest(
                session_id="session-mainline-verify",
                user_message=_user("please complete the report"),
                budget=BudgetLimits(
                    handoffs=10,
                    verification_attempts=5,
                    llm_cost_usd=10.0,
                    active_seconds=3600,
                ),
                idempotency_key=new_id(),
            )
        )
        await _wait_completed(service, created.objective_id)
        view = await service.get_view(created.objective_id)
        assert view is not None
        assert view.status == ObjectiveStatus.COMPLETED
        assert view.delivery_report
        roles = [r.role for r in view.root_runs]
        assert RunRole.WORK in roles
        assert RunRole.VERIFICATION in roles
        # Voluntary verifier handoff does not set the plan latch.
        assert view.verification_required is False
    finally:
        await service.stop_dispatcher()
        await scheduler.stop()
        await agent.close()


@pytest.mark.asyncio
async def test_verifier_reject_creates_fresh_work() -> None:
    store = InMemoryObjectiveStore()
    scheduler = Scheduler()
    await scheduler.start()
    model = ScriptedFinalizationModel(
        _scripted_run_turns(
            _finalization_json(report="work v1", target="verifier"),
            _finalization_json(report="reject", target="agent"),
            _finalization_json(report="work v2", target="verifier"),
            _finalization_json(
                report="ok delivered",
                target="user",
                expects_reply=False,
            ),
        )
    )
    agent = Agent(
        AgentConfig(
            name="mainline-reject",
            options=AgentOptions(enable_termination_summary=False),
        ),
        model=model,
        id="session-mainline-reject",
    )

    async def provider(session_id: str) -> Agent:
        del session_id
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    await service.start_dispatcher()
    try:
        created = await service.create_objective(
            CreateObjectiveRequest(
                session_id="session-mainline-reject",
                user_message=_user("do the work"),
                budget=BudgetLimits(
                    handoffs=20,
                    verification_attempts=10,
                    llm_cost_usd=10.0,
                    active_seconds=3600,
                ),
                idempotency_key=new_id(),
            )
        )
        await _wait_completed(service, created.objective_id)
        view = await service.get_view(created.objective_id)
        assert view is not None
        work_count = sum(1 for r in view.root_runs if r.role is RunRole.WORK)
        assert work_count >= 2
    finally:
        await service.stop_dispatcher()
        await scheduler.stop()
        await agent.close()
