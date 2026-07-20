"""SessionObjectiveGateway create-or-continue tests (P5-03)."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agiwo.agent.models.input import UserMessage
from agiwo.objective import (
    BudgetLimits,
    CommandResult,
    ObjectiveService,
    ObjectiveStatus,
)
from agiwo.objective.store.memory import InMemoryObjectiveStore

from server.models.session import Session
from server.services.objective_gateway import SessionObjectiveGateway
from server.services.session_store import InMemorySessionStore


@pytest.fixture
async def session_store():
    store = InMemorySessionStore()
    await store.connect()
    now = datetime.now(timezone.utc)
    await store.upsert_session(
        Session(
            id="sess-gw",
            chat_context_scope_id=None,
            base_agent_id="agent-1",
            created_by="TEST",
            created_at=now,
            updated_at=now,
        )
    )
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_no_active_objective_creates(session_store) -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    gateway = SessionObjectiveGateway(
        objective_service=service,
        session_store=session_store,
    )
    result = await gateway.handle_user_message(
        "sess-gw",
        UserMessage.from_value("first turn"),
        idempotency_key="gw-create",
        budget=BudgetLimits(
            handoffs=5,
            verification_attempts=3,
            llm_cost_usd=2.0,
            active_seconds=600,
        ),
    )
    assert result.objective_id
    views = await service.list_by_session("sess-gw")
    assert len(views) == 1
    assert not views[0].is_terminal


@pytest.mark.asyncio
async def test_active_objective_submits_input(session_store) -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    gateway = SessionObjectiveGateway(
        objective_service=service,
        session_store=session_store,
    )
    created = await gateway.handle_user_message(
        "sess-gw",
        UserMessage.from_value("start"),
        idempotency_key="gw-1",
        budget=BudgetLimits(
            handoffs=5,
            verification_attempts=3,
            llm_cost_usd=2.0,
            active_seconds=600,
        ),
    )
    continued = await gateway.handle_user_message(
        "sess-gw",
        UserMessage.from_value("more"),
        idempotency_key="gw-2",
    )
    assert continued.objective_id == created.objective_id
    views = await service.list_by_session("sess-gw")
    assert len(views) == 1


@pytest.mark.asyncio
async def test_terminal_objective_creates_new(session_store, monkeypatch) -> None:
    service = ObjectiveService(InMemoryObjectiveStore())
    gateway = SessionObjectiveGateway(
        objective_service=service,
        session_store=session_store,
    )

    terminal = SimpleNamespace(
        objective_id="obj_old",
        is_terminal=True,
        status=ObjectiveStatus.COMPLETED,
    )

    async def fake_list(_session_id: str):
        return [terminal]

    create_mock = AsyncMock(
        return_value=CommandResult(
            objective_id="obj_new",
            status=ObjectiveStatus.CREATED.value,
        )
    )
    monkeypatch.setattr(service, "list_by_session", fake_list)
    monkeypatch.setattr(service, "create_objective", create_mock)

    result = await gateway.handle_user_message(
        "sess-gw",
        UserMessage.from_value("after complete"),
        idempotency_key="gw-new",
    )
    assert result.objective_id == "obj_new"
    create_mock.assert_awaited_once()
