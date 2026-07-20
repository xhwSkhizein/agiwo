"""P1-02 ObjectiveLog fact construction tests."""

from datetime import datetime, timezone

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.objective.log import (
    FactBatch,
    ObjectiveFactKind,
    ObjectiveLogEntry,
    fact_objective_created,
    fact_objective_status_changed,
    fact_objective_user_input,
    fact_user_input_externalized,
)
from agiwo.objective.models import ObjectiveBudget, ObjectiveStatus, ObjectiveUserInput


def test_fact_roundtrip() -> None:
    budget = ObjectiveBudget.create(
        handoffs=3, verification_attempts=2, llm_cost_usd=1.0, active_seconds=100
    )
    entry = fact_objective_created(
        session_id="sess1",
        budget=budget,
    ).materialize(objective_id="obj1", sequence=1)
    assert entry.kind == ObjectiveFactKind.OBJECTIVE_CREATED
    restored = ObjectiveLogEntry.from_dict(entry.to_dict())
    assert restored.fact_id == entry.fact_id
    assert restored.sequence == 1


def test_user_input_fact_preserves_full_message() -> None:
    msg = UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text="build a plan")],
        is_user_provided=True,
    )
    ui = ObjectiveUserInput(input_id="inp1", message=msg)
    draft = fact_objective_user_input(user_input=ui)
    assert draft.payload["input_id"] == "inp1"
    assert draft.payload["message"]["content"][0]["text"] == "build a plan"


def test_externalized_fact_keeps_input_id() -> None:
    draft = fact_user_input_externalized(
        input_id="inp1",
        artifact_id="art1",
    )
    assert draft.payload["input_id"] == "inp1"
    assert draft.payload["artifact_id"] == "art1"


def test_fact_batch_allocates_contiguous_sequences() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    budget = ObjectiveBudget.create(
        handoffs=1, verification_attempts=1, llm_cost_usd=1.0, active_seconds=10
    )
    batch = FactBatch(objective_id="obj1", start_sequence=5, now=now)
    batch.add(fact_objective_created(session_id="s1", budget=budget))
    batch.add(
        fact_objective_status_changed(
            from_status=ObjectiveStatus.CREATED,
            to_status=ObjectiveStatus.RUNNING,
            reason="start",
        )
    )
    assert [f.sequence for f in batch.facts] == [5, 6]
    assert all(f.objective_id == "obj1" for f in batch.facts)
    assert all(f.occurred_at == now for f in batch.facts)
    assert batch.next_sequence == 7
