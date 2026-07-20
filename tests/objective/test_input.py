"""Run input assembly and history correspondence."""

from agiwo.agent import UserMessage
from agiwo.agent.models.input import ContentPart, ContentType
from agiwo.objective.input import (
    collect_history_input_ids,
    render_assignment_input,
    tag_user_message_with_input_id,
    verify_user_inputs_in_history,
)
from agiwo.objective.models import (
    BudgetLimits,
    ObjectiveStatus,
    ObjectiveUserInput,
    RunRole,
    utc_now,
)
from agiwo.objective.projection import ObjectiveView
from agiwo.objective.templates import default_run_templates


def test_render_assignment_input_is_system_attributed() -> None:
    view = ObjectiveView(
        objective_id="obj_1",
        session_id="sess_1",
        status=ObjectiveStatus.RUNNING,
        budget=BudgetLimits(
            handoffs=1,
            verification_attempts=1,
            llm_cost_usd=1.0,
            active_seconds=60,
        ).to_budget(),
        created_at=utc_now(),
        updated_at=utc_now(),
    )
    message, _template, digest = render_assignment_input(
        view,
        kind=RunRole.WORK,
        templates=default_run_templates(),
        run_id="run_1",
    )
    assert message.is_user_provided is False
    assert "Run boundary: run_id=run_1" in message.extract_text()
    assert digest


def test_history_gap_detection() -> None:
    msg = tag_user_message_with_input_id(
        UserMessage(content=[ContentPart(type=ContentType.TEXT, text="hi")]),
        "inp_1",
    )
    inputs = (
        ObjectiveUserInput(input_id="inp_1", message=msg),
        ObjectiveUserInput(
            input_id="inp_2",
            message=tag_user_message_with_input_id(
                UserMessage(content=[ContentPart(type=ContentType.TEXT, text="more")]),
                "inp_2",
            ),
        ),
    )
    history = collect_history_input_ids(
        [{"role": "user", "objective_input_id": "inp_1", "content": "hi"}]
    )
    missing = verify_user_inputs_in_history(
        inputs,
        history_input_ids=history,
        externalized_input_ids=set(),
    )
    assert missing == ["inp_2"]
