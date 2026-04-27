from agiwo.agent.introspect.models import IntrospectionOutcome
from agiwo.agent.introspect.repair import build_context_repair_plan


def test_step_back_repairs_only_after_previous_boundary() -> None:
    messages = [
        {"role": "tool", "tool_call_id": "tc-old", "content": "old", "_sequence": 2},
        {"role": "tool", "tool_call_id": "tc-new", "content": "new", "_sequence": 5},
        {
            "role": "tool",
            "tool_call_id": "tc-review",
            "content": "review",
            "_sequence": 6,
        },
    ]
    outcome = IntrospectionOutcome(
        aligned=False,
        mode="step_back",
        boundary_seq=6,
        experience="new search drifted",
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    plan = build_context_repair_plan(
        messages,
        outcome,
        previous_boundary_seq=3,
        step_lookup={"tc-new": {"id": "step-new", "sequence": 5}},
    )

    assert plan is not None
    assert plan.start_seq == 4
    assert plan.end_seq == 5
    assert [(u.tool_call_id, u.content) for u in plan.content_updates] == [
        ("tc-new", "[EXPERIENCE] new search drifted")
    ]


def test_aligned_review_cleans_prompt_notice() -> None:
    messages = [
        {
            "role": "tool",
            "tool_call_id": "tc-search",
            "content": "result\n<system-review>check</system-review>",
            "_sequence": 4,
        }
    ]
    outcome = IntrospectionOutcome(
        aligned=True,
        mode="metadata_only",
        boundary_seq=5,
        review_tool_call_id="tc-review",
        review_step_id="step-review",
    )

    plan = build_context_repair_plan(
        messages,
        outcome,
        previous_boundary_seq=0,
        step_lookup={"tc-search": {"id": "step-search", "sequence": 4}},
    )

    assert plan is not None
    assert plan.content_updates[0].content == "result"
    assert plan.notice_cleaned_step_ids == ["step-search"]
