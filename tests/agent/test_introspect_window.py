from agiwo.agent.introspect.tool import ReviewTrajectoryTool
from agiwo.agent.introspect.window import build_review_window


def test_review_window_exposes_only_tool_call_id_and_name() -> None:
    window = build_review_window(
        [
            {
                "role": "tool",
                "tool_call_id": "tc-old",
                "name": "search",
                "content": "old",
                "_sequence": 2,
            },
            {
                "role": "tool",
                "tool_call_id": "tc-new",
                "name": "read",
                "content": "new",
                "_sequence": 5,
            },
            {
                "role": "tool",
                "tool_call_id": "tc-review",
                "name": "review_trajectory",
                "content": "review",
                "_sequence": 6,
            },
        ],
        last_boundary_seq=3,
    )

    assert window == [
        {"tool_call_id": "tc-new", "tool_name": "read"},
    ]
    assert all(set(item) <= {"tool_call_id", "tool_name"} for item in window)


def test_review_tool_parameters_do_not_expose_internal_ids() -> None:
    params = ReviewTrajectoryTool().get_parameters()
    schema_text = str(params)
    assert "sequence" not in schema_text
    assert "step_id" not in schema_text
