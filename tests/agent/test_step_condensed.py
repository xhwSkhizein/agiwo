"""Tests for StepView.condensed_content and to_message() behaviour."""

from agiwo.agent.models.step import MessageRole, StepView


class TestStepViewCondensedContent:
    def test_to_message_uses_content_when_no_condensed(self):
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.TOOL,
            content="original result",
            tool_call_id="tc-1",
            name="bash",
        )
        msg = step.to_message()
        assert msg["content"] == "original result"

    def test_to_message_prefers_condensed_content(self):
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.TOOL,
            content="original verbose result",
            tool_call_id="tc-1",
            name="bash",
            condensed_content="[archived] Retrospect: short summary",
        )
        msg = step.to_message()
        assert msg["content"] == "[archived] Retrospect: short summary"

    def test_assistant_message_uses_condensed_content(self):
        """condensed_content applies to all roles via to_message()."""
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.ASSISTANT,
            content="thinking...",
            condensed_content="override",
        )
        msg = step.to_message()
        assert msg["content"] == "override"

    def test_condensed_content_defaults_to_none(self):
        step = StepView(
            session_id="s1",
            run_id="r1",
            sequence=1,
            role=MessageRole.TOOL,
            content="result",
            tool_call_id="tc-1",
            name="bash",
        )
        assert step.condensed_content is None
