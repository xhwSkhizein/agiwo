"""
Termination Summary - Generate summary when execution reaches limits.

Design Philosophy:
- Uses the same Model and System Prompt as the Agent (preserves LLM cache)
- Continues conversation by appending User message requesting summary
- Handles different termination positions (assistant with tools, tool results, etc.)
- Never truncates or discards existing context
"""

import json

from agiwo.agent.schema import TerminationReason
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

# Default user prompt for requesting termination summary
DEFAULT_TERMINATION_USER_PROMPT = """**IMPORTANT: Execution Limit Reached**

The execution has been interrupted due to %s. This is NOT a normal completion.

Please provide a summary report that includes:
1. **Original Request**: What was the user asking for
2. **Work Completed**: What has been accomplished so far (be specific, cite actual results)
3. **Pending Work**: What remains incomplete or was interrupted
4. **Key Findings & Refs**: Any important results, data, or conclusions discovered with references

**Requirements**:
- Base your summary ONLY on the actual work done and results obtained - do not fabricate or assume
- If you must make any assumptions, clearly mark them as such
- Clearly indicate this is an INCOMPLETE/INTERRUPTED execution report
- Use the same language as the original request"""


def build_termination_messages(
    messages: list[dict],
    termination_reason: TerminationReason | str,
    pending_tool_calls: list[dict] | None = None,
    custom_prompt: str | None = None,
) -> list[dict]:
    """
    Build messages for termination summary LLM call.

    This function prepares the message list for generating a termination summary
    by appending appropriate messages based on the termination position.

    Args:
        messages: Current conversation history (OpenAI format)
        termination_reason: Reason for termination (e.g., "max_steps", "timeout")
        pending_tool_calls: Unprocessed tool calls if terminated mid-execution
        custom_prompt: Custom user prompt for summary request

    Returns:
        New messages list with termination handling messages appended
    """
    # Create a copy to avoid modifying original
    result_messages = list(messages)

    # Build the user prompt requesting summary using Jinja2
    prompt_template = custom_prompt or DEFAULT_TERMINATION_USER_PROMPT
    termination_reason = _format_termination_reason(termination_reason)
    user_prompt = (
        prompt_template % termination_reason
        if "%s" in prompt_template
        else prompt_template
    )

    if pending_tool_calls:
        pending_tool_calls_json = json.dumps(
            pending_tool_calls,
            ensure_ascii=False,
            default=str,
        )
        user_prompt = (
            f"{user_prompt}\n\n"
            f"pending_tool_calls_not_executed_json: {pending_tool_calls_json}"
        )

    result_messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return result_messages


def _format_termination_reason(reason: TerminationReason | str) -> str:
    """Format termination reason for human readability."""
    # Convert Enum to string value if needed
    reason_val = reason.value if isinstance(reason, TerminationReason) else reason

    reason_mapping = {
        TerminationReason.MAX_STEPS.value: "reaching the maximum number of execution steps",
        TerminationReason.TIMEOUT.value: "execution timeout",
        TerminationReason.CANCELLED.value: "user cancellation",
        TerminationReason.TOOL_LIMIT.value: "reaching the tool call limit",
    }
    return reason_mapping.get(reason_val, reason_val)


__all__ = ["build_termination_messages", "DEFAULT_TERMINATION_USER_PROMPT"]
