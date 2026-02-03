"""
Termination Summary - Generate summary when execution reaches limits.

Design Philosophy:
- Uses the same Model and System Prompt as the Agent (preserves LLM cache)
- Continues conversation by appending User message requesting summary
- Handles different termination positions (assistant with tools, tool results, etc.)
- Never truncates or discards existing context
"""

from agiwo.agent.prompt_template import renderer
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

# Default user prompt for requesting termination summary
DEFAULT_TERMINATION_USER_PROMPT = """**IMPORTANT: Execution Limit Reached**

The execution has been interrupted due to {{ termination_reason }}. This is NOT a normal completion.

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
    termination_reason: str,
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

    # If there are pending tool calls, add placeholder tool results
    if pending_tool_calls:
        for tool_call in pending_tool_calls:
            call_id = tool_call.get("id", "unknown")
            tool_name = tool_call.get("function", {}).get("name", "unknown")
            result_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": tool_name,
                    "content": f"[Execution interrupted: {_format_termination_reason(termination_reason)}. "
                    f"This tool call was not executed.]",
                }
            )

    # Build the user prompt requesting summary using Jinja2
    prompt_template = custom_prompt or DEFAULT_TERMINATION_USER_PROMPT
    user_prompt = renderer.render(
        prompt_template,
        termination_reason=_format_termination_reason(termination_reason),
    )

    result_messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    return result_messages


def _format_termination_reason(reason: str) -> str:
    """Format termination reason for human readability."""
    reason_mapping = {
        "max_steps": "reaching the maximum number of execution steps",
        "max_iterations": "reaching the maximum number of iterations",
        "timeout": "execution timeout",
        "cancelled": "user cancellation",
        "tool_limit": "reaching the tool call limit",
    }
    return reason_mapping.get(reason, reason)


__all__ = [
    "build_termination_messages",
    "DEFAULT_TERMINATION_USER_PROMPT",
    "_format_termination_reason",
]
