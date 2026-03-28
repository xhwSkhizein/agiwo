"""Prompt helpers for interrupted-run termination summaries."""

from agiwo.agent.models.run import TerminationReason

TERMINATION_SUMMARY_REASONS = {
    TerminationReason.MAX_STEPS,
    TerminationReason.TIMEOUT,
    TerminationReason.MAX_OUTPUT_TOKENS,
    TerminationReason.MAX_INPUT_TOKENS_PER_CALL,
    TerminationReason.MAX_RUN_COST,
    TerminationReason.CANCELLED,
    TerminationReason.TOOL_LIMIT,
    TerminationReason.ERROR,
    TerminationReason.ERROR_WITH_CONTEXT,
}

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


def format_termination_reason(reason: TerminationReason | str) -> str:
    reason_val = reason.value if isinstance(reason, TerminationReason) else reason
    reason_mapping = {
        TerminationReason.MAX_STEPS.value: "reaching the maximum number of execution steps",
        TerminationReason.TIMEOUT.value: "execution timeout",
        TerminationReason.MAX_OUTPUT_TOKENS.value: "reaching model output token limit for one LLM call",
        TerminationReason.MAX_INPUT_TOKENS_PER_CALL.value: "reaching model input token limit for one LLM call",
        TerminationReason.MAX_RUN_COST.value: "reaching maximum token cost budget for this run",
        TerminationReason.CANCELLED.value: "user cancellation",
        TerminationReason.TOOL_LIMIT.value: "reaching the tool call limit",
        TerminationReason.ERROR.value: "internal error",
        TerminationReason.ERROR_WITH_CONTEXT.value: "error with context",
        TerminationReason.COMPLETED.value: "completed successfully",
        TerminationReason.SLEEPING.value: "sleeping/waiting",
    }
    return reason_mapping.get(reason_val, reason_val)


def render_termination_summary_prompt(
    prompt_template: str,
    reason: TerminationReason | str,
) -> str:
    reason_text = format_termination_reason(reason)
    if "%s" not in prompt_template:
        return prompt_template
    return prompt_template % reason_text


__all__ = [
    "DEFAULT_TERMINATION_USER_PROMPT",
    "TERMINATION_SUMMARY_REASONS",
    "format_termination_reason",
    "render_termination_summary_prompt",
]
