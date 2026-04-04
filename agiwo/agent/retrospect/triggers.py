"""Retrospect trigger detection and system-notice injection."""

from enum import Enum

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.run import RunLedger


class RetrospectTrigger(Enum):
    """Which condition fired the retrospect notice."""

    NONE = "none"
    LARGE_RESULT = "large_result"
    ROUND_INTERVAL = "round_interval"
    TOKEN_ACCUMULATED = "token_accumulated"


_NOTICE_LARGE_RESULT = (
    "\n\n<system-notice>"
    "This tool result is large. If it has limited value for your current "
    "goal, call retrospect_tool_result to replace it with a concise summary."
    "</system-notice>"
)

_NOTICE_ROUND_INTERVAL = (
    "\n\n<system-notice>"
    "You have made several tool calls since your last retrospect. Review "
    "whether the recent results meaningfully advance your current goal. "
    "If not, call retrospect_tool_result to consolidate your findings and "
    "refocus on the objective."
    "</system-notice>"
)

_NOTICE_TOKEN_ACCUMULATED = (
    "\n\n<system-notice>"
    "Tool results have accumulated significant context. To keep the "
    "conversation window effective, call retrospect_tool_result to replace "
    "low-value results with a concise summary of key findings."
    "</system-notice>"
)

_NOTICE_BY_TRIGGER = {
    RetrospectTrigger.LARGE_RESULT: _NOTICE_LARGE_RESULT,
    RetrospectTrigger.ROUND_INTERVAL: _NOTICE_ROUND_INTERVAL,
    RetrospectTrigger.TOKEN_ACCUMULATED: _NOTICE_TOKEN_ACCUMULATED,
}


def check_retrospect_trigger(
    config: AgentOptions,
    ledger: RunLedger,
    content: str,
    tool_name: str,
) -> RetrospectTrigger:
    """Return the trigger type, or ``NONE`` if no notice should be injected."""
    if not config.enable_tool_retrospect:
        return RetrospectTrigger.NONE
    if tool_name == "retrospect_tool_result":
        return RetrospectTrigger.NONE

    token_estimate = len(content) // 4

    if token_estimate >= config.retrospect_token_threshold:
        return RetrospectTrigger.LARGE_RESULT
    if ledger.retrospect_pending_rounds >= config.retrospect_round_interval:
        return RetrospectTrigger.ROUND_INTERVAL
    if (
        ledger.retrospect_pending_tokens
        >= config.retrospect_accumulated_token_threshold
    ):
        return RetrospectTrigger.TOKEN_ACCUMULATED
    return RetrospectTrigger.NONE


def update_retrospect_tracking(ledger: RunLedger, content: str) -> None:
    """Accumulate retrospect tracking counters after a tool result."""
    token_estimate = len(content) // 4
    ledger.retrospect_pending_tokens += token_estimate
    ledger.retrospect_pending_rounds += 1


def inject_system_notice(content: str, trigger: RetrospectTrigger) -> str:
    """Append the trigger-specific system-notice to *content*.

    Returns *content* unchanged when *trigger* is ``NONE``.
    """
    notice = _NOTICE_BY_TRIGGER.get(trigger)
    if notice is None:
        return content
    return content + notice


__all__ = [
    "RetrospectTrigger",
    "check_retrospect_trigger",
    "inject_system_notice",
    "update_retrospect_tracking",
]
