"""Streaming consent notifier — logs consent requests for Console monitoring."""

from agiwo.agent.tool_auth.notifier import ToolConsentNotifier
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class StreamingConsentNotifier(ToolConsentNotifier):
    """Log consent requests - Console layer monitors logs and pushes to streams."""

    async def notify_required(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        run_id: str,
        args_preview: str,
        reason: str,
        suggested_patterns: list[str] | None,
    ) -> None:
        """Log consent request with structured data."""
        logger.warning(
            "tool_consent_required",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=run_id,
            args_preview=args_preview[:200],
            reason=reason,
            suggested_patterns=suggested_patterns,
            message=(
                f"Tool '{tool_name}' requires user consent. "
                f"Call POST /api/consent/resolve with tool_call_id='{tool_call_id}' "
                f"and decision='allow' or 'deny' to proceed."
            ),
        )

    async def notify_denied(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        run_id: str,
        reason: str,
    ) -> None:
        """Log consent denial."""
        logger.info(
            "tool_consent_denied",
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            run_id=run_id,
            reason=reason,
        )


__all__ = ["StreamingConsentNotifier"]
