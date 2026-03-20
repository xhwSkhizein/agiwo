"""Web chat tool consent notifier — SSE event stream for user approval."""

from agiwo.agent.tool_auth.notifier import ToolConsentNotifier
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebChatToolConsentNotifier(ToolConsentNotifier):
    """Send consent requests via agent stream events for web chat."""

    def __init__(self, *, run_id: str, stream_callback) -> None:
        self._run_id = run_id
        self._stream_callback = stream_callback

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
        del suggested_patterns
        event = {
            "type": "consent_required",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "args_preview": args_preview,
            "reason": reason,
            "run_id": run_id,
        }

        try:
            await self._stream_callback(event)
            logger.info(
                "web_consent_notification_sent",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
            )
        except Exception as error:
            logger.error(
                "web_consent_notification_failed",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                error=str(error),
                exc_info=True,
            )

    async def notify_denied(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        run_id: str,
        reason: str,
    ) -> None:
        event = {
            "type": "consent_denied",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "reason": reason,
            "run_id": run_id,
        }

        try:
            await self._stream_callback(event)
            logger.info(
                "web_consent_denied_notification_sent",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
            )
        except Exception as error:
            logger.error(
                "web_consent_denied_notification_failed",
                tool_call_id=tool_call_id,
                error=str(error),
                exc_info=True,
            )


__all__ = ["WebChatToolConsentNotifier"]
