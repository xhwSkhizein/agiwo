"""Web chat tool consent notifier — SSE event stream for user approval."""

from collections.abc import Awaitable, Callable
from datetime import datetime

from agiwo.agent.runtime.stream_events import ConsentDeniedEvent, ConsentRequiredEvent
from agiwo.agent.tool_auth.notifier import ToolConsentNotifier
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class WebChatToolConsentNotifier(ToolConsentNotifier):
    """Send consent requests via agent stream events for web chat."""

    def __init__(
        self,
        *,
        session_id: str,
        agent_id: str,
        parent_run_id: str | None,
        depth: int,
        publish_callback: Callable[[ConsentRequiredEvent | ConsentDeniedEvent], Awaitable[None]],
    ) -> None:
        self._session_id = session_id
        self._agent_id = agent_id
        self._parent_run_id = parent_run_id
        self._depth = depth
        self._publish = publish_callback

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
        """Push consent_required event to web chat SSE stream."""
        event = ConsentRequiredEvent(
            session_id=self._session_id,
            run_id=run_id,
            agent_id=self._agent_id,
            parent_run_id=self._parent_run_id,
            depth=self._depth,
            timestamp=datetime.now(),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args_preview=args_preview,
            reason=reason,
            suggested_patterns=suggested_patterns,
        )

        try:
            await self._publish(event)
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
        """Push consent_denied event to web chat SSE stream."""
        event = ConsentDeniedEvent(
            session_id=self._session_id,
            run_id=run_id,
            agent_id=self._agent_id,
            parent_run_id=self._parent_run_id,
            depth=self._depth,
            timestamp=datetime.now(),
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            reason=reason,
        )

        try:
            await self._publish(event)
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
