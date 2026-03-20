"""Feishu tool consent notifier — interactive card for user approval."""

import json

from agiwo.agent.tool_auth.notifier import ToolConsentNotifier
from agiwo.utils.logging import get_logger

from server.channels.feishu.api_client import FeishuApiClient

logger = get_logger(__name__)


class FeishuToolConsentNotifier(ToolConsentNotifier):
    """Send interactive Feishu cards for tool consent requests."""

    def __init__(
        self,
        *,
        api: FeishuApiClient,
        chat_id: str,
    ) -> None:
        self._api = api
        self._chat_id = chat_id

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
        card_content = self._build_consent_card(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            args_preview=args_preview,
            reason=reason,
        )

        try:
            await self._api.create_post_message(self._chat_id, card_content)
            logger.info(
                "feishu_consent_notification_sent",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
                chat_id=self._chat_id,
            )
        except Exception as error:
            logger.error(
                "feishu_consent_notification_failed",
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
        message = f"🚫 工具调用被拒绝\n\n工具: {tool_name}\n原因: {reason}"
        try:
            await self._api.create_text_message(self._chat_id, message)
            logger.info(
                "feishu_consent_denied_notification_sent",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                run_id=run_id,
            )
        except Exception as error:
            logger.error(
                "feishu_consent_denied_notification_failed",
                tool_call_id=tool_call_id,
                error=str(error),
                exc_info=True,
            )

    def _build_consent_card(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        args_preview: str,
        reason: str,
    ) -> dict:
        """Build interactive card with approve/deny buttons."""
        return {
            "zh_cn": {
                "title": "🔐 工具授权请求",
                "content": [
                    [
                        {
                            "tag": "text",
                            "text": f"工具: {tool_name}\n",
                        }
                    ],
                    [
                        {
                            "tag": "text",
                            "text": f"参数: {args_preview}\n",
                        }
                    ],
                    [
                        {
                            "tag": "text",
                            "text": f"原因: {reason}\n",
                        }
                    ],
                    [
                        {
                            "tag": "text",
                            "text": "\n请选择是否允许执行此工具：",
                        }
                    ],
                    [
                        {
                            "tag": "button",
                            "text": {
                                "tag": "plain_text",
                                "content": "✅ 允许",
                            },
                            "type": "primary",
                            "value": json.dumps(
                                {
                                    "action": "consent_approve",
                                    "tool_call_id": tool_call_id,
                                }
                            ),
                        },
                        {
                            "tag": "button",
                            "text": {
                                "tag": "plain_text",
                                "content": "❌ 拒绝",
                            },
                            "type": "danger",
                            "value": json.dumps(
                                {
                                    "action": "consent_deny",
                                    "tool_call_id": tool_call_id,
                                }
                            ),
                        },
                    ],
                ],
            }
        }


__all__ = ["FeishuToolConsentNotifier"]
