"""Feishu card action handler for interactive consent buttons."""

import json

from agiwo.tool.authz import ConsentDecision
from agiwo.utils.logging import get_logger

from server.services.consent_manager import get_consent_manager

logger = get_logger(__name__)


async def handle_consent_card_action(action_value: str) -> dict[str, str]:
    """Handle consent approval/denial from Feishu card button clicks.
    
    Args:
        action_value: JSON string from button value field
        
    Returns:
        Response dict for Feishu API
    """
    try:
        action_data = json.loads(action_value)
        action = action_data.get("action")
        tool_call_id = action_data.get("tool_call_id")
        
        if not action or not tool_call_id:
            logger.error("invalid_consent_action_data", action_value=action_value)
            return {"msg": "invalid_action_data"}
        
        consent_manager = get_consent_manager()
        
        if action == "consent_approve":
            decision = ConsentDecision(decision="allow", patterns=[])
            await consent_manager.waiter.resolve(tool_call_id, decision)
            logger.info(
                "consent_approved_via_feishu",
                tool_call_id=tool_call_id,
            )
            return {"msg": "consent_approved"}
        
        elif action == "consent_deny":
            decision = ConsentDecision(decision="deny", patterns=[])
            await consent_manager.waiter.resolve(tool_call_id, decision)
            logger.info(
                "consent_denied_via_feishu",
                tool_call_id=tool_call_id,
            )
            return {"msg": "consent_denied"}
        
        else:
            logger.warning("unknown_consent_action", action=action)
            return {"msg": "unknown_action"}
            
    except json.JSONDecodeError as error:
        logger.error(
            "consent_action_json_decode_error",
            action_value=action_value,
            error=str(error),
        )
        return {"msg": "json_decode_error"}
    except Exception as error:
        logger.error(
            "consent_action_handler_error",
            error=str(error),
            exc_info=True,
        )
        return {"msg": "handler_error"}


__all__ = ["handle_consent_card_action"]
