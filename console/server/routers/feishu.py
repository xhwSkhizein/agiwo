"""
Feishu channel APIs.
"""

from typing import Any

from fastapi import APIRouter, Request

from server.channels.feishu.card_action_handler import handle_consent_card_action
from server.dependencies import ConsoleRuntimeDep

router = APIRouter(prefix="/api/channels/feishu", tags=["feishu"])


@router.get("/status")
async def feishu_status(runtime: ConsoleRuntimeDep) -> dict[str, Any]:
    service = runtime.feishu_channel_service
    if service is None:
        return {"enabled": False}
    status = service.get_status()
    return {"enabled": True, **status}


@router.post("/card-action")
async def handle_card_action(request: Request) -> dict[str, Any]:
    """Handle Feishu card action callbacks (e.g., consent button clicks)."""
    body = await request.json()
    
    # Extract action value from Feishu card action payload
    action_value = body.get("action", {}).get("value", "")
    
    if not action_value:
        return {"msg": "no_action_value"}
    
    # Check if this is a consent action
    if "consent_approve" in action_value or "consent_deny" in action_value:
        return await handle_consent_card_action(action_value)
    
    return {"msg": "unknown_action_type"}
