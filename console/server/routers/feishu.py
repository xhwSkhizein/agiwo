"""
Feishu channel APIs.
"""

from typing import Any

from fastapi import APIRouter

from server.dependencies import get_feishu_channel_service

router = APIRouter(prefix="/api/channels/feishu", tags=["feishu"])


@router.get("/status")
async def feishu_status() -> dict[str, Any]:
    service = get_feishu_channel_service()
    if service is None:
        return {"enabled": False}
    status = service.get_status()
    return {"enabled": True, **status}
