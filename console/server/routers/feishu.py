"""Feishu channel APIs."""

from typing import Any

from fastapi import APIRouter

from server.dependencies import ConsoleRuntimeDep

router = APIRouter(prefix="/api/channels/feishu", tags=["feishu"])


@router.get("/status")
async def feishu_status(runtime: ConsoleRuntimeDep) -> dict[str, Any]:
    service = runtime.feishu_channel_service
    if service is None:
        return {"enabled": False}
    status = service.get_status()
    return {"enabled": True, **status}
