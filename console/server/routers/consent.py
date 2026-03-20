"""Tool consent approval/denial API endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from agiwo.tool.authz import ConsentDecision
from agiwo.utils.logging import get_logger

from server.services.consent_manager import get_consent_manager

logger = get_logger(__name__)

router = APIRouter(prefix="/api/consent", tags=["consent"])


class ConsentRequest(BaseModel):
    tool_call_id: str
    decision: str  # "allow" | "deny"
    patterns: list[str] | None = None


@router.post("/resolve")
async def resolve_consent(request: ConsentRequest) -> dict[str, str]:
    """Resolve a pending tool consent request."""
    if request.decision not in ("allow", "deny"):
        raise HTTPException(status_code=400, detail="Invalid decision")

    consent_manager = get_consent_manager()
    decision = ConsentDecision(
        decision=request.decision,
        patterns=request.patterns or [],
    )

    await consent_manager.waiter.resolve(request.tool_call_id, decision)

    logger.info(
        "consent_resolved",
        tool_call_id=request.tool_call_id,
        decision=request.decision,
    )

    return {"status": "ok", "tool_call_id": request.tool_call_id}


__all__ = ["router"]
