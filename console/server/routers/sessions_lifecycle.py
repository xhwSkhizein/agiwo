"""Session lifecycle control routes (cancel / fork / archive / restore)."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from agiwo.agent import MainAgentState

from server.dependencies import (
    ConsoleRuntimeDep,
    get_session_context_service,
)
from server.models.view import CancelRequest, ForkSessionRequest
from server.services.runtime.session_turn_service import SessionTurnService

router = APIRouter(prefix="/api", tags=["sessions"])
_ARCHIVE_DRAIN_TIMEOUT_SECONDS = 30.0


@router.post("/sessions/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    body: CancelRequest,
    runtime: ConsoleRuntimeDep,
):
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    if runtime.agent_runtime_cache is None:
        raise RuntimeError("Agent runtime cache not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    main_agent = await runtime.agent_runtime_cache.get_or_create_main_agent(session)
    if main_agent.state is not MainAgentState.RUNNING:
        raise HTTPException(
            status_code=404,
            detail=f"No active orchestration found for session_id={session_id}",
        )

    session_turn = SessionTurnService(session_store=runtime.session_store)
    await session_turn.cancel_if_active(main_agent, session, reason=body.reason)
    return {"ok": True, "session_id": session_id, "state_id": session_id}


@router.post("/sessions/{session_id}/fork")
async def fork_session(
    session_id: str,
    body: ForkSessionRequest,
    runtime: ConsoleRuntimeDep,
):
    result = await get_session_context_service(runtime).fork_session_by_id(
        session_id=session_id,
        context_summary=body.context_summary,
        created_by="CONSOLE_FORK",
        update_chat_context=False,
    )
    return {
        "session_id": result.session.id,
        "source_session_id": result.session.source_session_id,
    }


@router.post("/sessions/{session_id}/archive")
async def archive_session_endpoint(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, object]:
    """Archive a session: cancel any active MainAgent run, then hide from listing."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    if runtime.agent_runtime_cache is None:
        raise RuntimeError("Agent runtime cache not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    main_agent = await runtime.agent_runtime_cache.get_or_create_main_agent(session)
    session_turn = SessionTurnService(session_store=runtime.session_store)
    await session_turn.cancel_if_active(main_agent, session, reason="user_archive")
    if not await session_turn.wait_until_idle(
        main_agent, timeout_seconds=_ARCHIVE_DRAIN_TIMEOUT_SECONDS
    ):
        raise HTTPException(
            status_code=409,
            detail="archive drain incomplete; root run still active",
        )

    session.archived_at = datetime.now(timezone.utc)
    await runtime.session_store.upsert_session(session)
    return {"ok": True, "session_id": session_id, "archived_at": session.archived_at}


@router.post("/sessions/{session_id}/restore")
async def restore_session_endpoint(
    session_id: str,
    runtime: ConsoleRuntimeDep,
) -> dict[str, bool]:
    """Clear a session's archived_at flag."""
    if runtime.session_store is None:
        raise RuntimeError("Session store not available")
    session = await runtime.session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    session.archived_at = None
    await runtime.session_store.upsert_session(session)
    return {"ok": True}
