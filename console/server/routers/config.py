"""Runtime config inspection and update API router."""

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from server.dependencies import ConsoleRuntimeDep
from server.models.runtime_config import (
    RuntimeConfigEditablePayload,
    RuntimeConfigResponse,
)

router = APIRouter(prefix="/api/config/runtime", tags=["config"])


@router.get("", response_model=RuntimeConfigResponse)
async def get_runtime_config(runtime: ConsoleRuntimeDep) -> RuntimeConfigResponse:
    if runtime.runtime_config_service is None:
        raise HTTPException(
            status_code=503,
            detail="Runtime config service not initialized",
        )
    return await runtime.runtime_config_service.get_snapshot()


@router.put("", response_model=RuntimeConfigResponse)
async def update_runtime_config(
    body: RuntimeConfigEditablePayload,
    runtime: ConsoleRuntimeDep,
) -> RuntimeConfigResponse:
    if runtime.runtime_config_service is None:
        raise HTTPException(
            status_code=503,
            detail="Runtime config service not initialized",
        )
    try:
        return await runtime.runtime_config_service.update(body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=jsonable_encoder(exc.errors()),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
