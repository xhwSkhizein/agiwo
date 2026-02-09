"""
Agiwo Console — FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.config import ConsoleConfig
from server.dependencies import (
    set_console_config,
    set_storage_manager,
    set_agent_registry,
)
from server.services.storage_manager import StorageManager
from server.services.agent_registry import AgentRegistry
from server.routers import sessions, traces, agents, chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    set_console_config(config)

    storage_manager = StorageManager(config)
    set_storage_manager(storage_manager)

    agent_registry = AgentRegistry(config)
    await agent_registry.initialize()
    set_agent_registry(agent_registry)

    yield

    await agent_registry.close()
    await storage_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agiwo Console",
        description="Control plane dashboard for Agiwo Agent SDK",
        version="0.1.0",
        lifespan=lifespan,
    )

    config = ConsoleConfig()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(sessions.router)
    app.include_router(traces.router)
    app.include_router(agents.router)
    app.include_router(chat.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "agiwo-console"}

    return app


app = create_app()
