"""
Agiwo Console — FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agiwo.agent.options import AgentOptions
from agiwo.config.settings import settings
from agiwo.scheduler.models import AgentStateStorageConfig, SchedulerConfig
from agiwo.scheduler.scheduler import Scheduler

from server.config import ConsoleConfig
from server.dependencies import (
    set_console_config,
    set_storage_manager,
    set_agent_registry,
    set_feishu_channel_service,
    set_scheduler,
)
from server.channels.feishu import FeishuChannelService
from server.services.storage_manager import StorageManager
from server.services.agent_registry import AgentRegistry, AgentConfigRecord
from server.routers import (
    sessions,
    traces,
    agents,
    chat,
    scheduler,
    scheduler_chat,
    feishu,
)

DEFAULT_AGENT_CONFIG = AgentConfigRecord(
    id="Walaha000",
    name="Walaha",
    description="",
    model_provider="deepseek",
    model_name="deepseek-reasoner",
    options={
        "max_steps": 20,
        "run_timeout": 600,
        "max_context_window_tokens": 128000,
        "max_tokens_per_run": 128000,
        "max_run_token_cost": None,
        "enable_termination_summary": True,
        "termination_summary_prompt": "",
        "enable_skill": False,
        "skills_dir": None,  # None means use {root_path}/skills
        "relevant_memory_max_token": 2048,  # max tokens for retrieved memories
        "stream_cleanup_timeout": 300.0,  # seconds),
    },
    model_params={},
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    set_console_config(config)

    storage_manager = StorageManager(config)
    set_storage_manager(storage_manager)

    agent_registry = AgentRegistry(config)
    await agent_registry.initialize()
    set_agent_registry(agent_registry)

    scheduler_config = SchedulerConfig(
        state_storage=AgentStateStorageConfig(
            storage_type="sqlite",
            config={"db_path": config.sqlite_db_path},
        ),
    )
    sched = Scheduler(scheduler_config)
    await sched.start()
    set_scheduler(sched)

    feishu_channel_service: FeishuChannelService | None = None
    if config.feishu_enabled:
        missing: list[str] = []
        if not config.feishu_app_id:
            missing.append("AGIWO_CONSOLE_FEISHU_APP_ID")
        if not config.feishu_app_secret:
            missing.append("AGIWO_CONSOLE_FEISHU_APP_SECRET")
        if not config.feishu_default_agent_name:
            missing.append("AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME")
        if missing:
            raise RuntimeError(
                "Feishu channel enabled but missing required config: "
                + ", ".join(missing)
            )
        base_agent = await agent_registry.get_agent_by_name(
            config.feishu_default_agent_name
        )
        if base_agent is None:
            base_agent = DEFAULT_AGENT_CONFIG

        feishu_channel_service = FeishuChannelService(
            config=config,
            scheduler=sched,
            agent_registry=agent_registry,
        )
        await feishu_channel_service.initialize()
        set_feishu_channel_service(feishu_channel_service)
    else:
        set_feishu_channel_service(None)

    yield

    if feishu_channel_service is not None:
        await feishu_channel_service.close()
    set_feishu_channel_service(None)
    await sched.stop()
    await agent_registry.close()
    await storage_manager.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Agiwo Console",
        description="Control plane dashboard for Agiwo Agent SDK",
        version="0.1.0",
        lifespan=lifespan,
    )

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
    app.include_router(scheduler.router)
    app.include_router(scheduler_chat.router)
    app.include_router(feishu.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "agiwo-console"}

    return app


app = create_app()
