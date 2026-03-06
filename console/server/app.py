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
from agiwo.utils.logging import get_logger

logger = get_logger("app")


def _build_default_agent_config(config: ConsoleConfig) -> AgentConfigRecord:
    """Build default agent config from ConsoleConfig."""
    default_options = {
        "config_root": "",
        "max_steps": 50,
        "run_timeout": 600,
        "max_context_window_tokens": 200000,
        "max_tokens_per_run": 200000,
        "max_run_token_cost": None,
        "enable_termination_summary": True,
        "termination_summary_prompt": "",
        "enable_skill": settings.is_skills_enabled,
        "skills_dirs": None,
        "relevant_memory_max_token": 2048,
        "stream_cleanup_timeout": 300.0,
        "compact_prompt": "",
    }

    return AgentConfigRecord(
        id=config.default_agent_id,
        name=config.default_agent_name,
        description=config.default_agent_description,
        model_provider=config.default_agent_model_provider,
        model_name=config.default_agent_model_name,
        system_prompt=config.default_agent_system_prompt,
        options=default_options,
        model_params=config.default_agent_model_params,
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

    if config.scheduler_state_storage_type == "sqlite":
        scheduler_state_storage = AgentStateStorageConfig(
            storage_type="sqlite",
            config={"db_path": config.sqlite_db_path},
        )
    else:
        scheduler_state_storage = AgentStateStorageConfig(storage_type="memory")

    scheduler_config = SchedulerConfig(
        state_storage=scheduler_state_storage,
    )
    sched = Scheduler(scheduler_config)
    await sched.start()
    # Keep scheduler APIs and storage-manager-based APIs on the same state store.
    await storage_manager.agent_state_storage.close()
    storage_manager.agent_state_storage = sched.store
    set_scheduler(sched)

    feishu_channel_service: FeishuChannelService | None = None
    if config.feishu_enabled:
        logger.info("feishu_channel_enabled", enabled=True)
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
            base_agent = _build_default_agent_config(config)
            await agent_registry.create_agent(base_agent)
            logger.info(
                "create Default Agent Config", name=config.feishu_default_agent_name
            )

        feishu_channel_service = FeishuChannelService(
            config=config,
            scheduler=sched,
            agent_registry=agent_registry,
        )
        await feishu_channel_service.initialize()
        set_feishu_channel_service(feishu_channel_service)
    else:
        logger.info("feishu_channel_disabled", enabled=False)
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
