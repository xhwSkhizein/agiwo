"""
Agiwo Console — FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agiwo.scheduler.engine import Scheduler
from agiwo.scheduler.models import SchedulerConfig
from agiwo.skill.manager import get_global_skill_manager

from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.channels.utils import safe_close_all
from server.channels.feishu import FeishuChannelService
from server.services.session_store import create_session_store
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.runtime import AgentRuntimeCache
from server.services.runtime_config import RuntimeConfigService
from server.services.storage_wiring import (
    build_agent_state_storage_config,
    create_run_step_storage,
    create_trace_storage,
)
from server.routers import (
    sessions,
    traces,
    overview,
    agents,
    config,
    scheduler,
    feishu,
)
from agiwo.utils.logging import get_logger

logger = get_logger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    run_step_storage = create_run_step_storage(config)
    trace_storage = create_trace_storage(config)
    agent_registry = AgentRegistry(config)
    await agent_registry.initialize()
    runtime_config_service = RuntimeConfigService(config)

    scheduler_config = SchedulerConfig(
        state_storage=build_agent_state_storage_config(config),
    )
    sched = Scheduler(scheduler_config)
    await sched.start()

    await get_global_skill_manager().initialize()

    console_session_store = create_session_store(
        db_path=config.sqlite_db_path,
        use_persistent_store=config.storage.metadata_type == "sqlite",
    )
    await console_session_store.connect()

    feishu_channel_service: FeishuChannelService | None = None
    if config.channels.feishu.enabled:
        logger.info("feishu_channel_enabled", enabled=True)
        missing: list[str] = []
        if not config.channels.feishu.app_id:
            missing.append("AGIWO_CONSOLE_FEISHU_APP_ID")
        if not config.channels.feishu.app_secret:
            missing.append("AGIWO_CONSOLE_FEISHU_APP_SECRET")
        if not config.channels.feishu.default_agent_name:
            missing.append("AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME")
        if missing:
            raise RuntimeError(
                "Feishu channel enabled but missing required config: "
                + ", ".join(missing)
            )
        base_agent = await agent_registry.get_agent_by_name(
            config.channels.feishu.default_agent_name
        )
        if base_agent is None:
            # 使用 .env 中的默认 Agent（不持久化到 DB）
            base_agent = agent_registry._build_default_agent_record()
            logger.info("using_default_agent_from_env", name=base_agent.name)

        feishu_channel_service = FeishuChannelService(
            config=config,
            scheduler=sched,
            agent_registry=agent_registry,
        )
        await feishu_channel_service.initialize()
    else:
        logger.info("feishu_channel_disabled", enabled=False)

    agent_runtime_cache = AgentRuntimeCache(
        scheduler=sched,
        agent_registry=agent_registry,
        console_config=config,
        session_store=console_session_store,
    )

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=agent_registry,
            scheduler=sched,
            feishu_channel_service=feishu_channel_service,
            session_store=console_session_store,
            agent_runtime_cache=agent_runtime_cache,
            runtime_config_service=runtime_config_service,
        ),
    )

    yield

    clear_console_runtime(app)
    closables: list[object] = []
    if feishu_channel_service is not None:
        closables.append(feishu_channel_service)
    closables.extend([agent_runtime_cache, agent_registry, run_step_storage])
    if trace_storage is not None:
        closables.append(trace_storage)
    try:
        await sched.stop()
    except Exception:  # noqa: BLE001
        logger.warning("resource_close_failed", resource="Scheduler", exc_info=True)
    await safe_close_all(*closables)


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
    app.include_router(overview.router)
    app.include_router(agents.router)
    app.include_router(config.router)
    app.include_router(scheduler.router)
    app.include_router(feishu.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "agiwo-console"}

    return app


app = create_app()
