"""
Agiwo Console — FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass

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
from server.services.agent_registry import AgentRegistry, build_default_agent_record
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


@dataclass
class LifespanResources:
    run_step_storage: object | None = None
    trace_storage: object | None = None
    agent_registry: AgentRegistry | None = None
    scheduler: Scheduler | None = None
    scheduler_started: bool = False
    console_session_store: object | None = None
    feishu_channel_service: FeishuChannelService | None = None
    agent_runtime_cache: AgentRuntimeCache | None = None


def _validate_feishu_config(config: ConsoleConfig) -> None:
    if not config.channels.feishu.enabled:
        logger.info("feishu_channel_disabled", enabled=False)
        return
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
            "Feishu channel enabled but missing required config: " + ", ".join(missing)
        )


async def _build_feishu_channel_service(
    config: ConsoleConfig,
    sched: Scheduler,
    agent_registry: AgentRegistry,
) -> FeishuChannelService | None:
    _validate_feishu_config(config)
    if not config.channels.feishu.enabled:
        return None
    base_agent = await agent_registry.get_agent_by_name(
        config.channels.feishu.default_agent_name
    )
    if base_agent is None:
        # 使用 .env 中的默认 Agent（不持久化到 DB）
        base_agent = build_default_agent_record(config.default_agent)
        logger.info("using_default_agent_from_env", name=base_agent.name)

    feishu_channel_service = FeishuChannelService(
        config=config,
        scheduler=sched,
        agent_registry=agent_registry,
    )
    await feishu_channel_service.initialize()
    return feishu_channel_service


async def _close_runtime_resources(resources: LifespanResources) -> None:
    closables: list[object] = []
    if resources.feishu_channel_service is not None:
        closables.append(resources.feishu_channel_service)
    if resources.agent_runtime_cache is not None:
        closables.append(resources.agent_runtime_cache)
    if resources.console_session_store is not None:
        closables.append(resources.console_session_store)
    if resources.agent_registry is not None:
        closables.append(resources.agent_registry)
    if resources.run_step_storage is not None:
        closables.append(resources.run_step_storage)
    if resources.trace_storage is not None:
        closables.append(resources.trace_storage)
    if resources.scheduler is not None and resources.scheduler_started:
        try:
            await resources.scheduler.stop()
        except Exception:  # noqa: BLE001
            logger.warning("resource_close_failed", resource="Scheduler", exc_info=True)
    await safe_close_all(*closables)


async def _startup_console_runtime(
    app: FastAPI,
    config: ConsoleConfig,
    runtime_config_service: RuntimeConfigService,
    resources: LifespanResources,
) -> None:
    resources.run_step_storage = create_run_step_storage(config)
    resources.trace_storage = create_trace_storage(config)
    resources.agent_registry = AgentRegistry(config)
    await resources.agent_registry.initialize()

    scheduler_config = SchedulerConfig(
        state_storage=build_agent_state_storage_config(config),
    )
    resources.scheduler = Scheduler(scheduler_config)
    await resources.scheduler.start()
    resources.scheduler_started = True

    await get_global_skill_manager().initialize()

    resources.console_session_store = create_session_store(
        db_path=config.sqlite_db_path,
        use_persistent_store=config.storage.metadata_type == "sqlite",
    )
    await resources.console_session_store.connect()

    resources.feishu_channel_service = await _build_feishu_channel_service(
        config,
        resources.scheduler,
        resources.agent_registry,
    )
    resources.agent_runtime_cache = AgentRuntimeCache(
        scheduler=resources.scheduler,
        agent_registry=resources.agent_registry,
        console_config=config,
        session_store=resources.console_session_store,
    )

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=resources.run_step_storage,
            trace_storage=resources.trace_storage,
            agent_registry=resources.agent_registry,
            scheduler=resources.scheduler,
            feishu_channel_service=resources.feishu_channel_service,
            session_store=resources.console_session_store,
            agent_runtime_cache=resources.agent_runtime_cache,
            runtime_config_service=runtime_config_service,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    runtime_config_service = RuntimeConfigService(config)
    resources = LifespanResources()

    try:
        await _startup_console_runtime(
            app,
            config,
            runtime_config_service,
            resources,
        )
    except Exception:
        await _close_runtime_resources(resources)
        raise

    try:
        yield
    finally:
        clear_console_runtime(app)
        await _close_runtime_resources(resources)


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
