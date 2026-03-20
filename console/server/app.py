"""
Agiwo Console — FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agiwo.scheduler.models import SchedulerConfig
from agiwo.scheduler.scheduler import Scheduler

from server.config import ConsoleConfig
from server.dependencies import (
    ConsoleRuntime,
    bind_console_runtime,
    clear_console_runtime,
)
from server.channels.base import safe_close_all
from server.channels.feishu import FeishuChannelService
from server.services.agent_registry import AgentRegistry, AgentConfigRecord
from server.services.agent_lifecycle import build_default_agent_options
from server.services.storage_wiring import (
    build_agent_state_storage_config,
    create_run_step_storage,
    create_trace_storage,
)
from server.routers import (
    sessions,
    traces,
    agents,
    chat,
    scheduler,
    feishu,
    consent,
)
from agiwo.utils.logging import get_logger

logger = get_logger("app")


def _build_default_agent_config(config: ConsoleConfig) -> AgentConfigRecord:
    """Build default agent config from ConsoleConfig."""
    return AgentConfigRecord(
        id=config.default_agent_id,
        name=config.default_agent_name,
        description=config.default_agent_description,
        model_provider=config.default_agent_model_provider,
        model_name=config.default_agent_model_name,
        system_prompt=config.default_agent_system_prompt,
        tools=config.default_agent_tools,
        options=build_default_agent_options(),
        model_params=config.default_agent_model_params,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = ConsoleConfig()
    run_step_storage = create_run_step_storage(config)
    trace_storage = create_trace_storage(config)
    agent_registry = AgentRegistry(config)
    await agent_registry.initialize()

    scheduler_config = SchedulerConfig(
        state_storage=build_agent_state_storage_config(config),
    )
    sched = Scheduler(scheduler_config)
    await sched.start()

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
    else:
        logger.info("feishu_channel_disabled", enabled=False)

    bind_console_runtime(
        app,
        ConsoleRuntime(
            config=config,
            run_step_storage=run_step_storage,
            trace_storage=trace_storage,
            agent_registry=agent_registry,
            scheduler=sched,
            feishu_channel_service=feishu_channel_service,
        ),
    )

    yield

    clear_console_runtime(app)
    closables: list[object] = []
    if feishu_channel_service is not None:
        closables.append(feishu_channel_service)
    closables.extend([agent_registry, run_step_storage])
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
    app.include_router(agents.router)
    app.include_router(chat.router)
    app.include_router(scheduler.router)
    app.include_router(feishu.router)
    app.include_router(consent.router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok", "service": "agiwo-console"}

    return app


app = create_app()
