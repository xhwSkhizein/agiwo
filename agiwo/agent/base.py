from abc import ABC
from dataclasses import dataclass
import os
import datetime
from uuid import uuid4
import asyncio
from typing import AsyncIterator

from agiwo.agent.wire import Wire
from agiwo.config.settings import settings
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.schema import RunOutput, StepAdapter, StepEvent
from agiwo.agent.step_factory import StepFactory
from agiwo.agent.run_lifecycle import RunLifecycle
from agiwo.agent.side_effect_processor import SideEffectProcessor
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.session.base import SessionStore, InMemorySessionStore
from agiwo.agent.session.sqlite import SQLiteSessionStore
from agiwo.agent.session.mongo import MongoSessionStore
from agiwo.agent.prompt_template import renderer
from agiwo.agent.executor import AgentExecutor
from agiwo.skills.manager import SkillManager
from agiwo.observability.collector import TraceCollector
from agiwo.observability.store import TraceStore
from agiwo.observability.sqlite_store import SQLiteTraceStore
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentConfigOptions:
    max_steps: int = 10
    timeout_per_step: int = 120  # seconds
    parallel_tool_calls: bool = True
    max_output_tokens: int = 8196
    max_history_messages: int = 10

    # Agent Loop Configuration
    enable_termination_summary: bool = False
    termination_summary_prompt: str = ""

    enable_auto_compact: bool = False
    auto_compact_prompt: str = ""
    auto_compact_threshold: float = 0.75

    # runtime vars
    work_dir: str | None = None
    date_yyyyMMdd: str | None = None

    # skills
    skill_manager: SkillManager | None = None

    # side-effect saving
    session_store: SessionStore | None = None
    trace_collector: TraceCollector | None = None

    def __post_init__(self):
        self.work_dir = os.getcwd() if self.work_dir is None else self.work_dir
        self.date_yyyyMMdd = (
            datetime.datetime.now().strftime("%Y-%m-%d")
            if self.date_yyyyMMdd is None
            else self.date_yyyyMMdd
        )

        if self.skill_manager is None:
            if settings.is_skills_enabled:
                self.skill_manager = SkillManager(skills_dirs=settings.skills_dirs)
            else:
                self.skill_manager = None

        if self.session_store is None:
            if settings.default_session_store == "sqlite":
                self.session_store = SQLiteSessionStore()
            elif settings.default_session_store == "mongo":
                self.session_store = MongoSessionStore()
            else:
                self.session_store = InMemorySessionStore()

        if self.trace_collector is None:
            if settings.default_trace_store == "mongo":
                trace_store = TraceStore(
                    mongo_uri=settings.mongo_uri,
                    db_name=settings.mongo_db_name,
                    collection_name=settings.trace_collection_name,
                    buffer_size=settings.trace_buffer_size,
                )
            elif settings.default_trace_store == "sqlite":
                trace_store = SQLiteTraceStore(
                    db_path=settings.sqlite_db_path,
                    collection_name=settings.trace_collection_name,
                    buffer_size=settings.trace_buffer_size,
                )
            else:
                trace_store = None
            self.trace_collector = (
                TraceCollector(store=trace_store) if trace_store else None
            )


class AgiwoAgent(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        model: Model,
        tools: list[BaseTool] = [],
        system_prompt: str = "",
        options: AgentConfigOptions | None = None,
    ):
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.options = options
        if options is None:
            self.options = AgentConfigOptions()

    @property
    def id(self) -> str:
        return self.name

    def get_name(self) -> str:
        return self.name

    def get_description(self) -> str:
        return self.description

    async def run(
        self,
        user_input: str,
        *,
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        """Execute Agent with Run lifecycle management."""

        side_effect_processor = SideEffectProcessor(
            context, self.options.session_store, self.options.trace_collector
        )
        lifecycle = RunLifecycle(self.name, user_input, context, side_effect_processor)
        async with lifecycle:
            result = await self._execute_core(
                user_input, context, side_effect_processor, abort_signal
            )
            lifecycle.set_output(result)

            return result

    async def _execute_core(
        self,
        user_input: str,
        context: ExecutionContext,
        side_effect_processor: SideEffectProcessor,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        session_id = context.session_id

        # 1) Handle Sequence & User Step
        seq = await side_effect_processor.allocate_sequence()

        step_factory = StepFactory(context)
        user_step = step_factory.user_step(
            sequence=seq,
            content=user_input,
            agent_id=self.name,
        )
        await side_effect_processor.commit_step(user_step)

        # 2) Render Prompt
        rendered_prompt = self._build_system_prompt()

        # 3) Build LLM messages
        if self.options.session_store:
            messages = await self.build_context_from_steps(
                session_id=session_id,
                system_prompt=rendered_prompt,
                agent_id=self.name,
            )
        else:
            messages = (
                [{"role": "system", "content": rendered_prompt}]
                if rendered_prompt
                else []
            )

        # 4) Execute
        executor = AgentExecutor(
            model=self.model,
            tools=self.tools,
            side_effect_processor=side_effect_processor,
            options=self.options,
        )
        return await executor.execute(
            messages=messages, context=context, abort_signal=abort_signal
        )

    def _build_system_prompt(self) -> str:
        prompt_context = {
            "work_dir": self.options.work_dir,
            "date": self.options.date_yyyyMMdd,
        }
        parts: list[str] = []
        base_prompt = renderer.render(self.system_prompt or "", **prompt_context)
        if base_prompt:
            parts.append(base_prompt)

        if self.options.skill_manager:
            skills_section = self.options.skill_manager.render_skills_section()
            if skills_section:
                parts.append(skills_section)
        return "\n\n".join(parts)

    async def build_context_from_steps(
        self,
        system_prompt: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> list[dict]:
        """
        Build LLM context from Steps using StepAdapter.

        Args:
            system_prompt: Optional system prompt to prepend
            session_id: Session ID
            run_id: Filter by run_id (optional, for isolating agent context)
            agent_id: Filter by agent_id (optional, for isolating agent steps)

        Returns:
            list[dict]: Messages in OpenAI format, ready to send to LLM
        """
        logger.debug(
            "building_context",
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )

        # 1. Query steps from session_store with optional filters
        steps = await self.options.session_store.get_steps(
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )

        logger.debug("context_steps_loaded", session_id=session_id, count=len(steps))

        # 2. Convert using StepAdapter
        messages = StepAdapter.steps_to_messages(steps)

        # 3. Optionally prepend system prompt
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        logger.debug(
            "context_built", session_id=session_id, message_count=len(messages)
        )

        return messages

    async def run_stream(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        trace_store: TraceStore | None = None,
        cleanup_timeout: float = 5.0,
    ) -> AsyncIterator[StepEvent]:
        """
        Streaming execution - creates Wire internally and yields events.

        This is the primary entry point for top-level execution.
        Wire and ExecutionContext are created internally.
        """
        run_id = str(uuid4())
        session_id = session_id or str(uuid4())
        wire = Wire()
        context = ExecutionContext(
            run_id=run_id,
            session_id=session_id,
            wire=wire,
            user_id=user_id,
            agent_id=self.name,
            metadata=metadata or {},
        )

        async def _run_task():
            try:
                await self.run(user_input, context=context)
            except Exception as e:
                logger.exception(
                    "run_stream_failed",
                    run_id=run_id,
                    agent_id=self.name,
                    error=str(e),
                    exc_info=True,
                )
                raise

        # Prepare event stream
        event_stream = wire.read()
        if trace_store:
            collector = TraceCollector(store=trace_store)
            # Use wrap_stream (Pull Mode) for now to ensure we catch all events,
            # including those from AgentExecutor which might not use Pipeline yet.
            event_stream = collector.wrap_stream(
                event_stream,
                agent_id=self.id,
                session_id=session_id,
                user_id=user_id,
                input_query=user_input,
            )

        # Start execution task
        async def _run_and_close():
            """Run task and close wire when done."""
            try:
                await _run_task()
            finally:
                # Close wire immediately after task completes
                # This signals the event stream to stop
                await wire.close()

        # Execute and stream events
        task = asyncio.create_task(_run_and_close())
        try:
            async for event in event_stream:
                # Check if task has failed while we're still waiting for events
                if task.done() and not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        raise exc
                yield event
        finally:
            # Wait for task to complete if still running
            if not task.done():
                try:
                    async with asyncio.timeout(cleanup_timeout):
                        await task
                except asyncio.TimeoutError:
                    logger.warning(
                        "run_stream_cleanup_timeout",
                        agent_id=self.id,
                        timeout=cleanup_timeout,
                    )
                    task.cancel()
            else:
                # Task is done, check for exceptions
                if not task.cancelled():
                    exc = task.exception()
                    if exc is not None:
                        raise exc
