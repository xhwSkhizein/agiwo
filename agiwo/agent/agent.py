from abc import ABC
import time
import asyncio
from typing import AsyncIterator
from uuid import uuid4

from agiwo.agent.stream_channel import Wire
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool
from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.schema import (
    Run,
    RunOutput,
    RunStatus,
    StreamEvent,
    StepRecord,
    steps_to_messages,
)
from agiwo.agent.side_effect_io import SideEffectIO
from agiwo.agent.store_factory import create_session_store, create_trace_store
from agiwo.utils.abort_signal import AbortSignal
from agiwo.agent.prompt_template import renderer
from agiwo.agent.executor import AgentExecutor
from agiwo.agent.options import AgentOptions
from agiwo.agent.session.base import SessionStore
from agiwo.observability.base import BaseTraceStore
from agiwo.observability.collector import TraceCollector
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)


class AgiwoAgent(ABC):
    """
    AgiwoAgent 是 Agent 执行的主要入口点。

    它负责协调 Agent 的生命周期：
    1. 配置 (Model, Tools, Prompts)
    2. 执行上下文 (Wire, Session, Run)
    3. 事件流与可观测性 (Tracing)
    4. 核心执行循环 (委托给 AgentExecutor)
    """

    def __init__(
        self,
        id: str,
        description: str,
        model: Model,
        tools: list[BaseTool] | None = None,
        system_prompt: str = "",
        options: AgentOptions | None = None,
    ):
        self.id = id
        self.description = description
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.options = options or AgentOptions()
        self._session_store: SessionStore = create_session_store()
        self._trace_store: BaseTraceStore | None = create_trace_store()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    async def run(
        self,
        user_input: str,
        *,
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
    ) -> RunOutput:
        """
        同步执行 Agent（阻塞直到完成）。

        此方法：
        1. 在后台任务中启动执行逻辑。
        2. 消费事件流（确保副作用如 Tracing 正常工作）。
        3. 等待并返回最终的 RunOutput。
        """
        task = self._start_execution_task(
            user_input, context, abort_signal, close_wire_on_complete=True
        )
        drain_task = self._start_stream_drain_task(
            context=context, user_input=user_input
        )

        try:
            return await task
        finally:
            await self._cleanup_stream_drain_task(drain_task)

    async def run_stream(
        self,
        user_input: str,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        执行 Agent 并实时产出事件流。

        此方法：
        1. 创建新的独立 ExecutionContext。
        2. 在后台任务中启动执行。
        3. 产出事件流（如果开启 Tracing 则会自动包装）。
        4. 处理任务清理和错误传播。
        """
        context = self._create_execution_context(session_id, user_id)

        # 启动执行逻辑
        task = self._start_execution_task(
            user_input, context, close_wire_on_complete=True
        )

        # 包装流以支持可观测性
        stream = self._build_observable_stream(context, user_input)

        # 产出事件同时监控任务健康状态
        try:
            async for event in stream:
                if task.done():
                    # 如果任务提前失败，立即抛出异常
                    if exc := task.exception():
                        raise exc
                yield event
        finally:
            # 确保后台任务被正确清理
            await self._cleanup_task(task)

    # ──────────────────────────────────────────────────────────────────────────
    # Core Execution Workflow
    # ──────────────────────────────────────────────────────────────────────────

    def _start_execution_task(
        self,
        user_input: str,
        context: ExecutionContext,
        abort_signal: AbortSignal | None = None,
        close_wire_on_complete: bool = True,
    ) -> asyncio.Task[RunOutput]:
        """启动后台执行任务。"""

        async def _wrapper():
            try:
                return await self._execute_workflow(user_input, context, abort_signal)
            except Exception as e:
                # 记录导致工作流崩溃的关键错误
                logger.exception(
                    "agent_execution_crashed",
                    run_id=context.run_id,
                    agent_id=self.id,
                    error=str(e),
                )
                raise
            finally:
                if close_wire_on_complete:
                    await context.wire.close()

        return asyncio.create_task(_wrapper())

    async def _execute_workflow(
        self,
        user_input: str,
        context: ExecutionContext,
        abort_signal: AbortSignal | None,
    ) -> RunOutput:
        """
        主要的执行脚本。

        生命周期：
        1. 初始化 Run & IO
        2. 发送 START 事件
        3. 准备上下文 (Prompt + History)
        4. 执行循环 (Executor)
        5. 发送 COMPLETE/FAIL 事件
        """
        run_io = SideEffectIO(context, self._session_store)
        run = Run(
            id=context.run_id,
            agent_id=context.agent_id,
            session_id=context.session_id,
            input_query=user_input,
            status=RunStatus.RUNNING,
            parent_run_id=context.parent_run_id,
        )
        run.metrics.start_at = time.time()

        await run_io.emit_run_started(run)

        try:
            # 1. 准备执行参数
            await self._record_user_step(user_input, context, run_io)
            messages = await self._prepare_llm_messages(context)

            # 2. 运行执行器
            executor = AgentExecutor(
                model=self.model,
                tools=self.tools or [],
                run_io=run_io,
                options=self.options,
            )
            result = await executor.execute(
                messages, context, abort_signal=abort_signal
            )

            # 3. 处理成功
            self._finalize_run(run, result)
            await run_io.emit_run_completed(run, result)
            return result

        except Exception as e:
            # 4. 处理失败
            self._finalize_run(run, error=e)
            await run_io.emit_run_failed(run, e)
            raise

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: Context & Data
    # ──────────────────────────────────────────────────────────────────────────

    def _create_execution_context(
        self, session_id: str | None, user_id: str | None
    ) -> ExecutionContext:
        return ExecutionContext(
            run_id=str(uuid4()),
            session_id=session_id or str(uuid4()),
            wire=Wire(),
            user_id=user_id,
            agent_id=self.id,
            metadata={},
        )

    async def _record_user_step(
        self, user_input: str, context: ExecutionContext, run_io: SideEffectIO
    ) -> None:
        """记录用户的输入作为序列中的一步。"""
        seq = await run_io.allocate_sequence()
        user_step = StepRecord.user(
            context,
            sequence=seq,
            content=user_input,
            agent_id=self.id,
        )
        await run_io.commit_step(user_step)

    async def _prepare_llm_messages(self, context: ExecutionContext) -> list[dict]:
        """构建 LLM 消息列表 (System Prompt + Session History)。"""
        system_prompt = self._render_system_prompt()

        steps = await self._session_store.get_steps(
            session_id=context.session_id,
            agent_id=self.id,  # 隔离 agent 记忆
        )
        messages = steps_to_messages(steps)
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        return messages

    def _render_system_prompt(self) -> str:
        """渲染带有变量和技能的 system prompt。"""
        context = {
            "work_dir": self.options.work_dir,
            "date": self.options.date_yyyyMMdd,
        }

        parts = []
        if self.system_prompt:
            parts.append(renderer.render(self.system_prompt, **context))

        if self.options.skill_manager:
            if skills := self.options.skill_manager.render_skills_section():
                parts.append(skills)

        return "\n\n".join(parts)

    def _finalize_run(
        self, run: Run, result: RunOutput | None = None, error: Exception | None = None
    ) -> None:
        """完成时更新 run 的指标和状态。"""
        run.metrics.end_at = time.time()
        run.metrics.duration_ms = (run.metrics.end_at - run.metrics.start_at) * 1000

        if error:
            run.status = RunStatus.FAILED
        elif result:
            run.status = RunStatus.COMPLETED
            run.response_content = result.response
            if result.metrics:
                run.metrics.total_tokens = result.metrics.total_tokens
                run.metrics.input_tokens = result.metrics.input_tokens
                run.metrics.output_tokens = result.metrics.output_tokens
                run.metrics.tool_calls_count = result.metrics.tool_calls_count
        else:
            run.status = RunStatus.FAILED  # 正常流程不应到达此处

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers: Observability & Stream
    # ──────────────────────────────────────────────────────────────────────────

    def _should_trace(self, context: ExecutionContext) -> bool:
        """决定此次执行是否启用 Tracing。"""
        if self._trace_store is None:
            return False
        # 避免重复 Trace：只 Trace 根运行，或防止嵌套 Agent 抢夺事件
        return context.parent_run_id is None and context.depth == 0

    def _build_observable_stream(
        self, context: ExecutionContext, user_input: str
    ) -> AsyncIterator[StreamEvent]:
        """如果需要，用 TraceCollector 包装 wire 的事件流。"""
        stream = context.wire.read()

        if self._should_trace(context) and self._trace_store:
            collector = TraceCollector(store=self._trace_store)
            stream = collector.wrap_stream(
                stream,
                agent_id=self.id,
                session_id=context.session_id,
                user_id=context.user_id,
                input_query=user_input,
            )

        return stream

    async def _cleanup_task(self, task: asyncio.Task) -> None:
        """安全地清理后台任务。"""
        if not task.done():
            task.cancel()
            try:
                async with asyncio.timeout(self.options.stream_cleanup_timeout):
                    await task
            except (asyncio.TimeoutError, asyncio.CancelledError):
                logger.warning(
                    "stream_task_cancelled_timeout",
                    agent_id=self.id,
                )
            except Exception as e:
                # 记录但不重新抛出，因为我们正在清理
                logger.error("stream_task_cleanup_error", error=str(e))

        # 如果任务在正常执行期间发生了异常，则抛出
        if task.done() and not task.cancelled():
            if exc := task.exception():
                raise exc

    def _start_stream_drain_task(
        self,
        *,
        context: ExecutionContext,
        user_input: str,
    ) -> asyncio.Task[None]:
        stream = self._build_observable_stream(context, user_input)
        return asyncio.create_task(self._drain_stream(stream))

    async def _drain_stream(self, stream: AsyncIterator[StreamEvent]) -> None:
        async for _ in stream:
            pass

    async def _cleanup_stream_drain_task(self, task: asyncio.Task[None]) -> None:
        try:
            async with asyncio.timeout(self.options.stream_cleanup_timeout):
                await task
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error("stream_drain_task_cleanup_error", error=str(e))
                return
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("stream_drain_task_failed", error=str(e), exc_info=True)
