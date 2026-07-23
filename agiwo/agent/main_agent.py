"""Session-scoped MainAgent executor skeleton (ADR 0049 / CONTEXT MainAgent)."""

import asyncio
import time
from collections.abc import AsyncIterator
from enum import Enum
from uuid import uuid4

from agiwo.agent.agent import Agent, AgentExecutionHandle
from agiwo.agent.hooks import HookRegistration, HookRegistry
from agiwo.agent.intent.base import SessionIntentStore
from agiwo.agent.intent.factory import create_session_intent_store
from agiwo.agent.intent.models import IntentEntry
from agiwo.agent.intent.report import summarize_run_report
from agiwo.agent.introspect.replay import build_introspect_state_from_entries
from agiwo.agent.models.execution import RunExecutionRequest, RunTreeRole
from agiwo.agent.models.input import UserInput, UserMessage
from agiwo.agent.models.plan import RunPlan
from agiwo.agent.models.run import RunOutput
from agiwo.agent.models.stream import AgentStreamItem
from agiwo.agent.queue import QueueItem, QueueItemKind
from agiwo.agent.session_history import append_session_user_message_to_history
from agiwo.agent.spec import AgentSpec
from agiwo.agent.completion_gates.context import CompletionGateContext
from agiwo.agent.storage.base import RunLogStorage
from agiwo.agent.worker_port import WorkerSchedulerPort
from agiwo.config.termination import TerminationReason
from agiwo.llm.base import Model
from agiwo.tool.base import BaseTool


class MainAgentState(str, Enum):
    """Lifecycle state for a session-bound MainAgent."""

    IDLE = "idle"
    RUNNING = "running"


class MainAgent:
    """Session-scoped live executor bound from an AgentSpec (ADR 0049).

    MainAgent stays alive across Runs within a Session. User input, gate
    feedback, and worker reports share one unified loop queue.

    SessionIntent writes (Wave C):
    - ``accept``: append full user text after RunLog user write (D1).
    - successful Run end: append summarized run report; optional last_run_plan.
    - cancel / fail / non-completed termination: no run_report append.
    """

    def __init__(
        self,
        session_id: str,
        agent_id: str,
        spec: AgentSpec,
        *,
        model: Model,
        tools: list[BaseTool] | None = None,
        hooks: HookRegistry | list[HookRegistration] | None = None,
        run_log_storage: RunLogStorage | None = None,
        session_intent_store: SessionIntentStore | None = None,
        worker_scheduler: WorkerSchedulerPort | None = None,
    ) -> None:
        self._session_id = session_id
        self._agent_id = agent_id
        self._spec = spec
        # Persistence is owned by the internal Agent (from spec.config). An
        # external run_log_storage argument is ignored for now.
        del run_log_storage
        self._agent = Agent(
            spec.config,
            model=model,
            tools=tools,
            hooks=hooks,
            id=agent_id,
        )
        if session_intent_store is None:
            self._session_intent_store = create_session_intent_store(
                spec.config.options.storage.run_log_storage
            )
        else:
            self._session_intent_store = session_intent_store
        self._state = MainAgentState.IDLE
        self._pending: list[QueueItem] = []
        self._handle: AgentExecutionHandle | None = None
        self._completion_task: asyncio.Task[None] | None = None
        self._worker_service = None
        if worker_scheduler is not None:
            from agiwo.agent.worker import WorkerService  # noqa: PLC0415
            from agiwo.agent.worker_tools import SpawnWorkerTool  # noqa: PLC0415

            self._worker_service = WorkerService(self, worker_scheduler)
            self._agent._inject_system_tools([SpawnWorkerTool(self._worker_service)])
        self._agent.bind_completion_gate_context(self._completion_gate_context())

    def _completion_gate_context(self) -> CompletionGateContext:
        service = self._worker_service

        def _active_worker_ids() -> frozenset[str]:
            if service is None:
                return frozenset()
            return service.active_worker_ids

        return CompletionGateContext(
            active_worker_ids=_active_worker_ids,
            on_gate_feedback=self._enqueue_gate_feedback,
        )

    async def _enqueue_gate_feedback(self, feedback_text: str) -> None:
        self.enqueue(
            QueueItem(
                kind=QueueItemKind.GATE_FEEDBACK,
                text=feedback_text,
                created_at=time.time(),
            )
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def spec(self) -> AgentSpec:
        return self._spec

    @property
    def state(self) -> MainAgentState:
        return self._state

    @property
    def agent(self) -> Agent:
        """Underlying SDK Agent used for this session-bound executor."""
        return self._agent

    @property
    def run_log_storage(self) -> RunLogStorage:
        return self._agent.run_log_storage

    @property
    def session_intent_store(self) -> SessionIntentStore:
        return self._session_intent_store

    async def accept(self, user_input: UserInput) -> AgentExecutionHandle | None:
        """Accept external user input into Session history and the live loop."""
        if self._worker_service is not None:
            await self._worker_service.ensure_started()
        UserMessage.require_user_provided(user_input)
        message = UserMessage.from_value(user_input)
        # User bubbles are committed once under a synthetic run_id so Session
        # history is not tied to any single live Run id (ADR 0048).
        await append_session_user_message_to_history(
            self._agent,
            session_id=self._session_id,
            user_message=message,
            run_id=f"session-history-{self._session_id}",
        )
        await self._append_user_input_intent(message)

        if self._handle is not None and self._handle.is_active:
            await self._handle.enqueue_message(message)
            self.enqueue(
                QueueItem(
                    kind=QueueItemKind.USER_INPUT,
                    message=message,
                    created_at=time.time(),
                )
            )
            return self._handle

        self._state = MainAgentState.RUNNING
        handle = self._agent.start_prevalidated(
            None,
            session_id=self._session_id,
            execution_request=RunExecutionRequest(
                run_id=str(uuid4()),
                run_tree_role=RunTreeRole.ROOT,
            ),
        )
        self._handle = handle
        self._completion_task = asyncio.create_task(self._await_run_completion(handle))
        return handle

    async def _append_user_input_intent(self, message: UserMessage) -> None:
        await self._session_intent_store.append_entry(
            self._session_id,
            IntentEntry(
                kind="user_input",
                text=message.extract_text(),
                at=int(time.time()),
            ),
        )

    async def _await_run_completion(self, handle: AgentExecutionHandle) -> None:
        output: RunOutput | None = None
        try:
            output = await handle.wait()
        except asyncio.CancelledError:
            pass
        finally:
            if output is not None:
                await self._maybe_append_run_report(handle, output)
            if self._handle is handle:
                self._handle = None
                self._completion_task = None
                self._state = MainAgentState.IDLE
                if self._worker_service is not None:
                    await self._worker_service.sync_parent_idle()
                await self._process_pending_worker_reports()

    async def _maybe_append_run_report(
        self,
        handle: AgentExecutionHandle,
        output: RunOutput,
    ) -> None:
        if not self._should_append_run_report(output):
            return
        summary = summarize_run_report(response=output.response)
        if not summary:
            return
        last_run_plan = await self._load_run_plan_snapshot(handle.run_id)
        await self._session_intent_store.append_entry(
            self._session_id,
            IntentEntry(
                kind="run_report",
                text=summary,
                at=int(time.time()),
                run_id=handle.run_id,
            ),
            last_run_plan=last_run_plan,
        )

    @staticmethod
    def _should_append_run_report(output: RunOutput) -> bool:
        return (
            output.error is None
            and output.termination_reason is TerminationReason.COMPLETED
        )

    async def _load_run_plan_snapshot(self, run_id: str) -> RunPlan | None:
        entries = await self.run_log_storage.list_entries(
            session_id=self._session_id,
            run_id=run_id,
            agent_id=self._agent_id,
        )
        replay = build_introspect_state_from_entries(entries)
        plan = replay.plan
        if not plan.milestones and plan.revision <= 0:
            return None
        return plan

    async def deliver_worker_report(self, run_id: str, report: str) -> None:
        """Enqueue an async Worker report on the unified loop queue."""
        self.enqueue(
            QueueItem(
                kind=QueueItemKind.WORKER_REPORT,
                text=report,
                run_id=run_id,
                created_at=time.time(),
            )
        )
        await self._process_pending_worker_reports()

    async def _process_pending_worker_reports(self) -> None:
        while self.peek_pending() is not None:
            item = self.peek_pending()
            if item is None or item.kind is not QueueItemKind.WORKER_REPORT:
                return
            self.ack_pending()
            if not item.text or not item.run_id:
                continue
            message = UserMessage.from_system(
                f"<worker-report>\n{item.text}\n</worker-report>"
            )
            if self._handle is not None and self._handle.is_active:
                await self._handle.enqueue_message(message)
                continue
            if self._state is MainAgentState.RUNNING:
                continue
            self._state = MainAgentState.RUNNING
            handle = self._agent.continue_completed_run(
                run_id=item.run_id,
                session_id=self._session_id,
                user_input=message,
            )
            self._handle = handle
            self._completion_task = asyncio.create_task(
                self._await_run_completion(handle)
            )

    def enqueue(self, item: QueueItem) -> None:
        """Append one item to the unified pending loop queue."""
        self._pending.append(item)

    def peek_pending(self) -> QueueItem | None:
        """Return the oldest pending queue item without removing it."""
        if not self._pending:
            return None
        return self._pending[0]

    def ack_pending(self) -> QueueItem | None:
        """Remove and return the oldest pending queue item."""
        if not self._pending:
            return None
        return self._pending.pop(0)

    async def cancel(self, reason: str | None = None) -> None:
        """Cancel the current run, Workers, and return to idle."""
        if self._worker_service is not None:
            await self._worker_service.cancel_all_workers(reason or "Cancelled by user")
        handle = self._handle
        completion_task = self._completion_task
        if handle is not None:
            handle.cancel(reason)
        if completion_task is not None:
            await completion_task
        self._handle = None
        self._completion_task = None
        self._state = MainAgentState.IDLE

    def subscribe(self) -> AsyncIterator[AgentStreamItem]:
        """Subscribe to live stream items for the current run."""
        if self._handle is not None:
            return self._handle.stream()

        async def _empty() -> AsyncIterator[AgentStreamItem]:
            if False:
                yield  # pragma: no cover - empty async generator

        return _empty()

    async def wait_current_run(self) -> RunOutput:
        """Wait for the current run to finish."""
        if self._handle is None:
            raise RuntimeError("No current run")
        return await self._handle.wait()

    async def close(self) -> None:
        """Release resources held by the underlying Agent."""
        if self._handle is not None and self._handle.is_active:
            await self.cancel()
        await self._agent.close()
        await self._session_intent_store.close()
