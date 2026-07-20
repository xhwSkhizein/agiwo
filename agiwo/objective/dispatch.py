"""Objective-private outbox dispatcher lifecycle."""

import asyncio
import time
from collections.abc import Awaitable, Callable

from agiwo.agent import (
    Agent,
    RunTreeRole,
    RunExecutionRequest,
    RunOutput,
    RunStatus,
    TerminationReason,
    UserMessage,
)
from agiwo.objective.active_time import facts_ensure_active_window
from agiwo.objective.errors import BudgetBoundaryHit, StoreError
from agiwo.objective.history import ensure_user_inputs_in_history
from agiwo.objective.llm_budget import ObjectiveLlmBudgetGate
from agiwo.objective.log import FactBatch, fact_root_run_started, fact_system_fault
from agiwo.objective.models import (
    OBJECTIVE_TERMINAL,
    CommandReceiptStatus,
    ObjectiveStatus,
    new_id,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.projection import project_objective
from agiwo.objective.store.base import CommandReceipt, ObjectiveStore, command_scope
from agiwo.scheduler import Scheduler, SchedulerExecutionRequest
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

AgentProvider = Callable[[str], Awaitable[Agent | None]]
RunFinishedHandler = Callable[[DispatchRequested, RunOutput], Awaitable[None]]

# Keep waiting well beyond a single lease tick so research Assignments can finish.
_DEFAULT_DISPATCH_WAIT_SECONDS = 900.0
_WAIT_CHUNK_SECONDS = 60.0


class OutboxDispatcher:
    """Claim DispatchRequested records and start Assignment root Runs."""

    def __init__(
        self,
        *,
        store: ObjectiveStore,
        scheduler: Scheduler,
        agent_provider: AgentProvider,
        on_run_finished: RunFinishedHandler | None = None,
        owner: str | None = None,
        poll_interval: float = 0.25,
        lease_seconds: float = 30.0,
        wait_timeout_seconds: float = _DEFAULT_DISPATCH_WAIT_SECONDS,
    ) -> None:
        self._store = store
        self._scheduler = scheduler
        self._agent_provider = agent_provider
        self._on_run_finished = on_run_finished
        self._owner = owner or new_id("disp_")
        self._poll_interval = poll_interval
        self._lease_seconds = lease_seconds
        self._wait_timeout_seconds = wait_timeout_seconds
        self._task: asyncio.Task | None = None
        self._stopping = False

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stopping = False
        self._task = asyncio.create_task(self._loop())
        logger.info("objective_dispatcher_started", owner=self._owner)

    async def stop(self) -> None:
        self._stopping = True
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("objective_dispatcher_stopped", owner=self._owner)

    async def _loop(self) -> None:
        while not self._stopping:
            try:
                claimed = await self._store.claim_dispatch(
                    owner=self._owner,
                    lease_seconds=self._lease_seconds,
                )
                if claimed is None:
                    await asyncio.sleep(self._poll_interval)
                    continue
                await self._process(claimed)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001
                logger.exception("objective_dispatcher_tick_failed", owner=self._owner)
                await asyncio.sleep(self._poll_interval)

    async def _process(self, record: DispatchRequested) -> None:  # noqa: PLR0911
        facts = await self._store.list_facts(objective_id=record.objective_id)
        view = project_objective(facts, objective_id=record.objective_id)
        if view is None or view.status in OBJECTIVE_TERMINAL:
            await self._store.complete_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                status="failed",
                last_error="objective_terminal_or_missing",
            )
            return
        if view.status in {
            ObjectiveStatus.DRAINING,
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.BUDGET_PAUSED,
            ObjectiveStatus.WAITING_USER,
        }:
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error=f"blocked_by_status:{view.status.value}",
            )
            return

        session_id = record.session_id or view.session_id
        state_id = record.state_id or session_id
        agent = await self._agent_provider(session_id)
        if agent is None:
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error="agent_provider_returned_none",
            )
            return

        if not record.run_input:
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error="missing_run_input",
            )
            return

        missing = await ensure_user_inputs_in_history(agent, view)
        if missing:
            await self._commit_history_gap_fault(record, missing)
            await self._store.complete_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                status="failed",
                last_error=f"history_gap:{','.join(missing)}",
            )
            return

        restored = UserMessage.from_storage_value(record.run_input)
        if restored is None:
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error="invalid_run_input",
            )
            return
        user_input = UserMessage.from_value(restored)

        async def _run_cost_reader(run_id: str) -> float:
            view = await self._scheduler.get_run_view(run_id)
            from agiwo.objective.llm_budget import run_cost_from_view  # noqa: PLC0415

            return run_cost_from_view(view)

        gate = ObjectiveLlmBudgetGate(
            self._store,
            objective_id=record.objective_id,
            run_cost_reader=_run_cost_reader,
        )
        await gate.seed_pending_from_run_log(record.run_id)
        agent.llm_budget_gate = gate
        try:
            await self._scheduler.dispatch_execution(
                agent,
                SchedulerExecutionRequest(
                    state_id=state_id,
                    session_id=session_id,
                    user_input=user_input,
                    execution=RunExecutionRequest(
                        run_id=record.run_id,
                        objective_id=record.objective_id,
                        run_tree_role=RunTreeRole.ROOT,
                        template_hash=record.template_hash,
                    ),
                    persistent=True,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error=str(exc),
            )
            logger.exception(
                "objective_dispatch_failed",
                dispatch_id=record.dispatch_id,
                run_id=record.run_id,
            )
            return

        run_output = await self._wait_for_run_output(
            record=record,
            state_id=state_id,
        )
        await self._finish_dispatch(record, run_output)

    async def _finish_dispatch(
        self,
        record: DispatchRequested,
        run_output: RunOutput | None,
    ) -> None:
        run_view = await self._scheduler.get_run_view(record.run_id)
        if run_view is None:
            await self._store.release_dispatch(
                dispatch_id=record.dispatch_id,
                owner=self._owner,
                last_error="run_started_missing",
            )
            return

        resolved = self._resolve_wait_output(record, run_output, run_view)
        if resolved is None:
            return

        await self._commit_execution_started(record)
        await self._store.complete_dispatch(
            dispatch_id=record.dispatch_id,
            owner=self._owner,
            status="dispatched",
        )
        if resolved.paused:
            logger.info(
                "objective_dispatch_run_paused",
                dispatch_id=record.dispatch_id,
                run_id=record.run_id,
                checkpoint_id=resolved.checkpoint_id,
            )
        if self._on_run_finished is None:
            return
        try:
            await self._on_run_finished(record, resolved)
        except BudgetBoundaryHit as hit:
            logger.info(
                "objective_budget_boundary_hit",
                dispatch_id=record.dispatch_id,
                objective_id=record.objective_id,
                dimension=hit.dimension,
                pending_action=hit.pending_action,
                used=hit.used,
                limit=hit.limit,
            )

    def _resolve_wait_output(
        self,
        record: DispatchRequested,
        run_output: RunOutput | None,
        run_view,
    ) -> RunOutput | None:
        timed_out = (
            run_output is None
            or run_output.termination_reason is TerminationReason.TIMEOUT
        )
        if timed_out:
            if run_view.status is RunStatus.COMPLETED:
                return RunOutput(
                    run_id=record.run_id,
                    session_id=run_view.session_id,
                    response=run_view.response,
                    metrics=run_view.metrics,
                    termination_reason=run_view.termination_reason
                    or TerminationReason.COMPLETED,
                    finalization=run_view.finalization,
                )
            if run_view.status is RunStatus.PAUSED:
                return RunOutput(
                    run_id=record.run_id,
                    session_id=run_view.session_id,
                    paused=True,
                    metadata={"pause_reason": "recoverable_pause"},
                )
            # Live Run still in progress: keep the claim for recovery.
            logger.warning(
                "objective_dispatch_wait_timeout",
                dispatch_id=record.dispatch_id,
                run_id=record.run_id,
                wait_timeout_seconds=self._wait_timeout_seconds,
                run_status=run_view.status.value,
            )
            return None
        assert run_output is not None
        if run_view.status is RunStatus.PAUSED and not run_output.paused:
            return RunOutput(
                run_id=record.run_id,
                session_id=run_view.session_id,
                paused=True,
                metadata={"pause_reason": "recoverable_pause"},
            )
        return run_output

    async def _wait_for_run_output(
        self,
        *,
        record: DispatchRequested,
        state_id: str,
    ) -> RunOutput | None:
        """Wait for a real terminal RunOutput, renewing the outbox lease."""
        deadline = time.monotonic() + self._wait_timeout_seconds
        last_output: RunOutput | None = None
        while not self._stopping:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return last_output
            chunk = min(_WAIT_CHUNK_SECONDS, remaining)
            try:
                await self._store.renew_dispatch_lease(
                    dispatch_id=record.dispatch_id,
                    owner=self._owner,
                    lease_seconds=max(self._lease_seconds, chunk + 15.0),
                )
            except StoreError:
                logger.warning(
                    "objective_dispatch_lease_renew_failed",
                    dispatch_id=record.dispatch_id,
                    exc_info=True,
                )
            try:
                last_output = await self._scheduler.wait_for(state_id, timeout=chunk)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "objective_dispatch_wait_failed",
                    dispatch_id=record.dispatch_id,
                    run_id=record.run_id,
                )
                return last_output
            if last_output.termination_reason is not TerminationReason.TIMEOUT:
                return last_output
        return last_output

    async def _commit_history_gap_fault(
        self,
        record: DispatchRequested,
        missing: list[str],
    ) -> None:
        seq = await self._store.get_max_sequence(record.objective_id)
        now = utc_now()
        entry = FactBatch(
            objective_id=record.objective_id,
            start_sequence=seq + 1,
            now=now,
        ).add(
            fact_system_fault(
                fault_code="objective_user_input_history_gap",
                message="ObjectiveUserInput missing from Session history",
                details={"missing_input_ids": missing},
            )
        )
        receipt = CommandReceipt(
            scope=command_scope(record.objective_id, "history_gap"),
            idempotency_key=f"history_gap:{record.dispatch_id}:{','.join(missing)}",
            request_hash=record.dispatch_id,
            status=CommandReceiptStatus.COMPLETED,
            response_payload={"missing_input_ids": missing},
            created_at=now,
            completed_at=now,
        )
        try:
            await self._store.commit_command(receipt=receipt, facts=[entry])
        except StoreError:
            logger.info(
                "objective_history_gap_fault_race",
                dispatch_id=record.dispatch_id,
            )

    async def _commit_execution_started(self, record: DispatchRequested) -> None:
        facts = await self._store.list_facts(objective_id=record.objective_id)
        # Idempotent: skip if already present.
        for fact in facts:
            if (
                fact.kind.value == "RootRunStarted"
                and fact.payload.get("run_id") == record.run_id
            ):
                return
        view = project_objective(facts, objective_id=record.objective_id)
        seq = await self._store.get_max_sequence(record.objective_id)
        now = utc_now()
        batch = FactBatch(
            objective_id=record.objective_id,
            start_sequence=seq + 1,
            now=now,
        )
        batch.add(fact_root_run_started(run_id=record.run_id))
        if view is not None:
            batch.add_many(facts_ensure_active_window(view, now=now))
        to_commit = batch.facts
        receipt = CommandReceipt(
            scope=command_scope(record.objective_id, "execution_started"),
            idempotency_key=f"{record.run_id}:started",
            request_hash=f"{record.run_id}",
            status=CommandReceiptStatus.COMPLETED,
            response_payload={"run_id": record.run_id},
            created_at=now,
            completed_at=now,
        )
        try:
            await self._store.commit_command(
                receipt=receipt,
                facts=to_commit,
            )
        except StoreError:
            # Concurrent reconciler won the race.
            logger.info(
                "objective_execution_started_race",
                run_id=record.run_id,
            )


__all__ = ["AgentProvider", "OutboxDispatcher"]
