"""ObjectiveService: deep-module facade for Objective commands."""

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agiwo.agent import Agent, RunOutput, TerminationReason
from agiwo.llm.base import Model
from agiwo.agent.models.finalization import (
    RunFinalizationResult,
    mechanical_agent_handoff_result,
)
from agiwo.objective.active_time import (
    check_active_time,
    default_clock,
    facts_close_active_window,
    facts_ensure_active_window,
    facts_resume_active_window,
)
from agiwo.objective.budget import (
    check_and_plan_consumption,
    usage_facts_for_consumption,
    validate_limit_not_below_used,
)
from agiwo.objective.drain import drain_facts_when_idle, resume_status_facts
from agiwo.objective.dispatch import OutboxDispatcher
from agiwo.objective.errors import (
    BudgetBoundaryHit,
    CommandUnavailable,
    IdempotencyConflict,
    InvariantViolation,
    ValidationError,
)
from agiwo.objective.finalization import (
    map_finalization_to_outcome,
    next_run_role_for_decision,
    should_deliver,
)
from agiwo.objective.complexity import (
    COMPLEXITY_PLANNING_THRESHOLD,
    assess_entry_complexity,
    planning_notice_for_score,
)
from agiwo.objective.history import (
    append_objective_user_input_to_history,
    estimate_required_input_tokens,
)
from agiwo.objective.input import (
    build_running_input_inject_message,
    plain_user_text,
    render_run_input,
    tag_user_message_with_input_id,
)
from agiwo.objective.llm_budget import ObjectiveLlmBudgetGate
from agiwo.objective.log import (
    FactBatch,
    expand_outcome_derived_facts,
    fact_artifact_registered,
    fact_root_run_requested,
    fact_run_outcome,
    fact_budget_adjusted,
    fact_context_capacity_exceeded,
    fact_decision_accepted,
    fact_drain_completed,
    fact_entry_complexity_assessed,
    fact_objective_created,
    fact_objective_delivered,
    fact_objective_status_changed,
    fact_objective_user_input,
    fact_root_run_paused,
    fact_user_input_externalized,
)
from agiwo.objective.plan_latch import ensure_verification_required_from_run_plan
from agiwo.agent.models.log import RunLogEntryKind
from agiwo.agent.models.run import RunStatus
from agiwo.objective.models import (
    AdjustBudgetRequest,
    Artifact,
    RunRole,
    RunOutcome,
    BudgetLimits,
    CommandReceiptStatus,
    CommandResult,
    CreateObjectiveRequest,
    ExternalizeUserInputRequest,
    HandoffDecision,
    HandoffTarget,
    ObjectiveBudget,
    ObjectiveStatus,
    ObjectiveUserInput,
    PauseObjectiveRequest,
    ResumeObjectiveRequest,
    SubmitUserInputRequest,
    is_objective_resumable,
    new_id,
    utc_now,
)
from agiwo.objective.outbox import DispatchRequested
from agiwo.objective.projection import ObjectiveView, project_objective
from agiwo.objective.store.base import (
    CommandReceipt,
    ObjectiveStore,
    SlotMutation,
    command_scope,
    create_scope,
)
from agiwo.objective.templates import (
    AssignmentTemplateSet,
    default_assignment_templates,
)
from agiwo.scheduler import Scheduler
from agiwo.utils.logging import get_logger

logger = get_logger(__name__)

AgentProvider = Callable[[str], Awaitable[Agent | None]]

# Soft physical ceiling for must-keep Objective inputs (chars/4 estimate).
# Real model context limits remain Agent-owned; this only gates Assignment
# creation when required user facts alone cannot fit.
DEFAULT_CONTEXT_LIMIT_TOKENS = 128_000


@dataclass(frozen=True, slots=True)
class _DecisionFollowOn:
    outbox_records: list[DispatchRequested]
    slot_mutation: SlotMutation | None
    status: ObjectiveStatus
    boundary_hit: BudgetBoundaryHit | None = None


@dataclass(frozen=True, slots=True)
class _WaitingUserFollowOn:
    outbox_records: list[DispatchRequested]
    status: ObjectiveStatus
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class _SubmitInputEffects:
    inject: bool = False
    deferred: bool = False
    continued: bool = False
    result_status: str = ""
    run_id: str | None = None
    outbox_records: tuple[DispatchRequested, ...] = ()


@dataclass(frozen=True, slots=True)
class _CommandEffects:
    facts: list
    result: CommandResult
    slot_mutation: SlotMutation | None = None
    outbox_records: list[DispatchRequested] | tuple = ()


def _canonical_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ObjectiveService:
    """Unique write/read facade for Objective commands and Assignment mainline."""

    def __init__(
        self,
        store: ObjectiveStore,
        *,
        artifacts_root: str | Path | None = None,
        scheduler: Scheduler | None = None,
        default_agent_provider: AgentProvider | None = None,
        complexity_model: Model | None = None,
        templates: AssignmentTemplateSet | None = None,
    ) -> None:
        self._store = store
        self._artifacts_root = Path(artifacts_root) if artifacts_root else None
        self._scheduler = scheduler
        self._default_agent_provider = default_agent_provider
        self._complexity_model = complexity_model
        self._templates = templates or default_assignment_templates()
        self._dispatcher: OutboxDispatcher | None = None
        self._clock = default_clock

    def set_clock(self, clock) -> None:
        """Inject a clock for active-time tests (production uses utc_now)."""
        self._clock = clock

    async def _run_command(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_payload: dict[str, Any],
        handler: Callable[[], Awaitable[_CommandEffects]],
    ) -> CommandResult:
        request_hash = _canonical_hash(request_payload)
        existing = await self._store.get_receipt(
            scope=scope,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            if existing.request_hash != request_hash:
                raise IdempotencyConflict(
                    scope=scope,
                    idempotency_key=idempotency_key,
                    existing_hash=existing.request_hash,
                    request_hash=request_hash,
                )
            return CommandResult.from_dict(
                {**existing.response_payload, "replayed": True}
            )

        effects = await handler()
        now = utc_now()
        receipt = CommandReceipt(
            scope=scope,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            status=CommandReceiptStatus.COMPLETED,
            response_payload=effects.result.to_dict(),
            created_at=now,
            completed_at=now,
        )
        outbox = list(effects.outbox_records) if effects.outbox_records else []
        await self._store.commit_command(
            receipt=receipt,
            facts=effects.facts,
            slot_mutation=effects.slot_mutation,
            outbox_records=outbox,
        )
        return effects.result

    async def _commit_internal(
        self,
        *,
        scope: str,
        idempotency_key: str,
        request_hash: str,
        facts: list,
        response_payload: dict[str, Any],
        slot_mutation: SlotMutation | None = None,
        outbox_records: list[DispatchRequested] | tuple = (),
    ) -> None:
        now = utc_now()
        receipt = CommandReceipt(
            scope=scope,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            status=CommandReceiptStatus.COMPLETED,
            response_payload=response_payload,
            created_at=now,
            completed_at=now,
        )
        outbox = list(outbox_records) if outbox_records else []
        await self._store.commit_command(
            receipt=receipt,
            facts=facts,
            slot_mutation=slot_mutation,
            outbox_records=outbox,
        )

    async def _require_agent(self, session_id: str) -> Agent:
        if self._default_agent_provider is None:
            raise CommandUnavailable(
                "objective_command",
                reason="agent_provider_required",
            )
        agent = await self._default_agent_provider(session_id)
        if agent is None:
            raise CommandUnavailable(
                "objective_command",
                reason="agent_provider_returned_none",
            )
        return agent

    async def _flush_run_llm_usage(
        self,
        *,
        objective_id: str,
        run_id: str,
        session_id: str | None = None,
    ) -> float:
        """Flush in-memory LLM pending for a root Run via the agent's injected gate."""
        if self._default_agent_provider is None:
            return 0.0
        sid = session_id
        if sid is None:
            view = await self.get_view(objective_id)
            if view is None:
                return 0.0
            sid = view.session_id
        agent = await self._default_agent_provider(sid)
        if agent is None:
            return 0.0
        gate = getattr(agent, "llm_budget_gate", None)
        if not isinstance(gate, ObjectiveLlmBudgetGate):
            return 0.0
        return await gate.flush_run_usage(run_id)

    async def _ensure_plan_latch(
        self,
        *,
        objective_id: str,
        run_id: str,
        session_id: str,
    ) -> bool:
        """Commit verification_required from RunLog plan facts before Decision."""
        del session_id
        if self._scheduler is None:
            return False
        entries = await self._scheduler.list_run_log_entries(
            run_id,
            kinds=[RunLogEntryKind.RUN_PLAN_UPDATED],
            limit=10_000,
        )
        return await ensure_verification_required_from_run_plan(
            self._store,
            objective_id=objective_id,
            run_id=run_id,
            plan_entries=entries,
        )

    # ── Queries ──────────────────────────────────────────────────────────────

    async def get_view(self, objective_id: str) -> ObjectiveView | None:
        facts = await self._store.list_facts(objective_id=objective_id)
        if not facts:
            return None
        return project_objective(facts, objective_id=objective_id)

    async def get_metrics(self, objective_id: str):
        """Project Objective-level metrics from the committed ObjectiveView."""
        from agiwo.objective.metrics import project_objective_metrics  # noqa: PLC0415

        view = await self.get_view(objective_id)
        if view is None:
            return None
        return project_objective_metrics(view)

    async def list_by_session(self, session_id: str) -> list[ObjectiveView]:
        ids = await self._store.list_objective_ids_for_session(session_id)
        views: list[ObjectiveView] = []
        for objective_id in ids:
            view = await self.get_view(objective_id)
            if view is not None:
                views.append(view)
        return views

    # ── Commands ─────────────────────────────────────────────────────────────

    async def create_objective(
        self,
        request: CreateObjectiveRequest,
    ) -> CommandResult:
        budget = request.budget.to_budget()
        scope = create_scope(request.session_id)
        request_payload = {
            "session_id": request.session_id,
            "user_message": request.user_message.to_dict(),
            "budget": request.budget.to_dict(),
            "in_reply_to_message_id": request.in_reply_to_message_id,
            "related_outcome_id": request.related_outcome_id,
            # Include client-supplied id only; generated ids are not part of the
            # request identity (receipt is looked up before id allocation matters).
            "objective_id": request.objective_id,
        }

        async def handler() -> _CommandEffects:
            objective_id = request.objective_id or new_id("obj_")
            input_id = new_id("inp_")
            tagged_message = tag_user_message_with_input_id(
                request.user_message, input_id
            )
            user_input = ObjectiveUserInput(
                input_id=input_id,
                message=tagged_message,
                in_reply_to_message_id=request.in_reply_to_message_id,
                related_outcome_id=request.related_outcome_id,
            )
            now = utc_now()
            batch = FactBatch(
                objective_id=objective_id,
                start_sequence=1,
                now=now,
            )
            batch.add(
                fact_objective_created(
                    session_id=request.session_id,
                    budget=budget,
                )
            )
            batch.add(fact_objective_user_input(user_input=user_input))

            outbox_records: list[DispatchRequested] = []
            run_id = None
            if self._scheduler is not None:
                agent = await self._require_agent(request.session_id)
                await append_objective_user_input_to_history(
                    agent,
                    session_id=request.session_id,
                    user_input=user_input,
                    objective_id=objective_id,
                )
                complexity_score = await assess_entry_complexity(
                    plain_user_text(tagged_message),
                    model=self._complexity_model,
                )
                if complexity_score is not None:
                    batch.add(
                        fact_entry_complexity_assessed(
                            score=complexity_score,
                            threshold=COMPLEXITY_PLANNING_THRESHOLD,
                        )
                    )
                planning_notice = planning_notice_for_score(
                    complexity_score,
                    threshold=COMPLEXITY_PLANNING_THRESHOLD,
                )
                provisional = project_objective(batch.facts, objective_id=objective_id)
                assert provisional is not None
                run_id = new_id("run_")
                rendered, _template, template_hash = render_run_input(
                    provisional,
                    role=RunRole.WORK,
                    templates=self._templates,
                    run_id=run_id,
                    thin=True,
                    planning_notice=planning_notice,
                )
                batch.add(
                    fact_root_run_requested(
                        run_id=run_id,
                        role=RunRole.WORK,
                    )
                )
                outbox_records.append(
                    DispatchRequested.create(
                        objective_id=objective_id,
                        run_id=run_id,
                        role=RunRole.WORK,
                        run_input=rendered.to_dict(),
                        template_hash=template_hash,
                        session_id=request.session_id,
                        state_id=request.session_id,
                    )
                )
            result = CommandResult(
                objective_id=objective_id,
                status=ObjectiveStatus.CREATED.value,
                replayed=False,
                payload={
                    "session_id": request.session_id,
                    "input_id": input_id,
                    "budget": budget.to_dict(),
                    "run_id": run_id,
                },
            )
            return _CommandEffects(
                facts=batch.facts,
                result=result,
                slot_mutation=SlotMutation(
                    action="acquire",
                    session_id=request.session_id,
                    objective_id=objective_id,
                    acquired_at=now,
                ),
                outbox_records=outbox_records,
            )

        result = await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )
        logger.info(
            "objective_created",
            objective_id=result.objective_id,
            session_id=request.session_id,
            input_id=result.payload.get("input_id"),
            run_id=result.payload.get("run_id"),
        )
        return result

    async def submit_user_input(
        self,
        request: SubmitUserInputRequest,
    ) -> CommandResult:
        view = await self.get_view(request.objective_id)
        if view is None:
            raise ValidationError(
                "objective not found",
                objective_id=request.objective_id,
            )
        if view.is_terminal:
            raise InvariantViolation(
                "terminal Objective cannot accept user input; create a new Objective",
                objective_id=request.objective_id,
                status=view.status.value,
            )

        scope = command_scope(request.objective_id, "submit_user_input")
        request_payload = {
            "objective_id": request.objective_id,
            "user_message": request.user_message.to_dict(),
            "in_reply_to_message_id": request.in_reply_to_message_id,
            "related_outcome_id": request.related_outcome_id,
        }

        async def handler() -> _CommandEffects:
            input_id = new_id("inp_")
            message_for_fact = (
                tag_user_message_with_input_id(request.user_message, input_id)
                if view.status is ObjectiveStatus.WAITING_USER
                else request.user_message
            )
            user_input = ObjectiveUserInput(
                input_id=input_id,
                message=message_for_fact,
                in_reply_to_message_id=request.in_reply_to_message_id,
                related_outcome_id=request.related_outcome_id,
            )
            now = utc_now()
            batch = FactBatch(
                objective_id=request.objective_id,
                start_sequence=(
                    await self._store.get_max_sequence(request.objective_id)
                )
                + 1,
                now=now,
            )
            batch.add(fact_objective_user_input(user_input=user_input))
            effects = await self._resolve_submit_user_input_effects(
                view, batch=batch, now=now
            )
            if effects.outbox_records:
                agent = await self._require_agent(view.session_id)
                await append_objective_user_input_to_history(
                    agent,
                    session_id=view.session_id,
                    user_input=user_input,
                    objective_id=view.objective_id,
                )
            result = CommandResult(
                objective_id=request.objective_id,
                status=effects.result_status,
                payload={
                    "input_id": input_id,
                    "injected": effects.inject,
                    "deferred_until_resume": effects.deferred,
                    "continued_root_run": effects.continued,
                    "run_id": effects.run_id,
                },
            )
            return _CommandEffects(
                facts=batch.facts,
                result=result,
                outbox_records=effects.outbox_records,
            )

        result = await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )
        if result.payload.get("injected"):
            assert self._scheduler is not None
            assert view.active_root_run is not None
            assert view.active_root_run.run_id
            notice = build_running_input_inject_message(
                request.user_message, input_id=result.payload["input_id"]
            )
            await self._scheduler.inject_user_message(
                view.active_root_run.run_id,
                notice,
            )
        logger.info(
            "objective_user_input_submitted",
            objective_id=request.objective_id,
            input_id=result.payload.get("input_id"),
            injected=result.payload.get("injected"),
            deferred_until_resume=result.payload.get("deferred_until_resume"),
            continued_root_run=result.payload.get("continued_root_run"),
            run_id=result.payload.get("run_id"),
        )
        return result

    async def _resolve_submit_user_input_effects(
        self,
        view: ObjectiveView,
        *,
        batch: FactBatch,
        now,
    ) -> _SubmitInputEffects:
        if view.status is ObjectiveStatus.RUNNING and view.active_root_run is not None:
            root_run_id = view.active_root_run.run_id
            if root_run_id and self._scheduler is not None:
                return _SubmitInputEffects(inject=True, result_status=view.status.value)
            if root_run_id and self._scheduler is None:
                raise CommandUnavailable(
                    "submit_user_input",
                    reason="scheduler_required_for_running_inject",
                )
            return _SubmitInputEffects(result_status=view.status.value)
        if view.status in {
            ObjectiveStatus.USER_PAUSED,
            ObjectiveStatus.BUDGET_PAUSED,
        }:
            return _SubmitInputEffects(deferred=True, result_status=view.status.value)
        if view.status is ObjectiveStatus.WAITING_USER:
            prior = await self._store.list_facts(objective_id=view.objective_id)
            follow = self._plan_waiting_user_continuation(
                view,
                batch=batch,
                all_facts_for_projection=[*prior, *batch.facts],
                now=now,
            )
            return _SubmitInputEffects(
                continued=follow.run_id is not None,
                result_status=follow.status.value,
                run_id=follow.run_id,
                outbox_records=tuple(follow.outbox_records),
            )
        return _SubmitInputEffects(result_status=view.status.value)

    async def externalize_user_input(
        self,
        request: ExternalizeUserInputRequest,
    ) -> CommandResult:
        view = await self.get_view(request.objective_id)
        if view is None:
            raise ValidationError(
                "objective not found",
                objective_id=request.objective_id,
            )
        if view.is_terminal:
            raise InvariantViolation(
                "terminal Objective cannot externalize inputs",
                objective_id=request.objective_id,
                status=view.status.value,
            )

        scope = command_scope(request.objective_id, "externalize_user_input")
        request_payload = {
            "objective_id": request.objective_id,
            "input_id": request.input_id,
            "summary": request.summary,
            "content_hash": request.content_hash,
        }

        externalized = False

        async def handler() -> _CommandEffects:
            nonlocal externalized
            user_input = next(
                (u for u in view.user_inputs if u.input_id == request.input_id),
                None,
            )
            if user_input is None:
                raise ValidationError(
                    "unknown input_id",
                    input_id=request.input_id,
                    objective_id=request.objective_id,
                )
            if any(e.input_id == request.input_id for e in view.externalized_inputs):
                existing_ext = next(
                    e
                    for e in view.externalized_inputs
                    if e.input_id == request.input_id
                )
                return _CommandEffects(
                    facts=[],
                    result=CommandResult(
                        objective_id=request.objective_id,
                        status=view.status.value,
                        payload={
                            "input_id": request.input_id,
                            "artifact_id": existing_ext.artifact_id,
                            "path": existing_ext.path,
                        },
                    ),
                )

            content = user_input.message.extract_text() or json.dumps(
                user_input.message.to_dict(), ensure_ascii=False
            )
            content_hash = (
                request.content_hash
                or hashlib.sha256(content.encode("utf-8")).hexdigest()
            )
            artifact_id = new_id("art_")
            rel_path = f"sessions/{view.session_id}/artifacts/{artifact_id}.txt"
            if self._artifacts_root is not None:
                abs_path = self._artifacts_root / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(content, encoding="utf-8")

            artifact = Artifact(
                artifact_id=artifact_id,
                path=rel_path,
                summary=request.summary,
                source_input_id=request.input_id,
                content_hash=content_hash,
            )
            now = utc_now()
            batch = FactBatch(
                objective_id=request.objective_id,
                start_sequence=(
                    await self._store.get_max_sequence(request.objective_id)
                )
                + 1,
                now=now,
            )
            batch.add(fact_artifact_registered(artifact=artifact))
            batch.add(
                fact_user_input_externalized(
                    input_id=request.input_id,
                    artifact_id=artifact_id,
                    authorized_at=now,
                )
            )
            externalized = True
            return _CommandEffects(
                facts=batch.facts,
                result=CommandResult(
                    objective_id=request.objective_id,
                    status=view.status.value,
                    payload={
                        "input_id": request.input_id,
                        "artifact_id": artifact_id,
                        "path": rel_path,
                        "content_hash": content_hash,
                    },
                ),
            )

        result = await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )
        if externalized:
            logger.info(
                "objective_user_input_externalized",
                objective_id=request.objective_id,
                input_id=request.input_id,
                artifact_id=result.payload.get("artifact_id"),
            )
        return result

    async def pause(
        self,
        request: PauseObjectiveRequest,
    ) -> CommandResult:
        """Pause: idle drain completes immediately; active Runs use PAUSED barrier."""
        view = await self.get_view(request.objective_id)
        if view is None:
            raise ValidationError(
                "objective not found",
                objective_id=request.objective_id,
            )
        if view.is_terminal:
            raise InvariantViolation(
                "terminal Objective cannot be paused",
                objective_id=request.objective_id,
            )

        scope = command_scope(request.objective_id, "pause")
        request_payload = {
            "objective_id": request.objective_id,
            "reason": request.reason,
        }
        reason = "user_archive" if request.reason == "user_archive" else "user_pause"
        barrier_run_ids = await self._barrier_run_ids(view)
        if barrier_run_ids and self._scheduler is None:
            raise CommandUnavailable(
                "pause",
                reason="scheduler_required_for_active_run_barrier",
            )

        async def handler() -> _CommandEffects:
            now = utc_now()
            batch = FactBatch(
                objective_id=request.objective_id,
                start_sequence=(
                    await self._store.get_max_sequence(request.objective_id)
                )
                + 1,
                now=now,
            )
            batch.add_many(
                drain_facts_when_idle(
                    view,
                    reason=reason,  # type: ignore[arg-type]
                    now=now,
                    source="user_pause",
                    barrier_run_ids=barrier_run_ids,
                )
            )
            return _CommandEffects(
                facts=batch.facts,
                result=CommandResult(
                    objective_id=request.objective_id,
                    status=(
                        ObjectiveStatus.DRAINING.value
                        if barrier_run_ids
                        else ObjectiveStatus.USER_PAUSED.value
                    ),
                    payload={
                        "reason": reason,
                        "barrier_run_ids": list(barrier_run_ids),
                    },
                ),
            )

        result = await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )
        if barrier_run_ids:
            assert self._scheduler is not None
            await self._scheduler.request_recoverable_pause(
                list(barrier_run_ids), reason
            )
            await self._complete_drain_barrier(
                objective_id=request.objective_id,
                next_status=ObjectiveStatus.USER_PAUSED,
                run_id=(
                    view.active_root_run.run_id
                    if view.active_root_run is not None
                    else None
                ),
            )
            result = CommandResult(
                objective_id=request.objective_id,
                status=ObjectiveStatus.USER_PAUSED.value,
                payload={"reason": reason, "barrier_run_ids": list(barrier_run_ids)},
            )
        return result

    async def resume(
        self,
        request: ResumeObjectiveRequest,
    ) -> CommandResult:
        """Two-phase resume: prepare Run barrier, then Objective RUNNING + release."""
        view = await self.get_view(request.objective_id)
        if view is None:
            raise ValidationError(
                "objective not found",
                objective_id=request.objective_id,
            )
        if not is_objective_resumable(view.status):
            raise InvariantViolation(
                "only paused Objective statuses can be resumed",
                objective_id=request.objective_id,
                status=view.status.value,
            )

        scope = command_scope(request.objective_id, "resume")
        request_payload = {
            "objective_id": request.objective_id,
            "reason": request.reason,
        }
        barrier_holder: list[tuple[str, ...]] = []

        async def handler() -> _CommandEffects:
            if view.status == ObjectiveStatus.BUDGET_PAUSED:
                self._assert_budget_resume_allowed(view)

            barrier_run_ids = await self._paused_barrier_run_ids(view)
            barrier_holder.append(barrier_run_ids)
            if barrier_run_ids:
                if self._scheduler is None:
                    raise CommandUnavailable(
                        "resume",
                        reason="scheduler_required_for_run_restore",
                    )
                await self._scheduler.prepare_resume(list(barrier_run_ids))

            now = utc_now()
            batch = FactBatch(
                objective_id=request.objective_id,
                start_sequence=(
                    await self._store.get_max_sequence(request.objective_id)
                )
                + 1,
                now=now,
            )
            batch.add_many(resume_status_facts(view, now=now))
            resumed_view = project_objective(
                list(await self._store.list_facts(objective_id=request.objective_id))
                + batch.facts,
                objective_id=request.objective_id,
            )
            if resumed_view is not None:
                batch.add_many(facts_resume_active_window(resumed_view, now=now))
            return _CommandEffects(
                facts=batch.facts,
                result=CommandResult(
                    objective_id=request.objective_id,
                    status=ObjectiveStatus.RUNNING.value,
                    payload={
                        "reason": request.reason,
                        "barrier_run_ids": list(barrier_run_ids),
                    },
                ),
            )

        result = await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )
        barrier_run_ids = barrier_holder[0] if barrier_holder else ()
        if barrier_run_ids and self._scheduler is not None:
            await self._scheduler.release_resume_barrier()
        return result

    def _assert_budget_resume_allowed(self, view: ObjectiveView) -> None:
        """BUDGET_PAUSED resume needs countable headroom; active window resets separately."""
        b = view.budget
        if (
            b.handoffs.used < b.handoffs.limit
            or b.verification_attempts.used < b.verification_attempts.limit
            or b.llm_cost_usd.used + 1e-12 < b.llm_cost_usd.limit
        ):
            return
        raise InvariantViolation(
            "budget resume requires raised countable limits "
            "(or only active_seconds was exhausted — raise a countable limit first)",
            objective_id=view.objective_id,
        )

    async def _barrier_run_ids(self, view: ObjectiveView) -> tuple[str, ...]:
        if view.active_root_run is None or not view.active_root_run.run_id:
            return ()
        root = view.active_root_run.run_id
        if self._scheduler is None:
            return (root,)
        tree = await self._scheduler.list_execution_tree(root)
        ids = [root]
        for node in tree:
            if node.run_id and node.run_id not in ids:
                ids.append(node.run_id)
        return tuple(ids)

    async def _paused_barrier_run_ids(self, view: ObjectiveView) -> tuple[str, ...]:
        """Runs that were barrier members when the Objective entered pause."""
        facts = await self._store.list_facts(objective_id=view.objective_id)
        for fact in reversed(facts):
            if fact.kind.value == "DrainStarted":
                raw = fact.payload.get("barrier_run_ids") or []
                return tuple(str(x) for x in raw if x)
        return ()

    async def _complete_drain_barrier(
        self,
        *,
        objective_id: str,
        next_status: ObjectiveStatus,
        run_id: str | None,
    ) -> None:
        if run_id is not None:
            await self._flush_run_llm_usage(
                objective_id=objective_id,
                run_id=run_id,
            )

        now = utc_now()
        batch = FactBatch(
            objective_id=objective_id,
            start_sequence=(await self._store.get_max_sequence(objective_id)) + 1,
            now=now,
        )
        batch.add(fact_drain_completed(next_status=next_status))
        if run_id is not None:
            batch.add(
                fact_root_run_paused(
                    run_id=run_id,
                    reason="drain_completed",
                )
            )
        await self._commit_internal(
            scope=command_scope(objective_id, "drain_complete"),
            idempotency_key=f"drain_complete:{objective_id}:{next_status.value}",
            request_hash=next_status.value,
            facts=batch.facts,
            response_payload={"status": next_status.value},
        )

    async def adjust_budget(
        self,
        request: AdjustBudgetRequest,
    ) -> CommandResult:
        view = await self.get_view(request.objective_id)
        if view is None:
            raise ValidationError(
                "objective not found",
                objective_id=request.objective_id,
            )
        if view.is_terminal:
            raise InvariantViolation(
                "terminal Objective cannot adjust budget",
                objective_id=request.objective_id,
            )

        scope = command_scope(request.objective_id, "adjust_budget")
        request_payload = {
            "objective_id": request.objective_id,
            "handoffs": request.handoffs,
            "verification_attempts": request.verification_attempts,
            "llm_cost_usd": request.llm_cost_usd,
            "active_seconds": request.active_seconds,
        }

        async def handler() -> _CommandEffects:
            b = view.budget
            validate_limit_not_below_used(
                b,
                handoffs=request.handoffs,
                verification_attempts=request.verification_attempts,
                llm_cost_usd=request.llm_cost_usd,
                active_seconds=request.active_seconds,
            )
            new_budget = ObjectiveBudget(
                handoffs=(
                    b.handoffs.with_limit(request.handoffs)
                    if request.handoffs is not None
                    else b.handoffs
                ),
                verification_attempts=(
                    b.verification_attempts.with_limit(request.verification_attempts)
                    if request.verification_attempts is not None
                    else b.verification_attempts
                ),
                llm_cost_usd=(
                    b.llm_cost_usd.with_limit(request.llm_cost_usd)
                    if request.llm_cost_usd is not None
                    else b.llm_cost_usd
                ),
                active_seconds=(
                    b.active_seconds.with_limit(request.active_seconds)
                    if request.active_seconds is not None
                    else b.active_seconds
                ),
            )
            now = utc_now()
            batch = FactBatch(
                objective_id=request.objective_id,
                start_sequence=(
                    await self._store.get_max_sequence(request.objective_id)
                )
                + 1,
                now=now,
            )
            batch.add(
                fact_budget_adjusted(
                    budget=new_budget,
                    reason="user_adjust",
                    previous_budget=b,
                )
            )
            return _CommandEffects(
                facts=batch.facts,
                result=CommandResult(
                    objective_id=request.objective_id,
                    status=view.status.value,
                    payload={"budget": new_budget.to_dict()},
                ),
            )

        return await self._run_command(
            scope=scope,
            idempotency_key=request.idempotency_key,
            request_payload=request_payload,
            handler=handler,
        )

    # ── Dispatcher / mainline ────────────────────────────────────────────────

    async def start_dispatcher(self) -> None:
        if self._scheduler is None or self._default_agent_provider is None:
            raise CommandUnavailable(
                "start_dispatcher",
                reason="scheduler_and_agent_provider_required",
            )
        if self._dispatcher is not None:
            return
        from agiwo.objective.recovery import reconcile_startup  # noqa: PLC0415

        await reconcile_startup(self._store, runs=self._scheduler, apply=True)
        self._dispatcher = OutboxDispatcher(
            store=self._store,
            scheduler=self._scheduler,
            agent_provider=self._default_agent_provider,
            on_run_finished=self._on_root_run_finished,
        )
        await self._dispatcher.start()

    async def reconcile_startup(self, *, apply: bool = True):
        """Scan open outbox / Run status and apply safe recovery actions."""
        from agiwo.objective.recovery import (  # noqa: PLC0415
            RecoveryReport,
            reconcile_startup,
        )

        report: RecoveryReport = await reconcile_startup(
            self._store, runs=self._scheduler, apply=apply
        )
        return report

    async def stop_dispatcher(self) -> None:
        if self._dispatcher is not None:
            await self._dispatcher.stop()
            self._dispatcher = None

    async def _on_root_run_finished(
        self,
        record: DispatchRequested,
        run_output: RunOutput,
    ) -> None:
        if run_output.paused:
            await self._flush_run_llm_usage(
                objective_id=record.objective_id,
                run_id=record.run_id,
            )
            await self._on_root_run_paused(record, run_output)
            return
        if run_output.termination_reason is TerminationReason.TIMEOUT:
            # Dispatcher wait timeout is not an Assignment terminal result.
            return
        finalization = getattr(run_output, "finalization", None)
        if finalization is None:
            # Direct runs without finalization: synthesize a mechanical handoff.
            finalization = mechanical_agent_handoff_result(
                (run_output.response or "").strip() or "(no report)",
                reason="missing_finalization_result",
                carry_forward=None,
            )
        elif (
            not (finalization.report or "").strip()
            and (run_output.response or "").strip()
        ):
            finalization.report = run_output.response or ""
        await self.apply_finalization(
            objective_id=record.objective_id,
            run_id=record.run_id,
            finalization=finalization,
            dispatch_id=record.dispatch_id,
        )

    async def _on_root_run_paused(
        self,
        record: DispatchRequested,
        run_output: RunOutput,
    ) -> None:
        """Paused runs must not produce Outcome; may complete a budget drain."""
        view = await self.get_view(record.objective_id)
        if view is None:
            return
        reason = None
        if isinstance(run_output.metadata, dict):
            reason = run_output.metadata.get("pause_reason")
        budgetish = isinstance(reason, str) and (
            "budget" in reason or "llm_cost" in reason or "active_seconds" in reason
        )
        if view.status == ObjectiveStatus.DRAINING:
            # User/budget pause already entered DRAINING; barrier completion is
            # owned by pause()/budget drain helpers.
            return
        if not budgetish:
            return
        barrier = await self._barrier_run_ids(view)
        now = utc_now()
        batch = FactBatch(
            objective_id=record.objective_id,
            start_sequence=(await self._store.get_max_sequence(record.objective_id))
            + 1,
            now=now,
        )
        batch.add_many(
            drain_facts_when_idle(
                view,
                reason="budget",
                now=now,
                source="llm_budget_denied",
                barrier_run_ids=barrier or (record.run_id,),
            )
        )
        if not batch.facts:
            return
        await self._commit_internal(
            scope=command_scope(record.objective_id, "budget_drain"),
            idempotency_key=f"budget_drain:{record.run_id}:{run_output.checkpoint_id}",
            request_hash=run_output.checkpoint_id or record.run_id,
            facts=batch.facts,
            response_payload={"status": ObjectiveStatus.DRAINING.value},
        )
        await self._complete_drain_barrier(
            objective_id=record.objective_id,
            next_status=ObjectiveStatus.BUDGET_PAUSED,
            run_id=record.run_id,
        )

    async def apply_finalization(
        self,
        *,
        objective_id: str,
        run_id: str,
        finalization: RunFinalizationResult,
        dispatch_id: str | None = None,
    ) -> CommandResult:
        view = await self.get_view(objective_id)
        if view is None:
            raise ValidationError("objective not found", objective_id=objective_id)
        active = view.active_root_run
        if active is None:
            raise ValidationError(
                "no active root run",
                objective_id=objective_id,
            )

        await self._flush_run_llm_usage(
            objective_id=objective_id,
            run_id=run_id,
            session_id=view.session_id,
        )
        await self._ensure_plan_latch(
            objective_id=objective_id,
            run_id=run_id,
            session_id=view.session_id,
        )
        view = await self.get_view(objective_id)
        if view is None:
            raise ValidationError("objective not found", objective_id=objective_id)
        active = view.active_root_run
        if active is None:
            raise ValidationError(
                "no active root run",
                objective_id=objective_id,
            )

        terminal = RunStatus.COMPLETED
        if finalization.mechanical_handoff and finalization.carry_forward:
            terminal = RunStatus.INTERRUPTED

        outcome, decision = map_finalization_to_outcome(
            view=view,
            run_id=run_id,
            role=active.role,
            result=finalization,
            terminal_status=terminal,
        )
        now = utc_now()
        batch = FactBatch(
            objective_id=objective_id,
            start_sequence=(await self._store.get_max_sequence(objective_id)) + 1,
            now=now,
        )
        batch.add(fact_run_outcome(outcome=outcome))
        batch.add_many(expand_outcome_derived_facts(outcome=outcome, occurred_at=now))
        batch.add(
            fact_decision_accepted(
                run_id=run_id,
                decision=decision,
            )
        )

        follow_on = await self._build_decision_follow_on(
            view=view,
            outcome=outcome,
            decision=decision,
            batch=batch,
            now=now,
        )
        outbox_records = follow_on.outbox_records
        slot_mutation = follow_on.slot_mutation
        status = follow_on.status

        result = CommandResult(
            objective_id=objective_id,
            status=status.value if isinstance(status, ObjectiveStatus) else str(status),
            payload={
                "run_id": run_id,
                "outcome_id": outcome.outcome_id,
                "decision": decision.to_dict(),
                "budget_boundary_hit": (
                    {
                        "code": follow_on.boundary_hit.code,
                        **follow_on.boundary_hit.details,
                    }
                    if follow_on.boundary_hit is not None
                    else None
                ),
            },
        )
        await self._commit_internal(
            scope=command_scope(objective_id, "apply_finalization"),
            idempotency_key=f"{run_id}:finalize",
            request_hash=_canonical_hash(
                {
                    "run_id": run_id,
                    "report": finalization.report,
                    "decision": finalization.decision,
                }
            ),
            facts=batch.facts,
            response_payload=result.to_dict(),
            slot_mutation=slot_mutation,
            outbox_records=outbox_records,
        )
        if dispatch_id is not None:
            # Best-effort: mark outbox completed after Outcome committed.
            try:
                await self._store.complete_dispatch(
                    dispatch_id=dispatch_id,
                    owner="objective_service",
                    status="completed",
                )
            except Exception:  # noqa: BLE001
                logger.info(
                    "objective_outbox_complete_skipped",
                    dispatch_id=dispatch_id,
                )
        if follow_on.boundary_hit is not None:
            raise follow_on.boundary_hit
        return result

    def _plan_waiting_user_continuation(
        self,
        view: ObjectiveView,
        *,
        batch: FactBatch,
        all_facts_for_projection: list,
        now,
    ) -> _WaitingUserFollowOn:
        """After WAITING_USER reply: capacity check then fresh WORK Assignment.

        Does not resume an ended Run. User replies never consume handoff quota.
        Without a scheduler, only the user-input fact is kept (same as create).
        """
        if view.active_root_run is not None:
            raise InvariantViolation(
                "WAITING_USER Objective must not have a non-terminal Assignment",
                objective_id=view.objective_id,
                run_id=view.active_root_run.run_id,
            )

        provisional = project_objective(
            all_facts_for_projection,
            objective_id=view.objective_id,
        )
        assert provisional is not None

        estimated = estimate_required_input_tokens(provisional)
        if estimated > DEFAULT_CONTEXT_LIMIT_TOKENS:
            batch.add(
                fact_context_capacity_exceeded(
                    context_limit_tokens=DEFAULT_CONTEXT_LIMIT_TOKENS,
                    estimated_input_tokens=estimated,
                    externalizable_input_ids=[
                        item.input_id for item in provisional.user_inputs
                    ],
                )
            )
            return _WaitingUserFollowOn(
                outbox_records=[],
                status=ObjectiveStatus.WAITING_USER,
            )

        if self._scheduler is None:
            return _WaitingUserFollowOn(
                outbox_records=[],
                status=ObjectiveStatus.WAITING_USER,
            )

        resume_facts = facts_resume_active_window(provisional, now=now)
        if not resume_facts:
            resume_facts = facts_ensure_active_window(provisional, now=now)
        batch.add_many(resume_facts)

        batch.add(
            fact_objective_status_changed(
                from_status=ObjectiveStatus.WAITING_USER,
                to_status=ObjectiveStatus.RUNNING,
                reason="user_reply_continue",
            )
        )

        next_run_id = new_id("run_")
        rendered, _template, template_hash = render_run_input(
            provisional,
            role=RunRole.WORK,
            templates=self._templates,
            run_id=next_run_id,
        )
        batch.add(
            fact_root_run_requested(
                run_id=next_run_id,
                role=RunRole.WORK,
            )
        )
        outbox = [
            DispatchRequested.create(
                objective_id=view.objective_id,
                run_id=next_run_id,
                role=RunRole.WORK,
                run_input=rendered.to_dict(),
                template_hash=template_hash,
                session_id=view.session_id,
                state_id=view.session_id,
            )
        ]
        return _WaitingUserFollowOn(
            outbox_records=outbox,
            status=ObjectiveStatus.RUNNING,
            run_id=next_run_id,
        )

    async def _build_decision_follow_on(
        self,
        *,
        view: ObjectiveView,
        outcome: RunOutcome,
        decision: HandoffDecision,
        batch: FactBatch,
        now,
    ) -> _DecisionFollowOn:
        outbox_records: list[DispatchRequested] = []
        status = view.status
        active = view.active_root_run
        assert active is not None

        if should_deliver(
            decision, role=active.role, verification_required=view.verification_required
        ):
            batch.add_many(facts_close_active_window(view, now=now))
            batch.add(
                fact_objective_delivered(
                    final_outcome_id=outcome.outcome_id,
                    report=outcome.report,
                    artifact_ids=[],
                )
            )
            return _DecisionFollowOn(
                outbox_records=outbox_records,
                slot_mutation=SlotMutation(
                    action="release",
                    session_id=view.session_id,
                    objective_id=view.objective_id,
                ),
                status=ObjectiveStatus.COMPLETED,
            )

        if decision.target is HandoffTarget.USER and decision.expects_reply:
            batch.add_many(
                facts_close_active_window(
                    view,
                    now=now,
                    waiting_reason="expects_user_reply",
                )
            )
            batch.add(
                fact_objective_status_changed(
                    from_status=view.status,
                    to_status=ObjectiveStatus.WAITING_USER,
                    reason="expects_user_reply",
                )
            )
            return _DecisionFollowOn(
                outbox_records=outbox_records,
                slot_mutation=None,
                status=ObjectiveStatus.WAITING_USER,
            )

        next_kind = next_run_role_for_decision(decision=decision)
        if next_kind is None:
            return _DecisionFollowOn(
                outbox_records=outbox_records,
                slot_mutation=None,
                status=status,
            )

        prior = list(await self._store.list_facts(objective_id=view.objective_id))
        provisional = project_objective(
            prior + batch.facts,
            objective_id=view.objective_id,
        )
        assert provisional is not None
        estimated = estimate_required_input_tokens(provisional)
        if estimated > DEFAULT_CONTEXT_LIMIT_TOKENS:
            batch.add(
                fact_context_capacity_exceeded(
                    context_limit_tokens=DEFAULT_CONTEXT_LIMIT_TOKENS,
                    estimated_input_tokens=estimated,
                    externalizable_input_ids=[
                        item.input_id for item in provisional.user_inputs
                    ],
                )
            )
            return _DecisionFollowOn(
                outbox_records=outbox_records,
                slot_mutation=None,
                status=ObjectiveStatus.WAITING_USER,
            )

        try:
            check_active_time(
                view,
                checked_at=self._clock(),
                pending_action="create_next_assignment",
            )
            consumption = check_and_plan_consumption(
                view.budget,
                decision,
                checked_at=now,
                pending_action="create_next_assignment",
            )
        except BudgetBoundaryHit as hit:
            provisional = project_objective(
                prior + batch.facts,
                objective_id=view.objective_id,
            )
            assert provisional is not None
            batch.add_many(
                drain_facts_when_idle(
                    provisional,
                    reason="budget",
                    now=now,
                    source="budget_boundary_hit",
                )
            )
            return _DecisionFollowOn(
                outbox_records=outbox_records,
                slot_mutation=None,
                status=ObjectiveStatus.BUDGET_PAUSED,
                boundary_hit=hit,
            )

        batch.add_many(
            usage_facts_for_consumption(
                budget_before=view.budget,
                plan=consumption,
                occurred_at=now,
            )
        )

        next_run_id = new_id("run_")
        carry_items = list(outcome.provenance.get("carry_forward_items") or [])
        rendered, _template, template_hash = render_run_input(
            provisional,
            role=next_kind,
            templates=self._templates,
            carry_forward=carry_items,
            run_id=next_run_id,
        )
        batch.add(
            fact_root_run_requested(
                run_id=next_run_id,
                role=next_kind,
            )
        )
        outbox_records.append(
            DispatchRequested.create(
                objective_id=view.objective_id,
                run_id=next_run_id,
                role=next_kind,
                run_input=rendered.to_dict(),
                template_hash=template_hash,
                session_id=view.session_id,
                state_id=view.session_id,
            )
        )
        return _DecisionFollowOn(
            outbox_records=outbox_records,
            slot_mutation=None,
            status=status,
        )


__all__ = ["ObjectiveService", "BudgetLimits"]
