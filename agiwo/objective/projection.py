"""Pure ObjectiveLog projector: facts -> ObjectiveView (no I/O)."""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Iterable

from agiwo.agent.models.run import RunStatus
from agiwo.objective.budget import budget_agent_projection
from agiwo.objective.errors import ProjectionError
from agiwo.objective.log import ObjectiveFactKind, ObjectiveLogEntry
from agiwo.objective.models import (
    ARTIFACT_INLINE_CONTENT_MAX_BYTES,  # noqa: F401 — re-export stability
    Artifact,
    ContributionAnnotation,
    CurrentGoalAnalysis,
    HandoffDecision,
    OBJECTIVE_TERMINAL,
    ObjectiveBudget,
    ObjectiveContribution,
    ObjectiveStatus,
    ObjectiveUserInput,
    RunOutcome,
    RunRole,
    assert_single_active_root_run,
    validate_objective_transition,
)
from agiwo.utils.serialization import parse_optional_datetime_or_default


@dataclass(frozen=True, slots=True)
class ExternalizedInputView:
    input_id: str
    artifact_id: str
    path: str
    summary: str
    authorized_at: datetime


@dataclass(frozen=True, slots=True)
class RootRunView:
    run_id: str
    role: RunRole
    status: RunStatus | None
    outcome: RunOutcome | None = None
    decision: HandoffDecision | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def is_active(self) -> bool:
        return self.outcome is None


@dataclass(frozen=True, slots=True)
class TimelineNode:
    sequence: int
    fact_id: str
    kind: ObjectiveFactKind
    occurred_at: datetime
    summary: str
    refs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BudgetView:
    budget: ObjectiveBudget

    def for_agent(self) -> dict[str, Any]:
        return budget_agent_projection(self.budget)


@dataclass(frozen=True, slots=True)
class ObjectiveView:
    objective_id: str
    session_id: str
    status: ObjectiveStatus
    budget: ObjectiveBudget
    verification_required: bool = False
    entry_complexity_score: int | None = None
    user_inputs: tuple[ObjectiveUserInput, ...] = ()
    externalized_inputs: tuple[ExternalizedInputView, ...] = ()
    contributions: tuple[ObjectiveContribution, ...] = ()
    current_goal: CurrentGoalAnalysis | None = None
    goal_revisions: tuple[CurrentGoalAnalysis, ...] = ()
    root_runs: tuple[RootRunView, ...] = ()
    artifacts: tuple[Artifact, ...] = ()
    outcomes: tuple[RunOutcome, ...] = ()
    timeline: tuple[TimelineNode, ...] = ()
    delivery_report: str | None = None
    delivery_outcome_id: str | None = None
    delivery_artifact_ids: tuple[str, ...] = ()
    first_started_at: datetime | None = None
    current_active_started_at: datetime | None = None
    last_sequence: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    context_capacity: dict[str, Any] | None = None
    system_faults: tuple[dict[str, Any], ...] = ()

    @property
    def active_root_run(self) -> RootRunView | None:
        for root_run in self.root_runs:
            if root_run.is_active:
                return root_run
        return None

    @property
    def is_terminal(self) -> bool:
        return self.status in OBJECTIVE_TERMINAL

    def budget_for_agent(
        self,
        *,
        checked_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Read-only limit/used/remaining projection for agent-facing templates."""
        window_used = None
        if checked_at is not None and self.current_active_started_at is not None:
            window_used = max(
                0.0, (checked_at - self.current_active_started_at).total_seconds()
            )
        return budget_agent_projection(
            self.budget,
            current_window_used_seconds=window_used,
        )


def project_objective(
    facts: Iterable[ObjectiveLogEntry],
    *,
    objective_id: str | None = None,
) -> ObjectiveView | None:
    """Project ordered facts into an ObjectiveView. Returns None if empty."""
    ordered = sorted(facts, key=lambda f: f.sequence)
    if not ordered:
        return None

    _validate_sequence(ordered)

    state = _ProjectorState()
    for fact in ordered:
        if objective_id is not None and fact.objective_id != objective_id:
            raise ProjectionError(
                "fact objective_id mismatch",
                expected=objective_id,
                actual=fact.objective_id,
                sequence=fact.sequence,
            )
        if state.objective_id is None:
            state.objective_id = fact.objective_id
        elif fact.objective_id != state.objective_id:
            raise ProjectionError(
                "mixed objective_id in fact stream",
                expected=state.objective_id,
                actual=fact.objective_id,
                sequence=fact.sequence,
            )
        if fact.fact_id in state.seen_fact_ids:
            raise ProjectionError(
                "duplicate fact_id",
                fact_id=fact.fact_id,
                sequence=fact.sequence,
            )
        state.seen_fact_ids.add(fact.fact_id)
        _apply_fact(state, fact)

    view = state.to_view()
    finalize_projection_invariants(view)
    return view


def project_timeline_page(
    facts: Iterable[ObjectiveLogEntry],
    *,
    after_sequence: int = 0,
    limit: int = 100,
) -> list[TimelineNode]:
    view = project_objective(facts)
    if view is None:
        return []
    nodes = [n for n in view.timeline if n.sequence > after_sequence]
    return list(nodes[:limit])


# ── Internal projector state ─────────────────────────────────────────────────


@dataclass
class _ProjectorState:
    objective_id: str | None = None
    session_id: str | None = None
    status: ObjectiveStatus | None = None
    budget: ObjectiveBudget | None = None
    verification_required: bool = False
    entry_complexity_score: int | None = None
    user_inputs: list[ObjectiveUserInput] = field(default_factory=list)
    externalized: dict[str, ExternalizedInputView] = field(default_factory=dict)
    contributions: dict[str, ObjectiveContribution] = field(default_factory=dict)
    goal_revisions: list[CurrentGoalAnalysis] = field(default_factory=list)
    root_runs: dict[str, RootRunView] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    outcomes: dict[str, RunOutcome] = field(default_factory=dict)
    timeline: list[TimelineNode] = field(default_factory=list)
    delivery_report: str | None = None
    delivery_outcome_id: str | None = None
    delivery_artifact_ids: list[str] = field(default_factory=list)
    first_started_at: datetime | None = None
    current_active_started_at: datetime | None = None
    last_sequence: int = 0
    created_at: datetime | None = None
    updated_at: datetime | None = None
    context_capacity: dict[str, Any] | None = None
    system_faults: list[dict[str, Any]] = field(default_factory=list)
    seen_fact_ids: set[str] = field(default_factory=set)

    def to_view(self) -> ObjectiveView:
        if (
            self.objective_id is None
            or self.session_id is None
            or self.status is None
            or self.budget is None
        ):
            raise ProjectionError("incomplete projection: missing create fact")
        active_ids = [run.run_id for run in self.root_runs.values() if run.is_active]
        assert_single_active_root_run(
            objective_id=self.objective_id,
            active_run_ids=active_ids,
        )
        return ObjectiveView(
            objective_id=self.objective_id,
            session_id=self.session_id,
            status=self.status,
            budget=self.budget,
            verification_required=self.verification_required,
            entry_complexity_score=self.entry_complexity_score,
            user_inputs=tuple(self.user_inputs),
            externalized_inputs=tuple(self.externalized.values()),
            contributions=tuple(self.contributions.values()),
            current_goal=self.goal_revisions[-1] if self.goal_revisions else None,
            goal_revisions=tuple(self.goal_revisions),
            root_runs=tuple(self.root_runs.values()),
            artifacts=tuple(self.artifacts.values()),
            outcomes=tuple(self.outcomes.values()),
            timeline=tuple(self.timeline),
            delivery_report=self.delivery_report,
            delivery_outcome_id=self.delivery_outcome_id,
            delivery_artifact_ids=tuple(self.delivery_artifact_ids),
            first_started_at=self.first_started_at,
            current_active_started_at=self.current_active_started_at,
            last_sequence=self.last_sequence,
            created_at=self.created_at,
            updated_at=self.updated_at,
            context_capacity=self.context_capacity,
            system_faults=tuple(self.system_faults),
        )


def _validate_sequence(facts: list[ObjectiveLogEntry]) -> None:
    if not facts:
        return
    expected = facts[0].sequence
    seen: set[int] = set()
    for fact in facts:
        if fact.sequence in seen:
            raise ProjectionError(
                "duplicate sequence",
                sequence=fact.sequence,
                fact_id=fact.fact_id,
            )
        seen.add(fact.sequence)
        if fact.sequence != expected:
            raise ProjectionError(
                "sequence gap or disorder",
                expected=expected,
                actual=fact.sequence,
                fact_id=fact.fact_id,
            )
        expected = fact.sequence + 1


def _timeline(
    fact: ObjectiveLogEntry,
    summary: str,
    **refs: Any,
) -> TimelineNode:
    return TimelineNode(
        sequence=fact.sequence,
        fact_id=fact.fact_id,
        kind=fact.kind,
        occurred_at=fact.occurred_at,
        summary=summary,
        refs=refs,
    )


def _require_created(state: _ProjectorState, fact: ObjectiveLogEntry) -> None:
    if state.status is None and fact.kind != ObjectiveFactKind.OBJECTIVE_CREATED:
        raise ProjectionError(
            "first fact must be ObjectiveCreated",
            kind=fact.kind.value,
            sequence=fact.sequence,
        )


def _set_status(
    state: _ProjectorState,
    to_status: ObjectiveStatus,
    *,
    fact: ObjectiveLogEntry,
) -> None:
    if state.status is None:
        state.status = to_status
        return
    validate_objective_transition(
        state.status,
        to_status,
        objective_id=state.objective_id,
    )
    state.status = to_status
    state.updated_at = fact.occurred_at


def _get_root_run(state: _ProjectorState, run_id: str) -> RootRunView:
    root_run = state.root_runs.get(run_id)
    if root_run is None:
        raise ProjectionError("unknown root run", run_id=run_id)
    return root_run


def _apply_fact(  # noqa: PLR0915
    state: _ProjectorState, fact: ObjectiveLogEntry
) -> None:
    _require_created(state, fact)
    state.last_sequence = fact.sequence
    state.updated_at = fact.occurred_at
    kind = fact.kind
    payload = fact.payload

    if kind == ObjectiveFactKind.OBJECTIVE_CREATED:
        if state.status is not None:
            raise ProjectionError("duplicate ObjectiveCreated", sequence=fact.sequence)
        state.session_id = payload["session_id"]
        state.budget = ObjectiveBudget.from_dict(payload["budget"])
        state.status = ObjectiveStatus.CREATED
        state.created_at = fact.occurred_at
        state.timeline.append(
            _timeline(fact, "Objective created", session_id=state.session_id)
        )
        return

    assert state.status is not None
    assert state.budget is not None

    if kind == ObjectiveFactKind.OBJECTIVE_STATUS_CHANGED:
        to_status = ObjectiveStatus(payload["to_status"])
        _set_status(state, to_status, fact=fact)
        state.timeline.append(
            _timeline(
                fact,
                f"Status -> {to_status.value}",
                reason=payload.get("reason"),
            )
        )
        return

    if kind == ObjectiveFactKind.OBJECTIVE_USER_INPUT:
        user_input = ObjectiveUserInput.from_dict(payload)
        if any(u.input_id == user_input.input_id for u in state.user_inputs):
            raise ProjectionError(
                "duplicate ObjectiveUserInput",
                input_id=user_input.input_id,
                sequence=fact.sequence,
            )
        state.user_inputs.append(user_input)
        state.timeline.append(
            _timeline(fact, "User input", input_id=user_input.input_id)
        )
        return

    if kind == ObjectiveFactKind.OBJECTIVE_USER_INPUT_EXTERNALIZED:
        input_id = payload["input_id"]
        artifact_id = payload["artifact_id"]
        if not any(u.input_id == input_id for u in state.user_inputs):
            raise ProjectionError(
                "externalize references unknown input_id",
                input_id=input_id,
                sequence=fact.sequence,
            )
        artifact = state.artifacts.get(artifact_id)
        if artifact is None:
            raise ProjectionError(
                "externalize references unknown artifact_id",
                artifact_id=artifact_id,
                sequence=fact.sequence,
            )
        if artifact.source_input_id and artifact.source_input_id != input_id:
            raise ProjectionError(
                "artifact source_input_id mismatch",
                input_id=input_id,
                artifact_id=artifact_id,
            )
        authorized_raw = payload.get("authorized_at")
        authorized_at = parse_optional_datetime_or_default(
            authorized_raw, fact.occurred_at
        )
        state.externalized[input_id] = ExternalizedInputView(
            input_id=input_id,
            artifact_id=artifact_id,
            path=artifact.path,
            summary=artifact.summary,
            authorized_at=authorized_at,
        )
        state.timeline.append(
            _timeline(
                fact,
                "User input externalized",
                input_id=input_id,
                artifact_id=artifact_id,
            )
        )
        return

    if kind == ObjectiveFactKind.CONTEXT_CAPACITY_EXCEEDED:
        state.context_capacity = dict(payload)
        if state.status not in OBJECTIVE_TERMINAL:
            _set_status(state, ObjectiveStatus.WAITING_USER, fact=fact)
        state.timeline.append(_timeline(fact, "Context capacity exceeded"))
        return

    if kind == ObjectiveFactKind.CONTRIBUTION_CREATED:
        contribution = ObjectiveContribution.from_dict(payload)
        if contribution.contribution_id in state.contributions:
            raise ProjectionError(
                "duplicate contribution",
                contribution_id=contribution.contribution_id,
            )
        state.contributions[contribution.contribution_id] = contribution
        state.timeline.append(
            _timeline(
                fact,
                "Contribution created",
                contribution_id=contribution.contribution_id,
            )
        )
        return

    if kind == ObjectiveFactKind.CONTRIBUTION_ANNOTATED:
        annotation = ContributionAnnotation.from_dict(payload)
        existing = state.contributions.get(annotation.contribution_id)
        if existing is None:
            raise ProjectionError(
                "annotation references unknown contribution",
                contribution_id=annotation.contribution_id,
            )
        new_annotations = existing.annotations + (annotation,)
        active = existing.active and not annotation.deactivate
        if annotation.deactivate:
            active = False
        state.contributions[annotation.contribution_id] = replace(
            existing,
            annotations=new_annotations,
            active=active,
        )
        state.timeline.append(
            _timeline(
                fact,
                "Contribution annotated",
                contribution_id=annotation.contribution_id,
            )
        )
        return

    if kind == ObjectiveFactKind.CURRENT_GOAL_REVISED:
        analysis = CurrentGoalAnalysis.from_dict(payload)
        if state.goal_revisions:
            prev = state.goal_revisions[-1]
            if analysis.revision != prev.revision + 1:
                raise ProjectionError(
                    "current_goal revision must be strictly incremental",
                    expected=prev.revision + 1,
                    actual=analysis.revision,
                )
        elif analysis.revision != 1 and analysis.revision != 0:
            if analysis.revision < 0:
                raise ProjectionError(
                    "invalid initial goal revision",
                    revision=analysis.revision,
                )
        state.goal_revisions.append(analysis)
        state.timeline.append(_timeline(fact, f"Goal revised r{analysis.revision}"))
        return

    if kind == ObjectiveFactKind.ROOT_RUN_REQUESTED:
        run_id = payload["run_id"]
        if run_id in state.root_runs:
            raise ProjectionError(
                "duplicate RootRunRequested",
                run_id=run_id,
            )
        active = [r.run_id for r in state.root_runs.values() if r.is_active]
        if active:
            raise ProjectionError(
                "cannot request root run while another is active",
                active=active,
                new=run_id,
            )
        view = RootRunView(
            run_id=run_id,
            role=RunRole(payload["role"]),
            status=None,
            created_at=fact.occurred_at,
            updated_at=fact.occurred_at,
        )
        state.root_runs[run_id] = view
        state.timeline.append(
            _timeline(fact, "Root run requested", run_id=run_id, role=view.role.value)
        )
        return

    if kind == ObjectiveFactKind.ROOT_RUN_STARTED:
        run_id = payload["run_id"]
        root_run = _get_root_run(state, run_id)
        if root_run.outcome is not None:
            raise ProjectionError(
                "cannot start terminal root run",
                run_id=run_id,
            )
        state.root_runs[run_id] = replace(
            root_run,
            status=RunStatus.RUNNING,
            updated_at=fact.occurred_at,
        )
        if state.status == ObjectiveStatus.CREATED:
            _set_status(state, ObjectiveStatus.RUNNING, fact=fact)
        if state.first_started_at is None:
            state.first_started_at = fact.occurred_at
        state.timeline.append(_timeline(fact, "Root run started", run_id=run_id))
        return

    if kind == ObjectiveFactKind.ROOT_RUN_PAUSED:
        run_id = payload["run_id"]
        root_run = _get_root_run(state, run_id)
        if root_run.outcome is not None:
            raise ProjectionError(
                "cannot pause terminal root run",
                run_id=run_id,
            )
        state.root_runs[run_id] = replace(
            root_run,
            status=RunStatus.PAUSED,
            updated_at=fact.occurred_at,
        )
        state.timeline.append(
            _timeline(
                fact,
                "Root run paused",
                run_id=run_id,
                reason=payload.get("reason"),
            )
        )
        return

    if kind == ObjectiveFactKind.ROOT_RUN_RESUMED:
        run_id = payload["run_id"]
        root_run = _get_root_run(state, run_id)
        if root_run.status is not RunStatus.PAUSED:
            raise ProjectionError(
                "root run resume requires PAUSED status",
                run_id=run_id,
                status=None if root_run.status is None else root_run.status.value,
            )
        state.root_runs[run_id] = replace(
            root_run,
            status=RunStatus.RUNNING,
            updated_at=fact.occurred_at,
        )
        state.timeline.append(
            _timeline(
                fact,
                "Root run resumed",
                run_id=run_id,
                reason=payload.get("reason"),
            )
        )
        return

    if kind == ObjectiveFactKind.ARTIFACT_REGISTERED:
        artifact = Artifact.from_dict(payload)
        if artifact.artifact_id in state.artifacts:
            raise ProjectionError(
                "duplicate artifact",
                artifact_id=artifact.artifact_id,
            )
        state.artifacts[artifact.artifact_id] = artifact
        state.timeline.append(
            _timeline(
                fact,
                "Artifact registered",
                artifact_id=artifact.artifact_id,
            )
        )
        return

    if kind == ObjectiveFactKind.DECISION_ACCEPTED:
        run_id = payload["run_id"]
        root_run = _get_root_run(state, run_id)
        decision = HandoffDecision.from_dict(payload["decision"])
        state.root_runs[run_id] = replace(
            root_run,
            decision=decision,
            updated_at=fact.occurred_at,
        )
        state.timeline.append(
            _timeline(
                fact,
                f"Decision accepted: {decision.target.value}",
                run_id=run_id,
            )
        )
        return

    if kind == ObjectiveFactKind.RUN_OUTCOME:
        outcome = RunOutcome.from_dict(payload)
        root_run = _get_root_run(state, outcome.run_id)
        if root_run.status is RunStatus.PAUSED:
            raise ProjectionError(
                "PAUSED root run cannot accept Outcome",
                run_id=outcome.run_id,
            )
        if root_run.outcome is not None:
            raise ProjectionError(
                "root run already has Outcome",
                run_id=outcome.run_id,
            )
        if outcome.outcome_id in state.outcomes:
            raise ProjectionError(
                "duplicate outcome_id",
                outcome_id=outcome.outcome_id,
            )
        state.root_runs[outcome.run_id] = replace(
            root_run,
            status=outcome.terminal_status,
            outcome=outcome,
            decision=outcome.decision or root_run.decision,
            updated_at=fact.occurred_at,
        )
        state.outcomes[outcome.outcome_id] = outcome
        state.timeline.append(
            _timeline(
                fact,
                "Run outcome",
                run_id=outcome.run_id,
                outcome_id=outcome.outcome_id,
            )
        )
        return

    if kind == ObjectiveFactKind.VERIFICATION_REQUIRED_SET:
        state.verification_required = True
        state.timeline.append(
            _timeline(
                fact,
                "Verification required",
                run_id=payload.get("run_id"),
            )
        )
        return

    if kind == ObjectiveFactKind.ENTRY_COMPLEXITY_ASSESSED:
        score = payload.get("score")
        if isinstance(score, int):
            state.entry_complexity_score = score
        state.timeline.append(
            _timeline(
                fact,
                "Entry complexity assessed",
                score=payload.get("score"),
            )
        )
        return

    if kind == ObjectiveFactKind.BUDGET_ADJUSTED:
        state.budget = ObjectiveBudget.from_dict(payload["budget"])
        state.timeline.append(
            _timeline(fact, "Budget adjusted", reason=payload.get("reason"))
        )
        return

    if kind == ObjectiveFactKind.BUDGET_USAGE_RECORDED:
        dimension = payload["dimension"]
        used_after = float(payload["used_after"])
        budget = state.budget
        mapping = {
            "handoffs": budget.handoffs,
            "verification_attempts": budget.verification_attempts,
            "llm_cost_usd": budget.llm_cost_usd,
            "active_seconds": budget.active_seconds,
        }
        if dimension not in mapping:
            raise ProjectionError("unknown budget dimension", dimension=dimension)
        updated = mapping[dimension].with_used(used_after)
        state.budget = ObjectiveBudget(
            handoffs=(updated if dimension == "handoffs" else budget.handoffs),
            verification_attempts=(
                updated
                if dimension == "verification_attempts"
                else budget.verification_attempts
            ),
            llm_cost_usd=(
                updated if dimension == "llm_cost_usd" else budget.llm_cost_usd
            ),
            active_seconds=(
                updated if dimension == "active_seconds" else budget.active_seconds
            ),
        )
        state.timeline.append(
            _timeline(fact, f"Budget usage: {dimension}", used_after=used_after)
        )
        return

    if kind == ObjectiveFactKind.ACTIVE_WINDOW_STARTED:
        started_raw = payload.get("started_at")
        started_at = parse_optional_datetime_or_default(started_raw, fact.occurred_at)
        if state.first_started_at is None:
            state.first_started_at = started_at
        state.current_active_started_at = started_at
        state.timeline.append(_timeline(fact, "Active window started"))
        return

    if kind == ObjectiveFactKind.ACTIVE_WINDOW_ENDED:
        state.current_active_started_at = None
        state.timeline.append(_timeline(fact, "Active window ended"))
        return

    if kind == ObjectiveFactKind.WAITING_INTERVAL_STARTED:
        state.timeline.append(
            _timeline(fact, "Waiting started", reason=payload.get("reason"))
        )
        return

    if kind == ObjectiveFactKind.WAITING_INTERVAL_ENDED:
        state.timeline.append(_timeline(fact, "Waiting ended"))
        return

    if kind == ObjectiveFactKind.CHECKPOINT_RECORDED:
        state.timeline.append(
            _timeline(
                fact,
                "Checkpoint",
                run_id=payload.get("run_id"),
            )
        )
        return

    if kind == ObjectiveFactKind.DRAIN_STARTED:
        if state.status not in OBJECTIVE_TERMINAL:
            _set_status(state, ObjectiveStatus.DRAINING, fact=fact)
        state.timeline.append(
            _timeline(fact, "Drain started", reason=payload.get("reason"))
        )
        return

    if kind == ObjectiveFactKind.DRAIN_COMPLETED:
        next_status = ObjectiveStatus(payload["next_status"])
        _set_status(state, next_status, fact=fact)
        state.timeline.append(
            _timeline(fact, f"Drain completed -> {next_status.value}")
        )
        return

    if kind == ObjectiveFactKind.OBJECTIVE_DELIVERED:
        final_outcome_id = payload["final_outcome_id"]
        if final_outcome_id not in state.outcomes:
            raise ProjectionError(
                "ObjectiveDelivered references unknown outcome",
                final_outcome_id=final_outcome_id,
            )
        report = payload.get("report") or ""
        if not report:
            raise ProjectionError("ObjectiveDelivered requires plain-text report")
        artifact_ids = list(payload.get("artifact_ids") or [])
        for aid in artifact_ids:
            if aid not in state.artifacts:
                raise ProjectionError(
                    "ObjectiveDelivered references unknown artifact",
                    artifact_id=aid,
                )
        state.delivery_report = report
        state.delivery_outcome_id = final_outcome_id
        state.delivery_artifact_ids = artifact_ids
        if state.status not in OBJECTIVE_TERMINAL:
            _set_status(state, ObjectiveStatus.COMPLETED, fact=fact)
        state.timeline.append(
            _timeline(
                fact,
                "Objective delivered",
                final_outcome_id=final_outcome_id,
            )
        )
        return

    if kind == ObjectiveFactKind.SYSTEM_FAULT:
        state.system_faults.append(dict(payload))
        state.timeline.append(
            _timeline(fact, "System fault", fault_code=payload.get("fault_code"))
        )
        return

    raise ProjectionError("unknown fact kind", kind=kind.value)


def finalize_projection_invariants(view: ObjectiveView) -> None:
    """Extra post-checks used by tests and service after project_objective."""
    for root_run in view.root_runs:
        if not root_run.is_active and root_run.outcome is None:
            raise ProjectionError(
                "non-active root run missing Outcome",
                run_id=root_run.run_id,
                status=None if root_run.status is None else root_run.status.value,
            )


__all__ = [
    "BudgetView",
    "ExternalizedInputView",
    "ObjectiveView",
    "RootRunView",
    "TimelineNode",
    "finalize_projection_invariants",
    "project_objective",
    "project_timeline_page",
]
