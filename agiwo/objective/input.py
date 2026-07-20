"""Assemble Run Input from ObjectiveView + templates."""

from agiwo.agent.models.input import ContentPart, ContentType, UserMessage
from agiwo.objective.models import ObjectiveUserInput, RunRole
from agiwo.objective.projection import ObjectiveView
from agiwo.objective.templates import (
    RunTemplateSet,
    default_run_templates,
    render_run_template,
    template_content_hash,
)


def build_template_context(
    view: ObjectiveView,
    *,
    role: RunRole,
    carry_forward: list[dict] | None = None,
) -> dict[str, str]:
    goal = view.current_goal
    current_goal = (
        f"intent={goal.intent!r}; scope={goal.scope!r}; "
        f"success_criteria={goal.success_criteria!r}; revision={goal.revision}"
        if goal is not None
        else "(none yet — derive from user inputs)"
    )
    contributions = (
        "\n".join(
            f"- [{c.contribution_id}] {c.summary or c.content[:120]}"
            for c in view.contributions
            if c.active
        )
        or "(none)"
    )
    budget = (
        f"handoffs={view.budget.handoffs.used}/{view.budget.handoffs.limit}; "
        f"verification={view.budget.verification_attempts.used}/"
        f"{view.budget.verification_attempts.limit}; "
        f"llm_cost_usd={view.budget.llm_cost_usd.used}/"
        f"{view.budget.llm_cost_usd.limit}; "
        f"active_seconds={view.budget.active_seconds.used}/"
        f"{view.budget.active_seconds.limit}"
    )
    outcomes = (
        "\n".join(
            f"- [{o.run_id}] {o.terminal_status.value}: {o.report[:200]}"
            for o in view.outcomes
        )
        or "(none)"
    )
    if carry_forward:
        outcomes += "\n\nCarry-forward plan items (re-declare in this RunPlan):\n"
        outcomes += "\n".join(
            f"- [{item.get('id')}] {item.get('status')}: {item.get('description')}"
            for item in carry_forward
        )
    return {
        "run_role": role.value,
        "current_goal": current_goal,
        "objective_contributions": contributions,
        "objective_budget": budget,
        "run_outcomes": outcomes,
    }


def render_run_input(
    view: ObjectiveView,
    *,
    role: RunRole,
    templates: RunTemplateSet | None = None,
    carry_forward: list[dict] | None = None,
    run_id: str,
    thin: bool = False,
    planning_notice: str | None = None,
) -> tuple[UserMessage, str, str]:
    """Return (system UserMessage, template_text, template_hash)."""
    if thin and role is RunRole.WORK and not view.verification_required:
        user_text = ""
        if view.user_inputs:
            parts = [
                p.text
                for p in view.user_inputs[-1].message.content
                if p.type == ContentType.TEXT and p.text
            ]
            user_text = "\n".join(parts)
        body = user_text or "(no user text)"
        if planning_notice:
            body = f"{planning_notice}\n\n{body}"
        message = UserMessage.from_system(body)
        return message, "", ""

    template_set = templates or default_run_templates()
    template = getattr(template_set, role.value)
    context = build_template_context(view, role=role, carry_forward=carry_forward)
    body = render_run_template(template, context)
    boundary = (
        f"\n\n---\nRun boundary: run_id={run_id}. "
        "The current RunPlan is empty until you call update_plan. "
        "Prior update_plan tool results belong to finished Runs only. "
        "Continue any carry_forward items by re-declaring them."
    )
    message = UserMessage.from_system(body + boundary)
    return message, template, template_content_hash(template)


def render_assignment_input(
    view: ObjectiveView,
    *,
    kind: RunRole,
    templates: RunTemplateSet | None = None,
    carry_forward: list[dict] | None = None,
    run_id: str,
    thin: bool = False,
) -> tuple[UserMessage, str, str]:
    return render_run_input(
        view,
        role=kind,
        templates=templates,
        carry_forward=carry_forward,
        run_id=run_id,
        thin=thin,
    )


def tag_user_message_with_input_id(
    message: UserMessage,
    input_id: str,
) -> UserMessage:
    """Attach objective_input_id so Session history can match ObjectiveUserInput."""
    return UserMessage(
        content=list(message.content),
        context=message.context,
        is_user_provided=message.is_user_provided,
        objective_input_id=input_id,
    )


def verify_user_inputs_in_history(
    user_inputs: tuple[ObjectiveUserInput, ...],
    *,
    history_input_ids: set[str],
    externalized_input_ids: set[str],
) -> list[str]:
    """Return missing input_ids that are not present in history or externalized."""
    missing: list[str] = []
    for item in user_inputs:
        if item.input_id in externalized_input_ids:
            continue
        if item.input_id not in history_input_ids:
            missing.append(item.input_id)
    return missing


def collect_history_input_ids(messages: list[dict]) -> set[str]:
    found: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        oid = message.get("objective_input_id")
        if isinstance(oid, str) and oid:
            found.add(oid)
        if message.get("__type") == "user_message" and message.get(
            "objective_input_id"
        ):
            found.add(str(message["objective_input_id"]))
    return found


def plain_user_text(message: UserMessage) -> str:
    parts = [
        part.text
        for part in message.content
        if part.type == ContentType.TEXT and part.text
    ]
    return "\n".join(parts)


def build_running_input_inject_message(
    user_message: UserMessage,
    *,
    input_id: str,
) -> UserMessage:
    """System-notice false-user message for mid-run Objective user input."""
    user_text = plain_user_text(user_message)
    body = (
        '<system-notice origin="running_user_input">\n'
        "The user sent a new instruction while this root Run is still running. "
        "Check consistency with the Objective goal and update the RunPlan if needed.\n"
        f"</system-notice>\n\n{user_text}"
    )
    return UserMessage.from_system(
        [ContentPart(type=ContentType.TEXT, text=body)],
        objective_input_id=input_id,
    )


__all__ = [
    "build_running_input_inject_message",
    "build_template_context",
    "collect_history_input_ids",
    "plain_user_text",
    "render_assignment_input",
    "render_run_input",
    "tag_user_message_with_input_id",
    "verify_user_inputs_in_history",
]
