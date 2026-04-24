"""Goal-directed review.

Public API consumed by ``run_tool_batch.py``:

* ``ReviewBatch`` — per-batch lifecycle object
* ``StepBackOutcome`` — result of a step-back pass
"""

from typing import Any

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import Milestone
from agiwo.agent.models.run import RunLedger
from agiwo.agent.review.goal_manager import (
    activate_next_milestone,
    complete_active_milestone,
    declare_milestones,
    get_active_milestone,
)
from agiwo.agent.review.review_enforcer import (
    ReviewTrigger,
    check_review_trigger,
    inject_system_review,
)
from agiwo.agent.review.step_back_executor import (
    StepBackOutcome,
    execute_step_back,
)
from agiwo.tool.base import BaseTool, ToolResult


class ReviewBatch:
    """Per-batch review lifecycle object.

    Replaces the legacy retrospect system.  Caller interacts through three methods:

    * ``process_result()``  — returns final content (may inject <system-review>)
    * ``register_step()``   — registers a committed step for later lookup
    * ``finalize()``        — returns ``StepBackOutcome``; caller applies
      via targeted content updates (NO rebuild_messages)
    """

    def __init__(
        self,
        config: AgentOptions,
        ledger: RunLedger,
        tools_map: dict[str, BaseTool],
    ) -> None:
        self._config = config
        self._ledger = ledger
        self._enabled = (
            config.enable_goal_directed_review
            and "review_trajectory" in tools_map
            and "declare_milestones" in tools_map
        )
        self._feedback: str | None = None
        self._review_seq: int | None = None
        self._review_tool_call_id: str | None = None
        self._step_lookup: dict[str, dict[str, Any]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process_result(
        self,
        result: ToolResult,
        *,
        current_seq: int = 0,
    ) -> str:
        """Process a tool result. Returns the (possibly transformed) content."""
        content = result.content or ""
        if not self._enabled:
            return content

        # Capture review_trajectory feedback.
        if result.tool_name == "review_trajectory" and result.is_success:
            self._ledger.review.consecutive_errors = 0
            output = result.output if isinstance(result.output, dict) else {}
            aligned = output.get("aligned")
            if aligned is True:
                self._ledger.review.last_review_seq = current_seq
                self._ledger.review.last_checkpoint_seq = current_seq
                self._ledger.review.is_review_pending = False
                return content
            if aligned is False:
                experience = output.get("experience")
                self._feedback = experience if isinstance(experience, str) else content
                self._review_seq = current_seq
                self._review_tool_call_id = result.tool_call_id or None
                return content
            return content

        if result.tool_name == "declare_milestones" and result.is_success:
            milestones = _parse_declared_milestones(result.output)
            if milestones:
                declare_milestones(
                    self._ledger.review,
                    milestones,
                    current_seq=current_seq,
                )
            return content

        # Track errors
        is_error = not result.is_success
        if is_error:
            self._ledger.review.consecutive_errors += 1
        else:
            self._ledger.review.consecutive_errors = 0

        # Check triggers
        trigger = check_review_trigger(
            state=self._ledger.review,
            enabled=True,
            is_error=is_error and self._config.review_on_error,
            step_interval=self._config.review_step_interval,
            error_threshold=2,
            tool_name=result.tool_name,
            current_seq=current_seq,
        )

        if trigger is not ReviewTrigger.NONE:
            milestone = get_active_milestone(self._ledger.review)
            step_count = max(current_seq - self._ledger.review.last_review_seq, 0)
            content = inject_system_review(content, milestone, step_count)

        return content

    def register_step(self, tool_call_id: str, step_id: str, sequence: int) -> None:
        """Register a committed step for later step-back lookup."""
        self._step_lookup[tool_call_id] = {
            "id": step_id,
            "sequence": sequence,
        }

    async def finalize(
        self,
        *,
        storage=None,
        session_id: str = "",
        run_id: str = "",
        agent_id: str = "",
    ) -> StepBackOutcome:
        """Build the step-back outcome if feedback was captured."""
        if not self._enabled or self._feedback is None:
            return StepBackOutcome()

        checkpoint_seq = self._ledger.review.last_checkpoint_seq
        messages = list(self._ledger.messages)

        outcome = await execute_step_back(
            messages=messages,
            checkpoint_seq=checkpoint_seq,
            experience=self._feedback,
            review_tool_call_id=self._review_tool_call_id,
            step_lookup=self._step_lookup,
            storage=storage,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )

        # Reset tracking
        if outcome.messages:
            max_seq = max(
                (msg.get("_sequence", 0) for msg in outcome.messages), default=0
            )
            self._ledger.review.last_review_seq = max(max_seq, self._review_seq or 0)
        self._ledger.review.is_review_pending = False

        return outcome


def _parse_declared_milestones(output: object) -> list[Milestone]:
    if not isinstance(output, dict):
        return []
    raw_milestones = output.get("milestones")
    if not isinstance(raw_milestones, list):
        return []

    milestones: list[Milestone] = []
    for raw in raw_milestones:
        if not isinstance(raw, dict):
            continue
        milestone_id = raw.get("id")
        if not isinstance(milestone_id, str) or not milestone_id.strip():
            continue
        description = raw.get("description", "")
        if not isinstance(description, str):
            description = ""
        status = raw.get("status", "pending")
        if status not in {"pending", "active", "completed", "abandoned"}:
            status = "pending"
        milestones.append(
            Milestone(
                id=milestone_id.strip(),
                description=description,
                status=status,
            )
        )
    return milestones


__all__ = [
    "ReviewBatch",
    "StepBackOutcome",
    "ReviewTrigger",
    "activate_next_milestone",
    "check_review_trigger",
    "complete_active_milestone",
    "declare_milestones",
    "execute_step_back",
    "get_active_milestone",
    "inject_system_review",
]
