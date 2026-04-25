"""Goal-directed review public API used by ``run_tool_batch.py``."""

from dataclasses import dataclass
from typing import Any

from agiwo.agent.models.config import AgentOptions
from agiwo.agent.models.review import Milestone, ReviewCheckpoint
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
    strip_system_review_notices,
)
from agiwo.agent.review.replay import build_review_state_from_entries
from agiwo.agent.review.step_back_executor import (
    ContentUpdate,
    StepBackOutcome,
    execute_step_back,
)
from agiwo.agent.storage.base import RunLogStorage
from agiwo.tool.base import BaseTool, ToolResult


class ReviewBatch:
    """Per-batch review lifecycle object."""

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
        self._pending_outcome: StepBackOutcome | None = None
        self._review_seq: int | None = None
        self._review_tool_call_id: str | None = None
        self._review_hidden_step_ids: list[str] = []
        self._feedback: str | None = None
        self._step_lookup: dict[str, dict[str, Any]] = {}
        self._pending_review_notice: ReviewNoticeRequest | None = None
        self._review_requested = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def process_result(
        self,
        result: ToolResult,
        *,
        current_seq: int = 0,
        assistant_step_id: str | None = None,
        tool_step_id: str | None = None,
    ) -> str:
        """Process a tool result and return the prompt-visible content."""

        self._pending_review_notice = None
        content = result.content or ""
        if not self._enabled:
            return content

        if result.tool_name == "review_trajectory" and result.is_success:
            return self._handle_review_result(
                result,
                content=content,
                current_seq=current_seq,
                assistant_step_id=assistant_step_id,
                tool_step_id=tool_step_id,
            )

        self._record_review_count_for_tool(result.tool_name)

        if result.tool_name == "declare_milestones" and result.is_success:
            milestones = _parse_declared_milestones(result.output)
            if milestones:
                declare_milestones(
                    self._ledger.review,
                    milestones,
                    current_seq=current_seq,
                )

        is_error = not result.is_success
        if is_error:
            self._ledger.review.consecutive_errors += 1
        else:
            self._ledger.review.consecutive_errors = 0

        if self._review_requested or _has_prompt_visible_system_review(
            self._ledger.messages
        ):
            return content

        trigger = check_review_trigger(
            state=self._ledger.review,
            enabled=True,
            is_error=is_error and self._config.review_on_error,
            step_interval=self._config.review_step_interval,
            error_threshold=2,
            tool_name=result.tool_name,
        )

        if trigger is ReviewTrigger.NONE:
            return content

        if trigger is ReviewTrigger.MILESTONE_SWITCH:
            self._ledger.review.pending_review_reason = None

        milestone = get_active_milestone(self._ledger.review)
        step_count = self._ledger.review.review_count_since_checkpoint
        self._pending_review_notice = ReviewNoticeRequest(
            content=content,
            milestone=milestone,
            step_count=step_count,
            trigger=trigger,
        )
        self._review_requested = True
        return inject_system_review(
            content,
            milestone,
            step_count,
            trigger_reason=trigger.value,
        )

    def consume_review_notice_request(self) -> "ReviewNoticeRequest | None":
        request = self._pending_review_notice
        self._pending_review_notice = None
        return request

    def register_step(self, tool_call_id: str, step_id: str, sequence: int) -> None:
        """Register a committed step for later step-back lookup."""

        self._step_lookup[tool_call_id] = {
            "id": step_id,
            "sequence": sequence,
        }

    def _record_review_count_for_tool(self, tool_name: str) -> None:
        if tool_name == "review_trajectory":
            return
        self._ledger.review.review_count_since_checkpoint += 1

    async def finalize(
        self,
        *,
        storage: RunLogStorage | None = None,
        session_id: str = "",
        run_id: str = "",
        agent_id: str = "",
    ) -> StepBackOutcome:
        """Return the structured cleanup outcome for this batch."""

        if not self._enabled or self._pending_outcome is None:
            return StepBackOutcome()

        if self._pending_outcome.mode != "step_back":
            cleanup_updates = await _build_system_review_cleanup_updates(
                self._ledger.messages,
                storage=storage,
                session_id=session_id,
                run_id=run_id,
                agent_id=agent_id,
                review_tool_call_id=self._pending_outcome.review_tool_call_id,
            )
            self._pending_outcome.content_updates.extend(cleanup_updates)
            return self._pending_outcome

        if storage is None:
            raise ValueError(
                "ReviewBatch.finalize() requires a non-None storage because "
                "execute_step_back() persists condensed content via storage."
            )

        checkpoint_seq = (
            self._ledger.review.latest_checkpoint.seq
            if self._ledger.review.latest_checkpoint is not None
            else 0
        )
        outcome = await execute_step_back(
            messages=list(self._ledger.messages),
            checkpoint_seq=checkpoint_seq,
            experience=self._feedback or "",
            review_tool_call_id=self._review_tool_call_id,
            step_lookup=self._step_lookup,
            storage=storage,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
        )
        outcome.hidden_step_ids = list(self._review_hidden_step_ids)
        self._ledger.review.last_review_seq = (
            self._review_seq or self._ledger.review.last_review_seq
        )
        self._ledger.review.pending_review_reason = None
        return outcome

    def _handle_review_result(
        self,
        result: ToolResult,
        *,
        content: str,
        current_seq: int,
        assistant_step_id: str | None,
        tool_step_id: str | None,
    ) -> str:
        self._ledger.review.consecutive_errors = 0
        self._review_seq = current_seq
        self._review_tool_call_id = result.tool_call_id or None
        self._review_hidden_step_ids = [
            step_id
            for step_id in (assistant_step_id, tool_step_id)
            if step_id is not None
        ]

        output = result.output if isinstance(result.output, dict) else {}
        aligned = output.get("aligned")
        if aligned is True:
            active_milestone = get_active_milestone(self._ledger.review)
            self._ledger.review.latest_checkpoint = ReviewCheckpoint(
                seq=current_seq,
                milestone_id=active_milestone.id
                if active_milestone is not None
                else "",
            )
            self._ledger.review.last_review_seq = current_seq
            self._ledger.review.review_count_since_checkpoint = 0
            self._ledger.review.pending_review_reason = None
            self._pending_outcome = StepBackOutcome(
                mode="metadata_only",
                review_tool_call_id=self._review_tool_call_id,
                hidden_step_ids=list(self._review_hidden_step_ids),
            )
            return content

        if aligned is False:
            experience = output.get("experience")
            self._feedback = experience if isinstance(experience, str) else content
            self._ledger.review.review_count_since_checkpoint = 0
            self._pending_outcome = StepBackOutcome(
                mode="step_back",
                review_tool_call_id=self._review_tool_call_id,
                hidden_step_ids=list(self._review_hidden_step_ids),
            )
            return content

        self._ledger.review.last_review_seq = current_seq
        self._ledger.review.review_count_since_checkpoint = 0
        self._ledger.review.pending_review_reason = None
        self._pending_outcome = StepBackOutcome(
            mode="metadata_only",
            review_tool_call_id=self._review_tool_call_id,
            hidden_step_ids=list(self._review_hidden_step_ids),
        )
        return content


@dataclass(frozen=True)
class ReviewNoticeRequest:
    content: str
    milestone: Milestone | None
    step_count: int
    trigger: ReviewTrigger


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
    "ReviewNoticeRequest",
    "ReviewTrigger",
    "StepBackOutcome",
    "activate_next_milestone",
    "build_review_state_from_entries",
    "check_review_trigger",
    "complete_active_milestone",
    "declare_milestones",
    "execute_step_back",
    "get_active_milestone",
    "inject_system_review",
    "strip_system_review_notices",
]


def _has_prompt_visible_system_review(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if message.get("role") == "tool" and _has_system_review_text(content):
            return True
    return False


def _has_system_review_text(content: object) -> bool:
    return isinstance(content, str) and "<system-review>" in content


async def _build_system_review_cleanup_updates(
    messages: list[dict[str, Any]],
    *,
    storage: RunLogStorage | None,
    session_id: str,
    run_id: str,
    agent_id: str,
    review_tool_call_id: str | None,
) -> list[ContentUpdate]:
    updates: list[ContentUpdate] = []
    for message in messages:
        if message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        if review_tool_call_id and tool_call_id == review_tool_call_id:
            continue

        content = message.get("content")
        if not _has_system_review_text(content):
            continue

        cleaned_content = strip_system_review_notices(content)
        step_id = ""
        if storage is not None:
            step = await storage.get_step_by_tool_call_id(session_id, tool_call_id)
            step_id = step.id if step is not None else ""
            if step_id:
                await storage.append_step_condensed_content(
                    session_id,
                    run_id,
                    agent_id,
                    step_id,
                    cleaned_content,
                )
        updates.append(
            ContentUpdate(
                step_id=step_id,
                tool_call_id=tool_call_id,
                content=cleaned_content,
            )
        )
    return updates
