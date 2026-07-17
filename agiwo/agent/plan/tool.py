"""Built-in update_plan system tool."""

import time
from typing import Any

from agiwo.tool.base import AbortSignal, BaseTool, ToolResult
from agiwo.tool.context import ToolContext

_MILESTONE_STATUSES = frozenset({"pending", "active", "completed", "abandoned"})


class UpdatePlanTool(BaseTool):
    """Declare or revise the current RunPlan via atomic incremental changes."""

    name = "update_plan"
    description = (
        "Declare or revise the work plan for the current run. Pass only the "
        "incremental changes for this call. New milestone ids require a "
        "description; existing ids update only the fields you provide; omitted "
        "milestones stay unchanged. The plan covers only work this run can "
        "complete and verify before the next Decision — not successor agents, "
        "verifiers, or the user. Prefer verifiable stage outcomes over tiny "
        "operational steps. There is no physical delete: mark items abandoned "
        "when they are no longer needed. Completed or abandoned items may be "
        "reopened in this same run.\n\n"
        'Example: [{"id":"understand","description":"Identify how auth tokens '
        'are validated"}, {"id":"fix","description":"Apply the fix and verify '
        'with tests"}]'
    )
    concurrency_safe = False

    def get_parameters(self) -> dict[str, Any]:
        # Schema approximates runtime rules: a brand-new milestone id always
        # needs description (enforced in apply_plan_changes). Updates to an
        # existing id may omit description when only changing status. Static
        # JSON Schema cannot know which ids already exist, so anyOf covers the
        # two legal shapes models should emit.
        change_properties = {
            "id": {
                "type": "string",
                "description": "Stable milestone id within this run's plan.",
            },
            "description": {
                "type": "string",
                "description": (
                    "Human-readable milestone outcome. REQUIRED when this id "
                    "is new to the current plan; optional when updating an "
                    "existing milestone."
                ),
            },
            "status": {
                "type": "string",
                "enum": sorted(_MILESTONE_STATUSES),
                "description": (
                    "Optional status: pending, active, completed, or abandoned."
                ),
            },
        }
        return {
            "type": "object",
            "properties": {
                "changes": {
                    "type": "array",
                    "description": (
                        "Ordered list of plan changes. Every item needs id. "
                        "New milestone ids MUST include a non-empty "
                        "description (create shape). Existing ids may omit "
                        "description when only updating status (update "
                        "shape). Omitted milestones stay unchanged."
                    ),
                    "items": {
                        "type": "object",
                        "properties": change_properties,
                        "required": ["id"],
                        "anyOf": [
                            {
                                "title": "create_or_set_description",
                                "required": ["id", "description"],
                                "description": (
                                    "Create a new milestone or revise "
                                    "description. Always use this shape for "
                                    "ids not already in the plan."
                                ),
                            },
                            {
                                "title": "update_existing_status",
                                "required": ["id", "status"],
                                "description": (
                                    "Update status on an existing milestone. "
                                    "Do not use for new ids — those need "
                                    "description."
                                ),
                            },
                        ],
                    },
                },
            },
            "required": ["changes"],
        }

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        changes = parameters.get("changes", [])
        if not isinstance(changes, list):
            return ToolResult.failed(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                error="changes must be an array",
                start_time=start_time,
            )
        if not changes:
            return ToolResult.failed(
                tool_name=self.name,
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                error="changes must be a non-empty array",
                start_time=start_time,
            )

        normalized_changes: list[dict[str, str]] = []
        seen_ids: set[str] = set()
        for index, change in enumerate(changes):
            if not isinstance(change, dict):
                return ToolResult.failed(
                    tool_name=self.name,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    error=f"changes[{index}] must be an object",
                    start_time=start_time,
                )
            raw_id = change.get("id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                return ToolResult.failed(
                    tool_name=self.name,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    error=f"changes[{index}].id must be a non-empty string",
                    start_time=start_time,
                )
            milestone_id = raw_id.strip()
            if milestone_id in seen_ids:
                return ToolResult.failed(
                    tool_name=self.name,
                    tool_call_id=context.tool_call_id,
                    input_args=parameters,
                    error=f"duplicate milestone id in changes: {milestone_id}",
                    start_time=start_time,
                )
            seen_ids.add(milestone_id)

            normalized: dict[str, str] = {"id": milestone_id}
            if "description" in change:
                description = change["description"]
                if not isinstance(description, str) or not description.strip():
                    return ToolResult.failed(
                        tool_name=self.name,
                        tool_call_id=context.tool_call_id,
                        input_args=parameters,
                        error=(
                            f"changes[{index}].description must be a "
                            "non-empty string when provided"
                        ),
                        start_time=start_time,
                    )
                normalized["description"] = description.strip()
            if "status" in change:
                status = change["status"]
                if not isinstance(status, str) or status not in _MILESTONE_STATUSES:
                    return ToolResult.failed(
                        tool_name=self.name,
                        tool_call_id=context.tool_call_id,
                        input_args=parameters,
                        error=f"changes[{index}].status is invalid: {status}",
                        start_time=start_time,
                    )
                normalized["status"] = status
            normalized_changes.append(normalized)

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content="Plan changes accepted",
            output={"changes": normalized_changes},
            start_time=start_time,
        )


__all__ = ["UpdatePlanTool"]
