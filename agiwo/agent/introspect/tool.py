"""Built-in review_trajectory system tool."""

import time
from typing import Any

from agiwo.agent.introspect.window import build_review_window
from agiwo.agent.runtime.context import RunContext
from agiwo.tool.base import AbortSignal, BaseTool, ToolResult
from agiwo.tool.context import RunContextLike, ToolContext

_VALID_SCORES = frozenset({0, 1, 2, 3})


class ReviewTrajectoryTool(BaseTool):
    """Respond to a <system-review> prompt with append-only trajectory metadata."""

    name = "review_trajectory"
    description = (
        "Respond to a <system-review> prompt by assessing whether your recent "
        "tool calls advance the active milestone.\n\n"
        "Parameters:\n"
        "- aligned (boolean, required): true if trajectory aligns with the "
        "active milestone, false if it drifted.\n"
        "- experience (string, optional): concise summary when aligned=false.\n"
        "- tool_usefulness (array, optional): score each tool in the current "
        "review window at most once. Scores: 0=harmful/misleading, 1=low value, "
        "2=useful, 3=critical. Omit tools you cannot judge; the system records "
        "them as unknown.\n\n"
        "This tool call and result remain in history."
    )
    concurrency_safe = False

    def get_parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "aligned": {
                    "type": "boolean",
                    "description": (
                        "true if your recent trajectory aligns with the "
                        "active milestone, false if it has drifted."
                    ),
                },
                "experience": {
                    "type": "string",
                    "description": (
                        "Optional when aligned=false. Concise summary of what "
                        "was attempted, learned, and how to adjust next."
                    ),
                },
                "tool_usefulness": {
                    "type": "array",
                    "description": (
                        "Optional usefulness scores for tools in the current "
                        "review window. Each item must include tool_call_id "
                        "and score (0-3)."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool_call_id": {"type": "string"},
                            "score": {
                                "type": "integer",
                                "enum": [0, 1, 2, 3],
                            },
                        },
                        "required": ["tool_call_id", "score"],
                    },
                },
            },
            "required": ["aligned"],
        }

    def build_context(
        self, run_context: RunContextLike, *, tool_call_id: str = ""
    ) -> ToolContext:
        context = super().build_context(run_context, tool_call_id=tool_call_id)
        if not isinstance(run_context, RunContext):
            return context
        review_window = build_review_window(
            run_context.ledger.messages,
            last_boundary_seq=run_context.ledger.introspection.last_boundary_seq,
            exclude_tool_call_id=tool_call_id or None,
        )
        metadata = dict(context.metadata)
        metadata["review_window"] = review_window
        return ToolContext(
            session_id=context.session_id,
            agent_id=context.agent_id,
            agent_name=context.agent_name,
            user_id=context.user_id,
            timeout_at=context.timeout_at,
            depth=context.depth,
            metadata=metadata,
            gate_checked=context.gate_checked,
            tool_call_id=context.tool_call_id,
        )

    async def execute(
        self,
        parameters: dict[str, Any],
        context: ToolContext,
        abort_signal: AbortSignal | None = None,
    ) -> ToolResult:
        del abort_signal
        start_time = time.time()
        aligned = parameters.get("aligned")
        experience = parameters.get("experience", "")
        raw_usefulness = parameters.get("tool_usefulness")

        if not isinstance(aligned, bool):
            return ToolResult.failed(
                tool_name=self.name,
                error="aligned must be a boolean",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
        if experience is None:
            experience = ""
        if not isinstance(experience, str):
            return ToolResult.failed(
                tool_name=self.name,
                error="experience must be a string",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )
        if raw_usefulness is not None and not isinstance(raw_usefulness, list):
            return ToolResult.failed(
                tool_name=self.name,
                error="tool_usefulness must be an array",
                tool_call_id=context.tool_call_id,
                input_args=parameters,
                start_time=start_time,
            )

        review_window = context.metadata.get("review_window")
        window_items = review_window if isinstance(review_window, list) else []
        window_by_id = {
            item["tool_call_id"]: item
            for item in window_items
            if isinstance(item, dict) and isinstance(item.get("tool_call_id"), str)
        }
        window_ids = set(window_by_id)

        accepted_scores: list[dict[str, Any]] = []
        rejected: list[str] = []
        seen_ids: set[str] = set()
        if isinstance(raw_usefulness, list):
            for index, entry in enumerate(raw_usefulness):
                if not isinstance(entry, dict):
                    rejected.append(f"tool_usefulness[{index}] must be an object")
                    continue
                tool_call_id = entry.get("tool_call_id")
                score = entry.get("score")
                if not isinstance(tool_call_id, str) or not tool_call_id:
                    rejected.append(
                        f"tool_usefulness[{index}].tool_call_id must be a string"
                    )
                    continue
                if tool_call_id in seen_ids:
                    rejected.append(f"duplicate tool_call_id: {tool_call_id}")
                    continue
                seen_ids.add(tool_call_id)
                if tool_call_id not in window_ids:
                    rejected.append(f"out-of-window tool_call_id: {tool_call_id}")
                    continue
                if not isinstance(score, int) or score not in _VALID_SCORES:
                    rejected.append(f"illegal score for {tool_call_id}: {score!r}")
                    continue
                accepted_scores.append(
                    {
                        "tool_call_id": tool_call_id,
                        "tool_name": window_by_id[tool_call_id].get("tool_name"),
                        "score": score,
                    }
                )

        unknown_tool_call_ids = sorted(window_ids - seen_ids)
        tool_usefulness = [
            *accepted_scores,
            *[
                {
                    "tool_call_id": tool_call_id,
                    "tool_name": window_by_id[tool_call_id].get("tool_name"),
                    "score": None,
                }
                for tool_call_id in unknown_tool_call_ids
            ],
        ]

        content_parts = [f"Trajectory review: aligned={aligned}."]
        if experience:
            content_parts.append(experience)
        if rejected:
            content_parts.append(f"Rejected scores: {'; '.join(rejected)}.")
        if unknown_tool_call_ids:
            content_parts.append(
                f"Unknown usefulness: {', '.join(unknown_tool_call_ids)}."
            )

        return ToolResult.success(
            tool_name=self.name,
            tool_call_id=context.tool_call_id,
            input_args=parameters,
            content=" ".join(content_parts),
            output={
                "aligned": aligned,
                "experience": experience,
                "tool_usefulness": tool_usefulness,
                "rejected_scores": rejected,
                "unknown_tool_call_ids": unknown_tool_call_ids,
            },
            start_time=start_time,
        )


__all__ = ["ReviewTrajectoryTool"]
