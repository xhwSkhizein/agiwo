"""Deterministic fixtures for Objective E2E scenarios (P6-01)."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from agiwo.agent import Agent, AgentOptions, UserMessage
from agiwo.agent.models.config import AgentConfig
from agiwo.agent.models.input import ContentPart, ContentType
from agiwo.llm.base import Model, StreamChunk
from agiwo.objective import ObjectiveService
from agiwo.objective.models import BudgetLimits, new_id
from agiwo.objective.store.memory import InMemoryObjectiveStore
from agiwo.objective.store.sqlite import SQLiteObjectiveStore
from agiwo.scheduler import Scheduler
from agiwo.tool.base import BaseTool, ToolIdempotency, ToolResult
from agiwo.tool.context import ToolContext


def user_message(text: str) -> UserMessage:
    return UserMessage(
        content=[ContentPart(type=ContentType.TEXT, text=text)],
        is_user_provided=True,
    )


def finalization_json(
    *,
    target: str,
    expects_reply: bool | None = None,
    report: str | None = None,
) -> str:
    """Build finalization JSON (no report field — work-loop text is authoritative).

    ``report`` is accepted only for call-site compatibility and ignored.
    """
    del report
    decision: dict[str, Any] = {"target": target}
    if expects_reply is not None:
        decision["expects_reply"] = expects_reply
    return json.dumps(
        {
            "decision": decision,
            "new_contributions": [],
            "contribution_annotations": [],
            "objective_update": None,
            "artifact_refs": [],
            "carry_forward": [],
        }
    )


def scripted_run_turns(*finals: str) -> list[str]:
    responses: list[str] = []
    for final in finals:
        responses.append("working")
        responses.append(final)
    return responses


# Back-compat alias for older e2e imports.
scripted_assignment_turns = scripted_run_turns


class ScriptedFinalizationModel(Model):
    """Ordinal response queue for ordinary report + finalization JSON pairs."""

    def __init__(self, responses: list[str]) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = list(responses)
        self.calls = 0

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools, messages
        self.calls += 1
        if not self._responses:
            text = finalization_json(report="fallback", target="agent")
        else:
            text = self._responses.pop(0)
        yield StreamChunk(content=text)
        yield StreamChunk(finish_reason="stop")


class ScriptedToolCallModel(Model):
    """Response queue supporting plain text and tool_calls dicts (update_plan, etc.)."""

    def __init__(self, responses: list[str | dict]) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = list(responses)
        self.calls: list[list[dict]] = []

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools
        self.calls.append(messages)
        if not self._responses:
            yield StreamChunk(
                content=finalization_json(report="fallback", target="agent")
            )
        else:
            response = self._responses.pop(0)
            if isinstance(response, dict):
                yield StreamChunk(tool_calls=[response])
            else:
                yield StreamChunk(content=response)
        yield StreamChunk(finish_reason="stop")


class CountingIdempotentTool(BaseTool):
    """Fake tool with explicit idempotency and side-effect counting."""

    name = "counting_tool"
    description = "counts external effects"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    idempotency = ToolIdempotency.GUARANTEED

    def __init__(self) -> None:
        self.side_effects = 0
        self.executions = 0

    async def execute(
        self, arguments: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        del context
        self.executions += 1
        self.side_effects += 1
        return ToolResult.success(output=f"ok:{arguments.get('value', '')}")


DEFAULT_BUDGET = BudgetLimits(
    handoffs=20,
    verification_attempts=10,
    llm_cost_usd=10.0,
    active_seconds=3600,
)


@dataclass
class E2ERuntime:
    service: ObjectiveService
    scheduler: Scheduler
    store: InMemoryObjectiveStore | SQLiteObjectiveStore
    agent: Agent
    model: Model
    session_id: str
    tools: list[BaseTool] = field(default_factory=list)


@asynccontextmanager
async def objective_e2e_runtime(
    *,
    responses: list[str] | list[str | dict],
    session_id: str | None = None,
    store: InMemoryObjectiveStore | SQLiteObjectiveStore | None = None,
    tools: list[BaseTool] | None = None,
    budget: BudgetLimits | None = None,
    model: Model | None = None,
) -> AsyncIterator[E2ERuntime]:
    del budget  # reserved for future create helpers
    sid = session_id or f"session-{new_id()[:8]}"
    owned_store = store is None
    store = store or InMemoryObjectiveStore()
    if isinstance(store, SQLiteObjectiveStore):
        await store.connect()
    scheduler = Scheduler()
    await scheduler.start()
    if model is None:
        model = ScriptedFinalizationModel(list(responses))  # type: ignore[arg-type]
    tool_list = list(tools or [])
    agent = Agent(
        AgentConfig(
            name="e2e",
            description="e2e",
            options=AgentOptions(
                enable_termination_summary=False,
                max_steps_per_run=20,
            ),
        ),
        model=model,
        tools=tool_list or None,
        id=sid,
    )

    async def provider(_session_id: str) -> Agent:
        return agent

    service = ObjectiveService(
        store,
        scheduler=scheduler,
        default_agent_provider=provider,
    )
    await service.start_dispatcher()
    runtime = E2ERuntime(
        service=service,
        scheduler=scheduler,
        store=store,
        agent=agent,
        model=model,
        session_id=sid,
        tools=tool_list,
    )
    try:
        yield runtime
    finally:
        await service.stop_dispatcher()
        await scheduler.stop()
        await agent.close()
        if owned_store and isinstance(store, SQLiteObjectiveStore):
            await store.close()


async def wait_for(
    predicate: Callable[[], Any],
    *,
    timeout_seconds: float = 8.0,
    interval: float = 0.1,
) -> Any:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last: Any = None
    while asyncio.get_running_loop().time() < deadline:
        last = predicate()
        if asyncio.iscoroutine(last):
            last = await last
        if last:
            return last
        await asyncio.sleep(interval)
    raise AssertionError(f"condition not met before timeout; last={last!r}")


async def wait_status(
    service: ObjectiveService,
    objective_id: str,
    *statuses: str,
    timeout_seconds: float = 8.0,
) -> Any:
    wanted = set(statuses)

    async def _check():
        view = await service.get_view(objective_id)
        if view is not None and view.status.value in wanted:
            return view
        return None

    return await wait_for(_check, timeout_seconds=timeout_seconds)


async def wait_completed(
    service: ObjectiveService, objective_id: str, *, timeout_seconds: float = 8.0
) -> Any:
    async def _check():
        view = await service.get_view(objective_id)
        if (
            view is not None
            and view.status.value == "COMPLETED"
            and view.delivery_report
        ):
            return view
        return None

    return await wait_for(_check, timeout_seconds=timeout_seconds)
