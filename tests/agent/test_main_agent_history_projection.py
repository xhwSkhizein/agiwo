"""Task 31.4: mid-run accept must project into the next Run exactly once."""

import asyncio
import json
import time
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    MainAgent,
    MainAgentState,
)
from agiwo.agent.hooks import HookRegistry
from agiwo.llm.base import Model, StreamChunk


class _ScriptedModel(Model):
    def __init__(
        self,
        responses: list[str | dict],
        *,
        start_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__(id="scripted", name="scripted", temperature=0.0)
        self._responses = iter(responses)
        self._start_event = start_event
        self.calls: list[list[dict]] = []

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del tools
        self.calls.append(list(messages))
        if self._start_event is not None and len(self.calls) == 1:
            await self._start_event.wait()
        response = next(self._responses)
        if isinstance(response, dict):
            yield StreamChunk(tool_calls=[response])
        else:
            yield StreamChunk(content=response)
        yield StreamChunk(finish_reason="stop")


def _build_main(*, model: Model) -> MainAgent:
    return MainAgent(
        session_id="session-history-proj",
        agent_id="agent-history-proj",
        spec=AgentSpec(
            config=AgentConfig(
                name="main-agent",
                description="history projection test",
                system_prompt="Test prompt",
                options=AgentOptions(max_steps_per_run=10),
            )
        ),
        model=model,
        hooks=HookRegistry(),
    )


def _update_plan_call(call_id: str) -> dict:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {
            "name": "update_plan",
            "arguments": json.dumps(
                {
                    "changes": [
                        {
                            "id": "inspect",
                            "description": "hold for mid-run accept",
                            "status": "completed",
                        }
                    ]
                }
            ),
        },
    }


def _user_texts(messages: list[dict]) -> list[str]:
    texts: list[str] = []
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content:
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        texts.append(text)
    return texts


async def _settle(main: MainAgent, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        completion = main._completion_task  # noqa: SLF001
        if completion is None and main.peek_pending() is None:
            return
        if completion is not None:
            await asyncio.wait({completion})
        else:
            await asyncio.sleep(0.01)
    raise AssertionError("MainAgent did not settle")


@pytest.mark.asyncio
async def test_mid_run_accept_appears_once_in_next_run_messages() -> None:
    """Dual-write (history + live pending) must not duplicate in the next Run."""
    gate = asyncio.Event()
    model = _ScriptedModel(
        [
            _update_plan_call("plan-1"),
            "first-run-done",
            "second-run-done",
        ],
        start_event=gate,
    )
    main = _build_main(model=model)

    first = await main.accept("msg-1")
    assert first is not None
    assert main.state is MainAgentState.RUNNING

    # Mid-run accept: history write + enqueue into the live Run.
    second = await main.accept("msg-2")
    assert second is first

    gate.set()
    await _settle(main)
    assert main.state is MainAgentState.IDLE

    calls_after_first_run = len(model.calls)
    assert calls_after_first_run >= 2  # plan turn + completion turn flushed pending

    third = await main.accept("msg-3")
    assert third is not None
    await _settle(main)

    # First LLM call of the new Run is the projection under test.
    new_run_first_call = model.calls[calls_after_first_run]
    user_texts = _user_texts(new_run_first_call)
    assert user_texts.count("msg-2") == 1, user_texts
    assert "msg-1" in user_texts
    assert "msg-3" in user_texts
