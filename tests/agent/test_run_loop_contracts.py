import asyncio
from collections.abc import AsyncIterator

import pytest

from agiwo.agent import (
    Agent,
    AgentConfig,
    AgentOptions,
    ContentPart,
    ContentType,
    UserMessage,
)
from agiwo.agent.hooks import HookPhase, HookRegistry, transform
from agiwo.agent.prompt import apply_steering_messages
from agiwo.agent.runtime.session import SessionRuntime
from agiwo.agent.storage.base import InMemoryRunLogStorage
from agiwo.llm.base import Model, StreamChunk


class _FixedResponseModel(Model):
    def __init__(self, response: str = "ok") -> None:
        super().__init__(id="fixed-model", name="fixed-model", temperature=0.0)
        self._response = response

    async def arun_stream(self, messages, tools=None) -> AsyncIterator[StreamChunk]:
        del messages, tools
        yield StreamChunk(content=self._response)
        yield StreamChunk(finish_reason="stop")


def _make_session_runtime() -> SessionRuntime:
    return SessionRuntime(
        session_id="session-1",
        run_log_storage=InMemoryRunLogStorage(),
    )


@pytest.mark.asyncio
async def test_enqueue_steer_accepts_image_only_input() -> None:
    session_runtime = _make_session_runtime()

    accepted = await session_runtime.enqueue_steer(
        UserMessage(
            content=[
                ContentPart(
                    type=ContentType.IMAGE,
                    url="https://example.com/diagram.png",
                )
            ]
        )
    )

    assert accepted is True


@pytest.mark.asyncio
async def test_apply_steering_messages_preserves_multimodal_payloads() -> None:
    session_runtime = _make_session_runtime()
    steering_input = UserMessage(
        content=[
            ContentPart(type=ContentType.TEXT, text="Look at this image"),
            ContentPart(type=ContentType.IMAGE, url="https://example.com/diagram.png"),
        ]
    )
    await session_runtime.enqueue_steer(steering_input)

    updated = apply_steering_messages(
        [{"role": "assistant", "content": "waiting"}],
        session_runtime.peek_pending_steer_inputs(),
    )

    assert updated[-1] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "Look at this image"},
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/diagram.png"},
            },
        ],
    }


@pytest.mark.asyncio
async def test_session_runtime_peek_steer_does_not_consume_until_ack() -> None:
    session_runtime = _make_session_runtime()

    await session_runtime.enqueue_steer("follow up")

    pending = session_runtime.peek_pending_steer_inputs()

    assert [UserMessage.from_value(item).extract_text() for item in pending] == [
        "follow up"
    ]
    assert [
        UserMessage.from_value(item).extract_text()
        for item in session_runtime.peek_pending_steer_inputs()
    ] == ["follow up"]

    session_runtime.ack_pending_steer_inputs(len(pending))

    assert session_runtime.peek_pending_steer_inputs() == []


@pytest.mark.asyncio
async def test_early_hooks_receive_initialized_context() -> None:
    observed: list[tuple[str, int, str, bool]] = []
    event = asyncio.Event()

    async def before_run(payload):
        user_input = payload["user_input"]
        context = payload["context"]
        del user_input
        observed.append(
            (
                "before_run",
                context.config.max_steps,
                context.config.config_root,
                context.hooks.has_phase(HookPhase.ASSEMBLE_CONTEXT),
            )
        )
        payload = dict(payload)
        payload["prelude_text"] = None
        return payload

    async def memory_retrieve(payload):
        user_input = payload["user_input"]
        context = payload["context"]
        del user_input
        observed.append(
            (
                "memory_retrieve",
                context.config.max_steps,
                context.config.config_root,
                context.hooks.has_phase(HookPhase.PREPARE),
            )
        )
        event.set()
        payload = dict(payload)
        payload["memories"] = []
        return payload

    agent = Agent(
        AgentConfig(
            name="hook-context",
            options=AgentOptions(max_steps=7, config_root="/tmp/agiwo-root"),
        ),
        model=_FixedResponseModel(),
        hooks=HookRegistry(
            [
                transform(HookPhase.PREPARE, "before_run", before_run),
                transform(
                    HookPhase.ASSEMBLE_CONTEXT,
                    "memory_retrieve",
                    memory_retrieve,
                ),
            ]
        ),
    )

    result = await agent.run("hello", session_id="hook-context-session")

    assert result.response == "ok"
    assert event.is_set()
    assert observed == [
        ("before_run", 7, "/tmp/agiwo-root", True),
        ("memory_retrieve", 7, "/tmp/agiwo-root", True),
    ]
