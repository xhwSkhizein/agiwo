import asyncio

import pytest

from agiwo.agent.execution_context import ExecutionContext
from agiwo.agent.stream_channel import StreamChannel
from agiwo.agent.tool_auth import ConsentWaiter, ToolAuthorizationRuntime
from agiwo.tool.authz import ConsentDecision, InMemoryConsentStore, PermissionPolicy, ToolPermissionProfile


class RecordingNotifier:
    def __init__(self) -> None:
        self.required_calls: list[dict] = []
        self.denied_calls: list[dict] = []

    async def notify_required(self, **kwargs) -> None:
        self.required_calls.append(kwargs)

    async def notify_denied(self, **kwargs) -> None:
        self.denied_calls.append(kwargs)


def build_context(*, user_id: str | None = "user-1") -> ExecutionContext:
    return ExecutionContext(
        session_id="session-1",
        run_id="run-1",
        agent_id="agent-1",
        agent_name="agent-1",
        channel=StreamChannel(),
        user_id=user_id,
    )


@pytest.mark.asyncio
async def test_authorize_allows_when_policy_missing() -> None:
    runtime = ToolAuthorizationRuntime()

    outcome = await runtime.authorize(
        tool_call_id="call-1",
        tool_name="bash",
        tool_args={"command": "ls"},
        context=build_context(),
    )

    assert outcome.allowed is True
    assert outcome.reason == "Authorization disabled"


@pytest.mark.asyncio
async def test_authorize_denies_when_policy_denies() -> None:
    notifier = RecordingNotifier()
    runtime = ToolAuthorizationRuntime(
        policy=PermissionPolicy(
            tool_profiles={"bash": ToolPermissionProfile(mode="deny")}
        ),
        notifier=notifier,
    )

    outcome = await runtime.authorize(
        tool_call_id="call-1",
        tool_name="bash",
        tool_args={"command": "rm -rf /"},
        context=build_context(),
    )

    assert outcome.allowed is False
    assert outcome.reason == "Tool denied by permission profile"
    assert len(notifier.denied_calls) == 1


@pytest.mark.asyncio
async def test_authorize_uses_cached_consent() -> None:
    store = InMemoryConsentStore()
    await store.save_consent(
        user_id="user-1",
        tool_name="bash",
        patterns=["bash(command=ls)"],
    )
    runtime = ToolAuthorizationRuntime(
        policy=PermissionPolicy(
            tool_profiles={"bash": ToolPermissionProfile(mode="require_consent")}
        ),
        consent_store=store,
    )

    outcome = await runtime.authorize(
        tool_call_id="call-1",
        tool_name="bash",
        tool_args={"command": "ls"},
        context=build_context(),
    )

    assert outcome.allowed is True
    assert outcome.from_cache is True
    assert outcome.reason == "Allowed by cached user consent"


@pytest.mark.asyncio
async def test_authorize_times_out_when_consent_not_resolved() -> None:
    notifier = RecordingNotifier()
    runtime = ToolAuthorizationRuntime(
        policy=PermissionPolicy(
            tool_profiles={"bash": ToolPermissionProfile(mode="require_consent")}
        ),
        waiter=ConsentWaiter(default_timeout=0.01),
        notifier=notifier,
    )

    outcome = await runtime.authorize(
        tool_call_id="call-1",
        tool_name="bash",
        tool_args={"command": "ls"},
        context=build_context(),
        timeout=0.01,
    )

    assert outcome.allowed is False
    assert outcome.reason == "User consent timed out"
    assert len(notifier.required_calls) == 1
    assert len(notifier.denied_calls) == 1


@pytest.mark.asyncio
async def test_authorize_saves_user_consent_after_approval() -> None:
    waiter = ConsentWaiter(default_timeout=1.0)
    store = InMemoryConsentStore()
    runtime = ToolAuthorizationRuntime(
        policy=PermissionPolicy(
            tool_profiles={"bash": ToolPermissionProfile(mode="require_consent")}
        ),
        consent_store=store,
        waiter=waiter,
    )

    task = asyncio.create_task(
        runtime.authorize(
            tool_call_id="call-1",
            tool_name="bash",
            tool_args={"command": "ls"},
            context=build_context(),
        )
    )
    await asyncio.sleep(0)
    await waiter.resolve(
        "call-1",
        ConsentDecision(decision="allow", patterns=["bash(command=ls)"]),
    )

    outcome = await task
    cached = await store.check_consent("user-1", "bash", {"command": "ls"})

    assert outcome.allowed is True
    assert outcome.reason == "Allowed by user consent"
    assert cached == "allowed"
