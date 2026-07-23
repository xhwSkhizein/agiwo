import dataclasses
import inspect
import time

from agiwo.agent import (
    AgentConfig,
    AgentOptions,
    AgentSpec,
    MainAgent,
    MainAgentState,
    QueueItem,
    QueueItemKind,
    UserMessage,
)
from agiwo.agent.hooks import HookRegistry
from agiwo.llm.base import Model


class MockModel(Model):
    async def arun_stream(self, messages, tools=None):
        if False:
            yield None


def _build_spec() -> AgentSpec:
    return AgentSpec(
        config=AgentConfig(
            name="main-agent",
            description="AB-01 test agent",
            system_prompt="Test prompt",
            options=AgentOptions(),
        )
    )


def _build_main_agent(
    *, session_id: str = "session-1", agent_id: str = "agent-1"
) -> MainAgent:
    return MainAgent(
        session_id=session_id,
        agent_id=agent_id,
        spec=_build_spec(),
        model=MockModel(id="mock", name="mock", provider="openai"),
        hooks=HookRegistry(),
    )


def test_agent_spec_holds_config_only() -> None:
    spec = _build_spec()

    assert spec.config.name == "main-agent"
    assert not hasattr(spec, "queue")
    assert "queue" not in {field.name for field in dataclasses.fields(spec)}


def test_queue_item_kind_includes_reserved_kinds() -> None:
    assert QueueItemKind.USER_INPUT.value == "user_input"
    assert QueueItemKind.GATE_FEEDBACK.value == "gate_feedback"
    assert QueueItemKind.WORKER_REPORT.value == "worker_report"


def test_main_agent_starts_idle_with_bound_ids() -> None:
    agent = _build_main_agent(session_id="sess-42", agent_id="root-99")

    assert agent.state is MainAgentState.IDLE
    assert agent.session_id == "sess-42"
    assert agent.agent_id == "root-99"
    assert agent.spec is not None


def test_main_agent_has_no_steer_or_inject_public_methods() -> None:
    public_methods = {
        name
        for name, member in inspect.getmembers(MainAgent, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    assert "steer" not in public_methods
    assert "inject_system_user_message" not in public_methods


def test_enqueue_user_input_appends_to_pending_queue() -> None:
    agent = _build_main_agent()
    message = UserMessage.from_value("hello")
    item = QueueItem(
        kind=QueueItemKind.USER_INPUT,
        message=message,
        created_at=time.time(),
    )

    agent.enqueue(item)

    assert agent.peek_pending() is item
    assert agent.ack_pending() is item
    assert agent.peek_pending() is None
