import importlib
import importlib.util


def test_nested_agent_modules_are_exposed_from_new_package() -> None:
    agent_tool_module = importlib.import_module("agiwo.agent.nested.agent_tool")
    context_module = importlib.import_module("agiwo.agent.nested.context")

    assert hasattr(agent_tool_module, "AgentTool")
    assert hasattr(context_module, "AgentToolContext")


def test_transport_package_has_been_removed_from_sdk_layer() -> None:
    assert importlib.util.find_spec("agiwo.agent.transport") is None


def test_hooks_module_exposes_hook_registry_and_memory_defaults() -> None:
    hooks_module = importlib.import_module("agiwo.agent.hooks")

    assert hasattr(hooks_module, "HookRegistry")
    assert hasattr(hooks_module, "DefaultMemoryHook")


def test_runtime_modules_expose_only_runtime_support_boundaries() -> None:
    runtime_module = importlib.import_module("agiwo.agent.runtime")
    context_module = importlib.import_module("agiwo.agent.runtime.context")
    session_module = importlib.import_module("agiwo.agent.runtime.session")
    state_writer_module = importlib.import_module("agiwo.agent.runtime.state_writer")
    state_ops_module = importlib.import_module("agiwo.agent.runtime.state_ops")

    assert hasattr(context_module, "RunContext")
    assert hasattr(context_module, "RunRuntime")
    assert hasattr(session_module, "SessionRuntime")
    assert hasattr(state_writer_module, "RunStateWriter")
    assert hasattr(state_ops_module, "track_step_state")
    assert hasattr(state_ops_module, "set_termination_reason")
    assert runtime_module.__all__ == [
        "RunContext",
        "RunRuntime",
        "RunStateWriter",
        "SessionRuntime",
    ]


def test_runtime_execution_shells_have_been_removed() -> None:
    removed_modules = (
        "agiwo.agent.runtime.run_engine",
        "agiwo.agent.runtime.step_committer",
        "agiwo.agent.runtime.hook_dispatcher",
    )

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None


def test_flat_conversion_modules_have_been_removed() -> None:
    removed_modules = (
        "agiwo.agent.input_codec",
        "agiwo.agent.message_adapters",
        "agiwo.agent.memory_hooks",
    )

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None
