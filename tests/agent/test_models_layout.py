import importlib
import importlib.util


def test_models_package_exposes_domain_specific_modules() -> None:
    config_module = importlib.import_module("agiwo.agent.models.config")
    input_module = importlib.import_module("agiwo.agent.models.input")
    run_module = importlib.import_module("agiwo.agent.models.run")
    step_module = importlib.import_module("agiwo.agent.models.step")
    stream_module = importlib.import_module("agiwo.agent.models.stream")

    assert hasattr(config_module, "AgentConfig")
    assert hasattr(input_module, "UserMessage")
    assert hasattr(input_module.UserMessage, "from_value")
    assert hasattr(input_module.UserMessage, "extract_text")
    assert hasattr(input_module.UserMessage, "serialize")
    assert hasattr(input_module.UserMessage, "deserialize")
    assert hasattr(input_module.UserMessage, "to_message_content")
    assert hasattr(run_module, "MemoryRecord")
    assert hasattr(run_module, "CompactMetadata")
    assert hasattr(run_module, "RunIdentity")
    assert hasattr(run_module, "RunLedger")
    assert hasattr(step_module, "StepView")
    assert hasattr(step_module.StepView, "to_message")
    assert hasattr(step_module, "LLMCallContext")
    assert hasattr(stream_module, "AgentStreamItem")


def test_flat_model_modules_have_been_removed() -> None:
    removed_modules = (
        "agiwo.agent.config",
        "agiwo.agent.input",
        "agiwo.agent.memory_types",
        "agiwo.agent.compact_types",
        "agiwo.agent.run_identity",
        "agiwo.agent.run_ledger",
        "agiwo.agent.records",
        "agiwo.agent.stream_events",
        "agiwo.agent.input_codec",
        "agiwo.agent.message_adapters",
    )

    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None
