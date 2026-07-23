"""AB-03: MainAgent must not depend on Scheduler Session facade APIs."""

import inspect

import agiwo.agent.main_agent as main_agent_module
import agiwo.scheduler.engine as engine_module


def test_main_agent_module_has_no_scheduler_session_facade_imports() -> None:
    source = inspect.getsource(main_agent_module)
    forbidden = (
        "route_root_input",
        "dispatch_execution",
        "inject_user_message",
        "from agiwo.scheduler",
        "import agiwo.scheduler",
    )
    for token in forbidden:
        assert token not in source, f"MainAgent must not reference {token!r}"


def test_scheduler_dropped_session_facade_methods() -> None:
    source = inspect.getsource(engine_module)
    for token in (
        "route_root_input",
        "dispatch_execution",
        "inject_user_message",
        "async def steer",
    ):
        assert token not in source

    assert not hasattr(engine_module.Scheduler, "steer")
    assert not hasattr(engine_module.Scheduler, "route_root_input")
    assert not hasattr(engine_module.Scheduler, "dispatch_execution")
    assert not hasattr(engine_module.Scheduler, "inject_user_message")
