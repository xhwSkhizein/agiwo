from agiwo.tool.builtin import ensure_builtin_tools_loaded
from agiwo.tool.builtin.registry import BUILTIN_TOOLS, DEFAULT_TOOLS


def test_ensure_builtin_tools_loaded_is_idempotent() -> None:
    ensure_builtin_tools_loaded()
    builtin_count = len(BUILTIN_TOOLS)
    default_count = len(DEFAULT_TOOLS)

    ensure_builtin_tools_loaded()

    assert len(BUILTIN_TOOLS) == builtin_count
    assert len(DEFAULT_TOOLS) == default_count
