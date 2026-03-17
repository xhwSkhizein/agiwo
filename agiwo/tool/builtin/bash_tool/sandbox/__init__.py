"""Shared sandbox helpers for bash tools."""

from pathlib import Path

from agiwo.config.settings import settings
from agiwo.tool.builtin.bash_tool.sandbox.local import LocalSandbox

_SHARED_LOCAL_SANDBOXES: dict[str, LocalSandbox] = {}


def get_shared_local_sandbox(
    *,
    workspace_dir: str | Path = settings.root_path,
    max_processes: int = 10,
) -> LocalSandbox:
    """Reuse one LocalSandbox per workspace so bash tools share job state."""
    key = "__default__"
    if workspace_dir is not None:
        key = str(Path(workspace_dir).resolve())

    sandbox = _SHARED_LOCAL_SANDBOXES.get(key)
    if sandbox is None:
        sandbox = LocalSandbox(
            workspace_dir=workspace_dir,
            max_processes=max_processes,
        )
        _SHARED_LOCAL_SANDBOXES[key] = sandbox
    return sandbox


__all__ = ["LocalSandbox", "get_shared_local_sandbox"]
