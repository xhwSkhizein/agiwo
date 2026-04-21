"""Shared executor helpers for bash tools."""

from pathlib import Path

from agiwo.config.settings import settings
from agiwo.tool.builtin.bash_tool.local_executor import LocalExecutor

_SHARED_LOCAL_EXECUTORS: dict[str, LocalExecutor] = {}


def get_shared_local_executor(
    *,
    workspace_dir: str | Path | None = None,
    max_processes: int = 10,
) -> LocalExecutor:
    """Reuse one LocalExecutor per workspace so bash tools share job state."""
    # Resolve default workspace_dir here to avoid delayed evaluation of settings.root_path
    if workspace_dir is None:
        workspace_dir = settings.root_path

    key = "__default__"
    if workspace_dir is not None:
        key = str(Path(workspace_dir).resolve())

    executor = _SHARED_LOCAL_EXECUTORS.get(key)
    if executor is None:
        executor = LocalExecutor(
            workspace_dir=workspace_dir,
            max_processes=max_processes,
        )
        _SHARED_LOCAL_EXECUTORS[key] = executor
    return executor


__all__ = ["LocalExecutor", "get_shared_local_executor"]
