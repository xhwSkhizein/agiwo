"""Architecture contracts for agiwo.objective public surface and import direction."""

import ast
from pathlib import Path

import agiwo.objective as objective_pkg


ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_DIR = ROOT / "agiwo" / "objective"
AGENT_DIR = ROOT / "agiwo" / "agent"
SCHEDULER_DIR = ROOT / "agiwo" / "scheduler"


def _imports_from(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


def _py_files(directory: Path) -> list[Path]:
    return [p for p in directory.rglob("*.py") if p.is_file()]


def test_public_export_surface() -> None:
    public = set(objective_pkg.__all__)
    required = {
        "ObjectiveService",
        "ObjectiveStore",
        "ObjectiveView",
        "ObjectiveMetrics",
        "project_objective_metrics",
        "create_objective_store",
        "CreateObjectiveRequest",
        "CommandResult",
        "ObjectiveStatus",
    }
    assert required.issubset(public)
    forbidden = {
        "ObjectiveLogEntry",
        "DispatchRequested",
        "project_objective",
        "InMemoryObjectiveStore",
        "SQLiteObjectiveStore",
        "ObjectiveFactKind",
    }
    assert public.isdisjoint(forbidden)


def test_agent_and_scheduler_do_not_import_objective() -> None:
    offenders: list[str] = []
    for directory in (AGENT_DIR, SCHEDULER_DIR):
        for path in _py_files(directory):
            for mod in _imports_from(path):
                if mod == "agiwo.objective" or mod.startswith("agiwo.objective."):
                    offenders.append(f"{path.relative_to(ROOT)} -> {mod}")
    assert offenders == []


def test_objective_does_not_import_agent_internals() -> None:
    # P2 allows objective -> scheduler facade; agent internals stay forbidden.
    forbidden_prefixes = (
        "agiwo.agent.storage",
        "agiwo.agent.run_loop",
        "agiwo.agent.runtime",
        "agiwo.agent.tool_executor",
        "agiwo.agent.compaction",
        "agiwo.agent.llm_caller",
        "agiwo.agent.prompt",
    )
    offenders: list[str] = []
    for path in _py_files(OBJECTIVE_DIR):
        for mod in _imports_from(path):
            if any(mod == p or mod.startswith(p + ".") for p in forbidden_prefixes):
                offenders.append(f"{path.relative_to(ROOT)} -> {mod}")
    assert offenders == []


def test_objective_does_not_call_route_root_input() -> None:
    """Objective paths must use the narrow Scheduler facade only."""
    offenders: list[str] = []
    for path in _py_files(OBJECTIVE_DIR):
        text = path.read_text(encoding="utf-8")
        if "route_root_input(" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_objective_scheduler_calls_use_narrow_facade() -> None:
    """Dispatch/wait/inject/pause-resume/tree — no mailbox routing."""
    allowed = {
        "dispatch_execution",
        "wait_for",
        "inject_user_message",
        "request_recoverable_pause",
        "prepare_resume",
        "release_resume_barrier",
        "list_execution_tree",
        "get_run_view",
        "get_run_status",
    }
    scheduler_methods: set[str] = set()
    for path in _py_files(OBJECTIVE_DIR):
        text = path.read_text(encoding="utf-8")
        for method in allowed:
            if f".{method}(" in text or f"self._scheduler.{method}(" in text:
                scheduler_methods.add(method)
    assert {"dispatch_execution", "wait_for"}.issubset(scheduler_methods)


def test_console_server_does_not_import_objective_store_backends() -> None:
    console_server = ROOT / "console" / "server"
    offenders: list[str] = []
    for path in _py_files(console_server):
        rel = path.relative_to(ROOT)
        if rel.as_posix() == "console/server/services/storage_wiring.py":
            continue
        for mod in _imports_from(path):
            if mod.startswith("agiwo.objective.store"):
                offenders.append(f"{rel} -> {mod}")
    assert offenders == []


def test_ordinary_console_paths_do_not_call_route_root_input() -> None:
    """Sessions/channels must use ObjectiveGateway; debug scheduler may still route."""
    allow = {
        ROOT / "console/server/routers/scheduler.py",
        ROOT / "console/server/services/runtime/session_runtime_service.py",
    }
    offenders: list[str] = []
    for path in _py_files(ROOT / "console" / "server"):
        if path in allow:
            continue
        if "services/runtime" in path.as_posix():
            continue
        text = path.read_text(encoding="utf-8")
        if "route_root_input(" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
