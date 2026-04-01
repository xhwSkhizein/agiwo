from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentWorkspace:
    root: Path
    workspace: Path
    memory_dir: Path
    work_dir: Path
    identity_path: Path
    soul_path: Path
    user_path: Path


def build_agent_workspace(
    *,
    root_path: str | Path,
    agent_name: str,
) -> AgentWorkspace:
    root = Path(root_path).expanduser().resolve()
    workspace = root / agent_name
    work_dir = workspace / "WORK"
    return AgentWorkspace(
        root=root,
        workspace=workspace,
        memory_dir=workspace / "MEMORY",
        work_dir=work_dir,
        identity_path=workspace / "IDENTITY.md",
        soul_path=workspace / "SOUL.md",
        user_path=workspace / "USER.md",
    )
