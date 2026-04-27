from dataclasses import dataclass
from pathlib import Path

from agiwo.utils.logging import get_logger
from agiwo.workspace.layout import AgentWorkspace

logger = get_logger(__name__)


@dataclass(frozen=True)
class WorkspaceDocuments:
    identity_text: str | None
    soul_text: str | None
    user_text: str | None
    tools_text: str | None
    change_token: str


class WorkspaceDocumentStore:
    """Read workspace-facing prompt documents and expose a change token."""

    def read(self, workspace: AgentWorkspace) -> WorkspaceDocuments:
        identity = self._read_optional(workspace.identity_path, "IDENTITY.md")
        soul = self._read_optional(workspace.soul_path, "SOUL.md")
        user = self._read_optional(workspace.user_path, "USER.md")
        tools = self._read_optional(workspace.tools_path, "TOOLS.md")
        return WorkspaceDocuments(
            identity_text=identity,
            soul_text=soul,
            user_text=user,
            tools_text=tools,
            change_token=self._build_change_token(
                workspace.identity_path,
                workspace.soul_path,
                workspace.user_path,
                workspace.tools_path,
            ),
        )

    def _read_optional(self, path: Path, label: str) -> str | None:
        if not path.exists():
            return None
        content = path.read_text()
        logger.info("loaded_workspace_document", label=label, path=str(path))
        return content

    def _build_change_token(self, *paths: Path) -> str:
        parts: list[str] = []
        for path in paths:
            if not path.exists():
                parts.append(f"{path.name}:missing")
                continue
            parts.append(f"{path.name}:{path.stat().st_mtime}")
        return "|".join(parts)
