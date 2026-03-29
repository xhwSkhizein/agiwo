import shutil
from pathlib import Path

from agiwo.utils.logging import get_logger
from agiwo.workspace.layout import AgentWorkspace

logger = get_logger(__name__)

_TEMPLATE_FILENAMES = ("IDENTITY.md", "SOUL.md", "USER.md")
_TEMPLATES_DIR = Path(__file__).resolve().parents[2] / "templates"
if not _TEMPLATES_DIR.is_dir():
    logger.warning("templates_dir_missing", path=str(_TEMPLATES_DIR))


class WorkspaceBootstrapper:
    """Create workspace directories and seed template documents."""

    async def ensure_memory_ready(self, workspace: AgentWorkspace) -> None:
        workspace.memory_dir.mkdir(parents=True, exist_ok=True)

    async def ensure_prompt_ready(self, workspace: AgentWorkspace) -> None:
        workspace.memory_dir.mkdir(parents=True, exist_ok=True)
        workspace.instance_work_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_template_files(workspace)

    def _ensure_template_files(self, workspace: AgentWorkspace) -> None:
        for filename in _TEMPLATE_FILENAMES:
            target_path = workspace.workspace / filename
            if target_path.exists():
                continue

            source_path = _TEMPLATES_DIR / filename
            if not source_path.exists():
                logger.warning("template_file_not_found", filename=str(source_path))
                continue

            try:
                shutil.copy2(source_path, target_path)
                logger.info(
                    "copied_template_file",
                    source=str(source_path),
                    target=str(target_path),
                )
            except Exception as error:  # noqa: BLE001 - bootstrap boundary
                logger.warning(
                    "failed_to_copy_template",
                    filename=filename,
                    error=str(error),
                )
