from datetime import datetime

from agiwo.agent.prompt.sections import (
    get_language_info,
    get_os_info,
    render_prompt,
)
from agiwo.agent.prompt.snapshot import (
    EnvironmentSnapshot,
    PromptSnapshot,
    ToolPromptSummary,
)
from agiwo.skill.prompt_catalog import SkillPromptProvider
from agiwo.utils.logging import get_logger
from agiwo.workspace import WorkspaceBootstrapper, WorkspaceDocumentStore
from agiwo.workspace.layout import AgentWorkspace

logger = get_logger(__name__)


class AgentPromptRuntime:
    """Build and cache the agent system prompt from prepared snapshots."""

    def __init__(
        self,
        *,
        base_prompt: str,
        workspace: AgentWorkspace,
        tools: list[object] | None = None,
        skill_manager: SkillPromptProvider | None = None,
        bootstrapper: WorkspaceBootstrapper | None = None,
        document_store: WorkspaceDocumentStore | None = None,
    ) -> None:
        self._base_prompt = base_prompt
        self._workspace = workspace
        self._tools = tools or []
        self._skill_manager = skill_manager
        self._bootstrapper = bootstrapper or WorkspaceBootstrapper()
        self._document_store = document_store or WorkspaceDocumentStore()
        self._initialized = False
        self._system_prompt: str | None = None
        self._change_token: str | None = None

    async def get_system_prompt(self) -> str:
        if not self._initialized:
            await self._bootstrapper.ensure_prompt_ready(self._workspace)
            if self._skill_manager is not None:
                await self._skill_manager.initialize()
            self._initialized = True
        elif self._skill_manager is not None:
            await self._skill_manager.refresh_if_changed()

        snapshot = self._build_snapshot()
        if snapshot.change_token != self._change_token:
            self._system_prompt = render_prompt(snapshot)
            self._change_token = snapshot.change_token
            logger.info("prompt_refreshed", change_token=snapshot.change_token)

        return self._system_prompt or ""

    def _build_snapshot(self) -> PromptSnapshot:
        documents = self._document_store.read(self._workspace)
        current_dt = datetime.now().astimezone()
        skills_section = ""
        skills_token = ""
        if self._skill_manager is not None:
            skills_snapshot = self._skill_manager.get_prompt_snapshot()
            skills_section = skills_snapshot.rendered_section
            skills_token = skills_snapshot.change_token

        tool_summaries = [
            ToolPromptSummary(
                name=tool.get_name(),
                description=tool.get_short_description(),
            )
            for tool in self._tools
        ]
        environment = EnvironmentSnapshot(
            os_info=get_os_info(),
            language_info=get_language_info(),
            timezone=str(current_dt.tzinfo),
            current_date=current_dt.strftime("%Y-%m-%d"),
        )
        tool_token = "|".join(tool.name for tool in tool_summaries)
        change_token = "|".join(
            [
                documents.change_token,
                skills_token,
                self._base_prompt,
                tool_token,
            ]
        )
        return PromptSnapshot(
            base_prompt=self._base_prompt,
            workspace=self._workspace,
            documents=documents,
            environment=environment,
            tools=tool_summaries,
            skills_section=skills_section,
            change_token=change_token,
        )
