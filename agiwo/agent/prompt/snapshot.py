from dataclasses import dataclass, field

from agiwo.workspace.documents import WorkspaceDocuments
from agiwo.workspace.layout import AgentWorkspace


@dataclass(frozen=True)
class ToolPromptSummary:
    name: str
    description: str


@dataclass(frozen=True)
class EnvironmentSnapshot:
    os_info: str
    language_info: str | None
    timezone: str
    current_date: str


@dataclass(frozen=True)
class PromptSnapshot:
    base_prompt: str
    workspace: AgentWorkspace
    documents: WorkspaceDocuments
    environment: EnvironmentSnapshot
    tools: list[ToolPromptSummary] = field(default_factory=list)
    skills_section: str = ""
    change_token: str = ""
