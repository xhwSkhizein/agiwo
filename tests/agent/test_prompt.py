from pathlib import Path
from unittest.mock import patch

import pytest

from agiwo.agent.prompt import build_system_prompt
from agiwo.workspace.documents import WorkspaceDocumentStore
from agiwo.workspace.layout import build_agent_workspace


class NoopBootstrapper:
    async def ensure_prompt_ready(self, workspace) -> None:
        workspace.workspace.mkdir(parents=True, exist_ok=True)
        workspace.work_dir.mkdir(parents=True, exist_ok=True)
        workspace.memory_dir.mkdir(parents=True, exist_ok=True)


class StubSkillManager:
    async def initialize(self) -> None:
        return None

    async def refresh_if_changed(self) -> None:
        return None

    def render_skills_section(self, allowed_skills=None) -> str:
        assert allowed_skills == ["brainstorming"]
        return (
            "## Skills\n\n"
            "If you are unsure, call `skill.search` with the user's original request before activating any skill.\n\n"
            "<available_skills>\n"
            "  <skill>\n"
            "    <name>brainstorming</name>\n"
            "    <description>Explore design before implementation.</description>\n"
            "  </skill>\n"
            "</available_skills>"
        )


@pytest.mark.asyncio
async def test_build_system_prompt_uses_reduced_skill_section(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")

    with patch(
        "agiwo.agent.prompt.get_global_skill_manager",
        return_value=StubSkillManager(),
    ):
        prompt = await build_system_prompt(
            base_prompt="Base system prompt",
            workspace=workspace,
            tools=[],
            allowed_skills=["brainstorming"],
            bootstrapper=NoopBootstrapper(),
            document_store=WorkspaceDocumentStore(),
        )

    assert "brainstorming" in prompt
    assert "<location>" not in prompt
    assert "skill.search" in prompt


@pytest.mark.asyncio
async def test_build_system_prompt_skips_skill_section_when_disabled(
    tmp_path: Path,
) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")

    prompt = await build_system_prompt(
        base_prompt="Base system prompt",
        workspace=workspace,
        tools=[],
        allowed_skills=[],
        bootstrapper=NoopBootstrapper(),
        document_store=WorkspaceDocumentStore(),
    )

    assert "## Skills" not in prompt


@pytest.mark.asyncio
async def test_build_system_prompt_includes_tools_document(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    workspace.workspace.mkdir(parents=True)
    workspace.tools_path.write_text("# TOOLS.md\n\nUse Browser CLI for rendered pages.")

    prompt = await build_system_prompt(
        base_prompt="Base system prompt",
        workspace=workspace,
        tools=[],
        allowed_skills=[],
        bootstrapper=NoopBootstrapper(),
        document_store=WorkspaceDocumentStore(),
    )

    assert "# TOOLS.md" in prompt
    assert "Use Browser CLI for rendered pages." in prompt


def test_workspace_document_store_reads_tools_document(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    workspace.workspace.mkdir(parents=True)
    workspace.tools_path.write_text("Tool practice")

    documents = WorkspaceDocumentStore().read(workspace)

    assert documents.tools_text == "Tool practice"
    assert "TOOLS.md:" in documents.change_token
