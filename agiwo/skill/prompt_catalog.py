from pathlib import Path
from typing import Protocol

from agiwo.skill.registry import SkillMetadata


class SkillPromptProvider(Protocol):
    async def initialize(self) -> None: ...

    async def refresh_if_changed(self) -> None: ...

    def render_skills_section(self) -> str: ...


class SkillPromptCatalog:
    """Render prompt-facing skill metadata and produce change fingerprints."""

    def render_section(self, metadata_items: list[SkillMetadata]) -> str:
        if not metadata_items:
            return ""

        lines = ["## Available Skills"]
        lines.append("\n")
        lines.append(
            "Skills are tools. Use them quietly. The user doesn't need to see the machinery."
        )
        lines.append(
            "These skills are discovered at startup. Each entry includes a name and description. "
            "Use the Skill tool to activate it when needed."
        )
        lines.append("")
        lines.append("<avaliable_skills>")
        for metadata in metadata_items:
            lines.append("  <skill>")
            lines.append(f"    <name>{metadata.name}</name>")
            lines.append(f"    <description>{metadata.description}</description>")
            lines.append(f"    <location>{metadata.path}</location>")
            lines.append("  </skill>")
        lines.append("</avaliable_skills>")
        lines.append("")
        lines.append("### How to use skills:")
        lines.append(
            "1. When a user task matches a skill's description, use the Skill tool to activate it."
        )
        lines.append(
            "2. After activation, follow the instructions in the skill's SKILL.md file."
        )
        lines.append(
            "3. Load reference files (references/) only when needed for specific steps."
        )
        lines.append(
            "4. Execute scripts (scripts/) only when the skill instructions require it."
        )
        lines.append(
            "5. Use assets (assets/) as templates or resources, don't load their content."
        )
        return "\n".join(lines)

    def compute_change_token(self, skills_dirs: list[Path]) -> str:
        fingerprints: list[str] = []
        for skills_dir in skills_dirs:
            if not skills_dir.exists():
                continue
            for skill_path in skills_dir.iterdir():
                if not skill_path.is_dir():
                    continue
                skill_md = skill_path / "SKILL.md"
                if skill_md.exists():
                    fingerprints.append(f"{skill_path.name}:{skill_md.stat().st_mtime}")
        return "|".join(sorted(fingerprints))


__all__ = [
    "SkillPromptCatalog",
    "SkillPromptProvider",
]
