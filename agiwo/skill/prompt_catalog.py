from pathlib import Path
from typing import Protocol

from agiwo.skill.registry import SkillMetadata


class SkillPromptProvider(Protocol):
    async def initialize(self) -> None: ...

    async def refresh_if_changed(self) -> None: ...

    def render_skills_section(self, allowed_skills: list[str] | None = None) -> str: ...


class SkillPromptCatalog:
    """Render prompt-facing skill metadata and produce change fingerprints."""

    def render_section(self, metadata_items: list[SkillMetadata]) -> str:
        if not metadata_items:
            return ""

        lines = ["## Skills", ""]
        lines.append(
            "Skills are optional. Do not use one unless it is clearly helpful."
        )
        lines.append(
            "If you are unsure, call `skill.search` with the user's original request before activating any skill."
        )
        lines.append("")
        lines.append("<available_skills>")
        for metadata in metadata_items:
            lines.append("  <skill>")
            lines.append(f"    <name>{metadata.name}</name>")
            lines.append(f"    <description>{metadata.description}</description>")
            lines.append("  </skill>")
        lines.append("</available_skills>")
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
