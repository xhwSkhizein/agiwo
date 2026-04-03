from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SkillDiscoveryConfig:
    skills_dirs: list[str] = field(default_factory=list)
    root_path: str = ""


def resolve_relative_path(path: str, root_path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(root_path).expanduser().resolve()
    return (root / candidate).resolve()


def resolve_skill_dirs(config: SkillDiscoveryConfig) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    for entry in config.skills_dirs:
        stripped = entry.strip()
        if not stripped:
            continue
        path = resolve_relative_path(stripped, config.root_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


__all__ = [
    "SkillDiscoveryConfig",
    "resolve_relative_path",
    "resolve_skill_dirs",
]
