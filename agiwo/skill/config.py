from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SkillDiscoveryConfig:
    configured_dirs: list[str] | None
    env_dirs: list[str]
    root_path: str


def normalize_skill_dirs(value: str | list[str] | None) -> list[str] | None:
    if value is None:
        return None

    entries = [value] if isinstance(value, str) else value
    normalized: list[str] = []
    for entry in entries:
        if not isinstance(entry, str):
            continue
        stripped = entry.strip()
        if stripped:
            normalized.append(stripped)
    return normalized or None


def resolve_relative_path(path: str, root_path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    root = Path(root_path).expanduser().resolve()
    return (root / candidate).resolve()


def resolve_skill_dirs(config: SkillDiscoveryConfig) -> list[Path]:
    resolved: list[Path] = []
    seen: set[str] = set()
    configured_dirs = config.configured_dirs
    if configured_dirs:
        entries = list(configured_dirs)
    else:
        default_dir = resolve_relative_path("skills", config.root_path)
        entries = [str(default_dir)] if default_dir.exists() else []
    entries.extend(config.env_dirs)

    for entry in entries:
        path = resolve_relative_path(entry, config.root_path)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


__all__ = [
    "SkillDiscoveryConfig",
    "normalize_skill_dirs",
    "resolve_relative_path",
    "resolve_skill_dirs",
]
