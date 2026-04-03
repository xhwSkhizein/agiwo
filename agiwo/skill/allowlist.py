def normalize_allowed_skills(
    patterns: object,
) -> tuple[str, ...] | None:
    if patterns is None:
        return None
    if isinstance(patterns, str) or not isinstance(patterns, (list, tuple)):
        raise ValueError("allowed_skills must be a list of strings")

    normalized: list[str] = []
    for pattern in patterns:
        if not isinstance(pattern, str):
            raise ValueError("allowed_skills must be a list of strings")
        stripped = pattern.strip()
        if stripped and stripped not in normalized:
            normalized.append(stripped)
    return tuple(normalized)


def skills_enabled(allowed_skills: list[str] | tuple[str, ...] | None) -> bool:
    normalized = normalize_allowed_skills(allowed_skills)
    return normalized is None or bool(normalized)


def _validate_allowed_skill_pattern(pattern: str) -> None:
    if "*" not in pattern:
        return
    if pattern == "*":
        return
    if pattern.count("*") != 1:
        raise ValueError(f"Malformed wildcard pattern: {pattern}")
    if pattern.startswith("*") ^ pattern.endswith("*"):
        return
    raise ValueError(f"Malformed wildcard pattern: {pattern}")


def contains_allowed_skill_patterns(
    skills: list[str] | tuple[str, ...] | None,
) -> bool:
    normalized = normalize_allowed_skills(skills)
    if normalized is None:
        return False
    for skill in normalized:
        _validate_allowed_skill_pattern(skill)
    return any("*" in skill for skill in normalized)


def validate_expanded_allowed_skills(
    skills: list[str] | tuple[str, ...] | None,
) -> None:
    normalized = normalize_allowed_skills(skills)
    if normalized is None:
        return
    patterns = [skill for skill in normalized if "*" in skill]
    if patterns:
        pattern_list = ", ".join(sorted(patterns))
        raise ValueError(
            "allowed_skills must be expanded to explicit skill names before "
            f"entering runtime data models: {pattern_list}"
        )


def validate_known_allowed_skills(
    skills: list[str] | tuple[str, ...] | None,
    available_skill_names: list[str] | tuple[str, ...],
) -> None:
    normalized = normalize_allowed_skills(skills)
    if normalized is None:
        return
    available = normalize_allowed_skills(available_skill_names) or ()
    available_set = set(available)
    missing = [skill for skill in normalized if skill not in available_set]
    if missing:
        available_text = ", ".join(available) if available else "(none)"
        missing_text = ", ".join(missing)
        raise ValueError(
            f"Unknown allowed skill(s): {missing_text}. Available skills: "
            f"{available_text}"
        )


def matches_allowed_skill(pattern: str, skill_name: str) -> bool:
    _validate_allowed_skill_pattern(pattern)
    if pattern == "*":
        return True
    if pattern.endswith("*") and not pattern.startswith("*"):
        return skill_name.startswith(pattern[:-1])
    if pattern.startswith("*") and not pattern.endswith("*"):
        return skill_name.endswith(pattern[1:])
    return pattern == skill_name


def expand_allowed_skills(
    patterns: object,
    available_skill_names: object,
) -> list[str] | None:
    normalized = normalize_allowed_skills(patterns)
    if normalized is None:
        return None
    if not normalized:
        return []
    available = normalize_allowed_skills(available_skill_names) or ()
    validate_known_allowed_skills(
        [pattern for pattern in normalized if "*" not in pattern],
        available,
    )

    expanded: list[str] = []
    for pattern in normalized:
        _validate_allowed_skill_pattern(pattern)
        if "*" not in pattern:
            if pattern not in expanded:
                expanded.append(pattern)
            continue
        for skill_name in available:
            if (
                matches_allowed_skill(pattern, skill_name)
                and skill_name not in expanded
            ):
                expanded.append(skill_name)
    return expanded


__all__ = [
    "contains_allowed_skill_patterns",
    "expand_allowed_skills",
    "matches_allowed_skill",
    "normalize_allowed_skills",
    "skills_enabled",
    "validate_expanded_allowed_skills",
    "validate_known_allowed_skills",
]
