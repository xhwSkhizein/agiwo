from agiwo.skill.allowlist import (
    contains_allowed_skill_patterns,
    expand_allowed_skills,
    matches_allowed_skill,
    normalize_allowed_skills,
    validate_expanded_allowed_skills,
)


def test_matches_allowed_skill_supports_exact_prefix_suffix_and_all() -> None:
    assert matches_allowed_skill("audit", "audit") is True
    assert matches_allowed_skill("skill*", "skill-review") is True
    assert matches_allowed_skill("*review", "skill-review") is True
    assert matches_allowed_skill("*", "anything") is True
    assert matches_allowed_skill("skill*", "review-skill") is False


def test_expand_allowed_skills_supports_wildcards() -> None:
    expanded = expand_allowed_skills(
        ["skill*", "*review"],
        ["skill-review", "skill-build", "code-review"],
    )

    assert expanded == ["skill-review", "skill-build", "code-review"]


def test_expand_allowed_skills_preserves_exact_names() -> None:
    expanded = expand_allowed_skills(
        ["custom-skill", "skill*"],
        ["custom-skill", "skill-review"],
    )

    assert expanded == ["custom-skill", "skill-review"]


def test_expand_allowed_skills_rejects_unknown_exact_names() -> None:
    try:
        expand_allowed_skills(["custom-skill"], ["skill-review"])
    except ValueError as exc:
        assert "Unknown allowed skill(s): custom-skill" in str(exc)
    else:
        raise AssertionError("expected unknown exact skill failure")


def test_expand_allowed_skills_empty_patterns_deny_all() -> None:
    assert expand_allowed_skills([], ["skill-review"]) == []


def test_normalize_allowed_skills_rejects_non_list_input() -> None:
    try:
        normalize_allowed_skills("skill-review")
    except ValueError as exc:
        assert "list of strings" in str(exc)
    else:
        raise AssertionError("expected invalid allowed_skills type failure")


def test_validate_expanded_allowed_skills_rejects_wildcard_patterns() -> None:
    assert contains_allowed_skill_patterns(["skill-review", "*audit"]) is True

    try:
        validate_expanded_allowed_skills(["skill-review", "*audit"])
    except ValueError as exc:
        assert "explicit skill names" in str(exc)
    else:
        raise AssertionError("expected wildcard validation failure")
