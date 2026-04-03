import pytest

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
    with pytest.raises(ValueError) as excinfo:
        expand_allowed_skills(["custom-skill"], ["skill-review"])
    assert "Unknown allowed skill(s): custom-skill" in str(excinfo.value)


def test_expand_allowed_skills_empty_patterns_deny_all() -> None:
    assert expand_allowed_skills([], ["skill-review"]) == []


def test_normalize_allowed_skills_rejects_non_list_input() -> None:
    with pytest.raises(ValueError) as excinfo:
        normalize_allowed_skills("skill-review")
    assert "list of strings" in str(excinfo.value)


def test_validate_expanded_allowed_skills_rejects_wildcard_patterns() -> None:
    assert contains_allowed_skill_patterns(["skill-review", "*audit"]) is True

    with pytest.raises(ValueError) as excinfo:
        validate_expanded_allowed_skills(["skill-review", "*audit"])
    assert "explicit skill names" in str(excinfo.value)


def test_matches_allowed_skill_rejects_malformed_wildcard_patterns() -> None:
    with pytest.raises(ValueError, match="Malformed wildcard pattern: foo\\*bar"):
        matches_allowed_skill("foo*bar", "foobar")


def test_expand_allowed_skills_rejects_malformed_wildcard_patterns() -> None:
    with pytest.raises(ValueError, match="Malformed wildcard pattern: foo\\*bar"):
        expand_allowed_skills(["foo*bar"], ["foobar"])
