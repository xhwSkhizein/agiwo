from pathlib import Path


def test_pre_push_branch_prefixes_allow_short_aliases() -> None:
    hook = Path(".githooks/pre-push").read_text(encoding="utf-8")

    assert "feature/" in hook
    assert "feat/" in hook
    assert "bugfix/" in hook
    assert "fix/" in hook
