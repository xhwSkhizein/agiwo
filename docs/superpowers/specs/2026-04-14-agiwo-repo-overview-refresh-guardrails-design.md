# Agiwo Repository Overview Refresh Guardrails Design

## Summary

The public repository overview page is intentionally generated manually so the resulting JSON can be reviewed before publishing. That keeps the process explicit, but it also leaves one failure mode open: the checked-in generated artifact can drift behind `README.md`, `AGENTS.md`, the generator rules, or the public architecture docs without CI noticing.

This design keeps manual generation as the source-of-truth workflow, while adding CI guardrails that fail when the generated repository overview artifact is stale. The goal is to preserve reviewability without letting outdated public content silently ship.

## Goals

- Keep repository overview generation manual and reviewable.
- Make CI fail when `website/src/generated/repo-overview.json` is out of date.
- Expand public-docs workflow triggers so relevant source changes re-run the check.
- Document the failure mode and the local refresh command clearly.

## Non-Goals

- Automatic regeneration on every push.
- Bots or workflows that commit generated output back to branches.
- A generic freshness checker for all docs pages.
- Replacing manual review of generated JSON diffs.

## Constraints And Assumptions

- The repository overview generator remains `scripts/generate_repo_overview.py`.
- The generated artifact remains committed to git at `website/src/generated/repo-overview.json`.
- Public Pages deploy continues to happen only from `main`.
- The repository already has a `Public Docs` GitHub Actions workflow that builds the Astro site.

## Recommended Approach

Add a deterministic generator freshness check to the existing `Public Docs` workflow, and expand its trigger paths to include the files that can change repo-overview output.

Why this approach:

- It preserves the current manual refresh workflow.
- It catches stale generated output before merge.
- It avoids workflow complexity around auto-commit, branch mutation, or regeneration races.
- It keeps the maintenance rule easy to understand: if source inputs changed, the generated JSON must be refreshed in the same branch.

Rejected alternatives:

- Automatic regeneration and commit: conflicts with the preferred review workflow.
- Separate dedicated workflow only for generator checking: unnecessary complexity for a small surface.
- No CI guardrail: too easy for public content drift to reappear.

## Trigger Scope

The `Public Docs` workflow should re-run when any repo-overview source of truth changes.

At minimum, the trigger paths should include:

- `website/**`
- `.github/workflows/public-docs.yml`
- `scripts/generate_repo_overview.py`
- `README.md`
- `AGENTS.md`
- `docs/architecture/**`
- `docs/guides/**`
- `docs/getting-started.md`

This keeps the build aligned with both presentation changes and generator input changes.

## CI Behavior

The build job should add a repo-overview freshness step before the Astro build.

Expected command:

```bash
python scripts/generate_repo_overview.py --check
```

Expected behavior:

- exit `0` when the committed JSON matches current generator output
- exit non-zero when the JSON is stale or missing

The workflow should fail immediately on a stale artifact instead of continuing to a misleading green docs build.

## Developer Workflow

The intended repository-overview maintenance loop remains:

1. update code, docs, or generator rules
2. run `python scripts/generate_repo_overview.py`
3. review the JSON diff
4. run the public site checks
5. commit both source changes and generated output together

The new CI check only enforces this workflow. It does not replace it.

## Documentation Changes

Update the public-site deployment / maintenance guide so contributors know:

- why CI can fail after editing repo-overview source inputs
- which command refreshes the generated artifact
- that generated JSON must be committed with the source change

The failure guidance should be concrete and short.

## Risks

- If trigger paths are too narrow, stale output can still slip through.
- If trigger paths are too broad, the workflow may run more often than necessary.
- If the CI failure message is unclear, contributors may not realize they need to regenerate the artifact locally.

## Success Criteria

- Editing repo-overview source inputs causes the `Public Docs` workflow to run.
- The workflow fails when `website/src/generated/repo-overview.json` is stale.
- Contributors can fix the failure by running one documented command locally.
- Manual regeneration and JSON diff review remain the normal publishing workflow.
