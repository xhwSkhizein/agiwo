# Python 3.10 Support Design

**Goal:** Lower the minimum supported Python version for both `agiwo` and `agiwo-console` from 3.11 to 3.10, and make the repository's code, packaging metadata, CI, and release flow consistently enforce that support level.

## Scope

This design covers the current two published Python packages in this repository:

1. `agiwo`
2. `agiwo-console`

The change includes:

- package metadata and install requirements
- runtime and script compatibility fixes for Python 3.10
- CI and release workflow updates so 3.10 is actually validated
- smoke-install verification on 3.10
- user-facing and maintainer-facing documentation updates

This design does not cover:

- adding support for Python 3.9 or lower
- broad dependency upgrades unrelated to Python 3.10 support
- frontend or npm runtime compatibility work
- unrelated refactoring triggered only by style preference

## Context

The repository currently declares Python 3.11+ in package metadata and docs. Most of the codebase already uses syntax that is compatible with Python 3.10, such as builtin generics and `X | None` unions. The main risk is not widespread syntax breakage, but a smaller set of 3.11-only runtime features and release-path assumptions:

- root and console `pyproject.toml` files declare `requires-python = ">=3.11"`
- some scripts rely on `tomllib`, which is only in the standard library starting from 3.11
- at least one runtime path uses `asyncio.timeout`, which is also 3.11+
- repository guardrails and docs still describe 3.11+ as the project standard
- CI and release workflows only install and validate Python 3.11, so current automation cannot prove 3.10 support

The practical requirement is not just "make local code run on 3.10". The repository must also stop publishing metadata and running automation that contradict that claim.

## Recommended Approach

Adopt Python 3.10 as the minimum supported version across both packages, and align every relevant layer to that baseline:

1. lower declared minimum Python version to 3.10
2. replace confirmed 3.11-only APIs with 3.10-compatible implementations
3. adjust repository guardrails and docs to describe 3.10+ as the standard
4. run CI and release smoke validation on Python 3.10 so the published support promise is enforced continuously

This is intentionally narrower than a full multi-version matrix rollout. The immediate problem is that the published minimum version is too high. The first fix is to make 3.10 the validated floor everywhere that matters.

## Compatibility Changes

### Runtime Code

Any runtime usage of 3.11-only standard-library APIs should be replaced with 3.10-safe equivalents while preserving behavior.

The currently confirmed case is `asyncio.timeout`. The replacement should:

- keep the existing per-chunk timeout behavior for streaming LLM responses
- raise a clear timeout error message when the stream stalls
- avoid hidden behavior changes in cancellation or exception handling

The implementation should prefer a small local compatibility pattern over adding a new abstraction layer unless more than one call site needs the same behavior.

### Tooling Scripts

Scripts that need to run in CI or release workflows must also work on Python 3.10.

The currently confirmed case is `scripts/check_release_metadata.py`, which imports `tomllib`. That script should gain a 3.10-safe path, preferably by:

- using `tomllib` when available
- falling back to a lightweight dependency or compatible parser only when running on Python 3.10

The fallback should stay limited to tooling paths. It should not drag extra compatibility dependencies into unrelated runtime code.

### Non-Issues

The repository already uses many modern annotations such as `list[str]` and `str | None`. Those remain valid in Python 3.10 and should not be rewritten to older `typing.List` or `Optional[...]` forms. The goal is real compatibility, not stylistic regression.

## Packaging And Repository Metadata

Both published packages should declare Python 3.10 as the minimum supported version:

- root `pyproject.toml`
- `console/pyproject.toml`

Related repository metadata should also align:

- Ruff target version should move from `py311` to `py310`
- any repository guardrails that explicitly describe 3.11+ as the project standard should be updated to 3.10+
- if lockfiles are committed and encode a stricter Python floor, they should be refreshed so local and CI workflows do not keep forcing 3.11

The rule is simple: there should be no first-party metadata that simultaneously says "3.10 supported" and "3.11 required".

## CI And Release Validation

### CI

The ordinary CI workflow should validate the minimum supported version by running Python 3.10 for the existing backend checks:

- lint
- SDK tests
- Console tests
- SDK package build and smoke install
- Console package build and smoke install

This keeps the repository honest: any new use of a 3.11-only feature will fail in ordinary pull-request validation.

The existing web job does not need Python-version changes beyond any incidental workflow consistency.

### Release Workflow

The GitHub release workflow should also build and smoke-test artifacts on Python 3.10 before publishing them.

That includes:

- applying release metadata
- syncing dependencies
- building both wheels
- smoke-installing the SDK wheel
- smoke-installing the paired SDK + console wheels

This matters because the published wheel metadata and the final install path must both reflect the same minimum version claim.

## Documentation Changes

User-facing and maintainer-facing documentation should be updated anywhere the repository currently promises or demonstrates Python 3.11+ only, including:

- root `README.md`
- getting-started documentation
- release documentation
- any build/test command sections that describe 3.11 as a hard prerequisite

The docs should consistently describe the supported floor as Python 3.10+.

Historical design notes or archived planning documents do not need mass editing unless they are used as current operational instructions. The focus is live documentation, not retroactive cleanup of every historical note.

## Testing And Validation

Acceptance should be based on the minimum-version path, not just developer-local success on a newer interpreter.

Required validation:

1. repository lint and guardrails pass with the updated 3.10 baseline
2. SDK tests pass on Python 3.10
3. Console backend tests pass on Python 3.10
4. SDK wheel builds and installs in a clean Python 3.10 environment
5. Console wheel builds and installs with the paired SDK wheel in a clean Python 3.10 environment

If any dependency turns out not to support Python 3.10 despite current assumptions, that is a release blocker. In that case the repository should not claim 3.10 support until the dependency choice or version constraint is corrected.

## Risks

### Dependency Floor Drift

Some third-party dependencies may have dropped Python 3.10 support even if the first-party code is compatible. This risk is highest in packaging and smoke-install steps, so build-time and install-time validation must be treated as first-class checks.

### Tooling Drift

CI may pass locally if developers use Python 3.11 by default, while release or fresh-install paths fail on 3.10. That is why the automation itself must run on 3.10, not just the package metadata.

### Hidden Runtime Paths

There may be a small number of additional 3.11-only APIs not yet surfaced by static search. Running the test suite and smoke-install flow under 3.10 is the backstop for catching these.

## Alternatives Considered

### 1. Metadata-Only Downgrade

Lower `requires-python` to 3.10 and patch only the known code breakages.

Why not chosen:

- it would still leave CI and release paths validating only 3.11
- the repository could publish a support claim that is not continuously enforced

### 2. Full 3.10 And 3.11 Test Matrix Immediately

Run all Python backend jobs on both 3.10 and 3.11 right now.

Why not chosen for this change:

- it is a larger operational change than the current release goal requires
- the immediate need is to make 3.10 the trusted floor, not to expand version coverage broadly

This can be added later once the minimum-version path is stable.

## Implementation Outline

The implementation should proceed in this order:

1. update metadata and repository guardrails to express a 3.10 baseline
2. patch confirmed 3.11-only runtime and script code paths
3. update CI and release workflows to run the backend path on Python 3.10
4. refresh live documentation to describe 3.10+ correctly
5. run lint, tests, and smoke-install verification on Python 3.10

This order keeps the declared baseline, actual code behavior, and automation aligned throughout the change.
