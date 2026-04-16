# GitHub Release Publish Design

**Goal:** Let a maintainer publish both `agiwo` and `agiwo-console` directly from a GitHub Release, with release-triggered CI building, validating, and uploading both packages to PyPI.

## Scope

This design covers the first release automation path for the two Python packages already present in this repository:

1. `agiwo`
2. `agiwo-console`

The design adds a dedicated GitHub Actions workflow for release publishing and the minimum supporting documentation needed for maintainers to operate it safely.

This design does not cover:

- npm or frontend package publishing
- automatic version bumping
- changelog generation
- rollback automation for failed PyPI releases
- broader production-hardening outside the publish path

## Context

The repository already has CI checks that lint, test, build both packages, and run fresh-install smoke checks. That is enough to block many bad releases, but the actual publish step is still manual.

The maintainer requirement is to publish from GitHub itself by creating a Release, instead of running a separate local release script. The tag format is `v0.1.0` rather than `0.1.0`. The maintainer also accepts that publishing is not atomic: if `agiwo` succeeds and `agiwo-console` fails, the workflow should fail overall but retain any package that was already uploaded successfully.

## Recommended Approach

Add a separate `.github/workflows/release.yml` workflow triggered by `release.published`.

This workflow should:

1. extract the version from the GitHub Release tag
2. verify that the repository package versions match that release version
3. rebuild and smoke-test the SDK artifacts
4. publish `agiwo`
5. rebuild and smoke-test the Console artifacts
6. publish `agiwo-console`

This keeps release operations separate from ordinary CI and maps directly to the maintainer's desired GitHub Release workflow.

## Workflow Architecture

### Trigger

The publish workflow triggers only on:

- `release.published`

It should not trigger on:

- branch pushes
- pull requests
- plain tag pushes

This keeps regular validation and formal publishing clearly separated.

### Job Structure

The workflow should have three jobs:

1. `prepare`
2. `publish-sdk`
3. `publish-console`

`publish-sdk` depends on `prepare`.

`publish-console` depends on `publish-sdk`.

This dependency order is deliberate. `agiwo-console` depends on `agiwo`, so the console package should not be published before the SDK publish has succeeded.

### Prepare Job

`prepare` is responsible for:

- reading `github.event.release.tag_name`
- requiring a `v` prefix
- normalizing `v0.1.0` to `0.1.0`
- exposing the normalized version as a workflow output

If the tag does not follow the required `v<semver>` style, the workflow should fail immediately before any build or upload starts.

## Version Validation Rules

Every publish job must explicitly validate package metadata against the normalized release version.

### SDK Version Check

The root `pyproject.toml` version must exactly equal the normalized release version.

If the release tag is `v0.1.0`, then:

- `pyproject.toml` must contain `version = "0.1.0"`

If not, `publish-sdk` fails before building or uploading.

### Console Version Check

`console/pyproject.toml` must also exactly equal the normalized release version.

If the release tag is `v0.1.0`, then:

- `console/pyproject.toml` must contain `version = "0.1.0"`

If not, `publish-console` fails before building or uploading.

### Console Dependency Check

Because the console package is released together with the SDK, the console dependency on `agiwo` must remain aligned with the same release line.

For the `0.1.0` release line, the workflow should validate that the console dependency entry stays compatible with that same line, for example:

- `agiwo ~= 0.1.0`

This prevents releasing a console package whose dependency range can drift away from the paired SDK release.

## Build And Validation Flow

Each publish job should rebuild artifacts in CI instead of trusting checked-in local outputs.

### SDK Publish Flow

`publish-sdk` should:

1. check out the repository
2. install Python 3.11 and `uv`
3. run `uv sync`
4. validate the root package version against the release version
5. build the SDK package with `uv build`
6. run the existing smoke script against the built SDK wheel
7. upload SDK artifacts to PyPI using Trusted Publisher

### Console Publish Flow

`publish-console` should:

1. check out the repository
2. install Python 3.11 and `uv`
3. run `(cd console && uv sync)`
4. validate `console/pyproject.toml` version against the release version
5. validate the console dependency on `agiwo` matches the same release line
6. rebuild the SDK package as the install source for smoke verification
7. build the console package with `(cd console && uv build)`
8. run the existing smoke script against both wheels, including `agiwo-console --help`
9. upload console artifacts to PyPI using Trusted Publisher

The console smoke should continue to install both wheels into a clean environment to verify the released pairing instead of validating the console wheel in isolation.

## Publishing Credentials Model

The workflow should use PyPI Trusted Publisher rather than long-lived API tokens stored in GitHub secrets.

That implies:

- the workflow job that uploads to PyPI must request `id-token: write`
- each package upload step must target its own PyPI project
- PyPI must be configured to trust this repository and this workflow for both `agiwo` and `agiwo-console`

This is the preferred first-release setup because it reduces secret handling and aligns with current PyPI best practice.

## Failure Semantics

The release process is intentionally non-atomic.

Expected behavior:

- if `publish-sdk` fails, `publish-console` does not run
- if `publish-sdk` succeeds and `publish-console` fails, the overall workflow fails
- any package already uploaded to PyPI remains published
- the maintainer fixes the issue and reruns the workflow to publish the missing package

The workflow should not try to delete releases, delete tags, or remove already-published PyPI artifacts.

## Re-Run Behavior

Workflow re-runs must be safe when one package was already published successfully.

That means the upload action or upload command should tolerate already-existing files so a rerun can:

- skip already uploaded SDK artifacts
- continue toward the missing publish step

The goal is operational recovery, not strict transactional behavior.

## Documentation Changes

Release-facing maintainer docs should add a short operational section that covers:

1. required PyPI Trusted Publisher setup for both projects
2. the fact that publishing is triggered by creating a GitHub Release
3. the required tag format, for example `v0.1.0`
4. the expected partial-failure behavior

This documentation should stay short and procedural. It only needs to explain how the release automation works and what prerequisites must exist outside the repository.

## Security And Permissions

The release workflow should use the minimum permissions needed:

- `contents: read` for checkout
- `id-token: write` only for jobs that publish to PyPI

No repository write-back is required. The workflow should not create commits, mutate tags, or edit release notes.

## Alternatives Considered

### 1. Publish On Tag Push

This would trigger on `push` of `v*` tags.

Why not chosen:

- the maintainer explicitly wants to publish from GitHub Release
- tag pushes are easier to trigger accidentally
- Release as the publish event is more legible operationally

### 2. Add Publish Logic Into Existing CI

This would extend `.github/workflows/ci.yml` with conditional publish steps.

Why not chosen:

- it mixes ordinary validation with formal release behavior
- it makes the main CI workflow harder to understand and maintain
- publish-only permissions and release triggers deserve a separate workflow boundary

### 3. Publish SDK And Console In Parallel

This would reduce total workflow time.

Why not chosen:

- `agiwo-console` depends on `agiwo`
- sequential publish preserves a cleaner failure story for the first release

## Acceptance Criteria

The release automation is complete when all of the following are true:

1. Creating a GitHub Release with tag `vX.Y.Z` triggers a dedicated publish workflow.
2. The workflow extracts `X.Y.Z` and validates both package versions before any upload begins.
3. The workflow builds and smoke-tests `agiwo` before publishing it.
4. The workflow builds and smoke-tests `agiwo-console` against the paired SDK wheel before publishing it.
5. The workflow publishes with PyPI Trusted Publisher rather than stored API tokens.
6. The console publish path validates its dependency on `agiwo` stays aligned with the same release line.
7. A partial failure leaves already-published artifacts intact and surfaces a failing GitHub Actions run for maintainer follow-up.
8. Maintainer-facing release docs explain the required tag format and the required external PyPI setup.

## Implementation Notes

Keep this release automation small and explicit.

Reuse the existing release smoke script and existing package build commands instead of inventing a new release toolchain. The value here is moving the last manual publish step into GitHub Actions, not redesigning packaging.
