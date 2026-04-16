# Release Publishing

## Prerequisites

Configure PyPI Trusted Publisher for both projects before the first release:

- `agiwo`
- `agiwo-console`

Point both PyPI projects at this GitHub repository and workflow:

- Repository: `xhwSkhizein/agiwo`
- Workflow file: `.github/workflows/release.yml`

Trusted Publisher is the intended publish path. The workflow does not require a long-lived PyPI API token in GitHub secrets.

## Tag Format

Create GitHub Releases with tags in this format:

- `v0.1.0`
- `v0.1.1`

Do not publish releases with bare tags like `0.1.0`.

## Publish Flow

1. Merge the release-ready changes into `main`.
2. Confirm the root `pyproject.toml` version matches the intended release version.
3. Confirm `console/pyproject.toml` uses the same version and still depends on the matching `agiwo` release line.
4. Create a GitHub Release with tag `vX.Y.Z`.
5. Publish the release.
6. Open the `Release Publish` workflow in GitHub Actions and confirm all jobs succeed.

The workflow will normalize the `v` tag, validate package metadata, build both packages, run the release smoke checks, publish `agiwo`, and then publish `agiwo-console`.

## Failure Handling

Publishing is not atomic.

- If `publish-sdk` fails, `agiwo-console` will not publish.
- If `publish-sdk` succeeds and `publish-console` fails, `agiwo` stays published.
- Fix the failure and rerun the workflow to publish the missing package.

The upload steps use duplicate-tolerant publishing so reruns can recover after a partial publish.
