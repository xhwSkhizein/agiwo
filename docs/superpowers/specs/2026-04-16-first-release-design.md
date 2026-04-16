# First Release Design

**Date:** 2026-04-16

## Goal

Prepare the repository for the first public release of both `agiwo` and `agiwo-console` with a simpler install story, accurate public docs, and release gates that catch broken packaging and broken quick-start paths before publish.

## Scope

This design covers four release-facing areas only:

1. Python package dependency and packaging layout for `agiwo` and `agiwo-console`
2. Public documentation and examples, with SDK-first positioning
3. Version surface alignment across SDK, Console backend, and Console web
4. CI checks required to block a bad release

This design does not include broader runtime refactors, API redesign, auth, or production-hardening work outside the minimum positioning and release-safety changes described below.

## Context

The current repository is close to release-ready from a testing perspective, but the publish surface still has several first-use failures:

- `pip install agiwo` does not match the documented user experience because public imports can fail when optional providers are missing
- README quick-start examples are not copy-paste safe
- builtin tool docs do not match the effective install/runtime surface
- CI does not exercise package build, clean-install smoke, or Console web verification
- Console messaging is currently too strong for its actual readiness level

For a first release, reducing surface complexity is more valuable than preserving a finely segmented extras model.

## Design Summary

### 1. Default install model

`agiwo` will install with the full current runtime dependency set by default. The package should no longer depend on optional extras for providers or web capabilities to make the documented primary path work.

Implications:

- Core dependencies in the root `pyproject.toml` include the currently used provider and web dependencies needed by the published SDK surface
- Existing extras may remain only where they still add real value, but the install story and docs no longer rely on them
- `agiwo-console` depends on `agiwo` directly rather than `agiwo[all]`

Rationale:

- This matches the maintainer decision to optimize for lower release complexity
- It removes the current mismatch between public imports and install instructions
- It keeps first-time installation predictable at the cost of a heavier wheel, which is acceptable for a `0.1.0`

### 2. Public SDK examples

Public docs should prefer the simplest constructor form that is valid in the released package. The canonical example becomes:

```python
from agiwo.llm import OpenAIModel

model = OpenAIModel(name="gpt-5.4")
```

The documentation should consistently prefer this style unless a page is explicitly teaching lower-level provider details. When an example needs a concrete `id`, it must still be internally consistent with the actual constructor contract.

Implications:

- README, getting-started, and public model/tool concept docs must be updated together
- Release-facing docs should avoid examples that imply optional install steps are required for the default SDK path

### 3. Console positioning

README and other public entry docs should continue to present the SDK as the main product. Console messaging should be reduced to a compact, accurate positioning:

- Console is a directly usable control-plane backend with a bundled web UI
- Current channel integration is limited to Feishu
- Recommended deployment model is internal/self-hosted use
- It is not yet production-ready

This is a messaging change, not a promise of new runtime constraints. The intent is to keep the release honest without removing the Console package from the release.

### 4. Version surface alignment

The release version must be represented consistently in:

- root `pyproject.toml`
- `console/pyproject.toml`
- FastAPI app version string
- Console web visible version label
- changelog release entry

The release version remains `0.1.0` unless changed separately, but all user-visible version surfaces must derive from or at least match the same value.

For this release, alignment is sufficient; introducing a new cross-language version generation system is optional and should only be done if it stays small.

### 5. Release gate in CI

CI must validate the same surfaces that are most likely to break the first release:

- Python lint
- Python test suites
- SDK package build
- Console package build
- clean-install smoke for the published SDK wheel
- Console web lint
- Console web tests
- Console web production build

The clean-install smoke should verify at least one documented public import path and one minimal constructor path from a fresh environment built from the wheel artifact, not the editable checkout.

## Implementation Shape

### Packaging

Root package changes are expected in:

- dependency declarations
- optional dependency cleanup
- any docs or tests that refer to extras-based installation

Console package changes are expected in:

- dependency declaration on `agiwo`
- release metadata consistency

### Documentation

Primary public docs to update:

- `README.md`
- `docs/getting-started.md`
- `docs/concepts/model.md`
- `docs/concepts/tool.md`
- any other release-facing page that still demonstrates the invalid `OpenAIModel(id=...)` quick-start or overstates Console readiness

Docs under internal planning or historical architecture notes do not need broad cleanup unless they directly conflict with the release entry surface.

### Verification

Verification should be split into:

- normal repo lint and tests
- package build checks
- fresh-environment smoke checks for built wheels
- Console web lint/test/build

## Error Handling and Failure Modes

### Install/runtime mismatch

Risk:
Public imports succeed in editable development but fail from an installed wheel.

Mitigation:
Require clean-install smoke in CI from built wheel artifacts.

### Doc drift

Risk:
README and docs point to constructors or install commands that no longer reflect shipped behavior.

Mitigation:
Update the public docs set together and include at least one smoke check that mirrors README usage.

### Console expectation gap

Risk:
Users interpret the Console package as production-ready because the README presents it as a flagship feature.

Mitigation:
Reduce README emphasis and add explicit internal-use / not-production-ready language.

## Testing Strategy

The release work is complete when all of the following pass:

- existing SDK lint and test commands
- existing Console backend lint and test commands
- `uv build` at repo root
- `uv build` in `console/`
- fresh-wheel smoke import and minimal model construction for `agiwo`
- `npm run lint`, `npm test`, and `npm run build` in `console/web`

At least one smoke test or CI command should prove that the documented `OpenAIModel(name="gpt-5.4")` path is valid from an installed wheel.

## Non-Goals

- No new provider abstractions
- No Console authentication or multi-tenant hardening
- No broad documentation rewrite beyond release-facing accuracy
- No runtime behavior changes unrelated to release correctness

## Acceptance Criteria

The repository is ready for `0.1.0` release when:

1. `pip install agiwo` provides a working public SDK import path without extra dependency selection
2. README quick start is copy-paste valid for the released package
3. Console is documented as an internal/self-hosted, not-yet-production-ready control plane with current Feishu-only channel integration
4. `agiwo` and `agiwo-console` both build successfully
5. CI verifies package build, clean-install smoke, and Console web quality gates
