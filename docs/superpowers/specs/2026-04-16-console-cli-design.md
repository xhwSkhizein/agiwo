# Console CLI Design

**Date:** 2026-04-16

## Goal

Make the published `agiwo-console` package startable through a user-facing command instead of exposing the internal ASGI module path `server.app:app`.

## Scope

This design covers only the installed-package startup experience for `agiwo-console`.

It includes:

1. adding a package entrypoint command
2. implementing a `serve` subcommand
3. updating release-facing docs to use the new command
4. adding minimal tests for CLI argument handling

It does not include:

- new Console features
- auth or production hardening
- web frontend packaging changes
- a broader CLI suite beyond the single startup command

## Current Problem

After `pip install agiwo-console`, the user currently has to start the server with:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8422
```

This works from an installed environment and does not require entering the source repo, but it is still poor UX because:

- it exposes internal module structure
- it forces users to learn Uvicorn invocation details
- it makes the package feel like source code rather than a product command

## Decision

Expose a console script named:

```bash
agiwo-console serve
```

This will be the documented startup path for the packaged Console.

## Command Surface

The CLI should support:

```bash
agiwo-console serve
agiwo-console serve --host 0.0.0.0
agiwo-console serve --port 8422
agiwo-console serve --env-file /path/to/.env
agiwo-console serve --reload
```

Behavior:

- defaults should match the existing Console defaults where possible
- `serve` should call `uvicorn.run("server.app:app", ...)`
- the CLI should stay thin and avoid duplicating app startup logic

## Packaging Design

`console/pyproject.toml` will publish a script entry similar to:

```toml
[project.scripts]
agiwo-console = "server.cli:main"
```

The implementation module should be small and live inside `console/server/`.

## Implementation Shape

Add a new module:

- `console/server/cli.py`

Responsibilities:

- build the top-level argument parser
- provide the `serve` subcommand
- forward normalized arguments into `uvicorn.run(...)`
- return a standard process exit code

The CLI should use the standard library `argparse` rather than introducing a new dependency just for one command.

## Docs Changes

Release-facing docs should replace:

```bash
uvicorn server.app:app ...
```

with:

```bash
agiwo-console serve ...
```

At minimum this should update:

- `README.md`
- `docs/console/overview.md`

If any installation or startup guidance for the packaged Console appears elsewhere, it should be kept consistent.

## Testing Strategy

Add minimal tests that verify:

1. `serve` dispatches to `uvicorn.run`
2. `--host`, `--port`, `--env-file`, and `--reload` are passed through correctly
3. the package exposes the script entry after build/install

The installed-command smoke can remain lightweight. It is enough to verify the console script exists and can print help from a fresh installed environment.

## Acceptance Criteria

The change is complete when:

1. `pip install agiwo-console` exposes an `agiwo-console` command
2. `agiwo-console serve` starts the same app currently started by `uvicorn server.app:app`
3. release-facing docs no longer tell packaged users to invoke the internal module path directly
4. CLI behavior is covered by at least one automated test and one installed-package smoke path
