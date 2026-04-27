# Console Browser CLI Tools Design

## Goal

Install the local development version of Browser CLI into the Console Docker
image, install its packaged skills into the default Agent skills directory, and
add a prompt-visible `TOOLS.md` workspace document that teaches Agents how to
use available tools and references.

## Scope

This change covers the first practical slice:

- Console Docker can install Browser CLI from a local source checkout such as
  `~/workspace/browser-cli`.
- Browser CLI packaged skills are installed into Agiwo's default skills
  directory inside the image.
- New workspaces receive `TOOLS.md` from `templates/TOOLS.md`.
- System prompt construction includes `TOOLS.md` together with the existing
  identity, soul, environment, runtime tool list, skill list, and user sections.

This change does not build the full failure-summary and experience-accumulation
data path. That should be a separate design because it needs durable runtime
facts, retrieval policy, prompt injection rules, and observability behavior.

## Architecture

### Browser CLI Image Install

`scripts/deploy_console.sh` should accept a `--browser-cli-source PATH` option.
When provided, the script builds a wheel from that source checkout before the
Console image build and passes the wheel into the Docker build.

The image should install the local wheel after Agiwo's normal dependencies are
installed. This intentionally overrides the published
`browser-control-and-automation-cli` dependency already declared by Agiwo. When
Browser CLI becomes stable and published, this local-source path can be removed
or left as an explicit development override while the default image uses the
published version.

The Docker build should not copy the Browser CLI repository into the Agiwo
repository or install from a host path at container runtime. The install happens
at image build time so a running Console container has a normal `browser-cli`
command on `PATH`.

### Browser CLI Skills

After the local Browser CLI wheel is installed, the image should run:

```bash
browser-cli install-skills --target /data/root/skills
```

The target is the default Agent skills directory for Docker deployments because
`agiwo-console container up` sets `AGIWO_ROOT_PATH=/data/root`. Installing into
that location keeps Browser CLI skills in the same discovery path as user-added
default skills.

The image build must create the target directory before installing skills. If
Docker later mounts a fresh host data directory over `/data`, those image-baked
files may be hidden by the mount. To keep the default-skill guarantee under a
mounted `/data`, the Console entrypoint should also run the same
`browser-cli install-skills --target /data/root/skills` command idempotently at
startup when `browser-cli` is available. Browser CLI replaces its packaged skill
directories safely on rerun.

### Host Network Browser Control

Console container lifecycle already supports `--network-mode host` on Linux.
The Browser CLI prompt guidance should make this runtime assumption explicit:
host networking is the intended mode when an Agent inside the Console container
needs to reach a host-side Browser CLI daemon or extension-connected browser.

Deployment docs should show the source-repo path:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data" \
  --network-mode host \
  --browser-cli-source "$HOME/workspace/browser-cli"
```

### TOOLS.md Prompt Document

Add `TOOLS.md` as a workspace prompt document, parallel to `IDENTITY.md`,
`SOUL.md`, and `USER.md`.

`TOOLS.md` is for durable tool practice guidance:

- what tools are available conceptually
- when to use Browser CLI versus direct API or shell work
- Browser CLI command patterns: `doctor`, `paths`, `status --json`, `read`,
  `open`, `snapshot`, `click`, `fill`, `wait`, `task validate`, `task run`,
  `automation publish`
- Browser CLI skills: `browser-cli-explore`, `browser-cli-converge`,
  `browser-cli-delivery`
- references to Browser CLI docs and Agiwo runtime-tool-boundary docs
- background-process handling: `bash(background=true)` returns a bash job ID,
  which must be inspected with `bash_process`, not `sleep_and_wait(wait_for=...)`
- guidance to record durable Browser CLI task findings in `task.meta.json`

`TOOLS.md` does not replace the existing dynamic runtime tool list rendered from
actual `BaseTool` instances. The dynamic section remains the authority for which
tools are currently callable. `TOOLS.md` supplies operating practice and links.

### Prompt Assembly

`WorkspaceDocumentStore` should read `TOOLS.md` and include it in its change
token. `AgentWorkspace` should expose `tools_path`. `WorkspaceBootstrapper`
should seed `TOOLS.md`.

`build_system_prompt()` should include `TOOLS.md` near the tool-related prompt
surface, after environment/review guidance and before or adjacent to the
dynamic runtime tool list. This keeps conceptual tool guidance close to the
actual tool inventory.

### Failure Experience Module

The later failure-summary system should be based on first-class runtime facts,
not free-form prompt scraping. A likely shape is:

- record structured tool failure and recovery lessons as append-only `RunLog`
  facts
- classify lessons by tool name, command family, workspace, and task context
- compact noisy outputs into concise lessons
- retrieve relevant lessons during prompt assembly based on the current task and
  available tools
- expose the facts through trace/Console views without relying on hidden prompt
  text as canonical state

That module should be designed separately because it crosses runtime state,
storage, review, memory, and prompt assembly.

## Error Handling

- If `--browser-cli-source` is provided and the path is missing or has no
  `pyproject.toml`, deployment should fail before Docker build starts.
- If Browser CLI wheel build fails, deployment should fail with the wheel-build
  output visible.
- If `browser-cli install-skills` fails during image build, the build should
  fail.
- If startup skill installation fails, the entrypoint should log the failure and
  continue starting Console, because a skills refresh problem should not make an
  otherwise healthy Console unavailable. The build-time install remains strict.

## Testing

Add focused tests for:

- `WorkspaceBootstrapper` seeds `TOOLS.md`.
- `WorkspaceDocumentStore` reads `TOOLS.md` and includes it in `change_token`.
- `build_system_prompt()` includes `TOOLS.md`.
- Docker runtime or deploy-script behavior validates `--browser-cli-source`
  input and passes the expected build argument/context.

Run at least:

```bash
uv run python scripts/lint.py ci
uv run pytest tests/agent/test_prompt.py -v
```

If Console Docker script or entrypoint behavior changes materially, also run:

```bash
uv run python scripts/check.py console-tests
```
