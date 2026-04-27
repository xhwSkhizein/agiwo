# Console Browser CLI Tools Design

## Goal

Install the local development checkout of Browser CLI into the Console Docker image, install its packaged skills into the Agent skill discovery path, and add a workspace `TOOLS.md` prompt document that teaches agents how to use available tools, Browser CLI, and reference docs.

## Scope

This phase includes:

- Docker build support for a local Browser CLI source checkout.
- Browser CLI packaged skill installation in the image.
- Default Console Docker skill discovery wiring for the installed Browser CLI skills.
- A new `templates/TOOLS.md` workspace prompt document.
- Prompt assembly changes so `TOOLS.md` appears in the system prompt.
- Tests for prompt document loading and Docker command/config behavior.

This phase does not include the durable tool-failure and experience-learning data path. That should be a later feature built on first-class runtime facts, not raw log injection.

## Browser CLI Source Installation

The source deployment path should accept a local Browser CLI checkout, initially `~/workspace/browser-cli`.

`scripts/deploy_console.sh` will gain:

```bash
--browser-cli-source PATH
```

When present, the script will:

1. Resolve `PATH`.
2. Verify it contains `pyproject.toml`.
3. Build a wheel from that checkout using `uv build --wheel --out-dir <temp-dir>`.
4. Pass the wheel directory into Docker build using a BuildKit additional context.
5. Set a Docker build arg enabling local Browser CLI installation.

The Docker image will first install Agiwo dependencies as it does today, then install the local Browser CLI wheel over the published dependency. This preserves the future release path: once Browser CLI is published, deployments can stop passing `--browser-cli-source` and rely on the normal `pyproject.toml` dependency.

## Browser CLI Skills

After the local Browser CLI wheel is installed, the Dockerfile will run:

```bash
browser-cli install-skills --target /opt/agiwo/browser-cli-skills
```

The image will make `/opt/agiwo/browser-cli-skills` readable by the runtime user.

Console Docker defaults will set:

```text
AGIWO_SKILLS_DIRS=["/opt/agiwo/browser-cli-skills","skills"]
```

This keeps built-in Browser CLI skills discoverable while preserving the existing default custom skills directory under `/data/root/skills`, because relative `skills` resolves under `AGIWO_ROOT_PATH=/data/root`.

The installed skills are expected to include:

- `browser-cli-explore`
- `browser-cli-converge`
- `browser-cli-delivery`

The default prompt should not render the full skill bodies. Existing reduced skill prompt behavior stays unchanged; agents can discover and activate skills through the skill runtime path.

## Host Browser Access

The existing Console container lifecycle already supports:

```bash
--network-mode host
```

This mode is required when the containerized Agent should operate a Browser CLI daemon or extension transport reachable on the host network. The docs and `TOOLS.md` should present host networking as the recommended deployment shape for host-browser operation on Linux.

No new Docker network abstraction is needed in this phase.

## TOOLS.md Prompt Document

Add `templates/TOOLS.md`.

The document is workspace-owned, like `SOUL.md`, `IDENTITY.md`, and `USER.md`. Workspace bootstrap copies it only when missing, so users and agents may refine it over time without later image upgrades overwriting local changes.

`TOOLS.md` should cover stable tool practice, not per-run tool schemas. It should include:

- When to use `bash` versus dedicated tools.
- How to manage `bash(background=true)` jobs with `bash_process`.
- A Browser CLI quick path:
  - start with `browser-cli doctor`, `browser-cli status`, and `browser-cli paths` when diagnosing runtime state;
  - use `browser-cli read <url> --snapshot` for one-shot exploration;
  - use `open -> snapshot -> click/fill/wait/verify` for interactive workflows;
  - refresh `snapshot` after page transitions or DOM changes;
  - use `X_AGENT_ID=<agent-id>` for multi-agent tab isolation;
  - use `task.py + task.meta.json` for reusable browser workflows;
  - validate reusable workflows with `browser-cli task validate` and `browser-cli task run`.
- Browser CLI references:
  - `/opt/agiwo/browser-cli-docs/README.md` if docs are copied into the image;
  - Browser CLI installed command help such as `browser-cli --help`, `browser-cli <command> --help`;
  - Agiwo runtime boundary docs where available.
- Skill practice:
  - activate `browser-cli-delivery` for reusable browser task delivery;
  - activate `browser-cli-explore` when site behavior still needs evidence;
  - activate `browser-cli-converge` once the path is validated and ready to encode;
  - keep durable Browser CLI discoveries in `task.meta.json` rather than only in chat.

The document should avoid claiming Browser CLI can control arbitrary user tabs. Extension mode manages Browser CLI-owned workspace windows and tabs.

## Prompt Assembly

Extend `AgentWorkspace` with `tools_path`.

Extend `WorkspaceDocuments` with `tools_text`.

`WorkspaceDocumentStore.read()` will load `TOOLS.md` and include it in the change token.

`build_system_prompt()` will render `TOOLS.md` near the runtime tool list:

1. identity
2. soul
3. base prompt
4. environment
5. goal-directed review
6. runtime tool list
7. `TOOLS.md`
8. skills reduced prompt section
9. user

This keeps the runtime list authoritative for what tools exist in the current run, while `TOOLS.md` explains how to use them well.

## Docker Documentation

Update Console Docker docs with a local Browser CLI deployment example:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data" \
  --network-mode host \
  --browser-cli-source "$HOME/workspace/browser-cli"
```

The docs should state that host networking is Linux-only in the first-class path and that the local Browser CLI source flag is intended for development until the desired Browser CLI version is published.

## Testing

Python-side tests:

- Prompt tests verify `TOOLS.md` is read and appears in the system prompt.
- Workspace document tests verify `TOOLS.md` participates in the change token.
- Docker runtime tests keep host-network behavior unchanged.
- Deploy script checks are shell-level and should be covered with low-risk static assertions where practical.

Required local checks after implementation:

```bash
uv run python scripts/lint.py ci
uv run pytest tests/agent/test_prompt.py tests/agent/test_skill_dirs.py tests/config/test_settings_env.py -v
uv run python scripts/check.py console-tests
```

If Dockerfile or deployment script changes are significant, run:

```bash
uv run python scripts/smoke_console_docker.py
```

The Docker smoke may be skipped locally only if Docker is unavailable; that skip must be reported.

## Later Experience Module

The later experience module should not parse free-form tool output as state. It should introduce structured facts such as tool failure classification, attempted recovery, durable lesson, and applicability keys. Prompt assembly can then load a bounded, relevant subset for the active task and tool family.

That design should integrate with RunLog and review/step-back facts so Console observability and prompt replay share one truth source.
