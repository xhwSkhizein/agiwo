# Console Browser CLI Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install a local Browser CLI build into the Console Docker image, install Browser CLI skills into the default Agent skills directory, and add `TOOLS.md` to workspace prompt assembly.

**Architecture:** Keep the Docker path explicit and reversible: source deployments build a Browser CLI wheel into a temporary ignored directory under `console/docker/`, and the Dockerfile installs any wheel found there after normal Agiwo dependencies. `TOOLS.md` becomes a fourth workspace prompt document, read and seeded alongside `IDENTITY.md`, `SOUL.md`, and `USER.md`.

**Tech Stack:** Bash deployment script, Dockerfile with pip install, Python workspace/prompt modules, pytest.

---

## File Structure

- Modify `agiwo/workspace/layout.py`: add `tools_path` to `AgentWorkspace`.
- Modify `agiwo/workspace/bootstrap.py`: seed `TOOLS.md`.
- Modify `agiwo/workspace/documents.py`: read `TOOLS.md` and include it in change tokens.
- Modify `agiwo/agent/prompt.py`: render `TOOLS.md` in the system prompt.
- Create `templates/TOOLS.md`: default tool practice guide.
- Modify `tests/agent/test_prompt.py`: cover `TOOLS.md` read/render behavior.
- Modify `console/Dockerfile`: install local Browser CLI wheel if present and install Browser CLI skills at image build.
- Modify `console/docker/entrypoint.sh`: idempotently refresh Browser CLI skills into `${AGIWO_ROOT_PATH}/skills` at startup.
- Create `console/docker/browser-cli-wheels/.gitkeep`: stable Docker COPY source.
- Modify `.gitignore`: ignore temporary Browser CLI wheel artifacts.
- Modify `scripts/deploy_console.sh`: add `--browser-cli-source`, build wheel, copy it for Docker, and clean it up.
- Modify `docs/console/docker.md`: document local Browser CLI + host network deployment.

## Task 1: Add `TOOLS.md` To Workspace Prompt Documents

**Files:**
- Modify: `agiwo/workspace/layout.py`
- Modify: `agiwo/workspace/bootstrap.py`
- Modify: `agiwo/workspace/documents.py`
- Modify: `agiwo/agent/prompt.py`
- Create: `templates/TOOLS.md`
- Test: `tests/agent/test_prompt.py`

- [ ] **Step 1: Add focused failing prompt tests**

Append these tests to `tests/agent/test_prompt.py`:

```python
@pytest.mark.asyncio
async def test_build_system_prompt_includes_tools_document(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    workspace.workspace.mkdir(parents=True)
    workspace.tools_path.write_text("# TOOLS.md\n\nUse Browser CLI for rendered pages.")

    prompt = await build_system_prompt(
        base_prompt="Base system prompt",
        workspace=workspace,
        tools=[],
        allowed_skills=[],
        bootstrapper=NoopBootstrapper(),
        document_store=WorkspaceDocumentStore(),
    )

    assert "# TOOLS.md" in prompt
    assert "Use Browser CLI for rendered pages." in prompt


def test_workspace_document_store_reads_tools_document(tmp_path: Path) -> None:
    workspace = build_agent_workspace(root_path=tmp_path, agent_name="planner")
    workspace.workspace.mkdir(parents=True)
    workspace.tools_path.write_text("Tool practice")

    documents = WorkspaceDocumentStore().read(workspace)

    assert documents.tools_text == "Tool practice"
    assert "TOOLS.md:" in documents.change_token
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/agent/test_prompt.py::test_build_system_prompt_includes_tools_document tests/agent/test_prompt.py::test_workspace_document_store_reads_tools_document -v
```

Expected: fail because `AgentWorkspace` has no `tools_path` and `WorkspaceDocuments` has no `tools_text`.

- [ ] **Step 3: Extend workspace layout**

Update `agiwo/workspace/layout.py` so the dataclass and builder include `TOOLS.md`:

```python
@dataclass(frozen=True)
class AgentWorkspace:
    root: Path
    workspace: Path
    memory_dir: Path
    work_dir: Path
    identity_path: Path
    soul_path: Path
    user_path: Path
    tools_path: Path
```

and in `build_agent_workspace()`:

```python
        tools_path=workspace / "TOOLS.md",
```

- [ ] **Step 4: Seed `TOOLS.md` during bootstrap**

Update `agiwo/workspace/bootstrap.py`:

```python
_TEMPLATE_FILENAMES = ("IDENTITY.md", "SOUL.md", "USER.md", "TOOLS.md")
```

- [ ] **Step 5: Read `TOOLS.md` from the document store**

Update `agiwo/workspace/documents.py`:

```python
@dataclass(frozen=True)
class WorkspaceDocuments:
    identity_text: str | None
    soul_text: str | None
    user_text: str | None
    tools_text: str | None
    change_token: str
```

In `read()`:

```python
        tools = self._read_optional(workspace.tools_path, "TOOLS.md")
        return WorkspaceDocuments(
            identity_text=identity,
            soul_text=soul,
            user_text=user,
            tools_text=tools,
            change_token=self._build_change_token(
                workspace.identity_path,
                workspace.soul_path,
                workspace.user_path,
                workspace.tools_path,
            ),
        )
```

- [ ] **Step 6: Render `TOOLS.md` in the system prompt**

Add this helper to `agiwo/agent/prompt.py` near `_render_tools()`:

```python
def _render_tools_document(documents: WorkspaceDocuments) -> str:
    content = documents.tools_text
    if not content:
        return ""
    return f"---\n\n{content}".strip()
```

Then include it in `sections` after `_render_goal_directed_review()` and before `_render_tools(...)`:

```python
        _render_tools_document(documents),
```

- [ ] **Step 7: Create default `TOOLS.md` template**

Create `templates/TOOLS.md` with this content:

````markdown
# TOOLS.md - Tool Practice And References

Use tools when they materially improve certainty, speed, or access to the real
environment. Prefer the smallest tool call that answers the next decision.

## Browser CLI

Use Browser CLI when the task depends on a rendered web page, login state,
interactive browser actions, downloads, screenshots, network evidence, or a
workflow that should become reusable automation.

Start with health and paths when the runtime is unfamiliar:

```bash
browser-cli doctor
browser-cli paths
browser-cli status --json
```

For one-shot page reading:

```bash
browser-cli read https://example.com
browser-cli read https://example.com --snapshot
browser-cli read https://example.com --snapshot --scroll-bottom
```

For interactive browser work, observe before acting and refresh the snapshot
after meaningful page changes:

```bash
browser-cli open https://example.com
browser-cli snapshot
browser-cli click @ref
browser-cli fill @input_ref "value"
browser-cli wait --text "Done"
browser-cli snapshot
```

Use `X_AGENT_ID=<stable-agent-id>` when multiple agents may share the same
Browser CLI daemon so tabs stay separated by agent.

When a flow becomes reusable, move from exploration to task artifacts:

```bash
browser-cli task template --output tasks/my_task
browser-cli task validate tasks/my_task
browser-cli task run tasks/my_task --set url=https://example.com
browser-cli automation publish tasks/my_task
```

Browser CLI skills may be available:

- `browser-cli-explore`: explore pages and record durable findings in
  `task.meta.json`
- `browser-cli-converge`: turn validated findings into stable `task.py`
- `browser-cli-delivery`: orchestrate exploration, convergence, validation, and
  optional automation packaging

## Browser CLI References

- Browser CLI installed guide:
  `/home/hongv/workspace/browser-cli/docs/installed-with-uv.md`
- Browser CLI usage guide:
  `/home/hongv/workspace/browser-cli/docs/browser-cli-usage-guide-zh.md`
- Browser CLI task examples:
  `/home/hongv/workspace/browser-cli/docs/examples/task-and-automation.md`
- Agiwo runtime tool boundaries:
  `docs/runtime-tool-boundaries.md`

## Bash Background Jobs

If you start a long-running shell command with `bash(background=true)`, the
returned ID is a bash process job ID. Inspect it with `bash_process`, not
`sleep_and_wait(wait_for=...)`.

```json
{"name": "bash_process", "arguments": {"action": "status", "job_id": "<job_id>"}}
{"name": "bash_process", "arguments": {"action": "logs", "job_id": "<job_id>", "tail": 200}}
```

Use `sleep_and_wait` for scheduler-managed child agents or intentional timer
wakes, not for bash process monitoring.

## Durable Tool Lessons

When Browser CLI exploration teaches something reusable, write it into the
task's `task.meta.json` sections such as `environment`, `success_path`,
`recovery_hints`, `failures`, and `knowledge`. Do not leave durable tool
lessons only in chat.
````

- [ ] **Step 8: Run focused prompt tests**

Run:

```bash
uv run pytest tests/agent/test_prompt.py -v
```

Expected: pass.

- [ ] **Step 9: Commit prompt document changes**

Run:

```bash
git add agiwo/workspace/layout.py agiwo/workspace/bootstrap.py agiwo/workspace/documents.py agiwo/agent/prompt.py templates/TOOLS.md tests/agent/test_prompt.py
git commit -m "feat: add tools prompt document"
```

## Task 2: Install Local Browser CLI Wheel In Console Docker

**Files:**
- Modify: `.gitignore`
- Modify: `console/Dockerfile`
- Create: `console/docker/browser-cli-wheels/.gitkeep`
- Modify: `scripts/deploy_console.sh`
- Test: shell syntax and help output

- [ ] **Step 1: Add Docker wheel staging directory**

Create `console/docker/browser-cli-wheels/.gitkeep` as an empty sentinel file.

- [ ] **Step 2: Ignore staged Browser CLI wheel artifacts**

Add this line to `.gitignore`:

```gitignore
console/docker/browser-cli-wheels/*.whl
```

- [ ] **Step 3: Add local wheel install to Dockerfile**

After the Agiwo SDK install in `console/Dockerfile`:

```dockerfile
COPY console/docker/browser-cli-wheels /tmp/browser-cli-wheels
RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
wheel="$(find /tmp/browser-cli-wheels -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
if [ -n "$wheel" ]; then
    PIP_CACHE_DIR=/root/.cache/pip pip install "$wheel"
fi
rm -rf /tmp/browser-cli-wheels
EOF
```

Keep this after normal dependency install so a local development wheel overrides the published Browser CLI version already pulled by Agiwo dependencies.

- [ ] **Step 4: Add image-build Browser CLI skill install**

After `/data` and `/home/agiwo` are created in `console/Dockerfile`, add:

```dockerfile
RUN install -d -m 0777 /data/root/skills \
    && if command -v browser-cli >/dev/null 2>&1; then \
        browser-cli install-skills --target /data/root/skills; \
    fi \
    && chown -R agiwo:agiwo /data/root
```

- [ ] **Step 5: Add deploy script argument and state**

In `scripts/deploy_console.sh` usage, add:

```text
  --browser-cli-source PATH
                        Build and install Browser CLI from a local source checkout
```

Add variables near the existing defaults:

```bash
BROWSER_CLI_SOURCE=""
BROWSER_CLI_WHEEL_DIR="$ROOT/console/docker/browser-cli-wheels"
BROWSER_CLI_STAGED_WHEEL=""
```

Parse the option in the `case` statement:

```bash
    --browser-cli-source)
      BROWSER_CLI_SOURCE="${2:-}"
      shift 2
      ;;
```

- [ ] **Step 6: Validate and build local Browser CLI wheel**

After `require_cmd uv`, add:

```bash
stage_browser_cli_wheel() {
  mkdir -p "$BROWSER_CLI_WHEEL_DIR"
  rm -f "$BROWSER_CLI_WHEEL_DIR"/*.whl

  if [[ -z "$BROWSER_CLI_SOURCE" ]]; then
    return 0
  fi

  if [[ ! -d "$BROWSER_CLI_SOURCE" ]]; then
    echo "Browser CLI source directory does not exist: $BROWSER_CLI_SOURCE" >&2
    exit 1
  fi

  local source_dir
  source_dir="$(cd "$BROWSER_CLI_SOURCE" && pwd)"
  if [[ ! -f "$source_dir/pyproject.toml" ]]; then
    echo "Browser CLI source must contain pyproject.toml: $source_dir" >&2
    exit 1
  fi

  echo "[deploy_console] building Browser CLI wheel from: $source_dir"
  (
    cd "$source_dir"
    uv build --wheel --out-dir "$BROWSER_CLI_WHEEL_DIR"
  )

  BROWSER_CLI_STAGED_WHEEL="$(find "$BROWSER_CLI_WHEEL_DIR" -maxdepth 1 -type f -name '*.whl' | sort | tail -n 1)"
  if [[ -z "$BROWSER_CLI_STAGED_WHEEL" || ! -f "$BROWSER_CLI_STAGED_WHEEL" ]]; then
    echo "Browser CLI wheel build did not produce a wheel" >&2
    exit 1
  fi
  echo "[deploy_console] staged Browser CLI wheel: $BROWSER_CLI_STAGED_WHEEL"
}

cleanup_browser_cli_wheel() {
  if [[ -n "$BROWSER_CLI_STAGED_WHEEL" ]]; then
    rm -f "$BROWSER_CLI_STAGED_WHEEL"
  fi
}

trap cleanup_browser_cli_wheel EXIT
```

Call `stage_browser_cli_wheel` immediately before the Docker build block.

- [ ] **Step 7: Include local-source flag in help smoke**

Run:

```bash
bash -n scripts/deploy_console.sh
scripts/deploy_console.sh --help | rg -- '--browser-cli-source'
```

Expected: syntax check passes and help includes the new flag.

- [ ] **Step 8: Verify wheel staging with the local Browser CLI checkout**

Run a controlled build staging command without starting Docker by using a temporary env file and `--no-build` is not sufficient because staging only happens before build. Instead, run the real wheel build command directly:

```bash
tmp_dir="$(mktemp -d)"
uv build --wheel --out-dir "$tmp_dir" /home/hongv/workspace/browser-cli
ls "$tmp_dir"/*.whl
rm -rf "$tmp_dir"
```

Expected: one Browser CLI wheel is produced.

- [ ] **Step 9: Commit Docker install changes**

Run:

```bash
git add .gitignore console/Dockerfile console/docker/browser-cli-wheels/.gitkeep scripts/deploy_console.sh
git commit -m "feat: install local browser-cli in console docker"
```

## Task 3: Refresh Browser CLI Skills At Container Startup

**Files:**
- Modify: `console/docker/entrypoint.sh`
- Test: shell syntax

- [ ] **Step 1: Add idempotent startup skill refresh**

In `console/docker/entrypoint.sh`, after:

```bash
mkdir -p "${AGIWO_ROOT_PATH}" "${AGIWO_ROOT_PATH}/skills" /data/runtime
```

add:

```bash
if command -v browser-cli >/dev/null 2>&1; then
  if ! browser-cli install-skills --target "${AGIWO_ROOT_PATH}/skills"; then
    echo "warning: failed to install Browser CLI skills into ${AGIWO_ROOT_PATH}/skills" >&2
  fi
fi
```

This must run before backend startup so the SDK skill manager can discover Browser CLI skills on first request.

- [ ] **Step 2: Run shell syntax check**

Run:

```bash
bash -n console/docker/entrypoint.sh
```

Expected: no output and exit code 0.

- [ ] **Step 3: Commit startup skill refresh**

Run:

```bash
git add console/docker/entrypoint.sh
git commit -m "feat: refresh browser-cli skills on console startup"
```

## Task 4: Document Host Network And Local Browser CLI Deployment

**Files:**
- Modify: `docs/console/docker.md`
- Test: docs review

- [ ] **Step 1: Add deployment documentation**

In `docs/console/docker.md`, add a section after "From a Cloned Repository":

````markdown
## Local Browser CLI Development Build

For deployments that should use a local Browser CLI checkout instead of the
published package version, pass `--browser-cli-source` when using the source
deployment script:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data" \
  --network-mode host \
  --browser-cli-source "$HOME/workspace/browser-cli"
```

The script builds a Browser CLI wheel from that checkout and installs it into
the Console image after Agiwo's normal dependencies. Browser CLI packaged
skills are installed into the default Agent skills directory,
`/data/root/skills`, and refreshed at container startup.

Use `--network-mode host` on Linux when Agents in the Console container need to
operate a host-side browser or Browser CLI extension-connected runtime.
````

- [ ] **Step 2: Commit docs update**

Run:

```bash
git add docs/console/docker.md
git commit -m "docs: document browser-cli console docker deployment"
```

## Task 5: Final Verification

**Files:**
- All files from prior tasks

- [ ] **Step 1: Run focused tests**

Run:

```bash
uv run pytest tests/agent/test_prompt.py -v
```

Expected: pass.

- [ ] **Step 2: Run Console backend tests**

Run:

```bash
uv run python scripts/check.py console-tests
```

Expected: pass.

- [ ] **Step 3: Run CI lint gate**

Run:

```bash
uv run python scripts/lint.py ci
```

Expected: pass.

- [ ] **Step 4: Optional Docker smoke**

If Docker is available and a full image build is acceptable, run:

```bash
scripts/deploy_console.sh \
  --env-file .env \
  --data-dir "$HOME/agiwo-data" \
  --network-mode host \
  --browser-cli-source "$HOME/workspace/browser-cli"
```

Then verify inside the container:

```bash
docker exec agiwo-console browser-cli --version
docker exec agiwo-console test -d /data/root/skills/browser-cli-explore
docker exec agiwo-console test -d /data/root/skills/browser-cli-converge
docker exec agiwo-console test -d /data/root/skills/browser-cli-delivery
```

Expected: Browser CLI version comes from the local checkout build, and all three Browser CLI skill directories exist.

- [ ] **Step 5: Review git status**

Run:

```bash
git status --short
```

Expected: only intentional changes remain. Do not touch unrelated untracked files such as `scripts/update_redeploy_console_docker.sh`.
