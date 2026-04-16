# First Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `agiwo` and `agiwo-console` ready for the first public release by simplifying installation, fixing the public SDK entry path, lowering Console marketing claims, and adding release gates that verify the built artifacts.

**Architecture:** Keep the runtime mostly intact and solve the release issues at the publish surface. The only intentional API change is making `OpenAIModel` support the documented single-argument constructor form; the rest of the work is packaging, docs, and CI hardening.

**Tech Stack:** Python 3.11+, Hatchling, uv, FastAPI, Next.js, Vitest, GitHub Actions

---

## File Structure

- Modify: `agiwo/llm/openai.py`
  - Relax the constructor so `OpenAIModel(name="gpt-5.4")` is valid and mirrors `name` into `id`.
- Modify: `tests/llm/test_openai.py`
  - Add unit coverage for the name-only constructor and missing-name/id failure mode.
- Modify: `pyproject.toml`
  - Move current runtime extras into the default dependency set and simplify optional dependencies to the dev-only surface.
- Modify: `console/pyproject.toml`
  - Depend on `agiwo` directly instead of `agiwo[all]`; remove the deprecated `tool.uv.dev-dependencies` block in favor of dependency groups.
- Create: `scripts/smoke_release_install.py`
  - Build-artifact smoke test that installs a wheel into a fresh venv and verifies the documented public import/constructor/tool surface.
- Modify: `README.md`
  - Make SDK the primary story, switch the public example to `OpenAIModel(name="gpt-5.4")`, simplify install instructions, and downscope Console positioning.
- Modify: `docs/getting-started.md`
  - Update first-run install and code examples to the new constructor and default install model.
- Modify: `docs/concepts/model.md`
  - Update the OpenAI example and shared model-field explanation to match the ergonomic constructor.
- Modify: `docs/concepts/tool.md`
  - Keep builtin tool docs aligned with the default install surface.
- Modify: `docs/console/overview.md`
  - Add explicit internal/self-hosted and not-production-ready language; keep Feishu as the only current channel integration.
- Modify: `docs/api/model.md`
  - Keep API docs aligned with the OpenAI example surface.
- Modify: `examples/01_hello_agent.py`
- Modify: `examples/02_streaming.py`
- Modify: `examples/03_custom_tool.py`
- Modify: `examples/04_builtin_tools.py`
- Modify: `examples/05_hooks.py`
- Modify: `examples/06_agent_as_tool.py`
- Modify: `examples/07_scheduler.py`
- Modify: `examples/08_multi_agent.py`
  - Update repo-visible sample code to the `OpenAIModel(name="gpt-5.4")` style wherever the example is using the default OpenAI path.
- Modify: `CHANGELOG.md`
  - Refresh the `0.1.0` release entry so the first release metadata matches the actual release-prep state.
- Modify: `.github/workflows/ci.yml`
  - Add package build jobs, wheel smoke validation, and `console/web` lint/test/build verification.

### Task 1: Make `OpenAIModel(name=...)` a valid public constructor

**Files:**
- Modify: `agiwo/llm/openai.py`
- Modify: `tests/llm/test_openai.py`
- Test: `tests/llm/test_openai.py`

- [ ] **Step 1: Add failing unit tests for the new constructor contract**

```python
@patch("agiwo.llm.openai.get_settings")
def test_openai_model_name_only_defaults_id(mock_get_settings):
    mock_settings = mock_get_settings.return_value
    mock_settings.openai_api_key = None

    model = OpenAIModel(name="gpt-5.4", api_key="test-key")

    assert model.id == "gpt-5.4"
    assert model.name == "gpt-5.4"


def test_openai_model_requires_name_or_id():
    with pytest.raises(
        TypeError,
        match="OpenAIModel requires at least one of id or name",
    ):
        OpenAIModel()
```

- [ ] **Step 2: Run the targeted test file to confirm the new contract currently fails**

Run: `uv run pytest tests/llm/test_openai.py -q`

Expected: FAIL with a `TypeError` showing that `name` is required or that `OpenAIModel()` cannot be called with only `name`.

- [ ] **Step 3: Update `OpenAIModel.__init__` to normalize missing `id`/`name`**

```python
class OpenAIModel(Model):
    def __init__(
        self,
        id: str | None = None,
        name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = "https://api.openai.com/v1",
        allow_env_fallback: bool = True,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_output_tokens: int = 4096,
        max_context_window: int = 200000,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        cache_hit_price: float = 0.0,
        input_price: float = 0.0,
        output_price: float = 0.0,
        provider: str = "openai",
    ):
        if id is None and name is None:
            raise TypeError("OpenAIModel requires at least one of id or name")

        resolved_id = id or name
        resolved_name = name or id

        config = LLMConfig(
            id=resolved_id,
            name=resolved_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            max_context_window=max_context_window,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            provider=provider,
            cache_hit_price=cache_hit_price,
            input_price=input_price,
            output_price=output_price,
        )
        super().__init__(config)
        self.allow_env_fallback = allow_env_fallback
        self.client = self._create_client()
```

- [ ] **Step 4: Run the targeted test file again**

Run: `uv run pytest tests/llm/test_openai.py -q`

Expected: PASS

- [ ] **Step 5: Run the full LLM test slice to catch constructor regressions in provider subclasses**

Run: `uv run pytest tests/llm/ -q`

Expected: PASS

- [ ] **Step 6: Commit the constructor contract change**

```bash
git add agiwo/llm/openai.py tests/llm/test_openai.py
git commit -m "feat: support name-only openai model construction"
```

### Task 2: Make the default `agiwo` install include the full runtime surface

**Files:**
- Modify: `pyproject.toml`
- Modify: `console/pyproject.toml`
- Create: `scripts/smoke_release_install.py`
- Test: `scripts/smoke_release_install.py`

- [ ] **Step 1: Add the release smoke script before changing package metadata**

```python
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/smoke_release_install.py <wheel-path>")

    wheel_path = Path(sys.argv[1]).resolve()
    if not wheel_path.is_file():
        raise SystemExit(f"Wheel not found: {wheel_path}")

    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("uv executable not found in PATH")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        venv_path = tmp_path / "venv"
        python_path = (
            venv_path / "Scripts" / "python.exe"
            if sys.platform == "win32"
            else venv_path / "bin" / "python"
        )

        run([uv, "venv", str(venv_path)])
        run([uv, "pip", "install", "--python", str(python_path), str(wheel_path)])
        run(
            [
                str(python_path),
                "-c",
                (
                    "from agiwo.llm import OpenAIModel; "
                    "from agiwo.tool.manager import ToolManager; "
                    "model = OpenAIModel(name='gpt-5.4', api_key='test-key'); "
                    "assert model.id == 'gpt-5.4'; "
                    "assert model.name == 'gpt-5.4'; "
                    "defaults = set(ToolManager().list_default_tool_names()); "
                    "assert {'bash', 'bash_process', 'web_search', 'web_reader', 'memory_retrieval'} <= defaults; "
                    "print('release smoke ok')"
                ),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Build the current wheel and prove the smoke script fails before the dependency cleanup**

Run: `uv build && uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl`

Expected: FAIL because the wheel-installed package is still missing provider/web dependencies or the default tool surface does not yet include `web_reader`.

- [ ] **Step 3: Move runtime extras into the root package default dependency list**

```toml
[project]
dependencies = [
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "structlog>=24.0.0",
    "tenacity>=8.0.0",
    "openai>=1.0.0",
    "anthropic>=0.34.0",
    "boto3>=1.42.59",
    "python-dotenv>=1.0.0",
    "orjson>=3.11.6",
    "aiosqlite>=0.22.1",
    "jinja2>=3.1.6",
    "pyyaml>=6.0.3",
    "tiktoken>=0.7.0",
    "httpx[socks]>=0.27.0",
    "aiofiles>=25.1.0",
    "browser-control-and-automation-cli>=0.1.3",
    "curl-cffi>=0.14.0",
    "bs4>=0.0.2",
    "trafilatura>=2.0.0",
    "lark-oapi>=1.5.3",
    "motor>=3.7.1",
    "opentelemetry-distro>=0.60b1",
    "opentelemetry-exporter-otlp>=1.39.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.14.0",
    "httpx>=0.27.0",
    "ruff>=0.15.5",
    "import-linter>=2.4.0",
]
```

- [ ] **Step 4: Update the Console package to depend on the base SDK package directly**

```toml
[project]
dependencies = [
    "agiwo",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "httpx[socks]>=0.27.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "sse-starlette>=2.0.0",
    "aiosqlite>=0.22.1",
    "motor>=3.7.1",
    "bs4>=0.0.2",
    "trafilatura>=2.0.0",
    "lark-oapi>=1.5.3",
    "sqlite-vec>=0.1.6",
    "aiofiles>=25.1.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.uv.sources]
agiwo = { path = "..", editable = true }
```

- [ ] **Step 5: Rebuild the wheel and rerun the release smoke**

Run: `uv build && uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl`

Expected: PASS with `release smoke ok`

- [ ] **Step 6: Rebuild the Console package to verify its dependency metadata still works**

Run: `cd console && uv build`

Expected: PASS

- [ ] **Step 7: Commit the packaging and smoke-test changes**

```bash
git add pyproject.toml console/pyproject.toml scripts/smoke_release_install.py
git commit -m "build: default agiwo install to full runtime surface"
```

### Task 3: Update public docs, examples, and release messaging

**Files:**
- Modify: `README.md`
- Modify: `docs/getting-started.md`
- Modify: `docs/concepts/model.md`
- Modify: `docs/concepts/tool.md`
- Modify: `docs/console/overview.md`
- Modify: `docs/api/model.md`
- Modify: `examples/01_hello_agent.py`
- Modify: `examples/02_streaming.py`
- Modify: `examples/03_custom_tool.py`
- Modify: `examples/04_builtin_tools.py`
- Modify: `examples/05_hooks.py`
- Modify: `examples/06_agent_as_tool.py`
- Modify: `examples/07_scheduler.py`
- Modify: `examples/08_multi_agent.py`
- Modify: `CHANGELOG.md`
- Test: `README.md`, `docs/getting-started.md`

- [ ] **Step 1: Rewrite the README install and SDK quick-start sections**

````md
### Install

```bash
pip install agiwo
```

### Minimal SDK Example

```python
import asyncio

from agiwo.agent import Agent, AgentConfig
from agiwo.llm import OpenAIModel


async def main() -> None:
    agent = Agent(
        AgentConfig(
            name="assistant",
            description="A helpful assistant",
            system_prompt="You are a concise assistant.",
        ),
        model=OpenAIModel(name="gpt-5.4"),
    )

    result = await agent.run("What is 2 + 2?")
    print(result.response)

    await agent.close()


asyncio.run(main())
```
````

- [ ] **Step 2: Reduce Console promotion in the README and make the readiness statement explicit**

```md
## Console

The Console is an optional control-plane backend with a bundled web UI for operating Agiwo agents.

- Current channel integration: Feishu only
- Recommended deployment model: internal/self-hosted use
- Readiness: useful today for operator workflows, but not yet production-ready
```

- [ ] **Step 3: Update repo docs and examples to use the same public OpenAI path**

```python
model = OpenAIModel(name="gpt-5.4")
```

Apply that constructor style anywhere the file is demonstrating the default OpenAI path:

- `docs/getting-started.md`
- `docs/concepts/model.md`
- `docs/api/model.md`
- `examples/01_hello_agent.py`
- `examples/02_streaming.py`
- `examples/03_custom_tool.py`
- `examples/04_builtin_tools.py`
- `examples/05_hooks.py`
- `examples/06_agent_as_tool.py`
- `examples/07_scheduler.py`
- `examples/08_multi_agent.py`

- [ ] **Step 4: Keep builtin tool and Console docs aligned with the new install and readiness surface**

```md
| `web_reader` | Fetch and extract web page content |
```

```md
## Console Overview

The Console is a self-hosted control plane for Agiwo. It includes a FastAPI backend and a bundled Next.js UI, and is best treated as an internal operations tool rather than a production-ready end-user product.

Current channel integrations:

- Feishu
```

- [ ] **Step 5: Refresh the changelog entry for the actual first release date**

```md
## [0.1.0] - 2026-04-16
```

- [ ] **Step 6: Run a focused documentation/example grep to ensure the old invalid OpenAI example is gone**

Run: `rg -n 'OpenAIModel\\(id=\"gpt-4o-mini\"|OpenAIModel\\(id=\"gpt-4o\"' README.md docs examples`

Expected: no matches in README, getting-started, model docs, or the updated examples

- [ ] **Step 7: Commit the public-docs release cleanup**

```bash
git add README.md docs/getting-started.md docs/concepts/model.md docs/concepts/tool.md docs/console/overview.md docs/api/model.md examples/01_hello_agent.py examples/02_streaming.py examples/03_custom_tool.py examples/04_builtin_tools.py examples/05_hooks.py examples/06_agent_as_tool.py examples/07_scheduler.py examples/08_multi_agent.py CHANGELOG.md
git commit -m "docs: align public release docs and examples"
```

### Task 4: Add release gates for wheel build, wheel smoke, and Console web

**Files:**
- Modify: `.github/workflows/ci.yml`
- Test: `.github/workflows/ci.yml`

- [ ] **Step 1: Add an SDK package job that builds the wheel and runs the smoke script**

```yaml
  package-sdk:
    name: Package SDK
    runs-on: ubuntu-latest
    needs: [lint, test-sdk]
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: uv sync

      - name: Build SDK package
        run: uv build

      - name: Smoke test built SDK wheel
        run: uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl
```

- [ ] **Step 2: Add a Console package job**

```yaml
  package-console:
    name: Package Console
    runs-on: ubuntu-latest
    needs: [lint, test-console]
    steps:
      - uses: actions/checkout@v4

      - uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Install dependencies
        run: cd console && uv sync

      - name: Build Console package
        run: cd console && uv build
```

- [ ] **Step 3: Add a `console/web` verification job**

```yaml
  test-web:
    name: Test Web
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: "22"
          cache: "npm"
          cache-dependency-path: console/web/package-lock.json

      - name: Install web dependencies
        run: npm --prefix console/web ci

      - name: Run web lint
        run: npm --prefix console/web run lint

      - name: Run web tests
        run: npm --prefix console/web test

      - name: Run web production build
        run: npm --prefix console/web run build
```

- [ ] **Step 4: Run the full local release gate before pushing**

Run:

```bash
uv run ruff check --ignore C901 --ignore PLR0911 --ignore PLR0912 agiwo/ console/server/ tests/ console/tests/ scripts/
uv run ruff format --check agiwo/ console/server/ tests/ console/tests/ scripts/
uv run python scripts/lint.py imports
uv run python scripts/repo_guard.py
uv run pytest tests/ -q
cd console && uv run pytest tests/ -q
cd ../console/web && npm run lint && npm test && npm run build
cd ../.. && uv build && uv run python scripts/smoke_release_install.py dist/agiwo-0.1.0-py3-none-any.whl
cd console && uv build
```

Expected: PASS end-to-end

- [ ] **Step 5: Commit the release-gate workflow changes**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add release packaging and web verification gates"
```

## Self-Review

- Spec coverage:
  - Simpler default install: Task 2
  - `OpenAIModel(name="gpt-5.4")` public path: Task 1 + Task 3 + Task 4 smoke
  - Console de-emphasis / internal-use positioning: Task 3
  - Version/release metadata refresh: Task 3
  - CI release gates: Task 4
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to” placeholders remain
- Type consistency:
  - The new OpenAI constructor contract is consistently `id: str | None = None`, `name: str | None = None`, with normalization to non-optional `LLMConfig` values
