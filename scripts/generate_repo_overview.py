from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "website" / "src" / "generated" / "repo-overview.json"
AGENTS_PATH = ROOT / "AGENTS.md"


FALLBACK_RESPONSIBILITIES: dict[str, str] = {
    "agiwo/agent": "Canonical agent runtime, execution loop, runtime state, models, nested-agent adapters, and persistence hooks.",
    "agiwo/llm": "Model abstraction, provider adapters, configuration policy, and factory construction.",
    "agiwo/tool": "Tool contracts, builtin tools, execution context, process registry, and tool-side persistence.",
    "agiwo/scheduler": "Agent-level orchestration, runtime tools, lifecycle management, and scheduler state persistence.",
    "agiwo/observability": "Trace and span storage, querying, and runtime trace adaptation.",
    "agiwo/embedding": "Embedding abstraction and provider-backed implementations.",
    "agiwo/skill": "Skill discovery, allowlisting, loading, and skill-to-tool bridging.",
    "agiwo/workspace": "Workspace path semantics, bootstrap, and runtime workspace helpers.",
    "agiwo/memory": "Shared workspace memory indexing, chunking, and search services.",
    "agiwo/config": "Global SDK configuration and shared provider settings.",
    "agiwo/utils": "Cross-module runtime utilities and shared storage support.",
    "console/server": "FastAPI control plane and runtime integration layer.",
    "console/server/routers": "HTTP and SSE API boundary for requests and responses.",
    "console/server/services": "Application services for runtime management, registry, tool catalog, session storage, and metrics.",
    "console/server/models": "Console-facing shared runtime, configuration, and view models.",
    "console/server/channels": "Channel adapters for delivery, parsing, and integration workflows.",
    "console/web": "Internal control-plane frontend for sessions, traces, scheduler, and settings.",
    "console/tests": "Console backend test suite.",
    "tests": "SDK test suite organized by subsystem.",
    "scripts": "Lint entrypoints, repo guardrails, and maintenance helpers.",
    "lint": "Import-linter and repository guard configuration.",
    "docs": "Repository-native design notes, concepts, and architecture documentation.",
    "templates": "Template content consumed by runtime features.",
    "trash": "Soft-delete landing area for removed files.",
}

LAYOUT_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        "sdk_layout",
        "SDK",
        [
            ("agiwo/agent", "Agent runtime"),
            ("agiwo/llm", "Model layer"),
            ("agiwo/tool", "Tool layer"),
            ("agiwo/scheduler", "Scheduler"),
            ("agiwo/observability", "Observability"),
            ("agiwo/embedding", "Embedding"),
            ("agiwo/skill", "Skills"),
            ("agiwo/workspace", "Workspace"),
            ("agiwo/memory", "Memory"),
            ("agiwo/config", "Configuration"),
            ("agiwo/utils", "Shared utilities"),
        ],
    ),
    (
        "console_layout",
        "Console",
        [
            ("console/server", "Control plane"),
            ("console/server/routers", "API boundary"),
            ("console/server/services", "Application services"),
            ("console/server/models", "Shared models"),
            ("console/server/channels", "Channel adapters"),
            ("console/web", "Internal web UI"),
            ("console/tests", "Console tests"),
        ],
    ),
    (
        "supporting_layout",
        "Supporting directories",
        [
            ("tests", "SDK tests"),
            ("scripts", "Repo scripts"),
            ("lint", "Guardrails"),
            ("docs", "Repository docs"),
            ("templates", "Templates"),
            ("trash", "Trash"),
        ],
    ),
]

RUNTIME_SURFACES: list[dict[str, object]] = [
    {
        "import_path": "agiwo.agent",
        "role": "Public entry for the canonical agent runtime, agent configuration, execution handles, and related types.",
        "source_paths": [
            "agiwo/agent",
            "agiwo/agent/__init__.py",
            "agiwo/agent/types.py",
        ],
    },
    {
        "import_path": "agiwo.scheduler",
        "role": "Public entry for orchestration, persistent roots, routing, waiting, and scheduler-backed agent coordination.",
        "source_paths": ["agiwo/scheduler", "agiwo/scheduler/engine.py"],
    },
    {
        "import_path": "agiwo.tool",
        "role": "Public entry for tool contracts, tool results, execution context, and builtin tool integration.",
        "source_paths": ["agiwo/tool", "agiwo/tool/manager.py"],
    },
    {
        "import_path": "agiwo.llm",
        "role": "Public entry for model abstractions, provider implementations, and model construction helpers.",
        "source_paths": ["agiwo/llm", "agiwo/llm/factory.py"],
    },
]

BOUNDARIES: list[str] = [
    "Scheduler sits on top of the agent runtime instead of the reverse direction.",
    "Console code should go through scheduler and agent facades rather than reading scheduler store internals directly.",
    "Public agent-facing types should enter through the correct facade instead of internal runtime modules when called from outside the agent package.",
    "The public docs site is static and separate from the internal Console web app.",
]

SOURCE_POINTER_FILES: list[tuple[str, str]] = [
    ("Repository guide", "AGENTS.md"),
    ("Public entry README", "README.md"),
    ("Architecture overview", "docs/architecture/overview.md"),
    ("Getting started guide", "docs/getting-started.md"),
    ("Multi-agent guide", "docs/guides/multi-agent.md"),
]


def parse_agents_responsibilities() -> dict[str, str]:
    text = AGENTS_PATH.read_text()
    responsibilities: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("| `"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) < 2:
            continue
        raw_path = cells[0]
        responsibility = cells[1]
        if not (raw_path.startswith("`") and raw_path.endswith("`")):
            continue
        normalized_path = raw_path.strip("`").rstrip("/")
        responsibilities[normalized_path] = responsibility
    return responsibilities


def first_heading(path: Path) -> str:
    if path.name == "README.md":
        return "Agiwo"
    if path.name == "AGENTS.md":
        return "AGENTS.md"
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem.replace("-", " ").title()


def build_layout_group(
    entries: list[tuple[str, str]],
    responsibilities: dict[str, str],
) -> list[dict[str, str]]:
    group: list[dict[str, str]] = []
    for path, label in entries:
        abs_path = ROOT / path
        if not abs_path.exists():
            continue
        agents_responsibility = responsibilities.get(path)
        fallback_responsibility = FALLBACK_RESPONSIBILITIES[path]
        responsibility = (
            agents_responsibility
            if agents_responsibility and agents_responsibility.isascii()
            else fallback_responsibility
        )
        group.append(
            {
                "path": path,
                "label": label,
                "responsibility": responsibility,
            }
        )
    return group


def build_source_pointers() -> list[dict[str, str]]:
    pointers: list[dict[str, str]] = []
    for label, relative_path in SOURCE_POINTER_FILES:
        path = ROOT / relative_path
        if not path.exists():
            continue
        pointer = {
            "label": label,
            "path": relative_path,
        }
        if path.suffix in {".md", ".mdx"}:
            pointer["title"] = first_heading(path)
        pointers.append(pointer)
    return pointers


def build_payload() -> dict[str, object]:
    responsibilities = parse_agents_responsibilities()
    layout_payload: dict[str, list[dict[str, str]]] = {}
    for key, _title, entries in LAYOUT_GROUPS:
        layout_payload[key] = build_layout_group(entries, responsibilities)

    return {
        "summary": (
            "Agiwo is organized around a canonical agent runtime, a separate scheduler "
            "orchestration layer, a tool abstraction, a model layer, and an internal control plane."
        ),
        "layout_sections": [
            {"key": key, "title": title} for key, title, _entries in LAYOUT_GROUPS
        ],
        **layout_payload,
        "runtime_surfaces": RUNTIME_SURFACES,
        "boundaries": BOUNDARIES,
        "source_pointers": build_source_pointers(),
    }


def render_payload() -> str:
    return json.dumps(build_payload(), indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a code-tree-driven public repository overview artifact."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the generated file differs from the expected output.",
    )
    args = parser.parse_args()

    expected = render_payload()
    if args.check:
        if OUTPUT_PATH.exists() and OUTPUT_PATH.read_text() == expected:
            return 0
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
