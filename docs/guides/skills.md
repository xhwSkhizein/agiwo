# Skills

Skills are file-based instruction sets that extend an agent's capabilities. Agiwo discovers them from configured directories, keeps their metadata in memory, and loads full `SKILL.md` content only when the agent explicitly activates a skill.

## How Skills Work

A skill is a directory containing:

```text
my_skill/
├── SKILL.md
├── scripts/
├── references/
└── assets/
```

`SKILL.md` is required. Its frontmatter provides the discovery metadata, and its body contains the instructions that are loaded when the skill is activated.

## Prompt-Time Visibility

The agent system prompt no longer includes the full discovered skill catalog.

Instead:

1. `AGIWO_DEFAULT_PROMPT_SKILLS` controls the small subset rendered into the prompt
2. that subset is still filtered by the agent's `allowed_skills`
3. if skills are disabled with `allowed_skills=[]`, no skill section is rendered

This keeps prompt size stable even when `skills_dirs` contains many skills.

## Runtime Discovery

Runtime discovery happens through the `skill` tool:

- `mode="search"` recommends one skill or `no_recommendation`
- `mode="activate"` loads the full `SKILL.md` body for a chosen skill

The normal flow is:

1. the model reads the user request
2. if a skill might help, it calls `skill` with `mode="search"` and the original user request
3. if search recommends a specific skill, the model calls `skill` again with `mode="activate"`
4. if search returns `no_recommendation`, the model continues without a skill

Discovery and activation both respect `allowed_skills`.

## Skill Discovery Config

Skills are discovered from the SDK-level `settings.skills_dirs` configuration, not from per-agent skill directory options:

```python
from agiwo.config.settings import settings
from agiwo.skill.manager import get_global_skill_manager

print(settings.skills_dirs)

manager = get_global_skill_manager()
await manager.initialize()

for skill in manager.list_available_skills():
    print(skill.name, skill.description)
```

For code references:

- `settings.skills_dirs` is the SDK-level source of truth
- `get_global_skill_manager()` is the normal runtime entrypoint
- `SkillManager` and `SkillDiscoveryConfig` are internal building blocks for that flow

Do not add per-agent skill directories or legacy `enable_skill` style toggles. Agent-level control should stay on `allowed_skills`.

Relevant SDK settings:

- `AGIWO_SKILLS_DIRS`
- `AGIWO_DEFAULT_PROMPT_SKILLS`
- `AGIWO_SKILL_SEARCH_ENABLED`
- `AGIWO_SKILL_SEARCH_TOP_K`

## Skill Tool

The `SkillTool` bridges skills to the agent runtime:

```python
from agiwo.skill import SkillTool

# mode="search" => recommend a skill or no skill
# mode="activate" => load the chosen SKILL.md body
```

The tool never auto-activates a skill during search.

## Creating A Skill

Minimal example:

```markdown
---
name: weather
description: Check weather conditions and summarize results.
---

Use the weather helper script when the user asks for weather information.
```

Best practices:

1. Keep `SKILL.md` focused and procedural.
2. Put long reference material in `references/` instead of the main skill body.
3. Keep scripts standalone so they can be run directly when the skill requires them.
4. Make the frontmatter description concrete, because discovery depends on metadata quality.
