# Skills

Skills are file-based instruction sets that extend an agent's capabilities. They're discovered from directories and loaded at runtime.

## How Skills Work

A skill is a directory containing:

```
my_skill/
├── SKILL.md          # Required: instructions for the agent
├── script.py         # Optional: executable scripts
├── reference.md      # Optional: reference documentation
└── config.yaml       # Optional: skill configuration
```

The `SKILL.md` file is the core — its contents are injected into the agent's system prompt when the skill is active.

## Skill Discovery

Skills are discovered from configured directories:

```python
from agiwo.skill import SkillManager, SkillConfig

config = SkillConfig(
    skill_dirs=["./skills", "~/.agiwo/skills"],
)

manager = SkillManager(config)
await manager.discover()

# List available skills
for skill in manager.list_skills():
    print(f"{skill.name}: {skill.description}")
```

## Skill Loading

Skills are loaded on demand when the agent decides to use them:

```python
# The agent's LLM sees available skills and can request one
# SkillManager.load_skill() reads SKILL.md and returns the content
# The content is injected into the system prompt
```

## Skill Configuration

```python
@dataclass
class SkillConfig:
    skill_dirs: list[str]           # Directories to scan for skills
    auto_load: bool = False         # Auto-load all discovered skills
    max_skill_tokens: int = 4096    # Max tokens per skill's SKILL.md
```

## Creating a Skill

### Minimal skill

```markdown
<!-- skills/weather/SKILL.md -->
# Weather Skill

You can check the weather using the weather_check script.

## Usage

```bash
python script.py --city "Beijing"
```

## Output Format

Return temperature, condition, and humidity.
```

### With configuration

```yaml
# skills/my_skill/config.yaml
name: my_skill
version: "1.0"
description: Does something useful
dependencies:
  - requests
```

## Skill Tool

The `SkillTool` bridges skills to the agent runtime:

```python
from agiwo.skill import SkillTool

# Skills are registered as tools
# The agent can invoke skill_tool(skill_name, action, params)
```

## Built-in Skills

Agiwo can load skills from:
- Project-local `./skills/` directory
- User-global `~/.agiwo/skills/` directory
- Any directory specified in `SkillConfig.skill_dirs`

## Best Practices

1. **Keep SKILL.md focused** — Clear instructions, not essays
2. **Use examples** — Show the agent how to use the skill
3. **Include error handling** — What to do when things go wrong
4. **Version your skills** — Use `config.yaml` to track versions
5. **Test skills independently** — Scripts should work standalone before integrating
