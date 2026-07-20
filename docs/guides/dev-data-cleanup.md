# Development data cleanup

Agiwo does **not** ship schema migrations for Objective / RunLog / Session stores. When models change, old SQLite files and local `.agiwo` trees must be deleted and recreated.

## What to delete

Typical local paths (relative to the repo or configured root):

| Path | Contents |
| --- | --- |
| `console/.agiwo/` | Console root (sessions, configs, sqlite) when using default Console root |
| `$AGIWO_ROOT_PATH` | SDK/Console shared root when overridden |
| `*.db` under the configured sqlite path | ObjectiveStore, RunLog, Scheduler state, session metadata |

Exact file names depend on `AGIWO_CONSOLE_*` / `AGIWO_*` storage settings (`metadata_type`, `run_log_type`, sqlite db path).

## Rebuild

```bash
# From repository root — wipe local console root then reinstall hooks/env as needed
rm -rf console/.agiwo
uv run python scripts/setup_dev_env.py
uv run python scripts/install_git_hooks.py
```

After cleanup, create agents/sessions again through Console or API. Do not attempt to “upgrade” old ObjectiveLog rows in place.

## Related vocabulary

- **ObjectiveBudget** owns handoffs / verification / LLM cost / active time.
- Scheduler **TaskGuard / TaskLimits** only protect the scheduling tree; they are not ObjectiveBudget.
