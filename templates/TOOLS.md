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
