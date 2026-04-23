# Feishu ACK Reaction Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Feishu message ACKs reliable by switching the default reaction to a safe emoji and extending the ACK fallback chain from reaction to reply to create-message.

**Architecture:** Keep ACK ownership inside `FeishuDeliveryService` and avoid adding new abstractions. Fix the unsafe default in `ConsoleConfig`, drive the ACK ladder with focused regression tests, and redeploy the managed `agiwo-console` container from the current source tree while preserving the existing data directory and host-network setup.

**Tech Stack:** Python 3.10+, FastAPI Console backend, Feishu channel, httpx, pytest, Docker, `scripts/deploy_console.sh`

---

## File Structure

- Modify: `console/server/config.py`
  Change the default Feishu ACK reaction from `Typing` to `OnIt`.
- Modify: `console/server/channels/feishu/delivery_service.py`
  Keep `send_ack()` as the ACK owner and extend it to try reaction, then reply, then create-message.
- Modify: `console/tests/test_feishu_service_components.py`
  Add regression tests for the three ACK branches.
- Modify: `console/tests/test_config_env.py`
  Add coverage for the new default ACK reaction value.
- Create: `/tmp/agiwo-console-redeploy.env`
  Export the current container environment into a temporary env file for replacement deployment.

## Task 1: Lock In the ACK Contract With Tests

**Files:**
- Modify: `console/tests/test_feishu_service_components.py`
- Modify: `console/tests/test_config_env.py`

- [ ] **Step 1: Add a failing config-default test**

```python
def test_console_config_uses_safe_default_feishu_ack_reaction() -> None:
    config = ConsoleConfig()

    assert config.channels.feishu.ack_reaction_emoji == "OnIt"
```

- [ ] **Step 2: Add a failing ACK reaction-success test**

```python
@pytest.mark.asyncio
async def test_delivery_service_send_ack_stops_after_reaction_success() -> None:
    api = SimpleNamespace(
        add_message_reaction=AsyncMock(),
        reply_text=AsyncMock(),
        create_text_message=AsyncMock(),
    )
    delivery = FeishuDeliveryService(
        api=api,
        config=ConsoleConfig(),
        truncate_for_log=lambda text: text,
    )

    await delivery.send_ack(_inbound_message())

    api.add_message_reaction.assert_awaited_once_with("msg-1", "OnIt")
    api.reply_text.assert_not_called()
    api.create_text_message.assert_not_called()
```

- [ ] **Step 3: Add a failing ACK reply-fallback test**

```python
@pytest.mark.asyncio
async def test_delivery_service_send_ack_falls_back_to_reply_text() -> None:
    api = SimpleNamespace(
        add_message_reaction=AsyncMock(side_effect=RuntimeError("reaction failed")),
        reply_text=AsyncMock(),
        create_text_message=AsyncMock(),
    )
    delivery = FeishuDeliveryService(
        api=api,
        config=ConsoleConfig(),
        truncate_for_log=lambda text: text,
    )

    await delivery.send_ack(_inbound_message())

    api.add_message_reaction.assert_awaited_once_with("msg-1", "OnIt")
    api.reply_text.assert_awaited_once_with("msg-1", "收到，正在处理。")
    api.create_text_message.assert_not_called()
```

- [ ] **Step 4: Add a failing ACK create-message fallback test**

```python
@pytest.mark.asyncio
async def test_delivery_service_send_ack_falls_back_to_create_message() -> None:
    api = SimpleNamespace(
        add_message_reaction=AsyncMock(side_effect=RuntimeError("reaction failed")),
        reply_text=AsyncMock(side_effect=RuntimeError("reply failed")),
        create_text_message=AsyncMock(),
    )
    delivery = FeishuDeliveryService(
        api=api,
        config=ConsoleConfig(),
        truncate_for_log=lambda text: text,
    )

    await delivery.send_ack(_inbound_message())

    api.add_message_reaction.assert_awaited_once_with("msg-1", "OnIt")
    api.reply_text.assert_awaited_once_with("msg-1", "收到，正在处理。")
    api.create_text_message.assert_awaited_once_with("chat-1", "收到，正在处理。")
```

- [ ] **Step 5: Run the focused tests to verify they fail**

Run: `uv run pytest console/tests/test_feishu_service_components.py console/tests/test_config_env.py -k "ack or safe_default_feishu_ack_reaction" -v`

Expected: FAIL because `ConsoleConfig` still defaults to `Typing` and `FeishuDeliveryService.send_ack()` currently stops after `reply_text()` failure.

- [ ] **Step 6: Commit the red test slice**

```bash
git add console/tests/test_feishu_service_components.py console/tests/test_config_env.py
git commit -m "test: cover feishu ack fallback ladder"
```

## Task 2: Implement the Safe Default and ACK Ladder

**Files:**
- Modify: `console/server/config.py`
- Modify: `console/server/channels/feishu/delivery_service.py`
- Test: `console/tests/test_feishu_service_components.py`
- Test: `console/tests/test_config_env.py`

- [ ] **Step 1: Change the default ACK reaction in config**

```python
class FeishuConfig(BaseModel):
    enabled: bool = False
    channel_instance_id: str = "feishu-main"
    api_base_url: str = "https://open.feishu.cn"
    app_id: str = ""
    app_secret: str = ""
    verification_token: str = ""
    encrypt_key: str = ""
    sdk_log_level: Literal["debug", "info", "warn", "error"] = "info"
    bot_open_id: str = ""
    default_agent_name: str = ""
    whitelist_open_ids: list[str] = Field(default_factory=list)
    debounce_ms: int = 3000
    max_batch_window_ms: int = 15000
    scheduler_wait_timeout: int = 900
    ack_reaction_emoji: str = "OnIt"
    ack_fallback_text: str = "收到，正在处理。"
```

- [ ] **Step 2: Extend `send_ack()` to a three-step ladder**

```python
    async def send_ack(self, inbound: InboundMessage) -> None:
        try:
            await self._api.add_message_reaction(
                inbound.message_id,
                self._config.feishu_ack_reaction_emoji,
            )
            logger.info(
                "feishu_ack_sent",
                channel="feishu",
                mode="reaction",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_ack_reaction_failed",
                message_id=inbound.message_id,
                error=str(exc),
            )

        try:
            await self._api.reply_text(
                inbound.message_id,
                self._config.feishu_ack_fallback_text,
            )
            logger.info(
                "feishu_ack_sent",
                channel="feishu",
                mode="reply_fallback",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
                text=self._truncate_for_log(self._config.feishu_ack_fallback_text),
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_ack_fallback_failed",
                message_id=inbound.message_id,
                error=str(exc),
            )

        try:
            await self._api.create_text_message(
                inbound.chat_id,
                self._config.feishu_ack_fallback_text,
            )
            logger.info(
                "feishu_ack_sent",
                channel="feishu",
                mode="create_message_fallback",
                chat_type=inbound.chat_type,
                chat_id=inbound.chat_id,
                message_id=inbound.message_id,
                text=self._truncate_for_log(self._config.feishu_ack_fallback_text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "feishu_ack_create_message_failed",
                message_id=inbound.message_id,
                chat_id=inbound.chat_id,
                error=str(exc),
            )
```

- [ ] **Step 3: Run the focused tests again**

Run: `uv run pytest console/tests/test_feishu_service_components.py console/tests/test_config_env.py -k "ack or safe_default_feishu_ack_reaction" -v`

Expected: PASS for the new config default assertion and all three ACK ladder cases.

- [ ] **Step 4: Run the full Feishu component test file**

Run: `uv run pytest console/tests/test_feishu_service_components.py -v`

Expected: PASS, confirming the ACK changes do not break command routing, group reply fallback, or attachment/message parsing coverage.

- [ ] **Step 5: Commit the implementation slice**

```bash
git add console/server/config.py console/server/channels/feishu/delivery_service.py console/tests/test_feishu_service_components.py console/tests/test_config_env.py
git commit -m "fix: harden feishu ack delivery"
```

## Task 3: Run Required Backend Checks

**Files:**
- Modify: none
- Test: `console/tests/test_feishu_service_components.py`
- Test: `console/tests/test_config_env.py`

- [ ] **Step 1: Run the repo lint gate required for Python changes**

Run: `uv run python scripts/lint.py ci`

Expected: PASS. If this fails because of unrelated pre-existing syntax errors outside the Feishu slice, capture the failing file path and keep it in the work log before moving on.

- [ ] **Step 2: Run the console backend test gate**

Run: `uv run python scripts/check.py console-tests`

Expected: PASS for the console backend suite, including the Feishu channel tests.

- [ ] **Step 3: Record the verification outcome**

No repo files change in this task. Keep the test output in the work log and do not create a verification-only commit here.

## Task 4: Redeploy the Managed Console Container

**Files:**
- Create: `/tmp/agiwo-console-redeploy.env`
- Modify: none

- [ ] **Step 1: Export the current container environment to a temporary env file**

Run:

```bash
docker inspect agiwo-console --format '{{range .Config.Env}}{{println .}}{{end}}' \
  | python - <<'PY'
import sys
from pathlib import Path

keep = []
for raw in sys.stdin:
    line = raw.strip()
    if not line or "=" not in line:
        continue
    key, value = line.split("=", 1)
    if key.startswith(("PATH", "LANG", "HOME", "PYTHON_", "PIP_", "GPG_KEY", "NEXT_TELEMETRY_DISABLED", "NPM_CONFIG_UPDATE_NOTIFIER")):
        continue
    keep.append(f"{key}={value}")

path = Path("/tmp/agiwo-console-redeploy.env")
path.write_text("\\n".join(keep) + "\\n", encoding="utf-8")
print(path)
PY
```

Expected: prints `/tmp/agiwo-console-redeploy.env` and creates a reusable env file containing the current Console and provider settings.

- [ ] **Step 2: Rebuild and replace the managed container from the current source tree**

Run:

```bash
scripts/deploy_console.sh \
  --env-file /tmp/agiwo-console-redeploy.env \
  --data-dir /home/hongv/.agiwo-console-data \
  --name agiwo-console \
  --image agiwo-console:local \
  --network-mode host \
  --mount /home/hongv/workspace/agiwo:agiwo-repo
```

Expected: image rebuild completes, the existing `agiwo-console` container is replaced, and the script prints `deployment complete`.

- [ ] **Step 3: Verify container health and the ACK runtime config**

Run:

```bash
docker inspect agiwo-console --format '{{.State.Health.Status}}'
curl -fsS http://127.0.0.1:8422/api/health
docker inspect agiwo-console --format '{{range .Config.Env}}{{println .}}{{end}}' | rg 'AGIWO_CONSOLE_CHANNELS__FEISHU__ACK_REACTION_EMOJI'
```

Expected:

- `healthy`
- `{"status":"ok","service":"agiwo-console"}`
- `AGIWO_CONSOLE_CHANNELS__FEISHU__ACK_REACTION_EMOJI=OnIt` unless the operator intentionally overrides it in the env file

- [ ] **Step 4: Verify the new code is inside the running container**

Run:

```bash
docker exec agiwo-console python - <<'PY'
from server.config import ConsoleConfig
print(ConsoleConfig().channels.feishu.ack_reaction_emoji)
PY
```

Expected: prints `OnIt`.

- [ ] **Step 5: Commit the deployment helper command updates if any repo files changed**

```bash
git status --short
```

Expected: no new repo changes from deployment. If the working tree is clean for this slice, no deployment commit is needed.
