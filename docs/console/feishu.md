# Feishu Integration

The Console includes a Feishu channel runtime for real-time messaging with agents.

## Features

- WebSocket long-connection via the Lark SDK
- direct chat and group `@mention` support
- slash-style command routing such as `/status`, `/new`, `/switch`, and `/cancel`
- attachment download and message-content normalization
- session-first scheduler routing through shared runtime services
- text chunking and delivery back to Feishu
- in-memory or SQLite-backed channel metadata

## Setup

### 1. Create a Feishu App

1. Go to the Feishu Open Platform.
2. Create an app and enable bot capability.
3. Configure the permissions required by your workflow.
4. Enable long-connection mode.

### 2. Configure Environment

Add to your `.env`:

```bash
AGIWO_CONSOLE_FEISHU_ENABLED=True
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxx
AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN=xxxx
AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY=xxxx
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME=assistant
AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_bot_xxxx
```

### 3. Start the Console

```bash
agiwo-console serve --env-file .env
```

## Runtime Architecture

```text
Feishu Platform
  -> FeishuConnection
  -> FeishuInboundHandler
  -> command handlers or normal-message flow
  -> SessionContextService
  -> AgentRuntimeCache
  -> SessionRuntimeService
  -> Scheduler
  -> FeishuDeliveryService
  -> Feishu Platform
```

The channel package owns Feishu transport concerns only. Session lifecycle and scheduler orchestration stay in `console/server/services/runtime/`.

## Module Reference

| Module | Responsibility |
|--------|----------------|
| `service.py` | top-level orchestration and lifecycle |
| `connection.py` | long-connection bridge |
| `inbound_handler.py` | event parsing, dedup, trigger checks, command routing |
| `message_parser.py` | Feishu envelope normalization |
| `message_builder.py` | `InboundMessage` to `UserMessage` conversion |
| `delivery_service.py` | replies, follow-up messages, acknowledgements |
| `api_client.py` | Feishu OpenAPI wrapper |
| `dedup_store.py` | claimed-event persistence |
| `group_history_store.py` | recent group-history cache |
| `commands/` | slash-style command handlers |

## Message Flow

1. `FeishuConnection` receives an event envelope.
2. `FeishuInboundHandler` parses, deduplicates, and checks whether the bot should trigger.
3. command messages stay inside `commands/`.
4. regular messages are converted into `UserMessage`.
5. `SessionContextService` resolves or creates the active session.
6. `AgentRuntimeCache` returns the stable runtime agent for that session.
7. `SessionRuntimeService` sends the input through the scheduler and returns a stream or acknowledgement.
8. `FeishuDeliveryService` sends the result back to Feishu.

## Storage

Feishu channel metadata is persisted via memory or SQLite.

Current SQLite-backed records include:

- chat-context bindings
- session bindings
- claimed-event deduplication state

## Debugging

Check channel status:

```bash
curl http://localhost:8422/api/channels/feishu/status
```
