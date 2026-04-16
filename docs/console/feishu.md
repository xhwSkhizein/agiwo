# Feishu Integration

The Console includes a full Feishu (Lark) channel integration for real-time messaging with agents.

## Features

- WebSocket long-connection via Lark SDK for real-time message receiving
- Direct (P2P) and group chat (with @mention) support
- Slash command routing (`/status`, `/new`, `/switch`, `/cancel`, etc.)
- Multi-format message handling: text, rich post, image, file, audio, video, sticker, card, shared chat
- Attachment download with MIME magic-byte detection
- Session lifecycle with debounce-based batch processing
- Agent executor with automatic scheduler state routing (submit / enqueue / steer)
- SSE-style streaming response delivery with text chunking
- Message history storage in memory or SQLite
- User-facing error localization

## Setup

### 1. Create a Feishu App

1. Go to [Feishu Open Platform](https://open.feishu.cn/)
2. Create a new app and enable bot capability
3. Configure the following permissions:
   - `im:message:receive` — Receive messages
   - `im:message:send_as_bot` — Send messages as bot
   - `im:chat:readonly` — Read chat info
   - `contact:user.id:readonly` — Read user IDs
4. Enable the **Long-Connection** mode (recommended)

### 2. Configure Environment

Add to your `.env`:

```bash
AGIWO_CONSOLE_FEISHU_ENABLED=True
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxx
AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN=xxxx
AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY=xxxx          # Optional, if encryption enabled
AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME=assistant
AgIWO_CONSOLE_FEISHU_BOT_OPEN_ID=ou_bot_xxxx    # Required for group @mention detection
```

### 3. Start the Console

The Feishu channel starts automatically with the Console server:

```bash
agiwo-console serve --env-file .env
```

## Architecture

```
Feishu Platform
    │
    │ WebSocket
    ▼
┌───────────────────────────────────────────────────────────┐
│ FeishuChannelService (service.py)                         │
│                                                           │
│  ┌─ Connection ──────────────────────────────────┐       │
│  │  FeishuConnection (connection.py)             │       │
│  │  └─ Runs in separate thread (Lark SDK)        │       │
│  └───────────────────────────────────────────────┘       │
│                        │ envelope                        │
│  ┌─ Inbound Handler ─────────────────────────────┐       │
│  │  FeishuInboundHandler (inbound_handler.py)    │       │
│  │  ├─ FeishuMessageParser  → parse / extract    │       │
│  │  ├─ FeishuContentExtractor → normalize text   │       │
│  │  ├─ FeishuSenderResolver   → cache sender     │       │
│  │  ├─ Event dedup (claim_event)                 │       │
│  │  ├─ Trigger rules (whitelist / @mention / DM) │       │
│  │  ├─ Command interceptor (CommandRegistry)     │       │
│  │  └─ FeishuDeliveryService  → ack / reply      │       │
│  └───────────────────────────────────────────────┘       │
│                        │ regular messages               │
│  ┌─ Session Management ──────────────────────────┐       │
│  │  SessionManager (channels/session/)            │       │
│  │  ├─ Debounce + batch window                   │       │
│  │  └─ SessionContextService → create/switch/fork│       │
│  └───────────────────────────────────────────────┘       │
│                        │ batch                           │
│  ┌─ Agent Executor ──────────────────────────────┐       │
│  │  AgentExecutor (agent_executor.py)            │       │
│  │  ├─ RuntimeAgentPool → cached instance         │       │
│  │  └─ scheduler.route_root_input() → stream      │       │
│  └───────────────────────────────────────────────┘       │
│                        │ stream output                   │
│  ┌─ Delivery ────────────────────────────────────┐       │
│  │  FeishuDeliveryService (delivery_service.py)  │       │
│  │  └─ FeishuApiClient (api_client.py)           │       │
│  └───────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────────┘
```

## Module Reference

| Module | File | Responsibility |
|--------|------|----------------|
| Service | `service.py` | Top-level orchestration, component wiring via factory |
| Factory | `factory.py` | `FeishuServiceFactory` — DI container for all components |
| Connection | `connection.py` | WebSocket long-connection via Lark SDK |
| Inbound Handler | `inbound_handler.py` | Event decryption, dedup, trigger rules, command routing |
| Message Parser | `message_parser.py` | Envelope → `InboundMessage`, sender resolution |
| Content Extractor | `content_extractor.py` | Text/post/image/file/audio/sticker/interactive card parsing |
| Message Builder | `message_builder.py` | `InboundMessage` → `UserMessage`, attachment download |
| Delivery Service | `delivery_service.py` | Reply/message/ack/command delivery to Feishu API |
| API Client | `api_client.py` | Feishu OpenAPI wrapper (auth, messaging, attachments) |
| Group History | `group_history_store.py` | In-memory recent group message cache for context |
| Session | `channels/session/` | Session lifecycle, context resolution, mutation plans |
| Agent Executor | `agent_executor.py` | `routeroute_root_input()` delegation with session tracking |
| Agent Pool | `runtime_agent_pool.py` | Agent instance cache with SHA1 config fingerprint |
| Commands | `commands/` | Slash command system (see below) |
| Store | `store/` | Channel metadata persistence |

## Message Flow

```
User sends message in Feishu
  │
  ▼
WebSocket event received (FeishuConnection)
  │
  ▼
Envelope decryption + signature validation
  │
  ▼
FeishuInboundHandler.process_envelope()
  ├─ parse_inbound_message() → InboundMessage
  ├─ claim_event() → deduplicate by event_id
  ├─ _should_trigger() → whitelist / @mention / DM check
  │   └─ false → ignore (silent drop)
  │
  ▼
Command check — CommandRegistry.try_parse()
  │
  ├─ Match → execute command handler → reply → done
  │
  └─ No match → regular message
       │
       ├─ send_ack() → "Typing" reaction or fallback text
       ├─ SessionManager.enqueue() → debounce
       │
       ▼
       _on_batch_ready()
         │
         ├─ _build_user_message() → attach download if any
         ├─ get_or_create_current_session()
         ├─ get_or_create_runtime_agent()
         ├─ AgentExecutor.execute() → scheduler.route_root_input()
         │
         ▼
         Stream output
           ├─ First chunk → _deliver_reply() (reply message)
           ├─ Subsequent chunks → _deliver_message() (new message)
           └─ Group chat → prefix with @mention
```

## Commands

Command handlers reside in `console/server/channels/feishu/commands/` and are wired via `build_feishu_command_registry()`.

### Session Management (`session.py`)

| Command | Description |
|---------|-------------|
| `/new` | Create a new session, resets current conversation context |
| `/list` | List historical sessions with overview metrics |
| `/switch {session_id}` | Switch to a specific session |
| `/fork` | Fork current session into a new one with weak lineage |

### Context Insight (`context.py`)

| Command | Description |
|---------|-------------|
| `/context` | View current session's system prompt and conversation overview |
| `/status` | View current conversation metrics (tokens, cost, duration) |

### Scheduler Control (`scheduler.py`)

| Command | Description |
|---------|-------------|
| `/agents` | List all scheduler agent states |
| `/detail {state_id}` | View detailed scheduler state for an agent |
| `/steer {state_id} {message}` | Send a steering message to a running agent |
| `/cancel {state_id}` | Cancel an active agent execution |
| `/resume {state_id} {message}` | Resume a paused persistent agent |

### Automatic

| Command | Description |
|---------|-------------|
| `/help` | Display all available commands |

## Storage

Feishu channel metadata (sessions, chat contexts, event deduplication) is persisted via two backends:

| Backend | Description |
|---------|-------------|
| `memory` | In-memory store — lost on restart |
| `sqlite` | Persistent store with schema migrations |

The backend is selected automatically based on the `AGIWO_CONSOLE_METADATA_STORAGE_TYPE` setting. When set to `"sqlite"`, the persistent SQLite store is used; otherwise the in-memory store is used.

SQLite tables:
- `feishu_chat_contexts` — chat context records
- `feishu_sessions` — session records with scheduler state binding
- `feishu_claimed_events` — event deduplication log

## Configuration

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `AGIWO_CONSOLE_FEISHU_ENABLED` | Enable Feishu channel | `False` |
| `AGIWO_CONSOLE_FEISHU_APP_ID` | Feishu app ID | Required |
| `AGIWO_CONSOLE_FEISHU_APP_SECRET` | Feishu app secret | Required |
| `AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN` | Verification token | `""` |
| `AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY` | Encryption key | `""` |
| `AGIWO_CONSOLE_FEISHU_BOT_OPEN_ID` | Bot open ID | `""` |
| `AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME` | Default agent name | Required |
| `AGIWO_CONSOLE_FEISHU_WHITELIST_OPEN_IDS` | Whitelisted user open IDs | `[]` |
| `AGIWO_CONSOLE_FEISHU_DEBOUNCE_MS` | Message debounce time | `3000` |
| `AGIWO_CONSOLE_FEISHU_MAX_BATCH_WINDOW_MS` | Max batch wait window | `15000` |
| `AGIWO_CONSOLE_FEISHU_SCHEDULER_WAIT_TIMEOUT` | Scheduler wait timeout | `900` |
| `AGIWO_CONSOLE_FEISHU_ACK_REACTION_EMOJI` | Ack reaction emoji | `"🤖"` |
| `AGIWO_CONSOLE_FEISHU_ACK_FALLBACK_TEXT` | Ack fallback text | `"收到，正在处理。"` |

## Error Handling

User-facing error mapping in `FeishuChannelService._to_user_facing_error()`:

| Error | User Message |
|-------|--------------|
| `PreviousTaskRunningError` | "上一条任务仍在处理中，请稍后再试。" |
| `BaseAgentNotFoundError` | "指定的 Agent '{id}' 不存在或已被删除，请验证该 Agent 是否存在于系统中。" |
| `DefaultAgentNameNotFoundError` | "当前默认 Agent 名称 '{name}' 不存在，请检查 AGIWO_CONSOLE_FEISHU_DEFAULT_AGENT_NAME。" |

## Debugging

### Check Connection Status

```bash
curl http://localhost:8422/api/channels/feishu/status
```

Response:
```json
{
  "mode": "long_connection",
  "long_connection_alive": true,
  "session_count": 5
}
```

### Log Keywords

| Log Key | Meaning |
|---------|---------|
| `feishu_message_received` | New message received |
| `feishu_message_ignored` | Message ignored (duplicate / no trigger) |
| `feishu_ack_sent` | Acknowledgment sent |
| `feishu_response_sent` | Agent response delivered |
| `feishu_command_received` | Command received |
| `feishu_long_connection_started` | Long-connection established |
| `feishu_long_connection_failed` | Long-connection failed |
