# Feishu Integration

The Console includes a full Feishu (Lark) channel integration for real-time messaging with agents.

## Features

- WebSocket-based real-time message receiving
- Direct and group message support
- Command parsing and routing
- Message history storage (memory or SQLite)
- SSE-based response delivery
- Rich message formatting

## Setup

### 1. Create a Feishu App

1. Go to [Feishu Open Platform](https://open.feishu.cn/)
2. Create a new app
3. Enable the following permissions:
   - `im:message:receive` — Receive messages
   - `im:message:send_as_bot` — Send messages as bot
   - `im:chat:readonly` — Read chat info
   - `contact:user.id:readonly` — Read user IDs

### 2. Configure Environment

Add to your `.env`:

```bash
AGIWO_CONSOLE_FEISHU_APP_ID=cli_xxxx
AGIWO_CONSOLE_FEISHU_APP_SECRET=xxxx
AGIWO_CONSOLE_FEISHU_ENCRYPT_KEY=xxxx      # Optional
AGIWO_CONSOLE_FEISHU_VERIFICATION_TOKEN=xxxx
```

### 3. Start the Console

The Feishu channel starts automatically with the Console server:

```bash
cd console
uv run uvicorn server.app:app --reload --env-file .env
```

## Architecture

```
Feishu Platform
    │
    │ WebSocket
    ▼
FeishuConnection (connection.py)
    │
    ▼
FeishuService (service.py)
    │
    ├── MessageParser (message_parser.py)
    │     └── Parse envelope, extract sender, content
    │
    ├── MessageBuilder (message_builder.py)
    │     └── Build UserMessage from parsed content
    │
    ├── InboundHandler (inbound_handler.py)
    │     └── Route to command handlers or agent executor
    │
    ├── CommandHandlers (commands/)
    │     ├── /status — Agent status
    │     ├── /session — Session management
    │     └── /scheduler — Scheduler control
    │
    ├── AgentExecutor (agent_executor.py)
    │     └── Execute agent, handle scheduler state routing
    │
    └── DeliveryService (delivery_service.py)
          └── Send responses back via Feishu API
```

## Message Flow

```
User sends message in Feishu
  │
  ▼
WebSocket event received
  │
  ▼
MessageParser extracts content and sender
  │
  ├── If command (/status, /session, etc.)
  │     └── CommandHandler processes and responds
  │
  └── If regular message
        │
        ▼
      SessionManager resolves session
        │
        ▼
      AgentExecutor.execute()
        │
        ├── Submit to Scheduler
        ├── Stream response via SSE
        └── DeliveryService sends reply
```

## Commands

| Command | Description |
|---------|-------------|
| `/status` | Show agent status and metrics |
| `/session info` | Show current session details |
| `/session reset` | Reset the current session |
| `/scheduler status` | Show scheduler state |

## Storage

Feishu message history supports two backends:

| Backend | Description |
|---------|-------------|
| `memory` | In-memory (lost on restart) |
| `sqlite` | Persistent SQLite storage |

Configured via `AGIWO_CONSOLE_FEISHU_STORE_TYPE`.

## Extending

To add a new command:

1. Create a handler in `console/server/channels/feishu/commands/`
2. Extend `BaseCommand`
3. Register in the command router

```python
from console.server.channels.feishu.commands.base import BaseCommand

class MyCommand(BaseCommand):
    async def execute(self, context: CommandContext) -> str:
        return "Command result"
```
