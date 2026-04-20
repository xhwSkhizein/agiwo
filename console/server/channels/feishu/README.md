# Feishu Channel

`console/server/channels/feishu/` is the Console's Feishu adapter. It handles Feishu transport concerns only and delegates session/runtime ownership to the shared runtime services in `console/server/services/runtime/`.

## Current Responsibilities

- maintain the long-connection client
- parse Feishu events into channel message models
- perform deduplication, trigger checks, and command interception
- resolve attachments and sender metadata
- deliver replies, follow-up messages, and acknowledgements
- hand normal messages to `SessionContextService`, `AgentRuntimeCache`, and `SessionRuntimeService`

## Directory Map

```text
feishu/
├── api_client.py
├── commands/
├── connection.py
├── content_extractor.py
├── dedup_store.py
├── delivery_service.py
├── group_history_store.py
├── inbound_handler.py
├── message_builder.py
├── message_parser.py
├── sender_resolver.py
└── service.py
```

## Runtime Collaboration

Regular user messages follow this path:

1. `FeishuConnection` receives an event envelope.
2. `FeishuInboundHandler` parses, deduplicates, and checks whether the message should trigger the bot.
3. command messages are handled inside `commands/`.
4. non-command messages are converted to `UserMessage`.
5. `SessionContextService` resolves or creates the active session.
6. `AgentRuntimeCache` returns the stable runtime agent for that session.
7. `SessionRuntimeService` sends the input through the scheduler and returns a stream or an acknowledgement.
8. `FeishuDeliveryService` sends the result back to Feishu.

## Design Boundary

Keep execution semantics out of the channel package:

- no duplicate scheduler state machine
- no separate runtime-agent ownership model
- no direct run/trace projection logic
