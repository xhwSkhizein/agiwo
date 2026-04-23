# Feishu ACK Reaction Repair

**Goal:** Repair Feishu channel acknowledgements so normal inbound messages get a lightweight ACK reliably, with reaction first and text fallback only when needed.

**Tech Stack:** Python 3.10+, Console Feishu channel, httpx, pytest

## Context

The current Feishu ACK path in `console/server/channels/feishu/delivery_service.py` is intended to:

1. add a reaction to the triggering message
2. fall back to a short text reply when reaction fails

Production logs on April 23, 2026 showed both branches failing:

- reaction failed with `code=99992402 field validation failed`
- fallback reply failed with `code=230001 invalid message content`

Direct API reproduction against the deployed app on April 23, 2026 clarified the reaction failure. The deployed environment sets:

- `AGIWO_CONSOLE_CHANNELS__FEISHU__ACK_REACTION_EMOJI=Typing`

Using the documented request body shape:

- `Typing` returned `231002 The operator has no permission to react on the specific message`
- `OnIt` succeeded on the same message
- `SMILE` also succeeded on the same message

That means the current default is not a safe ACK reaction for this bot context, even though Feishu documents `Typing` in the reaction catalog.

The fallback reply failure was not reproduced with a direct minimal API call. A normal text reply and a normal create-message call both succeeded against the same chat. The safer interpretation is:

- the primary root cause is the unsafe default ACK reaction
- the fallback path is still worth hardening because it currently stops after one reply attempt

## Decision

Keep the current ACK product behavior:

- normal inbound messages should try a lightweight reaction first
- if reaction cannot be sent, the system should still acknowledge receipt with a short text message

Do not introduce a new ACK strategy abstraction. The problem is not missing flexibility; it is an unsafe default reaction choice and an incomplete fallback chain.

## Design

### 1. Configuration

Replace the default Feishu ACK reaction value with a documented reaction `emoji_type` that works for the deployed bot context.

Recommended default:

- `OnIt`

Rationale:

- it is listed in Feishu's documented reaction catalog
- it matches the intended semantic of "received and working on it"
- it preserves the lightweight ACK UX better than a generic smile or thumbs-up

Existing explicit operator overrides should continue to work. The code should not silently translate arbitrary legacy aliases or maintain a hidden compatibility table. Operators who override the reaction value keep responsibility for choosing an emoji that is valid for their tenant and bot context.

### 2. Delivery Flow

Keep `FeishuDeliveryService.send_ack()` as the only ACK owner and make its control flow explicit:

1. try `add_message_reaction(message_id, emoji_type)`
2. if reaction fails, try `reply_text(message_id, fallback_text)`
3. if reply fails, try `create_text_message(chat_id, fallback_text)`

Logging should remain structured and make the selected ACK path obvious:

- `feishu_ack_sent` with `mode=reaction`
- `feishu_ack_sent` with `mode=reply_fallback`
- `feishu_ack_sent` with `mode=create_message_fallback`
- warning events for each failed step

This keeps the delivery contract local to the Feishu channel layer and avoids spreading ACK retry logic into the inbound handler.

### 3. API Surface

No new public API is needed.

Changes stay inside the existing Feishu channel boundary:

- `console/server/config.py`
- `console/server/channels/feishu/delivery_service.py`
- tests under `console/tests/`

`FeishuApiClient` does not need a new method because both fallback operations already exist.

## Error Handling

The ACK path should be best-effort and non-blocking:

- failure to add a reaction must not stop message enqueue
- failure to reply must not stop message enqueue
- failure to create a fallback message must still not stop message enqueue

The handler should continue to enqueue the inbound message after `send_ack()` returns, matching the current non-fatal semantics.

This design intentionally does not add retries or exponential backoff. ACK is latency-sensitive and should stay simple. If all three attempts fail, logs are the observability mechanism.

## Testing

Add focused regression coverage in `console/tests/test_feishu_service_components.py`:

1. reaction success:
   `send_ack()` sends reaction only
2. reaction failure, reply success:
   `send_ack()` falls back to `reply_text()`
3. reaction failure, reply failure:
   `send_ack()` falls back to `create_text_message()`

The tests should assert the exact call ordering so the intended ACK ladder does not regress later.

## Risks

The main product tradeoff is that when reaction fails, users may still see a visible fallback message. That is acceptable because visible acknowledgement is better than silent failure.

There is still a possibility that a tenant-specific Feishu policy rejects both reply and create-message fallback. This design does not try to mask that case; it only guarantees the code will use all available ACK paths before giving up.

## Acceptance Criteria

1. The default ACK reaction value is changed from `Typing` to a reaction that succeeds for the deployed bot context.
2. Normal inbound messages no longer fail immediately because of the default reaction config.
3. If reaction fails, ACK falls back to text reply.
4. If text reply fails, ACK falls back to creating a new message in the chat.
5. The Feishu regression tests cover all three ACK branches.
