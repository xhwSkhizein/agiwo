import asyncio


def apply_steering_messages(
    messages: list[dict],
    steering_queue: asyncio.Queue[object] | None,
) -> None:
    if steering_queue is None or steering_queue.empty():
        return

    parts: list[str] = []
    while not steering_queue.empty():
        try:
            parts.append(str(steering_queue.get_nowait()))
        except asyncio.QueueEmpty:
            break

    if not parts:
        return

    steering_text = "\n".join(parts)
    tag = f"\n\n<system-steering>{steering_text}</system-steering>"
    last_message = messages[-1] if messages else None
    if last_message and last_message.get("role") in ("user", "tool"):
        content = last_message.get("content", "")
        if isinstance(content, str):
            last_message["content"] = content + tag
        elif isinstance(content, list):
            content.append({"type": "text", "text": tag})
        else:
            last_message["content"] = tag
        return

    messages.append({"role": "user", "content": steering_text})


__all__ = ["apply_steering_messages"]
