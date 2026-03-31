"""Chat session request models."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class CreateSessionRequest(BaseModel):
    chat_context_scope_id: str
    channel_instance_id: str = "console-web"
    user_open_id: str = "console-user"


class SwitchSessionRequest(BaseModel):
    chat_context_scope_id: str
    target_session_id: str


class ForkSessionRequest(BaseModel):
    context_summary: str
