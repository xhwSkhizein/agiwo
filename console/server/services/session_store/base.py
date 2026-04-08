"""Session store protocol definition."""

from typing import Protocol

from server.models.session import (
    ChannelChatContext,
    Session,
    SessionWithContext,
)


class SessionStore(Protocol):
    """Protocol for session storage implementations.

    This is a generic session store used by both Console Web UI and Feishu channel.
    It manages Session and ChannelChatContext entities independently.
    """

    async def connect(self) -> None:
        """Initialize the store connection."""
        ...

    async def close(self) -> None:
        """Close the store connection."""
        ...

    # --- Chat Context operations ---

    async def get_chat_context(self, scope_id: str) -> ChannelChatContext | None:
        """Get chat context by scope_id."""
        ...

    async def upsert_chat_context(self, chat_context: ChannelChatContext) -> None:
        """Insert or update a chat context."""
        ...

    # --- Session operations ---

    async def get_session(self, session_id: str) -> Session | None:
        """Get a session by ID."""
        ...

    async def get_session_with_context(
        self,
        session_id: str,
    ) -> SessionWithContext | None:
        """Get a session with its associated chat context."""
        ...

    async def upsert_session(self, session: Session) -> None:
        """Insert or update a session."""
        ...

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if deleted, False if not found."""
        ...

    async def list_sessions_by_user(
        self, user_open_id: str
    ) -> list[SessionWithContext]:
        """List sessions for a specific user."""
        ...

    async def list_sessions_by_chat_context(
        self, chat_context_scope_id: str
    ) -> list[Session]:
        """List sessions belonging to a chat context."""
        ...

    async def list_sessions_by_base_agent(self, base_agent_id: str) -> list[Session]:
        """List sessions for a specific base agent."""
        ...

    async def list_sessions(self) -> list[Session]:
        """List all sessions."""
        ...
