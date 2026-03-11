"""Feishu channel metadata store abstractions and factory."""

from typing import Protocol

from server.channels.feishu.memory_store import InMemoryFeishuChannelStore
from server.channels.feishu.sqlite_store import SqliteFeishuChannelStore
from server.channels.models import ChannelChatSessionStore


class FeishuChannelStoreBackend(ChannelChatSessionStore, Protocol):
    async def connect(self) -> None: ...
    async def close(self) -> None: ...
    async def claim_event(self, channel_instance_id: str, event_id: str) -> bool: ...


def create_feishu_channel_store(
    *,
    db_path: str,
    use_persistent_store: bool,
) -> FeishuChannelStoreBackend:
    if use_persistent_store:
        return SqliteFeishuChannelStore(db_path=db_path)
    return InMemoryFeishuChannelStore()


class FeishuChannelStore:
    def __new__(
        cls,
        db_path: str,
        use_persistent_store: bool,
    ) -> FeishuChannelStoreBackend:
        return create_feishu_channel_store(
            db_path=db_path,
            use_persistent_store=use_persistent_store,
        )


__all__ = [
    "FeishuChannelStore",
    "FeishuChannelStoreBackend",
    "create_feishu_channel_store",
]
