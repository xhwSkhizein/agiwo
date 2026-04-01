"""Feishu channel metadata store abstractions and factory."""

from typing import Protocol

from server.channels.feishu.store.memory import InMemoryFeishuChannelStore
from server.channels.feishu.store.sqlite import SqliteFeishuChannelStore
from server.models.session import ChannelChatSessionStore


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


__all__ = [
    "FeishuChannelStoreBackend",
    "create_feishu_channel_store",
]
