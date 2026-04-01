"""
Feishu channel service factory — dependency injection and component wiring.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from agiwo.scheduler.engine import Scheduler

from server.channels.utils import truncate_for_log
from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.message_builder import (
    FeishuAttachmentResolver,
    FeishuUserMessageBuilder,
)
from server.channels.feishu.message_parser import FeishuMessageParser
from server.channels.feishu.sender_resolver import FeishuSenderResolver
from server.channels.feishu.store import (
    FeishuChannelStoreBackend,
    create_feishu_channel_store,
)
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry
from server.services.runtime import (
    AgentRuntimeCache,
    SessionContextService,
    SessionRuntimeService,
)


@dataclass
class FeishuServiceComponents:
    """Container for all Feishu channel service components."""

    api: FeishuApiClient
    store: FeishuChannelStoreBackend
    parser: FeishuMessageParser
    connection: FeishuConnection
    session_service: SessionContextService
    agent_pool: AgentRuntimeCache
    executor: SessionRuntimeService
    tmp_dir: Path
    message_builder: FeishuUserMessageBuilder
    delivery_service: FeishuDeliveryService
    content_extractor: FeishuContentExtractor
    group_history_store: FeishuGroupHistoryStore
    bot_open_id: str


class FeishuServiceFactory:
    """Factory for creating Feishu channel service components."""

    @staticmethod
    def create_components(
        *,
        config: ConsoleConfig,
        scheduler: Scheduler,
        agent_registry: AgentRegistry,
    ) -> FeishuServiceComponents:
        """Create all Feishu service components with proper dependency injection."""
        feishu = config.channels.feishu
        api = FeishuApiClient(
            app_id=feishu.app_id,
            app_secret=feishu.app_secret,
            api_base_url=feishu.api_base_url,
        )

        store = create_feishu_channel_store(
            db_path=config.sqlite_db_path,
            use_persistent_store=config.storage.metadata_type == "sqlite",
        )

        content_extractor = FeishuContentExtractor()
        group_history_store = FeishuGroupHistoryStore()
        sender_resolver = FeishuSenderResolver(api=api)

        parser = FeishuMessageParser(
            content_extractor=content_extractor,
            sender_resolver=sender_resolver,
            channel_instance_id=feishu.channel_instance_id,
            bot_open_id=feishu.bot_open_id,
        )

        connection = FeishuConnection(
            app_id=feishu.app_id,
            app_secret=feishu.app_secret,
            encrypt_key=feishu.encrypt_key,
            verification_token=feishu.verification_token,
            sdk_log_level=feishu.sdk_log_level,
        )

        session_service = SessionContextService(
            store=store,
            agent_registry=agent_registry,
            default_agent_name=feishu.default_agent_name,
        )

        agent_pool = AgentRuntimeCache(
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
            session_store=store,
        )

        executor = SessionRuntimeService(
            scheduler=scheduler,
            session_store=store,
            timeout=feishu.scheduler_wait_timeout,
        )

        tmp_dir = Path(tempfile.mkdtemp(prefix="feishu_attachments_"))

        attachment_resolver = FeishuAttachmentResolver(
            api=api,
            tmp_dir=tmp_dir,
        )

        message_builder = FeishuUserMessageBuilder(
            content_extractor=content_extractor,
            group_history_store=group_history_store,
            attachment_resolver=attachment_resolver,
        )

        delivery_service = FeishuDeliveryService(
            api=api,
            config=config,
            truncate_for_log=truncate_for_log,
        )

        return FeishuServiceComponents(
            api=api,
            store=store,
            parser=parser,
            connection=connection,
            session_service=session_service,
            agent_pool=agent_pool,
            executor=executor,
            tmp_dir=tmp_dir,
            message_builder=message_builder,
            delivery_service=delivery_service,
            content_extractor=content_extractor,
            group_history_store=group_history_store,
            bot_open_id=feishu.bot_open_id,
        )


__all__ = ["FeishuServiceComponents", "FeishuServiceFactory"]
