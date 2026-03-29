"""
Feishu channel service factory — dependency injection and component wiring.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

from agiwo.scheduler.engine import Scheduler

from server.channels.agent_executor import AgentExecutor
from server.channels.base import truncate_for_log
from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.message_builder import (
    FeishuAttachmentResolver,
    FeishuUserMessageBuilder,
)
from server.channels.feishu.message_parser import (
    FeishuMessageParser,
    FeishuSenderResolver,
)
from server.channels.feishu.store import (
    FeishuChannelStoreBackend,
    create_feishu_channel_store,
)
from server.channels.runtime_agent_pool import RuntimeAgentPool
from server.channels.session import SessionContextService
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry


@dataclass
class FeishuServiceComponents:
    """Container for all Feishu channel service components."""

    api: FeishuApiClient
    store: FeishuChannelStoreBackend
    parser: FeishuMessageParser
    connection: FeishuConnection
    session_service: SessionContextService
    agent_pool: RuntimeAgentPool
    executor: AgentExecutor
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
        api = FeishuApiClient(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            api_base_url=config.feishu_api_base_url,
        )

        store = create_feishu_channel_store(
            db_path=config.sqlite_db_path,
            use_persistent_store=config.metadata_storage_type == "sqlite",
        )

        content_extractor = FeishuContentExtractor()
        group_history_store = FeishuGroupHistoryStore()
        sender_resolver = FeishuSenderResolver(api=api)

        parser = FeishuMessageParser(
            content_extractor=content_extractor,
            sender_resolver=sender_resolver,
            channel_instance_id=config.feishu_channel_instance_id,
            bot_open_id=config.feishu_bot_open_id,
        )

        connection = FeishuConnection(
            app_id=config.feishu_app_id,
            app_secret=config.feishu_app_secret,
            encrypt_key=config.feishu_encrypt_key,
            verification_token=config.feishu_verification_token,
            sdk_log_level=config.feishu_sdk_log_level,
        )

        session_service = SessionContextService(
            store=store,
            agent_registry=agent_registry,
            default_agent_name=config.feishu_default_agent_name,
        )

        agent_pool = RuntimeAgentPool(
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
            store=store,
        )

        executor = AgentExecutor(
            scheduler=scheduler,
            store=store,
            timeout=config.feishu_scheduler_wait_timeout,
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
            bot_open_id=config.feishu_bot_open_id,
        )


__all__ = ["FeishuServiceComponents", "FeishuServiceFactory"]
