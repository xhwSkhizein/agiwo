"""
Feishu channel service factory — dependency injection and component wiring.
"""

import tempfile
from pathlib import Path

from agiwo.scheduler.scheduler import Scheduler

from server.channels.agent_executor import AgentExecutor
from server.channels.feishu.api_client import FeishuApiClient
from server.channels.feishu.commands import build_feishu_command_registry
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.inbound_handler import FeishuInboundHandler
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
from server.channels.session import SessionContextService, SessionManager
from server.config import ConsoleConfig
from server.services.agent_registry import AgentRegistry


class FeishuServiceComponents:
    """Container for all Feishu channel service components."""

    def __init__(
        self,
        *,
        api: FeishuApiClient,
        store: FeishuChannelStoreBackend,
        parser: FeishuMessageParser,
        connection: FeishuConnection,
        session_service: SessionContextService,
        agent_pool: RuntimeAgentPool,
        executor: AgentExecutor,
        session_manager: SessionManager,
        tmp_dir: Path,
        message_builder: FeishuUserMessageBuilder,
        delivery_service: FeishuDeliveryService,
        inbound_handler: FeishuInboundHandler,
        bot_open_id: str,
    ) -> None:
        self.api = api
        self.store = store
        self.parser = parser
        self.connection = connection
        self.session_service = session_service
        self.agent_pool = agent_pool
        self.executor = executor
        self.session_manager = session_manager
        self.tmp_dir = tmp_dir
        self.message_builder = message_builder
        self.delivery_service = delivery_service
        self.inbound_handler = inbound_handler
        self.bot_open_id = bot_open_id


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

        session_manager = SessionManager(
            on_batch_ready=None,  # Will be set by BaseChannelService
            debounce_ms=config.feishu_debounce_ms,
            max_batch_window_ms=config.feishu_max_batch_window_ms,
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
            truncate_for_log=lambda x: (
                x[:1200] + "...[truncated]" if len(x) > 1200 else x
            ),
        )

        command_registry = build_feishu_command_registry(
            session_service=session_service,
            agent_pool=agent_pool,
            executor=executor,
            session_manager=session_manager,
            scheduler=scheduler,
            agent_registry=agent_registry,
            console_config=config,
        )

        inbound_handler = FeishuInboundHandler(
            channel_instance_id=config.feishu_channel_instance_id,
            default_agent_name=config.feishu_default_agent_name,
            whitelist_open_ids=set(config.feishu_whitelist_open_ids),
            parser=parser,
            content_extractor=content_extractor,
            group_history_store=group_history_store,
            store=store,
            session_service=session_service,
            session_manager=session_manager,
            command_registry=command_registry,
            delivery_service=delivery_service,
            truncate_for_log=lambda x: (
                x[:1200] + "...[truncated]" if len(x) > 1200 else x
            ),
        )

        return FeishuServiceComponents(
            api=api,
            store=store,
            parser=parser,
            connection=connection,
            session_service=session_service,
            agent_pool=agent_pool,
            executor=executor,
            session_manager=session_manager,
            tmp_dir=tmp_dir,
            message_builder=message_builder,
            delivery_service=delivery_service,
            inbound_handler=inbound_handler,
            bot_open_id=config.feishu_bot_open_id,
        )


__all__ = ["FeishuServiceComponents", "FeishuServiceFactory"]
