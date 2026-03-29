from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agiwo.agent import ContentPart, ContentType
from server.channels.feishu.commands.base import CommandResult
from server.channels.feishu.content_extractor import FeishuContentExtractor
from server.channels.feishu.connection import FeishuConnection
from server.channels.feishu.factory import FeishuServiceComponents
from server.channels.feishu.delivery_service import FeishuDeliveryService
from server.channels.feishu.group_history_store import FeishuGroupHistoryStore
from server.channels.feishu.inbound_handler import FeishuInboundHandler
from server.channels.feishu.message_builder import (
    FeishuAttachmentResolver,
    FeishuUserMessageBuilder,
)
from server.channels.feishu.message_parser import (
    FeishuInboundEnvelope,
    FeishuMessageParser,
    FeishuSenderResolver,
)
from server.channels.session.models import Attachment, BatchContext, InboundMessage
from server.config import ConsoleConfig


def _inbound_message(
    *,
    chat_type: str = "p2p",
    is_at_bot: bool = False,
    text: str = "hello",
    attachments: list[Attachment] | None = None,
) -> InboundMessage:
    return InboundMessage(
        channel_instance_id="feishu-main",
        event_id="evt-1",
        message_id="msg-1",
        chat_id="chat-1",
        chat_type=chat_type,
        sender_id="user-1",
        sender_name="Alice",
        text=text,
        event_time_ms=1,
        raw_payload={"event": "payload"},
        is_at_bot=is_at_bot,
        attachments=attachments or [],
    )


def _batch_context(*, chat_type: str = "p2p") -> BatchContext:
    return BatchContext(
        chat_context_scope_id="scope-1",
        channel_instance_id="feishu-main",
        chat_id="chat-1",
        chat_type=chat_type,
        trigger_user_id="user-1",
        trigger_message_id="msg-1",
        base_agent_id="agent-1",
    )


def _envelope(
    *,
    event_type: str = "im.message.receive_v1",
) -> FeishuInboundEnvelope:
    return FeishuInboundEnvelope(
        event_type=event_type,
        event_id="evt-1",
        message_id="msg-1",
        chat_id="chat-1",
        chat_type="p2p",
        sender_open_id="user-1",
        message_type="text",
        content='{"text":"hello"}',
        event_time_ms=1,
    )


@pytest.mark.asyncio
async def test_attachment_resolver_downloads_image_to_tmp_dir(
    tmp_path: Path,
) -> None:
    api = SimpleNamespace(
        download_image=AsyncMock(return_value=b"\x89PNG\r\n\x1a\npayload"),
        download_message_resource=AsyncMock(),
    )
    resolver = FeishuAttachmentResolver(api=api, tmp_dir=tmp_path)  # type: ignore[arg-type]

    parts = await resolver.resolve_attachments(
        _inbound_message(
            attachments=[Attachment(type="image", key="img-1", name="sample.png")]
        )
    )

    assert len(parts) == 1
    part = parts[0]
    assert part.type == ContentType.IMAGE
    assert part.mime_type == "image/png"
    assert part.url is not None
    assert tmp_path.joinpath("img-1.png").exists()
    api.download_image.assert_awaited_once_with("img-1")
    api.download_message_resource.assert_not_called()


def test_connection_builds_typed_inbound_envelope_from_sdk_event() -> None:
    connection = FeishuConnection(
        app_id="app-id",
        app_secret="app-secret",
        encrypt_key="encrypt-key",
        verification_token="token",
        sdk_log_level="info",
    )
    event = SimpleNamespace(
        header=SimpleNamespace(
            event_id="evt-1",
            event_type="im.message.receive_v1",
        ),
        event=SimpleNamespace(
            message=SimpleNamespace(
                message_id="msg-1",
                chat_id="chat-1",
                chat_type="p2p",
                message_type="text",
                content='{"text":"hello"}',
                thread_id="thread-1",
                create_time="123",
                mentions=[SimpleNamespace(id=SimpleNamespace(open_id="ou_123"))],
            ),
            sender=SimpleNamespace(sender_id=SimpleNamespace(open_id="user-1")),
        ),
    )

    envelope = connection._build_inbound_envelope_from_sdk_event(event)

    assert envelope is not None
    assert envelope.event_type == "im.message.receive_v1"
    assert envelope.event_id == "evt-1"
    assert envelope.message_id == "msg-1"
    assert envelope.chat_id == "chat-1"
    assert envelope.sender_open_id == "user-1"
    assert envelope.event_time_ms == 123
    assert envelope.mentions[0].open_id == "ou_123"
    assert envelope.to_payload_dict()["event"]["message"]["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_message_parser_consumes_typed_envelope() -> None:
    parser = FeishuMessageParser(
        content_extractor=FeishuContentExtractor(),
        sender_resolver=FeishuSenderResolver(
            api=SimpleNamespace(get_user_display_name=AsyncMock(return_value="Alice"))  # type: ignore[arg-type]
        ),
        channel_instance_id="feishu-main",
        bot_open_id="bot-1",
    )
    envelope = FeishuInboundEnvelope(
        event_type="im.message.receive_v1",
        event_id="evt-1",
        message_id="msg-1",
        chat_id="chat-1",
        chat_type="group",
        sender_open_id="user-1",
        message_type="text",
        content='{"text":"<at user_id=\\"bot-1\\">Bot</at> hello"}',
        event_time_ms=123,
    )

    inbound = await parser.parse_inbound_message(envelope)

    assert inbound is not None
    assert inbound.message_id == "msg-1"
    assert inbound.sender_name == "Alice"
    assert inbound.raw_payload["event"]["message"]["message_id"] == "msg-1"


@pytest.mark.asyncio
async def test_sender_resolver_caches_display_name_lookup() -> None:
    api = SimpleNamespace(get_user_display_name=AsyncMock(return_value="Alice"))
    resolver = FeishuSenderResolver(api=api)  # type: ignore[arg-type]

    first = await resolver.resolve_sender_name("user-1")
    second = await resolver.resolve_sender_name("user-1")

    assert first == "Alice"
    assert second == "Alice"
    api.get_user_display_name.assert_awaited_once_with("user-1")


def test_group_history_store_tracks_recent_group_lines() -> None:
    history_store = FeishuGroupHistoryStore(window_ms=10**15)
    message = _inbound_message(chat_type="group", text="  hello   world  ")

    history_store.record_message(message, normalized_text="hello world")

    assert history_store.get_history_lines("chat-1", exclude_message_ids=set()) == [
        "Alice: hello world"
    ]


@pytest.mark.asyncio
async def test_message_builder_cleans_attachment_placeholders_and_adds_dm_history() -> (
    None
):
    resolver = SimpleNamespace(
        resolve_attachments=AsyncMock(
            return_value=[
                ContentPart(
                    type=ContentType.IMAGE,
                    url="/tmp/example.png",
                    mime_type="image/png",
                )
            ]
        )
    )
    content_extractor = SimpleNamespace(
        normalize_message_text=lambda text: text.strip(),
    )
    group_history_store = SimpleNamespace(get_history_lines=Mock(return_value=[]))
    builder = FeishuUserMessageBuilder(
        content_extractor=content_extractor,
        group_history_store=group_history_store,
        attachment_resolver=resolver,
    )

    message = await builder.build_user_message(
        _batch_context(),
        [
            _inbound_message(text=" earlier "),
            _inbound_message(text="[图片] 请看"),
        ],
    )

    assert message.content[0].text == "请看"
    assert message.content[1].type == ContentType.IMAGE
    assert message.context is not None
    assert message.context.metadata["recent_dm_messages"] == ["earlier"]


@pytest.mark.asyncio
async def test_delivery_service_falls_back_to_create_message_for_group_reply() -> None:
    api = SimpleNamespace(
        reply_text=AsyncMock(side_effect=RuntimeError("reply failed")),
        create_text_message=AsyncMock(),
        add_message_reaction=AsyncMock(),
    )
    delivery = FeishuDeliveryService(
        api=api,
        config=ConsoleConfig(),
        truncate_for_log=lambda text: text,
    )

    await delivery.deliver_reply(_batch_context(chat_type="group"), "done")

    api.reply_text.assert_awaited_once_with(
        "msg-1",
        '<at user_id="user-1">发起人</at> done',
    )
    api.create_text_message.assert_awaited_once_with(
        "chat-1",
        '<at user_id="user-1">发起人</at> done',
    )


@pytest.mark.asyncio
async def test_inbound_handler_executes_commands_before_ack_or_enqueue() -> None:
    parser = SimpleNamespace(
        parse_inbound_message=AsyncMock(return_value=_inbound_message()),
    )
    content_extractor = SimpleNamespace(
        normalize_message_text=Mock(return_value="hello")
    )
    group_history_store = SimpleNamespace(record_message=Mock())
    store = SimpleNamespace(claim_event=AsyncMock(return_value=True))
    session_service = SimpleNamespace(
        resolve_default_agent_config=AsyncMock(
            return_value=SimpleNamespace(id="agent-1")
        ),
        get_chat_context_and_current_session=AsyncMock(return_value=(None, None)),
    )
    session_manager = SimpleNamespace(enqueue=AsyncMock())
    delivery_service = SimpleNamespace(
        send_ack=AsyncMock(),
        send_command_response=AsyncMock(),
    )
    handler = SimpleNamespace(
        name="ping",
        execute=AsyncMock(return_value=CommandResult(text="pong")),
    )
    command_registry = SimpleNamespace(try_parse=Mock(return_value=(handler, "")))
    inbound_handler = FeishuInboundHandler(
        channel_instance_id="feishu-main",
        default_agent_name="default",
        whitelist_open_ids=set(),
        parser=parser,
        content_extractor=content_extractor,
        group_history_store=group_history_store,
        store=store,
        session_service=session_service,
        session_manager=session_manager,
        command_registry=command_registry,
        delivery_service=delivery_service,
        truncate_for_log=lambda text: text,
    )

    result = await inbound_handler.process_envelope(_envelope())

    assert result == {"msg": "command_executed"}
    handler.execute.assert_awaited_once()
    delivery_service.send_command_response.assert_awaited_once_with(
        _inbound_message(),
        "pong",
    )
    delivery_service.send_ack.assert_not_called()
    session_manager.enqueue.assert_not_called()
    group_history_store.record_message.assert_called_once()


@pytest.mark.asyncio
async def test_inbound_handler_acknowledges_and_enqueues_non_command_messages() -> None:
    inbound = _inbound_message(chat_type="group", is_at_bot=True, text="hello bot")
    parser = SimpleNamespace(
        parse_inbound_message=AsyncMock(return_value=inbound),
    )
    content_extractor = SimpleNamespace(
        normalize_message_text=Mock(return_value="hello bot")
    )
    group_history_store = SimpleNamespace(record_message=Mock())
    store = SimpleNamespace(claim_event=AsyncMock(return_value=True))
    session_service = SimpleNamespace(
        resolve_default_agent_config=AsyncMock(
            return_value=SimpleNamespace(id="agent-1")
        ),
    )
    session_manager = SimpleNamespace(enqueue=AsyncMock())
    delivery_service = SimpleNamespace(
        send_ack=AsyncMock(),
        send_command_response=AsyncMock(),
    )
    inbound_handler = FeishuInboundHandler(
        channel_instance_id="feishu-main",
        default_agent_name="default",
        whitelist_open_ids=set(),
        parser=parser,
        content_extractor=content_extractor,
        group_history_store=group_history_store,
        store=store,
        session_service=session_service,
        session_manager=session_manager,
        command_registry=SimpleNamespace(try_parse=Mock(return_value=None)),
        delivery_service=delivery_service,
        truncate_for_log=lambda text: text,
    )

    result = await inbound_handler.process_envelope(
        FeishuInboundEnvelope(
            event_type="im.message.receive_v1",
            event_id="evt-1",
            message_id="msg-1",
            chat_id="chat-1",
            chat_type="group",
            sender_open_id="user-1",
            message_type="text",
            content='{"text":"hello bot"}',
            event_time_ms=1,
        )
    )

    assert result == {"msg": "ok"}
    delivery_service.send_ack.assert_awaited_once_with(inbound)
    session_manager.enqueue.assert_awaited_once()
    args = session_manager.enqueue.await_args.args
    assert args[0] == "feishu:feishu-main:group:chat-1:user:user-1"
    assert args[1] is inbound
    assert args[2].base_agent_id == "agent-1"
    delivery_service.send_command_response.assert_not_called()
    group_history_store.record_message.assert_called_once_with(
        inbound,
        normalized_text="hello bot",
    )


@pytest.mark.asyncio
async def test_feishu_batch_uses_shared_conversation_service() -> None:
    """Verify that the workspace conversation service is wired through the Feishu factory."""
    components = FeishuServiceComponents(
        api=Mock(),
        store=Mock(),
        parser=Mock(),
        connection=Mock(),
        session_service=Mock(),
        agent_pool=Mock(),
        executor=Mock(),
        tmp_dir=Path("/tmp/test"),
        message_builder=Mock(),
        delivery_service=Mock(),
        inbound_handler=Mock(),
        bot_open_id="bot-1",
        workspace_conversation=Mock(),
    )
    assert components.workspace_conversation is not None
