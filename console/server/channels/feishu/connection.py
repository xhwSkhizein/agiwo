"""
Feishu long-connection (WebSocket) management via lark-oapi SDK.

Runs the lark WebSocket client in a dedicated daemon thread with its own
event loop, dispatching incoming message payloads back to the main asyncio
loop via a callback.
"""

import asyncio
import threading
import time
from concurrent.futures import Future
from collections.abc import Awaitable, Callable

from agiwo.utils.logging import get_logger
from server.channels.feishu.message_parser import FeishuInboundEnvelope, FeishuMention

try:
    import lark_oapi as _lark
except ImportError:
    _lark = None

try:
    from lark_oapi.ws import client as _lark_ws_client_module
except ImportError:
    _lark_ws_client_module = None


def _is_sdk_available() -> bool:
    return _lark is not None


def _build_event_handler(
    encrypt_key: str,
    verification_token: str,
    *,
    on_message: Callable[[object], None],
    on_message_read: Callable[[object], None],
) -> object:
    if _lark is None:
        raise RuntimeError("lark_oapi_not_installed")
    return (
        _lark.EventDispatcherHandler.builder(
            encrypt_key,
            verification_token,
        )
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_im_message_message_read_v1(on_message_read)
        .build()
    )


def _create_ws_client(
    *,
    app_id: str,
    app_secret: str,
    event_handler: object,
    log_level: object,
) -> object:
    if _lark is None:
        raise RuntimeError("lark_oapi_not_installed")
    return _lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=log_level,
    )


def _resolve_log_level(level_name: str) -> object:
    if _lark is None:
        return None
    normalized = level_name.upper()
    return getattr(_lark.LogLevel, normalized, _lark.LogLevel.INFO)


def _install_worker_loop(loop: asyncio.AbstractEventLoop) -> None:
    if _lark_ws_client_module is not None:
        _lark_ws_client_module.loop = loop


def _resolve_worker_loop(
    worker_loop: asyncio.AbstractEventLoop | None,
) -> asyncio.AbstractEventLoop | None:
    return worker_loop or getattr(_lark_ws_client_module, "loop", None)


def _disable_auto_reconnect(client: object | None) -> None:
    if client is None:
        return
    try:
        client._auto_reconnect = False
    except AttributeError:
        return


logger = get_logger(__name__)

OnEnvelopeCallback = Callable[[FeishuInboundEnvelope], Awaitable[object]]


class FeishuConnection:
    def __init__(
        self,
        *,
        app_id: str,
        app_secret: str,
        encrypt_key: str,
        verification_token: str,
        sdk_log_level: str,
    ) -> None:
        self._app_id = app_id
        self._app_secret = app_secret
        self._encrypt_key = encrypt_key
        self._verification_token = verification_token
        self._sdk_log_level = sdk_log_level

        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._on_envelope: OnEnvelopeCallback | None = None

        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_ready = threading.Event()
        self._ws_start_error: Exception | None = None
        self._ws_client: object | None = None
        self._closed = False

    def is_alive(self) -> bool:
        return self._ws_thread is not None and self._ws_thread.is_alive()

    async def start(
        self,
        main_loop: asyncio.AbstractEventLoop,
        on_envelope: OnEnvelopeCallback,
    ) -> None:
        if not _is_sdk_available():
            raise RuntimeError("lark_oapi_not_installed")

        self._main_loop = main_loop
        self._on_envelope = on_envelope
        self._closed = False
        self._ws_start_error = None
        self._ws_ready.clear()

        self._ws_thread = threading.Thread(
            target=self._run_worker,
            name="feishu-long-connection",
            daemon=True,
        )
        self._ws_thread.start()

        ready = await asyncio.to_thread(self._ws_ready.wait, 15.0)
        if not ready:
            raise RuntimeError("feishu_long_connection_start_timeout")
        if self._ws_start_error is not None:
            raise RuntimeError(
                f"feishu_long_connection_start_failed: {self._ws_start_error}"
            ) from self._ws_start_error
        if self._ws_thread is not None and not self._ws_thread.is_alive():
            raise RuntimeError("feishu_long_connection_worker_exited_early")

        logger.info("feishu_long_connection_started")

    async def stop(self) -> None:
        logger.info("feishu_long_connection_stopping")
        self._closed = True

        _disable_auto_reconnect(self._ws_client)
        ws_loop = _resolve_worker_loop(self._ws_loop)
        if ws_loop is not None and ws_loop.is_running():
            ws_loop.call_soon_threadsafe(ws_loop.stop)

        thread = self._ws_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 3.0)
        self._ws_thread = None
        self._ws_loop = None
        self._ws_client = None

    # -- Worker thread -------------------------------------------------------

    def _run_worker(self) -> None:
        if not _is_sdk_available():
            self._ws_start_error = RuntimeError("lark_oapi_not_installed")
            self._ws_ready.set()
            return

        thread_loop = asyncio.new_event_loop()
        self._ws_loop = thread_loop
        asyncio.set_event_loop(thread_loop)
        _install_worker_loop(thread_loop)

        try:
            event_handler = _build_event_handler(
                self._encrypt_key,
                self._verification_token,
                on_message=self._on_message,
                on_message_read=self._on_message_read,
            )
            self._ws_client = _create_ws_client(
                app_id=self._app_id,
                app_secret=self._app_secret,
                event_handler=event_handler,
                log_level=_resolve_log_level(self._sdk_log_level),
            )
            self._ws_ready.set()
            self._ws_client.start()
        except RuntimeError as e:
            if self._closed and "Event loop stopped before Future completed" in str(e):
                logger.info("feishu_long_connection_stopped")
                return
            self._ws_start_error = e
            self._ws_ready.set()
            logger.exception("feishu_long_connection_runtime_error", error=str(e))
        except Exception as e:
            self._ws_start_error = e
            self._ws_ready.set()
            logger.exception("feishu_long_connection_failed", error=str(e))
        finally:
            loop = self._ws_loop
            if loop is not None and not loop.is_closed():
                try:
                    self._drain_worker_loop(loop)
                except RuntimeError:
                    pass
                finally:
                    loop.close()
            self._ws_loop = None
            self._ws_client = None

    # -- SDK event handlers --------------------------------------------------

    def _on_message(self, data: object) -> None:
        envelope = self._build_inbound_envelope_from_sdk_event(data)
        if envelope is None:
            return

        loop = self._main_loop
        callback = self._on_envelope
        if loop is None or callback is None:
            return

        future = asyncio.run_coroutine_threadsafe(callback(envelope), loop)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: Future[object]) -> None:
        try:
            _ = future.result()
        except Exception as e:
            logger.exception("feishu_long_connection_event_failed", error=str(e))

    def _on_message_read(self, data: object) -> None:
        _ = data

    # -- Payload conversion --------------------------------------------------

    def _build_inbound_envelope_from_sdk_event(
        self,
        data: object,
    ) -> FeishuInboundEnvelope | None:
        event = getattr(data, "event", None)
        header = getattr(data, "header", None)
        message = getattr(event, "message", None)
        sender = getattr(event, "sender", None)
        sender_id = getattr(sender, "sender_id", None)

        message_id = getattr(message, "message_id", None)
        chat_id = getattr(message, "chat_id", None)
        chat_type = getattr(message, "chat_type", None)
        message_type = getattr(message, "message_type", None)
        sender_open_id = getattr(sender_id, "open_id", None)

        if not isinstance(message_id, str) or not message_id:
            return None
        if not isinstance(chat_id, str) or not chat_id:
            return None
        if not isinstance(chat_type, str) or not chat_type:
            return None
        if not isinstance(message_type, str) or not message_type:
            return None
        if not isinstance(sender_open_id, str) or not sender_open_id:
            return None

        mentions = self._extract_mentions(message)
        content = self._read_message_content(message)
        thread_id = self._read_thread_id(message)
        event_time_ms = self._coerce_event_time_ms(message)
        event_type = self._coerce_event_type(header)
        event_id = self._coerce_event_id(header, message_id)

        return FeishuInboundEnvelope(
            event_type=event_type,
            event_id=event_id,
            message_id=message_id,
            chat_id=chat_id,
            chat_type=chat_type,
            sender_open_id=sender_open_id,
            message_type=message_type,
            content=content,
            event_time_ms=event_time_ms,
            thread_id=thread_id,
            mentions=tuple(mentions),
        )

    def _drain_worker_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def _extract_mentions(self, message: object) -> list[FeishuMention]:
        mentions: list[FeishuMention] = []
        raw_mentions = getattr(message, "mentions", None) or []
        for mention in raw_mentions:
            mention_id = getattr(mention, "id", None)
            mention_open_id = getattr(mention_id, "open_id", None)
            if not isinstance(mention_open_id, str) or not mention_open_id:
                continue
            mentions.append(FeishuMention(open_id=mention_open_id))
        return mentions

    def _read_message_content(self, message: object) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return ""

    def _read_thread_id(self, message: object) -> str | None:
        thread_id = getattr(message, "thread_id", None)
        if isinstance(thread_id, str) and thread_id:
            return thread_id
        return None

    def _coerce_event_time_ms(self, message: object) -> int:
        create_time = getattr(message, "create_time", int(time.time() * 1000))
        try:
            return int(create_time)
        except (TypeError, ValueError):
            return int(time.time() * 1000)

    def _coerce_event_type(self, header: object) -> str:
        event_type = getattr(header, "event_type", "im.message.receive_v1")
        if isinstance(event_type, str) and event_type:
            return event_type
        return "im.message.receive_v1"

    def _coerce_event_id(self, header: object, message_id: str) -> str:
        event_id = getattr(header, "event_id", message_id)
        if isinstance(event_id, str) and event_id:
            return event_id
        return message_id
