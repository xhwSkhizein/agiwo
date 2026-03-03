"""
Feishu long-connection (WebSocket) management via lark-oapi SDK.

Runs the lark WebSocket client in a dedicated daemon thread with its own
event loop, dispatching incoming message payloads back to the main asyncio
loop via a callback.
"""

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

from agiwo.utils.logging import get_logger

try:
    import lark_oapi as lark
except Exception:
    lark = None

try:
    from lark_oapi.ws import client as lark_ws_client_module
except Exception:
    lark_ws_client_module = None

logger = get_logger(__name__)

OnPayloadCallback = Callable[[dict[str, Any]], Any]


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
        self._on_payload: OnPayloadCallback | None = None

        self._ws_thread: threading.Thread | None = None
        self._ws_loop: asyncio.AbstractEventLoop | None = None
        self._ws_ready = threading.Event()
        self._ws_start_error: Exception | None = None
        self._ws_client: Any = None
        self._closed = False

    def is_alive(self) -> bool:
        return self._ws_thread is not None and self._ws_thread.is_alive()

    async def start(
        self,
        main_loop: asyncio.AbstractEventLoop,
        on_payload: OnPayloadCallback,
    ) -> None:
        if lark is None:
            raise RuntimeError("lark_oapi_not_installed")

        self._main_loop = main_loop
        self._on_payload = on_payload
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

        if self._ws_client is not None:
            try:
                self._ws_client._auto_reconnect = False
            except Exception:
                pass

        ws_loop = self._ws_loop or getattr(lark_ws_client_module, "loop", None)
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
        if lark is None:
            self._ws_start_error = RuntimeError("lark_oapi_not_installed")
            self._ws_ready.set()
            return

        thread_loop = asyncio.new_event_loop()
        self._ws_loop = thread_loop
        asyncio.set_event_loop(thread_loop)
        if lark_ws_client_module is not None:
            lark_ws_client_module.loop = thread_loop

        try:
            event_handler = (
                lark.EventDispatcherHandler.builder(
                    self._encrypt_key,
                    self._verification_token,
                )
                .register_p2_im_message_receive_v1(self._on_message)
                .register_p2_im_message_message_read_v1(self._on_message_read)
                .build()
            )

            self._ws_client = lark.ws.Client(
                self._app_id,
                self._app_secret,
                event_handler=event_handler,
                log_level=self._resolve_log_level(),
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
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass
                finally:
                    loop.close()
            self._ws_loop = None
            self._ws_client = None

    # -- SDK event handlers --------------------------------------------------

    def _on_message(self, data: Any) -> None:
        payload = self._build_payload_from_sdk_event(data)
        if payload is None:
            return

        loop = self._main_loop
        callback = self._on_payload
        if loop is None or callback is None:
            return

        future = asyncio.run_coroutine_threadsafe(callback(payload), loop)
        future.add_done_callback(self._on_future_done)

    def _on_future_done(self, future: Any) -> None:
        try:
            _ = future.result()
        except Exception as e:
            logger.exception("feishu_long_connection_event_failed", error=str(e))

    def _on_message_read(self, data: Any) -> None:
        _ = data

    # -- Payload conversion --------------------------------------------------

    def _build_payload_from_sdk_event(self, data: Any) -> dict[str, Any] | None:
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

        mentions_payload: list[dict[str, Any]] = []
        mentions = getattr(message, "mentions", None) or []
        for mention in mentions:
            mention_id = getattr(mention, "id", None)
            mention_open_id = getattr(mention_id, "open_id", None)
            if not isinstance(mention_open_id, str) or not mention_open_id:
                continue
            mentions_payload.append({"id": {"open_id": mention_open_id}})

        return {
            "header": {
                "event_id": getattr(header, "event_id", message_id),
                "event_type": getattr(header, "event_type", "im.message.receive_v1"),
                "token": getattr(header, "token", ""),
            },
            "event": {
                "message": {
                    "message_id": message_id,
                    "chat_id": chat_id,
                    "chat_type": chat_type,
                    "thread_id": getattr(message, "thread_id", None),
                    "message_type": message_type,
                    "content": getattr(message, "content", ""),
                    "mentions": mentions_payload,
                    "create_time": getattr(message, "create_time", int(time.time() * 1000)),
                },
                "sender": {
                    "sender_id": {
                        "open_id": sender_open_id,
                    }
                },
            },
        }

    def _resolve_log_level(self) -> Any:
        if lark is None:
            return None

        level = self._sdk_log_level.lower()
        if level == "debug":
            return lark.LogLevel.DEBUG
        if level == "warn":
            warning_level = getattr(lark.LogLevel, "WARN", None)
            if warning_level is not None:
                return warning_level
            return lark.LogLevel.WARNING
        if level == "error":
            return lark.LogLevel.ERROR
        return lark.LogLevel.INFO
