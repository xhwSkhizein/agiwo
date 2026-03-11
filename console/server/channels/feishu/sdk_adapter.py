"""Private Feishu SDK compatibility boundary for the long-connection client."""

import asyncio
from collections.abc import Callable

try:
    import lark_oapi as lark
except ImportError:
    lark = None

try:
    from lark_oapi.ws import client as lark_ws_client_module
except ImportError:
    lark_ws_client_module = None


def is_sdk_available() -> bool:
    return lark is not None


def build_event_handler(
    encrypt_key: str,
    verification_token: str,
    *,
    on_message: Callable[[object], None],
    on_message_read: Callable[[object], None],
) -> object:
    if lark is None:
        raise RuntimeError("lark_oapi_not_installed")
    return (
        lark.EventDispatcherHandler.builder(
            encrypt_key,
            verification_token,
        )
        .register_p2_im_message_receive_v1(on_message)
        .register_p2_im_message_message_read_v1(on_message_read)
        .build()
    )


def create_ws_client(
    *,
    app_id: str,
    app_secret: str,
    event_handler: object,
    log_level: object,
) -> object:
    if lark is None:
        raise RuntimeError("lark_oapi_not_installed")
    return lark.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=log_level,
    )


def resolve_log_level(level_name: str) -> object:
    if lark is None:
        return None
    normalized = level_name.upper()
    return getattr(lark.LogLevel, normalized, lark.LogLevel.INFO)


def install_worker_loop(loop: asyncio.AbstractEventLoop) -> None:
    if lark_ws_client_module is not None:
        lark_ws_client_module.loop = loop


def resolve_worker_loop(
    worker_loop: asyncio.AbstractEventLoop | None,
) -> asyncio.AbstractEventLoop | None:
    return worker_loop or getattr(lark_ws_client_module, "loop", None)


def disable_auto_reconnect(client: object | None) -> None:
    if client is None:
        return
    try:
        client._auto_reconnect = False
    except AttributeError:
        return
