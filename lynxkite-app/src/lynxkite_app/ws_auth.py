"""WebSocket authentication and read-only client filtering."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any
from urllib.parse import parse_qs

from jose.exceptions import JWTError
from pycrdt import Channel, YMessageType, YSyncMessageType
from pycrdt.websocket.asgi_server import ASGIWebsocket
from pycrdt.websocket.websocket_server import WebsocketServer

from . import acl
from .auth import get_provider, is_auth_enabled


def extract_token_from_scope(scope: dict[str, Any]) -> str | None:
    raw = scope.get("query_string", b"")
    if isinstance(raw, str):
        query = raw
    else:
        query = raw.decode()
    values = parse_qs(query).get("access_token", [])
    return values[0] if values else None


def verify_token(token: str) -> dict[str, Any]:
    return get_provider().verify(token)


def authenticate_websocket_scope(scope: dict[str, Any], room_path: str) -> bool:
    """Populate scope auth fields. Returns True if the connection must be rejected."""
    if not is_auth_enabled():
        scope["lynxkite_write"] = True
        scope["lynxkite_user"] = {"sub": "user", "email": ""}
        return False

    token = extract_token_from_scope(scope)
    if token is None:
        return True
    try:
        user = verify_token(token)
    except (JWTError, Exception):
        return True

    folder = acl.resolve_folder(room_path)
    if not acl.has_permission(user, "read", folder, auth_enabled=True):
        return True

    scope["lynxkite_user"] = user
    scope["lynxkite_write"] = acl.has_permission(user, "write", folder, auth_enabled=True)
    return False


def _is_client_document_update(message: bytes) -> bool:
    return (
        len(message) >= 2
        and message[0] == YMessageType.SYNC
        and message[1] == YSyncMessageType.SYNC_UPDATE
    )


class WriteFilteringChannel:
    """Drop client-originated Yjs document updates when write access is denied."""

    def __init__(self, inner: Channel, *, can_write: bool):
        self._inner = inner
        self.can_write = can_write

    @property
    def path(self) -> str:
        return self._inner.path

    async def send(self, message: bytes) -> None:
        await self._inner.send(message)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        while True:
            message = await self._inner.__anext__()
            if self.can_write or not _is_client_document_update(message):
                return message


class LynxKiteASGIServer:
    """ASGI WebSocket entrypoint with LynxKite auth hooks and read-only filtering."""

    def __init__(
        self,
        websocket_server: WebsocketServer,
        on_connect: Callable[[dict[str, Any], dict[str, Any]], Awaitable[bool] | bool]
        | None = None,
        on_disconnect: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
    ):
        self._websocket_server = websocket_server
        self._on_connect = on_connect
        self._on_disconnect = on_disconnect

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ):
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        elif scope["type"] == "websocket":
            message = await receive()
            if message["type"] == "websocket.connect":
                if self._on_connect is not None:
                    reject = self._on_connect(message, scope)
                    if isawaitable(reject):
                        reject = await reject
                    if reject:
                        await send({"type": "websocket.close", "code": 4403})
                        return

                await send({"type": "websocket.accept"})
                websocket = ASGIWebsocket(receive, send, scope["path"], self._on_disconnect)
                wrapped = WriteFilteringChannel(
                    websocket, can_write=scope.get("lynxkite_write", True)
                )
                await self._websocket_server.serve(wrapped)
