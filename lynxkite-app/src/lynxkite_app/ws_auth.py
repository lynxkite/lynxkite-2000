"""WebSocket authentication and read-only client filtering."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs

from jose.exceptions import JWTError
from pycrdt import Channel, YMessageType, YSyncMessageType
from pycrdt.websocket.asgi_server import ASGIWebsocket
from pycrdt.websocket.websocket_server import WebsocketServer

from . import acl
from .auth import get_provider, is_auth_enabled


def _token_from_scope(scope: dict[str, Any]) -> str | None:
    raw = scope.get("query_string", b"")
    query = raw if isinstance(raw, str) else raw.decode()
    values = parse_qs(query).get("access_token", [])
    return values[0] if values else None


def authenticate_websocket(scope: dict[str, Any], room_path: str) -> bool:
    """Set scope['lynxkite_write']. Return True if the connection must be rejected."""
    if not is_auth_enabled():
        scope["lynxkite_write"] = True
        scope["lynxkite_user"] = {"sub": "user", "email": ""}
        return False

    token = _token_from_scope(scope)
    if token is None:
        return True
    try:
        user = get_provider().verify(token)
    except JWTError:
        return True

    if not acl.has_permission(user, "read", room_path):
        return True

    scope["lynxkite_user"] = user
    scope["lynxkite_write"] = acl.has_permission(user, "write", room_path)
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

    def __aiter__(self) -> WriteFilteringChannel:
        return self

    async def __anext__(self) -> bytes:
        return await self.recv()

    async def send(self, message: bytes) -> None:
        await self._inner.send(message)

    async def recv(self) -> bytes:
        while True:
            message = await self._inner.recv()
            if self.can_write or not _is_client_document_update(message):
                return message


async def serve(
    websocket_server: WebsocketServer,
    scope: dict[str, Any],
    receive,
    send,
    room_path: str,
) -> None:
    """Accept a WebSocket after ACL checks; filter writes when read-only."""
    message = await receive()
    if message["type"] != "websocket.connect":
        return
    if authenticate_websocket(scope, room_path):
        await send({"type": "websocket.close", "code": 4403})
        return
    await send({"type": "websocket.accept"})
    channel = WriteFilteringChannel(
        ASGIWebsocket(receive, send, room_path),
        can_write=scope.get("lynxkite_write", True),
    )
    await websocket_server.serve(channel)
