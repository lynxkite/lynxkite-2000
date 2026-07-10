"""Shared progress CRDT for workspace-level status."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import time
from typing import Any

import fastapi
import pycrdt
import pycrdt.websocket
from lynxkite_core import workspace
from lynxkite_core.workspace_progress import compute_workspace_progress, workspace_display_name

_active_workspace_ws_connections: dict[str, int] = {}
_progress_ydoc: pycrdt.Doc | None = None
_run_started_at: dict[str, float] = {}
_progress_server: ProgressWebsocketServer | None = None


def ws_exception_handler(exception, log):
    for ex in (
        exception.exceptions if isinstance(exception, builtins.ExceptionGroup) else [exception]
    ):
        log.exception(ex)
    return True


class ProgressWebsocketServer(pycrdt.websocket.WebsocketServer):
    async def init_room(self, name: str) -> pycrdt.websocket.YRoom:
        global _progress_ydoc
        ydoc = pycrdt.Doc()
        ydoc["workspaces"] = pycrdt.Map()
        ydoc["gpu_services"] = pycrdt.Text()
        _progress_ydoc = ydoc
        return pycrdt.websocket.YRoom(ydoc=ydoc, exception_handler=ws_exception_handler)

    async def get_room(self, name: str) -> pycrdt.websocket.YRoom:
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        room = self.rooms[name]
        await self.start_room(room)
        return room


def get_progress_server() -> ProgressWebsocketServer | None:
    return _progress_server


def reset_run_timer(room_name: str) -> None:
    _run_started_at[room_name] = time.monotonic()


def _elapsed_seconds(room_name: str) -> float | None:
    started = _run_started_at.get(room_name)
    if started is None:
        return None
    return max(0.0, time.monotonic() - started)


def _connected_workspace_rooms(server) -> list[tuple[str, Any]]:
    return [
        (room_name, room)
        for room_name in sorted(_active_workspace_ws_connections)
        if (room := server.rooms.get(room_name)) is not None
    ]


def _update_workspace_ws_connection(room_name: str, delta: int) -> None:
    count = _active_workspace_ws_connections.get(room_name, 0) + delta
    if count <= 0:
        _active_workspace_ws_connections.pop(room_name, None)
    else:
        _active_workspace_ws_connections[room_name] = count


def on_workspace_connection_open(room_name: str, ws_server) -> None:
    _update_workspace_ws_connection(room_name, +1)
    update_progress_workspaces(ws_server)


def on_workspace_connection_close(room_name: str, ws_server) -> None:
    _update_workspace_ws_connection(room_name, -1)
    update_progress_workspaces(ws_server)


def update_progress_workspaces(ws_server, k8s_workspace_gpus: dict | None = None) -> None:
    if _progress_ydoc is None or not hasattr(ws_server, "rooms"):
        return
    k8s_workspace_gpus = k8s_workspace_gpus or {}
    ws_map: pycrdt.Map = _progress_ydoc["workspaces"]
    entries_by_room: dict[str, str] = {}
    for room_name, room in _connected_workspace_rooms(ws_server):
        try:
            ws = workspace.Workspace.model_validate(room.ws.to_py())
            gpus = k8s_workspace_gpus.get(workspace_display_name(room_name))
            payload = compute_workspace_progress(
                ws,
                room_name=room_name,
                elapsed_seconds=_elapsed_seconds(room_name),
                gpus=gpus,
            )
            entries_by_room[room_name] = json.dumps(payload)
        except Exception as e:
            print(f"Error updating progress for workspace {room_name}: {e}")
    with _progress_ydoc.transaction():
        for room_name, entry in entries_by_room.items():
            ws_map[room_name] = entry
        for name in list(ws_map.keys()):
            if name not in entries_by_room:
                del ws_map[name]


def update_progress_gpu_services(gpu_services_data: list) -> None:
    if _progress_ydoc is None:
        return
    gpu_services_text: pycrdt.Text = _progress_ydoc["gpu_services"]
    new_content = json.dumps(gpu_services_data)
    if str(gpu_services_text) == new_content:
        return
    with _progress_ydoc.transaction():
        if len(gpu_services_text) > 0:
            del gpu_services_text[0 : len(gpu_services_text)]
        gpu_services_text += new_content


def delete_workspace_entry(name: str) -> None:
    _active_workspace_ws_connections.pop(name, None)
    _run_started_at.pop(name, None)
    if _progress_ydoc is None:
        return
    ws_map: pycrdt.Map = _progress_ydoc["workspaces"]
    if name in ws_map:
        with _progress_ydoc.transaction():
            del ws_map[name]


async def _progress_refresh_loop(ws_server, interval_seconds: float = 1.0) -> None:
    while True:
        update_progress_workspaces(ws_server)
        await asyncio.sleep(interval_seconds)


@contextlib.asynccontextmanager
async def lifespan_context(ws_server):
    global _progress_server
    server = ProgressWebsocketServer(auto_clean_rooms=False)
    _progress_server = server
    async with server:
        await server.get_room("progress")
        update_progress_workspaces(ws_server)
        refresh_task = asyncio.create_task(_progress_refresh_loop(ws_server))
        try:
            yield
        finally:
            refresh_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresh_task
    _progress_server = None


def register_routes(router, sanitize_path) -> None:
    @router.websocket("/ws/progress/crdt/{room_name:path}")
    async def progress_crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
        room_name = sanitize_path(room_name)
        progress_server = get_progress_server()
        if progress_server is None:
            await websocket.close(code=1008)
            return
        server = pycrdt.websocket.ASGIServer(progress_server)
        await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
