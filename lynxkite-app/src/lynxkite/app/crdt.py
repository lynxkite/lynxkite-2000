"""CRDT is used to synchronize workspace state for backend and frontend(s)."""

import asyncio
import contextlib
import enum
import pathlib
import fastapi
import os.path
import pycrdt
import pycrdt_websocket
import pycrdt_websocket.ystore
import uvicorn
import builtins
from lynxkite.core import workspace

router = fastapi.APIRouter()
DATA_PATH = pathlib.Path.cwd() / "data"
CRDT_PATH = pathlib.Path.cwd() / "crdt_data"


def ws_exception_handler(exception, log):
    if isinstance(exception, builtins.ExceptionGroup):
        for ex in exception.exceptions:
            if not isinstance(ex, uvicorn.protocols.utils.ClientDisconnected):
                log.exception(ex)
    else:
        log.exception(exception)
    return True


class WebsocketServer(pycrdt_websocket.WebsocketServer):
    async def init_room(self, name):
        path = CRDT_PATH / f"{name}.crdt"
        assert path.is_relative_to(CRDT_PATH)
        ystore = pycrdt_websocket.ystore.FileYStore(path)
        ydoc = pycrdt.Doc()
        ydoc["workspace"] = ws = pycrdt.Map()
        # Replay updates from the store.
        try:
            for update, timestamp in [
                (item[0], item[-1]) async for item in ystore.read()
            ]:
                ydoc.apply_update(update)
        except pycrdt_websocket.ystore.YDocNotFound:
            pass
        if "nodes" not in ws:
            ws["nodes"] = pycrdt.Array()
        if "edges" not in ws:
            ws["edges"] = pycrdt.Array()
        if "env" not in ws:
            ws["env"] = "unset"
            try_to_load_workspace(ws, name)
        room = pycrdt_websocket.YRoom(
            ystore=ystore, ydoc=ydoc, exception_handler=ws_exception_handler
        )
        room.ws = ws

        def on_change(changes):
            asyncio.create_task(workspace_changed(name, changes, ws))

        ws.observe_deep(on_change)
        return room

    async def get_room(self, name: str) -> pycrdt_websocket.YRoom:
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        room = self.rooms[name]
        await self.start_room(room)
        return room


last_ws_input = None


def clean_input(ws_pyd):
    for node in ws_pyd.nodes:
        node.data.display = None
        node.data.error = None
        node.position.x = 0
        node.position.y = 0
        if node.model_extra:
            for key in list(node.model_extra.keys()):
                delattr(node, key)


def crdt_update(crdt_obj, python_obj, boxes=set()):
    if isinstance(python_obj, dict):
        for key, value in python_obj.items():
            if key in boxes:
                crdt_obj[key] = value
            elif isinstance(value, dict):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Map()
                crdt_update(crdt_obj[key], value, boxes)
            elif isinstance(value, list):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Array()
                crdt_update(crdt_obj[key], value, boxes)
            elif isinstance(value, enum.Enum):
                crdt_obj[key] = str(value)
            else:
                crdt_obj[key] = value
    elif isinstance(python_obj, list):
        for i, value in enumerate(python_obj):
            if isinstance(value, dict):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Map())
                crdt_update(crdt_obj[i], value, boxes)
            elif isinstance(value, list):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Array())
                crdt_update(crdt_obj[i], value, boxes)
            else:
                if i >= len(crdt_obj):
                    crdt_obj.append(value)
                else:
                    crdt_obj[i] = value
    else:
        raise ValueError("Invalid type:", python_obj)


def try_to_load_workspace(ws, name):
    json_path = f"data/{name}"
    if os.path.exists(json_path):
        ws_pyd = workspace.load(json_path)
        crdt_update(ws, ws_pyd.model_dump(), boxes={"display"})


last_known_versions = {}
delayed_executions = {}


async def workspace_changed(name, changes, ws_crdt):
    ws_pyd = workspace.Workspace.model_validate(ws_crdt.to_py())
    # Do not trigger execution for superficial changes.
    # This is a quick solution until we build proper caching.
    clean_input(ws_pyd)
    if ws_pyd == last_known_versions.get(name):
        return
    last_known_versions[name] = ws_pyd.model_copy(deep=True)
    # Frontend changes that result from typing are delayed to avoid
    # rerunning the workspace for every keystroke.
    if name in delayed_executions:
        delayed_executions[name].cancel()
    delay = min(
        getattr(change, "keys", {}).get("__execution_delay", {}).get("newValue", 0)
        for change in changes
    )
    if delay:
        task = asyncio.create_task(execute(name, ws_crdt, ws_pyd, delay))
        delayed_executions[name] = task
    else:
        await execute(name, ws_crdt, ws_pyd)


async def execute(name, ws_crdt, ws_pyd, delay=0):
    if delay:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
    path = DATA_PATH / name
    assert path.is_relative_to(DATA_PATH)
    workspace.save(ws_pyd, path)
    await workspace.execute(ws_pyd)
    workspace.save(ws_pyd, path)
    with ws_crdt.doc.transaction():
        for nc, np in zip(ws_crdt["nodes"], ws_pyd.nodes):
            if "data" not in nc:
                nc["data"] = pycrdt.Map()
            # Display is added as an opaque Box.
            nc["data"]["display"] = np.data.display
            nc["data"]["error"] = np.data.error


@contextlib.asynccontextmanager
async def lifespan(app):
    global websocket_server
    websocket_server = WebsocketServer(
        auto_clean_rooms=False,
    )
    async with websocket_server:
        yield
    print("closing websocket server")


def sanitize_path(path):
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")


@router.websocket("/ws/crdt/{room_name}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    server = pycrdt_websocket.ASGIServer(websocket_server)
    await server({"path": room_name}, websocket._receive, websocket._send)
