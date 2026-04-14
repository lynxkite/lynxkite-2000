"""CRDT is used to synchronize workspace state for backend and frontend(s)."""

import asyncio
import contextlib
import enum
import json
import os
import pathlib
import re
from typing import Any
import fastapi
import os.path
import pycrdt.websocket
import pycrdt.store.file
import uvicorn.protocols.utils
import builtins
from lynxkite_core import workspace, ops
from watchdog import events, observers

router = fastapi.APIRouter()


def ws_exception_handler(exception, log):
    if isinstance(exception, builtins.ExceptionGroup):
        for ex in exception.exceptions:
            if not isinstance(ex, uvicorn.protocols.utils.ClientDisconnected):
                log.exception(ex)
    else:
        log.exception(exception)
    return True


class WorkspaceWebsocketServer(pycrdt.websocket.WebsocketServer):
    async def init_room(self, name: str) -> pycrdt.websocket.YRoom:
        """Initialize a room for the workspace with the given name.

        The workspace is loaded from ".crdt" if it exists there, or from a JSON file, or a new workspace is created.
        """
        crdt_path = pathlib.Path(".crdt")
        path = crdt_path / f"{name}.crdt"
        assert path.is_relative_to(crdt_path), f"Path '{path}' is invalid"
        ystore = pycrdt.store.file.FileYStore(str(path))
        ydoc = pycrdt.Doc()
        ydoc["workspace"] = ws = pycrdt.Map()
        # Replay updates from the store.
        try:
            for update, timestamp in [(item[0], item[-1]) async for item in ystore.read()]:
                ydoc.apply_update(update)
        except pycrdt.store.YDocNotFound:
            pass
        if "nodes" not in ws:
            ws["nodes"] = pycrdt.Array()
        if "edges" not in ws:
            ws["edges"] = pycrdt.Array()
        if "env" not in ws:
            ws["env"] = next(iter(ops.CATALOGS), "unset")
            # We have two possible sources of truth for the workspaces, the YStore and the JSON files.
            # In case we didn't find the workspace in the YStore, we try to load it from the JSON files.
            if not os.path.exists(name):
                _save_workspace(name, workspace.Workspace())
            else:
                load_workspace(ws, name)
        ws_simple = workspace.Workspace.model_validate(ws.to_py())
        ws_exec = ws_simple.model_copy(deep=True)
        clean_execution_input(ws_exec)
        last_execution_versions[name] = ws_exec
        ws_clean = ws_simple.model_copy(deep=True)
        clean_persisted_input(ws_clean)
        last_saved_file_contents[name] = ws_clean.model_dump_json_sorted()
        room = pycrdt.websocket.YRoom(
            ystore=ystore, ydoc=ydoc, exception_handler=ws_exception_handler
        )
        # We hang the YDoc pointer on the room, so it only gets garbage collected when the room does.
        room.ws = ws  # ty: ignore[unresolved-attribute]

        def on_change(changes):
            task = asyncio.create_task(workspace_changed(name, changes, ws))
            # We have no way to await workspace_changed(). The best we can do is to
            # dereference its result after it's done, so exceptions are logged normally.
            task.add_done_callback(lambda t: t.result())

        ws.observe_deep(on_change)

        # Observe the file too while the room exists.
        loop = asyncio.get_running_loop()
        file_change_handler = WorkspaceFileChangeHandler(ws, name, loop)
        file_change_handler.start()
        room.file_change_handler = file_change_handler  # ty: ignore[unresolved-attribute]
        return room

    async def get_room(self, name: str) -> pycrdt.websocket.YRoom:
        """Get a room by name.

        This method overrides the parent get_room method. The original creates an empty room,
        with no associated Ydoc. Instead, we want to initialize the the room with a Workspace
        object.
        """
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        room = self.rooms[name]
        await self.start_room(room)
        return room


class WorkspaceFileChangeHandler(events.FileSystemEventHandler):
    def __init__(self, ws_crdt: pycrdt.Map, file_path: str, loop: asyncio.AbstractEventLoop):
        self.file_path = file_path
        self.dir_path = os.path.dirname(file_path) or "."
        self.ws_crdt = ws_crdt
        self.loop = loop
        self.started = False

    def start(self):
        self.observer = observers.Observer()
        self.observer.schedule(self, path=self.dir_path, recursive=False)
        self.observer.start()
        self.started = True

    def stop(self):
        if self.started:
            self.observer.stop()
            self.observer.join()
            self.started = False

    def __del__(self):
        self.stop()

    def on_modified(self, event):
        if pathlib.Path(event.src_path) == pathlib.Path(self.file_path):
            try:
                current_content = pathlib.Path(event.src_path).read_text(encoding="utf-8")
            except FileNotFoundError:
                return
            if current_content == last_saved_file_contents.get(self.file_path):
                return
            last_saved_file_contents[self.file_path] = current_content
            print(f"Detected external changes in {event.src_path}. Updating workspace...")
            self.loop.call_soon_threadsafe(load_workspace, self.ws_crdt, self.file_path)


class CodeWebsocketServer(WorkspaceWebsocketServer):
    async def init_room(self, name: str) -> pycrdt.websocket.YRoom:
        """Initialize a room for a text document with the given name."""
        crdt_path = pathlib.Path(".crdt")
        path = crdt_path / f"{name}.crdt"
        assert path.is_relative_to(crdt_path), f"Path '{path}' is invalid"
        ystore = pycrdt.store.file.FileYStore(str(path))
        ydoc = pycrdt.Doc()
        ydoc["text"] = text = pycrdt.Text()
        # Replay updates from the store.
        try:
            for update, timestamp in [(item[0], item[-1]) async for item in ystore.read()]:
                ydoc.apply_update(update)
        except pycrdt.store.YDocNotFound:
            pass
        if len(text) == 0:
            if os.path.exists(name):
                with open(name, encoding="utf-8") as f:
                    text += f.read().replace("\r\n", "\n")
        room = pycrdt.websocket.YRoom(
            ystore=ystore, ydoc=ydoc, exception_handler=ws_exception_handler
        )
        # We hang the YDoc pointer on the room, so it only gets garbage collected when the room does.
        room.text = text  # ty: ignore[unresolved-attribute]

        def on_change(changes):
            asyncio.create_task(code_changed(name, changes, text))

        text.observe(on_change)
        return room


def clean_persisted_input(ws_pyd):
    """Delete only truly transient fields before persisting. Keep execution results (display/error/message/status/telemetry)."""
    for node in ws_pyd.nodes:
        node.data.input_metadata = None
        node.__execution_delay = 0
        if node.model_extra:
            for key in list(node.model_extra.keys()):
                delattr(node, key)


def clean_execution_input(ws_pyd):
    """Delete everything that should not trigger workspace execution."""
    clean_persisted_input(ws_pyd)
    for node in ws_pyd.nodes:
        node.data.display = None
        node.data.error = None
        node.data.message = None
        node.data.status = workspace.NodeStatus.done
        node.data.telemetry = None
        node.data.collapsed = False
        node.data.expanded_height = 0
        for p in list(node.data.params):
            if p.startswith("_"):
                del node.data.params[p]
        if node.data.op_id == "Comment":
            node.data.params = {}
        node.position.x = 0
        node.position.y = 0
        node.width = 0
        node.height = 0


def _save_workspace(name: str, ws_pyd: workspace.Workspace) -> None:
    """Save the workspace, skipping execution-only state. Stamps last_saved_file_contents
    so the file watcher won't reload a file we just wrote ourselves."""
    cwd = pathlib.Path()
    path = cwd / name
    assert path.is_relative_to(cwd), f"Path '{path}' is invalid"
    # Use model_dump/validate round-trip instead of deep copy to avoid pickling
    # private CRDT state that gets attached after connect_crdt() is called.
    ws_clean = workspace.Workspace.model_validate(ws_pyd.model_dump())
    clean_persisted_input(ws_clean)
    content = ws_clean.model_dump_json_sorted()
    if content == last_saved_file_contents.get(name):
        return
    last_saved_file_contents[name] = content
    ws_clean.save(path)


def crdt_update(
    crdt_obj: pycrdt.Map[Any] | pycrdt.Array[Any],
    python_obj: dict | list,
    non_collaborative_fields: set[str] = set(),
):
    """Update a CRDT object to match a Python object.

    The types between the CRDT object and the Python object must match. If the Python object
    is a dict, the CRDT object must be a Map. If the Python object is a list, the CRDT object
    must be an Array.

    Args:
        crdt_obj: The CRDT object, that will be updated to match the Python object.
        python_obj: The Python object to update with.
        non_collaborative_fields: List of fields to treat as a black box. Black boxes are
        updated as a whole, instead of having a fine-grained data structure to edit
        collaboratively. Useful for complex fields that contain auto-generated data or
        metadata.
        The default is an empty set.

    Raises:
        ValueError: If the Python object provided is not a dict or list.
    """
    if isinstance(python_obj, dict):
        assert isinstance(crdt_obj, pycrdt.Map), f"expected CRDT Map, got {type(crdt_obj)}"
        if crdt_obj.to_py() == python_obj:
            return
        for key, value in python_obj.items():
            if key in non_collaborative_fields:
                crdt_obj[key] = value
            elif isinstance(value, dict):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Map()
                crdt_update(crdt_obj[key], value, non_collaborative_fields)
            elif isinstance(value, list):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Array()
                crdt_update(crdt_obj[key], value, non_collaborative_fields)
            elif isinstance(value, enum.Enum):
                crdt_obj[key] = str(value.value)
            else:
                crdt_obj[key] = value
    elif isinstance(python_obj, list):
        assert isinstance(crdt_obj, pycrdt.Array), f"expected CRDT Array, got {type(crdt_obj)}"
        if crdt_obj.to_py() == python_obj:
            return
        for i, value in enumerate(python_obj):
            if isinstance(value, dict):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Map())  # ty: ignore[invalid-argument-type]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            elif isinstance(value, list):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Array())  # ty: ignore[invalid-argument-type]
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            else:
                if isinstance(value, enum.Enum):
                    value = str(value.value)
                if i >= len(crdt_obj):
                    crdt_obj.append(value)  # ty: ignore[invalid-argument-type]
                else:
                    crdt_obj[i] = value  # ty: ignore[invalid-assignment]
    else:
        raise ValueError("Invalid type:", python_obj)


def load_workspace(ws: pycrdt.Map, name: str):
    """Load the workspace `name`, if it exists, and update the `ws` CRDT object to match its contents.

    Args:
        ws: CRDT object to udpate with the workspace contents.
        name: Name of the workspace to load.
    """
    ws_pyd = workspace.Workspace.load(name)
    ws_clean = ws_pyd.model_copy(deep=True)
    clean_persisted_input(ws_clean)
    last_saved_file_contents[name] = ws_clean.model_dump_json_sorted()
    crdt_update(
        ws,
        ws_pyd.model_dump(),
        # We treat some fields as black boxes. They are not edited on the frontend.
        non_collaborative_fields={"display", "input_metadata", "meta"},
    )


last_execution_versions: dict[str, workspace.Workspace] = {}
last_saved_file_contents: dict[str, str] = {}
delayed_executions: dict[str, asyncio.Task] = {}


async def workspace_changed(name: str, changes: list[pycrdt.MapEvent], ws_crdt: pycrdt.Map):
    """Callback to react to changes in the workspace.

    Args:
        name: Name of the workspace.
        changes: Changes performed to the workspace.
        ws_crdt: CRDT object representing the workspace.
    """
    ws_pyd = workspace.Workspace.model_validate(ws_crdt.to_py())
    # Push the latest workspace state to the progress CRDT doc immediately.
    update_progress_workspaces()
    # Persist every change except pure execution-state noise (status, telemetry, display).
    # The watcher ignores files we wrote ourselves via last_saved_file_contents.
    _save_workspace(name, ws_pyd)
    # Do not trigger execution for superficial changes (layout, comments, etc.).
    ws_simple = ws_pyd.model_copy(deep=True)
    clean_execution_input(ws_simple)
    if ws_simple == last_execution_versions.get(name):
        return
    last_execution_versions[name] = ws_simple
    # Frontend changes that result from typing are delayed to avoid
    # rerunning the workspace for every keystroke.
    if name in delayed_executions:
        delayed_executions[name].cancel()
    # Frontend changes that result from typing are delayed to avoid
    # rerunning the workspace for every keystroke.
    delay = max(
        getattr(change, "keys", {}).get("__execution_delay", {}).get("newValue", 0) or 0
        for change in changes
    )
    # Check if workspace is paused - if so, skip automatic execution
    if getattr(ws_pyd, "paused", False):
        return

    task = asyncio.create_task(execute(name, ws_crdt, ws_pyd, delay=delay))
    delayed_executions[name] = task
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if delayed_executions.get(name) is task:
            del delayed_executions[name]


async def execute(name: str, ws_crdt: pycrdt.Map, ws_pyd: workspace.Workspace, *, delay: int = 0):
    """Execute the workspace and update the CRDT object with the results.

    Args:
        name: Name of the workspace.
        ws_crdt: CRDT object representing the workspace.
        ws_pyd: Workspace object to execute.
        delay: Wait time before executing the workspace. The default is 0.
    """
    if delay:
        await asyncio.sleep(delay)
    print(f"Running {name} in {ws_pyd.env}...")
    ops.load_user_scripts(name)
    ws_pyd.connect_crdt(ws_crdt)
    ws_pyd.update_metadata()
    ws_pyd.path = name
    ws_pyd.normalize()
    if not ws_pyd.has_executor():
        return
    with ws_crdt.doc.transaction():
        for nc in ws_crdt["nodes"]:
            nc["data"]["status"] = "planned"
            nc["data"]["message"] = None
    await ws_pyd.execute(workspace.WorkspaceExecutionContext(app=app))
    _save_workspace(name, ws_pyd)
    print(f"Finished running {name} in {ws_pyd.env}.")


async def code_changed(name: str, changes: pycrdt.TextEvent, text: pycrdt.Text):
    contents = str(text).strip() + "\n"
    with open(name, "w", encoding="utf-8") as f:
        f.write(contents)


ws_websocket_server: WorkspaceWebsocketServer
code_websocket_server: CodeWebsocketServer

# Tracks currently open workspace websocket connections by room name.
_active_workspace_ws_connections: dict[str, int] = {}

# Progress CRDT — singleton ephemeral room for the progress page.
_progress_ydoc: pycrdt.Doc | None = None


class ProgressWebsocketServer(pycrdt.websocket.WebsocketServer):
    """WebSocket server for the singleton progress CRDT room."""

    async def init_room(self, name: str) -> pycrdt.websocket.YRoom:
        global _progress_ydoc
        ydoc = pycrdt.Doc()
        ydoc["workspaces"] = pycrdt.Map()
        ydoc["gpu_services"] = pycrdt.Text()
        _progress_ydoc = ydoc
        return pycrdt.websocket.YRoom(ydoc=ydoc, exception_handler=ws_exception_handler)

    async def get_room(self, name: str) -> pycrdt.websocket.YRoom:
        """Get or initialize a progress room with the expected schema."""
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        room = self.rooms[name]
        await self.start_room(room)
        return room


progress_websocket_server: ProgressWebsocketServer


def _workspace_display_name(room_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "-", pathlib.Path(room_name).name.removesuffix(".lynxkite.json"))


def _extract_active_node(nodes: list) -> tuple[dict[str, Any] | None, float | None]:
    for node in nodes:
        if node.data.status != "active":
            continue
        telemetry = node.data.telemetry or {}
        tqdm_n = telemetry.get("n")
        tqdm_total = telemetry.get("total")
        tqdm_rate = telemetry.get("rate")
        active_node = {
            "id": node.id,
            "title": node.data.title,
            "tqdm": {"n": tqdm_n, "total": tqdm_total, "rate": tqdm_rate} if tqdm_total else None,
        }
        eta_seconds = None
        if tqdm_total and tqdm_rate and tqdm_rate > 0 and tqdm_n is not None:
            eta_seconds = (tqdm_total - tqdm_n) / tqdm_rate
        return active_node, eta_seconds
    return None, None


def _workspace_status(*, total: int, done: int, active: int, paused: bool) -> str:
    if total == 0:
        return "idle"
    if done == total:
        return "done"
    if paused:
        return "paused"
    if active > 0:
        return "active"
    return "running"


def _build_workspace_entry(room_name: str, room, k8s_workspace_gpus: dict[str, int]) -> str:
    display_name = _workspace_display_name(room_name)
    ws = workspace.Workspace.model_validate(room.ws.to_py())
    nodes = ws.nodes or []
    total = len(nodes)
    done = sum(1 for n in nodes if n.data.status == "done")
    active = sum(1 for n in nodes if n.data.status == "active")
    paused = bool(ws.paused)
    active_node, eta_seconds = _extract_active_node(nodes)
    status = _workspace_status(total=total, done=done, active=active, paused=paused)

    return json.dumps(
        {
            "name": display_name,
            "room_name": room_name,
            "status": status,
            "boxes_done": done,
            "boxes_total": total,
            "active_node": active_node,
            "eta_seconds": eta_seconds,
            "gpus": (ws.execution_options or {}).get("gpus")
            or k8s_workspace_gpus.get(display_name, 0),
            "paused": paused,
        }
    )


def _connected_workspace_rooms(server) -> list[tuple[str, Any]]:
    rooms = [
        (room_name, room)
        for room_name in _active_workspace_ws_connections
        if (room := server.rooms.get(room_name)) is not None
    ]
    rooms.sort(key=lambda item: item[0])
    return rooms


def _update_workspace_ws_connection(room_name: str, delta: int) -> None:
    count = _active_workspace_ws_connections.get(room_name, 0) + delta
    if count <= 0:
        _active_workspace_ws_connections.pop(room_name, None)
    else:
        _active_workspace_ws_connections[room_name] = count


def update_progress_workspaces(k8s_workspace_gpus: dict | None = None):
    """Recompute workspace status entries and push them into the progress CRDT doc.

    Called directly from workspace_changed (no K8s fallback) and periodically
    from the background refresh loop (with K8s GPU data).
    """
    if _progress_ydoc is None or not hasattr(ws_websocket_server, "rooms"):
        return
    if k8s_workspace_gpus is None:
        k8s_workspace_gpus = {}
    ws_map: pycrdt.Map = _progress_ydoc["workspaces"]
    connected_rooms = _connected_workspace_rooms(ws_websocket_server)
    entries_by_room: dict[str, str] = {}
    for room_name, room in connected_rooms:
        try:
            entries_by_room[room_name] = _build_workspace_entry(room_name, room, k8s_workspace_gpus)
        except Exception as e:
            print(f"Error updating progress for workspace {room_name}: {e}")

    # Sync the map in one transaction to avoid transient partial snapshots.
    with _progress_ydoc.transaction():
        for room_name, entry in entries_by_room.items():
            ws_map[room_name] = entry
        for name in list(ws_map.keys()):
            if name not in entries_by_room:
                del ws_map[name]


def update_progress_gpu_services(gpu_services_data: list):
    """Update the GPU services entry in the progress CRDT doc with fresh Kubernetes data."""
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


def get_room(name):
    return ws_websocket_server.get_room(name)


@contextlib.asynccontextmanager
async def lifespan(app):
    global ws_websocket_server
    global code_websocket_server
    global progress_websocket_server
    ws_websocket_server = WorkspaceWebsocketServer(auto_clean_rooms=False)
    code_websocket_server = CodeWebsocketServer(auto_clean_rooms=False)
    progress_websocket_server = ProgressWebsocketServer(auto_clean_rooms=False)
    async with ws_websocket_server:
        async with code_websocket_server:
            async with progress_websocket_server:
                # Pre-initialise the singleton room so clients get an immediate snapshot.
                await progress_websocket_server.get_room("progress")
                update_progress_workspaces()
                update_progress_gpu_services([])
                yield
    print("closing websocket server")


def delete_room(name: str):
    if name in ws_websocket_server.rooms:
        del ws_websocket_server.rooms[name]
    _active_workspace_ws_connections.pop(name, None)
    if name in delayed_executions:
        delayed_executions[name].cancel()
        del delayed_executions[name]
    last_execution_versions.pop(name, None)
    last_saved_file_contents.pop(name, None)
    # Remove the workspace entry from the progress doc.
    if _progress_ydoc is not None:
        ws_map: pycrdt.Map = _progress_ydoc["workspaces"]
        if name in ws_map:
            with _progress_ydoc.transaction():
                del ws_map[name]


def sanitize_path(path):
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")


app: fastapi.FastAPI | None = None


@router.websocket("/ws/crdt/{room_name:path}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    global app
    app = websocket.scope["app"]
    room_name = sanitize_path(room_name)
    server = pycrdt.websocket.ASGIServer(ws_websocket_server)
    _update_workspace_ws_connection(room_name, +1)
    update_progress_workspaces()
    try:
        await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
    finally:
        _update_workspace_ws_connection(room_name, -1)
        update_progress_workspaces()


@router.websocket("/ws/code/crdt/{room_name:path}")
async def code_crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    server = pycrdt.websocket.ASGIServer(code_websocket_server)
    await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)


@router.websocket("/ws/progress/crdt/{room_name:path}")
async def progress_room_crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    # Ensure the room (and _progress_ydoc) exists before handing over to ASGIServer.
    await progress_websocket_server.get_room(room_name)
    server = pycrdt.websocket.ASGIServer(progress_websocket_server)
    await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
