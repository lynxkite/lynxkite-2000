"""CRDT is used to synchronize workspace state for backend and frontend(s)."""

import asyncio
import contextlib
import pathlib
import posixpath
import fastapi
import os.path
import pycrdt.websocket
import pycrdt.store.file
import typing
from dataclasses import dataclass, field
import uvicorn.protocols.utils
import builtins
from lynxkite_core import workspace, ops
from watchdog import events, observers
from .crdt_update import crdt_update
from . import progress_crdt

try:
    import lynxkite_enterprise.backend as enterprise_backend  # ty: ignore[unresolved-import]
except ImportError:
    enterprise_backend = None

router = fastapi.APIRouter()
main_loop = None
WORKSPACE_CHANGED_THROTTLE_SECONDS = 1.0


@dataclass
class WorkspaceRuntimeState:
    last_known_version: typing.Any = None
    delayed_execution: asyncio.Task | None = None
    pending_workspace_changes: list[int] = field(default_factory=list)
    delayed_workspace_change: asyncio.Task | None = None
    next_allowed_flush_at: float = 0.0

    def destroy(self):
        self.last_known_version = None
        if self.delayed_workspace_change:
            self.delayed_workspace_change.cancel()
        self.pending_workspace_changes = []
        self.next_allowed_flush_at = 0.0
        if self.delayed_execution:
            self.delayed_execution.cancel()


state: dict[str, WorkspaceRuntimeState] = {}


def ws_exception_handler(exception, log):
    if isinstance(exception, builtins.ExceptionGroup):
        for ex in exception.exceptions:
            if not isinstance(ex, uvicorn.protocols.utils.ClientDisconnected):
                log.exception(ex)
    else:
        log.exception(exception)
    return True


def _task_result_callback(task: asyncio.Task):
    with contextlib.suppress(asyncio.CancelledError):
        task.result()


async def _flush_workspace_changes_async(name: str, ws: pycrdt.Map):
    this_task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    try:
        now = loop.time()
        next_allowed = state[name].next_allowed_flush_at
        if now < next_allowed:
            await asyncio.sleep(next_allowed - now)

        delays = state[name].pending_workspace_changes[:]
        state[name].pending_workspace_changes.clear()
        if delays:
            delay = max(delays)
            await workspace_changed(name, delay, ws)
            state[name].next_allowed_flush_at = loop.time() + WORKSPACE_CHANGED_THROTTLE_SECONDS
    except asyncio.CancelledError:
        pass
    finally:
        if name in state:
            if state[name].delayed_workspace_change is this_task:
                state[name].delayed_workspace_change = None
            # Ensure trailing changes are not dropped while throttling.
            if (
                state[name].pending_workspace_changes
                and state[name].delayed_workspace_change is None
            ):
                task = asyncio.create_task(_flush_workspace_changes_async(name, ws))
                state[name].delayed_workspace_change = task
                task.add_done_callback(_task_result_callback)


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
                workspace.Workspace().save(name)
            else:
                load_workspace(ws, name)
        # Set the last known version to the current state, so we don't trigger a change event.
        state[name] = WorkspaceRuntimeState()
        state[name].last_known_version = _workspace_fingerprint_from_dict(ws.to_py())
        room = pycrdt.websocket.YRoom(
            ystore=ystore, ydoc=ydoc, exception_handler=ws_exception_handler
        )
        # We hang the YDoc pointer on the room, so it only gets garbage collected when the room does.
        room.ws = ws  # ty: ignore[unresolved-attribute]

        def on_change(changes):
            # Frontend changes that result from typing are delayed to avoid
            # rerunning the workspace for every keystroke.
            delay = max(
                getattr(change, "keys", {}).get("__execution_delay", {}).get("newValue", 0) or 0
                for change in changes
            )
            state[name].pending_workspace_changes.append(delay)

            if state[name].delayed_workspace_change is None:
                task = asyncio.create_task(_flush_workspace_changes_async(name, ws))
                state[name].delayed_workspace_change = task
                task.add_done_callback(_task_result_callback)

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
            print(f"Detected changes in {event.src_path}. Updating workspace...")
            self.loop.call_soon_threadsafe(load_workspace, self.ws_crdt, self.file_path)

    def on_deleted(self, event):
        if pathlib.Path(event.src_path) == pathlib.Path(self.file_path):
            print(f"Detected deletion of {event.src_path}. Deleting workspace room...")
            delete_room(self.file_path)


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


def _workspace_fingerprint_from_dict(ws_dict):
    """Produce a lightweight fingerprint for change detection.

    Operates on a plain dict (e.g. from CRDT .to_py() or model_dump),
    avoiding expensive model_validate, deep copy, and pydantic __eq__.
    """
    # Work on a shallow copy so we don't mutate the input.
    ws_dict = dict(ws_dict)
    nodes = []
    for node in ws_dict.get("nodes", []):
        node = dict(node)
        data = dict(node.get("data", {}))
        data["display"] = None
        data["display_version"] = None
        data["input_metadata"] = None
        data["output_metadata"] = None
        data["error"] = None
        data["message"] = None
        data["telemetry"] = None
        data["collapsed"] = False
        data["expanded_height"] = 0
        data["status"] = "done"
        params = dict(data.get("params", {}))
        for p in list(params):
            if p.startswith("_"):
                del params[p]
        if data.get("op_id") == "Comment":
            params = {}
        data["params"] = params
        node["data"] = data
        node["position"] = {"x": 0, "y": 0}
        node["width"] = 0
        node["height"] = 0
        node["__execution_delay"] = 0
        if node.get("model_extra"):
            for key in list(node["model_extra"]):
                del node[key]
        nodes.append(node)
    ws_dict["nodes"] = nodes
    return ws_dict


def load_workspace(ws: pycrdt.Map, name: str):
    """Load the workspace `name`, if it exists, and update the `ws` CRDT object to match its contents.

    Args:
        ws: CRDT object to udpate with the workspace contents.
        name: Name of the workspace to load.
    """
    ws_pyd = workspace.Workspace.load(name)
    update_workspace(ws, ws_pyd)


def update_workspace(ws: pycrdt.Map, ws_pyd: workspace.Workspace):
    """Load the workspace `name`, if it exists, and update the `ws` CRDT object to match its contents.

    Args:
        ws: CRDT object to udpate with the workspace contents.
        ws_pyd: Workspace object to update the CRDT with.
    """
    with ws.doc.transaction():
        crdt_update(
            ws,
            ws_pyd.model_dump(),
            # We treat some fields as black boxes. They are not edited on the frontend.
            non_collaborative_fields={
                "display",
                "input_metadata",
                "meta",
                "position",  # Edited, but we don't want to track x and y separately.
                "output_metadata",
                "telemetry",
            },
        )


def print_diff(old, new, prefix=""):
    """Print the differences between two Python/Pydantic objects. For debugging."""
    if hasattr(old, "model_dump"):
        old = old.model_dump()
    if hasattr(new, "model_dump"):
        new = new.model_dump()
    if type(old) is not type(new):
        print(f"{prefix}- {old}")
        print(f"{prefix}+ {new}")
    elif isinstance(old, dict):
        for key in set(old.keys()) | set(new.keys()):
            print_diff(old.get(key), new.get(key), prefix + f"{key}.")
    elif isinstance(old, list):
        for i, (o, n) in enumerate(zip(old, new)):
            print_diff(o, n, prefix + f"{i}.")
        if len(old) < len(new):
            for i in range(len(old), len(new)):
                print(f"{prefix}+ {new[i]}")
        elif len(old) > len(new):
            for i in range(len(new), len(old)):
                print(f"{prefix}- {old[i]}")
    else:
        if old != new:
            print(f"{prefix}- {old}")
            print(f"{prefix}+ {new}")


async def workspace_changed(name: str, delay: int, ws_crdt: pycrdt.Map):
    """Callback to react to changes in the workspace.

    Args:
        name: Name of the workspace.
        changes: Changes performed to the workspace.
        ws_crdt: CRDT object representing the workspace.
    """
    raw = ws_crdt.to_py()
    ws_fingerprint = _workspace_fingerprint_from_dict(raw)
    if enterprise_backend is not None:
        enterprise_backend.on_workspace_changed(ws_websocket_server)
    ws_pyd = workspace.Workspace.model_validate(raw)
    ws_pyd.save(pathlib.Path() / name)
    # Do not trigger execution for superficial changes.
    # This is a quick solution until we build proper caching.
    if ws_fingerprint == state[name].last_known_version:
        return
    state[name].last_known_version = ws_fingerprint

    runtime_state = state.get(name)
    if runtime_state is not None and runtime_state.delayed_execution is not None:
        runtime_state.delayed_execution.cancel()
    # Check if workspace is paused - if so, skip automatic execution
    if getattr(ws_pyd, "paused", False):
        return

    task = asyncio.create_task(execute(name, ws_crdt, ws_pyd, delay=delay))
    state[name].delayed_execution = task
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        if state[name].delayed_execution is task:
            state[name].delayed_execution = None


async def execute(name: str, ws_crdt: pycrdt.Map, ws_pyd: workspace.Workspace, *, delay: int = 0):
    """Execute the workspace and update the CRDT object with the results.

    Args:
        room: The room associated with the workspace.
        name: Name of the workspace.
        ws_crdt: CRDT object representing the workspace.
        ws_pyd: Workspace object to execute.
        delay: Wait time before executing the workspace. The default is 0.
    """
    if delay:
        await asyncio.sleep(delay)
    progress_crdt.reset_run_timer(name)
    print(f"Running {name} in {ws_pyd.env}...")
    cwd = pathlib.Path()
    path = cwd / name
    assert path.is_relative_to(cwd), f"Path '{path}' is invalid"
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
    ws_pyd.save(path)
    print(f"Finished running {name} in {ws_pyd.env}.")


async def code_changed(name: str, changes: pycrdt.TextEvent, text: pycrdt.Text):
    contents = str(text).strip() + "\n"
    with open(name, "w", encoding="utf-8") as f:
        f.write(contents)


ws_websocket_server: WorkspaceWebsocketServer
code_websocket_server: CodeWebsocketServer


async def get_room(name):
    return await ws_websocket_server.get_room(name)


def get_room_or_none(name):
    return ws_websocket_server.rooms.get(name)


@contextlib.asynccontextmanager
async def lifespan(app):
    global main_loop
    global ws_websocket_server
    global code_websocket_server
    main_loop = asyncio.get_running_loop()
    ws_websocket_server = WorkspaceWebsocketServer(auto_clean_rooms=False)
    code_websocket_server = CodeWebsocketServer(auto_clean_rooms=False)
    async with ws_websocket_server:
        async with code_websocket_server:
            async with progress_crdt.lifespan_context(ws_websocket_server):
                if enterprise_backend is not None:
                    async with enterprise_backend.lifespan_context(ws_websocket_server):
                        yield
                else:
                    yield
    print("closing websocket server")


def delete_room(name: str):
    if name in ws_websocket_server.rooms:
        room_entry = ws_websocket_server.rooms.pop(name)
        del room_entry
    if name in state:
        state_entry = state.pop(name)
        state_entry.destroy()
    progress_crdt.delete_workspace_entry(name)
    if enterprise_backend is not None:
        enterprise_backend.on_workspace_deleted(name)


def sanitize_path(path):
    # Here we always assume posix paths, the posixpath module is the os.path module
    # for posix paths even on windows, so it will work correctly regardless of the host OS.
    return posixpath.normpath("/" + path.replace("\\", "/")).lstrip("/")


app: fastapi.FastAPI | None = None


@router.websocket("/ws/crdt/{room_name:path}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    global app
    app = websocket.scope["app"]
    room_name = sanitize_path(room_name)
    progress_crdt.on_workspace_connection_open(room_name, ws_websocket_server)
    if enterprise_backend is not None:
        enterprise_backend.on_workspace_connection_open(room_name, ws_websocket_server)
    server = pycrdt.websocket.ASGIServer(ws_websocket_server)
    try:
        await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
    finally:
        progress_crdt.on_workspace_connection_close(room_name, ws_websocket_server)
        if enterprise_backend is not None:
            enterprise_backend.on_workspace_connection_close(room_name, ws_websocket_server)


progress_crdt.register_routes(router, sanitize_path)


@router.websocket("/ws/code/crdt/{room_name:path}")
async def code_crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    server = pycrdt.websocket.ASGIServer(code_websocket_server)
    await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
