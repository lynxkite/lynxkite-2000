"""CRDT is used to synchronize workspace state for backend and frontend(s)."""

import asyncio
import contextlib
import enum
import pathlib
import fastapi
import os.path
import pycrdt.websocket
import pycrdt.store.file
import uvicorn.protocols.utils
import builtins
from lynxkite_core import workspace, ops

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
        ystore = pycrdt.store.file.FileYStore(path)
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
            try_to_load_workspace(ws, name)
        ws_simple = workspace.Workspace.model_validate(ws.to_py())
        clean_input(ws_simple)
        # Set the last known version to the current state, so we don't trigger a change event.
        last_known_versions[name] = ws_simple
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


class CodeWebsocketServer(WorkspaceWebsocketServer):
    async def init_room(self, name: str) -> pycrdt.websocket.YRoom:
        """Initialize a room for a text document with the given name."""
        crdt_path = pathlib.Path(".crdt")
        path = crdt_path / f"{name}.crdt"
        assert path.is_relative_to(crdt_path), f"Path '{path}' is invalid"
        ystore = pycrdt.store.file.FileYStore(path)
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


last_ws_input = None


def clean_input(ws_pyd):
    """Delete everything that we want to ignore for the purposes of change detection."""
    for node in ws_pyd.nodes:
        node.data.display = None
        node.data.input_metadata = None
        node.data.error = None
        node.data.status = workspace.NodeStatus.done
        for p in list(node.data.params):
            if p.startswith("_"):
                del node.data.params[p]
        if node.data.op_id == "Comment":
            node.data.params = {}
        node.position.x = 0
        node.position.y = 0
        node.width = 0
        node.height = 0
        if node.model_extra:
            for key in list(node.model_extra.keys()):
                delattr(node, key)


def crdt_update(
    crdt_obj: pycrdt.Map | pycrdt.Array,
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
        assert isinstance(crdt_obj, pycrdt.Map), "CRDT object must be a Map for a dict input"
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
        assert isinstance(crdt_obj, pycrdt.Array), "CRDT object must be an Array for a list input"
        for i, value in enumerate(python_obj):
            if isinstance(value, dict):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Map())
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            elif isinstance(value, list):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Array())
                crdt_update(crdt_obj[i], value, non_collaborative_fields)
            else:
                if isinstance(value, enum.Enum):
                    value = str(value.value)
                if i >= len(crdt_obj):
                    crdt_obj.append(value)
                else:
                    crdt_obj[i] = value
    else:
        raise ValueError("Invalid type:", python_obj)


def try_to_load_workspace(ws: pycrdt.Map, name: str):
    """Load the workspace `name`, if it exists, and update the `ws` CRDT object to match its contents.

    Args:
        ws: CRDT object to udpate with the workspace contents.
        name: Name of the workspace to load.
    """
    if os.path.exists(name):
        ws_pyd = workspace.Workspace.load(name)
        crdt_update(
            ws,
            ws_pyd.model_dump(),
            # We treat some fields as black boxes. They are not edited on the frontend.
            non_collaborative_fields={"display", "input_metadata", "meta"},
        )


last_known_versions = {}
delayed_executions = {}


async def workspace_changed(name: str, changes: list[pycrdt.MapEvent], ws_crdt: pycrdt.Map):
    """Callback to react to changes in the workspace.

    Args:
        name: Name of the workspace.
        changes: Changes performed to the workspace.
        ws_crdt: CRDT object representing the workspace.
    """
    ws_pyd = workspace.Workspace.model_validate(ws_crdt.to_py())
    # Do not trigger execution for superficial changes.
    # This is a quick solution until we build proper caching.
    ws_simple = ws_pyd.model_copy(deep=True)
    clean_input(ws_simple)
    if ws_simple == last_known_versions.get(name):
        return
    last_known_versions[name] = ws_simple
    # Frontend changes that result from typing are delayed to avoid
    # rerunning the workspace for every keystroke.
    if name in delayed_executions:
        delayed_executions[name].cancel()
    delay = min(
        getattr(change, "keys", {}).get("__execution_delay", {}).get("newValue", 0)
        for change in changes
    )
    # Check if workspace is paused - if so, skip automatic execution
    if getattr(ws_pyd, "paused", False):
        print(f"Skipping automatic execution for {name} in {ws_pyd.env} - workspace is paused")
        return
    if delay:
        task = asyncio.create_task(execute(name, ws_crdt, ws_pyd, delay))
        delayed_executions[name] = task
    else:
        await execute(name, ws_crdt, ws_pyd)


async def execute(name: str, ws_crdt: pycrdt.Map, ws_pyd: workspace.Workspace, delay: int = 0):
    """Execute the workspace and update the CRDT object with the results.

    Args:
        name: Name of the workspace.
        ws_crdt: CRDT object representing the workspace.
        ws_pyd: Workspace object to execute.
        delay: Wait time before executing the workspace. The default is 0.
    """
    if delay:
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            return
    print(f"Running {name} in {ws_pyd.env}...")
    cwd = pathlib.Path()
    path = cwd / name
    assert path.is_relative_to(cwd), f"Path '{path}' is invalid"
    # Save user changes before executing, in case the execution fails.
    ws_pyd.save(path)
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
    await ws_pyd.execute(workspace.WorkspaceExecutionContext(app=app))
    ws_pyd.save(path)
    print(f"Finished running {name} in {ws_pyd.env}.")


async def code_changed(name: str, changes: pycrdt.TextEvent, text: pycrdt.Text):
    contents = str(text).strip() + "\n"
    with open(name, "w", encoding="utf-8") as f:
        f.write(contents)


ws_websocket_server: WorkspaceWebsocketServer
code_websocket_server: CodeWebsocketServer


def get_room(name):
    return ws_websocket_server.get_room(name)


@contextlib.asynccontextmanager
async def lifespan(app):
    global ws_websocket_server
    global code_websocket_server
    ws_websocket_server = WorkspaceWebsocketServer(auto_clean_rooms=False)
    code_websocket_server = CodeWebsocketServer(auto_clean_rooms=False)
    async with ws_websocket_server:
        async with code_websocket_server:
            yield
    print("closing websocket server")


def delete_room(name: str):
    if name in ws_websocket_server.rooms:
        del ws_websocket_server.rooms[name]


def sanitize_path(path):
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")


app: fastapi.FastAPI | None = None


@router.websocket("/ws/crdt/{room_name:path}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    global app
    app = websocket.scope["app"]
    room_name = sanitize_path(room_name)
    server = pycrdt.websocket.ASGIServer(ws_websocket_server)
    await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)


@router.websocket("/ws/code/crdt/{room_name:path}")
async def code_crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    server = pycrdt.websocket.ASGIServer(code_websocket_server)
    await server({"path": room_name, "type": "websocket"}, websocket._receive, websocket._send)
