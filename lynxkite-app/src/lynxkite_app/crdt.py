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
from lynxkite.core import workspace, ops

router = fastapi.APIRouter()


def ws_exception_handler(exception, log):
    if isinstance(exception, builtins.ExceptionGroup):
        for ex in exception.exceptions:
            if not isinstance(ex, uvicorn.protocols.utils.ClientDisconnected):
                log.exception(ex)
    else:
        log.exception(exception)
    return True


class WebsocketServer(pycrdt_websocket.WebsocketServer):
    async def init_room(self, name: str) -> pycrdt_websocket.YRoom:
        """Initialize a room for the workspace with the given name.

        The workspace is loaded from "crdt_data" if it exists there, or from "data", or a new workspace is created.
        """
        crdt_path = pathlib.Path(".crdt")
        path = crdt_path / f"{name}.crdt"
        assert path.is_relative_to(crdt_path)
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
            ws["env"] = next(iter(ops.CATALOGS), "unset")
            # We have two possible sources of truth for the workspaces, the YStore and the JSON files.
            # In case we didn't find the workspace in the YStore, we try to load it from the JSON files.
            try_to_load_workspace(ws, name)
        ws_simple = workspace.Workspace.model_validate(ws.to_py())
        clean_input(ws_simple)
        # Set the last known version to the current state, so we don't trigger a change event.
        last_known_versions[name] = ws_simple
        room = pycrdt_websocket.YRoom(
            ystore=ystore, ydoc=ydoc, exception_handler=ws_exception_handler
        )
        room.ws = ws

        def on_change(changes):
            asyncio.create_task(workspace_changed(name, changes, ws))

        ws.observe_deep(on_change)
        return room

    async def get_room(self, name: str) -> pycrdt_websocket.YRoom:
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


last_ws_input = None


def clean_input(ws_pyd):
    for node in ws_pyd.nodes:
        node.data.display = None
        node.data.input_metadata = None
        node.data.error = None
        node.data.status = workspace.NodeStatus.done
        node.position.x = 0
        node.position.y = 0
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
        ws_pyd = workspace.load(name)
        crdt_update(
            ws,
            ws_pyd.model_dump(),
            # We treat some fields as black boxes. They are not edited on the frontend.
            non_collaborative_fields={"display", "input_metadata"},
        )


last_known_versions = {}
delayed_executions = {}


async def workspace_changed(name: str, changes: pycrdt.MapEvent, ws_crdt: pycrdt.Map):
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
    if delay:
        task = asyncio.create_task(execute(name, ws_crdt, ws_pyd, delay))
        delayed_executions[name] = task
    else:
        await execute(name, ws_crdt, ws_pyd)


async def execute(
    name: str, ws_crdt: pycrdt.Map, ws_pyd: workspace.Workspace, delay: int = 0
):
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
    assert path.is_relative_to(cwd), "Provided workspace path is invalid"
    # Save user changes before executing, in case the execution fails.
    workspace.save(ws_pyd, path)
    ws_pyd._crdt = ws_crdt
    with ws_crdt.doc.transaction():
        for nc, np in zip(ws_crdt["nodes"], ws_pyd.nodes):
            if "data" not in nc:
                nc["data"] = pycrdt.Map()
            nc["data"]["status"] = "planned"
            # Nodes get a reference to their CRDT maps, so they can update them as the results come in.
            np._crdt = nc
    await workspace.execute(ws_pyd)
    workspace.save(ws_pyd, path)
    print(f"Finished running {name} in {ws_pyd.env}.")


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
