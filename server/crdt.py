'''CRDT is used to synchronize workspace state for backend and frontend(s).'''
import asyncio
import contextlib
import fastapi
import os.path
import pycrdt
import pycrdt_websocket

import pycrdt_websocket.ystore

router = fastapi.APIRouter()

def ws_exception_handler(exception, log):
    print('exception', exception)
    log.exception(exception)
    return True

class WebsocketServer(pycrdt_websocket.WebsocketServer):
    async def init_room(self, name):
        ystore = pycrdt_websocket.ystore.FileYStore(f'crdt_data/{name}.crdt')
        ydoc = pycrdt.Doc()
        ydoc['workspace'] = ws = pycrdt.Map()
        # Replay updates from the store.
        try:
            for update, timestamp in [(item[0], item[-1]) async for item in ystore.read()]:
                ydoc.apply_update(update)
        except pycrdt_websocket.ystore.YDocNotFound:
            pass
        if 'nodes' not in ws:
            ws['nodes'] = pycrdt.Array()
        if 'edges' not in ws:
            ws['edges'] = pycrdt.Array()
        if 'env' not in ws:
            ws['env'] = 'unset'
        room = pycrdt_websocket.YRoom(ystore=ystore, ydoc=ydoc)
        room.ws = ws
        def on_change(changes):
            asyncio.create_task(workspace_changed(changes, ws))
        ws.observe_deep(on_change)
        return room

    async def get_room(self, name: str) -> pycrdt_websocket.YRoom:
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        room = self.rooms[name]
        await self.start_room(room)
        return room

websocket_server = WebsocketServer(exception_handler=ws_exception_handler, auto_clean_rooms=False)
asgi_server = pycrdt_websocket.ASGIServer(websocket_server)

last_ws_input = None
def clean_input(ws_pyd):
    for node in ws_pyd.nodes:
        node.data.display = None
        node.position.x = 0
        node.position.y = 0
        if node.model_extra:
            for key in list(node.model_extra.keys()):
                delattr(node, key)

def crdt_update(crdt_obj, python_obj):
    if isinstance(python_obj, dict):
        for key, value in python_obj.items():
            if isinstance(value, dict):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Map()
                crdt_update(crdt_obj[key], value)
            elif isinstance(value, list):
                if crdt_obj.get(key) is None:
                    crdt_obj[key] = pycrdt.Array()
                crdt_update(crdt_obj[key], value)
            else:
                print('set', key, value)
                crdt_obj[key] = value
    elif isinstance(python_obj, list):
        for i, value in enumerate(python_obj):
            if isinstance(value, dict):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Map())
                crdt_update(crdt_obj[i], value)
            elif isinstance(value, list):
                if i >= len(crdt_obj):
                    crdt_obj.append(pycrdt.Array())
                crdt_update(crdt_obj[i], value)
            else:
                if i >= len(crdt_obj):
                    crdt_obj.append(value)
                else:
                    print('set', i, value)
                    crdt_obj[i] = value
    else:
        raise ValueError('Invalid type:', python_obj)

async def workspace_changed(e, ws_crdt):
    global last_ws_input
    from . import workspace
    ws_pyd = workspace.Workspace.model_validate(ws_crdt.to_py())
    clean_input(ws_pyd)
    if ws_pyd == last_ws_input:
        return
    print('ws changed')
    last_ws_input = ws_pyd.model_copy(deep=True)
    workspace.execute(ws_pyd)
    for nc, np in zip(ws_crdt['nodes'], ws_pyd.nodes):
        if 'data' not in nc:
            nc['data'] = pycrdt.Map()
        # Display is added as an opaque Box.
        nc['data']['display'] = np.data.display

@contextlib.asynccontextmanager
async def lifespan(app):
    async with websocket_server:
        yield

def sanitize_path(path):
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")

@router.websocket("/ws/crdt/{room_name}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    await asgi_server({'path': room_name}, websocket._receive, websocket._send)
