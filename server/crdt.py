'''CRDT is used to synchronize workspace state for backend and frontend(s).'''
import contextlib
import fastapi
import os.path
import pycrdt
import pycrdt_websocket

import pycrdt_websocket.ystore

router = fastapi.APIRouter()

def ws_exception_handler(exception, log):
    log.exception(exception)
    return True

class WebsocketServer(pycrdt_websocket.WebsocketServer):
    async def init_room(self, name):
        ystore = pycrdt_websocket.ystore.FileYStore(f'crdt_data/{name}.crdt')
        ydoc = pycrdt.Doc()
        ydoc['workspace'] = ws = pycrdt.Map()
        ws['nodes'] = pycrdt.Array()
        ws['edges'] = pycrdt.Array()
        ws['env'] = 'unset'
        # Replay updates from the store.
        try:
            for update, timestamp in [(item[0], item[-1]) async for item in ystore.read()]:
                ydoc.apply_update(update)
        except pycrdt_websocket.ystore.YDocNotFound:
            pass
        print('init_room', name, ws)
        _subscription_id = ws.observe_deep(handle_deep_changes)
        return pycrdt_websocket.YRoom(ystore=ystore, ydoc=ydoc)

    async def get_room(self, name: str) -> pycrdt_websocket.YRoom:
        print('get_room', name, self.rooms)
        if name not in self.rooms:
            self.rooms[name] = await self.init_room(name)
        print('get_room2', name, self.rooms)
        room = self.rooms[name]
        await self.start_room(room)
        return room

print('new WebsocketServer')
websocket_server = WebsocketServer(exception_handler=ws_exception_handler, auto_clean_rooms=False)
asgi_server = pycrdt_websocket.ASGIServer(websocket_server)

def handle_deep_changes(events):
    print('events', events)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with websocket_server:
        yield

def sanitize_path(path):
    return os.path.relpath(os.path.normpath(os.path.join("/", path)), "/")

@router.websocket("/ws/crdt/{room_name}")
async def crdt_websocket(websocket: fastapi.WebSocket, room_name: str):
    room_name = sanitize_path(room_name)
    print('room_name', room_name)
    await asgi_server({'path': room_name}, websocket._receive, websocket._send)
