import json
from types import SimpleNamespace

import pycrdt
import pytest
from lynxkite_app import progress_crdt


class FakeRoom:
    def __init__(self):
        doc = pycrdt.Doc()
        doc["workspace"] = self.ws = pycrdt.Map({"nodes": pycrdt.Array(), "edges": pycrdt.Array()})


class FakeServer:
    def __init__(self, *room_names):
        self.rooms = {name: FakeRoom() for name in room_names}


@pytest.fixture
def progress_doc(monkeypatch):
    doc = pycrdt.Doc()
    doc["workspaces"] = pycrdt.Map()
    monkeypatch.setattr(progress_crdt, "_progress_ydoc", doc)
    monkeypatch.setattr(progress_crdt, "_active_workspace_ws_connections", {})
    return doc["workspaces"]


def _entry(ws_map, room_name):
    return json.loads(ws_map[room_name])


def test_progress_shows_connected_users(progress_doc):
    room = "a.lynxkite.json"
    server = FakeServer(room)
    progress_crdt.on_workspace_connection_open(room, server, "Bob")
    progress_crdt.on_workspace_connection_open(room, server, "Alice")
    progress_crdt.on_workspace_connection_open(room, server, "Bob")
    assert _entry(progress_doc, room)["user"] == "Alice, Bob"
    progress_crdt.on_workspace_connection_close(room, server, "Bob")
    assert _entry(progress_doc, room)["user"] == "Alice, Bob"
    progress_crdt.on_workspace_connection_close(room, server, "Alice")
    assert _entry(progress_doc, room)["user"] == "Bob"
    progress_crdt.on_workspace_connection_close(room, server, "Bob")
    assert room not in progress_doc


def test_progress_no_user_when_auth_off(progress_doc):
    room = "a.lynxkite.json"
    server = FakeServer(room)
    progress_crdt.on_workspace_connection_open(room, server, None)
    assert _entry(progress_doc, room)["user"] is None
    progress_crdt.on_workspace_connection_close(room, server, None)
    assert room not in progress_doc


def test_workspace_room_only_lists_its_workspace(progress_doc, monkeypatch):
    scoped = pycrdt.Doc()
    scoped["workspaces"] = pycrdt.Map()
    progress_server = SimpleNamespace(
        rooms={
            progress_crdt.PROGRESS_ROOM: SimpleNamespace(ydoc=pycrdt.Doc()),
            "a.lynxkite": SimpleNamespace(ydoc=scoped),
        }
    )
    monkeypatch.setattr(progress_crdt, "_progress_server", progress_server)
    server = FakeServer("a.lynxkite", "b.lynxkite")
    progress_crdt.on_workspace_connection_open("a.lynxkite", server, "Alice")
    progress_crdt.on_workspace_connection_open("b.lynxkite", server, "Bob")
    assert set(progress_doc.keys()) == {"a.lynxkite", "b.lynxkite"}
    assert set(scoped["workspaces"].keys()) == {"a.lynxkite"}
    assert _entry(scoped["workspaces"], "a.lynxkite")["user"] == "Alice"
    progress_crdt.on_workspace_connection_close("a.lynxkite", server, "Alice")
    assert "a.lynxkite" not in scoped["workspaces"]
