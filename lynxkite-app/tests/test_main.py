import os
import pathlib
import uuid

from fastapi.testclient import TestClient
from lynxkite_app import main
from lynxkite_core import ops, workspace

ops.user_script_root = None
client = TestClient(main.app)


def test_get_catalog():
    response = client.get("/api/catalog?workspace=test")
    assert response.status_code == 200


def test_list_dir():
    test_dir = pathlib.Path() / str(uuid.uuid4())
    test_dir.mkdir(parents=True, exist_ok=True)
    dir = test_dir / "test_dir"
    dir.mkdir(exist_ok=True)
    file = test_dir / "test_file.txt"
    file.touch()
    ws = test_dir / "test_workspace.lynxkite"
    ws.mkdir()
    response = client.get(f"/api/dir/list?path={test_dir.as_posix()}")
    assert response.status_code == 200
    assert response.json() == [
        {"name": f"{test_dir.as_posix()}/test_dir", "type": "directory"},
        {"name": f"{test_dir.as_posix()}/test_file.txt", "type": "file"},
        {"name": f"{test_dir.as_posix()}/test_workspace.lynxkite", "type": "workspace"},
    ]
    file.unlink()
    ws.rmdir()
    dir.rmdir()


def test_make_dir():
    dir_name = str(uuid.uuid4())
    response = client.post("/api/dir/mkdir", json={"path": dir_name})
    assert response.status_code == 200
    assert os.path.exists(dir_name)
    os.rmdir(dir_name)


def test_rename_workspace_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "data_path", tmp_path)
    monkeypatch.setattr(main.crdt, "delete_room", lambda name: None)
    old_path = tmp_path / "old.lynxkite"
    workspace.Workspace().save(old_path)
    crdt_path = tmp_path / ".crdt" / "old.lynxkite.crdt"
    crdt_path.parent.mkdir()
    crdt_path.touch()

    response = client.post(
        "/api/rename",
        json={"old_path": "old.lynxkite", "new_path": "new.lynxkite"},
    )

    assert response.status_code == 200
    assert (tmp_path / "new.lynxkite" / "workspace.json").is_file()
    assert not crdt_path.exists()


def test_rename_legacy_workspace_to_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "data_path", tmp_path)
    monkeypatch.setattr(main.crdt, "delete_room", lambda name: None)
    old_path = tmp_path / "old.lynxkite.json"
    workspace.Workspace().save(old_path)
    old_data_path = tmp_path / ".workspace_files" / old_path.name
    old_data_path.mkdir(parents=True)
    (old_data_path / "node-1.json").write_text("{}")

    response = client.post(
        "/api/rename",
        json={"old_path": "old.lynxkite.json", "new_path": "new.lynxkite"},
    )

    assert response.status_code == 200
    assert (tmp_path / "new.lynxkite" / "workspace.json").is_file()
    assert (tmp_path / "new.lynxkite" / "node_data" / "node-1" / "display.json").is_file()
    assert not old_data_path.exists()


def test_delete_workspace_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "data_path", tmp_path)
    monkeypatch.setattr(main.crdt, "delete_room", lambda name: None)
    bundle_path = tmp_path / "delete-me.lynxkite"
    workspace.Workspace().save(bundle_path)
    (bundle_path / "node_data" / "node-1").mkdir(parents=True)
    crdt_path = tmp_path / ".crdt" / "delete-me.lynxkite.crdt"
    crdt_path.parent.mkdir()
    crdt_path.touch()

    response = client.post("/api/delete", json={"path": "delete-me.lynxkite"})

    assert response.status_code == 200
    assert not bundle_path.exists()
    assert not crdt_path.exists()


def test_get_node_output_from_workspace_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "data_path", tmp_path)
    display_path = tmp_path / "view.lynxkite" / "node_data" / "node-1" / "display.json"
    display_path.parent.mkdir(parents=True)
    display_path.write_text('{"value": 42}')

    response = client.get("/api/node_output?workspace=view.lynxkite&node_id=node-1&version=1")

    assert response.status_code == 200
    assert response.json() == {"value": 42}
