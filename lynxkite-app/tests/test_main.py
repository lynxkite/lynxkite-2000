import os
import pathlib
import shutil
import tempfile
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


def test_search_dir():
    test_dir = pathlib.Path() / str(uuid.uuid4())
    (test_dir / "sub").mkdir(parents=True)
    (test_dir / "__hidden").mkdir()
    (test_dir / ".dot").mkdir()
    (test_dir / "sub" / "Needle.txt").touch()
    (test_dir / "sub" / "other.txt").touch()
    (test_dir / "__hidden" / "needle.txt").touch()
    (test_dir / ".dot" / "needle.txt").touch()
    (test_dir / "sub" / "settings.yaml").touch()
    try:
        response = client.get("/api/dir/search?q=NEEDLE")
        assert response.status_code == 200
        assert response.json() == [
            {"name": f"{test_dir.as_posix()}/sub/Needle.txt", "type": "file"},
        ]
        assert client.get("/api/dir/search?q=%20").json() == []
        # settings.yaml is never listed, even when it matches.
        assert client.get("/api/dir/search?q=settings.yaml").json() == []
    finally:
        shutil.rmtree(test_dir)


def test_search_dir_limit():
    test_dir = pathlib.Path() / str(uuid.uuid4())
    test_dir.mkdir()
    for i in range(25):
        (test_dir / f"file_{i}.txt").touch()
    try:
        response = client.get(f"/api/dir/search?q={test_dir.name}")
        assert response.status_code == 200
        assert len(response.json()) == 20
        response = client.get(f"/api/dir/search?q={test_dir.name}&limit=5")
        assert len(response.json()) == 5
    finally:
        shutil.rmtree(test_dir)


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


def test_upload_nested_folder():
    with tempfile.TemporaryDirectory(dir=".") as directory:
        root = pathlib.Path(directory)
        response = client.post(
            "/api/upload",
            params={"dir": root.name},
            files={"file": ("folder/subfolder/hello.txt", b"hello")},
        )
        assert response.status_code == 200
        assert (root / "folder/subfolder/hello.txt").read_bytes() == b"hello"

        response = client.post(
            "/api/upload",
            params={"dir": root.name},
            files=[
                ("directory", (None, "folder/empty/nested")),
                ("directory", (None, "folder/another-empty")),
            ],
        )
        assert response.status_code == 200
        assert (root / "folder/empty/nested").is_dir()
        assert (root / "folder/another-empty").is_dir()

        response = client.post(
            "/api/upload",
            params={"dir": root.name},
            files={"file": ("../outside.txt", b"no")},
        )
        assert response.status_code == 400
