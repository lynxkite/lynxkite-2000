import pathlib
import shutil
import uuid
from fastapi.testclient import TestClient
from lynxkite_app.main import app
from lynxkite_core import ops
import os


ops.user_script_root = None
client = TestClient(app)


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
    ws = test_dir / "test_workspace.lynxkite.json"
    ws.touch()
    response = client.get(f"/api/dir/list?path={test_dir.as_posix()}")
    assert response.status_code == 200
    assert response.json() == [
        {"name": f"{test_dir.as_posix()}/test_dir", "type": "directory"},
        {"name": f"{test_dir.as_posix()}/test_file.txt", "type": "file"},
        {"name": f"{test_dir.as_posix()}/test_workspace.lynxkite.json", "type": "workspace"},
    ]
    file.unlink()
    ws.unlink()
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
    try:
        response = client.get("/api/dir/search?q=NEEDLE")
        assert response.status_code == 200
        assert response.json() == [
            {"name": f"{test_dir.as_posix()}/sub/Needle.txt", "type": "file"},
        ]
        assert client.get("/api/dir/search?q=%20").json() == []
    finally:
        shutil.rmtree(test_dir)


def test_make_dir():
    dir_name = str(uuid.uuid4())
    response = client.post("/api/dir/mkdir", json={"path": dir_name})
    assert response.status_code == 200
    assert os.path.exists(dir_name)
    os.rmdir(dir_name)
