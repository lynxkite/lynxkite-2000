import pathlib
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
    response = client.get(f"/api/dir/list?path={str(test_dir)}")
    assert response.status_code == 200
    assert response.json() == [
        {"name": f"{test_dir}/test_dir", "type": "directory"},
        {"name": f"{test_dir}/test_file.txt", "type": "file"},
        {"name": f"{test_dir}/test_workspace.lynxkite.json", "type": "workspace"},
    ]
    file.unlink()
    ws.unlink()
    dir.rmdir()


def test_make_dir():
    dir_name = str(uuid.uuid4())
    response = client.post("/api/dir/mkdir", json={"path": dir_name})
    assert response.status_code == 200
    assert os.path.exists(dir_name)
    os.rmdir(dir_name)
