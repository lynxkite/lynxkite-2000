import uuid
from fastapi.testclient import TestClient
from lynxkite_app.main import app, detect_plugins, DATA_PATH
import os


client = TestClient(app)


def test_detect_plugins_with_plugins():
    # This test assumes that these plugins are installed as part of the testing process.
    plugins = detect_plugins()
    assert all(
        plugin in plugins.keys()
        for plugin in [
            "lynxkite_plugins.graph_analytics",
            "lynxkite_plugins.lynxscribe",
            "lynxkite_plugins.pillow_example",
        ]
    )


def test_get_catalog():
    response = client.get("/api/catalog")
    assert response.status_code == 200


def test_save_and_load():
    save_request = {
        "path": "test",
        "ws": {
            "env": "test",
            "nodes": [
                {
                    "id": "Node_1",
                    "type": "basic",
                    "data": {
                        "display": None,
                        "error": "Unknown operation.",
                        "title": "Test node",
                        "params": {"param1": "value"},
                    },
                    "position": {"x": -493.5496596237119, "y": 20.90123252513356},
                }
            ],
            "edges": [],
        },
    }
    response = client.post("/api/save", json=save_request)
    saved_ws = response.json()
    assert response.status_code == 200
    response = client.get("/api/load?path=test")
    assert response.status_code == 200
    assert saved_ws == response.json()


def test_list_dir():
    test_dir = str(uuid.uuid4())
    test_dir_full_path = DATA_PATH / test_dir
    test_dir_full_path.mkdir(exist_ok=True)
    test_file = test_dir_full_path / "test_file.txt"
    test_file.touch()
    response = client.get(f"/api/dir/list?path={str(test_dir)}")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["name"] == f"{test_dir}/test_file.txt"
    assert response.json()[0]["type"] == "workspace"
    test_file.unlink()
    test_dir_full_path.rmdir()


def test_make_dir():
    dir_name = str(uuid.uuid4())
    response = client.post("/api/dir/mkdir", json={"path": dir_name})
    assert response.status_code == 200
    assert os.path.exists(DATA_PATH / dir_name)
    os.rmdir(DATA_PATH / dir_name)
