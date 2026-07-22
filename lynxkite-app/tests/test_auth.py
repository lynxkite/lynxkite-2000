from fastapi.testclient import TestClient

from lynxkite_app.main import app

client = TestClient(app)


def test_permissions_auth_off():
    response = client.get("/api/permissions?path=foo.lynxkite.json")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}


def test_permissions_me_auth_off():
    response = client.get("/api/permissions/me")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}
