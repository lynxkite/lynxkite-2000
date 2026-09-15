from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from jose.exceptions import JWTError

from lynxkite_app import acl, auth
from lynxkite_app.main import app
from lynxkite_app.ws_auth import authenticate_websocket

client = TestClient(app)


class FakeProvider:
    def verify(self, token: str) -> dict:
        if token == "ok":
            return {"sub": "u1", "groups": ["lynxkite"]}
        raise JWTError("invalid token")


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setattr(auth, "issuer", "https://example.auth0.com/")
    monkeypatch.setattr(auth, "audience", "spa-client")
    monkeypatch.setattr(auth, "get_provider", lambda: FakeProvider())


@pytest.fixture
def guest_acl(tmp_path):
    acl.set_data_root(tmp_path)
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["anonymous", "*"], "write": ["group:lynxkite"]}})
    )
    yield
    acl.set_data_root(Path())


def test_permissions_auth_off():
    response = client.get("/api/permissions?path=foo.lynxkite.json")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}


def test_permissions_me_auth_off():
    response = client.get("/api/permissions/me")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}


def test_read_only(monkeypatch):
    monkeypatch.setenv("LYNXKITE_READ_ONLY", "1")
    assert client.get("/api/permissions/me").json() == {"read": True, "write": False}
    assert client.get("/api/config").json()["read_only"] is True
    assert client.post("/api/dir/mkdir", json={"path": "ro-forbidden"}).status_code == 403


def test_guest_is_anonymous(auth_on, guest_acl):
    assert client.get("/api/permissions?path=foo.lynxkite.json").json() == {
        "read": True,
        "write": False,
    }
    assert (
        client.get(
            "/api/permissions?path=foo.lynxkite.json",
            headers={"Authorization": "Bearer bad"},
        ).status_code
        == 401
    )
    assert client.get(
        "/api/permissions?path=foo.lynxkite.json",
        headers={"Authorization": "Bearer ok"},
    ).json() == {"read": True, "write": True}


def test_ws_guest(auth_on, guest_acl):
    scope: dict = {}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert scope["lynxkite_write"] is False


def test_ws_read_only(monkeypatch):
    monkeypatch.setenv("LYNXKITE_READ_ONLY", "1")
    scope: dict = {}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert scope["lynxkite_write"] is False
