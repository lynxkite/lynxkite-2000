from pathlib import Path

import pycrdt
import pytest
import yaml
from fastapi import HTTPException, Request
from fastapi.testclient import TestClient
from jose.exceptions import JWTError
from lynxkite_app import acl, auth, main, progress_crdt, ws_auth
from lynxkite_app.main import app
from lynxkite_app.ws_auth import authenticate_websocket
from pycrdt import (
    Channel,
    YMessageType,
    YSyncMessageType,
    create_sync_message,
    create_update_message,
)
from starlette.websockets import WebSocketDisconnect

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
    monkeypatch.setattr(ws_auth, "get_provider", lambda: FakeProvider())


@pytest.fixture
def guest_acl(tmp_path):
    acl.set_data_root(tmp_path)
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["anonymous", "*"], "write": ["group:lynxkite"]}})
    )
    yield
    acl.set_data_root(Path())


def test_permissions_auth_off():
    response = client.get("/api/permissions?path=foo.lynxkite")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}


def test_permissions_me_auth_off():
    response = client.get("/api/permissions/me")
    assert response.status_code == 200
    assert response.json() == {"read": True, "write": True}


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


def test_ws_logged_in_user_can_write(auth_on, guest_acl):
    scope: dict = {"query_string": b"access_token=ok"}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert scope["lynxkite_write"] is True
    assert scope["lynxkite_user"]["sub"] == "u1"


def test_ws_user_label_auth_off():
    scope: dict = {}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert ws_auth.user_label(scope) is None


def test_ws_user_label_auth_on(auth_on, guest_acl):
    scope: dict = {}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert ws_auth.user_label(scope) == "Guest"
    scope = {"query_string": b"access_token=ok"}
    assert authenticate_websocket(scope, "x.lynxkite.json") is False
    assert ws_auth.user_label(scope) == "u1"
    scope["lynxkite_user"] = {"sub": "u1", "email": "a@b.c", "name": "Alice"}
    assert ws_auth.user_label(scope) == "Alice"


def test_ws_bad_token_rejected(auth_on, guest_acl):
    assert authenticate_websocket({"query_string": b"access_token=bad"}, "x.lynxkite.json") is True


def test_ws_no_read_rejected(auth_on, tmp_path):
    acl.set_data_root(tmp_path)
    (tmp_path / "settings.yaml").write_text(yaml.dump({"acl": {"read": ["*"], "write": []}}))
    try:
        assert authenticate_websocket({}, "x.lynxkite.json") is True
    finally:
        acl.set_data_root(Path())


def test_readonly_client_messages():
    sync, awareness = YMessageType.SYNC, YMessageType.AWARENESS
    allowed = ws_auth._is_allowed_readonly_client_message
    assert allowed(bytes([sync, YSyncMessageType.SYNC_STEP1, 0]))
    assert allowed(bytes([awareness, 0]))
    assert not allowed(bytes([sync, YSyncMessageType.SYNC_STEP2, 0]))
    assert not allowed(bytes([sync, YSyncMessageType.SYNC_UPDATE, 0]))
    assert not allowed(bytes([sync]))
    assert not allowed(b"")
    assert not allowed(bytes([99]))


class FakeChannel(Channel):
    def __init__(self, messages: list[bytes]):
        self.messages = list(messages)

    @property
    def path(self) -> str:
        return "x"

    async def send(self, message: bytes) -> None:
        pass

    async def recv(self) -> bytes:
        return self.messages.pop(0)


async def test_write_filtering_channel_drops_updates():
    update = bytes([YMessageType.SYNC, YSyncMessageType.SYNC_UPDATE, 1])
    step1 = bytes([YMessageType.SYNC, YSyncMessageType.SYNC_STEP1, 0])
    readonly = ws_auth.WriteFilteringChannel(FakeChannel([update, step1]), can_write=False)
    assert await readonly.recv() == step1
    writable = ws_auth.WriteFilteringChannel(FakeChannel([update]), can_write=True)
    assert await writable.recv() == update


def _request(token: str | None = None) -> Request:
    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    return Request({"type": "http", "headers": headers})


async def test_assistant_requires_write(auth_on, guest_acl):
    with pytest.raises(HTTPException) as e:
        await main._assistant_requires_write(_request())
    assert e.value.status_code == 403
    await main._assistant_requires_write(_request("ok"))


async def test_assistant_open_when_auth_off():
    await main._assistant_requires_write(_request())


def test_provider_accepts_id_token_with_at_hash():
    # Keycloak ID tokens include at_hash. We verify the ID token without the access token.
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from jose import jwk, jwt

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public = jwk.construct(
        key.public_key().public_bytes(
            serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
        ),
        "RS256",
    ).to_dict()
    public["kid"] = "k1"
    claims = {
        "iss": "https://kc.example/realms/lynxkite",
        "aud": "lynxkite-web",
        "sub": "u1",
        "exp": 9999999999,
        "at_hash": "irrelevant",
    }
    token = jwt.encode(claims, pem, algorithm="RS256", headers={"kid": "k1"})
    provider = auth.OIDCProvider.__new__(auth.OIDCProvider)
    provider.audience = "lynxkite-web"
    provider.issuer = claims["iss"]
    provider.jwks = {"keys": [public]}
    assert provider.verify(token)["sub"] == "u1"


@pytest.fixture
def staff_only_acl(tmp_path):
    acl.set_data_root(tmp_path)
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["*"], "write": ["group:lynxkite"]}})
    )
    yield
    acl.set_data_root(Path())


def test_progress_ws_requires_root_read(auth_on, staff_only_acl):
    with (
        TestClient(app) as c,
        pytest.raises(WebSocketDisconnect) as e,
        c.websocket_connect("/ws/progress/crdt/progress"),
    ):
        pass
    assert e.value.code == 4403


def test_progress_ws_is_read_only(auth_on, staff_only_acl):
    fake = pycrdt.Doc()
    fake["workspaces"] = pycrdt.Map()
    fake["workspaces"]["fake.lynxkite.json"] = "{}"
    with TestClient(app) as c:
        with c.websocket_connect("/ws/progress/crdt/progress?access_token=ok") as ws:
            assert ws.receive_bytes()[0] == YMessageType.SYNC
            ws.send_bytes(create_update_message(fake.get_update()))
            # The server answers SyncStep1 in order, so the update was handled by then.
            ws.send_bytes(create_sync_message(pycrdt.Doc()))
            while True:
                msg = ws.receive_bytes()
                if msg[:2] == bytes([YMessageType.SYNC, YSyncMessageType.SYNC_STEP2]):
                    break
        assert "fake.lynxkite.json" not in progress_crdt._progress_ydoc["workspaces"]


@pytest.fixture
def folder_only_acl(tmp_path):
    """u1 can read the "mine" folder, but not the root."""
    acl.set_data_root(tmp_path)
    (tmp_path / "settings.yaml").write_text(
        yaml.dump({"acl": {"read": ["group:admins"], "write": ["group:admins"]}})
    )
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "settings.yaml").write_text(yaml.dump({"acl": {"read": ["sub:u1"]}}))
    yield
    acl.set_data_root(Path())


def test_progress_ws_workspace_room_needs_only_workspace_read(auth_on, folder_only_acl):
    with TestClient(app) as c:
        for room in ["progress", "other/b.lynxkite"]:
            with (
                pytest.raises(WebSocketDisconnect) as e,
                c.websocket_connect(f"/ws/progress/crdt/{room}?access_token=ok"),
            ):
                pass
            assert e.value.code == 4403
        with c.websocket_connect("/ws/progress/crdt/mine/a.lynxkite?access_token=ok") as ws:
            assert ws.receive_bytes()[0] == YMessageType.SYNC
