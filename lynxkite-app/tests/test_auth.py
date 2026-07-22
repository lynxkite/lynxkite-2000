import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from lynxkite_app import auth
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


@pytest.mark.asyncio
async def test_check_permission_rejects_invalid_action():
    request = Request({"type": "http", "method": "GET", "path": "/", "headers": []})
    with pytest.raises(HTTPException) as exc:
        await auth.check_permission(request, "execute", "x.lynxkite.json")
    assert exc.value.status_code == 400
