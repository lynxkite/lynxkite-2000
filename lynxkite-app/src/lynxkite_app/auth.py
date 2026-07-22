"""User authentication and permission checking using OpenID Connect."""

from functools import lru_cache
import os
import httpx
from jose import jwt
from jose.exceptions import JWTError
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import acl

security = HTTPBearer(auto_error=False)
issuer = os.environ.get("LYNXKITE_AUTH_ISSUER")
audience = os.environ.get("LYNXKITE_AUTH_AUDIENCE")


class OIDCProvider:
    def __init__(self, issuer: str, audience: str):
        self.issuer = issuer
        self.audience = audience
        self.config = httpx.get(f"{issuer}/.well-known/openid-configuration").json()
        self.jwks = httpx.get(self.config["jwks_uri"]).json()

    def verify(self, token: str) -> dict:
        payload = jwt.decode(
            token,
            self.jwks,
            issuer=self.issuer,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        # Depending on the provider, the audience may be in different fields.
        for field in ["aud", "azp"]:
            value = payload.get(field)
            if value == self.audience:
                return payload
            if isinstance(value, list) and self.audience in value:
                return payload
        raise JWTError("Invalid audience")


@lru_cache
def get_provider():
    assert issuer is not None and audience is not None, "Authentication is not configured"
    return OIDCProvider(issuer, audience)


def is_auth_enabled() -> bool:
    return bool(issuer and audience)


async def get_current_user(request: Request):
    if not is_auth_enabled():
        return {"sub": "user", "email": ""}
    credentials: HTTPAuthorizationCredentials | None = await security(request)
    if credentials is None:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return get_provider().verify(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_permission(request: Request, action: acl.Action, requested_path: str | None = None):
    if action not in acl.VALID_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action {action!r}. Must be 'read' or 'write'.",
        )
    user = await get_current_user(request)
    if is_auth_enabled() and not acl.has_permission(user, action, requested_path):
        raise HTTPException(status_code=403, detail="Forbidden")


async def effective_permissions(request: Request, path: str | None = None) -> dict[str, bool]:
    if not is_auth_enabled():
        return {"read": True, "write": True}
    user = await get_current_user(request)
    return acl.effective_permissions(user, path)
