"""User authentication and permission checking using OpenID Connect.

Auth is on when LYNXKITE_AUTH_ISSUER and LYNXKITE_AUTH_AUDIENCE are set. Audience is the SPA client id.
The browser sends the ID token. Who may log in is configured in the identity provider. This module only verifies JWTs.
"""

from functools import lru_cache
import os
import httpx
from jose import jwt
from jose.exceptions import JWTError
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import acl

security = HTTPBearer(auto_error=False)
issuer = os.environ.get("LYNXKITE_AUTH_ISSUER")  # https://dev-lynxkite.eu.auth0.com/
audience = os.environ.get("LYNXKITE_AUTH_AUDIENCE")  # CzxYq4nCYr3qvp2t9GDZFb1G7bRkNtD0


class OIDCProvider:
    def __init__(self, issuer: str, audience: str):
        self.audience = audience
        # Strip trailing slash so Auth0 issuers do not 404 on well-known discovery.
        discovery = f"{issuer.rstrip('/')}/.well-known/openid-configuration"
        self.config = httpx.get(discovery).json()
        self.issuer = self.config.get("issuer") or issuer
        self.jwks = httpx.get(self.config["jwks_uri"]).json()

    def verify(self, token: str) -> dict:
        payload = jwt.decode(
            token,
            self.jwks,
            issuer=self.issuer,
            algorithms=["RS256"],
            options={"verify_aud": False},
        )
        value = payload.get("aud")
        if value == self.audience:
            return payload
        if isinstance(value, list) and self.audience in value:
            return payload
        if payload.get("azp") == self.audience:
            return payload
        raise JWTError("Invalid audience")


@lru_cache
def get_provider():
    assert issuer is not None and audience is not None, "Authentication is not configured"
    return OIDCProvider(issuer, audience)


def is_auth_enabled() -> bool:
    return bool(issuer and audience)


async def get_current_user(request: Request) -> acl.User:
    if not is_auth_enabled():
        return {"sub": "user", "email": ""}
    credentials: HTTPAuthorizationCredentials | None = await security(request)
    if credentials is None:
        return {}
    try:
        return get_provider().verify(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_permission(request: Request, action: acl.Action, requested_path: str | None = None):
    if not is_auth_enabled():
        return
    user = await get_current_user(request)
    if not acl.has_permission(user, action, requested_path):
        raise HTTPException(status_code=403, detail="Forbidden")


async def effective_permissions(request: Request, path: str | None = None) -> dict[str, bool]:
    if not is_auth_enabled():
        return {"read": True, "write": True}
    user = await get_current_user(request)
    return acl.effective_permissions(user, path)
