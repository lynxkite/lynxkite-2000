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
audience = os.environ.get("LYNXKITE_AUTH_AUDIENCE")  # SPA / OIDC client id
# Optional API Identifier (Keycloak often = client id). Leave unset for Auth0 SPA
# so the UI sends the ID token instead of an opaque access token.
api_audience = os.environ.get("LYNXKITE_AUTH_API_AUDIENCE") or None


class OIDCProvider:
    def __init__(self, issuer: str, audience: str, api_audience: str | None):
        self.audience = audience
        self.api_audience = api_audience
        # Strip trailing `/` so Auth0 issuers do not 404 on `//.well-known/...`.
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
        accepted = {a for a in (self.audience, self.api_audience) if a}
        for field in ["aud", "azp"]:
            value = payload.get(field)
            if value in accepted:
                return payload
            if isinstance(value, list) and accepted & set(value):
                return payload
        raise JWTError("Invalid audience")


@lru_cache
def get_provider():
    assert issuer is not None and audience is not None, "Authentication is not configured"
    return OIDCProvider(issuer, audience, api_audience)


def is_auth_enabled() -> bool:
    return bool(issuer and audience)


def is_read_only() -> bool:
    return os.environ.get("LYNXKITE_READ_ONLY") == "1"


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
    user = await get_current_user(request)
    if not is_auth_enabled():
        if action == "write" and is_read_only():
            raise HTTPException(status_code=403, detail="Forbidden")
        return
    if not acl.has_permission(user, action, requested_path):
        raise HTTPException(status_code=403, detail="Forbidden")


async def effective_permissions(request: Request, path: str | None = None) -> dict[str, bool]:
    if not is_auth_enabled():
        return {"read": True, "write": not is_read_only()}
    user = await get_current_user(request)
    return acl.effective_permissions(user, path)
