"""User authentication and permission checking using OpenID Connect."""

from functools import lru_cache
import os
import httpx
from jose import jwt
from jose.exceptions import JWTError
from fastapi import HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

security = HTTPBearer(auto_error=False)
issuer = os.environ.get("LYNXKITE_AUTH_ISSUER")
audience = os.environ.get("LYNXKITE_AUTH_AUDIENCE")


class OIDCProvider:
    def __init__(self, issuer: str, audience: str):
        self.issuer = issuer
        self.audience = audience
        self.config = httpx.get(f"{issuer}/.well-known/openid-configuration").json()
        self.jwks = httpx.get(self.config["jwks_uri"]).json()

    def verify(self, token: str):
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
                break
            if isinstance(value, list) and self.audience in value:
                break
        else:
            raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Bearer"})
        return payload


@lru_cache
def get_provider():
    assert issuer is not None and audience is not None, "Authentication is not configured"
    return OIDCProvider(issuer, audience)


def _is_auth_enabled() -> bool:
    return bool(issuer and audience)


async def get_current_user(request: Request):
    if not _is_auth_enabled():
        return {"sub": "user", "email": ""}
    credentials: HTTPAuthorizationCredentials | None = await security(request)
    if credentials is None:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    try:
        payload = get_provider().verify(token)
        print("Authenticated user:", payload.get("email"))
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )


def _has_permission(user, action: str, requested_path: str | None) -> bool:
    print(
        f"Checking permissions for action '{action}' on path '{requested_path}' for user '{user.get('email')}'"
    )
    return True


async def check_permission(request: Request, action: str, requested_path: str | None = None):
    user = await get_current_user(request)
    if not _has_permission(user, action, requested_path):
        raise HTTPException(status_code=403, detail="Forbidden")
