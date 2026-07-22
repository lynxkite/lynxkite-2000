"""Folder-level ACL for LynxKite authorization, stored in settings.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from lynxkite_core.folder_settings import resolve_flat_section

Action = Literal["read", "write"]
VALID_ACTIONS = frozenset({"read", "write"})

_data_root: Path = Path()


def set_data_root(path: Path) -> None:
    """Set the data directory used to resolve per-folder settings.yaml files."""
    global _data_root
    _data_root = path


def get_data_root() -> Path:
    return _data_root


def resolve_folder(path: str | None) -> str:
    """Map a file or directory path to its containing folder for ACL lookup."""
    if not path:
        return ""
    path = path.replace("\\", "/").strip("/")
    if not path:
        return ""
    leaf = path.rsplit("/", 1)[-1]
    if leaf.endswith(".lynxkite.json") or ("." in leaf and not leaf.endswith(".")):
        if "/" in path:
            return path.rsplit("/", 1)[0] + "/"
        return ""
    return path + "/"


def user_principals(user: dict[str, Any]) -> set[str]:
    principals: set[str] = set()
    sub = user.get("sub")
    if sub:
        principals.add(f"sub:{sub}")
    groups = user.get("groups")
    if isinstance(groups, str):
        groups = [groups]
    if groups:
        for group in groups:
            principals.add(f"group:{group}")
    return principals


def _as_principal_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def resolve_acl(path: str | None) -> dict[str, list[str]]:
    """Effective acl.read / acl.write lists from settings.yaml root→folder merge."""
    lookup_path = path or ""
    section = resolve_flat_section(_data_root, lookup_path, "acl")
    return {
        "read": _as_principal_list(section.get("read")),
        "write": _as_principal_list(section.get("write")),
    }


def _principal_matches(allowed: list[str], principals: set[str], *, authenticated: bool) -> bool:
    if "*" in allowed:
        return authenticated
    return bool(principals & set(allowed))


def _allowed_for_action(user: dict[str, Any], action: Action, path: str | None) -> bool:
    allowed = resolve_acl(path)[action]
    authenticated = bool(user.get("sub"))
    return _principal_matches(allowed, user_principals(user), authenticated=authenticated)


def has_permission(
    user: dict[str, Any],
    action: Action,
    path: str | None,
    *,
    auth_enabled: bool,
) -> bool:
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action {action!r}. Must be 'read' or 'write'.")
    if not auth_enabled:
        return True
    folder = resolve_folder(path)
    if action == "read":
        return _allowed_for_action(user, "read", folder) or _allowed_for_action(
            user, "write", folder
        )
    return _allowed_for_action(user, "write", folder)


def effective_permissions(
    user: dict[str, Any],
    path: str | None,
    *,
    auth_enabled: bool,
) -> dict[str, bool]:
    if not auth_enabled:
        return {"read": True, "write": True}
    return {
        "read": has_permission(user, "read", path, auth_enabled=True),
        "write": has_permission(user, "write", path, auth_enabled=True),
    }
