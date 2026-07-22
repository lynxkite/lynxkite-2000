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


def resolve_acl(path: str | None) -> dict[str, list[str]]:
    """Effective acl.read / acl.write lists from settings.yaml root→folder merge."""
    section = resolve_flat_section(_data_root, path or "", "acl")

    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) for item in value]
        return []

    return {"read": as_list(section.get("read")), "write": as_list(section.get("write"))}


def _matches(allowed: list[str], principals: set[str], *, authenticated: bool) -> bool:
    if "*" in allowed:
        return authenticated
    return bool(principals & set(allowed))


def has_permission(user: dict[str, Any], action: Action, path: str | None) -> bool:
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action {action!r}. Must be 'read' or 'write'.")
    grants = resolve_acl(resolve_folder(path))
    principals = user_principals(user)
    authenticated = bool(user.get("sub"))
    if action == "write":
        return _matches(grants["write"], principals, authenticated=authenticated)
    # write implies read
    return _matches(grants["read"], principals, authenticated=authenticated) or _matches(
        grants["write"], principals, authenticated=authenticated
    )


def effective_permissions(user: dict[str, Any], path: str | None) -> dict[str, bool]:
    return {
        "read": has_permission(user, "read", path),
        "write": has_permission(user, "write", path),
    }
