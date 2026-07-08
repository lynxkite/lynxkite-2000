"""Per-folder settings.yaml for configs. It holds LIM box configs, and later can be used for ACL configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pydantic
import yaml

SETTINGS_FILENAME = "settings.yaml"


class LimBoxConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")
    cpu: str | None = None
    memory: str | None = None
    image: str | None = None
    port: int | None = None
    namespace: str | None = None
    env: list[dict[str, Any]] | None = None
    forward_env: list[str] | None = None
    storage_size: str | None = None
    storage_host_path: str | None = None
    timeout_seconds: int | None = None
    startup_timeout: int | None = None
    args: list[str] | None = None


class FolderSettings(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="ignore")
    lim: dict[str, LimBoxConfig] = pydantic.Field(default_factory=dict)


def _containing_dirs(data_root: Path, workspace_path: str) -> list[Path]:
    path = workspace_path.replace("\\", "/")
    if path.endswith(".lynxkite.json"):
        path = path.rsplit("/", 1)[0] if "/" in path else ""
    else:
        path = path.strip("/")
    dirs = [data_root]
    if path:
        parts = path.split("/")
        dirs.extend(data_root / "/".join(parts[: i + 1]) for i in range(len(parts)))
    return dirs


def resolve_lim_for_box(data_root: Path, workspace_path: str, op_id: str) -> dict[str, Any] | None:
    """Load settings.yaml from root → workspace folder; return merged LIM config for op_id."""
    merged: dict[str, dict[str, Any]] = {}
    for directory in _containing_dirs(data_root, workspace_path):
        settings_path = directory / SETTINGS_FILENAME
        if not settings_path.is_file():
            continue
        with settings_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        settings = FolderSettings.model_validate(raw)
        for box_id, box in settings.lim.items():
            merged.setdefault(box_id, {}).update(box.model_dump(exclude_none=True))
    result = merged.get(op_id)
    return result or None
