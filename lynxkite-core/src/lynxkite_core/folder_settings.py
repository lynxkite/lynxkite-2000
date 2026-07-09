"""Per-folder settings.yaml loading and hierarchical merge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

SETTINGS_FILENAME = "settings.yaml"


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


def resolve_settings_section(
    data_root: Path, workspace_path: str, section: str
) -> dict[str, dict[str, Any]]:
    """Load settings.yaml from root→workspace folder; merge requested top-level section."""
    merged: dict[str, dict[str, Any]] = {}
    for directory in _containing_dirs(data_root, workspace_path):
        settings_path = directory / SETTINGS_FILENAME
        if not settings_path.is_file():
            continue
        with settings_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        values = raw.get(section)
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            merged.setdefault(key, {}).update(value)
    return merged
