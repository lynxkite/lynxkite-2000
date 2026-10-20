"""Per-folder settings.yaml loading and hierarchical merge helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import deepmerge
import pydantic
import yaml

SETTINGS_FILENAME = "settings.yaml"


class BaseModel(pydantic.BaseModel):
    class Config:
        extra = "ignore"


class ACLSettings(BaseModel):
    """Access control list settings for a folder."""

    read: list[str] = pydantic.Field(
        default_factory=list, description="Users with read access, or `*` for anyone."
    )
    write: list[str] = pydantic.Field(
        default_factory=list, description="Users with write access, or `*` for anyone."
    )


class LIMSettings(BaseModel):
    """Configuration for LynxKite Inference Microservices."""


class LIMBoxConfig(BaseModel):
    """Configuration for one type of LynxKite Inference Microservice."""

    cpu: str | None = pydantic.Field(
        default=None, description="CPU resource request for the container"
    )
    memory: str | None = pydantic.Field(
        default=None, description="Memory resource request for the container"
    )
    image: str | None = pydantic.Field(default=None, description="Container image to use")
    port: int | None = pydantic.Field(default=None, description="Port to expose from the container")
    namespace: str | None = pydantic.Field(
        default=None, description="Kubernetes namespace for the container"
    )
    env: list[dict[str, Any]] | None = pydantic.Field(
        default=None, description="Environment variables for the container"
    )
    forward_env: list[str] | None = pydantic.Field(
        default=None, description="Environment variables to forward from the host"
    )
    storage_size: str | None = pydantic.Field(
        default=None, description="Requested storage size for the container"
    )
    storage_host_path: str | None = pydantic.Field(
        default=None, description="Host path for storage volume"
    )
    storage_pvc_claim_name: str | None = pydantic.Field(
        default=None, description="PVC claim name for storage volume"
    )
    timeout_seconds: int | None = pydantic.Field(
        default=None, description="Request timeout in seconds"
    )
    heartbeat_seconds: int | None = pydantic.Field(
        default=None, description="Heartbeat interval in seconds"
    )
    telemetry_interval_seconds: float | None = pydantic.Field(
        default=None, description="Telemetry reporting interval in seconds"
    )
    startup_timeout: int | None = pydantic.Field(
        default=None, description="Startup timeout in seconds"
    )
    args: list[str] | None = pydantic.Field(
        default=None, description="Command-line arguments for the container"
    )


class FolderSettings(BaseModel):
    """Folder-specific settings."""

    acl: ACLSettings = ACLSettings()
    lim: dict[str, LIMBoxConfig] = pydantic.Field(
        default_factory=dict,
        description="Overrides for LynxKite Inference Microservices.",
    )


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


def _iter_section_dicts(data_root: Path, workspace_path: str):
    """Yield raw configuration dicts from root→workspace settings.yaml files."""
    for directory in _containing_dirs(data_root, workspace_path):
        settings_path = directory / SETTINGS_FILENAME
        if not settings_path.is_file():
            continue
        with settings_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        yield raw


def get_settings(data_root: Path, workspace_path: str) -> FolderSettings:
    """Load settings applying to the given workspace."""
    merged: dict = {}
    for values in _iter_section_dicts(data_root, workspace_path):
        deepmerge.always_merger.merge(merged, values)
    return FolderSettings.model_validate(merged)


def get_settings_schema() -> dict:
    """Return the JSON schema for FolderSettings."""
    return FolderSettings.model_json_schema()
