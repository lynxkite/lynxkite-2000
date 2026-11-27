"""Filesystem paths associated with workspace files and workspace bundles."""

from os import PathLike
from pathlib import Path

WORKSPACE_BUNDLE_SUFFIX = ".lynxkite"
WORKSPACE_FILENAME = "workspace.json"
NODE_DATA_DIRNAME = "node_data"


def is_workspace_bundle(path: str | PathLike[str]) -> bool:
    """Return whether a path identifies a workspace bundle directory."""
    return Path(path).suffix == WORKSPACE_BUNDLE_SUFFIX


def _require_workspace_bundle(path: str | PathLike[str]) -> Path:
    path = Path(path)
    if not is_workspace_bundle(path):
        raise ValueError(
            f"Workspace paths must use the {WORKSPACE_BUNDLE_SUFFIX} extension: {path}"
        )
    return path


def workspace_file_path(path: str | PathLike[str]) -> Path:
    """Return the JSON file that stores a workspace definition."""
    return _require_workspace_bundle(path) / WORKSPACE_FILENAME


def workspace_data_dir(workspace_path: str | PathLike[str]) -> Path:
    """Return the directory containing data produced by workspace nodes."""
    return _require_workspace_bundle(workspace_path) / NODE_DATA_DIRNAME


def box_data_dir(workspace_path: str | PathLike[str], node_id: str) -> Path:
    """Return the directory reserved for files produced by a workspace node."""
    return workspace_data_dir(workspace_path) / node_id


def display_path(workspace_path: str | PathLike[str], node_id: str) -> Path:
    """Return the persisted display-data file for a workspace node."""
    return box_data_dir(workspace_path, node_id) / "display.json"
