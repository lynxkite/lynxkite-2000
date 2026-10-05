"""Filesystem paths associated with workspace files and workspace bundles."""

from os import PathLike
from pathlib import Path

WORKSPACE_BUNDLE_SUFFIX = ".lynxkite"
WORKSPACE_FILENAME = "workspace.json"
NODE_DATA_DIRNAME = "node_data"


def is_workspace_bundle(path: str | PathLike[str]) -> bool:
    """Return whether a path identifies a workspace bundle directory."""
    return Path(path).suffix == WORKSPACE_BUNDLE_SUFFIX


def workspace_file_path(path: str | PathLike[str]) -> Path:
    """Return the JSON file that stores a workspace definition."""
    path = Path(path)
    if is_workspace_bundle(path):
        return path / WORKSPACE_FILENAME
    return path


def workspace_data_dir(workspace_path: str | PathLike[str]) -> Path:
    """Return the directory containing data produced by workspace nodes."""
    workspace_path = Path(workspace_path)
    if is_workspace_bundle(workspace_path):
        return workspace_path / NODE_DATA_DIRNAME
    return workspace_path.parent / ".workspace_files" / workspace_path.name


def box_data_dir(workspace_path: str | PathLike[str], node_id: str) -> Path:
    """Return the directory reserved for files produced by a workspace node."""
    return workspace_data_dir(workspace_path) / node_id


def display_path(workspace_path: str | PathLike[str], node_id: str) -> Path:
    """Return the persisted display-data file for a workspace node."""
    if not is_workspace_bundle(workspace_path):
        return box_data_dir(workspace_path, node_id).with_suffix(".json")
    return box_data_dir(workspace_path, node_id) / "display.json"
