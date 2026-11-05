"""Filesystem paths associated with workspace files and workspace bundles."""

import shlex
from os import PathLike
from pathlib import Path

WORKSPACE_BUNDLE_SUFFIX = ".lynxkite"
LEGACY_WORKSPACE_SUFFIX = ".lynxkite.json"
WORKSPACE_FILENAME = "workspace.json"
NODE_DATA_DIRNAME = "node_data"


class LegacyWorkspaceError(ValueError):
    """Raised when an unmigrated legacy workspace is used."""

    def __init__(self, path: str | PathLike[str], data_dir: str | PathLike[str] | None = None):
        self.path = Path(path)
        if data_dir is None:
            data_arg = "<DATA_DIR>"
        else:
            data_arg = shlex.quote(str(data_dir))
        command = f"python3 scripts/migrate_workspaces.py {data_arg}"
        super().__init__(
            f"Old workspace detected at '{self.path}'. Stop LynxKite and, from the "
            f"repository root, run `{command}`."
        )


def is_workspace_bundle(path: str | PathLike[str]) -> bool:
    """Return whether a path identifies a workspace bundle directory."""
    return Path(path).suffix == WORKSPACE_BUNDLE_SUFFIX


def is_legacy_workspace(path: str | PathLike[str]) -> bool:
    """Return whether a path uses the retired workspace suffix."""
    name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
    return name.endswith(LEGACY_WORKSPACE_SUFFIX)


def reject_legacy_workspace(path: str | PathLike[str]) -> None:
    """Raise a migration error when a path uses the retired workspace suffix."""
    path = Path(path)
    if is_legacy_workspace(path):
        raise LegacyWorkspaceError(path)


def _require_workspace_bundle(path: str | PathLike[str]) -> Path:
    path = Path(path)
    reject_legacy_workspace(path)
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
