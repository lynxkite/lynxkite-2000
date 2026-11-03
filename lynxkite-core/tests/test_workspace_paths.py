from pathlib import Path

import pytest
from lynxkite_core.workspace_paths import (
    LegacyWorkspaceError,
    box_data_dir,
    display_path,
    workspace_data_dir,
    workspace_file_path,
)


def test_workspace_bundle_paths():
    workspace_path = Path("team/analysis.lynxkite")

    assert workspace_file_path(workspace_path) == workspace_path / "workspace.json"
    assert workspace_data_dir(workspace_path) == workspace_path / "node_data"
    assert box_data_dir(workspace_path, "node-1") == workspace_path / "node_data" / "node-1"
    assert display_path(workspace_path, "node-1") == (
        workspace_path / "node_data" / "node-1" / "display.json"
    )


def test_legacy_workspace_paths_require_migration():
    workspace_path = Path("team/analysis.lynxkite.json")

    with pytest.raises(LegacyWorkspaceError, match="python3 scripts/migrate_workspaces.py"):
        workspace_file_path(workspace_path)
    with pytest.raises(LegacyWorkspaceError, match="python3 scripts/migrate_workspaces.py"):
        workspace_data_dir(workspace_path)
    with pytest.raises(LegacyWorkspaceError, match="python3 scripts/migrate_workspaces.py"):
        display_path(workspace_path, "node-1")


def test_workspace_file_path_requires_bundle():
    with pytest.raises(ValueError, match="must use the .lynxkite extension"):
        workspace_file_path(Path("team/analysis.json"))
