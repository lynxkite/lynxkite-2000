from pathlib import Path

from lynxkite_core.workspace_paths import (
    box_data_dir,
    display_path,
    is_workspace_bundle,
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


def test_legacy_workspace_file_is_not_a_bundle():
    assert not is_workspace_bundle(Path("team/analysis.lynxkite.json"))
