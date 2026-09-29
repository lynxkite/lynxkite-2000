from pathlib import Path

from lynxkite_core.workspace_paths import box_data_dir, display_path, workspace_file_path


def test_workspace_bundle_paths():
    workspace_path = Path("team/analysis.lynxkite")

    assert workspace_file_path(workspace_path) == workspace_path / "workspace.json"
    assert box_data_dir(workspace_path, "node-1") == workspace_path / "node_data" / "node-1"
    assert display_path(workspace_path, "node-1") == (
        workspace_path / "node_data" / "node-1" / "display.json"
    )


def test_legacy_workspace_paths():
    workspace_path = Path("team/analysis.lynxkite.json")

    assert workspace_file_path(workspace_path) == workspace_path
    assert box_data_dir(workspace_path, "node-1") == (
        Path("team/.workspace_files/analysis.lynxkite.json/node-1")
    )
    assert display_path(workspace_path, "node-1") == (
        Path("team/.workspace_files/analysis.lynxkite.json/node-1.json")
    )
