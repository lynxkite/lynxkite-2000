from lynxkite_assistant import python_workspace_conversion, workspace_backend
from pathlib import Path
import pytest
from lynxkite_core import ops, workspace
import os


@pytest.fixture
def data_path():
    cwd = os.getcwd()
    fp = os.path.realpath(__file__)
    assert cwd in fp, f"Current working directory {cwd} is not in the file path {fp}"
    relative_path = Path(os.path.relpath(fp, cwd))
    ops.detect_plugins()
    # Disable CRDT for testing
    workspace_backend.crdt = None  # type: ignore
    ops.user_script_root = relative_path.parent
    relative_data_dir = relative_path.parent / "files"
    (relative_data_dir / "modified.lynxkite.json").touch()
    yield relative_data_dir
    # os.remove(relative_data_dir / "modified.lynxkite.json")


def test_workspace_unchanged(data_path):
    og_ws_path = data_path / "original.lynxkite.json"
    mod_ws_path = data_path / "modified.lynxkite.json"
    with open(mod_ws_path, "w") as f, open(og_ws_path) as og_f:
        f.write(og_f.read())
    og_ws = workspace.Workspace.load(og_ws_path)
    resp = python_workspace_conversion.workspace_to_python(og_ws)
    workspace_backend.set_workspace_file_content(mod_ws_path, resp)
    mod_ws = workspace.Workspace.load(mod_ws_path)
    assert len(og_ws.edges) == len(mod_ws.edges)
    sorted_og_nodes = sorted(og_ws.nodes, key=lambda n: n.data.op_id)
    sorted_mod_nodes = sorted(mod_ws.nodes, key=lambda n: n.data.op_id)
    for og_node, mod_node in zip(sorted_og_nodes, sorted_mod_nodes):
        assert og_node.width == mod_node.width
        assert og_node.height == mod_node.height
        assert og_node.type == mod_node.type
        assert og_node.data.op_id == mod_node.data.op_id
        assert og_node.data.title == mod_node.data.title
        assert og_node.data.params == mod_node.data.params
        assert og_node.position == mod_node.position


def test_workspace_changed(data_path):
    expected_ws_path = data_path / "after_change.lynxkite.json"
    mod_ws_path = data_path / "modified.lynxkite.json"
    with open(mod_ws_path, "w") as f, open(expected_ws_path) as og_f:
        f.write(og_f.read())

    resp = open(data_path / "workspace_files/modified.py").read()
    ops.load_user_scripts(
        "files/modified.lynxkite.json"
    )  # we pretend that the code is running from the tests folder
    workspace_backend.set_workspace_file_content(mod_ws_path, resp)
    mod_ws = workspace.Workspace.load(mod_ws_path)
    expected_ws = workspace.Workspace.load(expected_ws_path)
    assert len(expected_ws.edges) == len(mod_ws.edges)
    sorted_expected_nodes = sorted(expected_ws.nodes, key=lambda n: n.data.op_id)
    sorted_mod_nodes = sorted(mod_ws.nodes, key=lambda n: n.data.op_id)
    for expected_node, mod_node in zip(sorted_expected_nodes, sorted_mod_nodes):
        assert expected_node.width == mod_node.width
        assert expected_node.height == mod_node.height
        assert expected_node.type == mod_node.type
        assert expected_node.data.op_id == mod_node.data.op_id
        assert expected_node.data.title == mod_node.data.title
        assert expected_node.data.params == mod_node.data.params
        assert expected_node.position == mod_node.position
