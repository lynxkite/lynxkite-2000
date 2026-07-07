from lynxkite_assistant import python_workspace_conversion, workspace_backend
from pathlib import Path
import pytest
from lynxkite_core import ops, workspace
import os


@pytest.fixture(scope="module", autouse=True)
def setup_workspace():
    ops.detect_plugins()
    ops_usr = ops.user_script_root
    wb_crdt = workspace_backend.crdt
    ops.user_script_root = Path()
    # Disable CRDT for testing
    workspace_backend.crdt = None  # type: ignore

    yield

    ops.user_script_root = ops_usr
    workspace_backend.crdt = wb_crdt  # type: ignore


@pytest.fixture
def create_temp_file():
    created_files = []

    def _create_temp_file(filename):
        Path(filename).touch()
        created_files.append(filename)
        return filename

    yield _create_temp_file
    for file in created_files:
        if os.path.exists(file):
            os.remove(file)


def get_example_jsons():
    # we assume that the test are run from the root of the repository
    for root, dirs, files in os.walk("examples"):
        if "generated_samples" in root:
            continue
        for file in files:
            if file.endswith(".lynxkite.json"):
                yield os.path.join(root, file)
    # hand-made test files:
    yield "lynxkite-assistant/tests/files/original.lynxkite.json"


@pytest.mark.parametrize("og_ws_path", get_example_jsons())
def test_workspace_unchanged(og_ws_path, create_temp_file):
    if "Model definition" in og_ws_path:
        pytest.skip(
            "Skipping test because Repeat block causes cyclical nodes, whose handling is not implemented."
        )
    if "sql" in og_ws_path or "Gradio/Example" in og_ws_path:
        pytest.skip("Skipping test because SQL and Gradio examples have faulty blocks.")
    mod_ws_path = create_temp_file(
        og_ws_path.replace(".lynxkite.json", ".modified.lynxkite.json")
    )
    with open(mod_ws_path, "w") as f, open(og_ws_path) as og_f:
        f.write(og_f.read())
    ops.load_user_scripts(og_ws_path)
    og_ws = workspace.Workspace.load(og_ws_path)
    resp = python_workspace_conversion.workspace_to_python(og_ws)
    workspace_backend.set_workspace_file_content(mod_ws_path, resp)
    mod_ws = workspace.Workspace.load(mod_ws_path)

    def get_idx(node_ids, nid):
        return node_ids.index(nid) if nid in node_ids else -1

    og_ids = [n.id for n in og_ws.nodes]
    mod_ids = [n.id for n in mod_ws.nodes]
    s1 = set(
        (get_idx(og_ids, e.source), get_idx(og_ids, e.target)) for e in og_ws.edges
    )
    s2 = set(
        (get_idx(mod_ids, e.source), get_idx(mod_ids, e.target)) for e in mod_ws.edges
    )
    assert s1 == s2, "Edges differ"
    og_group_ids = {None: None}
    for og_node, mod_node in zip(og_ws.nodes, mod_ws.nodes):
        assert og_node.width == mod_node.width
        assert og_node.height == mod_node.height
        assert og_node.type == mod_node.type
        assert og_node.data.op_id == mod_node.data.op_id
        assert og_node.data.title == mod_node.data.title
        og_group = getattr(og_node.data, "parentId", None)
        mod_group = getattr(mod_node.data, "parentId", None)
        if og_group in og_group_ids:
            assert og_group_ids[og_group] == mod_group
        else:
            og_group_ids[og_group] = mod_group
        if (
            og_node.type == "comment" and mod_node.type == "comment"
        ):  # for comments, match based on text without whitespace and line breaks
            assert og_node.data.params.get("text", "").replace(" ", "").replace(
                "\n", ""
            ) == mod_node.data.params.get("text", "").replace(" ", "").replace("\n", "")
        else:
            assert og_node.data.params == mod_node.data.params
        assert og_node.position == mod_node.position


@pytest.fixture
def data_path():
    cwd = os.getcwd()
    fp = os.path.realpath(__file__)
    assert cwd in fp, f"Current working directory {cwd} is not in the file path {fp}"
    relative_path = Path(os.path.relpath(fp, cwd))
    yield relative_path.parent / "files"


def test_workspace_changed(data_path, create_temp_file):
    expected_ws_path = data_path / "after_change.lynxkite.json"
    mod_ws_path = create_temp_file(data_path / "modified.lynxkite.json")
    with open(mod_ws_path, "w") as f, open(expected_ws_path) as og_f:
        f.write(og_f.read())

    resp = open(data_path / "workspace_files/modified.py").read()
    ops.load_user_scripts(
        data_path / "modified.lynxkite.json"
    )  # we pretend that the code is running from the tests folder
    workspace_backend.set_workspace_file_content(mod_ws_path, resp)
    mod_ws = workspace.Workspace.load(mod_ws_path)
    expected_ws = workspace.Workspace.load(expected_ws_path)

    def get_idx(nodes, nid):
        ids = [n.id for n in nodes]
        return ids.index(nid) if nid in ids else -1

    s1 = set(
        (get_idx(expected_ws.nodes, e.source), get_idx(expected_ws.nodes, e.target))
        for e in expected_ws.edges
    )
    s2 = set(
        (get_idx(mod_ws.nodes, e.source), get_idx(mod_ws.nodes, e.target))
        for e in mod_ws.edges
    )
    assert s1 == s2, "Edges differ"
    for expected_node, mod_node in zip(expected_ws.nodes, mod_ws.nodes):
        assert expected_node.width == mod_node.width
        assert expected_node.height == mod_node.height
        assert expected_node.type == mod_node.type
        assert expected_node.data.op_id == mod_node.data.op_id
        assert expected_node.data.title == mod_node.data.title
        assert expected_node.data.params == mod_node.data.params
        assert expected_node.position == mod_node.position
