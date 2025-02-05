import os
import pytest
import tempfile
from lynxkite.core import workspace


def test_save_load():
    ws = workspace.Workspace(env="test")
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="1",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Node 1", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="2",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Node 2", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    ws.edges.append(
        workspace.WorkspaceEdge(
            id="edge1",
            source="1",
            target="2",
            sourceHandle="",
            targetHandle="",
        )
    )
    path = os.path.join(tempfile.gettempdir(), "test_workspace.json")

    try:
        workspace.save(ws, path)
        assert os.path.exists(path)
        loaded_ws = workspace.load(path)
        assert loaded_ws.env == ws.env
        assert len(loaded_ws.nodes) == len(ws.nodes)
        assert len(loaded_ws.edges) == len(ws.edges)
        sorted_ws_nodes = sorted(ws.nodes, key=lambda x: x.id)
        sorted_loaded_ws_nodes = sorted(loaded_ws.nodes, key=lambda x: x.id)
        # We do manual assertion on each attribute because metadata is added at
        # loading time, which makes the objects different.
        for node, loaded_node in zip(sorted_ws_nodes, sorted_loaded_ws_nodes):
            assert node.id == loaded_node.id
            assert node.type == loaded_node.type
            assert node.data.title == loaded_node.data.title
            assert node.data.params == loaded_node.data.params
            assert node.position.x == loaded_node.position.x
            assert node.position.y == loaded_node.position.y
        sorted_ws_edges = sorted(ws.edges, key=lambda x: x.id)
        sorted_loaded_ws_edges = sorted(loaded_ws.edges, key=lambda x: x.id)
        for edge, loaded_edge in zip(sorted_ws_edges, sorted_loaded_ws_edges):
            assert edge.id == loaded_edge.id
            assert edge.source == loaded_edge.source
            assert edge.target == loaded_edge.target
            assert edge.sourceHandle == loaded_edge.sourceHandle
            assert edge.targetHandle == loaded_edge.targetHandle
    finally:
        os.remove(path)


@pytest.fixture(scope="session", autouse=True)
def populate_ops_catalog():
    from lynxkite.core import ops

    ops.register_passive_op("test", "Test Operation", [])


def test_update_metadata():
    ws = workspace.Workspace(env="test")
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="1",
            type="basic",
            data=workspace.WorkspaceNodeData(title="Test Operation", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="2",
            type="basic",
            data=workspace.WorkspaceNodeData(title="Unknown Operation", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    updated_ws = workspace._update_metadata(ws)
    assert updated_ws.nodes[0].data.meta.name == "Test Operation"
    assert updated_ws.nodes[0].data.error is None
    assert not hasattr(updated_ws.nodes[1].data, "meta")
    assert updated_ws.nodes[1].data.error == "Unknown operation."


def test_update_metadata_with_empty_workspace():
    ws = workspace.Workspace(env="test")
    updated_ws = workspace._update_metadata(ws)
    assert len(updated_ws.nodes) == 0
