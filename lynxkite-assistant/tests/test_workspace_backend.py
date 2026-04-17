from lynxkite_core import workspace

from lynxkite_assistant.workspace_backend import _update_node_ids, _update_ws_positions


def test_update_node_ids_matches_by_op_id_and_params_and_updates_edges():
    source = workspace.Workspace()
    source.add_node(id="src-load-a", title="Load", params={"path": "a.csv"})
    source.add_node(id="src-load-b", title="Load", params={"path": "b.csv"})

    target = workspace.Workspace()
    target.add_node(id="tmp-1", title="Load", params={"path": "b.csv"})
    target.add_node(id="tmp-2", title="Load", params={"path": "a.csv"})
    target.add_edge("tmp-2", "output", "tmp-1", "input")

    _update_node_ids(source=source, target=target)

    assert {node.id for node in target.nodes} == {"src-load-a", "src-load-b"}
    assert [
        (edge.source, edge.target, edge.sourceHandle, edge.targetHandle) for edge in target.edges
    ] == [("src-load-a", "src-load-b", "output", "input")]


def test_update_node_ids_uses_params_when_source_has_duplicate_op_ids():
    source = workspace.Workspace()
    source.add_node(id="src-load-a", title="Load", params={"path": "a.csv"})
    source.add_node(id="src-load-b", title="Load", params={"path": "b.csv"})

    target = workspace.Workspace()
    target.add_node(id="tmp-only", title="Load", params={"path": "b.csv"})

    _update_node_ids(source=source, target=target)

    assert [node.id for node in target.nodes] == ["src-load-b"]


def test_update_node_ids_uses_params_when_target_has_duplicate_op_ids():
    source = workspace.Workspace()
    source.add_node(id="src-load-b", title="Load", params={"path": "b.csv"})

    target = workspace.Workspace()
    target.add_node(id="tmp-b", title="Load", params={"path": "b.csv"})
    target.add_node(id="tmp-a", title="Load", params={"path": "a.csv"})

    _update_node_ids(source=source, target=target)

    target_ids = {node.id for node in target.nodes}
    assert "src-load-b" in target_ids
    assert "tmp-a" in target_ids
    assert "tmp-b" not in target_ids


def test_update_node_ids_uses_neighbors_when_params_are_empty():
    source = workspace.Workspace()
    source_load_connected = source.add_node(id="src-load-connected", title="Load", params={})
    source.add_node(id="src-load-free", title="Load", params={})
    source_save = source.add_node(id="src-save", title="Save", params={})
    source.add_edge(source_load_connected, "output", source_save, "input")

    target = workspace.Workspace()
    target_load_free = target.add_node(id="tmp-load-free", title="Load", params={})
    target_load_connected = target.add_node(id="tmp-load-connected", title="Load", params={})
    target_save = target.add_node(id="tmp-save", title="Save", params={})
    target.add_edge(target_load_connected, "output", target_save, "input")

    _update_node_ids(source=source, target=target)

    assert target_load_connected.id == "src-load-connected"
    assert target_load_free.id == "src-load-free"
    assert target_save.id == "src-save"
    assert [
        (edge.source, edge.target, edge.sourceHandle, edge.targetHandle) for edge in target.edges
    ] == [("src-load-connected", "src-save", "output", "input")]


def test_update_ws_positions_copies_geometry_for_matching_ids_only():
    source = workspace.Workspace()
    source.add_node(
        id="shared-id",
        title="Source",
        position=workspace.Position(x=123, y=456),
        width=321,
        height=654,
    )

    target = workspace.Workspace()
    target.add_node(
        id="shared-id",
        title="Target",
        position=workspace.Position(x=0, y=0),
        width=10,
        height=20,
    )
    target.add_node(
        id="unmatched-id",
        title="Unmatched",
        position=workspace.Position(x=1, y=2),
        width=30,
        height=40,
    )

    _update_ws_positions(source=source, target=target)

    shared = next(node for node in target.nodes if node.id == "shared-id")
    unmatched = next(node for node in target.nodes if node.id == "unmatched-id")

    assert shared.position == workspace.Position(x=123, y=456)
    assert shared.width == 321
    assert shared.height == 654
    assert unmatched.position == workspace.Position(x=1, y=2)
    assert unmatched.width == 30
    assert unmatched.height == 40
