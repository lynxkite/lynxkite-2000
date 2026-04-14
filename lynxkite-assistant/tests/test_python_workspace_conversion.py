import pytest
from lynxkite_core import workspace

from lynxkite_assistant.python_workspace_conversion import python_to_workspace, workspace_to_python


def test_python_to_workspace_builds_chain_with_constants_and_references():
    code = "\n".join(
        [
            "data = load(path='/tmp/data.csv')",
            "transformed = transform(input=data, n=3)",
            "sink(inp=transformed, flag=True)",
        ]
    )

    ws = python_to_workspace(code)

    assert [node.id for node in ws.nodes] == [
        "load on line 1",
        "transform on line 2",
        "sink on line 3",
    ]
    assert [node.data.title for node in ws.nodes] == ["load", "transform", "sink"]
    assert ws.nodes[0].data.params == {"path": "/tmp/data.csv"}
    assert ws.nodes[1].data.params == {"n": 3}
    assert ws.nodes[2].data.params == {"flag": True}

    assert [(edge.source, edge.target, edge.targetHandle) for edge in ws.edges] == [
        ("load on line 1", "transform on line 2", "input"),
        ("transform on line 2", "sink on line 3", "inp"),
    ]
    assert ws.nodes[0].position == workspace.Position(x=0, y=0)
    assert ws.nodes[1].position == workspace.Position(x=500, y=0)
    assert ws.nodes[2].position == workspace.Position(x=1000, y=0)


def test_python_to_workspace_stacks_parallel_inputs_on_different_rows():
    code = "\n".join(
        [
            "import boxes",
            "left_result = left()",
            "right_result = right()",
            "combine(x=left_result, y=right_result)",
        ]
    )

    ws = python_to_workspace(code)
    positions = {node.id: node.position for node in ws.nodes}

    assert positions["left on line 2"] == workspace.Position(x=0, y=0)
    assert positions["right on line 3"] == workspace.Position(x=0, y=450)
    assert positions["combine on line 4"] == workspace.Position(x=500, y=0)


def test_python_to_workspace_rejects_positional_arguments():
    with pytest.raises(AssertionError, match="Unexpected statement on line 1"):
        python_to_workspace("op(1)")


def test_workspace_to_python_ignores_edges_pointing_to_missing_nodes():
    ws = workspace.Workspace()
    ws.add_node(id="node-a", title="source", params={"k": 1})
    ws.add_edge("missing-node", "output", "node-a", "x")

    code = workspace_to_python(ws)

    assert code.splitlines()[3] == "source(k=1)"


def test_workspace_to_python_orders_dependencies_and_handles():
    ws = workspace.Workspace()
    ws.add_node(id="a", title="alpha", params={})
    ws.add_node(id="b", title="beta", params={})
    ws.add_node(id="c", title="merge", params={"const": 5})
    ws.add_edge("a", "output", "c", "z")
    ws.add_edge("b", "output", "c", "a")

    code = workspace_to_python(ws)
    lines = code.splitlines()

    assert lines[3] == "res_alpha_1 = alpha()"
    assert lines[4] == "res_beta_2 = beta()"
    assert lines[5] == "merge(a=res_beta_2, z=res_alpha_1, const=5)"
