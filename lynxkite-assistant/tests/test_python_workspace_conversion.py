import pytest
from lynxkite_core import workspace

from lynxkite_assistant.python_workspace_conversion import (
    python_to_workspace,
    workspace_to_python,
)


def test_python_to_workspace_builds_chain_with_constants_and_references():
    code = "\n".join(
        [
            "data = load(path='/tmp/data.csv')",
            "transformed = transform(input=data, n=3)",
            "sink(inp=transformed, flag=True)",
        ]
    )

    ws = python_to_workspace(code, error_on_unknown_ops=False)

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


def test_python_to_workspace_rejects_positional_arguments():
    with pytest.raises(AssertionError, match="Unexpected statement on line 1"):
        python_to_workspace("op(1)")


def test_python_to_workspace_rejects_kwargs_expansion():
    with pytest.raises(AssertionError, match=r"\*\*kwargs expansion is not supported"):
        python_to_workspace("op(**params)")


def test_python_to_workspace_rejects_unknown_variable_references():
    with pytest.raises(AssertionError, match="Unknown variable reference: missing"):
        python_to_workspace("sink(inp=missing)")


def test_workspace_to_python_ignores_edges_pointing_to_missing_nodes():
    ws = workspace.Workspace()
    ws.add_node(id="node-a", title="source", params={"k": 1})
    ws.add_edge("missing-node", "output", "node-a", "x")

    code = workspace_to_python(ws)
    lines = [
        line for line in code.splitlines() if line.strip() and not line.startswith("#")
    ]

    assert lines[1] == "res_source_1 = source(k=1)  # node-a"


def test_workspace_to_python_orders_dependencies_and_handles():
    ws = workspace.Workspace()
    ws.add_node(id="a", title="alpha", params={})
    ws.add_node(id="b", title="beta", params={})
    ws.add_node(id="c", title="merge", params={"const": 5})
    ws.add_edge("a", "output", "c", "z")
    ws.add_edge("b", "output", "c", "a")

    code = workspace_to_python(ws)
    lines = [
        line for line in code.splitlines() if line.strip() and not line.startswith("#")
    ]

    assert lines[1] == "res_alpha_1 = alpha()  # a"
    assert lines[2] == "res_beta_2 = beta()  # b"
    assert lines[3] == "res_merge_3 = merge(a=res_beta_2, z=res_alpha_1, const=5)  # c"
