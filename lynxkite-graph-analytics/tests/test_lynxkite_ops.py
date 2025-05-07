import pandas as pd
import pytest
import networkx as nx

from lynxkite.core import workspace, ops
from lynxkite_graph_analytics.core import Bundle, execute, ENV


async def test_execute_operation_not_in_catalog():
    ws = workspace.Workspace(env=ENV)
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="1",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Non existing op", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    await execute(ws)
    assert ws.nodes[0].data.error == "Operation not found in catalog"


async def test_execute_operation_inputs_correct_cast():
    # Test that the automatic casting of operation inputs works correctly.

    op = ops.op_registration("test")

    @op("Create Bundle")
    def create_bundle() -> Bundle:
        df = pd.DataFrame({"source": [1, 2, 3], "target": [4, 5, 6]})
        return Bundle(dfs={"edges": df})

    @op("Bundle to Graph")
    def bundle_to_graph(graph: nx.Graph) -> nx.Graph:
        return graph

    @op("Graph to Bundle")
    def graph_to_bundle(bundle: Bundle) -> pd.DataFrame:
        return list(bundle.dfs.values())[0]

    @op("Dataframe to Bundle")
    def dataframe_to_bundle(bundle: Bundle) -> Bundle:
        return bundle

    ws = workspace.Workspace(env="test")
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="1",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Create Bundle", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="2",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Bundle to Graph", params={}),
            position=workspace.Position(x=100, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="3",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Graph to Bundle", params={}),
            position=workspace.Position(x=200, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="4",
            type="node_type",
            data=workspace.WorkspaceNodeData(title="Dataframe to Bundle", params={}),
            position=workspace.Position(x=300, y=0),
        )
    )
    ws.edges = [
        workspace.WorkspaceEdge(
            id="1", source="1", target="2", sourceHandle="output", targetHandle="graph"
        ),
        workspace.WorkspaceEdge(
            id="2", source="2", target="3", sourceHandle="output", targetHandle="bundle"
        ),
        workspace.WorkspaceEdge(
            id="3", source="3", target="4", sourceHandle="output", targetHandle="bundle"
        ),
    ]

    await execute(ws)

    assert all([node.data.error is None for node in ws.nodes])


async def test_multiple_inputs():
    """Make sure each input goes to the right argument."""
    op = ops.op_registration("test")

    @op("One")
    def one():
        return 1

    @op("Two")
    def two():
        return 2

    @op("Smaller?", view="visualization")
    def is_smaller(a, b):
        return a < b

    ws = workspace.Workspace(env="test")
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="one",
            type="cool",
            data=workspace.WorkspaceNodeData(title="One", params={}),
            position=workspace.Position(x=0, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="two",
            type="cool",
            data=workspace.WorkspaceNodeData(title="Two", params={}),
            position=workspace.Position(x=100, y=0),
        )
    )
    ws.nodes.append(
        workspace.WorkspaceNode(
            id="smaller",
            type="cool",
            data=workspace.WorkspaceNodeData(title="Smaller?", params={}),
            position=workspace.Position(x=200, y=0),
        )
    )
    ws.edges = [
        workspace.WorkspaceEdge(
            id="one",
            source="one",
            target="smaller",
            sourceHandle="output",
            targetHandle="a",
        ),
        workspace.WorkspaceEdge(
            id="two",
            source="two",
            target="smaller",
            sourceHandle="output",
            targetHandle="b",
        ),
    ]

    await execute(ws)

    assert ws.nodes[-1].data.display is True
    # Flip the inputs.
    ws.edges[0].targetHandle = "b"
    ws.edges[1].targetHandle = "a"
    await execute(ws)
    assert ws.nodes[-1].data.display is False


async def test_optional_inputs():
    @ops.op("test", "one")
    def one():
        return 1

    @ops.op("test", "maybe add")
    def maybe_add(a: int, b: int | None = None):
        return a + (b or 0)

    assert maybe_add.__op__.inputs == [
        ops.Input(name="a", type=int, position="left"),
        ops.Input(name="b", type=int | None, position="left"),
    ]
    ws = workspace.Workspace(env="test", nodes=[], edges=[])
    a = ws.add_node(one)
    b = ws.add_node(maybe_add)
    await execute(ws)
    assert b.data.error == "Missing input: a"
    ws.add_edge(a, "output", b, "a")
    outputs = await execute(ws)
    assert outputs[b.id, "output"] == 1


if __name__ == "__main__":
    pytest.main()
