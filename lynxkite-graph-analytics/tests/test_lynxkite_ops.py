import pandas as pd
import pytest
import networkx as nx

from lynxkite_core import workspace, ops
from lynxkite_graph_analytics.core import Bundle, execute, ENV
from lynxkite_graph_analytics.ops.file_ops import FileFormat, export_to_file


async def test_execute_operation_not_in_catalog():
    ws = workspace.Workspace(env=ENV)
    ws.add_node(
        id="1",
        type="node_type",
        title="Non existing op",
        position=workspace.Position(x=0, y=0),
    )
    await execute(ws)
    assert ws.nodes[0].data.error == "Unknown operation."


@pytest.mark.parametrize(
    "file_format, method_name",
    [
        (FileFormat.csv, "to_csv"),
        (FileFormat.json, "to_json"),
        (FileFormat.parquet, "to_parquet"),
        (FileFormat.excel, "to_excel"),
    ],
)
async def test_export_to_file(monkeypatch, file_format, method_name):
    df = pd.DataFrame({"id": [1, 2], "val": ["a", "b"]})
    bundle = Bundle(dfs={"data": df})
    path = "some/path/file.out"

    called = {}

    def fake_writer(self, filepath, **kwargs):
        called["path"] = filepath
        called["kwargs"] = kwargs

    monkeypatch.setattr(pd.DataFrame, method_name, fake_writer)

    export_to_file(
        bundle,
        table_name="data",
        filename=path,
        file_format=file_format,
    )

    assert called["path"] == path
    assert isinstance(called["kwargs"], dict)


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
    ws.add_node(
        id="1",
        type="node_type",
        title="Create Bundle",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="2",
        type="node_type",
        title="Bundle to Graph",
        position=workspace.Position(x=100, y=0),
    )
    ws.add_node(
        id="3",
        type="node_type",
        title="Graph to Bundle",
        position=workspace.Position(x=200, y=0),
    )
    ws.add_node(
        id="4",
        type="node_type",
        title="Dataframe to Bundle",
        position=workspace.Position(x=300, y=0),
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
    ws.add_node(
        id="one",
        type="cool",
        title="One",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="two",
        type="cool",
        title="Two",
        position=workspace.Position(x=100, y=0),
    )
    ws.add_node(
        id="smaller",
        type="cool",
        title="Smaller?",
        position=workspace.Position(x=200, y=0),
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
        ops.Input(name="a", type=int, position=ops.Position.LEFT),
        ops.Input(name="b", type=int | None, position=ops.Position.LEFT),
    ]
    ws = workspace.Workspace(env="test", nodes=[], edges=[])
    a = ws.add_node(one)
    b = ws.add_node(maybe_add)
    await execute(ws)
    assert b.data.error == "Missing input: a"
    ws.add_edge(a, "output", b, "a")
    result = await execute(ws)
    assert result.outputs[b.id, "output"] == 1


if __name__ == "__main__":
    pytest.main()
