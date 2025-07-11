import pandas as pd

from lynxkite.core import workspace, ops
from lynxkite_graph_analytics.core import Bundle, execute


async def test_multi_input_box():
    ws = workspace.Workspace(env="test")
    op = ops.op_registration("test")

    @op("Create Bundle")
    def create_bundle() -> Bundle:
        df = pd.DataFrame({"source": [1, 2, 3], "target": [4, 5, 6]})
        return Bundle(dfs={"edges": df})

    @op("Multi input op")
    def multi_input_op(bundles: list[Bundle]) -> int:
        return len(bundles)

    ws.add_node(
        id="1",
        type="node_type",
        title="Create Bundle",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="2",
        type="node_type",
        title="Create Bundle",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="3",
        type="node_type",
        title="Multi input op",
        position=workspace.Position(x=0, y=0),
    )
    ws.edges = [
        workspace.WorkspaceEdge(
            id="1", source="1", target="3", sourceHandle="output", targetHandle="bundles"
        ),
        workspace.WorkspaceEdge(
            id="2", source="2", target="3", sourceHandle="output", targetHandle="bundles"
        ),
    ]
    output = await execute(ws)
    assert all([node.data.error is None for node in ws.nodes])
    assert output[("3", "output")] == 2, (
        "Multi input op should return the correct number of bundles"
    )
