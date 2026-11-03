import pandas as pd
import pytest
from pydantic_core import to_json
from lynxkite_core import ops, workspace
from lynxkite_graph_analytics.core import Bundle, execute


@pytest.mark.parametrize("view", ["visualization", "graph_creation_view"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_visualization_display_is_saved_after_execution(tmp_path, view, is_async):
    env = "test_graph_display_saving"
    display = {"display_value": 1}

    def sync_display_op():
        return display

    async def async_display_op():
        return display

    display_op = ops.op(env, "display", view=view)(
        async_display_op if is_async else sync_display_op
    )
    ws_path = str(tmp_path / "example.lynxkite.json")
    ws = workspace.Workspace(env=env, path=ws_path)
    node = ws.add_node(display_op)
    path = ops.build_output_path(ws_path, node.id)

    for value, expected_version in [(1, 1), (1, 1), (2, 2)]:
        display["display_value"] = value
        await execute(ws)

        assert node.data.error is None
        assert node.data.status == workspace.NodeStatus.done
        if view == "graph_creation_view":
            assert node.data.display == display
            assert node.data.display_version is None
            assert not path.exists()
        else:
            assert node.data.display is None
            assert node.data.display_version == expected_version
            assert path.read_bytes() == to_json(display)


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
    result = await execute(ws)
    assert all(node.data.error is None for node in ws.nodes)
    assert result.outputs[("3", "output")] == 2, (
        "Multi input op should return the correct number of bundles"
    )


async def test_failed_box_does_not_run_dependents():
    ws = workspace.Workspace(env="test")
    op = ops.op_registration("test")
    downstream_ran = False

    @op("Create value")
    def create_value() -> int:
        return 1

    @op("Fail")
    def fail(value: int) -> int:
        raise ValueError("expected failure")

    @op("Should not run")
    def should_not_run(value: int) -> int:
        nonlocal downstream_ran
        downstream_ran = True
        return value

    ws.add_node(
        id="0",
        type="node_type",
        title="Create value",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="1",
        type="node_type",
        title="Fail",
        position=workspace.Position(x=0, y=0),
    )
    ws.add_node(
        id="2",
        type="node_type",
        title="Should not run",
        position=workspace.Position(x=0, y=0),
    )
    ws.edges = [
        workspace.WorkspaceEdge(
            id="0", source="0", target="1", sourceHandle="output", targetHandle="value"
        ),
        workspace.WorkspaceEdge(
            id="1", source="1", target="2", sourceHandle="output", targetHandle="value"
        ),
    ]

    result = await execute(ws)

    assert next(node for node in ws.nodes if node.id == "1").data.error == (
        "ValueError: expected failure"
    )
    assert not downstream_ran
    assert ("1", "output") not in result.outputs
    assert next(node for node in ws.nodes if node.id == "2").data.error is None
