from lynxkite.core import ops, workspace
from lynxkite.core.executors import simple


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
    simple.register("test")
    ws = workspace.Workspace(env="test", nodes=[], edges=[])
    a = ws.add_node(one)
    b = ws.add_node(maybe_add)
    await ws.execute()
    assert b.data.error == "Missing input: a"
    ws.add_edge(a, "output", b, "a")
    outputs = await ws.execute()
    assert outputs[b.id, "output"] == 1
