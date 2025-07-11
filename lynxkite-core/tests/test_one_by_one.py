from lynxkite.core import ops, workspace
from lynxkite.core.executors import one_by_one


async def test_optional_inputs():
    @ops.op("test", "one")
    def one():
        return 1

    @ops.input_position(a="bottom", b="bottom")
    @ops.op("test", "maybe add")
    def maybe_add(a: list[int], b: list[int] | None = None):
        return [a + b for a, b in zip(a, b)] if b else a

    assert maybe_add.__op__.inputs == [
        ops.Input(name="a", type=list[int], position=ops.Position.BOTTOM),
        ops.Input(name="b", type=list[int] | None, position=ops.Position.BOTTOM),
    ]
    one_by_one.register("test")
    ws = workspace.Workspace(env="test", nodes=[], edges=[])
    a = ws.add_node(one)
    b = ws.add_node(maybe_add)
    outputs = await ws.execute()
    assert b.data.error == "Missing input: a"
    ws.add_edge(a, "output", b, "a")
    outputs = await ws.execute()
    assert outputs[b.id].last_result == [1]
