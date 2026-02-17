from lynxkite_core import ops, workspace
from lynxkite_core.executors import simple


async def test_optional_inputs():
    @ops.op("test", "one")
    def one():
        return 1

    @ops.op("test", "maybe add")
    def maybe_add(a: int, b: int | None = None):
        """b is optional"""
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


async def test_message_does_not_propagate_to_later_ops():
    @ops.op("test_msg", "op1")
    def op1(self):
        self.print("message from op1", append=False)
        return 2

    @ops.op("test_msg", "op2")
    def op2(a: int):
        return a * 2

    simple.register("test_msg")
    ws = workspace.Workspace(env="test_msg", nodes=[], edges=[])
    a = ws.add_node(op1)
    b = ws.add_node(op2)
    ws.add_edge(a, "output", b, "a")

    outputs = await ws.execute()
    assert outputs[b.id, "output"] == 4
    assert a.data.message == "message from op1\n"
    assert b.data.message is None


async def test_message_stays_when_cache_hit_on_slow_op():
    call_count = {"n": 0}
    old_cache_wrapper = ops.CACHE_WRAPPER

    import joblib
    import tempfile

    with tempfile.TemporaryDirectory() as cache_dir:
        mem = joblib.Memory(cache_dir, verbose=0)
        ops.CACHE_WRAPPER = mem.cache

        @ops.op("test_msg_cache", "slow_op", slow=True)
        def slow_op(self):
            call_count["n"] += 1
            self.print("message from slow_op", append=False)
            return 2

        try:
            simple.register("test_msg_cache")
            ws = workspace.Workspace(env="test_msg_cache", nodes=[], edges=[])
            a = ws.add_node(slow_op)
            outputs = await ws.execute()
            assert outputs[a.id, "output"] == 2
            assert a.data.message == "message from slow_op\n"
            # Execute again to hit the cache.
            outputs = await ws.execute()
            assert outputs[a.id, "output"] == 2
            assert a.data.message == "message from slow_op\n"
            assert call_count["n"] == 1
        finally:
            ops.CACHE_WRAPPER = old_cache_wrapper
